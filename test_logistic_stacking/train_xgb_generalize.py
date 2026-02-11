# train_xgb_generalize.py
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

# Force non-GUI matplotlib backend (fixes your Tk/Tcl crash)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# ----------------------------
# CONFIG
# ----------------------------
DOLOS_FILE  = r"D:\video_project\dolos_openface_merged_final.csv"
KAGGLE_FILE = r"D:\video_project\kaggle_openface_merged.csv"
OUT_DIR     = r"D:\video_project\xgb_out"

CONF_MIN   = 0.80
SEED       = 42

# choose: "all" | "aus_all" | "casme_top"
FEATURE_MODE = "all"

# CASME top AUs (only used when FEATURE_MODE="casme_top")
TOP_AUS = ["AU07_r","AU04_r","AU14_r","AU45_r","AU06_r","AU10_r","AU12_r","AU01_r","AU17_r","AU26_r","AU25_r","AU02_r"]

# If True: do Platt scaling calibration on VALID (recommended if you report probabilities)
DO_CALIBRATION = True

# ----------------------------
# Helpers
# ----------------------------
def label_from_name(name: str):
    n = str(name).lower()
    if "truth" in n:
        return "truth"
    if "lie" in n:
        return "lie"
    return None

def pick_name_col(df: pd.DataFrame) -> str:
    for c in ["file_name", "video_id", "source_file"]:
        if c in df.columns:
            return c
    raise RuntimeError("No name column found (expected file_name/video_id/source_file).")

def pick_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    # numeric candidates
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # drop obvious non-features if present
    drop = {"frame", "timestamp", "face_id", "success", "confidence"}
    numeric_cols = [c for c in numeric_cols if c not in drop]

    if mode == "all":
        return numeric_cols

    if mode == "aus_all":
        # AUxx_r and AUxx_c (classic AU set)
        au_cols = [c for c in df.columns if re.match(r"^AU\d{2}_[rc]$", c)]
        au_cols = [c for c in au_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not au_cols:
            raise RuntimeError("No AU columns found like AU01_r / AU12_c etc.")
        return au_cols

    if mode == "casme_top":
        cols = [c for c in TOP_AUS if c in df.columns]
        if not cols:
            raise RuntimeError("None of TOP_AUS found in dataset columns.")
        return cols

    raise ValueError(f"Unknown FEATURE_MODE: {mode}")

def aggregate_per_video(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Per-frame rows -> one row per video.
    For each feature column compute: mean, std, max, madiff (mean abs diff consecutive frames).
    If multiple faces exist, aggregate each face then average faces.
    """
    if "face_id" not in df.columns:
        df["face_id"] = 0

    sort_cols = [c for c in ["frame", "timestamp"] if c in df.columns]
    out_rows = []

    # group by video_id
    for vid, clip in df.groupby("video_id", sort=False):
        face_rows = []
        for _, g in clip.groupby("face_id", sort=False):
            if sort_cols:
                g = g.sort_values(sort_cols)

            x = g[feat_cols].to_numpy(dtype=float)

            meanv = np.nanmean(x, axis=0)
            stdv  = np.nanstd(x, axis=0)
            maxv  = np.nanmax(x, axis=0)

            if x.shape[0] >= 2:
                madiff = np.nanmean(np.abs(np.diff(x, axis=0)), axis=0)
            else:
                madiff = np.zeros_like(meanv)

            d = {}
            # build columns without DataFrame insert loops (avoid fragmentation)
            for i, c in enumerate(feat_cols):
                d[f"mean_{c}"]   = meanv[i]
                d[f"std_{c}"]    = stdv[i]
                d[f"max_{c}"]    = maxv[i]
                d[f"madiff_{c}"] = madiff[i]
            face_rows.append(d)

        if not face_rows:
            continue

        face_df = pd.DataFrame(face_rows)
        clip_feat = face_df.mean(axis=0).to_dict()
        clip_feat["video_id"] = vid
        clip_feat["label"] = clip["label"].iloc[0]
        out_rows.append(clip_feat)

    return pd.DataFrame(out_rows).fillna(0.0)

def build_video_table(csv_path: str, mode: str) -> tuple[pd.DataFrame, list[str]]:
    print("Reading:", csv_path)
    df = pd.read_csv(csv_path)

    # frame-level filters
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= CONF_MIN]

    name_col = pick_name_col(df)
    df["video_id"] = df[name_col].astype(str).str.replace(r"\.csv$", "", regex=True)

    df["label"] = df["video_id"].apply(label_from_name)
    df = df.dropna(subset=["label"]).copy()

    feat_cols = pick_feature_columns(df, mode)
    video_df = aggregate_per_video(df, feat_cols)

    final_feat_cols = [c for c in video_df.columns if c not in ["video_id", "label"]]
    return video_df, final_feat_cols

def split_train_valid_test_video(df: pd.DataFrame, seed: int = 42, frac_train=0.6, frac_valid=0.2):
    """
    Stratified split by label (video-level).
    Returns train, valid, test dataframes.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    y = df["label"].values

    train_idx, valid_idx, test_idx = [], [], []

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(frac_train * n)
        n_valid = int(frac_valid * n)

        train_idx.extend(cls_idx[:n_train])
        valid_idx.extend(cls_idx[n_train:n_train + n_valid])
        test_idx.extend(cls_idx[n_train + n_valid:])

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )

def save_roc(y_true: np.ndarray, p: np.ndarray, out_png: str, title: str):
    fpr, tpr, _ = roc_curve(y_true, p)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_pr(y_true: np.ndarray, p: np.ndarray, out_png: str, title: str):
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_calibration_curve(y_true: np.ndarray, p: np.ndarray, out_png: str, title: str, bins: int = 10):
    # simple reliability plot
    p = np.clip(p, 1e-6, 1 - 1e-6)
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    grp = df.groupby("bin", observed=True)
    mean_p = grp["p"].mean().values
    frac_pos = grp["y"].mean().values

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(mean_p, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def train_xgb_cv(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[XGBClassifier, dict]:
    """
    Small CV search. AUC is used (threshold-free).
    """
    params_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 800, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 900, "subsample": 0.8, "colsample_bytree": 0.7},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 400, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_auc = -1.0
    best_params = None

    for p in params_grid:
        aucs = []
        for tr, va in skf.split(X, y):
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                reg_lambda=1.0,
                reg_alpha=0.0,
                **p
            )
            clf.fit(X[tr], y[tr])
            pv = clf.predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y[va], pv))

        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc = m
            best_params = p

    final = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        **best_params
    )
    final.fit(X, y)

    return final, {"best_cv_auc": best_auc, **best_params}

def find_best_threshold_by_youden(y_true: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, p)
    J = tpr - fpr
    return float(thr[int(np.argmax(J))])

def platt_calibrate(valid_proba: np.ndarray, valid_y: np.ndarray) -> LogisticRegression:
    """
    Platt scaling: fit logistic regression on model probabilities (1D) using VALID set.
    Returns a calibrator that maps raw proba -> calibrated proba.
    """
    Xv = valid_proba.reshape(-1, 1)
    cal = LogisticRegression(solver="lbfgs")
    cal.fit(Xv, valid_y)
    return cal

def apply_platt(cal: LogisticRegression, raw_proba: np.ndarray) -> np.ndarray:
    return cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

def report_block(title: str, y: np.ndarray, p: np.ndarray, threshold: float):
    pred = (p >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, p)
    cm = confusion_matrix(y, pred)  # [[tn, fp],[fn,tp]]

    print(f"\n{title}")
    print("Acc:", float(acc), "AUC:", float(auc))
    print("CM [rows=Actual lie/truth, cols=Pred lie/truth]:")
    print(np.array([[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]]))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("FEATURE_MODE =", FEATURE_MODE)

    # ------------------ DOLOS ------------------
    print("\nBuilding DOLOS video features...")
    dolos, feat_cols = build_video_table(DOLOS_FILE, FEATURE_MODE)
    print("DOLOS clips:", len(dolos), "label counts:", dolos["label"].value_counts().to_dict())
    print("Feature columns:", len(feat_cols))

    tr, va, te = split_train_valid_test_video(dolos, seed=SEED)

    Xtr = tr[feat_cols].to_numpy(dtype=float)
    ytr = (tr["label"].astype(str) == "truth").astype(int).to_numpy()

    Xva = va[feat_cols].to_numpy(dtype=float)
    yva = (va["label"].astype(str) == "truth").astype(int).to_numpy()

    Xte = te[feat_cols].to_numpy(dtype=float)
    yte = (te["label"].astype(str) == "truth").astype(int).to_numpy()

    model, info = train_xgb_cv(Xtr, ytr, seed=SEED)
    print("\nDOLOS training (CV):", info)

    # raw probs
    p_va_raw = model.predict_proba(Xva)[:, 1]
    p_te_raw = model.predict_proba(Xte)[:, 1]

    # threshold tuning on VALID (on raw probs)
    best_thr = find_best_threshold_by_youden(yva, p_va_raw)
    print(f"\nBest threshold from DOLOS validation (Youden J): {best_thr:.4f}")

    # optional calibration
    if DO_CALIBRATION:
        cal = platt_calibrate(p_va_raw, yva)
        p_va = apply_platt(cal, p_va_raw)
        p_te = apply_platt(cal, p_te_raw)
        print("Calibration: Platt scaling on VALID enabled.")
    else:
        p_va = p_va_raw
        p_te = p_te_raw
        print("Calibration: OFF (raw model probabilities).")

    # Evaluate on DOLOS VALID/TEST
    report_block("DOLOS VALID", yva, p_va, best_thr)
    report_block("DOLOS TEST (threshold from VALID)", yte, p_te, best_thr)

    # Save DOLOS ROC/PR/Cal
    save_roc(yte, p_te, os.path.join(OUT_DIR, f"roc_dolos_xgb_{FEATURE_MODE}.png"),
             f"DOLOS ROC (XGB:{FEATURE_MODE}) AUC={roc_auc_score(yte, p_te):.3f}")
    save_pr(yte, p_te, os.path.join(OUT_DIR, f"pr_dolos_xgb_{FEATURE_MODE}.png"),
            f"DOLOS PR (XGB:{FEATURE_MODE})")
    save_calibration_curve(
        yte, p_te,
        os.path.join(OUT_DIR, f"cal_dolos_xgb_{FEATURE_MODE}.png"),
        f"DOLOS Calibration (XGB:{FEATURE_MODE})"
    )

    # Feature importance (gain)
    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": list(imp.keys()), "gain": list(imp.values())})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(os.path.join(OUT_DIR, f"importance_gain_{FEATURE_MODE}.csv"), index=False)

    # ------------------ KAGGLE ------------------
    print("\nBuilding KAGGLE video features...")
    kag, _ = build_video_table(KAGGLE_FILE, FEATURE_MODE)
    print("Kaggle clips:", len(kag), "label counts:", kag["label"].value_counts().to_dict())

    # Align Kaggle to DOLOS training columns in ONE go (no fragmentation loops)
    kag = kag.set_index(["video_id", "label"])
    kag = kag.reindex(columns=feat_cols, fill_value=0.0).reset_index()

    Xk = kag[feat_cols].to_numpy(dtype=float)
    yk = (kag["label"].astype(str) == "truth").astype(int).to_numpy()

    pk_raw = model.predict_proba(Xk)[:, 1]
    if DO_CALIBRATION:
        pk = apply_platt(cal, pk_raw)
    else:
        pk = pk_raw

    report_block("KAGGLE GENERALIZATION (same threshold from DOLOS VALID)", yk, pk, best_thr)

    # Save Kaggle ROC/PR/Cal
    save_roc(yk, pk, os.path.join(OUT_DIR, f"roc_kaggle_xgb_{FEATURE_MODE}.png"),
             f"Kaggle ROC (XGB:{FEATURE_MODE}) AUC={roc_auc_score(yk, pk):.3f}")
    save_pr(yk, pk, os.path.join(OUT_DIR, f"pr_kaggle_xgb_{FEATURE_MODE}.png"),
            f"Kaggle PR (XGB:{FEATURE_MODE})")
    save_calibration_curve(
        yk, pk,
        os.path.join(OUT_DIR, f"cal_kaggle_xgb_{FEATURE_MODE}.png"),
        f"Kaggle Calibration (XGB:{FEATURE_MODE})"
    )

    # Save extra numeric metrics (useful in report)
    metrics = {
        "FEATURE_MODE": FEATURE_MODE,
        "calibration": bool(DO_CALIBRATION),
        "threshold_from_valid": float(best_thr),

        "dolos_test_auc": float(roc_auc_score(yte, p_te)),
        "dolos_test_brier": float(brier_score_loss(yte, p_te)),
        "dolos_test_pr_auc": float(average_precision_score(yte, p_te)),

        "kaggle_auc": float(roc_auc_score(yk, pk)),
        "kaggle_brier": float(brier_score_loss(yk, pk)),
        "kaggle_pr_auc": float(average_precision_score(yk, pk)),
    }
    pd.Series(metrics).to_csv(os.path.join(OUT_DIR, f"metrics_{FEATURE_MODE}.csv"))

    # Save predictions
    out_pred = kag[["video_id", "label"]].copy()
    out_pred["proba_truth"] = pk
    out_pred["pred"] = np.where((pk >= best_thr).astype(int) == 1, "truth", "lie")
    out_csv = os.path.join(OUT_DIR, f"pred_kaggle_xgb_{FEATURE_MODE}.csv")
    out_pred.to_csv(out_csv, index=False)

    print("\nSaved outputs in:", OUT_DIR)
    print(" -", out_csv)
    print(" - ROC/PR/Calibration PNGs (DOLOS + Kaggle)")
    print(" -", os.path.join(OUT_DIR, f"importance_gain_{FEATURE_MODE}.csv"))
    print(" -", os.path.join(OUT_DIR, f"metrics_{FEATURE_MODE}.csv"))

if __name__ == "__main__":
    main()
