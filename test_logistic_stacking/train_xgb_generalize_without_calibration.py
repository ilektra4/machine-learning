# train_xgb_generalize.py
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


# IMPORTANT: no GUI backend (fix Tk/Tcl issue)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    roc_curve, f1_score
)

from xgboost import XGBClassifier

# ----------------------------
# CONFIG
# ----------------------------
DOLOS_FILE  = r"D:\video_project\dolos_openface_merged_final.csv"
KAGGLE_FILE = r"D:\video_project\kaggle_openface_merged.csv"
OUT_DIR     = r"D:\video_project\xgb_out_new"

CONF_MIN   = 0.80
SEED       = 42

# CASME top AUs (your output)
TOP_AUS = ["AU07_r","AU04_r","AU14_r","AU45_r","AU06_r","AU10_r","AU12_r","AU01_r","AU17_r","AU26_r","AU25_r","AU02_r"]

# choose: "all" | "aus_all" | "casme_top"
FEATURE_MODE = "all"


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
    if "file_name" in df.columns:
        return "file_name"
    if "video_id" in df.columns:
        return "video_id"
    if "source_file" in df.columns:
        return "source_file"
    raise RuntimeError("No name column found (expected file_name/video_id/source_file).")

def pick_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = {"frame", "timestamp", "face_id", "success", "confidence"}
    numeric_cols = [c for c in numeric_cols if c not in drop]

    if mode == "all":
        return numeric_cols

    if mode == "aus_all":
        au_cols = [c for c in df.columns if re.match(r"^AU\d{2}_[rc]$", c)]
        au_cols = [c for c in au_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not au_cols:
            raise RuntimeError("No AU columns found like AU01_r / AU12_c etc.")
        return au_cols

    if mode == "casme_top":
        cols = [c for c in TOP_AUS if c in df.columns]
        if not cols:
            raise RuntimeError("None of TOP_AUS found in the dataset columns.")
        return cols

    raise ValueError(f"Unknown FEATURE_MODE: {mode}")

def aggregate_per_video(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    if "face_id" not in df.columns:
        df["face_id"] = 0

    sort_cols = [c for c in ["frame", "timestamp"] if c in df.columns]

    for vid, clip in df.groupby("video_id", sort=False):
        faces = []
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
            for i, c in enumerate(feat_cols):
                d[f"mean_{c}"]   = meanv[i]
                d[f"std_{c}"]    = stdv[i]
                d[f"max_{c}"]    = maxv[i]
                d[f"madiff_{c}"] = madiff[i]
            faces.append(d)

        if not faces:
            continue

        face_df = pd.DataFrame(faces)
        clip_feat = face_df.mean(axis=0).to_dict()
        clip_feat["video_id"] = vid
        clip_feat["label"] = clip["label"].iloc[0]
        rows.append(clip_feat)

    return pd.DataFrame(rows).fillna(0.0)

def build_video_table(csv_path: str, mode: str) -> tuple[pd.DataFrame, list[str]]:
    print("Reading:", csv_path)
    df = pd.read_csv(csv_path)

    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= CONF_MIN]

    name_col = pick_name_col(df)
    df["video_id"] = df[name_col].astype(str).str.replace(r"\.csv$", "", regex=True)

    df["label"] = df["video_id"].apply(label_from_name)
    df = df.dropna(subset=["label"]).copy()
    df = df.copy()

    feat_cols = pick_feature_columns(df, mode)
    video_df = aggregate_per_video(df, feat_cols)

    final_feat_cols = [c for c in video_df.columns if c not in ["video_id", "label"]]
    return video_df, final_feat_cols

def split_train_valid_test(df: pd.DataFrame, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    y = df["label"].values

    train_idx, valid_idx, test_idx = [], [], []

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(0.6 * n)
        n_valid = int(0.2 * n)

        train_idx.extend(cls_idx[:n_train])
        valid_idx.extend(cls_idx[n_train:n_train+n_valid])
        test_idx.extend(cls_idx[n_train+n_valid:])

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )

def find_best_threshold(y_true: np.ndarray, proba: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresholds[best_idx])

    pred = (proba >= best_thr).astype(int)
    f1 = float(f1_score(y_true, pred))
    return best_thr, f1

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

def train_xgb_cv(X: np.ndarray, y: np.ndarray, seed: int = 42):
    params_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 800, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 400, "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 900, "subsample": 0.8, "colsample_bytree": 0.7},
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
                n_jobs=0,
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
        n_jobs=0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        **best_params
    )
    final.fit(X, y)
    return final, {"best_cv_auc": best_auc, **best_params}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("FEATURE_MODE =", FEATURE_MODE)

    # ------------------ DOLOS ------------------
    print("\nBuilding DOLOS video features...")
    dolos, feat_cols = build_video_table(DOLOS_FILE, FEATURE_MODE)
    print("DOLOS clips:", len(dolos), "label counts:", dolos["label"].value_counts().to_dict())
    print("Feature columns:", len(feat_cols))

    train_df, valid_df, test_df = split_train_valid_test(dolos, seed=SEED)

    Xtr = train_df[feat_cols].to_numpy(dtype=float)
    ytr = (train_df["label"].astype(str) == "truth").astype(int).to_numpy()

    Xva = valid_df[feat_cols].to_numpy(dtype=float)
    yva = (valid_df["label"].astype(str) == "truth").astype(int).to_numpy()

    Xte = test_df[feat_cols].to_numpy(dtype=float)
    yte = (test_df["label"].astype(str) == "truth").astype(int).to_numpy()

    model, info = train_xgb_cv(Xtr, ytr, seed=SEED)
    print("\nDOLOS training (CV):", info)

    # threshold tuning on DOLOS VALID ONLY
    p_val = model.predict_proba(Xva)[:, 1]
    best_thr, f1_val = find_best_threshold(yva, p_val)
    print(f"\nBest threshold from DOLOS validation: {best_thr:.4f}")
    print(f"Validation F1 at best threshold: {f1_val:.4f}")

    # Evaluate on DOLOS TEST with tuned threshold
    p_te = model.predict_proba(Xte)[:, 1]
    pred_te = (p_te >= best_thr).astype(int)
    acc_te = accuracy_score(yte, pred_te)
    auc_te = roc_auc_score(yte, p_te)
    cm_te = confusion_matrix(yte, pred_te)

    print("\nDOLOS TEST (threshold from DOLOS valid)")
    print("Acc:", acc_te, "AUC:", auc_te)
    print("CM [rows=Actual lie/truth, cols=Pred lie/truth]:")
    print(np.array([[cm_te[0,0], cm_te[0,1]], [cm_te[1,0], cm_te[1,1]]]))

    save_roc(yte, p_te, os.path.join(OUT_DIR, f"roc_dolos_xgb_{FEATURE_MODE}.png"),
             f"DOLOS ROC (XGB:{FEATURE_MODE}) AUC={auc_te:.3f}")

    # feature importance (gain)
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

    # align Kaggle columns to training columns (fast)
    kag = kag.set_index(["video_id", "label"])
    kag = kag.reindex(columns=feat_cols, fill_value=0.0).reset_index()

    Xk = kag[feat_cols].to_numpy(dtype=float)
    yk = (kag["label"].astype(str) == "truth").astype(int).to_numpy()

    pk = model.predict_proba(Xk)[:, 1]
    predk = (pk >= best_thr).astype(int)
    acck = accuracy_score(yk, predk)
    auck = roc_auc_score(yk, pk)
    cmk = confusion_matrix(yk, predk)

    print("\nKAGGLE GENERALIZATION (same tuned threshold)")
    print("Acc:", acck, "AUC:", auck)
    print("CM [rows=Actual lie/truth, cols=Pred lie/truth]:")
    print(np.array([[cmk[0,0], cmk[0,1]], [cmk[1,0], cmk[1,1]]]))

    save_roc(yk, pk, os.path.join(OUT_DIR, f"roc_kaggle_xgb_{FEATURE_MODE}.png"),
             f"Kaggle ROC (XGB:{FEATURE_MODE}) AUC={auck:.3f}")

    # Save predictions
    out_pred = kag[["video_id", "label"]].copy()
    out_pred["proba_truth"] = pk
    out_pred["pred"] = np.where(predk == 1, "truth", "lie")
    out_csv = os.path.join(OUT_DIR, f"pred_kaggle_xgb_{FEATURE_MODE}.csv")
    out_pred.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(" -", out_csv)
    print(" -", os.path.join(OUT_DIR, f"roc_dolos_xgb_{FEATURE_MODE}.png"))
    print(" -", os.path.join(OUT_DIR, f"roc_kaggle_xgb_{FEATURE_MODE}.png"))
    print(" -", os.path.join(OUT_DIR, f"importance_gain_{FEATURE_MODE}.csv"))


if __name__ == "__main__":
    main()
