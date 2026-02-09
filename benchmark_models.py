# benchmark_models.py
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from xgboost import XGBClassifier


# ----------------------------
# CONFIG
# ----------------------------
DOLOS_FILE  = r"D:\video_project\dolos_openface_merged_final.csv"
KAGGLE_FILE = r"D:\video_project\kaggle_openface_merged.csv"
OUT_DIR     = r"D:\video_project\benchmark_out"

CONF_MIN   = 0.80
SEED       = 42

TOP_AUS = ["AU07_r","AU04_r","AU14_r","AU45_r","AU06_r","AU10_r","AU12_r","AU01_r","AU17_r","AU26_r","AU25_r","AU02_r"]

# Choose which feature sets to benchmark
FEATURE_MODES = ["aus_all", "casme_top", "all"]   # recommended order
FACE_MODE = "dominant"  # "dominant" or "mean_all_faces"

# Calibration settings
CALIBRATE_LINEAR_SVC = True      # recommended (LinearSVC has no predict_proba)
CALIBRATE_XGB = False            # optional; set True to test


# ----------------------------
# Data + Features
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

def infer_subject_id(video_id: str) -> str | None:
    s = str(video_id)
    m = re.search(r"(?:^|[_\-])(s\d{1,3}|subject\d{1,3}|p\d{1,3}|person\d{1,3})(?:[_\-]|$)", s, flags=re.I)
    if m:
        return m.group(1).lower()
    tok = re.split(r"[_\-]", s)[0]
    if re.match(r"^(s\d{1,3}|subject\d{1,3}|p\d{1,3}|person\d{1,3})$", tok, flags=re.I):
        return tok.lower()
    return None

def aggregate_face_track(g: pd.DataFrame, feat_cols: list[str]) -> dict:
    sort_cols = [c for c in ["frame", "timestamp"] if c in g.columns]
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

    out = {}
    for i, c in enumerate(feat_cols):
        out[f"mean_{c}"]   = meanv[i]
        out[f"std_{c}"]    = stdv[i]
        out[f"max_{c}"]    = maxv[i]
        out[f"madiff_{c}"] = madiff[i]

    out["_n_frames"] = int(x.shape[0])
    if "confidence" in g.columns:
        out["_conf"] = float(np.nanmean(g["confidence"].to_numpy(dtype=float)))
    else:
        out["_conf"] = 0.0
    return out

def aggregate_per_video(df: pd.DataFrame, feat_cols: list[str], face_mode: str = "dominant") -> pd.DataFrame:
    rows = []
    if "face_id" not in df.columns:
        df["face_id"] = 0

    for vid, clip in df.groupby("video_id", sort=False):
        face_rows = []
        for _, g in clip.groupby("face_id", sort=False):
            face_rows.append(aggregate_face_track(g, feat_cols))

        if not face_rows:
            continue

        face_df = pd.DataFrame(face_rows)

        if face_mode == "dominant":
            face_df = face_df.sort_values(["_n_frames", "_conf"], ascending=False)
            chosen = face_df.iloc[0].to_dict()
        elif face_mode == "mean_all_faces":
            chosen = face_df.mean(axis=0).to_dict()
        else:
            raise ValueError("FACE_MODE must be 'dominant' or 'mean_all_faces'")

        chosen.pop("_n_frames", None)
        chosen.pop("_conf", None)
        chosen["video_id"] = vid
        chosen["label"] = clip["label"].iloc[0]
        rows.append(chosen)

    return pd.DataFrame(rows).fillna(0.0)

def build_video_table(csv_path: str, mode: str, face_mode: str) -> tuple[pd.DataFrame, list[str]]:
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

    df["subject_id"] = df["video_id"].apply(infer_subject_id)

    feat_cols = pick_feature_columns(df, mode)
    video_df = aggregate_per_video(df, feat_cols, face_mode=face_mode)

    subj_map = df.groupby("video_id")["subject_id"].agg(lambda x: x.dropna().iloc[0] if x.dropna().shape[0] else None)
    video_df["subject_id"] = video_df["video_id"].map(subj_map)

    final_feat_cols = [c for c in video_df.columns if c not in ["video_id", "label", "subject_id"]]
    return video_df, final_feat_cols

def split_train_valid_test(df: pd.DataFrame, seed: int = 42):
    if "subject_id" in df.columns and df["subject_id"].notna().any():
        g = df["subject_id"].fillna(df["video_id"]).to_numpy()
        y = df["label"].to_numpy()

        gss1 = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
        tr_idx, rest_idx = next(gss1.split(df, y, groups=g))

        rest = df.iloc[rest_idx].reset_index(drop=True)
        g_rest = rest["subject_id"].fillna(rest["video_id"]).to_numpy()
        y_rest = rest["label"].to_numpy()

        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
        va_idx, te_idx = next(gss2.split(rest, y_rest, groups=g_rest))

        return (
            df.iloc[tr_idx].reset_index(drop=True),
            rest.iloc[va_idx].reset_index(drop=True),
            rest.iloc[te_idx].reset_index(drop=True),
        )

    # fallback: stratified per-class split
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

def align_and_impute(train_df: pd.DataFrame, other_df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    train_means = train_df[feat_cols].mean(axis=0)
    other = other_df.set_index(["video_id", "label"])
    other = other.reindex(columns=feat_cols)
    other = other.fillna(train_means).reset_index()
    return other

def find_best_threshold_f1(y_true: np.ndarray, proba: np.ndarray):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1

def eval_with_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float):
    pred = (proba >= thr).astype(int)
    acc = accuracy_score(y_true, pred)
    f1  = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, proba)
    cm  = confusion_matrix(y_true, pred)
    return acc, f1, auc, cm

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


# ----------------------------
# Models
# ----------------------------
def train_xgb_cv(X: np.ndarray, y: np.ndarray, seed: int = 42):
    params_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 800,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 900,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 500,  "subsample": 0.9, "colsample_bytree": 0.9},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 1200, "subsample": 0.8, "colsample_bytree": 0.7},
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
                min_child_weight=1.0,
                gamma=0.0,
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
        min_child_weight=1.0,
        gamma=0.0,
        **best_params
    )
    final.fit(X, y)
    return final, {"best_cv_auc": best_auc, **best_params}

def make_models(seed: int = 42):
    models = {}

    # Logistic Regression baseline
    models["logreg_l2"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=seed
        ))
    ])

    # Linear SVM + calibration to get probabilities
    base_svc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0,
            random_state=seed
        ))
    ])
    if CALIBRATE_LINEAR_SVC:
        models["linear_svm_calib"] = CalibratedClassifierCV(
            estimator=base_svc,
            method="sigmoid",
            cv=5
        )
    else:
        # no proba -> will use decision_function scaled to [0,1] (not ideal)
        models["linear_svm_nocalib"] = base_svc

    # ExtraTrees
    models["extratrees"] = ExtraTreesClassifier(
        n_estimators=800,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    # XGBoost (selected via CV)
    models["xgb_cv"] = None  # placeholder; trained separately

    # Optional: XGB calibrated
    models["xgb_cv_calib"] = None  # placeholder; trained separately

    return models


# ----------------------------
# Main benchmark
# ----------------------------
def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].astype(str) == "truth").astype(int).to_numpy()

def get_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback: decision_function -> sigmoid-ish scaling (diagnostic only)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.clip(s, -20, 20)
        return 1.0 / (1.0 + np.exp(-s))
    raise RuntimeError("Model cannot produce probabilities.")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows = []

    for FEATURE_MODE in FEATURE_MODES:
        print("\n" + "="*70)
        print(f"FEATURE_MODE = {FEATURE_MODE} | FACE_MODE = {FACE_MODE}")
        print("="*70)

        # Build DOLOS
        print("\nBuilding DOLOS video features...")
        dolos, feat_cols = build_video_table(DOLOS_FILE, FEATURE_MODE, face_mode=FACE_MODE)
        print("DOLOS clips:", len(dolos), "label counts:", dolos["label"].value_counts().to_dict())
        print("Feature columns:", len(feat_cols))

        train_df, valid_df, test_df = split_train_valid_test(dolos, seed=SEED)

        Xtr = train_df[feat_cols].to_numpy(dtype=float); ytr = to_y(train_df)
        Xva = valid_df[feat_cols].to_numpy(dtype=float); yva = to_y(valid_df)
        Xte = test_df[feat_cols].to_numpy(dtype=float);  yte = to_y(test_df)

        # Build Kaggle (raw)
        print("\nBuilding KAGGLE video features...")
        kag, _ = build_video_table(KAGGLE_FILE, FEATURE_MODE, face_mode=FACE_MODE)
        print("Kaggle clips:", len(kag), "label counts:", kag["label"].value_counts().to_dict())

        # Align Kaggle to DOLOS feature columns (and impute from train means)
        kag_aligned = align_and_impute(train_df, kag, feat_cols)
        Xk = kag_aligned[feat_cols].to_numpy(dtype=float)
        yk = to_y(kag_aligned)

        models = make_models(SEED)

        # Train XGB-CV once per feature mode
        print("\nTraining XGB (CV grid)...")
        xgb_model, xgb_info = train_xgb_cv(Xtr, ytr, seed=SEED)
        print("XGB best:", xgb_info)
        models["xgb_cv"] = xgb_model

        if CALIBRATE_XGB:
            # calibrate using VALID only (FrozenEstimator stops refit)
            calib = CalibratedClassifierCV(
                estimator=FrozenEstimator(xgb_model),
                method="sigmoid",
                cv=5
            )
            calib.fit(Xva, yva)
            models["xgb_cv_calib"] = calib
            print("XGB calibration: sigmoid on DOLOS valid.")
        else:
            models.pop("xgb_cv_calib", None)

        # Train / evaluate all models
        for name, model in models.items():
            if model is None:
                continue

            print("\n---", name, "---")

            # Fit (XGB already fit)
            if name not in ("xgb_cv", "xgb_cv_calib"):
                model.fit(Xtr, ytr)

            # Tune threshold on DOLOS valid (max F1)
            p_val = get_proba(model, Xva)
            thr, f1v = find_best_threshold_f1(yva, p_val)

            # DOLOS test
            p_te = get_proba(model, Xte)
            acc_te, f1_te, auc_te, cm_te = eval_with_threshold(yte, p_te, thr)

            # Kaggle
            p_k = get_proba(model, Xk)
            acc_k, f1_k, auc_k, cm_k = eval_with_threshold(yk, p_k, thr)

            print(f"thr(valid maxF1)={thr:.3f} | DOLOS: AUC={auc_te:.3f} Acc={acc_te:.3f} F1={f1_te:.3f} | "
                  f"KAGGLE: AUC={auc_k:.3f} Acc={acc_k:.3f} F1={f1_k:.3f}")
            print("DOLOS CM [[TN FP],[FN TP]]:")
            print(np.array([[cm_te[0,0], cm_te[0,1]], [cm_te[1,0], cm_te[1,1]]]))
            print("KAGGLE CM [[TN FP],[FN TP]]:")
            print(np.array([[cm_k[0,0], cm_k[0,1]], [cm_k[1,0], cm_k[1,1]]]))

            # Save ROC plots
            roc_dolos = os.path.join(OUT_DIR, f"roc_dolos_{FEATURE_MODE}_{name}.png")
            roc_kag   = os.path.join(OUT_DIR, f"roc_kaggle_{FEATURE_MODE}_{name}.png")
            save_roc(yte, p_te, roc_dolos, f"DOLOS ROC | {FEATURE_MODE} | {name} | AUC={auc_te:.3f}")
            save_roc(yk,  p_k,  roc_kag,   f"KAGGLE ROC | {FEATURE_MODE} | {name} | AUC={auc_k:.3f}")

            # Save Kaggle predictions
            predk = (p_k >= thr).astype(int)
            out_pred = kag_aligned[["video_id", "label"]].copy()
            out_pred["proba_truth"] = p_k
            out_pred["pred"] = np.where(predk == 1, "truth", "lie")
            out_csv = os.path.join(OUT_DIR, f"pred_kaggle_{FEATURE_MODE}_{name}.csv")
            out_pred.to_csv(out_csv, index=False)

            summary_rows.append({
                "feature_mode": FEATURE_MODE,
                "model": name,
                "thr_from_valid_maxF1": thr,
                "dolos_test_auc": auc_te,
                "dolos_test_acc": acc_te,
                "dolos_test_f1": f1_te,
                "kaggle_auc": auc_k,
                "kaggle_acc": acc_k,
                "kaggle_f1": f1_k,
                "dolos_TN": int(cm_te[0,0]), "dolos_FP": int(cm_te[0,1]),
                "dolos_FN": int(cm_te[1,0]), "dolos_TP": int(cm_te[1,1]),
                "kaggle_TN": int(cm_k[0,0]), "kaggle_FP": int(cm_k[0,1]),
                "kaggle_FN": int(cm_k[1,0]), "kaggle_TP": int(cm_k[1,1]),
            })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["feature_mode", "kaggle_auc", "dolos_test_auc"], ascending=[True, False, False]
    )
    out_sum = os.path.join(OUT_DIR, "benchmark_summary.csv")
    summary.to_csv(out_sum, index=False)

    print("\nDONE. Saved summary:")
    print(" -", out_sum)
    print("\nTop 10 by Kaggle AUC:")
    print(summary[["feature_mode","model","kaggle_auc","dolos_test_auc","kaggle_acc","dolos_test_acc","thr_from_valid_maxF1"]].head(10))


if __name__ == "__main__":
    main()
