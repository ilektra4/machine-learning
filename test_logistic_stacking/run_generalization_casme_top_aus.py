import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs

import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve
)

import matplotlib.pyplot as plt


# ----------------------------
# CONFIG (edit only if needed)
# ----------------------------
TOP_AUS = ["AU07_r","AU04_r","AU14_r","AU45_r","AU06_r","AU10_r","AU12_r","AU01_r","AU17_r","AU26_r","AU25_r","AU02_r"]

DOLOS_FILE  = r"D:\video_project\dolos_openface_merged_final.csv"
KAGGLE_FILE = r"D:\video_project\kaggle_openface_merged.csv"
OUT_DIR     = r"D:\video_project\dolos_with_casme_aus_pipeline_out"

CONF_MIN   = 0.80
THRESHOLD  = 0.50   # keep fixed; we can tune later but this keeps it honest


# ----------------------------
# Helpers
# ----------------------------
def list_csvs(root: str):
    files = []
    for base, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(".csv"):
                files.append(os.path.join(base, f))
    return files

def label_from_name(name: str):
    n = name.lower()
    if "truth" in n:
        return "truth"
    if "lie" in n:
        return "lie"
    return None

def safe_read_csv(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def build_features_from_openface_csv(path: str, top_aus):
    print("Reading:", path)
    df = pd.read_csv(path)

    # filters
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= CONF_MIN]

    if "face_id" not in df.columns:
        df["face_id"] = 0

    # get label from file_name / video_id
    name_col = "file_name" if "file_name" in df.columns else "video_id"
    df["video_id"] = df[name_col].astype(str).str.replace(r"\.csv$", "", regex=True)

    def label_from_name(name):
        n = name.lower()
        if "truth" in n: return "truth"
        if "lie" in n:   return "lie"
        return None

    df["label"] = df["video_id"].apply(label_from_name)
    df = df.dropna(subset=["label"])

    aus = [c for c in top_aus if c in df.columns]
    if not aus:
        raise RuntimeError("No CASME AUs found in CSV!")

    rows = []

    for vid, clip in df.groupby("video_id"):
        faces = []

        for fid, g in clip.groupby("face_id"):
            g = g.sort_values([c for c in ["frame","timestamp"] if c in g.columns])

            x = g[aus].to_numpy(dtype=float)

            meanv = np.nanmean(x, axis=0)
            varv  = np.nanvar(x, axis=0)
            maxv  = np.nanmax(x, axis=0)

            if x.shape[0] >= 2:
                madiff = np.nanmean(np.abs(np.diff(x, axis=0)), axis=0)
            else:
                madiff = np.zeros_like(meanv)

            act = np.nanmean(x > 1.0, axis=0)

            feat = {}
            for i, au in enumerate(aus):
                feat[f"mean_{au}"]   = meanv[i]
                feat[f"var_{au}"]    = varv[i]
                feat[f"max_{au}"]    = maxv[i]
                feat[f"madiff_{au}"] = madiff[i]
                feat[f"act_{au}"]    = act[i]

            faces.append(feat)

        if not faces:
            continue

        face_df = pd.DataFrame(faces)
        clip_feat = face_df.mean(axis=0).to_dict()
        clip_feat["video_id"] = vid
        clip_feat["label"] = clip["label"].iloc[0]
        rows.append(clip_feat)

    feats = pd.DataFrame(rows).fillna(0)
    return feats


def split_by_video(df, seed=42):
    # df has one row per video_id already, stratify by label
    rng = np.random.RandomState(seed)
    vids = df["video_id"].values
    y = df["label"].astype(str).values

    # stratified split: 70/30 train/test
    idx = np.arange(len(df))
    # simple stratified shuffle split without sklearn dependency
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        cut = int(np.floor(0.70 * len(cls_idx)))
        train_idx.extend(cls_idx[:cut])
        test_idx.extend(cls_idx[cut:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

def fit_logreg_cv(X, y, seed=42):
    # Elastic-net logistic with CV over (C, l1_ratio)
    Cs = np.logspace(-3, 3, 13)
    L1S = [0.2, 0.5, 0.8, 1.0]   # 1.0 = pure L1, 0.0 would be pure L2

    best_auc = -1.0
    best_C = None
    best_l1 = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for l1 in L1S:
        for C in Cs:
            aucs = []
            for tr, va in skf.split(X, y):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=l1,
                        C=C,
                        max_iter=5000,
                        random_state=seed
                    ))
                ])
                pipe.fit(X[tr], y[tr])
                p = pipe.predict_proba(X[va])[:, 1]
                aucs.append(roc_auc_score(y[va], p))

            m = float(np.mean(aucs))
            if m > best_auc:
                best_auc = m
                best_C = C
                best_l1 = l1

    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=best_l1,
            C=best_C,
            max_iter=5000,
            random_state=seed
        ))
    ])
    final.fit(X, y)

    return final, best_C, best_l1, best_auc


    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=best_C,
            max_iter=5000,
            random_state=seed
        ))
    ])
    final.fit(X, y)
    return final, best_C, best_auc

def save_roc(y_true, p_true, path, title):
    fpr, tpr, _ = roc_curve(y_true, p_true)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building DOLOS features from:", DOLOS_FILE)
    dolos = build_features_from_openface_csv(DOLOS_FILE, TOP_AUS)
    print("DOLOS clips:", len(dolos), " label counts:", dolos["label"].value_counts().to_dict())

    # Train/test split on DOLOS (video-level)
    tr, te = split_by_video(dolos, seed=42)

    feat_cols = [c for c in dolos.columns if c not in ["video_id", "label"]]
    Xtr = tr[feat_cols].to_numpy(dtype=float)
    ytr = (tr["label"].astype(str) == "truth").astype(int).to_numpy()

    Xte = te[feat_cols].to_numpy(dtype=float)
    yte = (te["label"].astype(str) == "truth").astype(int).to_numpy()

    model, best_C, best_l1, cv_auc = fit_logreg_cv(Xtr, ytr, seed=42)
    print(f"\nDOLOS training: best C={best_C}, best l1_ratio={best_l1} (CV AUC≈{cv_auc:.3f})")


    # Evaluate on DOLOS test
    p_te = model.predict_proba(Xte)[:, 1]
    pred_te = (p_te >= THRESHOLD).astype(int)
    acc_te = accuracy_score(yte, pred_te)
    auc_te = roc_auc_score(yte, p_te)
    cm_te = confusion_matrix(yte, pred_te)

    print("\nDOLOS TEST (fixed threshold=0.5)")
    print("Acc:", acc_te, "AUC:", auc_te)
    print("CM [rows=Actual lie/truth, cols=Pred lie/truth]:")
    # y=1 is truth, y=0 is lie => map to [lie, truth]
    # confusion_matrix gives [[tn, fp],[fn,tp]]
    print(np.array([[cm_te[0,0], cm_te[0,1]],[cm_te[1,0], cm_te[1,1]]]))

    save_roc(yte, p_te, os.path.join(OUT_DIR, "roc_dolos_casmeTopAUs.png"),
             f"DOLOS ROC (CASME top AUs) AUC={auc_te:.3f}")

    # Kaggle generalization (same feature engineering, same columns)
    print("\nBuilding KAGGLE features from:", KAGGLE_FILE)
    kag = build_features_from_openface_csv(KAGGLE_FILE, TOP_AUS)
    print("Kaggle clips:", len(kag), " label counts:", kag["label"].value_counts().to_dict())

    # align columns
    for c in feat_cols:
        if c not in kag.columns:
            kag[c] = 0.0
    kag = kag[["video_id", "label"] + feat_cols].fillna(0)

    Xk = kag[feat_cols].to_numpy(dtype=float)
    yk = (kag["label"].astype(str) == "truth").astype(int).to_numpy()

    pk = model.predict_proba(Xk)[:, 1]
    predk = (pk >= THRESHOLD).astype(int)
    acck = accuracy_score(yk, predk)
    auck = roc_auc_score(yk, pk)
    cmk = confusion_matrix(yk, predk)

    print("\nKAGGLE GENERALIZATION (fixed threshold=0.5)")
    print("Acc:", acck, "AUC:", auck)
    print("CM [rows=Actual lie/truth, cols=Pred lie/truth]:")
    print(np.array([[cmk[0,0], cmk[0,1]],[cmk[1,0], cmk[1,1]]]))

    save_roc(yk, pk, os.path.join(OUT_DIR, "roc_kaggle_casmeTopAUs.png"),
             f"Kaggle ROC (CASME top AUs) AUC={auck:.3f}")

    # Save predictions
    out_pred = kag[["video_id", "label"]].copy()
    out_pred["proba_truth"] = pk
    out_pred["pred"] = np.where(predk == 1, "truth", "lie")
    out_csv = os.path.join(OUT_DIR, "pred_kaggle_casmeTopAUs.csv")
    out_pred.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)
    print("Saved ROCs:",
          os.path.join(OUT_DIR, "roc_dolos_casmeTopAUs.png"),
          os.path.join(OUT_DIR, "roc_kaggle_casmeTopAUs.png"))

if __name__ == "__main__":
    main()
