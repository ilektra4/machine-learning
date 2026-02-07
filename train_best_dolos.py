import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from io_utils import load_parquet, ensure_dir
from config import OUT_DIR, SEED

DOLOS_STATIC = os.path.join(OUT_DIR, "features_static.parquet")
DOLOS_TEMPORAL = os.path.join(OUT_DIR, "features_temporal.parquet")

MODEL_PATH = os.path.join(OUT_DIR, "best_dolos_both_logistic.joblib")
FEATCOLS_PATH = os.path.join(OUT_DIR, "best_dolos_feature_columns.txt")


def standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "video_id" not in df.columns and "file_name" in df.columns:
        df = df.rename(columns={"file_name": "video_id"})
    return df

def split_by_video(features: pd.DataFrame, train_frac=0.7, valid_frac_of_rest=0.5, seed=42):
    vids = features["video_id"].astype(str).unique()
    vids = np.array(vids, dtype=object)

    rng = np.random.default_rng(seed)
    rng.shuffle(vids)

    n_train = int(np.floor(train_frac * len(vids)))
    train_vid = set(vids[:n_train])
    rest = vids[n_train:]
    n_val = int(np.floor(valid_frac_of_rest * len(rest)))
    val_vid = set(rest[:n_val])
    test_vid = set(rest[n_val:])

    tr = features[features["video_id"].isin(train_vid)].copy()
    va = features[features["video_id"].isin(val_vid)].copy()
    te = features[features["video_id"].isin(test_vid)].copy()
    return tr, va, te

def make_xy(df: pd.DataFrame):
    y = df["label"].astype(str)
    drop_cols = [c for c in ["video_id","file_name","face_id","label","n_frames"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0)
    return X, y

def main():
    ensure_dir(OUT_DIR)

    a = standardize_keys(load_parquet(DOLOS_STATIC))
    b = standardize_keys(load_parquet(DOLOS_TEMPORAL)).drop(columns=["n_frames"], errors="ignore")

    feats = a.merge(b, on=["video_id","face_id","label"], how="left").fillna(0)

    tr, va, te = split_by_video(feats, seed=SEED)

    Xtr, ytr = make_xy(tr)
    Xva, yva = make_xy(va)
    Xte, yte = make_xy(te)

    # save feature columns order
    feat_cols = Xtr.columns.tolist()
    with open(FEATCOLS_PATH, "w", encoding="utf-8") as f:
        for c in feat_cols:
            f.write(c + "\n")
    print("Saved feature columns:", FEATCOLS_PATH, "n=", len(feat_cols))

    # scale + logistic
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(
        solver="liblinear",
        random_state=SEED,
        max_iter=5000
    )
    clf.fit(Xtr_s, ytr)

    # quick sanity metrics on DOLOS split (optional)
    p_va = clf.predict_proba(Xva_s)[:, 1]
    p_te = clf.predict_proba(Xte_s)[:, 1]

    # choose threshold=0.5 for this saved model (simple & stable)
    pred_va = np.where(p_va > 0.5, "truth", "lie")
    pred_te = np.where(p_te > 0.5, "truth", "lie")

    print("DOLOS valid acc:", accuracy_score(yva, pred_va))
    print("DOLOS test  acc:", accuracy_score(yte, pred_te))
    print("DOLOS test  AUC:", roc_auc_score((yte=="truth").astype(int), p_te))

    joblib.dump({"scaler": scaler, "model": clf}, MODEL_PATH)
    print("Saved model:", MODEL_PATH)

if __name__ == "__main__":
    main()
