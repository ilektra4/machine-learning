from __future__ import annotations
import os
import pandas as pd
import numpy as np

# --------------------------------------------
# Experiments execution and producing results 
# --------------------------------------------

from config import OUT_DIR, VIDEO_COL, LABEL_COL, SEED, LABELS_ORDER
from io_utils import load_parquet, ensure_dir, save_parquet
from pca_step import fit_pca
from model_logistic import fit_logistic
from model_stacking_tree import fit_tree, fit_meta_logistic, predict_meta
from model_lasso import fit_lasso
from evaluate import youden_threshold, evaluate, plot_roc, plot_explained_variance

# --------------------------------------------
# Split dataset 
# --------------------------------------------

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

    tr = features[features[VIDEO_COL].isin(train_vid)].copy()
    va = features[features[VIDEO_COL].isin(val_vid)].copy()
    te = features[features[VIDEO_COL].isin(test_vid)].copy()
    return tr, va, te

def make_xy(df: pd.DataFrame):
    y = df[LABEL_COL].astype(str)

    drop_cols = [c for c in ["video_id","file_name","face_id","label","n_frames"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    X = X.fillna(0)
    return X, y


# --------------------------------------------
# Aggregation type exploration 
# --------------------------------------------

def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "video_id" not in df.columns and "file_name" in df.columns:
        df = df.rename(columns={"file_name": "video_id"})
    return df

def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "video_id" not in df.columns and "file_name" in df.columns:
        df = df.rename(columns={"file_name": "video_id"})
    return df


def load_feature_set(which: str) -> pd.DataFrame:
    if which == "static":
        return _standardize_keys(load_parquet(os.path.join(OUT_DIR, "features_static.parquet")))

    if which == "temporal":
        return _standardize_keys(load_parquet(os.path.join(OUT_DIR, "features_temporal.parquet")))

    if which == "both":
        a = _standardize_keys(load_parquet(os.path.join(OUT_DIR, "features_static.parquet")))
        b = _standardize_keys(load_parquet(os.path.join(OUT_DIR, "features_temporal.parquet")))
        b = b.drop(columns=["n_frames"], errors="ignore")
        return a.merge(b, on=["video_id","face_id","label"], how="left").fillna(0)

    raise ValueError("which must be static|temporal|both")



def run_one(feature_set: str, model_kind: str, pos_label: str = "truth"):
    feats = load_feature_set(feature_set)
    tr, va, te = split_by_video(feats, seed=SEED)

    Xtr, ytr = make_xy(tr)
    Xva, yva = make_xy(va)
    Xte, yte = make_xy(te)

 # ----------------------------------------------
# Models - various versions with/without PCA etc.
# -----------------------------------------------
    if model_kind == "pca_logistic":
        Ztr, Zva, Zte, pca, scaler, k, cumvar = fit_pca(Xtr, Xva, Xte, variance_keep=0.95, seed=SEED)
        plot_explained_variance(cumvar, os.path.join(OUT_DIR, f"pca_cumvar_{feature_set}.png"))

        log = fit_logistic(Ztr, ytr, seed=SEED)
        p_va = log.predict_proba(Zva)[:, 1]
        p_te = log.predict_proba(Zte)[:, 1]

        thr = youden_threshold(yva, p_va, pos_label)
        res = evaluate(yte, p_te, pos_label, thr)
        plot_roc(yte, p_te, pos_label, os.path.join(OUT_DIR, f"roc_{feature_set}_{model_kind}.png"))
        return res

    if model_kind == "logistic":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)

        log = fit_logistic(Xtr_s, ytr, seed=SEED)
        p_va = log.predict_proba(Xva_s)[:, 1]
        p_te = log.predict_proba(Xte_s)[:, 1]

        thr = youden_threshold(yva, p_va, pos_label)
        res = evaluate(yte, p_te, pos_label, thr)
        plot_roc(yte, p_te, pos_label, os.path.join(OUT_DIR, f"roc_{feature_set}_{model_kind}.png"))
        return res

    if model_kind == "stacking_tree":
        Ztr, Zva, Zte, pca, scaler, k, cumvar = fit_pca(Xtr, Xva, Xte, variance_keep=0.95, seed=SEED)

        log = fit_logistic(Ztr, ytr, seed=SEED)
        tree = fit_tree(Ztr, ytr, seed=SEED)

        p_log_tr = log.predict_proba(Ztr)[:, 1]
        tree_tr = (tree.predict(Ztr) == pos_label).astype(int)
        meta = fit_meta_logistic(p_log_tr, tree_tr, ytr, seed=SEED)

        p_log_va = log.predict_proba(Zva)[:, 1]
        tree_va = (tree.predict(Zva) == pos_label).astype(int)
        p_va = predict_meta(meta, p_log_va, tree_va)

        p_log_te = log.predict_proba(Zte)[:, 1]
        tree_te = (tree.predict(Zte) == pos_label).astype(int)
        p_te = predict_meta(meta, p_log_te, tree_te)

        thr = youden_threshold(yva, p_va, pos_label)
        res = evaluate(yte, p_te, pos_label, thr)
        plot_roc(yte, p_te, pos_label, os.path.join(OUT_DIR, f"roc_{feature_set}_{model_kind}.png"))
        return res

    if model_kind == "lasso":
        lasso = fit_lasso(Xtr, ytr, seed=SEED)
        p_va = lasso.predict_proba(Xva)[:, 1]
        p_te = lasso.predict_proba(Xte)[:, 1]

        thr = youden_threshold(yva, p_va, pos_label)
        res = evaluate(yte, p_te, pos_label, thr)
        plot_roc(yte, p_te, pos_label, os.path.join(OUT_DIR, f"roc_{feature_set}_{model_kind}.png"))
        return res

    raise ValueError("model_kind must be pca_logistic|logistic|stacking_tree|lasso")

def main():
    ensure_dir(OUT_DIR)

    feature_sets = ["static", "temporal", "both"]
    models = ["pca_logistic", "logistic", "stacking_tree", "lasso"]

    rows = []
    for fs in feature_sets:
        for mk in models:
            res = run_one(fs, mk, pos_label="truth")
            rows.append({
                "feature_set": fs,
                "model": mk,
                "acc": res["acc"],
                "auc": res["auc"],
                "cm_lie_lie": int(res["cm"][0,0]),
                "cm_lie_truth": int(res["cm"][0,1]),
                "cm_truth_lie": int(res["cm"][1,0]),
                "cm_truth_truth": int(res["cm"][1,1]),
            })
            print(fs, mk, "acc", res["acc"], "auc", res["auc"])

    out = pd.DataFrame(rows).sort_values(["auc","acc"], ascending=False)
    out_path = os.path.join(OUT_DIR, "experiment_results.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
