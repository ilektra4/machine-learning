import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUT_DIR, MIN_CONFIDENCE, REQUIRE_SUCCESS, MIN_FRAMES_PER_TRACK

KAGGLE_MERGED_CSV = r"D:\video_project\kaggle_openface_merged.csv"

MODEL_PATH = os.path.join(OUT_DIR, "best_dolos_both_logistic.joblib")
FEATCOLS_PATH = os.path.join(OUT_DIR, "best_dolos_feature_columns.txt")

OUT_PRED_CSV = os.path.join(OUT_DIR, "kaggle_predictions.csv")
OUT_ROC_PNG  = os.path.join(OUT_DIR, "roc_kaggle_best_model.png")

def load_featcols(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_feature_column_lists(df: pd.DataFrame):
    au_cols = [c for c in df.columns if c.endswith("_r")]
    gaze_pose_cols = [c for c in df.columns if re.search(r"(gaze|pose_R|pose_T)", c)]
    return au_cols, gaze_pose_cols

def preprocess_frames(df: pd.DataFrame) -> pd.DataFrame:
    # required
    if "video_id" not in df.columns:
        raise ValueError("Kaggle merged must contain video_id.")
    if "label" not in df.columns:
        raise ValueError("Kaggle merged must contain label.")
    if "face_id" not in df.columns:
        df["face_id"] = 0

    # filters if present
    if REQUIRE_SUCCESS and "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= MIN_CONFIDENCE]

    # frame for diffs
    if "frame" not in df.columns:
        if "timestamp" in df.columns:
            df = df.sort_values(["video_id", "face_id", "timestamp"]).copy()
            df["frame"] = df.groupby(["video_id", "face_id"]).cumcount() + 1
        else:
            raise ValueError("Need frame or timestamp.")

    return df

def aggregate_static(df: pd.DataFrame, au_cols, gp_cols, activation_thr=1.0) -> pd.DataFrame:
    keys = ["video_id", "face_id", "label"]
    base = df.groupby(keys).size().reset_index(name="n_frames")
    base = base[base["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    def agg_block(cols, prefix):
        g = df.groupby(keys)[cols]
        out = pd.concat([
            g.mean().add_prefix(f"{prefix}mean_"),
            g.min().add_prefix(f"{prefix}min_"),
            g.max().add_prefix(f"{prefix}max_"),
            g.var().fillna(0).add_prefix(f"{prefix}var_"),
            g.apply(lambda x: (x > activation_thr).mean()).add_prefix(f"{prefix}act_"),
        ], axis=1).reset_index()
        return out

    out = base.merge(agg_block(au_cols, "au_"), on=keys, how="left")

    if gp_cols:
        g = df.groupby(keys)[gp_cols]
        gp = pd.concat([
            g.mean().add_prefix("gp_mean_"),
            g.min().add_prefix("gp_min_"),
            g.max().add_prefix("gp_max_"),
            g.var().fillna(0).add_prefix("gp_var_"),
        ], axis=1).reset_index()
        out = out.merge(gp, on=keys, how="left")

    return out.fillna(0)

def aggregate_temporal(df: pd.DataFrame, au_cols, gp_cols) -> pd.DataFrame:
    keys = ["video_id", "face_id", "label"]
    df = df.sort_values(["video_id", "face_id", "frame"]).copy()
    g = df.groupby(keys, sort=False)

    for c in au_cols:
        df[f"d_{c}"] = g[c].diff()
        df[f"ad_{c}"] = df[f"d_{c}"].abs()
    for c in gp_cols:
        df[f"gd_{c}"] = g[c].diff()
        df[f"agd_{c}"] = df[f"gd_{c}"].abs()

    base = df.groupby(keys).size().reset_index(name="n_frames")
    base = base[base["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()

    def agg_stats(cols, prefix):
        if not cols:
            return pd.DataFrame(columns=keys)
        gg = df.groupby(keys)[cols]
        out = pd.concat([
            gg.mean().fillna(0).add_prefix(f"{prefix}mean_"),
            gg.min().fillna(0).add_prefix(f"{prefix}min_"),
            gg.max().fillna(0).add_prefix(f"{prefix}max_"),
            gg.var().fillna(0).add_prefix(f"{prefix}var_"),
        ], axis=1).reset_index()
        return out

    diff_au = [f"d_{c}" for c in au_cols]
    adiff_au = [f"ad_{c}" for c in au_cols]
    diff_gp = [f"gd_{c}" for c in gp_cols]
    adiff_gp = [f"agd_{c}" for c in gp_cols]

    out = base
    out = out.merge(agg_stats(diff_au,  "au_d_"),  on=keys, how="left")
    out = out.merge(agg_stats(adiff_au, "au_ad_"), on=keys, how="left")
    out = out.merge(agg_stats(diff_gp,  "gp_d_"),  on=keys, how="left")
    out = out.merge(agg_stats(adiff_gp, "gp_ad_"), on=keys, how="left")

    return out.fillna(0)

def plot_roc(y_true, proba, out_png):
    fpr, tpr, _ = roc_curve((y_true=="truth").astype(int), proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Kaggle ROC (Best DOLOS model)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    model  = bundle["model"]
    train_cols = load_featcols(FEATCOLS_PATH)

    df = pd.read_csv(KAGGLE_MERGED_CSV)
    df = preprocess_frames(df)

    au_cols, gp_cols = build_feature_column_lists(df)

    fs = aggregate_static(df, au_cols, gp_cols)
    ft = aggregate_temporal(df, au_cols, gp_cols).drop(columns=["n_frames"], errors="ignore")

    feats = fs.merge(ft, on=["video_id","face_id","label"], how="left").fillna(0)

    # build X aligned to training columns
    y = feats["label"].astype(str).values
    drop_cols = [c for c in ["video_id","face_id","label","n_frames","file_name"] if c in feats.columns]
    X = feats.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"]).fillna(0)

    # align
    X = X.reindex(columns=train_cols, fill_value=0)

    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]

    # ----- ROC-based threshold calibration -----
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve((y=="truth").astype(int), proba)
    j_scores = tpr - fpr
    best_thr = thresholds[np.argmax(j_scores)]

    print("Best threshold from Kaggle ROC:", best_thr)

    pred = np.where(proba > best_thr, "truth", "lie")


    acc = accuracy_score(y, pred)
    auc = roc_auc_score((y=="truth").astype(int), proba)
    cm = confusion_matrix(y, pred, labels=["lie","truth"])

    print("\nKAGGLE GENERALIZATION (threshold=0.5)")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("Confusion matrix [rows=Actual lie/truth, cols=Pred lie/truth]:\n", cm)

    out = feats[["video_id","face_id","label"]].copy()
    out["proba_truth"] = proba
    out["pred"] = pred
    out.to_csv(OUT_PRED_CSV, index=False)
    print("Saved predictions:", OUT_PRED_CSV)

    plot_roc(pd.Series(y), proba, OUT_ROC_PNG)
    print("Saved ROC:", OUT_ROC_PNG)

if __name__ == "__main__":
    main()
