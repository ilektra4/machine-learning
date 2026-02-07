from __future__ import annotations
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    OUT_DIR, KAGGLE_OPENFACE_CSV,
    VIDEO_COL, FACE_COL, LABEL_COL, FRAME_COL,
    REQUIRE_SUCCESS, MIN_CONFIDENCE, MIN_FRAMES_PER_TRACK,
    FEAT_STATIC_AU, FEAT_TEMPORAL_AU, FEAT_TEMPORAL_AU_Z,
    SEED
)
from io_utils import load_parquet
from evaluate import save_roc_plot

# ---- EDIT THESE AFTER running run_experiments_au.py ----
CHOSEN_FEATURE_SET = "temporal_au_z"  # static_au / temporal_au / temporal_au_z
CHOSEN_MODEL = "lasso"               # logistic / lasso

def au_cols_from_df(df: pd.DataFrame):
    return [c for c in df.columns if c.endswith("_r")]

def build_features_from_frames(frames: pd.DataFrame, mode: str) -> pd.DataFrame:
    au_cols = au_cols_from_df(frames)
    keys = [VIDEO_COL, FACE_COL, LABEL_COL]

    frames = frames.sort_values(keys + [FRAME_COL]).copy()
    grp = frames.groupby(keys, sort=False)

    if mode == "static_au":
        stat = grp[au_cols].agg(["mean", "var", "max"]).reset_index()
        stat.columns = ["_".join([x for x in col if x]) for col in stat.columns.to_flat_index()]
        act = grp[au_cols].apply(lambda x: (x > 1).mean()).reset_index()
        act = act.rename(columns={c: f"act_{c}" for c in au_cols})
        n_frames = grp.size().reset_index(name="n_frames")
        out = stat.merge(n_frames, on=keys, how="left")
        out = out[out["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()
        out = out.merge(act, on=keys, how="left").fillna(0)
        return out

    if mode == "temporal_au":
        for c in au_cols:
            frames[f"d_{c}"] = grp[c].diff().abs()
        d_cols = [f"d_{c}" for c in au_cols]
        out = grp[d_cols].agg(["mean", "var", "max"]).reset_index()
        out.columns = ["_".join([x for x in col if x]) for col in out.columns.to_flat_index()]
        n_frames = grp.size().reset_index(name="n_frames")
        out = out.merge(n_frames, on=keys, how="left")
        out = out[out["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()
        return out.fillna(0)

    if mode == "temporal_au_z":
        for c in au_cols:
            mu = grp[c].transform("mean")
            sd = grp[c].transform("std").replace(0, np.nan)
            frames[f"z_{c}"] = ((frames[c] - mu) / sd).fillna(0)
        for c in au_cols:
            frames[f"dz_{c}"] = grp[f"z_{c}"].diff().abs()
        dz_cols = [f"dz_{c}" for c in au_cols]
        out = grp[dz_cols].agg(["mean", "var", "max"]).reset_index()
        out.columns = ["_".join([x for x in col if x]) for col in out.columns.to_flat_index()]
        n_frames = grp.size().reset_index(name="n_frames")
        out = out.merge(n_frames, on=keys, how="left")
        out = out[out["n_frames"] >= MIN_FRAMES_PER_TRACK].copy()
        return out.fillna(0)

    raise ValueError("Unknown mode")

def split_by_video(df: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    vids = df[VIDEO_COL].astype(str).unique()
    vids = np.array(vids)
    rng.shuffle(vids)

    n = len(vids)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)

    tr_vid = set(vids[:n_tr])
    va_vid = set(vids[n_tr:n_tr+n_va])
    te_vid = set(vids[n_tr+n_va:])

    tr = df[df[VIDEO_COL].astype(str).isin(tr_vid)].copy()
    va = df[df[VIDEO_COL].astype(str).isin(va_vid)].copy()
    te = df[df[VIDEO_COL].astype(str).isin(te_vid)].copy()
    return tr, va, te

def make_xy(df: pd.DataFrame):
    y = (df[LABEL_COL].astype(str) == "truth").astype(int).values
    X = df.drop(columns=[VIDEO_COL, FACE_COL, LABEL_COL], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.values, y, X.columns.tolist()

def best_threshold(y_true, proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])

def train_best_on_dolos(feat_path: str, model_kind: str):
    df = load_parquet(feat_path)
    tr, va, te = split_by_video(df, seed=SEED)

    Xtr, ytr, cols = make_xy(tr)
    Xva, yva, _ = make_xy(va)
    Xte, yte, _ = make_xy(te)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    if model_kind == "logistic":
        model = LogisticRegression(max_iter=5000, solver="lbfgs")
        model.fit(Xtr_s, ytr)
    else:
        Cs = np.logspace(-3, 1, 12)
        best = None
        best_auc = -1
        for C in Cs:
            m = LogisticRegression(max_iter=5000, solver="saga", penalty="l1", C=C)
            m.fit(Xtr_s, ytr)
            pva = m.predict_proba(Xva_s)[:, 1]
            auc_va = roc_auc_score(yva, pva)
            if auc_va > best_auc:
                best_auc = auc_va
                best = m
        model = best

    pva = model.predict_proba(Xva_s)[:, 1]
    t = best_threshold(yva, pva)

    # DOLOS test (sanity)
    pte = model.predict_proba(Xte_s)[:, 1]
    pred = (pte >= t).astype(int)
    print(f"DOLOS sanity ({model_kind}, threshold from DOLOS valid={t:.4f})")
    print("Acc:", accuracy_score(yte, pred), "AUC:", roc_auc_score(yte, pte))

    return model, scaler, cols, t

def load_kaggle_frames():
    df = pd.read_csv(KAGGLE_OPENFACE_CSV)

    # ensure core cols exist
    if VIDEO_COL not in df.columns and "video_id" in df.columns:
        df[VIDEO_COL] = df["video_id"]

    if FACE_COL not in df.columns:
        df[FACE_COL] = 0

    # frame col fallback
    if FRAME_COL not in df.columns and "frame" in df.columns:
        pass

    # filters
    if REQUIRE_SUCCESS and "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= MIN_CONFIDENCE]

    # label from filename/video_id if missing
    if LABEL_COL not in df.columns:
        vid = df[VIDEO_COL].astype(str).str.lower()
        df[LABEL_COL] = np.where(vid.str.contains("truth"), "truth",
                         np.where(vid.str.contains("lie"), "lie", np.nan))
    df = df.dropna(subset=[LABEL_COL]).copy()

    return df

def main():
    # pick correct dolos feature parquet for the chosen mode
    feat_map = {
        "static_au": os.path.join(OUT_DIR, FEAT_STATIC_AU),
        "temporal_au": os.path.join(OUT_DIR, FEAT_TEMPORAL_AU),
        "temporal_au_z": os.path.join(OUT_DIR, FEAT_TEMPORAL_AU_Z),
    }
    dolos_feat_path = feat_map[CHOSEN_FEATURE_SET]

    model, scaler, train_cols, threshold = train_best_on_dolos(dolos_feat_path, CHOSEN_MODEL)

    # Kaggle: build same feature mode from frames
    kag_frames = load_kaggle_frames()
    kag_feats = build_features_from_frames(kag_frames, CHOSEN_FEATURE_SET)

    # align columns to training
    Xk = kag_feats.drop(columns=[VIDEO_COL, FACE_COL, LABEL_COL], errors="ignore")
    Xk = Xk.select_dtypes(include=[np.number]).copy()
    for c in train_cols:
        if c not in Xk.columns:
            Xk[c] = 0
    Xk = Xk[train_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    yk = (kag_feats[LABEL_COL].astype(str) == "truth").astype(int).values

    Xk_s = scaler.transform(Xk.values)
    pk = model.predict_proba(Xk_s)[:, 1]
    pred = (pk >= threshold).astype(int)

    acc = accuracy_score(yk, pred)
    auc = roc_auc_score(yk, pk)
    cm = confusion_matrix(yk, pred, labels=[0, 1])

    print("\nKAGGLE GENERALIZATION (no Kaggle threshold tuning)")
    print("Feature set:", CHOSEN_FEATURE_SET, "Model:", CHOSEN_MODEL)
    print("Threshold (from DOLOS valid):", threshold)
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("CM [rows=Actual lie/truth, cols=Pred lie/truth]:\n", cm)

    save_roc_plot(yk, pk, os.path.join(OUT_DIR, f"roc_kaggle_{CHOSEN_FEATURE_SET}_{CHOSEN_MODEL}.png"))

if __name__ == "__main__":
    main()
