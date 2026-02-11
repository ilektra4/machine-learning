from __future__ import annotations
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    OUT_DIR, VIDEO_COL, FACE_COL, LABEL_COL, SEED,
    FEAT_STATIC_AU, FEAT_TEMPORAL_AU, FEAT_TEMPORAL_AU_Z
)
from io_utils import load_parquet, save_csv
from evaluate import save_roc_plot

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
    # drop any non-numeric just in case
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.values, y, X.columns.tolist()

def best_threshold(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])

def train_eval(feature_name: str, df: pd.DataFrame):
    tr, va, te = split_by_video(df, seed=SEED)

    Xtr, ytr, cols = make_xy(tr)
    Xva, yva, _ = make_xy(va)
    Xte, yte, _ = make_xy(te)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    results = []

    # Logistic (L2)
    log = LogisticRegression(max_iter=5000, solver="lbfgs")
    log.fit(Xtr_s, ytr)

    pva = log.predict_proba(Xva_s)[:, 1]
    pte = log.predict_proba(Xte_s)[:, 1]
    t = best_threshold(yva, pva)

    pred = (pte >= t).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, pte)
    cm = confusion_matrix(yte, pred, labels=[0, 1])  # 0=lie,1=truth

    save_roc_plot(yte, pte, os.path.join(OUT_DIR, f"roc_{feature_name}_logistic.png"))

    results.append(dict(
        feature_set=feature_name,
        model="logistic",
        acc=acc,
        auc=auc,
        threshold=t,
        cm_lie_lie=int(cm[0,0]),
        cm_lie_truth=int(cm[0,1]),
        cm_truth_lie=int(cm[1,0]),
        cm_truth_truth=int(cm[1,1]),
        n_test=len(yte)
    ))

    # LASSO (L1)
    # Note: saga supports l1. Use C (inverse of regularization). Try a small grid quickly.
    Cs = np.logspace(-3, 1, 12)
    best = None
    best_auc = -1

    for C in Cs:
        l1 = LogisticRegression(max_iter=5000, solver="saga", penalty="l1", C=C)
        l1.fit(Xtr_s, ytr)
        pva = l1.predict_proba(Xva_s)[:, 1]
        auc_va = roc_auc_score(yva, pva)
        if auc_va > best_auc:
            best_auc = auc_va
            best = l1

    pva = best.predict_proba(Xva_s)[:, 1]
    pte = best.predict_proba(Xte_s)[:, 1]
    t = best_threshold(yva, pva)

    pred = (pte >= t).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, pte)
    cm = confusion_matrix(yte, pred, labels=[0, 1])

    save_roc_plot(yte, pte, os.path.join(OUT_DIR, f"roc_{feature_name}_lasso.png"))

    results.append(dict(
        feature_set=feature_name,
        model="lasso",
        acc=acc,
        auc=auc,
        threshold=t,
        best_C=float(best.C),
        cm_lie_lie=int(cm[0,0]),
        cm_lie_truth=int(cm[0,1]),
        cm_truth_lie=int(cm[1,0]),
        cm_truth_truth=int(cm[1,1]),
        n_test=len(yte)
    ))

    return results, cols, scaler, best

def main():
    feats = {
        "static_au": load_parquet(os.path.join(OUT_DIR, FEAT_STATIC_AU)),
        "temporal_au": load_parquet(os.path.join(OUT_DIR, FEAT_TEMPORAL_AU)),
        "temporal_au_z": load_parquet(os.path.join(OUT_DIR, FEAT_TEMPORAL_AU_Z)),
    }

    all_rows = []
    for name, df in feats.items():
        rows, _, _, _ = train_eval(name, df)
        all_rows.extend(rows)

    res = pd.DataFrame(all_rows).sort_values(["auc","acc"], ascending=False)
    out_csv = os.path.join(OUT_DIR, "experiment_results_au.csv")
    res.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(res[["feature_set","model","acc","auc","threshold"]].to_string(index=False))

if __name__ == "__main__":
    main()
