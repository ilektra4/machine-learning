import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# CHANGE THESE if needed
OUT_DIR = r"D:\video_project\xgb_out\all_features"
PRED_KAGGLE = os.path.join(OUT_DIR, "pred_kaggle_xgb_all.csv")

def save_pr_curve(y, p, out_png, title):
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} | AP={ap:.3f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return ap

def save_calibration(y, p, out_png, title, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(PRED_KAGGLE)
    # y=1 for truth, y=0 for lie (matches your training)
    y = (df["label"].astype(str) == "truth").astype(int).to_numpy()
    p = df["proba_truth"].to_numpy(dtype=float)

    roc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    print("KAGGLE metrics from saved predictions:")
    print("ROC AUC:", roc)
    print("PR AUC (Average Precision):", ap)
    print("Brier score (calibration error-ish):", brier)

    save_pr_curve(y, p, os.path.join(OUT_DIR, "pr_kaggle_all.png"), "Kaggle PR curve")
    save_calibration(y, p, os.path.join(OUT_DIR, "cal_kaggle_all.png"), "Kaggle Calibration")

    # --- sanity: shuffle labels baseline ---
    rng = np.random.RandomState(42)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    roc_shuf = roc_auc_score(y_shuf, p)
    ap_shuf = average_precision_score(y_shuf, p)

    print("\nSanity check (same probabilities, SHUFFLED labels):")
    print("ROC AUC shuffled:", roc_shuf)
    print("PR AUC shuffled:", ap_shuf)
    print("\nSaved:")
    print(" -", os.path.join(OUT_DIR, "pr_kaggle_all.png"))
    print(" -", os.path.join(OUT_DIR, "cal_kaggle_all.png"))

if __name__ == "__main__":
    main()
