from __future__ import annotations
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# -----------------------------------------------------------------------------------------------
# Evaluation metrics - ROC curve, AUC score, accuracy score (important here), confusion matrix
# -----------------------------------------------------------------------------------------------

def youden_threshold(y_true, proba_pos, pos_label: str):
    fpr, tpr, thr = roc_curve(y_true, proba_pos, pos_label=pos_label)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])

def evaluate(y_true, proba_pos, pos_label: str, threshold: float):
    pred = np.where(proba_pos > threshold, pos_label, _other_label(pos_label))
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score((y_true == pos_label).astype(int), proba_pos)
    cm = confusion_matrix(y_true, pred, labels=["lie","truth"])
    return {"acc": acc, "auc": auc, "cm": cm}

def _other_label(pos_label: str) -> str:
    return "lie" if pos_label == "truth" else "truth"

def plot_roc(y_true, proba_pos, pos_label: str, out_png: str):
    fpr, tpr, _ = roc_curve(y_true, proba_pos, pos_label=pos_label)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_explained_variance(cumvar, out_png: str):
    plt.figure()
    plt.plot(range(1, len(cumvar)+1), cumvar)
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Explained Variance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_roc_plot(y_true, y_prob, path, title="ROC"):
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
