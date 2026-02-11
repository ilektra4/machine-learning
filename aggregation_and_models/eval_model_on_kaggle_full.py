import os
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

# ========= ΡΥΘΜΙΣΕΙΣ ΠΟΥ ΠΡΕΠΕΙ ΝΑ ΕΙΝΑΙ ΣΩΣΤΕΣ =========
# 1) Ο φάκελος με τα αρχεία σου
BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 2) Το RAW kaggle αρχείο με τα frames
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_openface_merged.csv")

# 3) Το ΟΝΟΜΑ του εκπαιδευμένου μοντέλου που έχεις ήδη σώσει
MODEL_PATH = os.path.join(MODELS_DIR, "logreg_from_train_dau.pkl")

# γι’ αυτό που είχαμε κάνει για DAU
TIME_CANDIDATES = ["timestamp", "frame"]


def find_time_col(frames: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in frames.columns:
            return c
    raise ValueError(
        f"Για DAU χρειάζομαι μία στήλη χρόνου. "
        f"Δεν βρέθηκε καμία από: {TIME_CANDIDATES}"
    )


def add_dau(frames: pd.DataFrame, au_cols):
    """Υπολογίζει d_AUxx_r = διαφορά από προηγούμενο frame μέσα στο ίδιο video."""
    time_col = find_time_col(frames)
    frames = frames.sort_values(["video_id", time_col]).copy()

    for c in au_cols:
        frames[f"d_{c}"] = frames.groupby("video_id")[c].diff().fillna(0)

    return frames


def aggregate_dau(frames: pd.DataFrame, au_cols):
    """Aggregation ανά video_id για τα d_AUxx_r (mean/std/max)."""
    dau_cols = [f"d_{c}" for c in au_cols if f"d_{c}" in frames.columns]
    if not dau_cols:
        raise ValueError("Δεν βρέθηκαν DAU columns. Κάτι πήγε στραβά στο add_dau().")

    X = frames.groupby("video_id")[dau_cols].agg(["mean", "std", "max"])
    X.columns = [f"{c}_{stat}" for c, stat in X.columns]  # flatten multiindex

    # label ανά video_id
    y = frames.groupby("video_id")["label"].first()
    return X.join(y).reset_index()


def main():
    # ----- Φόρτωμα kaggle raw αρχείου -----
    print(">>> Φορτώνω kaggle αρχείο από:", KAGGLE_CSV)
    df = pd.read_csv(KAGGLE_CSV)

    # basic checks
    need = ["video_id", "label", "success", "confidence"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Λείπουν βασικές στήλες: {missing}")
    print("Rows (frames):", len(df))

    # AU στήλες (_r)
    au_cols = [c for c in df.columns if c.endswith("_r")]
    if not au_cols:
        raise ValueError("Δεν βρέθηκαν AU *_r στήλες στο kaggle αρχείο.")
    print("Βρέθηκαν AU columns:", len(au_cols))

    # ----- DAU + aggregation ανά video_id -----
    print(">>> Υπολογίζω DAU (delta AU) ανά frame...")
    df_dau = add_dau(df, au_cols)

    print(">>> Κάνω aggregation (mean/std/max) ανά video_id...")
    full_agg = aggregate_dau(df_dau, au_cols)

    print("Full aggregated shape (videos x features+label):", full_agg.shape)

    X_all = full_agg.drop(columns=["video_id", "label"])
    y_all = full_agg["label"]

    print("X_all shape:", X_all.shape, "| αριθμός videos:", len(full_agg))

    # ----- Φόρτωμα εκπαιδευμένου μοντέλου -----
    print(">>> Φορτώνω εκπαιδευμένο μοντέλο από:", MODEL_PATH)
    clf = joblib.load(MODEL_PATH)

    # ----- Predict σε ΟΛΑ τα videos -----
    print(">>> Predict σε όλα τα videos (aggregated DAU)...")
    pred_all = clf.predict(X_all)

    acc_all = accuracy_score(y_all, pred_all)

    # F1 scores
    f1_macro = f1_score(y_all, pred_all, average="macro")
    f1_weighted = f1_score(y_all, pred_all, average="weighted")

    # AUC: positive class = "truth"
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_all)
        classes = list(clf.classes_)
        if "truth" in classes:
            idx_truth = classes.index("truth")
            scores = proba[:, idx_truth]
            y_bin = (y_all == "truth").astype(int)
            auc = roc_auc_score(y_bin, scores)
        else:
            auc = None
            print("ΠΡΟΣΟΧΗ: δεν βρέθηκε κλάση 'truth' στο clf.classes_.")
    else:
        auc = None
        print("ΠΡΟΣΟΧΗ: Το μοντέλο δεν υποστηρίζει predict_proba, δεν μπορώ να βγάλω AUC.")

    print("\n=== Evaluation on FULL KAGGLE (aggregated DAU per video) ===")
    print("Accuracy:", acc_all)
    print("F1 macro:", f1_macro)
    print("F1 weighted:", f1_weighted)
    if auc is not None:
        print("ROC AUC (positive='truth'):", auc)
    print(classification_report(y_all, pred_all))


if __name__ == "__main__":
    main()
