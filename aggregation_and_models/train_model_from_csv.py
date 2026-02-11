import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ========= ΡΥΘΜΙΣΕΙΣ =========

# Φάκελος όπου είναι το csv
BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

# ΕΔΩ βάζεις το train αρχείο σου.
# Αν θες static αντί για dau, άλλαξε το σε "train_agg_static.csv"
TRAIN_CSV_NAME = "train_agg_dau.csv"

# Όνομα για το αποθηκευμένο μοντέλο
MODEL_NAME = "logreg_from_train_dau.pkl"


def main():
    print(">>> Ξεκινάω training πάνω στο:", TRAIN_CSV_NAME)

    # ----- Φόρτωμα train csv -----
    train_path = os.path.join(BASE_DIR, "data", TRAIN_CSV_NAME)
    print("Φορτώνω train από:", train_path)

    train = pd.read_csv(train_path)

    # Περιμένουμε στήλες: video_id, label, + features
    if "video_id" not in train.columns or "label" not in train.columns:
        raise ValueError("Πρέπει το csv να έχει στήλες 'video_id' και 'label'.")

    X_train = train.drop(columns=["video_id", "label"])
    y_train = train["label"]

    print("Train shape (rows, features):", X_train.shape)

    # ----- Ορισμός μοντέλου (pipeline) -----
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=42
        )
    )

    # ----- Εκπαίδευση -----
    print(">>> Κάνω fit στο train...")
    clf.fit(X_train, y_train)
    print(">>> Το μοντέλο εκπαιδεύτηκε.")

    # ----- Προαιρετικός έλεγχος πάνω στο train -----
    print(">>> Υπολογίζω απόδοση πάνω στο train (για έλεγχο)...")
    pred_train = clf.predict(X_train)

    acc_train = accuracy_score(y_train, pred_train)
    print("\n=== Evaluation on TRAIN ===")
    print("Train accuracy:", acc_train)
    print(classification_report(y_train, pred_train))

    # ----- Αποθήκευση εκπαιδευμένου μοντέλου -----
    model_path = os.path.join(BASE_DIR, MODEL_NAME)
    joblib.dump(clf, model_path)
    print("\n>>> Αποθήκευσα το εκπαιδευμένο μοντέλο στο:", model_path)
    print(">>> ΤΕΛΟΣ.")


if __name__ == "__main__":
    main()
