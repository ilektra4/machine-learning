import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# Φάκελος που είναι το script + τα .pkl
SCRIPT_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(SCRIPT_DIR, "svm_linear_movaver.pkl")
TEST_PATH  = os.path.join(SCRIPT_DIR, "movaver_test.pkl")

def main():
    # 1. Φόρτωμα μοντέλου
    print(">>> Φορτώνω SVM Linear movaver από:", MODEL_PATH)
    clf = joblib.load(MODEL_PATH)

    # 2. Φόρτωμα movaver_test
    print(">>> Φορτώνω movaver_test από:", TEST_PATH)
    df = pd.read_pickle(TEST_PATH)
    print("Test shape:", df.shape)

    if "label" not in df.columns:
        raise ValueError("Το movaver_test.pkl δεν έχει στήλη 'label'.")

    # Πετάμε label + ό,τι δεν είναι feature
    drop_cols = ["label"]
    for c in ["video_id", "face_id"]:
        if c in df.columns:
            drop_cols.append(c)

    X_test = df.drop(columns=drop_cols)
    y_test = df["label"]

    print("X_test shape:", X_test.shape)

    # 3. Predict
    print(">>> Predict στο movaver_test με SVM Linear...")
    pred = clf.predict(X_test)

    # 4. Metrics
    acc = accuracy_score(y_test, pred)
    cm  = confusion_matrix(y_test, pred, labels=["lie", "truth"])
    f1_macro    = f1_score(y_test, pred, average="macro")
    f1_weighted = f1_score(y_test, pred, average="weighted")

    print("\n=== SVM Linear movaver on movaver_test ===")
    print("Accuracy:", acc)
    print("F1 macro:", f1_macro)
    print("F1 weighted:", f1_weighted)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, pred))

if __name__ == "__main__":
    main()
