import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

def main():
    # 1. import
    model_path = os.path.join(BASE_DIR, "models", "final", "logreg_dau.pkl")
    clf = joblib.load(model_path)
    print("Loaded model from:", model_path)

    # 2. import TEST set (DAU)
    test_path = os.path.join(BASE_DIR, "data", "kaggle_openface_merged.csv")
    print("Loading test data from:", test_path)
    test = pd.read_csv(test_path)

    X_test = test.drop(columns=["video_id", "label"])
    y_test = test["label"]

    print("Test shape:", X_test.shape)

    # 3. Predict στο TEST
    print(">>> Predict στο test...")
    pred_test = clf.predict(X_test)

    # 4. Metrics
    acc_test = accuracy_score(y_test, pred_test)
    cm_test  = confusion_matrix(y_test, pred_test, labels=["lie", "truth"])

    print("\n=== Evaluation on TEST (DAU) ===")
    print("Test accuracy:", acc_test)
    print("Test confusion matrix:\n", cm_test)
    print(classification_report(y_test, pred_test))

if __name__ == "__main__":
    main()
