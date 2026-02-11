import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

def main():
    print(">>> Ξεκινάω training script (DAU)")

    # === Φόρτωμα DAU datasets ===
    train_path = os.path.join(BASE_DIR, "data", "train_agg_dau.csv")
    test_path  = os.path.join(BASE_DIR, "data", "test_agg_dau.csv")
    print("Φορτώνω:", train_path)
    print("Φορτώνω:", test_path)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    X_train = train.drop(columns=["video_id", "label"])
    y_train = train["label"]
    X_test  = test.drop(columns=["video_id", "label"])
    y_test  = test["label"]

    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

    # === Pipeline μοντέλου ===
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
    )

    # === Εκπαίδευση στο TRAIN ===
    print(">>> Κάνω fit στο train...")
    clf.fit(X_train, y_train)
    print(">>> Το μοντέλο εκπαιδεύτηκε.")

    # === Αξιολόγηση στο TEST ===
    print(">>> Υπολογίζω προβλέψεις στο test...")
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=["lie", "truth"])
    rep = classification_report(y_test, pred, output_dict=True)

    print("=== Logistic Regression (DAU) ===")
    print("Accuracy (test):", acc)
    print("Confusion matrix (test):\n", cm)
    print(classification_report(y_test, pred))

    # ---------- Plot: Confusion Matrix ----------
    plt.figure()
    plt.imshow(cm)
    plt.title(f"LogReg DAU - Confusion Matrix (acc={acc:.3f})")
    plt.xticks([0, 1], ["lie", "truth"])
    plt.yticks([0, 1], ["lie", "truth"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    cm_path = os.path.join(BASE_DIR, "cm_logistic_dau.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    print("Saved confusion matrix image:", cm_path)

    # ---------- Plot: Precision/Recall/F1 per class ----------
    classes = ["lie", "truth"]
    prec = [rep[c]["precision"] for c in classes]
    rec  = [rep[c]["recall"] for c in classes]
    f1   = [rep[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    plt.figure()
    plt.bar(x - w, prec, width=w, label="precision")
    plt.bar(x,     rec,  width=w, label="recall")
    plt.bar(x + w, f1,   width=w, label="f1")
    plt.xticks(x, classes)
    plt.ylim(0, 1)
    plt.title("LogReg DAU - Metrics per class")
    plt.legend()

    metrics_path = os.path.join(BASE_DIR, "metrics_logistic_dau.png")
    plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
    print("Saved metrics image:", metrics_path)

    # === Αποθήκευση εκπαιδευμένου μοντέλου ===
    model_path = os.path.join(BASE_DIR, "logreg_dau.pkl")
    joblib.dump(clf, model_path)
    print("Saved trained model to:", model_path)

    print(">>> ΤΕΛΟΣ training script (DAU)")

if __name__ == "__main__":
    main()
