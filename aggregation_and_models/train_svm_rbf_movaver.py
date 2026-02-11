import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(__file__)

TRAIN_PATH = os.path.join(SCRIPT_DIR, "movaver_train.pkl")
TEST_PATH  = os.path.join(SCRIPT_DIR, "movaver_test.pkl")
MODEL_PATH = os.path.join(SCRIPT_DIR, "svm_rbf_movaver.pkl")

def main():
    # ====== LOAD TRAIN ======
    print(">>> Φορτώνω movaver_train από:", TRAIN_PATH)
    train_df = pd.read_pickle(TRAIN_PATH)
    print("Train shape:", train_df.shape)

    if "label" not in train_df.columns:
        raise ValueError("Δεν βρήκα στήλη 'label' στο movaver_train.pkl")

    drop_cols = ["label"]
    for c in ["video_id", "face_id"]:
        if c in train_df.columns:
            drop_cols.append(c)

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["label"]

    print("X_train shape:", X_train.shape)

    # ====== SVM RBF MODEL ======
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale")
    )

    print(">>> Κάνω fit στο movaver_train (SVM RBF)...")
    clf.fit(X_train, y_train)
    print(">>> Το μοντέλο εκπαιδεύτηκε.")

    # ====== LOAD TEST ======
    print(">>> Φορτώνω movaver_test από:", TEST_PATH)
    test_df = pd.read_pickle(TEST_PATH)
    print("Test shape:", test_df.shape)

    if "label" not in test_df.columns:
        raise ValueError("Δεν βρήκα στήλη 'label' στο movaver_test.pkl")

    drop_cols_test = ["label"]
    for c in ["video_id", "face_id"]:
        if c in test_df.columns:
            drop_cols_test.append(c)

    X_test = test_df.drop(columns=drop_cols_test)
    y_test = test_df["label"]

    print("X_test shape:", X_test.shape)

    print(">>> Predict στο movaver_test (SVM RBF)...")
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm  = confusion_matrix(y_test, pred, labels=["lie", "truth"])

    print("\n=== SVM RBF on movaver_test ===")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, pred))

    # Confusion matrix plot
    plt.figure()
    plt.imshow(cm)
    plt.title(f"SVM RBF movaver - Confusion (acc={acc:.3f})")
    plt.xticks([0, 1], ["lie", "truth"])
    plt.yticks([0, 1], ["lie", "truth"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    cm_path = os.path.join(SCRIPT_DIR, "cm_svm_rbf_movaver.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    print("Saved confusion matrix:", cm_path)

    # SAVE MODEL
    joblib.dump(clf, MODEL_PATH)
    print("Saved SVM RBF movaver model to:", MODEL_PATH)

if __name__ == "__main__":
    main()
