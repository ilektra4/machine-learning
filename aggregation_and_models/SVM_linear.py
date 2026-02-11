import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")

train = pd.read_csv(os.path.join(DATA_DIR, "train_agg_static.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test_agg_static.csv"))

X_train = train.drop(columns=["video_id", "label"])
y_train = train["label"]

X_test = test.drop(columns=["video_id", "label"])
y_test = test["label"]

clf = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LinearSVC(C=1.0, random_state=42, max_iter=20000)
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=["lie", "truth"])
rep = classification_report(y_test, pred, output_dict=True)

print("=== SVM (Linear) ===")
print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print(classification_report(y_test, pred))

# -----------------------------
# Plot 1: Confusion Matrix
# -----------------------------
plt.figure()
plt.imshow(cm)
plt.title(f"SVM (Linear) - Confusion Matrix (acc={acc:.3f})")
plt.xticks([0, 1], ["lie", "truth"])
plt.yticks([0, 1], ["lie", "truth"])
plt.xlabel("Predicted")
plt.ylabel("True")

for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
cm_path = os.path.join(OUTPUTS_DIR, "cm_svm_linear.png")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
print("Saved:", cm_path)

# -----------------------------
# Plot 2 (optional): Metrics bars
# -----------------------------
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
plt.title("SVM (Linear) - Metrics per class")
plt.legend()

metrics_path = os.path.join(OUTPUTS_DIR, "metrics_svm_linear.png")
plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
print("Saved:", metrics_path)
