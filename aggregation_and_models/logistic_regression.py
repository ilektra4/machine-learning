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

# ----------------- ΡΥΘΜΙΣΕΙΣ PATHS -----------------

# Φάκελος που βρίσκεται ΤΟ SCRIPT (δηλαδή ο FINAL_DOLOS)
SCRIPT_DIR = os.path.dirname(__file__)

# Εκεί είναι τα train/test csv (δεν τα πειράζουμε)
DATA_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs\data"

# Εκεί θες να αποθηκεύονται ΟΛΑ τα outputs (png + pkl)
OUTPUTS_DIR = SCRIPT_DIR

# ---------------------------------------------------

train = pd.read_csv(os.path.join(DATA_DIR, "train_agg_static.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test_agg_static.csv"))

X_train = train.drop(columns=["video_id", "label"])
y_train = train["label"]
X_test  = test.drop(columns=["video_id", "label"])
y_test  = test["label"]

clf = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
)

print(">>> Κάνω fit στο train...")
clf.fit(X_train, y_train)
print(">>> Το μοντέλο εκπαιδεύτηκε.")

pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=["lie", "truth"])
rep = classification_report(y_test, pred, output_dict=True)

print("=== Logistic Regression (static) ===")
print("Accuracy:", acc)
print("Confusion matrix:\n", cm)
print(classification_report(y_test, pred))

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------- Confusion Matrix ----------
plt.figure()
plt.imshow(cm)
plt.title(f"Logistic Regression - Confusion Matrix (acc={acc:.3f})")
plt.xticks([0, 1], ["lie", "truth"])
plt.yticks([0, 1], ["lie", "truth"])
plt.xlabel("Predicted")
plt.ylabel("True")

for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")

cm_path = os.path.join(OUTPUTS_DIR, "cm_logistic_static.png")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
print("Saved:", cm_path)

# ---------- Precision/Recall/F1 ----------
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
plt.title("Logistic Regression - Metrics per class")
plt.legend()

metrics_path = os.path.join(OUTPUTS_DIR, "metrics_logistic_static.png")
plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
print("Saved:", metrics_path)

# ---------- 💾 Αποθήκευση εκπαιδευμένου μοντέλου ----------
model_path = os.path.join(OUTPUTS_DIR, "logreg_static.pkl")
joblib.dump(clf, model_path)
print("Saved trained Logistic Regression model to:", model_path)
