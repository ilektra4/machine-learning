import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # 💾 για αποθήκευση/φόρτωμα μοντέλου

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
)

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")

# Φόρτωμα train/test (STATIC aggregation)
train = pd.read_csv(os.path.join(DATA_DIR, "train_agg_static.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test_agg_static.csv"))

X_train = train.drop(columns=["video_id", "label"])
y_train = train["label"]

X_test = test.drop(columns=["video_id", "label"])
y_test = test["label"]

# ✅ Imputer -> Scaler -> SVM (RBF)
clf = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

print(">>> Κάνω fit στο train...")
clf.fit(X_train, y_train)
print(">>> Το μοντέλο εκπαιδεύτηκε.")

pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred, labels=["lie", "truth"])
rep_txt = classification_report(y_test, pred)

# --- F1 scores ---
f1_macro = f1_score(y_test, pred, average="macro")
f1_weighted = f1_score(y_test, pred, average="weighted")

# --- AUC ---
# Χρησιμοποιούμε την απόσταση από το υπερεπίπεδο (decision_function)
# και ορίζουμε ως "θετική" κλάση την "truth"
scores = clf.decision_function(X_test)
y_bin = (y_test == "truth").astype(int)
auc = roc_auc_score(y_bin, scores)

print("=== SVM (RBF) ===")
print("Accuracy:", acc)
print("F1 macro:", f1_macro)
print("F1 weighted:", f1_weighted)
print("ROC AUC (positive='truth'):", auc)
print("Confusion matrix:\n", cm)
print(rep_txt)

# --------- Confusion matrix plot ----------
plt.figure()
plt.imshow(cm)
plt.title(f"SVM (RBF) - Confusion Matrix (acc={acc:.3f})")
plt.xticks([0, 1], ["lie", "truth"])
plt.yticks([0, 1], ["lie", "truth"])
plt.xlabel("Predicted")
plt.ylabel("True")

for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

out_cm = os.path.join(OUTPUTS_DIR, "cm_svm_rbf.png")
plt.savefig(out_cm, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", out_cm)

# --------- 💾 Αποθήκευση εκπαιδευμένου μοντέλου ----------
model_path = os.path.join(OUTPUTS_DIR, "svm_rbf_static.pkl")
joblib.dump(clf, model_path)
print("Saved trained SVM model to:", model_path)
