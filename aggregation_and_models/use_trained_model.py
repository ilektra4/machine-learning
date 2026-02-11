import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

# 1. Φόρτωσε το αποθηκευμένο μοντέλο
model_path = os.path.join(BASE_DIR, "logreg_from_train_dau.pkl")
clf = joblib.load(model_path)
print("Loaded model from:", model_path)

# 2. Φόρτωσε κάποιο csv με ΙΔΙΑ features (π.χ. test_agg_dau.csv)
test_path = os.path.join(BASE_DIR, "test_agg_dau.csv")
test = pd.read_csv(test_path)

X_test = test.drop(columns=["video_id", "label"])
y_test = test["label"]

# 3. Κάνε predict
pred = clf.predict(X_test)

# 4. Δες πόσο καλά πάει
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
