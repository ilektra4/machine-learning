import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"

train = pd.read_csv(f"{BASE_DIR}\\data\\train_agg_static.csv")
test  = pd.read_csv(f"{BASE_DIR}\\data\\test_agg_static.csv")

X_train = train.drop(columns=["video_id", "label"])
y_train = train["label"]

X_test = test.drop(columns=["video_id", "label"])
y_test = test["label"]

clf = make_pipeline(
    SimpleImputer(strategy="median"),   # <-- fix για NaN
    RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
