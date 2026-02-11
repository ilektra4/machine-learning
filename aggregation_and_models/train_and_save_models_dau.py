import os
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import joblib

BASE_DIR = r"C:\Users\ilekt\Downloads\dolos_outputs"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(data_dir, prefix="dau"):
    train_path = os.path.join(data_dir, f"train_agg_{prefix}.csv")
    test_path = os.path.join(data_dir, f"test_agg_{prefix}.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def prepare_xy(df):
    X = df.drop(columns=["video_id", "label"])
    y = df["label"]
    return X, y

def build_models():
    models = {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
        ),
        "svm_rbf": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)
        ),
        "svm_linear": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(C=1.0, random_state=42, max_iter=20000)
        )
    }
    return models

def train_and_save(base_dir, data_dir, prefix="dau"):
    train, test = load_data(data_dir, prefix)
    X_train, y_train = prepare_xy(train)
    X_test = test.drop(columns=["video_id", "label"])

    models = build_models()
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        out_path = os.path.join(MODELS_DIR, f"{name}_dau.pkl")
        joblib.dump(clf, out_path)
        print(f"Saved model: {out_path}")

    # Create a kaggle-style submission using majority model (logistic here)
    # You can change which model to use for the submission by changing `pred_clf`.
    pred_clf = models["logistic"]
    preds = pred_clf.predict(X_test)

    submission = pd.DataFrame({
        "video_id": test["video_id"],
        "label": preds
    })
    sub_path = os.path.join(base_dir, "kaggle_submission_dau.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission CSV: {sub_path}")

if __name__ == "__main__":
    train_and_save(BASE_DIR, DATA_DIR, prefix="dau")
