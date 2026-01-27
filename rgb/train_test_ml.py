import argparse
from joblib import load

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def build_model(model_name: str, pca_dim: int | None, seed: int):
    """
    Returns a sklearn Pipeline.
    - For logreg: StandardScaler -> (optional PCA) -> LogisticRegression
    - For rf: (optional PCA) -> RandomForest (no scaler needed generally)
    """

    if model_name == "logreg":
        steps = [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
        if pca_dim is not None and pca_dim > 0:
            steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
        steps.append(
            (
                "clf",
                LogisticRegression(
                    C=1.0,                 # μπορείς να το κάνεις arg αν θες tuning
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",         # σταθερό για binary
                    n_jobs=None,
                ),
            )
        )
        return Pipeline(steps)

    if model_name == "rf":
        steps = []
        # PCA πριν από RF καμιά φορά βοηθάει σε πολύ μεγάλα dims
        if pca_dim is not None and pca_dim > 0:
            steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                ),
            )
        )
        return Pipeline(steps)

    raise ValueError("model must be one of: logreg, rf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to *.joblib (from extract_features.py)")
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    ap.add_argument("--pca", type=int, default=0, help="PCA dims (0 = no PCA). Suggested: 128/256/512")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load(args.features)
    X, y, groups = data["X"], data["y"], data["groups"]

    # Split by participant (groups) to avoid leakage
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(X, y, groups=groups))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(
        "Train:", len(y_tr),
        "Test:", len(y_te),
        "| truth:", int((y_te == 0).sum()),
        "deception:", int((y_te == 1).sum())
    )

    pca_dim = args.pca if args.pca and args.pca > 0 else None
    model = build_model(args.model, pca_dim, args.seed)

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)

    print(f"Model={args.model} PCA={pca_dim if pca_dim else 'none'}")
    print(f"Test acc={acc:.4f} f1={f1:.4f}")
    print("\nReport:\n", classification_report(y_te, pred, target_names=["truth", "deception"]))


if __name__ == "__main__":
    main()
