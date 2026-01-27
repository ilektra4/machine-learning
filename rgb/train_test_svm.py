import argparse
from joblib import load

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features.joblib (e.g., hog_K16.joblib)")
    ap.add_argument("--pca", type=int, default=256, help="PCA dims (0 = disable PCA). Suggested: 128/256/512")
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC C")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load(args.features)
    X, y, groups = data["X"], data["y"], data["groups"]

    # Group split to avoid leakage (same participant in train & test)
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

    steps = [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ]
    if args.pca and args.pca > 0:
        steps.append(("pca", PCA(n_components=args.pca, random_state=args.seed)))
    steps.append(("svm", LinearSVC(C=args.C, class_weight="balanced", max_iter=10000)))

    clf = Pipeline(steps)

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred)

    pca_str = str(args.pca) if (args.pca and args.pca > 0) else "none"
    print(f"SVM model: LinearSVC(C={args.C}) | PCA={pca_str}")
    print(f"Test acc={acc:.4f} f1={f1:.4f}")
    print("\nReport:\n", classification_report(y_te, pred, target_names=["truth", "deception"]))


if __name__ == "__main__":
    main()
