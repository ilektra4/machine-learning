from sklearn.linear_model import LogisticRegression

def fit_logistic(X_train, y_train, C: float = 1.0, seed: int = 42):
    clf = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=1.0,
        C=C,
        max_iter=5000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf

