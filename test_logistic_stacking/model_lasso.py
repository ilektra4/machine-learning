from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# -------------------------------------------------------------------------------------
# LASSO Logistic - Third experimental model selection - remove PCA and check efficiency
# -------------------------------------------------------------------------------------

def fit_lasso(X_train, y_train, C: float = 1.0, seed: int = 42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            random_state=seed,
            max_iter=10000,
            n_jobs=-1
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe
