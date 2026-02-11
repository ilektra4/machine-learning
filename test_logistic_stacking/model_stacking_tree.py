from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------------------------
# Decision Tree with Stacking - Second experimental model selection - attempt to improve accuracy
# -----------------------------------------------------------------------------------------------

def fit_tree(X_train, y_train, seed: int = 42):
    tree = DecisionTreeClassifier(
        random_state=seed,
        min_samples_leaf=10,
    )
    tree.fit(X_train, y_train)
    return tree

def fit_meta_logistic(p_log_train: np.ndarray, tree_pred_train: np.ndarray, y_train, seed: int = 42):
    X_meta = np.column_stack([p_log_train, tree_pred_train])
    meta = LogisticRegression(solver="liblinear", random_state=seed, max_iter=5000)
    meta.fit(X_meta, y_train)
    return meta

def predict_meta(meta, p_log: np.ndarray, tree_pred: np.ndarray) -> np.ndarray:
    X_meta = np.column_stack([p_log, tree_pred])
    return meta.predict_proba(X_meta)[:, 1]
