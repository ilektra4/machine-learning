from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# PCA - for feature selection - could be optional step hence isolated
# ---------------------------------------------------------------------

def fit_pca(X_train, X_valid, X_test, variance_keep: float = 0.95, seed: int = 42):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_valid)
    Xte = scaler.transform(X_test)

    pca_full = PCA(random_state=seed)
    pca_full.fit(Xtr)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.argmax(cum >= variance_keep) + 1)

    pca = PCA(n_components=k, random_state=seed)
    Ztr = pca.fit_transform(Xtr)
    Zva = pca.transform(Xva)
    Zte = pca.transform(Xte)
    return Ztr, Zva, Zte, pca, scaler, k, cum
