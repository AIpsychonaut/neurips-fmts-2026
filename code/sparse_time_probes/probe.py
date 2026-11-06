"""Held-out ridge probes. Series-level Pearson r; never token-pooled p-values."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from sparse_time_probes.config import N_SPLITS, RIDGE_ALPHA


def pearson_r(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    if y_true.std() < 1e-12 or y_hat.std() < 1e-12:
        return 0.0
    c = np.corrcoef(y_true, y_hat)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def ridge_cv_r(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
    n_splits: int = N_SPLITS,
    seed: int = 0,
) -> float:
    """Mean Pearson r of ridge predictions on shuffled K-fold held-out splits."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if X.shape[0] < max(16, n_splits * 3):
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train, test in kf.split(X):
        if test.size < 4:
            continue
        model = Ridge(alpha=alpha)
        model.fit(X[train], y[train])
        scores.append(pearson_r(y[test], model.predict(X[test])))
    if not scores:
        return float("nan")
    return float(np.mean(scores))
