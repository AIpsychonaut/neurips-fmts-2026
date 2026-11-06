"""Index-only control tasks for residualized observation time."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from sparse_time_probes.config import POLY_DEGREE, SPLINE_DEGREE, SPLINE_KNOTS
from sparse_time_probes.geometry import residualize, zscore
from sparse_time_probes.probe import ridge_cv_r


def poly1_features(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    return np.column_stack([np.ones_like(k), k])


def poly3_features(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
    return PolynomialFeatures(degree=POLY_DEGREE, include_bias=True).fit_transform(k)


def spline_features(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
    return SplineTransformer(
        n_knots=SPLINE_KNOTS,
        degree=SPLINE_DEGREE,
        include_bias=True,
    ).fit_transform(k)


def control_task_scores(k: np.ndarray, m: np.ndarray, seed: int) -> dict[str, float]:
    """Held-out ridge r for poly1 / poly3 / spline of k predicting m^perp."""
    m_perp = zscore(residualize(m, k))
    return {
        "r_poly1": ridge_cv_r(poly1_features(k), m_perp, seed=seed),
        "r_poly3": ridge_cv_r(poly3_features(k), m_perp, seed=seed),
        "r_spline": ridge_cv_r(spline_features(k), m_perp, seed=seed),
    }
