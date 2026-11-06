"""Sparse-time probe diagnostics for order-preserving observation masks."""

from sparse_time_probes.geometry import (
    beta_quantile,
    observed_times,
    psi_curve,
    residualize,
    unit_rank,
    zscore,
)
from sparse_time_probes.keep import keep_beta
from sparse_time_probes.probe import pearson_r, ridge_cv_r

__all__ = [
    "beta_quantile",
    "keep_beta",
    "observed_times",
    "pearson_r",
    "psi_curve",
    "residualize",
    "ridge_cv_r",
    "unit_rank",
    "zscore",
]
