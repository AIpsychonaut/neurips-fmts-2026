"""Observation-time geometry: rank maps, OLS residualization, quantile warp."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

from sparse_time_probes.config import KEEP_ALPHA, KEEP_BETA


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        return x - x.mean()
    return (x - x.mean()) / sd


def unit_rank(k: np.ndarray, n: int | None = None) -> np.ndarray:
    """Normalized rank \(\tilde k = (k-1)/(n-1)\) for 1-based index k."""
    k = np.asarray(k, dtype=np.float64)
    n_tok = int(k.size if n is None else n)
    if n_tok < 2:
        raise ValueError("need at least two tokens")
    return (k - 1.0) / (n_tok - 1.0)


def observed_times(keep_t: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 1-based index k, 0-based i, and normalized times m in [0, 1]."""
    keep_t = np.asarray(keep_t)
    n = keep_t.size
    i = np.arange(n, dtype=np.float64)
    k = i + 1.0
    m = keep_t.astype(np.float64) / float(max(T - 1, 1))
    return k, i, m


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y on [1, x]. Affine in x, so 0-based and 1-based index agree."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def ols_coef(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def ols_fit(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    coef = ols_coef(y, x)
    return coef[0] + coef[1] * x


def beta_quantile(
    tilde_k: np.ndarray,
    alpha: float = KEEP_ALPHA,
    beta: float = KEEP_BETA,
) -> np.ndarray:
    """Inverse CDF \(Q_{\mathrm{Beta}}(\tilde k)\)."""
    u = np.clip(np.asarray(tilde_k, dtype=np.float64), 0.0, 1.0)
    return beta_dist.ppf(u, alpha, beta)


def mean_kept_times(
    T: int,
    p_drop: float,
    n_rep: int = 400,
    seed: int = 12_345,
) -> np.ndarray:
    """Monte Carlo \(\mathbb{E}[m_k]\) under the paper keep operator."""
    from sparse_time_probes.keep import keep_beta

    samples = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        keep = keep_beta(T, p_drop, rng)
        _, _, m = observed_times(keep, T)
        samples.append(m)
    return np.mean(np.stack(samples, axis=0), axis=0)


def psi_curve(
    k: np.ndarray,
    n: int | None = None,
    alpha: float = KEEP_ALPHA,
    beta: float = KEEP_BETA,
) -> np.ndarray:
    """Quantile warp minus its OLS line, with OLS fitted on the n token indices."""
    k = np.asarray(k, dtype=np.float64)
    n_tok = int(k.size if n is None else n)
    k_fit = np.arange(1.0, n_tok + 1.0)
    q_fit = beta_quantile(unit_rank(k_fit, n=n_tok), alpha=alpha, beta=beta)
    coef = ols_coef(q_fit, k_fit)
    q = beta_quantile(unit_rank(k, n=n_tok), alpha=alpha, beta=beta)
    return q - (coef[0] + coef[1] * k)
