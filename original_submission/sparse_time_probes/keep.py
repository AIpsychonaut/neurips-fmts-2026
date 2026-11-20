"""Order-preserving keep operators with pinned endpoints."""

from __future__ import annotations

import numpy as np

from sparse_time_probes.config import KEEP_ALPHA, KEEP_BETA


def _n_keep(T: int, p_drop: float) -> int:
    n_keep = int(round((1.0 - p_drop) * T))
    return max(8, min(T, n_keep))


def keep_beta(
    T: int,
    p_drop: float,
    rng: np.random.Generator,
    alpha: float = KEEP_ALPHA,
    beta: float = KEEP_BETA,
) -> np.ndarray:
    """Return sorted unique grid indices of length ~ (1 - p_drop) T.

    Endpoints ``0`` and ``T - 1`` are always kept. Remaining times are sampled
    from a Beta(alpha, beta) density on the unit interval, mapped onto the
    discrete grid, and rejected until the target cardinality is met.
    Token order is chronological.
    """
    if T < 2:
        raise ValueError("T must be at least 2")
    if p_drop <= 0:
        return np.arange(T, dtype=np.int64)
    n_keep = _n_keep(T, p_drop)
    pts = {0, T - 1}
    guard = 0
    while len(pts) < n_keep and guard < 50_000:
        guard += 1
        t = int(np.clip(rng.beta(alpha, beta) * (T - 1), 0, T - 1))
        pts.add(t)
    if len(pts) < n_keep:
        raise RuntimeError("failed to assemble keep set")
    return np.sort(np.fromiter(pts, dtype=np.int64))


def keep_iid(
    T: int,
    p_drop: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fixed-cardinality iid keep: endpoints pinned, remaining indices uniform without replacement."""
    if T < 2:
        raise ValueError("T must be at least 2")
    if p_drop <= 0:
        return np.arange(T, dtype=np.int64)
    n_keep = _n_keep(T, p_drop)
    pts = {0, T - 1}
    interior = np.arange(1, T - 1, dtype=np.int64)
    need = n_keep - 2
    if need > 0:
        if interior.size < need:
            raise RuntimeError("T too small for requested keep")
        chosen = rng.choice(interior, size=need, replace=False)
        pts.update(int(x) for x in chosen)
    return np.sort(np.fromiter(pts, dtype=np.int64))
