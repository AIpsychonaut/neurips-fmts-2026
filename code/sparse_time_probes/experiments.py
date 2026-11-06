"""CPU control-task and optional Chronos encoder audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparse_time_probes.config import (
    BASE_SEED,
    DATASETS,
    LAYERS,
    MODEL_ID,
    P_GRID,
    P_PRIMARY,
    PRIMARY_LAYER,
)
from sparse_time_probes.control_task import control_task_scores
from sparse_time_probes.geometry import observed_times, residualize, zscore
from sparse_time_probes.keep import keep_beta
from sparse_time_probes.probe import pearson_r, ridge_cv_r
from sparse_time_probes.results import load_paper_results


def run_control_task() -> dict[str, Any]:
    """Recompute poly/spline ceilings from keep geometry (no model weights)."""
    frozen = {row["name"]: row for row in load_paper_results()["series"]}
    rows = []
    for idx, name in enumerate(DATASETS):
        spec = frozen[name]
        T = int(spec["T"])
        seed = BASE_SEED + idx
        rng = np.random.default_rng(seed)
        keep = keep_beta(T, P_PRIMARY, rng)
        k, i, m = observed_times(keep, T)
        scores = control_task_scores(k, m, seed=seed)
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "T": T,
                "n": int(keep.size),
                "corr_i_m": pearson_r(i, m),
                **scores,
            }
        )
    keys = ("corr_i_m", "r_poly1", "r_poly3", "r_spline")
    medians = {key: float(np.median([row[key] for row in rows])) for key in keys}
    return {"rows": rows, "medians": medians}


def run_chronos_audit(cache_dir, *, p_grid=P_GRID, layers=LAYERS):
    """Full encoder audit. Requires the optional Chronos extra and a GPU."""
    import torch
    from chronos import ChronosPipeline

    from sparse_time_probes.chronos_encoder import encoder_states
    from sparse_time_probes.ucr import load_series

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chronos encoder audit")

    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16
    )
    all_rows = []
    for idx, name in enumerate(DATASETS):
        y = load_series(name, cache_dir)
        seed = BASE_SEED + idx
        for p_drop in p_grid:
            all_rows.extend(
                _condition(pipeline, y, name, p_drop, seed, layers, noise=False)
            )
        all_rows.extend(
            _condition(
                pipeline, y, name, P_PRIMARY, seed, (PRIMARY_LAYER,), noise=True
            )
        )
    return {"model": MODEL_ID, "encoder_only": True, "rows": all_rows}


def _condition(pipeline, y, name, p_drop, seed, layers, noise):
    from sparse_time_probes.chronos_encoder import encoder_states

    rng = np.random.default_rng(seed)
    T = int(y.size)
    keep = keep_beta(T, p_drop, rng)
    x = y[keep].copy()
    if noise:
        x = rng.normal(size=x.shape)
    k, i, m = observed_times(keep, T)
    hidden = encoder_states(pipeline, x, layers=layers)
    n = min(next(iter(hidden.values())).shape[0], keep.size)
    i, m, k = i[:n], m[:n], k[:n]
    m_perp = residualize(m, k)
    rows = []
    for layer, H in hidden.items():
        H = H[:n]
        rows.append(
            {
                "dataset": name,
                "p_drop": float(p_drop),
                "layer": int(layer),
                "noise": bool(noise),
                "n": int(H.shape[0]),
                "T": T,
                "corr_i_m": pearson_r(i, m),
                "r_ord": ridge_cv_r(H, zscore(i), seed=seed),
                "r_perp": ridge_cv_r(H, zscore(m_perp), seed=seed),
            }
        )
    return rows
