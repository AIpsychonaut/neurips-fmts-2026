from __future__ import annotations

import numpy as np

from sparse_time_probes.config import BASE_SEED, DATASETS, P_PRIMARY
from sparse_time_probes.geometry import observed_times, residualize, unit_rank
from sparse_time_probes.keep import keep_beta
from sparse_time_probes.probe import pearson_r
from sparse_time_probes.results import load_paper_results


def test_pinned_endpoints_and_rank():
    rng = np.random.default_rng(0)
    T = 200
    keep = keep_beta(T, 0.5, rng)
    assert keep[0] == 0 and keep[-1] == T - 1
    k, _, m = observed_times(keep, T)
    rank = unit_rank(k)
    assert abs(rank[0] - 0.0) < 1e-12
    assert abs(rank[-1] - 1.0) < 1e-12
    assert abs(m[0] - 0.0) < 1e-12
    assert abs(m[-1] - 1.0) < 1e-12


def test_residual_orthogonal_to_index():
    rng = np.random.default_rng(1)
    keep = keep_beta(400, 0.5, rng)
    k, _, m = observed_times(keep, 400)
    m_perp = residualize(m, k)
    assert abs(pearson_r(m_perp, k)) < 1e-10


def test_unit_rank_uses_token_count_not_grid_length():
    n = 50
    grid = np.linspace(1.0, n, 400)
    rank = unit_rank(grid, n=n)
    assert abs(rank[0] - 0.0) < 1e-12
    assert abs(rank[-1] - 1.0) < 1e-12


def test_keep_matches_frozen_corr():
    frozen = {row["name"]: row for row in load_paper_results()["series"]}
    for idx, name in enumerate(DATASETS):
        row = frozen[name]
        rng = np.random.default_rng(BASE_SEED + idx)
        keep = keep_beta(int(row["T"]), P_PRIMARY, rng)
        _, i, m = observed_times(keep, int(row["T"]))
        got = pearson_r(i, m)
        assert abs(got - row["corr_i_m"]) < 1e-9, (name, got, row["corr_i_m"])
