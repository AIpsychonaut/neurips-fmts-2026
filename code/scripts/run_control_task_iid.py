#!/usr/bin/env python3
"""IID-keep control-task ceilings on the frozen series lengths (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sparse_time_probes.config import BASE_SEED, DATASETS, P_PRIMARY
from sparse_time_probes.control_task import control_task_scores
from sparse_time_probes.geometry import observed_times
from sparse_time_probes.keep import keep_iid
from sparse_time_probes.probe import pearson_r
from sparse_time_probes.results import load_paper_results


def main() -> None:
    frozen = {row["name"]: row for row in load_paper_results()["series"]}
    rows = []
    for idx, name in enumerate(DATASETS):
        T = int(frozen[name]["T"])
        seed = BASE_SEED + idx
        rng = np.random.default_rng(seed)
        keep = keep_iid(T, P_PRIMARY, rng)
        k, i, m = observed_times(keep, T)
        scores = control_task_scores(k, m, seed=seed)
        rows.append(
            {
                "dataset": name,
                "T": T,
                "n": int(keep.size),
                "corr_i_m": pearson_r(i, m),
                **scores,
            }
        )
    med = {
        key: float(np.median([r[key] for r in rows]))
        for key in ("corr_i_m", "r_poly1", "r_poly3", "r_spline")
    }
    payload = {"keep": "iid_fixed_cardinality", "p_drop": P_PRIMARY, "rows": rows, "medians": med}
    out = ROOT / "results" / "control_task_iid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(med, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
