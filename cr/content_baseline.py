"""Exploratory (not pre-registered) content-only baseline for the dt target.

Same keeps, series, targets and cross-series probe as the model runs, but the
"hidden states" are hand-built local features of the kept values:
  - step / bidirectional: y_{k-2..k+2}, first differences around k, |.| and squares
  - step / causal:        y_{k-3..k}, backward differences, |.| and squares (fair to a causal decoder)
  - patch:                per-patch summary of within-patch values and differences
Output: cr/results/content_baseline.csv

python cr/content_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402


def seed_for(s, law, p):  # identical to modal_run.seed_for
    return 42 + 1000 * s + 100 * core.LAWS.index(law) + int(round(p * 10))


def shift(x, s):
    out = np.full_like(x, np.nan)
    if s > 0:
        out[s:] = x[:-s]
    elif s < 0:
        out[:s] = x[-s:]
    else:
        out[:] = x
    return out


def step_feats(v, causal):
    lags = range(0, 4) if causal else range(-2, 3)
    cols = [shift(v, s) for s in lags]
    d1 = v - shift(v, 1)
    diffs = [d1, shift(d1, 1)] if causal else [d1, shift(d1, -1), shift(d1, 1)]
    cols += diffs + [np.abs(d) for d in diffs] + [d ** 2 for d in diffs]
    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0)


def patch_feats(v):
    pidx, n_p = core.patch_index(v.size)
    d = np.r_[np.nan, np.diff(v)]
    rows = []
    for q in range(n_p):
        sel = pidx == q
        vv, dd = v[sel], d[sel]
        dd = dd[~np.isnan(dd)]
        rows.append([vv.mean(), vv.std(), np.ptp(vv), np.abs(dd).mean() if dd.size else 0,
                     (dd ** 2).mean() if dd.size else 0, np.abs(dd).max() if dd.size else 0, sel.sum()])
    return np.asarray(rows)


def main():
    d = np.load(HERE / "series.npz")
    names = [k[3:] for k in d.files if k.startswith("s__")]
    ys = [d["s__" + n] for n in names]
    rows = []
    for law in core.LAWS:
        for p in core.PS:
            keeps = [core.draw_keep(law, y.size, p, np.random.default_rng(seed_for(s, law, p))) for s, y in enumerate(ys)]
            for fam in ("step_bidir", "step_causal", "patch"):
                patchy = fam == "patch"
                tg = [core.scaled(core.patch_targets(k, y.size) if patchy else core.step_targets(k, y.size))
                      for k, y in zip(keeps, ys)]
                Y = [np.column_stack([t[x] for x in core.TARGETS]) for t in tg]
                for inp in ("real", "noise"):
                    Hs = []
                    for s, (k, y) in enumerate(zip(keeps, ys)):
                        v = y[k] if inp == "real" else np.random.default_rng(seed_for(s, law, p) + 7_777_777).normal(size=k.size)
                        Hs.append(patch_feats(v) if patchy else step_feats(v, causal=(fam == "step_causal")))
                    per, pooled = core.cross_series_probe(Hs, Y, seed=seed_for(0, law, p))
                    for j, t in enumerate(core.TARGETS):
                        rows.append(dict(baseline=fam, law=law, p=p, input=inp, target=t, series="__pooled__", r=pooled[j]))
                        for s, nm in enumerate(names):
                            rows.append(dict(baseline=fam, law=law, p=p, input=inp, target=t, series=nm, r=per[s, j]))
            print(law, p, flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "content_baseline.csv", index=False)


if __name__ == "__main__":
    main()
