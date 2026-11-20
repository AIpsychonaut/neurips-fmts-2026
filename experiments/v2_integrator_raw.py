"""Running-sum baseline exactly as integrator_baseline.py (ridge on raw, unscaled dt), for the v2 laws/draws.
Output: experiments/results_v2/integrator_raw_v2.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402
import v2core  # noqa: E402
from content_baseline import step_feats  # noqa: E402

d = np.load(HERE / "series.npz")
names = [k[3:] for k in d.files if k.startswith("s__")]
ys = [d["s__" + n] for n in names]
rows = []
for law in v2core.LAWS_V2:
    for p in v2core.PS_V2:
        for dr in v2core.DRAWS:
            keeps = [v2core.draw_keep_v2(law, y, p, np.random.default_rng(v2core.seed_v2(s, law, p, dr))) for s, y in enumerate(ys)]
            raw = [core.step_targets(k, y.size) for k, y in zip(keeps, ys)]
            for causal in (False, True):
                Hs = [step_feats(y[k], causal) for k, y in zip(keeps, ys)]
                g = np.concatenate([np.full(len(h), s) for s, h in enumerate(Hs)])
                X = np.concatenate(Hs)
                Y = np.concatenate([r["dt"] for r in raw])
                pred = np.full_like(Y, np.nan)
                perm = np.random.default_rng(v2core.seed_v2(0, law, p, dr)).permutation(len(Hs))[g]
                for tr, te in GroupKFold(5).split(X, groups=perm):
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                    ok = ~np.isnan(Y[tr])
                    W, xm, ym = core._ridge_fit(((X[tr] - mu) / sd)[ok], Y[tr][ok][:, None], 1.0)
                    pred[te] = (((X[te] - mu) / sd - xm) @ W + ym)[:, 0]
                for s in range(len(Hs)):
                    pdt = pred[g == s].copy()
                    pdt[0] = 0.0
                    r = core.pearson(core.residualize(np.cumsum(pdt), raw[s]["k"]), raw[s]["m_perp"])
                    rows.append(dict(law=law, p=p, draw=dr, family="step_causal" if causal else "step_bidir", series=names[s], r=r))
            print(law, p, dr, flush=True)
df = pd.DataFrame(rows)
df.to_csv(HERE / "results_v2" / "integrator_raw_v2.csv", index=False)
print(df.groupby(["family", "law", "p", "draw"]).r.median().unstack("draw").round(3).to_string())
