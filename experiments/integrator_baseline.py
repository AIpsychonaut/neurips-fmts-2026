"""Exploratory: does integrating a content-only gap reader recover accumulated time (m_perp)?
Cross-series ridge predicts dt from local features (as content_baseline step_bidir/step_causal),
then predicted gaps are cumulatively summed per series, residualized on k, and correlated with true m_perp.
python experiments/integrator_baseline.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core
from content_baseline import step_feats, seed_for
from sklearn.model_selection import GroupKFold

def main():
    d = np.load(HERE / "series.npz"); names = [k[3:] for k in d.files if k.startswith("s__")]; ys = [d["s__"+n] for n in names]
    rows = []
    for law in ("iid", "block", "beta"):
        for p in core.PS:
            keeps = [core.draw_keep(law, y.size, p, np.random.default_rng(seed_for(s, law, p))) for s, y in enumerate(ys)]
            raw = [core.step_targets(k, y.size) for k, y in zip(keeps, ys)]
            for causal in (False, True):
                for inp in ("real", "noise"):
                    Hs, Ys = [], []
                    for s, (k, y) in enumerate(zip(keeps, ys)):
                        v = y[k] if inp == "real" else np.random.default_rng(seed_for(s, law, p) + 7_777_777).normal(size=k.size)
                        Hs.append(step_feats(v, causal)); Ys.append(raw[s]["dt"])
                    g = np.concatenate([np.full(len(h), s) for s, h in enumerate(Hs)])
                    X = np.concatenate(Hs); Y = np.concatenate(Ys); pred = np.full_like(Y, np.nan)
                    perm = np.random.default_rng(seed_for(0, law, p)).permutation(len(Hs))[g]
                    for tr, te in GroupKFold(5).split(X, groups=perm):
                        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                        ok = ~np.isnan(Y[tr])
                        W, xm, ym = core._ridge_fit(((X[tr]-mu)/sd)[ok], Y[tr][ok][:, None], 1.0)
                        pred[te] = (((X[te]-mu)/sd - xm) @ W + ym)[:, 0]
                    for s in range(len(Hs)):
                        pdt = pred[g == s].copy(); pdt[0] = 0.0
                        t_hat = np.cumsum(pdt)
                        kk = raw[s]["k"]
                        r = core.pearson(core.residualize(t_hat, kk), raw[s]["m_perp"])
                        rows.append(dict(law=law, p=p, causal=causal, input=inp, series=names[s], r=r))
            print(law, p, flush=True)
    df = pd.DataFrame(rows); df.to_csv(HERE / "results" / "integrator_baseline.csv", index=False)
    print(df.groupby(["law", "causal", "input", "p"]).r.median().unstack("p").round(3).to_string())

if __name__ == "__main__":
    main()
