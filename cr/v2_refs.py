"""No-model references for the v2 batch (Modal GPU for the MLP readers).

For every (law, p, draw): difference probes on input values (ridge and MLP), their running sums, and a
learned position-only reference (spline of rank). Families:
  step_bidir (raw values), step_bidir_quant (Chronos-T5 quantized values), step_causal (raw, past-only),
  patch (patch summaries, v2 patch targets).
Output: cr/results_v2/references_v2.csv with columns
  law,p,draw,family,probe,quantity,series,r   quantity in {k,m_perp,dt,running_sum,position_m_perp,position_dt}

  modal run cr/v2_refs.py
"""
from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers>=4.48,<4.58", "chronos-forecasting==2.3.2",
                      "scikit-learn==1.5.2", "pandas==2.2.3", "numpy<2.1")
         .add_local_file(HERE / "core.py", "/root/core.py")
         .add_local_file(HERE / "v2core.py", "/root/v2core.py")
         .add_local_file(HERE / "content_baseline.py", "/root/content_baseline.py")
         .add_local_file(HERE / "series.npz", "/root/series.npz"))
app = modal.App("fmts-cr-v2-refs")


def ridge_pred(Hs, Ys, seed):
    import numpy as np
    import core
    from sklearn.model_selection import GroupKFold
    S = len(Hs)
    g = np.concatenate([np.full(len(h), s) for s, h in enumerate(Hs)])
    X, Y = np.concatenate(Hs).astype(float), np.concatenate(Ys).astype(float)
    pred = np.full_like(Y, np.nan)
    perm = np.random.default_rng(seed).permutation(S)[g]
    for tr, te in GroupKFold(n_splits=5).split(X, groups=perm):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        for j in range(Y.shape[1]):
            ok = ~np.isnan(Y[tr, j])
            W, xm, ym = core._ridge_fit(Xtr[ok], Y[tr][ok][:, [j]], 1.0)
            pred[te, j] = ((Xte - xm) @ W + ym)[:, 0]
    return pred, g


@app.function(image=image, gpu="A10G", timeout=3600, cpu=4.0, memory=16384)
def cell(law, p, dr):
    import numpy as np
    import pandas as pd
    import torch
    import core
    import v2core
    from content_baseline import patch_feats, step_feats
    from chronos import BaseChronosPipeline

    tok = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", torch_dtype=torch.float32).tokenizer
    centers = tok.centers.cpu().numpy() if torch.is_tensor(tok.centers) else np.asarray(tok.centers)
    n_special = tok.config.n_special_tokens

    def quant(vals):
        ctx = torch.tensor(np.asarray(vals, float), dtype=torch.float32)[None]
        ids, attn, scale = tok.context_input_transform(ctx)
        ids = ids[0][attn[0].bool()].numpy()[: len(vals)]
        return centers[np.clip(ids - n_special, 0, centers.size - 1)] * float(scale.reshape(-1)[0])

    d = np.load("/root/series.npz")
    names = [k[3:] for k in d.files if k.startswith("s__")]
    ys = [d["s__" + n] for n in names]
    keeps = [v2core.draw_keep_v2(law, y, p, np.random.default_rng(v2core.seed_v2(s, law, p, dr))) for s, y in enumerate(ys)]
    seed = v2core.seed_v2(0, law, p, dr)
    rows = []
    for fam in ("step_bidir", "step_bidir_quant", "step_causal", "patch"):
        patchy = fam == "patch"
        tg = [core.scaled(v2core.patch_targets_v2(k, y.size) if patchy else core.step_targets(k, y.size))
              for k, y in zip(keeps, ys)]
        Ys = [np.column_stack([t[x] for x in core.TARGETS]) for t in tg]
        Hs = []
        for k, y in zip(keeps, ys):
            v = quant(y[k]) if fam == "step_bidir_quant" else y[k]
            Hs.append(patch_feats(v) if patchy else step_feats(v, causal=(fam == "step_causal")))
        jd, jm = core.TARGETS.index("dt"), core.TARGETS.index("m_perp")
        preds = {"ridge": ridge_pred(Hs, Ys, seed)}
        _, _, mp = v2core.mlp_cross_series(Hs, Ys, seed=seed, device="cuda")
        preds["mlp"] = (mp, preds["ridge"][1])
        for probe, (pred, g) in preds.items():
            for s, nm in enumerate(names):
                sel = g == s
                if sel.sum() < 8:
                    continue
                for j, t in enumerate(core.TARGETS):
                    rows.append(dict(law=law, p=p, draw=dr, family=fam, probe=probe, quantity=t, series=nm,
                                     r=core.pearson(Ys[s][:, j], pred[sel, j])))
                pdt = pred[sel, jd].copy()
                pdt[0] = 0.0 if not patchy else pdt[0]
                t_hat = np.cumsum(np.nan_to_num(pdt))
                kk = np.arange(sel.sum(), dtype=float)
                rows.append(dict(law=law, p=p, draw=dr, family=fam, probe=probe, quantity="running_sum", series=nm,
                                 r=core.pearson(core.residualize(t_hat, kk), Ys[s][:, jm])))
        # learned position-only reference (same targets, spline of rank)
        Ps = [core.spline_features(np.arange(len(t["k"]))) for t in tg]
        per, _ = core.cross_series_probe(Ps, Ys, seed=seed)
        for s, nm in enumerate(names):
            rows.append(dict(law=law, p=p, draw=dr, family=fam, probe="ridge", quantity="position_m_perp", series=nm, r=per[s, jm]))
            rows.append(dict(law=law, p=p, draw=dr, family=fam, probe="ridge", quantity="position_dt", series=nm, r=per[s, jd]))
    return pd.DataFrame(rows).to_csv(index=False)


@app.local_entrypoint()
def main():
    import io
    import pandas as pd
    import v2core
    cells = [(law, p, dr) for law in v2core.LAWS_V2 for p in v2core.PS_V2 for dr in v2core.DRAWS]
    out = [pd.read_csv(io.StringIO(c)) for c in cell.starmap(cells)]
    (HERE / "results_v2").mkdir(exist_ok=True)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(HERE / "results_v2" / "references_v2.csv", index=False)
    print(df[df.series != "__pooled__"].groupby(["family", "probe", "law", "p", "quantity"]).r.median().unstack("quantity").round(3).to_string())
