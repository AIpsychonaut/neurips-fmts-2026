"""Camera-ready experiment runner (Modal).

  modal run cr/modal_run.py --mode smoke        # 6 series, 1 condition, every model
  modal run cr/modal_run.py --mode replicate    # submitted-paper replication check
  modal run cr/modal_run.py --mode references   # model-free MC oracle + spline references
  modal run cr/modal_run.py --mode full         # all models, all conditions

Results land in cr/results/ as CSV.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code" / "sparse_time_probes"

base_pkgs = ["scikit-learn==1.5.2", "pandas==2.2.3", "numpy<2.1"]
chronos_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers>=4.48,<4.58", "chronos-forecasting==2.3.2", *base_pkgs)
    .add_local_file(HERE / "core.py", "/root/core.py")
    .add_local_file(HERE / "series.npz", "/root/series.npz")
    .add_local_dir(CODE, "/root/sparse_time_probes")
)
timemoe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "transformers==4.40.1", "accelerate==0.34.2", "safetensors", *base_pkgs)
    .add_local_file(HERE / "core.py", "/root/core.py")
    .add_local_file(HERE / "series.npz", "/root/series.npz")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*base_pkgs)
    .add_local_file(HERE / "core.py", "/root/core.py")
    .add_local_file(HERE / "series.npz", "/root/series.npz")
)
app = modal.App("fmts-cr")

MODELS = {
    "t5-tiny": ("t5", "amazon/chronos-t5-tiny"),
    "t5-mini": ("t5", "amazon/chronos-t5-mini"),
    "t5-small": ("t5", "amazon/chronos-t5-small"),
    "t5-base": ("t5", "amazon/chronos-t5-base"),
    "t5-large": ("t5", "amazon/chronos-t5-large"),
    "bolt-base": ("bolt", "amazon/chronos-bolt-base"),
    "chronos-2": ("c2", "amazon/chronos-2"),
    "timemoe-50m": ("timemoe", "Maple728/TimeMoE-50M"),
}
PATCH_KINDS = {"bolt", "c2"}


# ============================================================================ adapters
def make_extractor(kind, model_id):
    import numpy as np
    import torch

    dev = "cuda"
    if kind == "timemoe":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                     torch_dtype=torch.float32).to(dev).eval()

        def extract(values):
            x = np.asarray(values, float)
            x = (x - x.mean()) / (x.std() + 1e-8)
            t = torch.tensor(x, dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                o = model.model(input_ids=t, output_hidden_states=True, return_dict=True)
            return [h[0].float().cpu().numpy() for h in o.hidden_states]
        return extract

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=dev, torch_dtype=torch.float32)

    if kind == "t5":
        enc = pipe.model.model.encoder

        def extract(values):
            ctx = torch.tensor(np.asarray(values, float), dtype=torch.float32)[None]
            ids, attn, _ = pipe.tokenizer.context_input_transform(ctx)
            with torch.no_grad():
                o = enc(input_ids=ids.to(dev), attention_mask=attn.to(dev), output_hidden_states=True, return_dict=True)
            msk = attn[0].bool().numpy()
            n = len(values)
            out = [h[0].float().cpu().numpy()[msk][:n] for h in o.hidden_states]
            assert out[0].shape[0] == n, (out[0].shape, n)
            return out
        return extract

    m = pipe.model

    def _patch_inputs(ctx):
        mask = torch.ones_like(ctx)
        c, _ = m.instance_norm(ctx)
        pc = m.patch(c)
        pm = torch.nan_to_num(m.patch(mask), nan=0.0)
        return c, pc, pm

    if kind == "bolt":
        def extract(values):
            n = len(values)
            ctx = torch.tensor(np.asarray(values, float), dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                c, pc, pm = _patch_inputs(ctx)
                n_p = pc.shape[1]
                assert n_p == int(np.ceil(n / 16)), (n_p, n)
                if n % 16:
                    assert pm[0, 0].sum().item() == n % 16, "expected left padding"
                pc = torch.where(pm > 0, pc, 0.0)
                emb = m.input_patch_embedding(torch.cat([pc, pm], dim=-1))
                am = (pm.sum(-1) > 0).to(emb.dtype)
                reg = m.shared(torch.full((1, 1), m.config.reg_token_id, device=dev))
                emb = torch.cat([emb, reg], dim=-2)
                am = torch.cat([am, torch.ones(1, 1, device=dev)], dim=-1)
                o = m.encoder(attention_mask=am, inputs_embeds=emb, output_hidden_states=True, return_dict=True)
            return [h[0, :n_p].float().cpu().numpy() for h in o.hidden_states]
        return extract

    if kind == "c2":
        import torch.nn as nn
        blocks = None
        for _, mod in m.encoder.named_children():
            if isinstance(mod, nn.ModuleList):
                blocks = mod
                break
        assert blocks is not None, "no ModuleList in Chronos-2 encoder"
        store = {}

        def hook(i):
            def f(_mod, _inp, out):
                if i == 0 and 0 in store:  # input embedding fires again for the future patch
                    return
                t = out
                if isinstance(t, (tuple, list)):
                    t = t[0]
                if not torch.is_tensor(t):
                    cand = None
                    for attr in ("hidden_states", "last_hidden_state"):
                        v = getattr(t, attr, None)
                        if torch.is_tensor(v):
                            cand = v
                            break
                    if cand is None and hasattr(t, "__getitem__"):
                        cand = t[0]
                    t = cand
                assert torch.is_tensor(t) and t.dim() == 3, f"block output {type(out)}"
                store[i] = t.detach()
            return f

        for i, b in enumerate(blocks):
            b.register_forward_hook(hook(i + 1))
        m.input_patch_embedding.register_forward_hook(hook(0))

        def extract(values):
            n = len(values)
            ctx = torch.tensor(np.asarray(values, float), dtype=torch.float32, device=dev)[None]
            store.clear()
            with torch.no_grad():
                _, pc, pm = _patch_inputs(ctx)
                n_p = pc.shape[1]
                assert n_p == int(np.ceil(n / 16)), (n_p, n)
                if n % 16:
                    assert pm[0, 0].sum().item() == n % 16, "expected left padding"
                m.encode(context=ctx, num_output_patches=1)
            # input embedding hook fires twice (context, then future); keep context version via first n_p rows
            return [store[i][0, :n_p].float().cpu().numpy() for i in sorted(store)]
        return extract

    raise ValueError(kind)


# ============================================================================ experiment
def load_series(legacy=False):
    import numpy as np
    d = np.load("/root/series.npz")
    pref = "legacy__" if legacy else "s__"
    names = [k[len(pref):] for k in d.files if k.startswith(pref)]
    return names, [d[pref + n] for n in names]


def seed_for(s, law, p):
    from core import LAWS
    return 42 + 1000 * s + 100 * LAWS.index(law) + int(round(p * 10))


def run_conditions(model_key, laws, ps, max_series=None, inputs=("real", "noise")):
    import numpy as np
    import pandas as pd
    import time
    import core

    kind, mid = MODELS[model_key]
    patchy = kind in PATCH_KINDS
    extract = make_extractor(kind, mid)
    names, ys = load_series()
    if max_series:
        names, ys = names[:max_series], ys[:max_series]
    rows = []
    t0 = time.time()
    for law in laws:
        for p in ps:
            keeps = [core.draw_keep(law, y.size, p, np.random.default_rng(seed_for(s, law, p)))
                     for s, y in enumerate(ys)]
            tg = [core.scaled(core.patch_targets(k, y.size) if patchy else core.step_targets(k, y.size))
                  for k, y in zip(keeps, ys)]
            Y = [np.column_stack([t[x] for x in core.TARGETS]) for t in tg]
            for inp in inputs:
                Hs = []
                for s, (k, y) in enumerate(zip(keeps, ys)):
                    vals = y[k] if inp == "real" else np.random.default_rng(seed_for(s, law, p) + 7_777_777).normal(size=k.size)
                    Hs.append(extract(vals))
                L = len(Hs[0])
                for layer in range(L):
                    per, pooled = core.cross_series_probe([h[layer] for h in Hs], Y, seed=seed_for(0, law, p))
                    for j, t in enumerate(core.TARGETS):
                        rows.append(dict(model=model_key, law=law, p=p, input=inp, layer=layer, n_layers=L - 1,
                                         target=t, cv="cross", series="__pooled__", r=pooled[j]))
                        for s, nm in enumerate(names):
                            rows.append(dict(model=model_key, law=law, p=p, input=inp, layer=layer, n_layers=L - 1,
                                             target=t, cv="cross", series=nm, r=per[s, j]))
                if not patchy:
                    for layer in sorted({(L - 1) // 2, (2 * (L - 1)) // 3, L - 1}):
                        for mode in ("shuffled", "blocked"):
                            for j, t in enumerate(core.TARGETS):
                                if t == "k":
                                    continue
                                for s, nm in enumerate(names):
                                    r = core.within_series_r(Hs[s][layer], Y[s][:, j], mode, seed=seed_for(s, law, p))
                                    rows.append(dict(model=model_key, law=law, p=p, input=inp, layer=layer,
                                                     n_layers=L - 1, target=t, cv="within_" + mode, series=nm, r=r))
                print(f"{model_key} {law} p={p} {inp} L={L - 1} n_series={len(ys)} t={time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows).to_csv(index=False)


vol = modal.Volume.from_name("fmts-cr-vol", create_if_missing=True)
GPU_KW = dict(gpu="A10G", timeout=60 * 60 * 4, memory=32768, cpu=8.0, volumes={"/vol": vol})


@app.function(image=chronos_image, **GPU_KW)
def run_chronos(model_key, laws, ps, max_series=None):
    csv = run_conditions(model_key, laws, ps, max_series)
    Path(f"/vol/{model_key}.csv").write_text(csv)
    vol.commit()
    return model_key, csv


@app.function(image=timemoe_image, **GPU_KW)
def run_timemoe(model_key, laws, ps, max_series=None):
    csv = run_conditions(model_key, laws, ps, max_series)
    Path(f"/vol/{model_key}.csv").write_text(csv)
    vol.commit()
    return model_key, csv


@app.function(image=chronos_image, gpu="A10G", timeout=3600)
def replicate():
    """Exact submitted protocol: legacy keep sampler + legacy ridge, 8 series capped at 400, bf16 like the original."""
    import numpy as np
    import torch
    from chronos import BaseChronosPipeline
    from sparse_time_probes.keep import keep_beta
    from sparse_time_probes.probe import ridge_cv_r
    import core

    names, ys = load_series(legacy=True)
    out = {}
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-base", device_map="cuda", torch_dtype=dtype)
        enc = pipe.model.model.encoder
        res = {"real": [], "noise": []}
        for s, y in enumerate(ys):
            for noise in (False, True):
                rng = np.random.default_rng(42 + s)
                keep = keep_beta(y.size, 0.5, rng)
                x = y[keep].copy()
                if noise:
                    x = rng.normal(size=x.shape)
                ctx = torch.tensor(x, dtype=torch.float32)[None]
                ids, attn, _ = pipe.tokenizer.context_input_transform(ctx)
                with torch.no_grad():
                    o = enc(input_ids=ids.cuda(), attention_mask=attn.cuda(), output_hidden_states=True, return_dict=True)
                H = o.hidden_states[12][0].float().cpu().numpy()[attn[0].bool().numpy()][: keep.size]
                i = np.arange(keep.size, dtype=float)
                m = keep / float(y.size - 1)
                mp = core.zscore(core.residualize(m, i))
                res["noise" if noise else "real"].append(ridge_cv_r(H, mp, seed=42 + s))
        out[dtype_name] = {"names": names, **res,
                           "median_real": float(np.median(res["real"])), "median_noise": float(np.median(res["noise"]))}
    return out


@app.function(image=cpu_image, cpu=4.0, timeout=3600)
def reference_cell(law, p, T, patch):
    import core
    tab = core.mc_oracle(law, T, p, n_draws=200, seed=12345 + T, patch=16 if patch else None)
    return (law, p, T, patch, {k: v.tolist() for k, v in tab.items()})


@app.function(image=cpu_image, cpu=8.0, memory=16384, timeout=3600)
def references(tables_json):
    """Per-series MC-oracle r and cross-series spline(k) r, for step and patch targets."""
    import numpy as np
    import pandas as pd
    import core
    tables = {tuple(json.loads(k)): {t: np.array(v) for t, v in d.items()} for k, d in json.loads(tables_json).items()}
    names, ys = load_series()
    rows = []
    for patch in (False, True):
        for law in core.LAWS:
            for p in core.PS:
                keeps = [core.draw_keep(law, y.size, p, np.random.default_rng(seed_for(s, law, p))) for s, y in enumerate(ys)]
                tg = [core.scaled(core.patch_targets(k, y.size) if patch else core.step_targets(k, y.size))
                      for k, y in zip(keeps, ys)]
                Y = [np.column_stack([t[x] for x in core.TARGETS]) for t in tg]
                Hs = [core.spline_features(np.arange(len(t["k"]))) for t in tg]
                per, pooled = core.cross_series_probe(Hs, Y, seed=seed_for(0, law, p))
                for s, nm in enumerate(names):
                    tab = tables[(law, p, int(ys[s].size), patch)]
                    pred = core.oracle_predict(tab, np.arange(len(tg[s]["k"])))
                    for j, t in enumerate(core.TARGETS):
                        rows.append(dict(ref="spline_cross", patch=patch, law=law, p=p, target=t, series=nm, r=per[s, j]))
                        rows.append(dict(ref="mc_oracle", patch=patch, law=law, p=p, target=t, series=nm,
                                         r=core.pearson(tg[s][t], pred[t])))
                for j, t in enumerate(core.TARGETS):
                    rows.append(dict(ref="spline_cross", patch=patch, law=law, p=p, target=t, series="__pooled__", r=pooled[j]))
    return pd.DataFrame(rows).to_csv(index=False)


@app.local_entrypoint()
def main(mode: str = "smoke", models: str = ""):
    import core  # local import for constants
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    if mode == "replicate":
        res = replicate.remote()
        (out / "replication.json").write_text(json.dumps(res, indent=1))
        for k, v in res.items():
            print(k, "median real", round(v["median_real"], 3), "noise", round(v["median_noise"], 3), "(paper 0.821 / 0.759)")
        return
    if mode == "references":
        import numpy as np
        d = np.load(HERE / "series.npz")
        Ts = sorted({int(d[k].size) for k in d.files if k.startswith("s__")})
        cells = [(law, p, T, patch) for patch in (False, True) for law in core.LAWS for p in core.PS for T in Ts]
        tables = {}
        for law, p, T, patch, tab in reference_cell.starmap(cells):
            tables[json.dumps([law, p, T, patch])] = tab
        (out / "oracle_tables.json").write_text(json.dumps(tables))
        (out / "references.csv").write_text(references.remote(json.dumps(tables)))
        print("references written")
        return
    if mode == "smoke":
        laws, ps, ms = ["iid"], [0.5], 6
    else:
        laws, ps, ms = list(core.LAWS), list(core.PS), None
    calls = []
    wanted = set(models.split(",")) if models else set(MODELS)
    for key, (kind, _) in MODELS.items():
        if key not in wanted:
            continue
        fn = run_timemoe if kind == "timemoe" else run_chronos
        calls.append(fn.spawn(key, laws, ps, ms))
    for c in calls:
        try:
            key, csv = c.get()
            name = f"smoke_{key}.csv" if mode == "smoke" else f"{key}.csv"
            (out / name).write_text(csv)
            print("saved", name, len(csv.splitlines()), "rows")
        except Exception as e:
            print("FAILED", repr(e)[:500])
