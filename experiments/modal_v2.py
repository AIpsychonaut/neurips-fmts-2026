"""Robustness batch (v2) on Modal. See V2_PLAN.md.

  modal run experiments/modal_v2.py --mode smoke                 # 6 series, iid p=0.7 draw 0, every model
  modal run experiments/modal_v2.py --mode full [--models a,b]   # iid/block/mnar x p 0.5/0.7 x 3 draws

Per model: ridge (all layers) + MLP (selected layers) held-out-series probes, real and noise inputs.
Hidden states of the MLP layers are cached to the volume (float16) for later CPU-only probes.
Results: experiments/results_v2/<model>.csv ; quantized Chronos values: experiments/results_v2/quantized.npz
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
base_pkgs = ["scikit-learn==1.5.2", "pandas==2.2.3", "numpy<2.1"]


def _add(img):
    return (img.add_local_file(HERE / "core.py", "/root/core.py")
            .add_local_file(HERE / "v2core.py", "/root/v2core.py")
            .add_local_file(HERE / "extractors.py", "/root/extractors.py")
            .add_local_file(HERE / "series.npz", "/root/series.npz"))


chronos_image = _add(modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.6.0", "transformers>=4.48,<4.58", "chronos-forecasting==2.3.2", *base_pkgs))
timemoe_image = _add(modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1", "transformers==4.40.1", "accelerate==0.34.2", "safetensors", *base_pkgs))
app = modal.App("fmts-cr-v2")
vol = modal.Volume.from_name("fmts-cr-vol", create_if_missing=True)
GPU_KW = dict(gpu="A10G", timeout=60 * 60 * 6, memory=32768, cpu=8.0, volumes={"/vol": vol})

# pre-registered gap layer per model (experiments/eda/summary.json, P3_keeping_time)
P3_LAYER = {"t5-tiny": 4, "t5-mini": 4, "t5-small": 5, "t5-base": 12, "t5-large": 23,
            "bolt-base": 9, "chronos-2": 2, "timemoe-50m": 7}


def run_v2(model_key, laws, ps, draws, max_series=None, do_mlp=True):
    import time
    import numpy as np
    import pandas as pd
    import torch
    import core
    import v2core
    from extractors import MODELS, PATCH_KINDS, make_extractor

    kind, mid = MODELS[model_key]
    patchy = kind in PATCH_KINDS
    extract = make_extractor(kind, mid)
    d = np.load("/root/series.npz")
    names = [k[3:] for k in d.files if k.startswith("s__")]
    ys = [d["s__" + n] for n in names]
    if max_series:
        names, ys = names[:max_series], ys[:max_series]

    quant = None
    if model_key == "t5-base":  # Chronos-T5 sizes share one tokenizer config
        from chronos import BaseChronosPipeline
        tok = BaseChronosPipeline.from_pretrained(mid, device_map="cpu", torch_dtype=torch.float32).tokenizer
        centers = tok.centers.cpu().numpy() if torch.is_tensor(tok.centers) else np.asarray(tok.centers)
        n_special = tok.config.n_special_tokens

        def quant(vals):
            ctx = torch.tensor(np.asarray(vals, float), dtype=torch.float32)[None]
            ids, attn, scale = tok.context_input_transform(ctx)
            ids = ids[0][attn[0].bool()].numpy()[: len(vals)]
            q = centers[np.clip(ids - n_special, 0, centers.size - 1)] * float(scale.reshape(-1)[0])
            return q
    qstore = {}

    rows, t0 = [], time.time()
    sdir = Path(f"/vol/v2states/{model_key}")
    sdir.mkdir(parents=True, exist_ok=True)
    for law in laws:
        for p in ps:
            for dr in draws:
                keeps = [v2core.draw_keep_v2(law, y, p, np.random.default_rng(v2core.seed_v2(s, law, p, dr)))
                         for s, y in enumerate(ys)]
                tg = [core.scaled(v2core.patch_targets_v2(k, y.size) if patchy else core.step_targets(k, y.size))
                      for k, y in zip(keeps, ys)]
                Y = [np.column_stack([t[x] for x in core.TARGETS]) for t in tg]
                if quant is not None:
                    for s, (k, y) in enumerate(zip(keeps, ys)):
                        qstore[f"{law}|{p}|{dr}|{names[s]}"] = quant(y[k]).astype(np.float32)
                for inp in ("real", "noise"):
                    Hs = []
                    for s, (k, y) in enumerate(zip(keeps, ys)):
                        nseed = v2core.seed_v2(s, law, p, dr) + 7_777_777
                        vals = y[k] if inp == "real" else np.random.default_rng(nseed).normal(size=k.size)
                        Hs.append(extract(vals))
                    L = len(Hs[0]) - 1
                    base = dict(model=model_key, law=law, p=p, draw=dr, input=inp, n_layers=L)
                    for layer in range(L + 1):
                        per, pooled = core.cross_series_probe([h[layer] for h in Hs], Y, seed=v2core.seed_v2(0, law, p, dr))
                        for j, t in enumerate(core.TARGETS):
                            rows.append(dict(base, layer=layer, probe="ridge", target=t, series="__pooled__", r=pooled[j]))
                            rows += [dict(base, layer=layer, probe="ridge", target=t, series=nm, r=per[s, j])
                                     for s, nm in enumerate(names)]
                    mls = v2core.mlp_layers(L, min(P3_LAYER[model_key], L))
                    for layer in mls:
                        np.savez_compressed(sdir / f"{law}_{p}_{dr}_{inp}_L{layer}.npz",
                                            **{nm: Hs[s][layer].astype(np.float16) for s, nm in enumerate(names)})
                        if do_mlp:
                            per, pooled, _ = v2core.mlp_cross_series([h[layer] for h in Hs], Y,
                                                                     seed=v2core.seed_v2(0, law, p, dr), device="cuda")
                            for j, t in enumerate(core.TARGETS):
                                rows.append(dict(base, layer=layer, probe="mlp", target=t, series="__pooled__", r=pooled[j]))
                                rows += [dict(base, layer=layer, probe="mlp", target=t, series=nm, r=per[s, j])
                                         for s, nm in enumerate(names)]
                    print(f"{model_key} {law} p={p} d={dr} {inp} L={L} t={time.time() - t0:.0f}s", flush=True)
                vol.commit()
    q_bytes = None
    if qstore:
        buf = io.BytesIO()
        np.savez_compressed(buf, **qstore)
        q_bytes = buf.getvalue()
    return model_key, pd.DataFrame(rows).to_csv(index=False), q_bytes


@app.function(image=chronos_image, **GPU_KW)
def run_chronos(model_key, laws, ps, draws, max_series=None):
    return run_v2(model_key, laws, ps, draws, max_series)


@app.function(image=timemoe_image, **GPU_KW)
def run_timemoe(model_key, laws, ps, draws, max_series=None):
    return run_v2(model_key, laws, ps, draws, max_series)


@app.local_entrypoint()
def main(mode: str = "smoke", models: str = ""):
    import v2core
    from extractors import MODELS
    out = HERE / ("results_v2" if mode == "full" else "results_v2_smoke")
    out.mkdir(exist_ok=True)
    if mode == "smoke":
        laws, ps, draws, ms = ["iid"], [0.7], [0], 6
    else:
        laws, ps, draws, ms = list(v2core.LAWS_V2), list(v2core.PS_V2), list(v2core.DRAWS), None
    wanted = set(models.split(",")) if models else set(MODELS)
    calls = [(run_timemoe if kind == "timemoe" else run_chronos).spawn(key, laws, ps, draws, ms)
             for key, (kind, _) in MODELS.items() if key in wanted]
    for c in calls:
        try:
            key, csv, q = c.get()
            (out / f"{key}.csv").write_text(csv)
            if q:
                (out / "quantized.npz").write_bytes(q)
            print("saved", key, len(csv.splitlines()), "rows", flush=True)
        except Exception as e:
            print("FAILED", repr(e)[:800], flush=True)
