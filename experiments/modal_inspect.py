"""Inspect model internals on Modal before writing adapters.

modal run experiments/modal_inspect.py
"""
import modal

chronos_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.6.0", "transformers>=4.48,<4.58", "chronos-forecasting>=2.0", "numpy<2.1"
)
timemoe_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1", "transformers==4.40.1", "accelerate", "numpy<2.1", "safetensors"
)
app = modal.App("fmts-cr-inspect")


@app.function(image=chronos_image, gpu="T4", timeout=1200)
def inspect_chronos():
    import inspect, importlib.metadata as md
    import torch
    import chronos
    out = {"chronos": md.version("chronos-forecasting"), "transformers": md.version("transformers")}
    from chronos import BaseChronosPipeline
    for mid in ["amazon/chronos-t5-tiny", "amazon/chronos-bolt-base", "amazon/chronos-2"]:
        try:
            p = BaseChronosPipeline.from_pretrained(mid, device_map="cuda", torch_dtype=torch.float32)
            m = p.model
            info = {"pipeline": type(p).__name__, "model": type(m).__name__,
                    "children": [n for n, _ in m.named_children()]}
            cfg = getattr(m, "config", None)
            if cfg is not None:
                info["config_keys"] = {k: v for k, v in cfg.to_dict().items() if not isinstance(v, (list, dict)) or k in ("chronos_config",)}
            for meth in ["encode", "forward", "_prepare_patched_context", "_prepare_patched_future"]:
                if hasattr(m, meth):
                    try:
                        info["src_" + meth] = inspect.getsource(getattr(m, meth))[:6000]
                    except Exception as e:
                        info["src_" + meth] = repr(e)
            out[mid] = info
        except Exception as e:
            out[mid] = {"error": repr(e)}
    # Chronos-2 module source tree
    import os
    root = os.path.dirname(chronos.__file__)
    out["files"] = sorted(os.listdir(root))
    try:
        import chronos.chronos2 as c2
        out["c2_files"] = sorted(os.listdir(os.path.dirname(c2.__file__)))
    except Exception as e:
        out["c2_files"] = repr(e)
    return out


@app.function(image=timemoe_image, gpu="T4", timeout=1200)
def inspect_timemoe():
    import inspect, torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("Maple728/TimeMoE-50M", trust_remote_code=True, torch_dtype=torch.float32).cuda()
    out = {"model": type(m).__name__, "children": [n for n, _ in m.named_children()],
           "config": {k: v for k, v in m.config.to_dict().items() if not isinstance(v, (list, dict))}}
    try:
        out["src_model_forward"] = inspect.getsource(type(m.model).forward)[:5000]
    except Exception as e:
        out["src_model_forward"] = repr(e)
    x = torch.randn(1, 64).cuda()
    with torch.no_grad():
        o = m.model(input_ids=x, output_hidden_states=True, return_dict=True)
    out["n_hidden"] = len(o.hidden_states)
    out["hidden_shape"] = list(o.hidden_states[-1].shape)
    return out


@app.local_entrypoint()
def main():
    import json
    res = {"chronos": inspect_chronos.remote(), "timemoe": inspect_timemoe.remote()}
    from pathlib import Path
    p = Path(__file__).parent / "inspect_out.json"
    p.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("wrote", p)
