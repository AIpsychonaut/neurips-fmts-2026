"""Hidden-state extractors (copied verbatim from modal_run.py so v2 can import them without Modal app definitions)."""
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
