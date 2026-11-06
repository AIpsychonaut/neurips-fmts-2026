"""Chronos-T5 encoder hidden states (optional extra)."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def encoder_states(
    pipeline,
    values: np.ndarray,
    layers: Iterable[int] = (4, 8, 12),
) -> dict[int, np.ndarray]:
    """Return encoder hidden states at the requested layers, attention-masked.

    Uses ``pipeline.model.model.encoder`` only (no decoder / horizon tokens).
    """
    import torch

    context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    token_ids, attn, _ = pipeline.tokenizer.context_input_transform(context)
    token_ids = token_ids.to(pipeline.model.device)
    attn = attn.to(pipeline.model.device)
    enc = pipeline.model.model.encoder
    with torch.no_grad():
        out = enc(
            input_ids=token_ids,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
    mask = attn[0].bool().cpu().numpy()
    hidden = out.hidden_states
    result: dict[int, np.ndarray] = {}
    for layer in layers:
        H = hidden[int(layer)][0].float().cpu().numpy()[mask]
        result[int(layer)] = H
    return result
