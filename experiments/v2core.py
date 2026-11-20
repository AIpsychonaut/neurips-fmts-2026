"""Robustness batch (v2) helpers: value-dependent (MNAR) drops, patch-advance targets, multi-draw seeds,
and a grouped-CV MLP probe. Extends core.py; see V2_PLAN.md.
"""
from __future__ import annotations

import numpy as np

import core

LAWS_V2 = ("iid", "block", "mnar")
PS_V2 = (0.5, 0.7)
DRAWS = (0, 1, 2)
LAW_IDX = {"beta": 0, "iid": 1, "block": 2, "mnar": 3}
MNAR_STRENGTH = 1.5  # drop weight exp(1.5 * z(y_t)): high values are more likely to be missing


def seed_v2(s, law, p, d):
    """Draw 0 reproduces the original seeds for beta/iid/block."""
    return 42 + 1000 * s + 100 * LAW_IDX[law] + int(round(p * 10)) + 10_000_000 * d


def keep_mnar(y, p, rng, strength=MNAR_STRENGTH):
    """Fixed-cardinality drop, weighted towards high values (saturating sensor); endpoints kept."""
    T = y.size
    n = core.n_keep(T, p)
    z = core.zscore(y)
    w = np.exp(strength * z[1:-1])
    drop = rng.choice(np.arange(1, T - 1), size=(T - 2) - (n - 2), replace=False, p=w / w.sum())
    mask = np.ones(T, dtype=bool)
    mask[drop] = False
    return np.flatnonzero(mask)


def draw_keep_v2(law, y, p, rng):
    if law == "mnar":
        return keep_mnar(y, p, rng)
    return core.draw_keep(law, y.size, p, rng)


def patch_targets_v2(keep, T, patch=core.PATCH):
    """Patch-level targets that keep timing variance:
    m_end = time of the last observation in the patch; dt = time advanced since the previous patch's end
    (for the first patch: since t=0); m_perp = m_end with its linear trend in patch index removed."""
    m = keep / float(T - 1)
    pidx, n_p = core.patch_index(keep.size, patch)
    m_end = np.array([m[pidx == q].max() for q in range(n_p)])
    dt = np.diff(np.concatenate([[0.0], m_end]))
    j = np.arange(n_p, dtype=float)
    return {"k": j, "m": m_end, "m_perp": core.residualize(m_end, j), "dt": dt}


def mlp_layers(L, p3_layer):
    """Hidden-state layers probed with the MLP: 1/3, 1/2, 2/3 depth, last, and the pre-registered gap layer."""
    return sorted({max(1, round(L / 3)), max(1, round(L / 2)), max(1, round(2 * L / 3)), L, p3_layer})


# ----------------------------------------------------------------------------- MLP probe
def mlp_cross_series(H_list, Y_list, n_splits=5, seed=0, min_tokens=8, device="cpu",
                     hidden=256, epochs=400, patience=40, lr=3e-3, wd=1e-4, dropout=0.1):
    """Same GroupKFold split as core.cross_series_probe; 2-hidden-layer MLP, masked MSE over targets,
    early stopping on 15% of training series. Returns per-series r (S, nT) and pooled r."""
    import torch
    from sklearn.model_selection import GroupKFold

    torch.manual_seed(seed)
    S = len(H_list)
    nT = Y_list[0].shape[1]
    groups = np.concatenate([np.full(len(h), s) for s, h in enumerate(H_list)])
    X = np.concatenate(H_list).astype(np.float32)
    Y = np.concatenate(Y_list).astype(np.float32)
    pred = np.full_like(Y, np.nan)
    perm = np.random.default_rng(seed).permutation(S)
    g2 = perm[groups]
    rng = np.random.default_rng(seed + 1)
    for tr, te in GroupKFold(n_splits=min(n_splits, S)).split(X, groups=g2):
        tr_groups = np.unique(groups[tr])
        val_g = rng.choice(tr_groups, size=max(2, int(0.15 * tr_groups.size)), replace=False)
        is_val = np.isin(groups[tr], val_g)
        fit, val = tr[~is_val], tr[is_val]
        mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
        t = lambda a: torch.tensor((X[a] - mu) / sd, device=device)
        Xf, Xv, Xt = t(fit), t(val), t(te)
        Yf = torch.tensor(Y[fit], device=device)
        Yv = torch.tensor(Y[val], device=device)
        Mf, Mv = ~torch.isnan(Yf), ~torch.isnan(Yv)
        Yf, Yv = torch.nan_to_num(Yf), torch.nan_to_num(Yv)
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, nT)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            loss = (((net(Xf) - Yf) ** 2) * Mf).sum() / Mf.sum()
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                vl = float((((net(Xv) - Yv) ** 2) * Mv).sum() / Mv.sum())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred[te] = net(Xt).cpu().numpy()
    per = np.full((S, nT), np.nan)
    for s in range(S):
        sel = groups == s
        if sel.sum() >= min_tokens:
            for j in range(nT):
                per[s, j] = core.pearson(Y[sel, j], pred[sel, j])
    pooled = np.array([core.pearson(Y[:, j], pred[:, j]) for j in range(nT)])
    return per, pooled, pred
