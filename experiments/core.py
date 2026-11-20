"""Camera-ready core: keep laws, targets, patch mapping, MC oracle, probes.

Pure numpy / sklearn; no model code here.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import SplineTransformer

PATCH = 16
LAWS = ("beta", "iid", "block")
PS = (0.1, 0.3, 0.5, 0.7)
TARGETS = ("k", "m_perp", "dt")


# ----------------------------------------------------------------------------- keeps
def n_keep(T, p):
    return max(8, min(T, int(round((1.0 - p) * T))))


def keep_beta(T, p, rng, a=0.35, b=4.0):
    """Same law as the submitted paper (first-n unique Beta draws, endpoints pinned),
    vectorised so high keep fractions finish."""
    n = n_keep(T, p)
    need = n - 2
    chosen = np.empty(0, dtype=np.int64)
    pool = np.empty(0, dtype=np.int64)
    while True:
        draws = np.clip((rng.beta(a, b, size=200_000) * (T - 1)).astype(np.int64), 0, T - 1)
        pool = np.concatenate([pool, draws])
        vals, first = np.unique(pool, return_index=True)
        keep_mask = (vals != 0) & (vals != T - 1)
        vals, first = vals[keep_mask], first[keep_mask]
        if vals.size >= need:
            chosen = vals[np.argsort(first)][:need]
            break
        if pool.size > 50_000_000:
            raise RuntimeError("beta keep failed")
    return np.sort(np.concatenate([[0, T - 1], chosen]))


def keep_iid(T, p, rng):
    n = n_keep(T, p)
    interior = rng.choice(np.arange(1, T - 1), size=n - 2, replace=False)
    return np.sort(np.concatenate([[0, T - 1], interior]))


def keep_block(T, p, rng, mean_gap=10.0):
    """Alternating observed / missing runs with geometric lengths (sensor outages).
    Expected drop fraction p; endpoints pinned. Cardinality varies by draw."""
    mean_obs = mean_gap * (1 - p) / p
    mask = np.zeros(T, dtype=bool)
    t, observed = 0, True
    while t < T:
        L = rng.geometric(1.0 / max(mean_obs if observed else mean_gap, 1.0))
        if observed:
            mask[t : t + L] = True
        t += L
        observed = not observed
    mask[0] = mask[T - 1] = True
    idx = np.flatnonzero(mask)
    if idx.size < 8:  # pathological draw: fall back to iid with same p
        return keep_iid(T, p, rng)
    return idx


def draw_keep(law, T, p, rng):
    return {"beta": keep_beta, "iid": keep_iid, "block": keep_block}[law](T, p, rng)


# ----------------------------------------------------------------------------- targets
def zscore(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else x - np.nanmean(x)


def residualize(y, x):
    X = np.column_stack([np.ones_like(x, dtype=float), x])
    ok = ~np.isnan(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return y - X @ coef


def step_targets(keep, T):
    """Per-token targets for one-token-per-observation models. Returns raw (unscaled)."""
    n = keep.size
    k = np.arange(n, dtype=float)
    m = keep / float(T - 1)
    dt = np.full(n, np.nan)
    dt[1:] = np.diff(keep) / float(T - 1)
    return {"k": k, "m": m, "m_perp": residualize(m, k), "dt": dt}


def patch_index(n, patch=PATCH):
    """Compressed index -> patch index under left padding to a multiple of `patch`."""
    n_p = int(np.ceil(n / patch))
    pad = n_p * patch - n
    return (np.arange(n) + pad) // patch, n_p


def patch_targets(keep, T, patch=PATCH):
    st = step_targets(keep, T)
    pidx, n_p = patch_index(keep.size, patch)
    j = np.arange(n_p, dtype=float)
    m = np.array([st["m"][pidx == q].mean() for q in range(n_p)])
    dt = np.array([np.nanmean(st["dt"][pidx == q]) if np.any(~np.isnan(st["dt"][pidx == q])) else np.nan
                   for q in range(n_p)])
    return {"k": j, "m": m, "m_perp": residualize(m, j), "dt": dt}


def scaled(targets):
    """z-score each target within series (as in the submitted paper)."""
    return {t: zscore(targets[t]) for t in TARGETS}


# ----------------------------------------------------------------------------- oracle
def mc_oracle(law, T, p, n_draws=200, seed=0, patch=None, bins=50):
    """E[target | normalized rank] under the keep law, by Monte Carlo (no learning).
    Returns dict target -> callable(rank_norm array) -> oracle prediction (z-scale agnostic)."""
    rng = np.random.default_rng(seed)
    acc = {t: [[] for _ in range(bins)] for t in TARGETS}
    for _ in range(n_draws):
        keep = draw_keep(law, T, p, rng)
        tg = patch_targets(keep, T, patch) if patch else step_targets(keep, T)
        n = tg["k"].size
        r = tg["k"] / max(n - 1, 1)
        b = np.minimum((r * bins).astype(int), bins - 1)
        for t in TARGETS:
            v = zscore(tg[t])
            for bi, vi in zip(b, v):
                if not np.isnan(vi):
                    acc[t][bi].append(vi)
    table = {t: np.array([np.mean(x) if x else np.nan for x in acc[t]]) for t in TARGETS}
    # fill empty bins by interpolation
    for t in TARGETS:
        v = table[t]
        ok = ~np.isnan(v)
        if ok.sum() >= 2:
            table[t] = np.interp(np.arange(bins), np.flatnonzero(ok), v[ok])
    return table


def oracle_predict(table, k):
    bins = len(next(iter(table.values())))
    r = k / max(k.size - 1, 1)
    b = np.minimum((r * bins).astype(int), bins - 1)
    return {t: table[t][b] for t in table}


# ----------------------------------------------------------------------------- probes
def pearson(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _ridge_fit(X, Y, lam):
    """Ridge with intercept; X standardized by caller. Y may have several columns."""
    xm = X.mean(0)
    ym = Y.mean(0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ (Y - ym))
    return W, xm, ym


def cross_series_probe(H_list, Y_list, lam=1.0, n_splits=5, seed=0, min_tokens=8):
    """GroupKFold by series. H_list[s]: (n_s, d); Y_list[s]: (n_s, n_targets) z-scored, NaN allowed.
    Returns per-series r array (n_series, n_targets) and pooled held-out r (n_targets,)."""
    S = len(H_list)
    nT = Y_list[0].shape[1]
    groups = np.concatenate([np.full(len(h), s) for s, h in enumerate(H_list)])
    X = np.concatenate(H_list).astype(np.float64)
    Y = np.concatenate(Y_list).astype(np.float64)
    pred = np.full_like(Y, np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, S))
    # shuffle group assignment deterministically
    perm = np.random.default_rng(seed).permutation(S)
    g2 = perm[groups]
    for tr, te in gkf.split(X, groups=g2):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        for j in range(nT):
            ok = ~np.isnan(Y[tr, j])
            W, xm, ym = _ridge_fit(Xtr[ok], Y[tr][ok][:, [j]], lam)
            pred[te, j] = ((Xte - xm) @ W + ym)[:, 0]
    per = np.full((S, nT), np.nan)
    for s in range(S):
        sel = groups == s
        if sel.sum() >= min_tokens:
            for j in range(nT):
                per[s, j] = pearson(Y[sel, j], pred[sel, j])
    pooled = np.array([pearson(Y[:, j], pred[:, j]) for j in range(nT)])
    return per, pooled


def within_series_r(H, y, mode, lam=1.0, n_splits=5, seed=0):
    """Legacy protocol (raw H, sklearn-style ridge), shuffled or blocked folds."""
    from sklearn.linear_model import Ridge
    ok = ~np.isnan(y)
    H, y = H[ok], y[ok]
    if H.shape[0] < max(16, n_splits * 3):
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=(mode == "shuffled"), random_state=seed if mode == "shuffled" else None)
    rs = []
    for tr, te in kf.split(H):
        if te.size < 4:
            continue
        mdl = Ridge(alpha=lam).fit(H[tr], y[tr])
        rs.append(pearson(y[te], mdl.predict(H[te])))
    return float(np.nanmean(rs)) if rs else np.nan


def spline_features(k):
    r = (k / max(k.size - 1, 1)).reshape(-1, 1)
    return SplineTransformer(n_knots=8, degree=3, include_bias=True).fit_transform(r)


def bootstrap_median_ci(x, n=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))
