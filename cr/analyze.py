"""EDA for the camera-ready experiments. Reads cr/results/, writes cr/eda/ (figures, tables, summary.json).

python cr/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

RES = HERE / "results"
OUT = HERE / "eda"
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["t5-tiny", "t5-mini", "t5-small", "t5-base", "t5-large", "bolt-base", "chronos-2", "timemoe-50m"]
PARAMS = {"t5-tiny": 8e6, "t5-mini": 20e6, "t5-small": 46e6, "t5-base": 200e6, "t5-large": 710e6,
          "bolt-base": 205e6, "chronos-2": 120e6, "timemoe-50m": 50e6}
PATCHY = {"bolt-base", "chronos-2"}
COL = {"real": "#1f77b4", "noise": "#e69f00"}


def load():
    frames = [pd.read_csv(RES / f"{m}.csv") for m in ORDER if (RES / f"{m}.csv").exists()]
    d = pd.concat(frames, ignore_index=True)
    d["rel_layer"] = d.layer / d.n_layers
    ref = pd.read_csv(RES / "references.csv") if (RES / "references.csv").exists() else None
    rep = json.loads((RES / "replication.json").read_text()) if (RES / "replication.json").exists() else None
    return d, ref, rep


def per_series(d):
    return d[(d.cv == "cross") & (d.series != "__pooled__")]


def pooled(d):
    return d[(d.cv == "cross") & (d.series == "__pooled__")]


def summ(d, model, **kw):
    """median per-series r by layer (or pooled r for patch models where per-series is sparse)."""
    x = per_series(d)
    for k, v in kw.items():
        x = x[x[k] == v]
    return x[x.model == model].groupby("layer").r.median()


def main():
    d, ref, rep = load()
    models = [m for m in ORDER if m in d.model.unique()]
    S = {"replication": rep, "models": models}
    ps = per_series(d)
    S["n_series"] = int(ps.series.nunique())
    S["nan_frac_by_model"] = ps.groupby("model").r.apply(lambda s: round(float(s.isna().mean()), 3)).to_dict()

    # ---------------------------------------------------------------- P1 counting
    p1 = {}
    for m in models:
        rows = {}
        for law in core.LAWS:
            for inp in ("noise", "real"):
                prof = summ(d, m, law=law, p=0.5, input=inp, target="k")
                rows[f"{law}_{inp}"] = {"best_r": round(float(prof.max()), 3), "best_layer": int(prof.idxmax()),
                                        "layer0_r": round(float(prof.loc[0]), 3), "last_r": round(float(prof.iloc[-1]), 3)}
        p1[m] = rows
    S["P1_counting"] = p1

    # ---------------------------------------------------------------- references
    refs = {}
    if ref is not None:
        rr = ref[ref.series != "__pooled__"]
        for patch in (False, True):
            for law in core.LAWS:
                for p in core.PS:
                    for t in core.TARGETS:
                        for kind in ("mc_oracle", "spline_cross"):
                            x = rr[(rr.patch == patch) & (rr.law == law) & (rr.p == p) & (rr.target == t) & (rr.ref == kind)]
                            refs[f"{'patch' if patch else 'step'}|{law}|{p}|{t}|{kind}"] = round(float(x.r.median()), 3)
    S["references"] = refs

    # ---------------------------------------------------------------- P2 the trap
    p2 = {}
    for m in models:
        fam = "patch" if m in PATCHY else "step"
        entry = {}
        for law in core.LAWS:
            for p in core.PS:
                noise = summ(d, m, law=law, p=p, input="noise", target="m_perp")
                real = summ(d, m, law=law, p=p, input="real", target="m_perp")
                o = refs.get(f"{fam}|{law}|{p}|m_perp|mc_oracle", np.nan)
                e = {"cross_noise_best": round(float(noise.max()), 3), "cross_real_best": round(float(real.max()), 3),
                     "oracle": o, "noise_over_oracle": round(float(noise.max() / o), 3) if o and o > 0.2 else None}
                if m not in PATCHY:
                    w = d[(d.model == m) & (d.law == law) & (d.p == p) & (d.target == "m_perp")]
                    last = w.n_layers.max()
                    for mode in ("within_shuffled", "within_blocked"):
                        for inp in ("noise", "real"):
                            e[f"{mode}_{inp}_last"] = round(float(w[(w.cv == mode) & (w.input == inp) & (w.layer == last)].r.median()), 3)
                entry[f"{law}|{p}"] = e
        p2[m] = entry
    S["P2_trap"] = p2

    # ---------------------------------------------------------------- P3 keeping time (pre-registered)
    rng = np.random.default_rng(2026)
    all_series = sorted(ps.series.unique())
    dev = set(rng.choice(all_series, size=int(round(0.3 * len(all_series))), replace=False))
    S["P3_dev_series"] = sorted(dev)

    def p3(m, law="iid", p=0.5, target="dt"):
        x = ps[(ps.model == m) & (ps.law == law) & (ps.p == p) & (ps.target == target)]
        devx = x[x.series.isin(dev) & (x.input == "real")]
        layer = int(devx.groupby("layer").r.median().idxmax())
        ev = x[~x.series.isin(dev) & (x.layer == layer)].pivot_table(index="series", columns="input", values="r")
        delta = (ev["real"] - ev["noise"]).dropna()
        med, lo, hi = core.bootstrap_median_ci(delta.values, seed=7)
        # pooled (for patch models)
        po = pooled(d)
        po = po[(po.model == m) & (po.law == law) & (po.p == p) & (po.target == target) & (po.layer == layer)]
        return {"layer": layer, "n_eval": int(delta.size), "median_real": round(float(ev["real"].median()), 3),
                "median_noise": round(float(ev["noise"].median()), 3), "median_delta": round(med, 3),
                "ci95": [round(lo, 3), round(hi, 3)],
                "keeps_time": bool(med > 0.10 and lo > 0),
                "pooled_real": round(float(po[po.input == "real"].r.iloc[0]), 3) if len(po) else None,
                "pooled_noise": round(float(po[po.input == "noise"].r.iloc[0]), 3) if len(po) else None}

    S["P3_keeping_time"] = {m: p3(m) for m in models}
    # exploratory grid (not pre-registered)
    S["P3_grid_exploratory"] = {m: {f"{law}|{p}": p3(m, law, p) for law in ("iid", "block") for p in core.PS} for m in models}

    # ---------------------------------------------------------------- P4 scale
    S["P4_scale"] = {m: {"params": PARAMS[m], "counting_best_noise_k_iid": p1[m]["iid_noise"]["best_r"],
                         "keeping_time_delta": S["P3_keeping_time"][m]["median_delta"]}
                     for m in models if m.startswith("t5-")}

    # ---------------------------------------------------------------- content-clock exploratory: which series earn time?
    ser = np.load(HERE / "series.npz")
    props = {}
    for k in ser.files:
        if not k.startswith("s__"):
            continue
        y = ser[k]
        f = np.abs(np.fft.rfft(y - y.mean())) ** 2
        f = f[1:] / f[1:].sum()
        props[k[3:]] = {"lag1_ac": float(np.corrcoef(y[:-1], y[1:])[0, 1]),
                        "roughness": float(np.std(np.diff(y)) / np.std(y)),
                        "spec_entropy": float(-(f * np.log(f + 1e-12)).sum() / np.log(f.size)),
                        "T": int(y.size)}
    pr = pd.DataFrame(props).T
    cc = {}
    for m in models:
        info = S["P3_keeping_time"][m]
        x = ps[(ps.model == m) & (ps.law == "iid") & (ps.p == 0.5) & (ps.target == "dt") & (ps.layer == info["layer"])]
        dl = x.pivot_table(index="series", columns="input", values="r")
        dl = (dl["real"] - dl["noise"]).rename("delta").to_frame().join(pr, how="inner").dropna()
        cc[m] = {c: round(float(dl["delta"].corr(dl[c], method="spearman")), 3) for c in ("lag1_ac", "roughness", "spec_entropy", "T")}
        if m == "t5-base":
            dl.sort_values("delta").to_csv(OUT / "t5-base_iid_dt_delta_by_series.csv")
    S["content_clock_spearman"] = cc

    # ---------------------------------------------------------------- content-only baseline (exploratory)
    cb_path = RES / "content_baseline.csv"
    if cb_path.exists():
        cb = pd.read_csv(cb_path)
        cb = cb[cb.series != "__pooled__"]
        S["content_baseline_median"] = {
            f"{b}|{law}|{p}|{inp}|{t}": round(float(g.r.median()), 3)
            for (b, law, p, inp, t), g in cb.groupby(["baseline", "law", "p", "input", "target"])}
        vs = {}
        for m in models:
            fam = "patch" if m in PATCHY else ("step_causal" if m.startswith("timemoe") else "step_bidir")
            vs[m] = {}
            for law in ("iid", "block"):
                for p in core.PS:
                    info = S["P3_grid_exploratory"][m][f"{law}|{p}"]
                    x = ps[(ps.model == m) & (ps.law == law) & (ps.p == p) & (ps.target == "dt") &
                           (ps.layer == info["layer"]) & (ps.input == "real") & ~ps.series.isin(dev)].set_index("series").r
                    b = cb[(cb.baseline == fam) & (cb.law == law) & (cb.p == p) & (cb.target == "dt") &
                           (cb.input == "real")].set_index("series").r
                    diff = (x - b).dropna()
                    med, lo, hi = core.bootstrap_median_ci(diff.values, seed=11)
                    vs[m][f"{law}|{p}"] = {"baseline": fam, "model_real": round(float(x.median()), 3),
                                           "baseline_real": round(float(b.reindex(x.index).median()), 3),
                                           "median_model_minus_baseline": round(med, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        S["model_vs_content_baseline"] = vs

    # ---------------------------------------------------------------- accumulated time vs trivial integrator (exploratory)
    ib_path = RES / "integrator_baseline.csv"
    if ib_path.exists():
        ib = pd.read_csv(ib_path)
        acc = {}
        for m in models:
            causal = m.startswith("timemoe")
            acc[m] = {}
            if m in PATCHY:
                continue
            for law in ("iid", "block"):
                for p in core.PS:
                    x = ps[(ps.model == m) & (ps.law == law) & (ps.p == p) & (ps.target == "m_perp") & (ps.input == "real")]
                    layer = int(x[x.series.isin(dev)].groupby("layer").r.median().idxmax())
                    xm = x[(x.layer == layer) & ~x.series.isin(dev)].set_index("series").r
                    b = ib[(ib.law == law) & (ib.p == p) & (ib.causal == causal) & (ib.input == "real")].set_index("series").r
                    diff = (xm - b).dropna()
                    med, lo, hi = core.bootstrap_median_ci(diff.values, seed=13)
                    acc[m][f"{law}|{p}"] = {"layer": layer, "model_m_perp": round(float(xm.median()), 3),
                                            "integrator_m_perp": round(float(b.reindex(xm.index).median()), 3),
                                            "median_model_minus_integrator": round(med, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        S["accumulated_vs_integrator"] = acc

    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))

    # ================================================================ figures
    # F1 layer profiles: k, m_perp, dt ; iid p=.5 and beta p=.5 ; real vs noise
    for law in ("beta", "iid", "block"):
        fig, axes = plt.subplots(len(models), 3, figsize=(11, 2.0 * len(models)), sharex=False)
        for i, m in enumerate(models):
            for j, t in enumerate(core.TARGETS):
                ax = axes[i, j]
                for inp in ("real", "noise"):
                    prof = summ(d, m, law=law, p=0.5, input=inp, target=t)
                    ax.plot(prof.index, prof.values, "-o", ms=3, color=COL[inp], label=inp)
                fam = "patch" if m in PATCHY else "step"
                o = refs.get(f"{fam}|{law}|0.5|{t}|mc_oracle")
                if o is not None and t != "k":
                    ax.axhline(o, ls="--", color="green", lw=1, label="MC oracle (position only)")
                ax.axhline(0, color="grey", lw=0.5)
                ax.set_ylim(-0.4, 1.02)
                if i == 0:
                    ax.set_title(t)
                if j == 0:
                    ax.set_ylabel(m, fontsize=8)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f"Cross-series probe r by layer — {law} keep, p=0.5 (median over series)")
        fig.tight_layout()
        fig.savefig(FIG / f"layers_{law}.png", dpi=110)
        plt.close(fig)

    # F2 heatmap: best-layer noise r(m_perp) cross vs oracle, law x p, per model
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, law in enumerate(core.LAWS):
        M = np.array([[p2[m][f"{law}|{p}"]["cross_noise_best"] for p in core.PS] for m in models])
        im = axes[j].imshow(M, vmin=-0.2, vmax=1, cmap="viridis")
        axes[j].set_xticks(range(4), core.PS)
        axes[j].set_yticks(range(len(models)), models if j == 0 else [""] * len(models))
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                axes[j].text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7, color="w")
        o = [refs.get(f"step|{law}|{p}|m_perp|mc_oracle") for p in core.PS]
        axes[j].set_title(f"{law}: noise r(m⊥)\noracle(step) " + " ".join(f"{x:.2f}" for x in o), fontsize=8)
        axes[j].set_xlabel("drop rate p")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIG / "trap_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # F3 interpolation: within-shuffled vs within-blocked vs cross, noise, m_perp, step models
    step = [m for m in models if m not in PATCHY]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for j, law in enumerate(core.LAWS):
        for k_, (lab, key) in enumerate([("within shuffled", "within_shuffled_noise_last"),
                                         ("within blocked", "within_blocked_noise_last"),
                                         ("cross-series", "cross_noise_best")]):
            vals = [p2[m][f"{law}|0.5"][key] for m in step]
            axes[j].bar(np.arange(len(step)) + 0.27 * (k_ - 1), vals, width=0.27, label=lab)
        axes[j].axhline(refs.get(f"step|{law}|0.5|m_perp|mc_oracle", np.nan), color="green", ls="--", label="MC oracle")
        axes[j].set_xticks(range(len(step)), step, rotation=45, fontsize=7)
        axes[j].set_title(f"{law}, p=0.5: noise-input r(m⊥) by CV scheme", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "cv_schemes.png", dpi=110)
    plt.close(fig)

    # F4 keeping time: per-model delta with CI (iid dt p=.5)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, m in enumerate(models):
        e = S["P3_keeping_time"][m]
        ax.errorbar(i, e["median_delta"], yerr=[[e["median_delta"] - e["ci95"][0]], [e["ci95"][1] - e["median_delta"]]],
                    fmt="o", color="k", capsize=3)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(0.10, color="red", ls=":", lw=1, label="pre-registered 0.10 bar")
    ax.set_xticks(range(len(models)), models, rotation=40, fontsize=8)
    ax.set_ylabel("median Δr (real − noise)")
    ax.set_title("Keeping time: iid keep, Δt target, cross-series probe, p=0.5 (95% bootstrap CI)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "keeping_time.png", dpi=110)
    plt.close(fig)

    # F5 counting vs keeping-time plane
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for m in models:
        x = p1[m]["iid_noise"]["best_r"]
        y = S["P3_keeping_time"][m]["median_delta"]
        ax.scatter(x, y, s=40 + 60 * np.log10(PARAMS[m] / 8e6 + 1))
        ax.annotate(m, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("counting: best-layer r(H, k) on white noise")
    ax.set_ylabel("keeping time: median Δr on Δt (real − noise)")
    ax.set_title("Counting vs keeping time (iid keep, p=0.5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "counting_vs_keeping.png", dpi=110)
    plt.close(fig)

    # F6 keeping time vs drop rate (exploratory)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for j, law in enumerate(("iid", "block")):
        for m in models:
            axes[j].plot(core.PS, [S["P3_grid_exploratory"][m][f"{law}|{p}"]["median_delta"] for p in core.PS], "-o", ms=3, label=m)
        axes[j].axhline(0, color="grey", lw=0.6)
        axes[j].set_title(f"{law}: median Δr on Δt vs drop rate (exploratory)", fontsize=9)
        axes[j].set_xlabel("p")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "keeping_vs_p.png", dpi=110)
    plt.close(fig)

    # F7 content clock: per-series earned time vs smoothness
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.8), sharey=True)
    for i, m in enumerate(models):
        info = S["P3_keeping_time"][m]
        x = ps[(ps.model == m) & (ps.law == "iid") & (ps.p == 0.5) & (ps.target == "dt") & (ps.layer == info["layer"])]
        dl = x.pivot_table(index="series", columns="input", values="r")
        dl = (dl["real"] - dl["noise"]).rename("delta").to_frame().join(pr, how="inner").dropna()
        axes[i].scatter(dl.lag1_ac, dl.delta, s=8)
        axes[i].set_title(f"{m}\nρ={cc[m]['lag1_ac']:+.2f}", fontsize=8)
        axes[i].set_xlabel("lag-1 autocorr", fontsize=7)
        axes[i].axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("Δr on Δt (real − noise)")
    fig.suptitle("Earned time tracks how smooth the signal is (iid, p=0.5)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "content_clock.png", dpi=110)
    plt.close(fig)

    # F8 model vs content-only baseline (real input, dt)
    if "model_vs_content_baseline" in S:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        for j, law in enumerate(("iid", "block")):
            for i, m in enumerate(models):
                vals = [S["model_vs_content_baseline"][m][f"{law}|{p}"]["median_model_minus_baseline"] for p in core.PS]
                axes[j].plot(core.PS, vals, "-o", ms=3, label=m)
            axes[j].axhline(0, color="grey", lw=0.6)
            axes[j].set_title(f"{law}: model − content-only baseline, r on Δt (real input)", fontsize=9)
            axes[j].set_xlabel("p")
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(FIG / "model_vs_baseline.png", dpi=110)
        plt.close(fig)

    print(json.dumps({k: S[k] for k in ("n_series", "nan_frac_by_model", "P3_keeping_time", "P4_scale", "content_clock_spearman")}, indent=1))


if __name__ == "__main__":
    main()
