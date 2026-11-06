"""Analysis of the v2 robustness batch. Reads cr/results_v2/, writes cr/results_v2/summary_v2.json and prints tables.

Layer choice: for each (model, law, p, probe, target), the layer with the best real-input median r on the
fixed 30% development series (same seed-2026 split as analyze.py); everything is reported on the other 71.
Per-series r is averaged over the 3 draws before taking medians / paired differences.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core  # noqa: E402

R2 = HERE / "results_v2"
ORDER = ["t5-tiny", "t5-mini", "t5-small", "t5-base", "t5-large", "timemoe-50m", "bolt-base", "chronos-2"]
PATCHY = {"bolt-base", "chronos-2"}
CAUSAL = {"timemoe-50m"}


def dev_split(series):
    rng = np.random.default_rng(2026)
    allS = sorted(series)
    return set(rng.choice(allS, size=int(round(0.3 * len(allS))), replace=False))


def ci(x, seed=7):
    m, lo, hi = core.bootstrap_median_ci(np.asarray(x, float), seed=seed)
    return [round(m, 3), round(lo, 3), round(hi, 3)]


def main():
    frames = [pd.read_csv(R2 / f"{m}.csv") for m in ORDER if (R2 / f"{m}.csv").exists()]
    d = pd.concat(frames, ignore_index=True)
    d = d[d.series != "__pooled__"]
    dev = dev_split(d.series.unique())
    # average over draws per series
    g = d.groupby(["model", "law", "p", "probe", "input", "layer", "target", "series"]).r.mean().reset_index()
    ref = pd.read_csv(R2 / "references_v2.csv")
    ref = ref[ref.series != "__pooled__"].groupby(["law", "p", "family", "probe", "quantity", "series"]).r.mean().reset_index()
    integ = pd.read_csv(R2 / "integrator_raw_v2.csv").groupby(["law", "p", "family", "series"]).r.mean().reset_index()

    S = {"models": sorted(g.model.unique(), key=ORDER.index), "n_eval": None, "cells": {}}
    for m in S["models"]:
        fam = "patch" if m in PATCHY else ("step_causal" if m in CAUSAL else "step_bidir")
        for law in ("iid", "block", "mnar"):
            for p in (0.5, 0.7):
                cell = {}
                for probe in ("ridge", "mlp"):
                    for target in ("k", "dt", "m_perp"):
                        x = g[(g.model == m) & (g.law == law) & (g.p == p) & (g.probe == probe) & (g.target == target)]
                        if x.empty:
                            continue
                        sel_in = "noise" if target == "k" else "real"
                        devx = x[x.series.isin(dev) & (x.input == sel_in)]
                        layer = int(devx.groupby("layer").r.median().idxmax())
                        ev = x[~x.series.isin(dev) & (x.layer == layer)].pivot_table(index="series", columns="input", values="r")
                        S["n_eval"] = int(ev.shape[0])
                        c = dict(layer=layer, real=round(float(ev["real"].median()), 3), noise=round(float(ev["noise"].median()), 3),
                                 delta=ci((ev["real"] - ev["noise"]).dropna()))
                        # references (same probe class, real input)
                        if target == "dt":
                            b = ref[(ref.law == law) & (ref.p == p) & (ref.family == fam) & (ref.probe == probe) & (ref.quantity == "dt")].set_index("series").r
                            diff = (ev["real"] - b).dropna()
                            c["reader"] = round(float(b[~b.index.isin(dev)].median()), 3)
                            c["minus_reader"] = ci(diff)
                            if fam == "step_bidir":
                                bq = ref[(ref.law == law) & (ref.p == p) & (ref.family == "step_bidir_quant") & (ref.probe == probe) & (ref.quantity == "dt")].set_index("series").r
                                c["reader_quant"] = round(float(bq[~bq.index.isin(dev)].median()), 3)
                                c["minus_reader_quant"] = ci((ev["real"] - bq).dropna())
                        if target == "m_perp":
                            b = ref[(ref.law == law) & (ref.p == p) & (ref.family == fam) & (ref.probe == probe) & (ref.quantity == "running_sum")].set_index("series").r
                            c["running_sum"] = round(float(b[~b.index.isin(dev)].median()), 3)
                            c["minus_running_sum"] = ci((ev["real"] - b).dropna())
                            if probe == "ridge" and fam != "patch":
                                bi = integ[(integ.law == law) & (integ.p == p) & (integ.family == fam)].set_index("series").r
                                c["running_sum_raw"] = round(float(bi[~bi.index.isin(dev)].median()), 3)
                                c["minus_running_sum_raw"] = ci((ev["real"] - bi).dropna())
                            pos = ref[(ref.law == law) & (ref.p == p) & (ref.family == fam) & (ref.quantity == "position_m_perp")].set_index("series").r
                            c["position_only"] = round(float(pos[~pos.index.isin(dev)].median()), 3)
                        cell[f"{probe}|{target}"] = c
                # draw-to-draw spread of the ridge dt / m_perp medians at the chosen layer
                for target in ("dt", "m_perp"):
                    key = f"ridge|{target}"
                    if key in cell:
                        x = d[(d.model == m) & (d.law == law) & (d.p == p) & (d.probe == "ridge") & (d.target == target) &
                              (d.layer == cell[key]["layer"]) & (d.input == "real") & ~d.series.isin(dev)]
                        cell[key]["per_draw_median"] = [round(float(v), 3) for v in x.groupby("draw").r.median().values]
                S["cells"][f"{m}|{law}|{p}"] = cell
    (R2 / "summary_v2.json").write_text(json.dumps(S, indent=1))

    # ---- compact printout
    def row(m, law, p, probe, t, keys):
        c = S["cells"].get(f"{m}|{law}|{p}", {}).get(f"{probe}|{t}")
        return "  ".join(f"{k}={c.get(k)}" for k in keys) if c else "-"
    for law in ("iid", "block", "mnar"):
        for p in (0.5, 0.7):
            print(f"\n===== {law} p={p}")
            for m in S["models"]:
                for probe in ("ridge", "mlp"):
                    print(f"{m:12s} {probe:5s} k: {row(m, law, p, probe, 'k', ['noise'])}")
                    print(f"{'':12s} {'':5s} dt: {row(m, law, p, probe, 'dt', ['real', 'noise', 'reader', 'minus_reader', 'minus_reader_quant'])}")
                    print(f"{'':12s} {'':5s} m_perp: {row(m, law, p, probe, 'm_perp', ['real', 'noise', 'running_sum', 'minus_running_sum', 'running_sum_raw', 'minus_running_sum_raw', 'position_only'])}")


if __name__ == "__main__":
    main()
