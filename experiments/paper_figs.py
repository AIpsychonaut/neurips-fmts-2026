"""Camera-ready figures + table numbers. Reads experiments/results and experiments/eda/summary.json, writes paper/figures/.

python experiments/paper_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((HERE / "eda" / "summary.json").read_text())

STEP = ["t5-tiny", "t5-mini", "t5-small", "t5-base", "t5-large", "timemoe-50m"]
ALL = ["t5-tiny", "t5-mini", "t5-small", "t5-base", "t5-large", "timemoe-50m", "bolt-base", "chronos-2"]
LABEL = {"t5-tiny": "Chronos-T5 tiny", "t5-mini": "Chronos-T5 mini", "t5-small": "Chronos-T5 small",
         "t5-base": "Chronos-T5 base", "t5-large": "Chronos-T5 large", "timemoe-50m": "Time-MoE 50M",
         "bolt-base": "Chronos-Bolt base", "chronos-2": "Chronos-2"}

# validated categorical slots 1-2 (dataviz reference palette, all-pairs safe) + neutrals
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def load():
    return pd.concat([pd.read_csv(HERE / "results" / f"{m}.csv") for m in ALL], ignore_index=True)


def med(d, **kw):
    x = d[d.series != "__pooled__"]
    for k, v in kw.items():
        x = x[x[k] == v]
    return float(x.r.median())


# ---------------------------------------------------------------- Figure 1: the trap
def fig_trap(d):
    laws = [("beta", "Beta keep"), ("iid", "Random dropout"), ("block", "Block outages")]
    schemes = [("within_shuffled", "Shuffled\nfolds"), ("within_blocked", "Contiguous\nfolds"), ("cross", "Held-out\nseries")]
    p = 0.5
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.55), sharey=True)
    rng = np.random.default_rng(0)
    vals = {}
    off = {"real": -0.17, "noise": 0.17}
    for ax, (law, title) in zip(axes, laws):
        oracle = S["references"][f"step|{law}|{p}|m_perp|mc_oracle"]
        ax.axhline(oracle, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for j, (cv, _) in enumerate(schemes):
            ax.axvspan(j - 0.42, j + 0.42, color="#f6f5f2", zorder=0, lw=0)
            for inp in ("real", "noise"):
                rs = []
                for m in STEP:
                    x = d[(d.model == m) & (d.law == law) & (d.p == p) & (d.input == inp) & (d.target == "m_perp") & (d.cv == cv)]
                    r = med(x, layer=x.n_layers.iloc[0])
                    rs.append(r)
                    if inp == "noise":
                        vals[(law, cv, m)] = r
                    tm = m == "timemoe-50m"
                    col = BLUE if inp == "noise" else REF
                    ax.scatter(j + off[inp] + rng.uniform(-0.06, 0.06), r, s=13 if tm else 11, zorder=3,
                               marker="^" if tm else "o", facecolor=col if inp == "noise" else "white",
                               edgecolor=col, linewidth=0.8)
                ax.plot([j + off[inp] - 0.11, j + off[inp] + 0.11], [np.median(rs)] * 2,
                        color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
        ax.set_xticks(range(3), [s for _, s in schemes])
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, color=INK, pad=3)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    # annotate the two failure modes where they occur
    axes[0].annotate("noise matches\nthe oracle", xy=(2.05, 0.93), xytext=(1.0, 0.62), fontsize=6, color=INK2,
                     ha="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    axes[1].annotate("interpolation:\nhigh $r$ from noise", xy=(0.2, 0.68), xytext=(1.55, 0.5), fontsize=6, color=INK2,
                     ha="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    axes[0].set_ylim(-0.15, 1.08)
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_ylabel("$r$ for elapsed time $m^{\\perp}$")
    hs = [axes[0].scatter([], [], s=11, color=BLUE, label="white-noise input"),
          axes[0].scatter([], [], s=11, facecolor="white", edgecolor=REF, lw=0.8, label="real input"),
          axes[0].scatter([], [], s=13, marker="^", facecolor="white", edgecolor=INK2, lw=0.8, label="Time-MoE (others: Chronos-T5)"),
          axes[0].plot([], [], color=INK, lw=1.1, label="median")[0],
          axes[0].plot([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="position-only oracle")[0]]
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.85))
    fig.legend(handles=hs, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.53, 1.0),
               handletextpad=0.3, columnspacing=1.1, fontsize=6.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_trap.{ext}", bbox_inches="tight")
    plt.close(fig)
    return vals


# ---------------------------------------------------------------- Figure 2: count / perceive / keep
def fig_ladder():
    P1, P3, MB, AI = S["P1_counting"], S["P3_keeping_time"], S["model_vs_content_baseline"], S["accumulated_vs_integrator"]
    rows = ALL[::-1]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharey=True)
    titles = ["Token position", "Gap size", "Elapsed time"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, color=INK)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.axhspan(-0.5, 1.5, color="#f3f2ee", zorder=0)  # patch models band
        ax.tick_params(axis="y", length=0)
    # count
    ax = axes[0]
    for i, m in enumerate(rows):
        ax.plot([P1[m]["iid_noise"]["layer0_r"], P1[m]["iid_noise"]["best_r"]], [i, i], color=GRID, lw=1.2, zorder=1)
        ax.scatter(P1[m]["iid_noise"]["layer0_r"], i, s=16, facecolor="white", edgecolor=BLUE, lw=1.0, zorder=3)
        ax.scatter(P1[m]["iid_noise"]["best_r"], i, s=18, color=BLUE, zorder=3)
    ax.set_xlabel("$r$ (white-noise input)")
    # perceive
    ax = axes[1]
    for i, m in enumerate(rows):
        ref = MB[m]["iid|0.5"]["baseline_real"]
        ax.plot([ref, ref], [i - 0.32, i + 0.32], color=REF, lw=2.0, solid_capstyle="round", zorder=2)
        ax.scatter(P3[m]["median_real"], i, s=18, color=BLUE, zorder=3)
    ax.set_xlabel("$r$ (real input)")
    # keep
    ax = axes[2]
    for i, m in enumerate(rows):
        if AI.get(m):
            a = AI[m]["iid|0.7"]
            ax.plot([a["integrator_m_perp"]] * 2, [i - 0.32, i + 0.32], color=REF, lw=2.0, solid_capstyle="round", zorder=2)
            ax.scatter(a["model_m_perp"], i, s=18, color=BLUE, zorder=3)
        else:
            ax.text(0.02, i, "n/a (patch targets)", va="center", fontsize=6.2, color=INK2)
    ax.set_xlabel("$r$ (real input)")
    axes[0].set_yticks(y, [LABEL[m] for m in rows])
    # legend
    hs = [axes[0].scatter([], [], s=18, color=BLUE, label="model (best layer)"),
          axes[0].scatter([], [], s=16, facecolor="white", edgecolor=BLUE, lw=1.0, label="model input layer"),
          axes[0].plot([], [], color=REF, lw=2.0, label="no-model reference")[0]]
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.89))
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.6, 1.0), handletextpad=0.3, columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_ladder.{ext}", bbox_inches="tight")
    plt.close(fig)


def table_numbers(trapvals):
    P1, P3, MB, AI = S["P1_counting"], S["P3_keeping_time"], S["model_vs_content_baseline"], S["accumulated_vs_integrator"]
    out = {}
    for m in ALL:
        r = dict(count=P1[m]["iid_noise"]["best_r"], count_in=P1[m]["iid_noise"]["layer0_r"],
                 perceive_real=P3[m]["median_real"], perceive_noise=P3[m]["median_noise"],
                 p3_delta=P3[m]["median_delta"], p3_ci=P3[m]["ci95"], p3_layer=P3[m]["layer"],
                 over_reader=MB[m]["iid|0.5"]["median_model_minus_baseline"], over_reader_ci=MB[m]["iid|0.5"]["ci95"],
                 reader=MB[m]["iid|0.5"]["baseline_real"])
        if AI.get(m):
            for k in ("iid|0.7", "block|0.7", "iid|0.5", "block|0.5"):
                r["keep_" + k] = [AI[m][k]["model_m_perp"], AI[m][k]["integrator_m_perp"],
                                  AI[m][k]["median_model_minus_integrator"], AI[m][k]["ci95"]]
        out[m] = r
    out["trap"] = {"|".join(k): v for k, v in trapvals.items()}
    (OUT / "numbers.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    d = load()
    tv = fig_trap(d)
    # fig_ladder() is the earlier single-draw version of Figure 2; the paper uses paper_figs_v2.py
    nums = table_numbers(tv)
    print(json.dumps(nums, indent=0)[:6000])
