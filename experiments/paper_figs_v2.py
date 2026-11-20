"""Figure 2 from the three-draw (v2) results. Ridge probes; MLP numbers are in Table 1.

python experiments/paper_figs_v2.py   ->  paper/figures/fig2_v2.{pdf,png}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_figs as pf  # shared style, palette, labels

HERE = Path(__file__).resolve().parent
S2 = json.loads((HERE / "results_v2" / "summary_v2.json").read_text())
S1 = json.loads((HERE / "eda" / "summary.json").read_text())
ALL = pf.ALL


def cell(m, law, p, key):
    return S2["cells"][f"{m}|{law}|{p}"][key]


def main():
    rows = ALL[::-1]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), sharey=True)
    titles = ["(a) Token position", "(b) Gap size vs difference probe", "(c) Elapsed time vs running sum"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, color=pf.INK, pad=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.xaxis.grid(True, color=pf.GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.axhspan(-0.5, 1.5, color="#f3f2ee", zorder=0)
        ax.tick_params(axis="y", length=0)

    def ref_tick(ax, x, i, model_r=None):
        ax.plot([x, x], [i - 0.32, i + 0.32], color=pf.REF, lw=2.0, solid_capstyle="round", zorder=2)
        if model_r is not None:  # draw the model-minus-reference difference
            ax.plot([x, model_r], [i, i], color=pf.REF, lw=0.9, zorder=1)

    for i, m in enumerate(rows):
        # (a) position from noise, random dropout p=0.5: best layer (v2, 3 draws) and input embedding (v1)
        a = cell(m, "iid", 0.5, "ridge|k")["noise"]
        a0 = S1["P1_counting"][m]["iid_noise"]["layer0_r"]
        axes[0].plot([a0, a], [i, i], color=pf.GRID, lw=1.2, zorder=1)
        axes[0].scatter(a0, i, s=16, facecolor="white", edgecolor=pf.BLUE, lw=1.0, zorder=3)
        axes[0].scatter(a, i, s=18, color=pf.BLUE, zorder=3)
        # (b) gap, real input, random dropout p=0.5, vs difference probe
        b = cell(m, "iid", 0.5, "ridge|dt")
        ref_tick(axes[1], b["reader"], i, b["real"])
        axes[1].scatter(b["real"], i, s=18, color=pf.BLUE, zorder=3)
        # (c) elapsed time, real input, value-dependent p=0.7, vs running sum (same reference as Table 1)
        c = cell(m, "mnar", 0.7, "ridge|m_perp")
        ref_tick(axes[2], c["running_sum"], i, c["real"])
        axes[2].scatter(c["real"], i, s=18, color=pf.BLUE, zorder=3)
    axes[0].text(0.5, 0.5, "patch tokens", fontsize=6, color=pf.INK2, style="italic", ha="center", va="center")
    axes[0].set_xlabel("$r$, white-noise input")
    axes[1].set_xlabel("$r$, real input")
    axes[2].set_xlabel("$r$, real input")
    axes[0].set_yticks(y, [pf.LABEL[m] for m in rows])
    hs = [axes[0].scatter([], [], s=18, color=pf.BLUE, label="model (best layer)"),
          axes[0].scatter([], [], s=16, facecolor="white", edgecolor=pf.BLUE, lw=1.0, label="model input layer"),
          axes[0].plot([], [], color=pf.REF, lw=2.0, label="no-model reference")[0]]
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.87))
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.6, 1.0),
               handletextpad=0.5, columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(pf.OUT / f"fig2_v2.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
