"""Paper figures. Vector PDF plus 300 dpi PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from sparse_time_probes.config import (
    BASE_SEED,
    DATASETS,
    FIGURE1_DATASET,
    P_PRIMARY,
    T_CAP,
)
from sparse_time_probes.geometry import (
    beta_quantile,
    mean_kept_times,
    observed_times,
    ols_fit,
    psi_curve,
    residualize,
    unit_rank,
)
from sparse_time_probes.keep import keep_beta
from sparse_time_probes.probe import pearson_r
from sparse_time_probes.results import load_paper_results
from sparse_time_probes.style import BLUE, FILL, GREEN, INK, MUTED, ORANGE, VERMILLION, apply_style


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, tight: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    save_kw = {} if tight else {"bbox_inches": None, "pad_inches": 0.02}
    fig.savefig(pdf, **save_kw)
    fig.savefig(png, **save_kw)
    plt.close(fig)
    return [pdf, png]


def _panel_tag(ax, tag: str, *, y: float = 1.06) -> None:
    ax.text(
        0.0,
        y,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def figure1_geometry(out_dir: Path, dataset: str = FIGURE1_DATASET) -> list[Path]:
    """Dual collinearity: m vs k, then m^perp vs k with quantile residual psi."""
    apply_style()
    results = load_paper_results()
    row = next(s for s in results["series"] if s["name"] == dataset)
    T = int(row["T"])
    seed = BASE_SEED + DATASETS.index(dataset)
    rng = np.random.default_rng(seed)
    keep = keep_beta(T, P_PRIMARY, rng)
    k, i, m = observed_times(keep, T)
    m_perp = residualize(m, k)
    m_ols = ols_fit(m, k)
    r_im = pearson_r(i, m)

    n = int(k.size)
    k_grid = np.linspace(1.0, float(n), 400)
    q_grid = beta_quantile(unit_rank(k_grid, n=n))
    m_mean = mean_kept_times(T, P_PRIMARY)
    m_mean_perp = residualize(m_mean, k)
    psi_grid = psi_curve(k_grid, n=n)
    psi = psi_curve(k, n=n)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.75, 2.55),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.scatter(k, m, s=11, c=FILL, alpha=0.42, linewidths=0, zorder=2, rasterized=True)
    ax.plot(k, m_ols, color=BLUE, lw=1.35, zorder=3, label=r"OLS$(m\mid 1,k)$")
    ax.plot(k, m_mean, color=VERMILLION, lw=1.5, zorder=4, label=r"keep-law $\mathbb{E}[m_k]$")
    ax.plot(k_grid, q_grid, color=VERMILLION, lw=1.0, ls=(0, (3.2, 1.6)), zorder=5, label=r"$Q_{\mathrm{Beta}}(\tilde{k})$")
    ax.set_xlabel(r"token index $k$")
    ax.set_ylabel(r"normalized time $m_k$")
    ax.set_xlim(1, k.size)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="upper left", handletextpad=0.4)
    ax.text(
        0.97,
        0.06,
        rf"{dataset}, $T={T}$, $p=0.5$" + "\n" + rf"$r(i,m)={r_im:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=MUTED,
        linespacing=1.35,
    )
    _panel_tag(ax, r"(a)")

    ax = axes[1]
    ax.axhline(0.0, color=BLUE, lw=1.0, ls=(0, (3, 1.6)), zorder=1)
    ax.scatter(k, m_perp, s=11, c=FILL, alpha=0.42, linewidths=0, zorder=2, rasterized=True)
    ax.plot(k, m_mean_perp, color=VERMILLION, lw=1.5, zorder=4, label=r"$\mathbb{E}[m_k]^\perp$")
    ax.plot(k_grid, psi_grid, color=VERMILLION, lw=1.0, ls=(0, (3.2, 1.6)), zorder=5, label=r"$\psi(k)$")
    ax.set_xlabel(r"token index $k$")
    ax.set_ylabel(r"residual $m_k^{\perp}$")
    ax.set_xlim(1, k.size)
    yabs = max(np.max(np.abs(m_perp)), np.max(np.abs(psi)), np.max(np.abs(m_mean_perp))) * 1.12
    ax.set_ylim(-yabs, yabs)
    ax.legend(loc="upper right", handletextpad=0.4)
    ax.text(
        0.03,
        0.06,
        r"$\psi(k)=Q(\tilde{k})-\mathrm{OLS}(Q\mid 1,k)$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    _panel_tag(ax, r"(b)")

    return _save(fig, out_dir, "fig1_geometry")


def _forest_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color=BLUE, ls="none", markersize=4.6, label="Chronos, real $y$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=ORANGE,
            ls="none",
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.15,
            label="Chronos, white noise",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color=GREEN,
            ls="none",
            markersize=9,
            markeredgewidth=1.6,
            label=r"spline$(k)$ ceiling",
        ),
    ]


def _draw_forest(ax, *, legend: bool = True) -> tuple[float, float, float]:
    rows = load_paper_results()["series"]
    names = [r["name"] for r in rows]
    y = np.arange(len(names), dtype=float)
    real = np.array([r["r_perp_real"] for r in rows])
    noise = np.array([r["r_perp_noise"] for r in rows])
    spline = np.array([r["r_spline"] for r in rows])
    med_real = float(np.median(real))
    med_noise = float(np.median(noise))
    med_spline = float(np.median(spline))

    ax.axvline(med_spline, color=GREEN, lw=0.9, ls=(0, (4, 1.8)), zorder=1)
    ax.axvline(med_noise, color=ORANGE, lw=0.7, ls=(0, (1.2, 1.2)), alpha=0.85, zorder=1)
    ax.axvline(med_real, color=BLUE, lw=0.7, ls=(0, (1.2, 1.2)), alpha=0.85, zorder=1)

    # Slight vertical dodge so near-coincident markers stay separable.
    for yi, s, n, p in zip(y, spline, noise, real):
        ax.plot([s], [yi + 0.14], marker="|", color=GREEN, markersize=9, markeredgewidth=1.6, zorder=3)
        ax.plot(
            [n],
            [yi - 0.14],
            marker="o",
            color=ORANGE,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.15,
            zorder=4,
        )
        ax.plot([p], [yi], marker="o", color=BLUE, markersize=4.4, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"held-out $r(H, m^{\perp})$")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(len(names) - 0.55, -0.55)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    if legend:
        ax.legend(
            handles=_forest_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            borderaxespad=0.0,
            columnspacing=0.9,
            handletextpad=0.3,
            fontsize=6.0,
            frameon=False,
        )
    return med_spline, med_real, med_noise


def _draw_layers(ax, *, legend: bool = True) -> None:
    layers = load_paper_results()["layers"]
    xs = np.array([4, 8, 12], dtype=float)
    r_ord = np.array([layers["r_ord"][str(L)] for L in (4, 8, 12)])
    r_perp = np.array([layers["r_perp"][str(L)] for L in (4, 8, 12)])

    ax.plot(xs, r_ord, color=MUTED, marker="s", markersize=4.5, lw=1.35, label=r"$r(H, i)$")
    ax.plot(xs, r_perp, color=BLUE, marker="o", markersize=5.0, lw=1.45, label=r"$r(H, m^{\perp})$")
    ax.set_xticks([4, 8, 12])
    ax.set_xticklabels(["4", "8", "12"])
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("median held-out $r$")
    ax.set_ylim(0.78, 1.02)
    ax.set_xlim(3.2, 12.8)
    if legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            borderaxespad=0.0,
            columnspacing=1.0,
            handletextpad=0.3,
            fontsize=6.0,
            frameon=False,
        )


def figure2_forest(out_dir: Path) -> list[Path]:
    """Standalone forest panel (also used by the combined audit figure)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    med_spline, med_real, med_noise = _draw_forest(ax)
    ax.text(
        0.5,
        -0.22,
        rf"medians: spline ${med_spline:.3f}$ $\mid$ real ${med_real:.3f}$ $\mid$ noise ${med_noise:.3f}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=MUTED,
        clip_on=False,
    )
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.22)
    return _save(fig, out_dir, "fig2_forest", tight=False)


def figure3_layers(out_dir: Path) -> list[Path]:
    """Standalone layers panel with the same canvas geometry as the forest panel."""
    apply_style()
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    _draw_layers(ax)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.22)
    return _save(fig, out_dir, "fig3_layers", tight=False)


def figure_audit(out_dir: Path) -> list[Path]:
    """Matched two-panel audit figure for the paper (aligned axes boxes)."""
    apply_style()
    fig = plt.figure(figsize=(6.85, 2.85))
    # Identical plot boxes (figure fraction): same bottom/height/width.
    box_w, box_h, box_y = 0.36, 0.58, 0.20
    ax0 = fig.add_axes([0.11, box_y, box_w, box_h])
    ax1 = fig.add_axes([0.60, box_y, box_w, box_h])

    med_spline, med_real, med_noise = _draw_forest(ax0, legend=True)
    _draw_layers(ax1, legend=True)
    _panel_tag(ax0, r"(a)", y=1.22)
    _panel_tag(ax1, r"(b)", y=1.22)
    ax0.text(
        0.5,
        -0.24,
        rf"medians: spline ${med_spline:.3f}$ $\mid$ real ${med_real:.3f}$ $\mid$ noise ${med_noise:.3f}$",
        transform=ax0.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=MUTED,
        clip_on=False,
    )
    return _save(fig, out_dir, "fig2_audit", tight=False)


def make_all(out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    paths.extend(figure1_geometry(out_dir))
    paths.extend(figure_audit(out_dir))
    paths.extend(figure2_forest(out_dir))
    paths.extend(figure3_layers(out_dir))
    return paths
