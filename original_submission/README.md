# Original submission code

Code for the submitted version of the paper, titled **Dual Collinearity in Sparse Time Probes of Chronos** at review time. The camera-ready experiments are in [`../experiments`](../experiments); this package is kept because the camera-ready replication check (`experiments/modal_run.py --mode replicate`) reuses its keep sampler and probe.

Under order-preserving sparse sampling, residualized observation time \(m^\perp\) remains a smooth function of token index. Linear probes of this target measure positional feature capacity, not metric-time inference. This repository reproduces the keep geometry, index-only control tasks, and paper figures. The Chronos encoder audit is optional and requires a GPU.

## Requirements

Python 3.10+ with NumPy, SciPy, scikit-learn, and Matplotlib.

```bash
python -m pip install -r requirements.txt
```

The Chronos encoder audit additionally needs PyTorch, `chronos-forecasting==1.5.0`, and `transformers==4.46.3`.

## Reproduce figures (CPU)

Figure 1 is generated from the Beta keep law (no model weights). Figures 2–3 plot the frozen encoder-audit numbers shipped in `sparse_time_probes/assets/paper_results.json`.

```bash
python scripts/make_figures.py --out figures
```

Outputs:

| File | Content |
|---|---|
| `figures/fig1_geometry.{pdf,png}` | Dual collinearity: \(m\) vs \(k\), then \(m^\perp\) vs \(k\) with quantile residual \(\psi(k)\) |
| `figures/fig2_forest.{pdf,png}` | Per-series held-out \(r(H,m^\perp)\) vs the spline(\(k\)) ceiling |
| `figures/fig3_layers.{pdf,png}` | Encoder-layer medians for \(r(H,i)\) and \(r(H,m^\perp)\) |

## Control-task ceilings (CPU)

```bash
python scripts/run_control_task.py
```

This redraws the Beta keep for each dataset length, residualizes \(m\) against linear index, and reports 5-fold ridge Pearson \(r\) from \(\mathrm{poly}_1(k)\), \(\mathrm{poly}_3(k)\), and a cubic spline of \(k\).

## IID keep robustness (CPU)

```bash
python scripts/run_control_task_iid.py
```

Same CV protocol under fixed-cardinality iid keep (endpoints pinned). Writes `results/control_task_iid.json` (median \(\mathrm{poly}_3=0.675\), \(\mathrm{spline}=0.802\)).

## Chronos encoder audit (GPU, optional)

```bash
python scripts/run_chronos_audit.py --cache data --out results/chronos_audit.json
```

Protocol:

- Model: `amazon/chronos-t5-base`, **encoder context tokens only**
- Data: first TRAIN series of eight UCR Archive 2018 sets, truncated to \(T\le 400\), z-scored
- Keep: Beta(\(0.35,4.0\)), endpoints pinned, seed \(=42+\) dataset index
- Probe: `Ridge(alpha=1.0)`, 5-fold shuffled `KFold`, held-out Pearson \(r\), z-scored target
- Primary row: layer 12, drop rate \(0.5\); noise confound uses the same keep, \(y\sim\mathcal{N}(0,1)\)

The UCR 2018 zip is password-protected by the archive maintainers (password `someone`). See [the UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

## Layout

```
sparse_time_probes/   library
scripts/              figure and experiment entry points
tests/                keep-geometry checks
assets/               frozen table numbers used by Figures 2–3
```

## License

MIT.
