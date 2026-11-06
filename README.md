# Position and Elapsed Time in Sparse-Time Probes of Time Series Foundation Models

Code, analysis plan and per-series results for the FMTS @ NeurIPS 2026 workshop paper (camera-ready).
Akanksha Gupta, 3am Labs.

When missing observations are dropped and the remaining values are passed to a time series foundation model (TSFM) without timestamps, do the model's representations still reflect elapsed time? This repository contains:
- the probing experiments on eight TSFMs (Chronos-T5 tiny to large, Chronos-Bolt-base, Chronos-2, Time-MoE-50M) over 102 UCR series;
- the controls the probe needs: held-out-series evaluation, white-noise inputs, a position-only oracle, and no-model reference probes;
- every result in the paper, per series.

## Layout

| Path | Contents |
|---|---|
| `code/` | The package from the original submission (`sparse_time_probes`): keep laws, probes, control tasks, original figures. Used by the replication check. |
| `cr/core.py`, `cr/v2core.py` | Missingness patterns, probe targets, Monte Carlo position-only oracle, ridge and MLP held-out-series probes |
| `cr/extractors.py` | Hidden-state extraction for each model family |
| `cr/modal_run.py` | Main sweep on Modal: 8 models × {Beta, random, block} × p ∈ {0.1, 0.3, 0.5, 0.7}, real and noise inputs, three evaluation schemes; replication and reference modes |
| `cr/modal_v2.py` | Robustness sweep: value-dependent missingness, three draws, MLP probes, patch-level targets |
| `cr/content_baseline.py`, `cr/integrator_baseline.py`, `cr/v2_refs.py`, `cr/v2_integrator_raw.py` | No-model references: difference probe and its running sum |
| `cr/analyze.py`, `cr/v2_analyze.py` | Analyses producing `cr/eda/summary.json` and `cr/results_v2/summary_v2.json` |
| `cr/paper_figs.py`, `cr/paper_figs_v2.py`, `cr/gen_appendix.py` | Paper figures and appendix tables |
| `cr/PREREGISTRATION.md` | Analysis plan, written before the main sweep |
| `cr/RESULTS_EDA.md`, `cr/RESULTS_V2.md` | Exploratory reports on the main and robustness sweeps |
| `cr/results/`, `cr/results_v2/` | Per-series probe correlations (gzipped CSV), summaries, replication and oracle tables |
| `cr/series.npz`, `cr/series_meta.json` | The 102 z-scored UCR series used (selection rule in `cr/prep_series.py`) |

## Reproducing

Analyses and figures run on CPU from the shipped results:

```bash
pip install numpy pandas scikit-learn matplotlib torch
cd cr
gunzip -k results/*.csv.gz results_v2/*.csv.gz
python analyze.py            # main sweep summary
python v2_analyze.py         # robustness sweep summary
python paper_figs.py && python paper_figs_v2.py && python gen_appendix.py
```

Model runs use [Modal](https://modal.com) with one NVIDIA A10G per model (about 5 GPU-hours in total):

```bash
modal run cr/modal_run.py --mode replicate    # reproduces the original 0.821 / 0.759
modal run cr/modal_run.py --mode references
modal run cr/modal_run.py --mode full
modal run cr/modal_v2.py --mode full
modal run cr/v2_refs.py
```

Pinned versions:
- Chronos models: `torch==2.6.0`, `transformers>=4.48,<4.58`, `chronos-forecasting==2.3.2`.
- Time-MoE: `torch==2.4.1`, `transformers==4.40.1`.

## Data and models

UCR Time Series Archive 2018 (Dau et al., 2019); please cite the archive if you use `cr/series.npz`. Model weights are public on Hugging Face:
- `amazon/chronos-t5-{tiny,mini,small,base,large}`
- `amazon/chronos-bolt-base`
- `amazon/chronos-2`
- `Maple728/TimeMoE-50M`

## Citation

```bibtex
@inproceedings{gupta2026position,
  title     = {Position and Elapsed Time in Sparse-Time Probes of Time Series Foundation Models},
  author    = {Gupta, Akanksha},
  booktitle = {NeurIPS 2026 Workshop on Foundation Models for Temporal Systems},
  year      = {2026}
}
```

## Licence

MIT (see `LICENSE`).
