# Position and Elapsed Time in Sparse-Time Probes of Time Series Foundation Models

Akanksha Gupta (3am Labs) · NeurIPS 2026 Workshop on Foundation Models for Temporal Systems (FMTS)

**Paper:** [`paper/camera_ready.pdf`](paper/camera_ready.pdf)

When missing steps are dropped and the remaining values are passed to a time series foundation model (TSFM) without timestamps, do the model's hidden states still reflect elapsed time? A natural linear-probe protocol answers yes, even from white-noise inputs. This repository contains:
- the controls that expose that failure: held-out-series probes, matched-keep white noise, a position-only oracle, and probes that use no model;
- the results of applying those controls to eight TSFMs on 102 UCR series.

**Main results**
- All eight models encode token position from white noise (r ≥ 0.94).
- They encode the size of individual gaps about as well as a no-model probe on local value changes.
- Where the values reveal the gaps, a running sum of that probe's gap estimates tracks elapsed time better than every model, under linear and MLP probes.

Models: Chronos-T5 tiny/mini/small/base/large, Chronos-Bolt-base, Chronos-2, Time-MoE-50M.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Camera-ready LaTeX source, PDF, figures, bibliography, NeurIPS checklist, and full appendix tables (`APPENDIX_TABLES.md`) |
| `experiments/` | All camera-ready experiment and analysis code, per-series results, and summaries |
| `docs/ANALYSIS_PLAN.md` | Analysis plan, written before the main sweep. It uses the paper's working title. |
| `docs/notes/` | Working notes from the two sweeps; the paper is authoritative where they differ |
| `original_submission/` | Code for the submitted version; reused by the replication check |

Inside `experiments/`:

| File | Role |
|---|---|
| `core.py`, `v2core.py` | Missingness patterns, probe targets, Monte Carlo position-only oracle, held-out-series ridge and MLP probes |
| `extractors.py` | Hidden-state extraction for each model family |
| `modal_run.py` | Main sweep (three missingness patterns × four drop rates × real/noise × three evaluation schemes), replication and reference modes |
| `modal_v2.py` | Robustness sweep: value-dependent missingness, three draws, MLP probes, patch-level targets |
| `content_baseline.py`, `integrator_baseline.py`, `v2_refs.py`, `v2_integrator_raw.py` | No-model references: difference probe and running sum |
| `analyze.py`, `v2_analyze.py` | Summaries: `eda/summary.json`, `results_v2/summary_v2.json` |
| `paper_figs.py`, `paper_figs_v2.py`, `gen_appendix.py` | Paper figures and appendix tables |
| `prep_series.py`, `series.npz`, `series_meta.json` | Series selection rule and the 102 z-scored UCR series |
| `results/`, `results_v2/` | Per-series probe correlations (gzipped CSV), replication and oracle tables |

## Reproducing the analysis (CPU)

```bash
pip install -r requirements.txt
cd experiments
gunzip -k results/*.csv.gz results_v2/*.csv.gz
python analyze.py          # main-sweep summary   -> eda/summary.json
python v2_analyze.py       # robustness summary   -> results_v2/summary_v2.json
python paper_figs.py && python paper_figs_v2.py && python gen_appendix.py   # -> ../paper/
```

## Re-running the models (GPU)

The model runs use [Modal](https://modal.com), one NVIDIA A10G per model, about 5 GPU-hours in total.

```bash
modal run experiments/modal_run.py --mode replicate    # reproduces the submitted 0.821 / 0.759
modal run experiments/modal_run.py --mode references
modal run experiments/modal_run.py --mode full
modal run experiments/modal_v2.py --mode full
modal run experiments/v2_refs.py
```

Pinned versions:
- Chronos models: `torch==2.6.0`, `transformers>=4.48,<4.58`, `chronos-forecasting==2.3.2`.
- Time-MoE: `torch==2.4.1`, `transformers==4.40.1`.

## Data and models

UCR Time Series Archive 2018 ([Dau et al., 2019](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)); please cite the archive if you use `experiments/series.npz`.

Model weights on Hugging Face (Apache-2.0):
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

## License

MIT. See [`LICENSE`](LICENSE).
