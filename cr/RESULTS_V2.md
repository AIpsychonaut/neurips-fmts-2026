# Robustness batch (v2): what changed after fixing the limitations

**Run:** 6–7 Nov 2026, Modal A10G, all 8 models complete. **Code:**
- [modal_v2.py](modal_v2.py), the GPU run;
- [v2core.py](v2core.py), the new drop pattern, patch targets and MLP probe;
- [v2_refs.py](v2_refs.py) and [v2_integrator_raw.py](v2_integrator_raw.py), the no-model references;
- [v2_analyze.py](v2_analyze.py), the analysis.

**Numbers:** `results_v2/summary_v2.json`; full printout in `results_v2/analysis_print.txt`. Hidden states for the MLP layers are cached on the Modal volume (`fmts-cr-vol/v2states/`), so further probes are CPU-only.

## What was run

| Limitation | Fix |
|---|---|
| Linear probes only | 2-hidden-layer MLP probe (256 units, early stopping on held-out training series, same folds as the ridge probe) at 5 layers per model: 1/3, 1/2 and 2/3 depth, the last layer, and the pre-specified gap layer. The no-model references were also given an MLP version, so each probe is compared with a reference of the same class. |
| One drop draw per series | 3 draws per series and condition. Draw 0 reproduces the original seeds. Per-series r is averaged over draws before the median. |
| Difference probe sees raw values | Added a version that sees Chronos-T5's quantized values. |
| Patch-model targets were weak | New patch targets: time of the last observation in each patch (elapsed time), and time advanced since the previous patch (gap). |
| No missing-not-at-random data | New **value-dependent** drop pattern: steps are dropped with weight exp(1.5·z(y)), so high values are more likely to be missing (a saturating sensor). Fixed count; endpoints kept. |

**Scope:**
- Drop patterns: random dropout, block outages, value-dependent.
- Drop rates: 0.5 and 0.7.
- 3 draws, 102 series, real and noise inputs.
- Layer chosen on the same 30% development series as before; reported on the other 71 (step models).

## Results

### 1. Quantization: no effect (limitation resolved)
- Quantized values differ from raw by < 0.01 (correlation 1.000).
- The difference probe scores the same either way: model-minus-probe differences change by ≤ 0.01. Example: Chronos-T5-base, random dropout, p = 0.5: +0.062 on raw vs +0.055 on quantized.

### 2. Nonlinear probe: finds slightly more, and so does the nonlinear reference (the conclusion holds)
- **The MLP changes elapsed-time r by −0.03 to +0.10 for step models** (up to +0.29 for patch models). For example, Chronos-T5-large under outages at p = 0.7: 0.27 with ridge, 0.37 with the MLP.
- **The MLP reference gains as much or more.** Its running sum goes from 0.37 to 0.50 under outages at p = 0.7.
- **So "linear probes only" no longer weakens the claim.** Model-minus-running-sum differences have the same sign under both probe classes.

### 3. Elapsed time vs running sum, averaged over 3 draws (corrects the earlier single-draw claim)
Model minus running sum, elapsed time at p = 0.7. Ridge / MLP; * means the 95% CI excludes 0.

| Model | Random dropout | Block outages | Value-dependent |
|---|---|---|---|
| Chronos-T5 tiny | −0.07* / −0.05 | −0.20* / −0.25* | −0.27* / −0.26* |
| Chronos-T5 mini | −0.07 / −0.05* | −0.14* / −0.21* | −0.20* / −0.18* |
| Chronos-T5 small | −0.09* / −0.05 | −0.12* / −0.22* | −0.24* / −0.24* |
| Chronos-T5 base | −0.05 / −0.02 | −0.13* / −0.18* | −0.17* / −0.15* |
| Chronos-T5 large | −0.09* / −0.01 | −0.15* / −0.15* | −0.14* / −0.13* |
| Time-MoE | −0.06 / −0.01 | −0.13* / −0.22* | −0.20* / −0.23* |
| Chronos-Bolt | −0.05 / −0.07 | −0.14* / −0.14* | −0.25* / −0.07* |
| Chronos-2 | +0.06 / −0.01 | −0.20* / −0.15* | −0.09* / −0.10* |

**Readings:**
- **Random dropout:** the original draw (seed 0) happened to favour the running sum. Averaged over 3 draws, models are slightly below it or tied. Elapsed time is barely decodable by anything here: step models 0.10–0.21, running sum 0.15–0.25. Under random dropout there is little elapsed time beyond position to accumulate. **The draft's "beats every model at 70% random dropout" must go.**
- **Block outages:** every model is below the running sum under both probe classes, with every CI excluding 0. The same holds at p = 0.5 except Time-MoE with ridge.
- **Value-dependent drops:** the strongest new result. The values reveal the gaps, so elapsed time becomes quite decodable from the models (0.44–0.77). The running sum of a probe on local differences still reaches 0.80–0.92. Every model falls below it under both probes, with every CI excluding 0.
- **Draw-to-draw spread** of the series median is about ±0.05 (`per_draw_median` in the summary).

### 4. Gap size vs difference probe (3 draws)
- Random dropout, p = 0.5, ridge: Chronos-T5 small, base and large exceed the difference probe by +0.04, +0.06 and +0.03 (CIs above 0).
- Mini and Time-MoE tie with it; tiny is −0.12.
- Against the MLP difference probe, the margins are +0.03 to +0.04.
- Under value-dependent drops with the MLP, most models are slightly *below* the MLP difference probe.
- The original conclusion holds: models are within about 0.1 of the probe, with a small edge for mid-to-large Chronos-T5.

### 5. Patch models with the new targets (now conclusive)
- **Gap (time advanced per patch) is now strongly decodable:**
  - real data: 0.44–0.85 depending on pattern;
  - noise: ≤ 0.15 (except random dropout at p = 0.7: about 0.5, because the left-padded first patch is shorter and position predicts it).
- **Use real-minus-noise, not raw r, for patch gaps.** The patch reference shows the same first-patch effect. Patch models are at or slightly below the patch difference probe.
- **Elapsed time (time at patch end):** both patch models fall below the patch running sum under outages and value-dependent drops (every CI excludes 0). Under random dropout they tie.
- **The patch rows now follow the same pattern as the step models.** About 22% of per-series values are still missing for patch models (short series with too few patches); the medians use the rest.

### 6. Position (counting): unchanged
r(k) from noise is 0.94–0.996 for all models under all three patterns, including value-dependent.

## What this changes in the draft

1. **The elapsed-time claim** now rests on block outages and value-dependent drops, where every model falls below the running sum under both linear and nonlinear probes. Under random dropout, report a tie: little elapsed time is available beyond position there.
2. **Add the value-dependent pattern** to Setup and Table 1. It is the most realistic and the most demanding test, because the values reveal the gaps.
3. **Add the MLP probe and 3 draws** to Setup. Remove the corresponding limitations; keep "MNAR is one specific mechanism".
4. **Patch models** move from "inconclusive" to "same pattern", with the first-patch caveat.
5. **Remove the quantization limitation.**
