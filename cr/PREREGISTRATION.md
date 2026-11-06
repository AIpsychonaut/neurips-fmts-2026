# Camera-ready pre-registration: Counting Steps Is Not Keeping Time

**Written:** 5 Nov 2026, before any camera-ready GPU run. Do not edit after the first run. Corrections go in a dated addendum at the bottom.
**Scope:** experiments that answer the FMTS reviews: iid keep, keep laws from practice, more series and sparsity levels, a second model and more, blocked/transfer CV, spline treated as a reference.

## 1. Fixed choices

| Item | Choice |
|---|---|
| Models | `amazon/chronos-t5-{tiny,mini,small,base,large}` (encoder states), `amazon/chronos-bolt-base` (encoder, patch tokens), `amazon/chronos-2` (encoder, patch tokens), `Maple728/TimeMoE-50M` (causal decoder, one token per step) |
| Series | Every UCR 2018 univariate dataset; first TRAIN row; NaN padding dropped; truncated to the first 512 steps; kept if length ≥ 128 and std > 0; z-scored. Rule fixed before any data is seen |
| Keep laws | `beta` (Beta(0.35,4), endpoints pinned; the submitted paper's law), `iid` (uniform without replacement, fixed cardinality, endpoints pinned; MCAR point dropout), `block` (alternating observed/missing runs, geometric lengths, mean outage 10 steps, endpoints pinned; sensor-outage pattern) |
| Drop rates p | 0.1, 0.3, 0.5, 0.7 |
| Inputs | `real` (kept values), `noise` (N(0,1) at the same kept positions; same keep draw) |
| Feeding | Compressed: kept values only, in order, no timestamps (the protocol under audit) |
| Targets (step-token models) | `k` (token rank), `m_perp` (normalized time, OLS-residualized on [1,k] per series), `dt` (gap to previous kept point / T) |
| Targets (patch models) | Per patch of compressed tokens: patch index, mean m, the same residualization, mean dt |
| Probe | Ridge λ = 1 on standardized hidden states, targets z-scored within series |
| **Primary CV** | **Cross-series**: 5-fold GroupKFold by series. Per held-out series Pearson r; summary = median over series |
| Secondary CV | Within-series shuffled 5-fold (the legacy protocol), within-series blocked 5-fold (contiguous). Step-token models only |
| References | k-only spline (8 knots) learned with the same CV; **Monte Carlo oracle** E[target given k, keep law, T, p], 200 draws, with no learning |
| Replication check | chronos-t5-base, beta, p = 0.5, the 8 original series truncated at 400, within-series shuffled CV, final encoder layer: should reproduce the submitted 0.821 (real) and 0.759 (noise) within ±0.05. If it doesn't, stop and debug before reading any other result |

## 2. Predictions and kill criteria

**P1, counting (all models).** On noise input, some layer decodes `k` with cross-series median r ≥ 0.9.
*Kill:* any model below 0.7 at every layer. Then "counting" doesn't hold for that model, and the paper says so for that row.

**P2, the trap is geometry plus interpolation.**
- (a) Under beta, the cross-series noise r(m_perp) reaches at least 70% of the MC oracle r.
- (b) Under iid and block, the oracle is near 0, so the cross-series noise r(m_perp) is near 0. But within-series shuffled CV shows high r (interpolation).

*Kill (a):* noise below 50% of the oracle for most models. Then the models don't exploit the position geometry.
*Kill (b):* within-series shuffled noise r(m_perp) under iid below 0.3. Then interpolation is not the mechanism.

**P3, keeping time (the honest test).** iid keep, `dt` target, cross-series probe, p = 0.5. The median per-series paired Δr (real − noise) is the earned-time estimate.
- Evidence of keeping time: median Δr > 0.10 **and** a series-bootstrap 95% CI (2,000 resamples) that excludes 0.
- Otherwise the report reads "no evidence of keeping time under this probe", with the CI.

Layer for P3: the layer with the best real-input r on `dt`, chosen on a fixed 30% development subset of series (seeded), and evaluated on the remaining 70%.

**P4, scale (Chronos-T5 sizes).** No directional prediction. Report counting and keeping-time (the P3 metric) against parameter count.

## 3. What is reported regardless of outcome
- Every model × keep × p × input × layer cell, in CSV.
- The replication check.
- Any series or model that failed to run, and why.

## Addenda

**A1 (5 Nov 2026, before any camera-ready result; implementation clarifications only):**
1. Series rule: the archive's `Missing_value_and_variable_length_datasets_adjusted` folder duplicates 11 datasets. We use the original folder (NaN padding dropped) and exclude the adjusted copies.
2. Within-series CVs (secondary) use the submitted paper's exact ridge (raw hidden states, sklearn `Ridge(alpha=1)`) for continuity. They run at three layers per model (mid, two-thirds, last) and on the `m_perp` and `dt` targets.
3. Per-series r needs ≥ 8 tokens (patches for patch models) in that series; otherwise it is NaN for that series. Pooled held-out r over all series is reported alongside the median per-series r.
4. The beta sampler is vectorized (first-n unique draws in draw order, the same law). The replication check uses the original sequential sampler.
5. Hidden-state layer 0 is the input embedding for every model.
6. Primary GPU dtype is float32. The replication check is run in both bf16 (as submitted) and fp32.
