# Camera-ready experiments: results and EDA

**Run:** 5 Nov 2026, Modal A10G. **Status:** complete for all 8 models. No paper text has been written from this yet; this file is for discussion.
**Pre-registration:** [PREREGISTRATION.md](PREREGISTRATION.md) (written before any result; addendum A1 also written before any result).
**Raw data:** `cr/results/*.csv` (one row per model × drop pattern × drop rate × input × layer × target × CV × series). **Summary numbers:** `cr/eda/summary.json`. **Figures:** `cr/eda/figs/`.
**Code:** [modal_run.py](modal_run.py), [core.py](core.py), [analyze.py](analyze.py), [content_baseline.py](content_baseline.py), [integrator_baseline.py](integrator_baseline.py).

---

## 0. The short version

1. **Every model counts steps.** On pure white noise, all 8 models decode a token's position in the sequence at r = 0.95–0.996. Chronos-T5's input layer carries no position at all (r ≈ 0), so the count is built inside the network. Chronos-2's input layer already carries it (r = 0.96), because Chronos-2 feeds a step counter as an input feature.
2. **The trap is real under the drop patterns people actually use, not only under our Beta keep.** With random point dropout (iid) and sensor-outage dropout (block), the submitted protocol (shuffled folds within one series) reports r ≈ 0.54–0.75 for "elapsed time" **on white noise**. Contiguous folds bring that to ≈ 0.03–0.12, and training on some series while testing on others brings it to ≈ 0.0–0.1. The mechanism is interpolation along the sequence, exactly as reviewer v2FR suspected. Under the Beta keep it is position geometry: models match the position-only oracle (≈ 0.92).
3. **Models do see the size of individual gaps, but only as much as the data shows.** In the honest test (iid drops, gap target, cross-series probe), one-token-per-step models decode each gap at r ≈ 0.4 on real data and ≈ 0 on noise. This passes the pre-registered bar. But a trivial content reader (a linear probe on neighbouring values and their differences, no model) gets most of the way. The best models add only +0.03 to +0.10 over it, and T5-tiny, Bolt and Chronos-2 fall *below* it. Gap visibility tracks how smooth the series is (Spearman ρ ≈ 0.6–0.7 with lag-1 autocorrelation). It is a property of the signal, not a clock in the model.
4. **No model accumulates gaps into elapsed time.** Accumulated time (the timestamp after removing the position trend) is barely decodable from any model (r ≈ 0.1–0.29). At high sparsity and under outages, a trivial integrator (add up the content reader's gap estimates) beats every one-token-per-step model, with CIs excluding zero. So the models count steps and perceive local change, but don't keep time.
5. **Scale buys counting, not timekeeping.** Across Chronos-T5 from 8M to 710M parameters, counting rises from 0.973 to 0.996. Gap perception rises from tiny to mini, then is flat (0.40, 0.40, 0.43, 0.43). The margin over the trivial reader peaks around small/base and shrinks at large. Accumulation does not improve.
6. **Patch models and Chronos-2 behave differently.** They count perfectly, but their per-patch time targets are weak and noisy, and they show little gap perception beyond the trivial reader. Chronos-2 is the only model that fails the pre-registered keeping-time test. Treat patch-model rows with care (§7).

**In one line for the title:** *counting steps* is near-perfect everywhere. *Keeping time*, meaning accumulating elapsed time beyond what a trivial reader of the data gets, is absent. The title holds, but the evidence is now more precise and more interesting than the submitted paper's.

---

## 1. What was run

| | |
|---|---|
| Models | Chronos-T5 tiny / mini / small / base / large (8M → 710M; encoder; one token per observation) · Chronos-Bolt-base (T5 encoder; 16-step patches) · Chronos-2 (encoder, RoPE, step-counter input feature; 16-step patches) · Time-MoE-50M (causal decoder, RoPE, one token per observation) |
| Series | 102 UCR 2018 univariate datasets (first training series; length 128–512; z-scored), selected by a rule fixed in advance |
| Drop patterns | **beta** (the submitted paper's Beta(0.35,4) keep: dense head, sparse tail) · **iid** (random point dropout) · **block** (outages: alternating observed/missing runs, mean outage 10 steps) |
| Drop rates | 0.1, 0.3, 0.5, 0.7 |
| Inputs | real values at kept positions · white noise at the same positions |
| Targets | `k` (token position) · `m_perp` (accumulated time after removing the linear position trend, as submitted) · `dt` (gap to the previous kept point) |
| Probes | ridge λ = 1 · **primary: cross-series** (train on 80% of series, test on held-out series) · secondary: within-series shuffled folds (the submitted protocol) and within-series contiguous folds |
| References (no model) | Monte Carlo oracle E[target given position] · spline of position with the same cross-series CV · content-only reader of local values (exploratory) · integrator of the content reader (exploratory) |

**Replication check: passed.** The new pipeline reproduces the submitted numbers exactly: median 0.821 (real) and 0.759 (noise) at L12, in both bf16 and fp32 (`results/replication.json`).

**Data hygiene:**
- Per-series results are missing for 0.1% of step-model cells and 22.5% of patch-model cells. The patch gap comes from short series at high drop rates having fewer than 8 patches; pooled r is reported alongside.
- Layer for the keeping-time test is chosen on a fixed 30% of series (seed 2026) and evaluated on the other 70% (71 series).

---

## 2. Finding 1: counting is universal (P1 passes for all 8)

Best-layer cross-series r for token position, iid drops, p = 0.5:

| Model | Noise input | Real input | Input layer (noise) |
|---|---|---|---|
| t5-tiny | 0.973 | 0.919 | −0.01 |
| t5-mini | 0.987 | 0.964 | −0.01 |
| t5-small | 0.989 | 0.964 | −0.01 |
| t5-base | 0.995 | 0.982 | 0.00 |
| t5-large | 0.996 | 0.987 | −0.01 |
| bolt-base | 0.989 | 0.948 | 0.19 |
| chronos-2 | 0.995 | 0.987 | **0.96** |
| timemoe-50m | 0.951 | 0.929 | 0.00 |

What stands out:
- **The count is built, not given (T5, Time-MoE).** The input embedding has no position (r ≈ 0); one or two layers later it's 0.65–0.85; by mid-stack it's ≥ 0.95 ([figs/layers_iid.png](eda/figs/layers_iid.png)). Chronos-T5's encoder is bidirectional with only *relative* position biases. NLP work predicted such encoders would struggle to learn absolute position without position embeddings (Haviv et al. 2022). Chronos learns it anyway.
- **Chronos-2 is handed the count** (input-layer r = 0.96), matching its design (a "time index" that is a step counter).
- **Bolt's input layer is at 0.19.** Plausibly the left-padded first patch and the mask channel leak position; minor.
- **Real data slightly *lowers* counting** (by 0.01–0.05), consistently. Content competes with position in the representation.
- Results are the same across drop patterns. Counting doesn't care how points were dropped, as expected if it is pure step counting.

---

## 3. Finding 2: the trap under drop patterns used in practice (P2)

### 3a. Under the submitted Beta keep: position geometry (P2a)
Cross-series noise r for m_perp, best layer, against the position-only oracle (≈ 0.76–0.94 across drop rates) ([figs/trap_heatmap.png](eda/figs/trap_heatmap.png)):

- **All Chronos-T5 sizes, Bolt, Chronos-2:** 0.91–0.98, at or above the oracle (the spline reference is ≈ 0.94; the oracle table is binned). P2a passes: these models exploit the full position geometry.
- **Time-MoE: 0.11–0.39.** P2a fails for this row (below 50% of the oracle). A causal decoder doesn't linearly expose the Beta warp from noise. A token can't see how many tokens come after it, and the warp depends on the token's rank relative to the whole sequence.
  This is a real architectural difference, not a bug. Time-MoE still counts (0.95), but its count is "how many came before", not "where am I in the window". Worth a sentence in the paper.

### 3b. Under iid and block drops: interpolation (P2b passes)
Noise-input r(m_perp), final layer, p = 0.5 ([figs/cv_schemes.png](eda/figs/cv_schemes.png)):

| Model | iid shuffled (submitted protocol) | iid contiguous | iid cross-series | block shuffled | block contiguous |
|---|---|---|---|---|---|
| t5-tiny | 0.63 | 0.08 | 0.05 | 0.68 | 0.12 |
| t5-mini | 0.70 | 0.08 | 0.02 | 0.71 | 0.11 |
| t5-small | 0.68 | 0.04 | 0.02 | 0.70 | 0.11 |
| t5-base | 0.66 | 0.03 | 0.06 | 0.65 | 0.09 |
| t5-large | 0.68 | 0.10 | 0.01 | 0.65 | 0.10 |
| timemoe-50m | 0.70 | 0.11 | 0.00 | 0.71 | 0.09 |
| *position-only oracle* | *≈ 0* | | | *≈ 0* | |

**What this means:**
- Under the drop patterns people actually use, the target has no position geometry (oracle ≈ 0). But the submitted protocol still reports r ≈ 0.54–0.75 **from white noise**. On real data the same protocol reports 0.67–0.93, which looks like strong evidence of a clock.
- The illusion comes from shuffled folds. Every held-out token has its neighbours in the training set, and accumulated time is smooth along the sequence, so any representation that varies smoothly with position interpolates it.
- This directly answers the reviewers' request: re-run under iid (meta-review, v2FR #1), keep laws from practice (v2FR #1), and blocked CV (v2FR #4). **The trap is not specific to Beta(0.35,4). It takes a different form under each drop pattern: geometry under Beta, interpolation under iid and block.**

### 3c. Patch models under iid/block
Bolt and Chronos-2 show noise r(m_perp) of 0.07–0.37 under iid. That is consistent with their *patch-level* position-only oracle (0.45 at iid, p = 0.5): averaging timestamps within a left-padded patch creates position structure. This is an artefact of the patch target, not evidence about the models.

---

## 4. Finding 3: gaps are visible, roughly as much as the data shows (P3 and exploratory)

### 4a. Pre-registered test (iid, Δt target, cross-series, p = 0.5; 71 evaluation series)

| Model | Layer | Real | Noise | **Median Δ (real − noise)** | 95% CI | Pre-registered "keeps time"? |
|---|---|---|---|---|---|---|
| t5-tiny | 4/4 | 0.265 | 0.016 | 0.205 | [0.19, 0.31] | yes |
| t5-mini | 4/4 | 0.422 | 0.015 | 0.400 | [0.29, 0.46] | yes |
| t5-small | 5/6 | 0.429 | 0.016 | 0.398 | [0.30, 0.47] | yes |
| t5-base | 12/12 | 0.449 | 0.017 | 0.433 | [0.29, 0.53] | yes |
| t5-large | 23/24 | 0.434 | 0.000 | 0.432 | [0.29, 0.51] | yes |
| bolt-base | 9/12 | 0.108 | −0.022 | 0.139 | [0.08, 0.20] | yes (weak) |
| chronos-2 | 2/12 | 0.098 | −0.001 | −0.027 | [−0.13, 0.21] | **no** |
| timemoe-50m | 7/12 | 0.398 | 0.001 | 0.375 | [0.32, 0.45] | yes |

**By the pre-registered criterion, 7 of 8 models "keep time".** Say this plainly: it's the opposite of what I expected, and we wrote the criterion down before looking. The next two analyses (not pre-registered, labelled exploratory) show what it consists of.

### 4b. Exploratory: is it more than reading local change?
A content-only reader: the same cross-series ridge, but the "hidden state" is just neighbouring values and their differences (bidirectional for the encoder models, past-only for Time-MoE, patch summaries for patch models). On real data it gets r = 0.31–0.37 for gaps (iid, p = 0.5); on noise, ≈ 0.

Paired per-series difference, model minus reader (real input, gap target), with 95% CI ([figs/model_vs_baseline.png](eda/figs/model_vs_baseline.png)):

| Model | iid p=0.3 | iid p=0.5 | iid p=0.7 | block p=0.5 |
|---|---|---|---|---|
| t5-tiny | −0.11 [−0.14, −0.05] | −0.10 [−0.13, −0.07] | −0.13 [−0.17, −0.09] | −0.06 [−0.12, −0.03] |
| t5-mini | +0.01 [−0.02, 0.03] | +0.01 [−0.02, 0.03] | +0.01 [−0.02, 0.04] | +0.03 [0.00, 0.07] |
| t5-small | +0.08 [0.04, 0.10] | +0.04 [0.02, 0.06] | +0.05 [0.02, 0.08] | +0.10 [0.07, 0.12] |
| t5-base | +0.08 [0.05, 0.14] | +0.05 [0.04, 0.08] | +0.04 [0.02, 0.06] | +0.08 [0.06, 0.11] |
| t5-large | +0.08 [0.03, 0.10] | +0.03 [0.00, 0.06] | +0.01 [−0.03, 0.03] | +0.08 [0.07, 0.10] |
| bolt-base | −0.06 [−0.15, 0.06] | +0.10 [−0.08, 0.19] | −0.12 [−0.24, −0.04] | −0.09 [−0.16, −0.05] |
| chronos-2 | +0.03 [−0.08, 0.15] | −0.00 [−0.12, 0.09] | −0.14 [−0.43, 0.14] | −0.13 [−0.27, −0.01] |
| timemoe-50m | +0.03 [0.00, 0.07] | +0.01 [−0.01, 0.05] | −0.00 [−0.03, 0.07] | +0.06 [0.04, 0.08] |

**Reading:** almost all of the "keeping time" signal is information any trivial reader of the values gets. Chronos-T5 small/base/large add a small, reliable margin (+0.03 to +0.10, likely nonlinear reading of change through the quantized tokens). That margin does not grow with size beyond small/base.

### 4c. Exploratory: gap visibility is a property of the signal
Per-series earned time against series smoothness ([figs/content_clock.png](eda/figs/content_clock.png)):

| | t5-tiny | t5-mini | t5-small | t5-base | t5-large | timemoe | bolt | chronos-2 |
|---|---|---|---|---|---|---|---|---|
| Spearman ρ with lag-1 autocorrelation | 0.52 | 0.60 | 0.68 | 0.69 | 0.70 | 0.68 | −0.08 | 0.26 |

Smooth series reveal their gaps (a big jump means a lot of time passed); rough or noisy series don't. Series length plays no role (ρ ≈ 0). This is Mach's "time is an abstraction we arrive at through the changes of things", measured. It holds for the step models, not for the patch models.

---

## 5. Finding 4: no accumulation, so no clock (exploratory)

A clock *adds up* ticks. If a model kept time, accumulated time (`m_perp` under iid/block, where position carries no information) should be decodable. It barely is: best-layer real r ≈ 0.10–0.29 for every one-token-per-step model, against ≈ 0.4 for individual gaps ([figs/layers_iid.png](eda/figs/layers_iid.png), middle column).

Comparison with a **trivial integrator**: take the content reader's gap estimates and add them up. Model minus integrator, paired per series (real input, m_perp):

| Model | iid p=0.5 | iid p=0.7 | block p=0.5 | block p=0.7 |
|---|---|---|---|---|
| t5-tiny | −0.04 [−0.20, 0.16] | **−0.23** [−0.40, −0.07] | **−0.21** [−0.30, −0.10] | **−0.20** [−0.29, −0.11] |
| t5-mini | −0.09 [−0.18, 0.15] | **−0.23** [−0.30, −0.17] | **−0.15** [−0.20, −0.05] | **−0.13** [−0.25, −0.10] |
| t5-small | −0.02 [−0.16, 0.11] | **−0.20** [−0.35, −0.08] | **−0.20** [−0.30, −0.12] | **−0.21** [−0.27, −0.10] |
| t5-base | +0.04 [−0.07, 0.18] | **−0.30** [−0.42, −0.20] | −0.06 [−0.23, 0.03] | **−0.17** [−0.24, −0.06] |
| t5-large | −0.04 [−0.12, 0.06] | **−0.25** [−0.33, −0.12] | **−0.09** [−0.19, −0.01] | **−0.19** [−0.21, −0.09] |
| timemoe-50m | −0.02 [−0.17, 0.22] | **−0.24** [−0.37, −0.03] | −0.09 [−0.19, 0.07] | **−0.16** [−0.28, −0.01] |

(Bold = CI excludes 0. Integrator median r: iid 0.08/0.41, block 0.33/0.41 at p = 0.5/0.7.)

**Reading:**
- When there's little time information to accumulate (iid, p = 0.5), nobody does well, and models tie with the integrator.
- When gaps are large and informative (high sparsity, outages), a five-line running sum beats every model, and the models don't improve with size.
- **The models perceive local change but don't integrate it into elapsed time.**

This is the sharpest support for the title, and it is new relative to the submitted paper.

---

## 6. Finding 5: scale (P4, Chronos-T5 family, iid p = 0.5)

| Size | Params | Counting (noise, best layer) | Gap perception (pre-registered Δ) | Margin over trivial reader | Accumulation vs integrator (iid 0.7) |
|---|---|---|---|---|---|
| tiny | 8M | 0.973 | 0.21 | −0.10 | −0.23 |
| mini | 20M | 0.987 | 0.40 | +0.01 | −0.23 |
| small | 46M | 0.989 | 0.40 | +0.04 | −0.20 |
| base | 200M | 0.995 | 0.43 | +0.05 | −0.30 |
| large | 710M | 0.996 | 0.43 | +0.03 | −0.25 |

**Scale buys counting, not timekeeping.** Counting saturates. Gap perception jumps from tiny to mini and is then flat. The margin over a trivial reader peaks at base. Accumulation stays below a running sum at every size. Five points from one family is a trend, not a law; say so.

Layer note: the best layer for gaps moves deeper with size (tiny 4/4 … large 23/24), while counting peaks mid-to-late stack. The submitted paper's L8 peak for m_perp (beta) persists for t5-base.

---

## 7. Patch models (Bolt, Chronos-2): what we can and can't say

- **Can say:** they count perfectly (0.99). Chronos-2 is handed the count as an input. Under Beta drops they show the same trap (0.96–0.97, matching the patch oracle 0.997).
- **Can't say much about timekeeping.** Per-patch targets average 16 observations. Under iid drops the mean gap per patch has little variance, so the Δt target is weak, and 22.5% of per-series values are missing (too few patches). Gap perception is weak (Bolt 0.14; Chronos-2 −0.03 with a wide CI) and generally at or below the patch-level trivial reader.
- **Design point worth one sentence in the discussion:** Chronos-2's grid + mask + step-counter design is the right one *when a grid exists*. Fed compressed observations (this protocol), its counter is a counter, not a clock. We did **not** test Chronos-2 on the masked grid (NaN inputs); that would be the natural follow-up for the forecast gap test.

---

## 8. Pre-registration scorecard

| Prediction | Result |
|---|---|
| **P1** counting ≥ 0.9 on noise, all models | **Pass, 8/8** (0.95–0.996) |
| **P2a** under Beta, noise ≥ 70% of oracle | **Pass, 7/8**; **fails for Time-MoE** (causal decoder; 12–42% of oracle) |
| **P2b** under iid, within-series shuffled noise ≥ 0.3 (interpolation) | **Pass, all step models** (0.54–0.75) |
| **P3** keeping time: median Δ > 0.10 and CI > 0 | **Pass 7/8**, Chronos-2 fails. Exploratory analyses (§4b, §5) show the signal is mostly content any reader gets, and is not accumulated |
| **P4** scale | Reported (§6) |

**Not pre-registered, flagged as exploratory:** content-only reader, integrator, smoothness correlation, all drop rates other than 0.5 for P3.

---

## 9. What this changes for the paper (to discuss before writing)

1. **The claim gets stronger and more precise.** Submitted: "the probe can't tell position from time (one model, one keep)". Now: across 8 models, 3 drop patterns, 4 drop rates and 102 series:
   - the naive probe is fooled in two different ways (geometry under Beta; interpolation under iid and block);
   - models count steps almost perfectly;
   - the time information in their states is what the data's own changes reveal, and it is not accumulated into elapsed time.
2. **Title fits.** "Counting Steps Is Not Keeping Time" is now directly supported by §2 + §5. Watch one tension: P3 passes. The paper must say models *perceive gaps*, explain that this is the data's change, and avoid "no time information at all".
3. **Reviewer requests: all addressed.**
   - iid re-run: §3b.
   - Keep laws from practice: iid and block.
   - Trap to avoid: §3, reframed.
   - More series: 8 → 102. More sparsity levels: 4.
   - Second model: 3 additional architectures plus the size sweep.
   - Blocked CV: §3b.
   - Spline as a reference: oracle and spline both reported.
4. **Practitioner sentence (reviewer 8Fbz's request for concrete motivation):** "If your pipeline drops missing points and feeds the rest to a TSFM, the model knows which step it's on, sees how much the values jumped, and nothing more. It does not track how much time has passed."
5. **Candidate figures for 4 pages:**
   - (1) dense-head/sparse-tail picture plus the CV-scheme bars ([cv_schemes.png](eda/figs/cv_schemes.png));
   - (2) layer profiles for k / Δt / m_perp, real vs noise, a few models ([layers_iid.png](eda/figs/layers_iid.png));
   - (3) the counting-vs-keeping plane or the scale table;
   - (4) model vs trivial reader and integrator.

## 10. Caveats

- One keep draw per series per condition; variability is across series, not across draws.
- Linear probes only. A nonlinear probe might find more accumulated time. That would be a stronger claim to make than to refute, and we don't make it.
- The trivial reader uses raw floats, which Chronos sees quantized, so the reader is a slightly generous comparison for Chronos-T5.
- Time-MoE input is z-scored per series (its documented usage); Chronos models use their own scaling.
- Patch-model targets are patch averages (§7).
- UCR first training rows are short, curated signals. Real irregular data (missing not at random, asynchronous) is not tested.
- §4b, §4c and §5 are exploratory and should be labelled that way in the paper.

## 11. Questions for our discussion

1. Lead with the trap (methodological, the accepted paper's identity) and build to "count, perceive, don't accumulate"? Or lead with the three-part model finding? The camera-ready rules favour the first.
2. How much of §5 (integrator) goes in the main text vs the appendix, given it's exploratory?
3. Time-MoE's Beta exception (causal models don't see the warp): main text sentence or appendix?
4. Do we add the masked-grid (NaN) Chronos-2 run tomorrow morning? It's about 20 GPU-minutes and makes the practical "drop vs NaN" sentence evidence-based. But it's new evidence for a design claim, so it could also wait for the poster.
