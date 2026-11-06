# Results: Six A100 Sessions

These are the unchanged first-release results under the [baseline protocol](BASELINE_PROTOCOL.md). The current default pipeline is described in [PROTOCOL.md](PROTOCOL.md). The [retrospective selection review](../analysis/review_selection/README.md) is a separate analysis of known data, not a replacement of the original decisions or new GPU evidence.

The first release includes six measured sessions: FP16 and BF16 across three recorded A100 devices. The data support two distinct findings: the primary timing labels were consistent across these sessions, and model selection on the small development set did not reliably choose the lowest held-out-error candidate.

All percentages below are absolute percentage errors unless stated otherwise. Figures are rounded for readability; the [summary CSVs](../results/summary/) retain full precision.

## 1. Evidence and environment

| Item | Recorded scope |
|---|---|
| GPU model | NVIDIA A100-SXM4-40GB |
| Device identities | Three distinct recorded UUIDs |
| Sessions | Three FP16 and three BF16; each same-number pair shares a UUID |
| Shape set | The same 96 modeling/acquisition shapes in every session, plus eight diagnostics |
| Main splits | 24 calibration, 12 development, 24 test, 12 challenge |
| Acquisition pool | 24 labels per session |
| Python / PyTorch | 3.12.13 / 2.11.0+cu128 |
| CUDA runtime / driver | 12.8 / 580.82.07 |
| Primary target | Warm-reuse graph replay event time divided by 50 GEMMs |

The precision order was FP16 before BF16 within each pair. The experiment does not randomize dtype order or control every clock, power, temperature, and concurrent-load effect. Three recorded device identities also do not represent a broad sample of all A100 hardware or cloud runtimes.

Per-session source records and publication checksums are indexed in the [dataset guide](../results/README.md).

## 2. Official development-selected results

Each model family fits only the 24 calibration labels. Development scores select hyperparameters and the official model. The table retains the choice made before held-out collection.

| Session | Selected model | Dev median | Test median | Test P90 | Challenge median | Challenge P90 |
|---|---|---:|---:|---:|---:|---:|
| FP16 s01 | Shape-log Ridge | 13.72% | 25.66% | 79.30% | 31.96% | 76.45% |
| FP16 s02 | Shape-log Ridge | 14.12% | 26.53% | 78.97% | 31.75% | 76.51% |
| FP16 s03 | Roofline + residual KRR | 13.96% | 17.18% | 37.09% | 21.17% | 37.56% |
| BF16 s01 | Roofline, three parameters | 13.63% | 48.28% | 60.59% | 51.83% | 71.28% |
| BF16 s02 | Roofline, three parameters | 13.47% | 48.50% | 60.54% | 51.63% | 71.40% |
| BF16 s03 | Roofline, three parameters | 12.77% | 48.35% | 59.85% | 51.64% | 71.38% |

Sources: [selected pipeline](../results/summary/selected_pipeline.csv), [all model metrics](../results/summary/model_metrics.csv).

The BF16 selection is a clear failure of development-set selection to generalize to these held-out shapes: development median error is about 13%, while test median error is about 48%. For FP16 s03, residual KRR beats Ridge on development by only **0.1143 percentage points**. That small margin changes the selected family.

The different M values and family coverage make an unrepresentative development set a plausible explanation, but the measurements do not isolate a single cause. Changing the selector after reading these results would be a new design decision requiring new untouched evaluation data.

## 3. All registered candidate models

The ranges below are the minimum and maximum of **three per-session median APE values**, not confidence intervals or a pooled estimate. All five model families were registered and frozen before each session's held-out evaluation.

| Dtype | Candidate | Test median range | Challenge median range | Selected sessions |
|---|---|---:|---:|---:|
| FP16 | FLOP-only | 59.77-60.08% | 67.63-67.80% | 0/3 |
| FP16 | Roofline, two parameters | 54.06-54.28% | 54.46-55.66% | 0/3 |
| FP16 | Roofline, three parameters | 21.78-22.36% | 32.44-32.74% | 0/3 |
| FP16 | Shape-log Ridge | 25.66-26.53% | 31.75-31.96% | 2/3 |
| FP16 | Roofline + residual KRR | 16.56-17.18% | 21.00-21.17% | 1/3 |
| BF16 | FLOP-only | 54.99-55.74% | 61.96-63.27% | 0/3 |
| BF16 | Roofline, two parameters | 57.36-57.57% | 59.65-59.78% | 0/3 |
| BF16 | Roofline, three parameters | 48.28-48.50% | 51.63-51.83% | 3/3 |
| BF16 | Shape-log Ridge | 24.93-25.32% | 40.56-40.69% | 0/3 |
| BF16 | Roofline + residual KRR | 16.44-16.75% | 24.25-24.30% | 0/3 |

Source: [model_metrics.csv](../results/summary/model_metrics.csv).

Residual KRR has the lowest test median APE among these candidates in all six sessions. That is a descriptive result of this registered comparison; it does not make KRR the official selected pipeline in five sessions where development chose something else.

KRR also does not dominate every metric. For example, BF16 s01 challenge RMSE is approximately **152.02 us for KRR** and **99.64 us for Ridge**, despite KRR's lower median APE. A good typical relative error can coexist with costly absolute errors on larger shapes. Full per-case predictions and tail metrics should accompany any accuracy claim.

## 4. Measurement quality

Every recorded session passed its **5,468 internal audit checks** and its declared timing gates. Those checks cover integrity, aggregation, coverage, numerical witnesses, model decisions, and chronology. The count is not a number of independent GPU experiments or a third-party certification.

| Session | Maximum within-visit CV | Maximum block-median ratio | Maximum control ratio |
|---|---:|---:|---:|
| FP16 s01 | 3.03% | 1.0652 | 1.0808 |
| FP16 s02 | 3.25% | 1.0896 | 1.0349 |
| FP16 s03 | 3.63% | 1.0349 | 1.0516 |
| BF16 s01 | 2.18% | 1.0327 | 1.0415 |
| BF16 s02 | 3.23% | 1.0241 | 1.0521 |
| BF16 s03 | 2.03% | 1.0639 | 1.1724 |

Source: [quality_and_audits.csv](../results/summary/quality_and_audits.csv). Ratio columns are maximum divided by minimum; the declared limits are 1.2. The within-visit CV limit is 10%.

### BF16 s03 control qualification

Control `x03` spans **19.865-23.290 us**, a maximum/minimum ratio of **1.172428**, or **17.24% relative to the minimum**. It passes the 1.2 gate but remains a visible timing qualification. No observations were removed or rescaled, and no causal explanation is established.

Across the three rounds, the largest per-shape maximum/minimum ratio among the 96 primary/pool targets was **1.02717 for FP16** and **1.01933 for BF16**, corresponding to spans of 2.72% and 1.93% relative to the minimum. This supports consistency of those aggregate labels in this dataset. It does not contradict the larger control excursion, which samples a different shape repeatedly within a session. See [cross-round spread](../results/summary/cross_round_latency_spread.csv).

## 5. Frozen-source transfer

The first session of each dtype is the declared source. Its already-fitted model is applied unchanged to the same registered shapes measured in later sessions; the source is not chosen by test score.

| Source model | Target | Test median APE | Challenge median APE |
|---|---|---:|---:|
| FP16 s01 Ridge | FP16 s02 | 26.51% | 31.98% |
| FP16 s01 Ridge | FP16 s03 | 26.25% | 31.86% |
| BF16 s01 three-parameter Roofline | BF16 s02 | 48.37% | 51.75% |
| BF16 s01 three-parameter Roofline | BF16 s03 | 48.67% | 51.78% |

Source: [frozen_selected_transfer.csv](../results/summary/frozen_selected_transfer.csv).

The errors transfer consistently, including the large BF16 errors. Stability is not the same as accuracy. The [frozen KRR candidate transfer](../results/summary/frozen_krr_candidate_transfer.csv) is retained as a separate registered-candidate analysis, without replacing the source model selected by development.

This is same-model hardware transfer among the recorded A100 devices and sessions. It does not establish transfer to an H100, another GPU architecture, another runtime, or another timing contract.

## 6. Uncertainty in model comparisons

The reported comparison resamples N/K families in paired fashion, with 1,000 resamples. For FP16 s03, selected KRR minus three-parameter Roofline gives:

| Split | Observed median-APE difference | Descriptive 95% percentile range |
|---|---:|---:|
| Test | -5.18 percentage points | -33.62 to +3.56 |
| Challenge | -11.27 percentage points | -48.23 to +2.53 |

Source: [`fp16_s03/results.json`](../results/a100/fp16_s03/results.json), `selected_vs_roofline_3p` in each split's evaluation.

Negative favors KRR, but both ranges cross zero. The observed gains do not establish a uniform or broadly generalizable advantage. The few families, shared shape structure, and session effects constrain inference. These ranges are not prediction intervals for a new shape.

## 7. Acquisition and batching

The release preserves every heuristic/random acquisition result at 2, 4, 8, and 12 revealed labels across ten seeds. This is an **offline label-revelation experiment** over a fully measured 24-shape pool. The [all-budget table](../results/summary/acquisition_all_budgets.csv) supports comparison without choosing a favorable test seed. It does not measure GPU collection time saved by a deployed adaptive sampler.

A separate offline batching experiment compares 64 GEMMs with M = 16 against one GEMM with M = 1024, with N = K = 1024 and equal total arithmetic work. Recorded total-work time ratios range from **25.57x to 26.72x** across sessions. The [batching data](../results/summary/offline_batching.csv) preserve both times.

That ratio describes the controlled offline workload. It does not include request arrivals, queueing, batching delay, model execution outside this GEMM, or service-level constraints, so it is not an end-to-end LLM serving speedup.

## 8. Conclusions supported by this release

The benchmark retains evidence that can be checked on a CPU: raw timing samples, aggregation, frozen predictions, model choices, and small numerical witnesses. The six-session experiment exposes a practical modeling limitation: an accurate development score can coexist with poor held-out generalization, even when measurement variation is modest.

Residual KRR is a promising candidate under the measured contract. The evidence does not establish a universally reliable selector, calibrated per-shape prediction intervals, cross-architecture accuracy, an observed hardware bottleneck, or full-model inference acceleration. Future work should preserve this baseline and evaluate changes on new untouched shapes, additional environments, and separately specified application workloads.
