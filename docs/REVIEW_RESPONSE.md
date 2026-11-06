# Design Review Response

This release addresses a review of the first published experiment. The changes focus on the selection objective, training coverage, workload realism, and operational usefulness. The dataset is collected before fitting, with training, selection, and evaluation labels kept separate during analysis.

The completed [6 November 2026 A100 experiment](RESULTS.md) now provides measured evidence for this workflow. Its original archive and full metrics are preserved under [`experiments/a100_20261106/`](../experiments/a100_20261106/README.md). Implementation, measured results, and remaining limitations are distinguished below.

## Implemented changes

| Review concern | Implementation | Evidence boundary |
|---|---|---|
| Held-out data measured only after model freeze | Collect every split before fitting; enforce train-only inputs to grouped CV | Test labels may exist on disk. No freeze-before-measurement marker is required. |
| Median selection hides bad tails | Weighted mean APE selection, P90 tie-break, full tail reporting | The objective is declared, not a universal best metric. |
| One small development split | Grouped CV by N/K family; refit on all non-test training rows | CV scores are tuning evidence, not independent test results. |
| Handcrafted workload grid | Pinned public model configurations, operation-derived shapes and weights | Configurations are not real traffic or runtime traces. |
| Irregular shapes dominate ordinary reports | Separate irregular stress split | Stress remains visible; it is not removed to improve an aggregate score. |
| Missing wave structure | Shared geometric features with candidate-tile/SM proxies | Actual kernel tile identity is not assumed or leaked from a profiling run. |
| Warm-reuse-only contract | Separate rotating-buffer collection in eligible L2 regimes | Rotation does not prove cold caches or DRAM traffic. |
| Fixed 50-GEMM cost for every shape | Bounded pilot and recorded adaptive replay count | The A100 records retain actual graph counts from 1 to 256; the 2 ms replay target is not guaranteed for every case. |
| FP16 always precedes BF16 | Seeded interleaving within a collection | Instantaneous hardware state is still not perfectly controlled. |
| Three entire repeated grids | One full collection plus a deterministic 77-of-609 repeat subset in the default workflow | The completed repeat is within the same session; it does not establish independent-session reproducibility. |
| Profiling only two shapes | Separate kernel-name profiling for every shape and precision after timing | All 1,218 primary and 154 repeat captures contain CUDA kernel events; each is warm reuse, with no per-rotation capture or hardware counters. |
| Acquisition uses a weak model | Remove it from the mandatory new pipeline; retain historical appendix evidence | No current active-learning time-saving claim. |
| No layer-level check | Optional random-weight decoder block and hybrid component estimate | Not pretrained model quality, full-model prediction, or serving performance. |
| No cross-architecture evidence | Explicit fixed-source transfer command for a second GPU dataset | Command support is not evidence until target GPU data are collected. |

## Review conclusions that require qualification

The stored candidate scores support the concern that median APE can conceal large errors. A different development-only objective can select another stored candidate. Because this change was proposed after examining published outcomes, it is a retrospective explanation and design proposal. It does not establish that the metric was the only cause of generalization error.

The historical buffer-rotation sensitivity shows that the cache contract matters for specific shapes. It does not establish that four buffers reproduce a full LLM's cache behavior or that every small-M case has the same penalty.

Tile strings in profiler names can help investigate a measured case, but requiring them as predictive features would make an unseen prediction depend on executing/profiling that case first. This release therefore uses declared geometric proxies and records kernel names separately.

The review's GPU-time estimates are planning estimates. The collector uses memory and elapsed-time limits and preserves partial output. The completed worker took approximately 38 minutes 56 seconds on its assigned A100; that observation is not a duration guarantee for a different run or device.

## Measured findings and open model issues

The CV-selected residual KRR achieves 7.71%–10.32% weighted mean APE on ordinary tests, lower than the fitted three-parameter Roofline comparator in each context. These are same-run prediction-error comparisons. They do not measure GPU acceleration, and changes in workload or protocol prevent a causal comparison with the first-release percentages.

The measured results also expose limitations that this publication does not silently correct:

- Warm unseen-family challenge error is about 29%; FLOP-only performs better on that particular held-out scope. The original CV choice remains unchanged.
- Small narrow matrices can require corrections larger than the implemented 4x upper bound. Passing timing-quality checks do not remove this modeling limitation.
- The training alignment indicators are constant zero. Irregular stress cases exercise unseen alignment states and produce 49.47%–57.31% weighted mean error; their residuals are not actively clipped.
- Six primary rows and one repeat control group exceed declared timing-quality thresholds. Their data remain present. Typical repeat changes are about 1%–2%, with a maximum of 28.19%.

These findings motivate future overhead-aware models, training-only selection of correction policies, irregular training coverage, and targeted timing diagnostics. They are not implemented improvements or new untouched-test results. See [RESULTS.md](RESULTS.md) for case-level examples and [VALIDATION.md](VALIDATION.md) for the audit scope.

## Historical baseline

The recorded first-release data, original model selections, and publication checksums remain unchanged under `results/`, with their findings in [BASELINE_RESULTS.md](BASELINE_RESULTS.md). Historical freeze and claim files describe the acquisition protocol that actually produced those data. They are not requirements of the current collector.

The [retrospective review tables](../analysis/review_selection/README.md) are stored separately from the original evidence. New measured results belong in new run directories, with a clear measurement kind and actual environment.

## Further hardware and application evidence

The completed A100 run establishes the recorded single-device experiment. Cross-architecture transfer, hardware-counter explanations, production request frequencies, and the optional decoder-layer comparison need separate measurements. CPU tests verify data-use boundaries and software behavior, while the published GPU archive establishes the current measured execution. Neither is a serving-system speedup or benchmark certification.
