# Grouped Model Selection and Latency Prediction

The current model workflow selects configurations on an already measured offline dataset. Its primary objective is weighted mean absolute percentage error, evaluated out of fold by N/K family. The earlier median-only selector is retained only for verifying the recorded [baseline models](BASELINE_MODELS.md).

## Why change the selection objective

A median can hide severe errors affecting fewer than half the examples. Mean APE makes every shape's relative error contribute to selection. The workload weights describe model operation occurrences under a declared M sampling grid; they are not inferred from held-out errors or claimed to represent live traffic. Unweighted and tail metrics remain available so weighting does not hide failures.

P90 APE breaks a tie in the primary score, followed by deterministic model order. This choice is an engineering objective, not a universal industry accuracy standard. If a future use case optimizes kernel ranking, capacity, or absolute delay, its selection objective should be declared separately and evaluated on new held-out data.

## Training and validation

Only rows with `split="train"` enter `fit_cv_suite`. For historical reanalysis, calibration and development rows are explicitly mapped into the training pool; test/challenge rows are not.

Whole N/K families are allocated to folds. With a small number of families, a family is left out in each fold; larger sets use a bounded grouped partition. Every feature scaler and model is fitted on the training portion of that fold. Candidate scores come from validation predictions, with fold scores and dispersion reported.

After selection, the chosen configuration for each family is fitted again using all training rows. Test, challenge, and irregular stress rows are evaluated only after this step. Their existence on disk does not prevent ordinary offline model training.

Cross-validation scores are used for tuning. They are not advertised as independent test scores or as calibrated confidence intervals.

## Candidate families

| Candidate | Purpose |
|---|---|
| FLOP-only | Tests how much a fixed term and arithmetic amount explain. |
| Roofline 2p | A compact compute/memory proxy with no fitted offset. |
| Roofline 3p | Adds a fixed term for effects missing from the resource proxy. |
| Shape-log Ridge | Fits log latency using regularized geometric features. |
| Roofline residual KRR | Learns a multiplicative correction to the two-parameter baseline. |

The two-parameter residual baseline is retained as a declared simple reference: it limits the interaction between a separately fitted offset and the residual correction. This is a modeling choice, not a claim that it is always better than a three-parameter residual baseline. A direct 2p/3p residual ablation would be a new comparison.

## Geometry and hardware

Ridge and KRR share a declared geometric feature policy: dimensions, arithmetic/data proxies, aspect ratio, small-M behavior, alignment, tile waste, and SM-aware candidate-wave features. Means and scales are learned from training data only.

The wave features use candidate tile geometries and the recorded number of streaming multiprocessors. They are available without executing an unseen case. They do not assert that cuBLAS selected one of those exact tiles. Kernel names captured by the diagnostic profiler are excluded from predictor inputs.

A missing SM count is explicitly represented according to the implementation's fallback policy, rather than silently assuming 108 SMs. Changing architecture can change algorithm selection, cache behavior, and effective coefficients in ways a wave proxy cannot repair.

## Units and interpretation

Predictions use milliseconds. The conventional operation count is `F = 2*M*N*K`; `Q = bytes_per_element*(M*K + N*K + M*N)` is a logical tensor-byte proxy.

Effective compute, bandwidth, and offset coefficients are regression parameters. They are not measured hardware peaks, actual DRAM traffic, or directly isolated CPU launch overhead. Positivity, numerical bounds, and prediction clipping protect the fitting procedure; they do not create prediction intervals.

The predictor is evaluated separately for each environment, dtype and cache contract. No warm and rotating-buffer labels are silently mixed, and simulated observations cannot be mixed with CUDA measurements.

## Retrospective versus new evidence

Changing the criterion over the five previously fitted candidates is different from repeating hyperparameter search and refitting on a larger training pool. The review analysis keeps these comparisons separate.

Analysis of the historical six-session A100 snapshot is retrospective: those
test outcomes informed design discussion. Original selected-model scores remain
unchanged. The [completed v1.1.0 experiment](RESULTS.md) evaluates the revised
workflow on its own declared splits. Those results have now been inspected to
diagnose limitations; future model changes require new final evaluation cases
for fresh generalization claims.

## Observed limitations in the completed A100 experiment

The residual KRR multiplies a zero-offset Roofline 2p baseline by a correction
restricted to 0.25–4. On several small ordinary-test cases, the upper bound is
active and prevents a sufficiently large prediction. The bound is part of the
measured version, not a rule inferred from the test answers. An additive-overhead
baseline or a different correction bound remains a future training-CV comparison.

All N/K alignment indicators are constant zero in the training workload, while
the separate irregular stress set introduces N+1 and K+1 dimensions. Features
alone do not supply examples of the changed execution behavior. The stress
results expose this missing training support; their poor errors are not explained
by active correction clipping. Warm-reuse predictions for unseen N/K families
also retain substantial errors. See [RESULTS.md](RESULTS.md) for the exact
per-context metrics and failure examples. The predictor implementation has not
been adjusted after inspecting these results.
