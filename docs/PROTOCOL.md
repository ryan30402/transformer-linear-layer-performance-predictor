# Dataset Collection and Evaluation Protocol

This document describes protocol 1.1.0. The first-release measurement snapshot retains its original [baseline protocol](BASELINE_PROTOCOL.md) and verifier. Historical labels and model decisions are not rewritten.

## 1. Offline data workflow

1. Build a workload manifest from pinned public model configurations. Assign train, test, challenge, and irregular stress roles using shapes only.
2. Collect all declared splits and precision/cache contexts. No predictor is fitted during this stage.
3. Pass only training rows to family-grouped cross-validation. Fit preprocessing on each training fold, tune hyperparameters, and select the model.
4. Refit each model family's chosen configuration using every training row.
5. Evaluate the already available test, challenge, and stress labels. Report all candidates and preserve the selected model.

The held-out boundary is a data-use boundary, not an acquisition-time boundary. Test labels may have been measured before fitting. They must not influence feature selection, hyperparameters, the model family, or reporting choices. The software does not create a one-way held-out claim marker or require model hashes to predate test measurement.

Changes informed by the published baseline are development decisions. Re-evaluating them on the same known labels is retrospective analysis. New generalization claims require a new evaluation set.

## 2. Workload construction

`linear_perf/workloads.py` derives GEMM dimensions from pinned model configuration files. It records hidden/intermediate widths, attention and key/value heads, vocabulary size, operation counts, and the source revision and checksum.

The manifest separates:

- New M values within training N/K families (`test`).
- Entire held-out N/K families (`challenge`).
- Irregular dimensions (`stress`), excluded from model selection and reported separately.

Training M values cover small decode-sized rows and larger prefill-sized rows. The shape plan is not a runtime trace. Operation-occurrence weights and the declared M grid are a controlled evaluation distribution; they are not production request frequencies. Shared shapes are deduplicated before splitting, and previously published shapes are excluded from new held-out membership.

The resource guard may exclude a full-vocabulary projection or a large prefill case. Exclusions retain the original dimensions and reason. The generator does not shrink a published model to make a case fit.

## 3. Measurement target

The primary operator is preallocated `torch.mm(X, W.T, out=Y)`, with represented FP16 or BF16 inputs, inference mode, and a recorded precision policy. CUDA events surround a graph replay and the elapsed time is divided by the actual captured GEMM count.

Input/weight/output GPU accesses and replay gaps inside the interval are included. Allocation, CPU/GPU transfers, graph capture, correctness computation, and request queues are excluded. This is a GEMM-level target, not an isolated hardware-counter measurement or a full-model response time.

Each case has multiple timed visits within one collection. Within-visit event means and visit medians are retained separately. The complete dataset does not require three repeated full collection campaigns. A deterministic shape-only 12.5 percent subset can be collected in another session to assess repeatability.

## 4. Adaptive replay and cache policies

A bounded pilot estimates how many GEMMs are needed for the configured replay interval. The final count has explicit minimum/maximum bounds and is saved alongside the raw elapsed times. This reduces excessive repetition for large GEMMs while keeping very short intervals measurable.

Warmup uses a declared elapsed-time and replay-count rule with a cap. Adaptation uses pilot duration, not model error or a preferred benchmark result. Quality warnings remain visible.

`warm_reuse` repeatedly uses one buffer set. `rotating_buffers` cycles through multiple sets and forms a separate dataset/model context. Rotation is eligible for the declared small-weight-to-L2 regime, subject to memory limits. It is not a guarantee of cold cache or complete L2 eviction. The operation count covers whole buffer cycles.

The collector obtains L2 capacity from supported runtime/driver metadata. An explicit configured override is recorded. If capacity is unavailable, the rotation policy is marked unavailable rather than guessing a GPU model's cache size.

## 5. Precision ordering, hardware and correctness

The schedule interleaves FP16 and BF16 in a seeded order instead of collecting all FP16 before all BF16. Both rows retain their actual device/environment identity. Each dtype/cache context is fitted independently.

GPU model, UUID where available, SM count, software stack, math policy, resource decisions, controls, and coarse telemetry are retained. No claim of locked clocks, absent interference, or hardware counters is made unless actually measured.

Correctness checks are outside the event interval and use the same represented inputs. A precision capability failure is explicit. There is no automatic fallback from BF16 to FP16, from GPU to CPU, or from real measurement to simulation.

## 6. Profiling and geometric features

The all-case profiling phase is separate from primary timing. Torch traces record kernel names and execution events for the warm-reuse contract. Missing profiler capability is a recorded status; profiler durations do not replace training labels.

Predictive features use shapes and known hardware metadata. Candidate tile sizes produce wave-utilization proxies. Parsed kernel names, actual measured tile choices, and profiler timings are diagnostic evidence, not mandatory inputs for an unseen prediction. A kernel name is not a hardware-counter proof.

## 7. Evaluation

The primary selection objective is out-of-fold weighted mean APE, with P90 APE and fixed candidate order used for ties. N/K families do not cross training/validation fold boundaries. All eligible training data are used for final refitting.

Reports also show unweighted errors, median and P90/worst APE, absolute-time errors, signed errors, and per-shape predictions where provided. A cross-validation tuning score is not an unbiased estimate of the selected model's future performance. Test and challenge results serve that separate role.

Cache contexts and hardware identities are not silently pooled. A cross-GPU analysis uses a fixed source model with target hardware metadata and no target-label refitting, and is labeled as transfer. Repeated measurements of known shapes test repeatability, not new-shape generalization.

## 8. Separate experiments

The original two-parameter-Roofline acquisition experiment remains historical appendix evidence. It is not a mandatory stage in the new pipeline and is not evidence that active sampling saves GPU time for the revised selector.

The decoder-layer experiment measures a random-weight block and its standalone components. A composition of predicted GEMM costs plus independently measured non-GEMM costs is a hybrid estimate. It is compared with the separately measured whole block. It is not a fully predicted LLM, a serving benchmark, or a measurement of pretrained task quality.

## 9. Validation boundaries

CPU tests and simulated runs validate software. A GPU run is needed to validate CUDA execution, cache effects, hardware support, and the revised predictor's empirical accuracy. Historical data can verify calculations and guide design but cannot establish prospective improvement for a method chosen after examining those data.

The current standard manifest has 609 shapes per dtype: 311 train, 134 ordinary test, 118 family challenge and 46 irregular stress. Planning exclusions are explicit. Two ordinary-test vocabulary projections exceed their own surviving training M range after memory exclusions; per-case coverage metadata identifies them. Ordinary test therefore means unseen M in a known N/K family, not guaranteed interpolation in every case.

## Complete fresh experiment orchestration

The default Colab entry point executes primary collection and analysis, a deterministic 12.5% repeat, and a CPU comparison in one workflow. It uses no previous timing datasets or fitted predictors. Primary collection completes all splits before train-only modeling. The repeat reuses the exact workload manifest and selected source models; its labels never trigger refitting. A combined report links raw data, model evaluations and stage-specific quality qualifications. Each collection has its recorded wall-time budget. An unrestricted default total-GEMM-call counter avoids terminating large valid grids at an arbitrary fixed operation count; explicit operation caps remain supported, while time, per-graph, warmup and memory bounds still apply.

A complete default result requires primary, repeat and comparison to complete. Timing-quality qualifications are reported independently. Required-stage failures retain completed evidence and a failed/interrupted experiment status. An explicitly requested optional layer stage that cannot complete is not silently marked successful.

## Durable checkpoint boundaries

The Colab entry point keeps active datasets on local VM storage and requires a
verified initial copy in the selected Google Drive directory before GPU
collection. Evidence snapshots are then created synchronously after each full
timing visit, after timing completes and before profiler capture starts, after
validated collection, and after major experiment stages. The final result ZIP is
also copied and verified. Existing session folders and published files are not
overwritten.

Each snapshot is first compressed locally, copied to a unique temporary name in
the backup directory, flushed, read back and checked with SHA256. The final ZIP
and its checksum receipt are published only after successful copy verification.
These checks confirm bytes read from the mounted filesystem; they do not attest
the storage provider's internal replication or guarantee protection for work
completed after the most recent verified snapshot.

No backup compression or upload runs concurrently with a CUDA event timing
interval. Checkpoint start/end times, outcome and duration are retained in
`run/checkpoints.jsonl`. Backup pauses count toward the collection wall-time
budget and may alter temperature or clocks before the next case. Declared
warmup, controls, telemetry and timing-quality qualifications still apply; the
implementation does not claim that checkpoint pauses have zero hardware effect.

Checkpoint archives preserve the recorded state. A snapshot taken before
profiling or finalization can contain complete timing records while its dataset
is still incomplete under the full validation contract. Recovery never rewrites
that status, silently completes missing stages, or merges measurements from a
replacement GPU/runtime. The separate download notebook can retrieve existing
archives without the original Python variables. See [COLAB_RECOVERY.md](COLAB_RECOVERY.md).
