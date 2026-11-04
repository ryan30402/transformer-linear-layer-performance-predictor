# Recorded A100 Results

This directory contains the measured-data snapshot for **Transformer Linear Layer Performance Predictor**. It includes six sessions on NVIDIA A100-SXM4-40GB GPUs: three FP16 sessions and three BF16 sessions, with the same 96 primary matrix shapes in every session. Each FP16/BF16 pair has the same recorded GPU UUID; the three pairs have three distinct recorded UUIDs.

The workload is `torch.mm(X, W.T, out=Y)` with preallocated buffers. The primary target is warm-reuse CUDA Graph replay time per matrix multiplication, measured using CUDA events around a replay of 50 captured operations. It includes replay overhead and inter-kernel gaps. Allocation, graph capture, CPU-GPU transfers, and end-to-end serving are outside this target.

## Read the results

| File | Contents |
| --- | --- |
| [summary/selected_pipeline.csv](summary/selected_pipeline.csv) | Official model selected using development labels, with held-out errors |
| [summary/model_metrics.csv](summary/model_metrics.csv) | All five candidates on development, test, and challenge shapes |
| [summary/quality_and_audits.csv](summary/quality_and_audits.csv) | Timing variation, control drift, and recorded acquisition audit outcomes |
| [summary/case_latencies.csv](summary/case_latencies.csv) | All 576 shape-session latency labels |
| [summary/cross_round_latency_spread.csv](summary/cross_round_latency_spread.csv) | Per-shape variation across three sessions within each dtype |
| [summary/paired_dtype_latencies.csv](summary/paired_dtype_latencies.csv) | FP16/BF16 observations paired by shape and session round |
| [summary/frozen_selected_transfer.csv](summary/frozen_selected_transfer.csv) | The first session's official model evaluated on subsequent sessions without refitting |
| [summary/frozen_krr_candidate_transfer.csv](summary/frozen_krr_candidate_transfer.csv) | A separate descriptive transfer analysis of the predeclared residual KRR candidate |
| [summary/acquisition_all_budgets.csv](summary/acquisition_all_budgets.csv) | Offline heuristic/random acquisition at budgets of 2, 4, 8, and 12 labels |
| [summary/offline_batching.csv](summary/offline_batching.csv) | A separate equal-work batching experiment |
| [summary/summary.json](summary/summary.json) | Concise machine-readable findings and explicit limitations |
| [summary/cross_session_summary.json](summary/cross_session_summary.json) | Detailed descriptive analysis across sessions |

### Official selected-model results

The selected model is retained even when another candidate later performs better on held-out shapes. No winner is substituted after examining test results.

| Session | Development-selected model | Test median APE | Challenge median APE |
| --- | --- | ---: | ---: |
| FP16 s01 | Shape-log Ridge | 25.66% | 31.96% |
| FP16 s02 | Shape-log Ridge | 26.53% | 31.75% |
| FP16 s03 | Roofline residual KRR | 17.18% | 21.17% |
| BF16 s01 | Three-parameter Roofline | 48.28% | 51.83% |
| BF16 s02 | Three-parameter Roofline | 48.50% | 51.63% |
| BF16 s03 | Three-parameter Roofline | 48.35% | 51.64% |

APE is absolute prediction error divided by measured latency. These medians describe errors across distinct matrix shapes, not the distribution of individual request latency.

All six runs passed the registered timing criteria. BF16 s03 still has a visible qualification: control case x03 has a maximum/minimum latency ratio of **1.172428**, below the declared 1.2 cutoff but a **17.24%** span relative to its minimum. No samples were removed, rescaled, or recollected to improve that result. Passing measurement checks does not establish accurate prediction or reliable model selection.

## Session evidence

Each directory under `a100/` contains the following evidence:

| Artifact | Purpose |
| --- | --- |
| `plan.json` | All 104 registered shapes, including 96 primary/pool shapes and eight separate diagnostic shapes |
| `environment.json` | Recorded GPU identity, software stack, precision, timing, and math policy |
| `initial_measurements.jsonl` | Aggregated labels for 24 calibration and 12 development shapes |
| `candidate_measurements.jsonl` | Aggregated labels for the 24-shape acquisition pool |
| `heldout_measurements.jsonl` | Aggregated labels for 24 test and 12 challenge shapes |
| `initial/`, `candidate_pool/`, `heldout/` | Per-visit event means, declared visit schedules, quality records, and telemetry snapshots |
| `controls.json`, `quality.json` | Interleaved controls and recorded timing criteria |
| `model_suite.json` | Five fitted candidates, coefficients, feature scaling, hyperparameter search, and development selection |
| `frozen_predictions.json`, `freeze.json`, `heldout_claim.json`, `journal.json` | Recorded predictions and chronology around the one-time held-out stage |
| `results.json`, `predictions.csv`, `group_metrics.csv` | Per-shape errors, tail metrics, grouping, and descriptive family-cluster bootstrap results |
| `acquisition_frozen.json`, `acquisition_results.json` | Frozen acquisition checkpoints and complete policy/seed/budget evaluations |
| `diagnostics/` | Eager/graph sensitivity, buffer rotation, actual replay profile traces, and small numerical witnesses |
| `batching.json` | Separate offline batching measurements and correctness records |
| `recorded_source_identity.json`, `recorded_acquisition_seal.json` | Recorded acquisition identifiers and original-file hashes, distinct from publication-file hashes |

For each main shape, three visits contribute nine event means each. The label is the median of the three visit medians. The snapshot therefore retains **15,552 primary event means**, plus separate control and diagnostic measurements. Event means, visits, and sessions are different sampling units.

Each session also includes four small NPZ numerical witnesses: zero, identity, alternating signs, and random values. The archives preserve the represented low-precision inputs, GPU outputs, and FP32 references as numeric arrays. CPU FP64 recomputation checks these 24 archived records. These are four repeated patterns, not 24 independent patterns, and they do not preserve every large benchmark tensor.

## Verify on a CPU

From the repository root, after installing the analysis dependencies:

```bash
python scripts/verify_published_results.py
```

To save a fresh verification report outside the recorded evidence:

```bash
python scripts/verify_published_results.py --output runs/published-results-verification.json
```

The verifier checks publication SHA-256 hashes, numeric payload digests, exact shape coverage, raw-repeat medians, three-visit aggregation, recorded freeze chronology, numerical witnesses, development-only selection, stored-parameter prediction replay, test/challenge errors, bootstrap intervals, acquisition checkpoints, control drift, and cross-session tables. It performs **no model fitting and no GPU work**.

Stored model coefficients are replayed using a temporary in-memory envelope compatible with the current prediction API. This envelope is never written back to the data and is never presented as a recorded pre-measurement seal. The recorded predictions and scientific decisions remain unchanged.

The `audit_checks` value of 5,468 in the summary tables belongs to the recorded acquisition reviews. It is separate from the number of checks performed by this publication verifier.

## Provenance and publication format

This is a normalized publication snapshot of already recorded experiments. The publication source tree and its metadata have different bytes from the acquisition source archive. This directory does not claim that the publication source hash generated the original observations.

During export, the acquisition archives were read without modification and their original file seals were checked. Every retained numeric JSON value was compared exactly before and after presentation normalization. Numerical witness NPZ files and the summary CSV files are byte-identical to their source artifacts. The exported snapshot retains the original measured values, fitted parameters, frozen predictions, timestamps, official model choices, and qualifications.

Presentation normalization gives labels neutral names, makes execution-local paths portable, and omits development-only registries, protocol revision labels, source-file inventories, launcher logs, generated reports, and a duplicated diagnostic summary. The original acquisition archives remain separately retained by the project owner.

[MANIFEST.json](MANIFEST.json) records both the original input hashes and the hashes of the exported files. **Embedded acquisition hashes, signatures, and environment IDs remain recorded metadata.** They are not silently recalculated to make normalized files resemble byte-identical acquisition records. Use the manifest's `sha256` fields for the published bytes. Its `original_sha256` fields and `recorded_inputs` identify acquisition artifacts; reconstructing omitted original source or full archives is outside the public verifier's scope.

Hash verification and CPU recomputation establish internal consistency of retained evidence. They do not independently attest that GPU execution occurred, eliminate shared-environment effects, or certify a benchmark standard.

## Limits on interpretation

- Only one GPU model and one software stack were measured. Cross-architecture generalization has not been measured.
- FP16 ran before BF16 within every round. Execution order and machine state can confound a causal comparison between precisions.
- Repeated event means are not independent physical GPU experiments. The sessions reuse the same 96 shapes.
- The byte-count feature is ideal tensor traffic, not measured DRAM traffic. No hardware bandwidth or utilization counters were collected.
- The acquisition study reveals labels from an already measured fixed pool. Its budgets are label counts, not GPU time saved.
- Bootstrap intervals are descriptive resampling of a small set of shape families. They are not per-prediction confidence bounds or guaranteed population coverage.
- The batching experiment measures equal offline matrix work. Its ratio is not an end-to-end LLM serving speedup.
- The recorded held-out results are now visible. Future model improvements need a newly registered unseen evaluation set for a fresh prospective accuracy claim.
