# Release Validation

This record separates local software validation, offline evidence checks, and the completed **6 November 2026 A100 experiment** for protocol 1.1.0. Local checks do not establish the status of a GitHub Actions run; hardware findings come from the separately preserved measured archive.

## Completed A100 experiment

The [published experiment](../experiments/a100_20261106/README.md) completed primary collection, training-only grouped CV and refit, held-out evaluation, the predetermined repeat, and repeat comparison. Simulation was disabled. The optional decoder-layer stage was not requested.

| Check | Observed evidence |
|---|---|
| Primary dataset | 609 distinct shapes; 2,256 aggregate rows, 6,768 raw visits, and 378 control visits |
| Repeat dataset | 77 predetermined shapes; 280 aggregate rows, 840 raw visits, and 72 control visits |
| Dataset record validation | 486,647 primary and 61,547 repeat consistency checks passed |
| Raw event normalization | All 72,522 retained intervals, including controls, match elapsed time divided by captured GEMM count |
| Saved-model replay and metrics | 5,510 per-case predictions, 60 candidate metric groups, and all 12 selected summaries reproduced |
| Numerical-check records | 39,812 before/after per-buffer records, including repeated controls and profiles, satisfy their declared limits |
| Actual trace coverage | All 1,218 primary and 154 repeat trace files contain CUDA kernel events |
| Archive integrity | SHA-256 verified for all 1,444 manifest-listed files; the embedded source matches the measured release snapshot |
| Recorded hardware/software identity | Primary and repeat have the same recorded GPU UUID and matching precision/cache environment identities |
| Checkpoints | 15 verified copy/readback receipts; no failed receipt recorded |
| Completion | Every requested stage completed; worker exit code 0, no timeout or interruption |
| Timing quality | Qualified: 6 primary rows and 1 repeat control group exceed their declared thresholds |

The checks above inspect retained records and saved models without new GPU execution or refitting. Check counts are not numbers of independent experiments or external certification. Numerical checks compare with FP32 IEEE-mode `torch.mm` on the same already-represented low-precision inputs. The output/reference matrices are not archived, so offline review cannot independently recompute those products.

Prediction errors, failure cases, and baseline comparisons are reported in [RESULTS.md](RESULTS.md). Ordinary-test weighted mean APE is 7.71%–10.32%; warm unseen-family error is about 29%, and irregular stress error is 49.47%–57.31%. Every eligible observation, including quality-flagged rows, remains in the reported results. The 77-shape repeat is a later subset in the same session, not an independent-day or cross-device validation.

The exact original archive, summary CSVs, and detailed offline review are indexed in the [evidence guide](../experiments/a100_20261106/README.md). Sources inside the archive include `run/summary.json`, `run/{primary,repeat}/audit.json`, both raw datasets, `run/primary/analysis/`, and `run/checkpoints.jsonl`. Final complete-archive Drive-copy success is reported outside the ZIP and cannot be established solely from its checkpoint receipts.

## CPU software and data checks

The full unit suite and historical-evidence verifier were rerun for this publication. The integration and recovery checks below also retain the recorded release-validation evidence; simulation remains explicitly separate from the measured A100 run.

| Check | Observed result |
|---|---|
| Existing full `unittest` discovery, before the publication fixtures were added | 322 tests: 320 passed, 2 optional CPU PyTorch mathematical tests skipped because PyTorch is unavailable in the primary environment; 207.538 seconds |
| Fresh-publication integrity fixtures | 10 tests passed separately, including altered hashes, unsafe archive paths, missing entries, provenance and metric-copy mismatches |
| Fresh A100 publication verifier | Passed: all 1,444 manifest-listed archive members, primary/repeat record audits, 5,510 saved per-case predictions, 60 candidate metric groups, 12 selected summaries, grouped-CV records and paired-repeat comparisons; no refit or new GPU work |
| Optional layer suite in an isolated PyTorch CPU environment | All 9 tests passed, including the 2 mathematical checks skipped above |
| Published-result verifier | 39,708 checks passed; 264 files and 70,445,615 bytes verified |
| Original `results/` tree | All 266 tracked files unchanged relative to commit `603fcde80644fe08f2ba502feffc7e58f591a008` |
| Paired-precision simulated smoke dataset | 32 shapes, 4 precision/cache contexts, 128 aggregate rows, 256 visits |
| Smoke dataset raw-record audit | 8,947 checks passed |
| Deterministic smoke repeat subset | 4 shapes, 16 aggregate rows, 32 visits; 1,779 audit checks passed |
| Repeat comparison | 4 source contexts; no target refitting or mutation of source/target evidence |
| Independent metric recomputation | 60 metric rows recomputed from 340 per-case predictions |
| Reanalysis of existing labels | All 4 fitted model suites, metrics CSV and predictions CSV byte-identical |
| Colab payload build | Source checksums and every embedded Python file compile; notebooks have no saved execution output |
| Complete fresh-workflow integration | New primary data, train-only fitting, fixed repeat, unchanged source-model comparison and combined report completed in CPU simulation |
| Historical-data isolation | Integration test blocks historical `results/` reads while the fresh workflow completes |
| Packaged full-workflow smoke | Embedded source unpacked and primary/repeat/comparison completed; one downloadable archive produced |
| Packaged standard-profile simulation | All 609 primary shapes and the 77-shape repeat completed; 2,048 primary rows and 252 repeat rows across four precision/cache contexts; combined report and one archive exported |
| Standard-profile simulated record audits | 242,135 primary checks and 30,767 repeat checks passed; no GPU timing or accuracy claim |
| Packaged missing-CUDA check | Failed explicitly, retained partial records and failure report, exported archive; no simulated fallback |
| Fresh-archive CPU review | The exported complete experiment archive was reanalyzed successfully through its `run/primary` dataset |
| Verified backup publication | Existing destination refusal, competing-writer protection, checksum readback, incomplete upload handling and source/link checks passed |
| Pre-profiling evidence recovery | Timing records and their hashes survived deletion of a simulated local VM directory; the checkpoint retained its original incomplete status |
| Packaged workflow with backups | Full smoke primary/repeat/comparison completed with 13 verified checkpoints and a final archive copy |
| Download after Python state loss | The actual main notebook download cell selected and verified a saved result in a fresh namespace with no `result` variable |

The fresh-experiment tests also exercise required-stage failures, retained primary results, optional-layer unavailability, quality qualifications, and cumulative worker budgets. The total GEMM-call cap is optional by default; explicit caps remain tested and timing/memory limits remain enforced.

Required checkpoint tests verify that initial backup failure prevents collector
execution, a failed visit backup stops later visits, and completed primary data
remain available when a subsequent backup fails. Archive recovery refuses an
ambiguous experiment choice and a checksum mismatch. A separate CLI collection
simulation also produced and verified six snapshots through the real command
path, without mocking the CLI.

Backup tests use temporary local directories to represent mounted storage. They
validate software copying, integrity and recovery behavior, not Google Drive
OAuth, remote provider replication, or actual Colab VM termination. The published
hardware run separately records successful checkpoint copying in Colab. New runs
still require authorization and a verified initial copy in their own runtime.
The implementation does not claim it can recover measurements lost before any
backup was created, resume an arbitrary partial checkpoint, or prevent every
runtime failure.

The full primary test run used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0, with BLAS/OpenMP thread counts set to one. The publication rerun took 207.538 seconds. This duration is not a promise for another machine or CI runner. The isolated mathematical checks used PyTorch 2.8.0+cpu without altering the project dependency lock.

## Boundaries exercised

Tests cover training-only model inputs; complete-family CV isolation; preprocessing fitted inside each fold; full-training refitting; held-out perturbation invariance; exclusion of profiler measurements from features; raw event divisors and visit aggregation; rotation coverage; failure retention; evidence/model association; unknown-hardware transfer; and archive extraction/checksum behavior.

The real CPU layer tests compare GQA SDPA with explicit causal attention and check the complete random-weight decoder against its component execution. The optional FP32 reference explicitly uses the SDPA math backend. These checks do not execute CUDA.

## Remaining evidence limits

The measured archive establishes execution and conditional accuracy for its recorded A100 environment. It does not establish:

- Zero-retraining transfer of the current predictors to a different GPU architecture or software environment.
- Independent-day reproducibility of the entire workload; the recorded repeat is a later subset in one session.
- Fixed clocks, identical thermals, absence of concurrent activity, or a causal explanation of timing exceptions.
- Hardware-counter measurements of cache hits, HBM traffic, occupancy, Tensor Core utilization, or stall causes.
- The optional decoder-layer hybrid comparison, full-model inference prediction, or serving-system behavior.
- Calibrated prediction intervals or reliable extrapolation to arbitrary shapes and irregular dimensions.

The current run has real timing-quality qualifications, retained in full. A consistency audit passing does not turn them into passing timing gates. The separate [retrospective baseline analysis](../analysis/review_selection/README.md) continues to describe already-known labels; it is not a replacement for the fresh measured evidence or a new final test for future model changes.

## Reproduce the CPU checks

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python scripts/verify_published_results.py
python scripts/verify_fresh_results.py
python release_tools/build_release.py --out dist --notebook-dir notebooks
python dist/linear_perf_colab.py --self-check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_demo.py --out runs/new_cpu_demo
```

For the complete fresh experiment, start with [START_HERE.md](START_HERE.md). Detailed measurement commands remain in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
