# Reproducing and Extending the Project

For a first fresh GPU experiment, follow [START_HERE.md](START_HERE.md). `Run_Experiment.ipynb` performs the full primary + repeat + comparison workflow, saves verified checkpoints to the mounted Google Drive, and exports one final archive. `Download_Saved_Results.ipynb` retrieves saved evidence after a lost connection or runtime. The lower-level commands below remain available for separate stages.

The default release workflow collects an offline dataset, then runs grouped cross-validation, refits on all training rows and evaluates held-out rows. Held-out measurements may already exist. There is no freeze-before-measurement requirement. The training-only model interface is the data-isolation boundary.

| Task | Hardware | Entry point |
|---|---|---|
| Verify the completed v1.1.0 A100 experiment | CPU, including a Mac | `python scripts/verify_fresh_results.py` |
| Verify the published first-release A100 snapshot | CPU, including a Mac | `python scripts/verify_published_results.py` |
| Exercise the new pipeline with synthetic labels | CPU | `python scripts/run_demo.py --out runs/synthetic_demo` |
| Run the complete fresh experiment with Drive checkpoints | NVIDIA CUDA GPU in Colab | `Run_Experiment.ipynb` |
| Retrieve a saved final archive or checkpoint | CPU in Colab | `Download_Saved_Results.ipynb` |
| Collect and analyze a new paired-precision dataset | NVIDIA CUDA GPU | `python -m linear_perf run ...` |
| Analyze an existing new-format dataset | CPU | `python -m linear_perf analyze ...` |
| Check raw-to-aggregate consistency | CPU | `python -m linear_perf audit ...` |
| Repeat a deterministic 12.5% subset | NVIDIA CUDA GPU | `python -m linear_perf collect ... --repeat-subset` |
| Validate a random-weight decoder-layer composition | The source CUDA GPU/runtime | `python -m linear_perf layer ...` |

Run commands from the repository root. Keep new runs and reports outside the immutable `results/` snapshot.

## 1. Install the analysis environment

Use Python 3.11 or newer. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis-lock.txt
python -m pip install --no-deps -e .
```

The lock pins NumPy, SciPy and Matplotlib; it does not install or replace PyTorch/CUDA. GPU collection needs a compatible CUDA-enabled PyTorch installation and NVIDIA driver. The editable checkout supplies the pinned model configuration resources; this guide does not assume a standalone wheel bundles those resources.

```bash
python -m linear_perf --help
python -m linear_perf doctor
```

The console command `linear-perf` exposes the same CLI. A Mac/CPU can analyze real exported data and run simulations; it cannot produce NVIDIA CUDA measurements.

## 2. Verify the published experiments

The completed v1.1.0 A100 experiment is retained under
[`experiments/a100_20261106/`](../experiments/a100_20261106/README.md). Its
`results.zip` is the unchanged original export, including raw samples, complete
profiles, fitted models, exact measured source, and the original reports.

```bash
python scripts/verify_fresh_results.py
```

This checks the archive and source provenance, publication copies, raw-to-label
consistency, and saved-model results using the checked-out verification code.
It does not execute source from the archive, refit models, run a GPU measurement,
or convert timing-quality flags into passes. Extract the archive into a new
directory and open `run/report.html` to inspect the original report. The
[review](../experiments/a100_20261106/review.html) and [results](RESULTS.md) explain
the measured accuracy and retained limitations.

The source embedded in the measured result remains the authority for that
experiment. Rebuilding the current notebooks after documentation and publication
updates can change their source-archive hashes without changing the measured
predictor. The original archive and its recorded source hash are retained for
that reason.

### Historical baseline and retrospective analysis

```bash
python scripts/verify_published_results.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/review_selection.py \
  --compact --output runs/review_analysis
```

The first command checks the committed publication export using its original contract. The second leaves those bytes unchanged and separately reports (a) selection over the five unchanged stored candidates and (b) new grouped-CV/refit modeling of the old 36 non-test labels. Old test answers were already known during this revision. Those errors are retrospective, not a new independent evaluation. See [the reviewed tables](../analysis/review_selection/README.md).

The original recorded models, failures, freeze files and acquisition appendix remain historical evidence. The [baseline protocol](BASELINE_PROTOCOL.md) explains that original experiment. Its chronological freeze rule is not a requirement of the new workflow. The checked-in public dataset is an English publication export; recorded capture digests remain provenance records, not claims that transformed publication bytes equal their original capture archives.

## 3. Exercise the new pipeline without a GPU

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_demo.py --out runs/synthetic_demo
python -m linear_perf audit --run runs/synthetic_demo
```

Open `runs/synthetic_demo/analysis/report.html`. Every timing artifact is labeled simulated; this checks data flow, grouping, fitting, raw aggregation and reporting. It establishes no CUDA performance, CUDA numerical correctness or hardware accuracy.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The [CPU validation workflow](../.github/workflows/cpu-validation.yml) also verifies the historical and current published datasets and builds the Colab payload. A local pass does not establish that a future GitHub Actions run has passed. GPU capture/timing requires a separate actual GPU execution. Optional CPU PyTorch tests check the decoder's arithmetic when PyTorch is installed; they do not execute CUDA.

## 4. Collect one paired-precision dataset

The standard manifest contains 609 distinct `(M,N,K)` shapes per precision. The 32-shape smoke manifest is for plumbing, not representative performance evidence. Plans retain pinned model sources, exclusions and declared workload weights.

```bash
python -m linear_perf run \
  --config configs/workloads.json \
  --out runs/paired_s01 \
  --session-id paired_s01
python -m linear_perf audit --run runs/paired_s01
```

The seeded schedule interleaves FP16/BF16 within a single campaign. Each precision/cache context has its own fitted model. The collector completes every split before fitting begins. A BF16 request fails clearly on unsupported hardware; there is no silent CPU or FP16 fallback. Each attempt needs a new output path.

Warm-reuse timing covers every planned shape. Four-buffer rotation is an additional context when the weight is at most twice the observed L2 size and the memory guard permits it. Actual L2 discovery takes priority. If discovery is unavailable, `--l2-cache-mib` supplies an explicit documented fallback; absent both, the rotation comparison is marked unavailable. Rotation does not prove cache eviction or HBM traffic.

Adaptive graph counts target an approximately 2 ms replay, subject to bounds; the exact captured operation count is retained for each visit and divides its event interval. Repeated visits, numerical checks, controls and telemetry remain in the dataset. Profiles run separately after timing and never become prediction labels/features. `--skip-profiles` is available for an explicitly different diagnostic scope. The collection budget is configurable with `--max-wall-seconds`; expiry preserves partial data and fails the run, rather than substituting simulated data.

To separate collection and analysis:

```bash
python -m linear_perf collect --config configs/workloads.json \
  --out runs/data_only --session-id data_only
python -m linear_perf analyze --dataset runs/data_only --out runs/data_only/analysis
```

This is also how previously collected test data are used normally: fit on train records, evaluate the existing test records afterwards. No timestamp-based model-freeze gate is applied.

### Output layout

| Path under a run | Purpose |
|---|---|
| `status.json`, `workload_manifest.json` | Overall status and exact workload plan |
| `dataset/raw_visits.jsonl` | Event intervals, capture divisors, numerical checks and visit metadata |
| `dataset/measurements.jsonl` | Per-shape targets derived from visit medians |
| `dataset/environments.json`, `schedule.json` | Actual hardware/software/cache contexts and measurement order |
| `dataset/controls.jsonl`, `quality.json`, `telemetry.jsonl` | Drift/stability context and retained quality qualifications |
| `dataset/profiles.jsonl` | Separate kernel diagnostics, including unavailable/error records |
| `analysis/report.html`, `metrics.csv`, `predictions.csv` | Human report and all-candidate error tables |
| `analysis/group_*/model_suite.json` | CV trials, chosen configuration and all-training refit |
| `analysis/group_*/test.json`, `challenge.json`, `stress.json` | Separate held-out scores, subgroup metrics and domain warnings |

Passing record-consistency checks does not imply timing stability. Inspect `measurement_quality` and controls in the analysis metadata. All observations are retained; diagnostic quality flags do not silently remove slow cases.

## 5. Use Colab for the complete experiment

1. Upload [Run_Experiment.ipynb](../notebooks/Run_Experiment.ipynb) to Colab.
2. Select **Runtime > Change runtime type > GPU**. The collector checks actual hardware and FP16/BF16 support.
3. Keep `PROFILE = "standard"`, `SIMULATE = False`, `INCLUDE_REPEAT = True`, `INCLUDE_LAYER = False`, and `SKIP_PROFILES = False`. The notebook generates a unique `EXPERIMENT_ID`.
4. Run the cells in order and authorize the Google Drive mount when prompted. Check the printed backup directory and the initial verified checkpoint before GPU collection begins.
5. The experiment collects and models the new primary dataset, collects the deterministic repeat subset, compares the source model without refitting, and writes a combined report. Checkpoint copies preserve evidence at the declared boundaries.
6. Download the single final results ZIP and check `Final backup status`. A `verified` final backup is also available in Drive. If the final copy fails, download the retained local archive immediately. Open `run/report.html` after extraction, then inspect stage reports and timing-quality qualifications.

The default 7,200-second cap applies separately to primary and repeat collection. The combined worker ceiling accounts for both plus analysis allowance; setup and export add time. These are maximum budgets, not predicted duration. A full run can exceed two hours. The code does not automatically rent a GPU, retry failed measurements, or substitute synthetic data.

The required default stages produce `run/primary/`, `run/repeat/`, and `run/repeat_comparison/`. Optional layer validation can be requested before running by setting `INCLUDE_LAYER = True`. Its results are a hybrid estimate and are never predictor-training labels. It is not required when disabled.

The notebook embeds the source and pinned architecture configurations. It does not read the old published A100 data. `Run_Benchmark.ipynb` is a compatibility copy of this new complete notebook. The separate lower-level `run` and `collect` commands still support single-session collection and explicitly declared cross-GPU studies.

### Persistent checkpoints

The notebook passes `backup_directory=BACKUP_DIRECTORY` to `launcher.run_experiment`. By default, that parent directory is:

```text
/content/drive/MyDrive/Transformer_Linear_Layer_Performance_Predictor/experiments
```

The launcher creates a new `linear-perf_<EXPERIMENT_ID>` folder under it. Existing experiment folders are preserved, and an attempted reuse is rejected. Mount Drive before using this path; creating a similarly named local directory alone does not make storage persistent.

The initial checkpoint is required before measurement begins. Later snapshots are taken after full timing visits, before profiling, after successful collection and at major experiment-stage boundaries. Timing and trace files remain local during measurement; snapshots are packed locally and copied with readback verification. Transfers happen outside CUDA event intervals. Their pauses can change wall time and thermal state, so the recorded pauses, controls and timing-quality checks remain part of interpretation. Pauses during collection count toward that collection's wall-time budget.

For the generated launcher CLI, `--backup-directory` names the **parent folder**; the launcher adds the session folder. For a direct `python -m linear_perf experiment`, `run` or `collect` command, `--checkpoint-directory` names the **exact destination for that attempt**. These options are optional for non-notebook use; an ordinary run directory on temporary storage does not survive deletion of that storage. The launcher also passes its source hash through `--checkpoint-source-sha256` for checkpoint provenance.

A verified checkpoint can still contain incomplete measurements or unfinished analysis. It preserves those statuses and cannot automatically satisfy the completed-dataset audit. Required checkpoint failures stop the affected path and retain local evidence; final archive-copy status is reported separately from experiment completion.

### Retrieve files after losing Python state or the runtime

If a browser download fails while the original runtime is available, rerun only the download cell. It can locate a saved archive without the in-memory `result` variable after Drive is mounted. Running the experiment cell again starts a new attempt.

For a replaced runtime, open [Download_Saved_Results.ipynb](../notebooks/Download_Saved_Results.ipynb) on CPU and mount the same Google account. Run the listing cell, then set `RECOVERY_EXPERIMENT_ID` to the exact listed ID if multiple experiments exist. It may remain blank when only one experiment is available. Leave `RECOVERY_ARCHIVE_NAME` blank to prefer that experiment's final results ZIP, falling back to its latest verified checkpoint; enter an exact listed filename to retrieve a particular earlier archive.

The download verifies the selected archive and prints its kind and recorded status. A checksum failure is reported rather than silently selecting another artifact. This is file retrieval, not automatic resumption of measurement or permission to combine observations from different GPUs. See [COLAB_RECOVERY.md](COLAB_RECOVERY.md).

[Review_Results.ipynb](../notebooks/Review_Results.ipynb) is an optional CPU-only tool. In `analyze` mode, it resolves a complete experiment archive to `run/primary/` and reanalyzes that dataset. The combined repeat report is already inside the original archive. A failed later stage does not prevent CPU analysis of a completed primary dataset; the reviewer preserves the original failure status and validates the primary raw records. Advanced two-archive `repeat`/`gpu` modes remain available.

Preserve failed and incomplete archives; a partial experiment may contain a valid primary evaluation even when repeat collection or comparison failed. An abrupt runtime loss before a checkpoint finishes can still lose the latest unbacked work. If no copy exists and the original runtime is gone, saved notebook output cannot reconstruct the raw measurements. Do not change an incomplete status to `completed` to force normal analysis.

To rebuild the source and notebooks after an intentional code change:

```bash
python release_tools/build_release.py --out dist --notebook-dir notebooks
python dist/linear_perf_colab.py --self-check
```

## 6. Repeat a small subset and compare another GPU

Use the source's exact workload manifest to retain identical measurement settings:

```bash
python -m linear_perf collect --config runs/paired_s01/workload_manifest.json \
  --out runs/paired_s01_repeat --session-id paired_s01_repeat --repeat-subset
python -m linear_perf compare --source runs/paired_s01 --target runs/paired_s01_repeat \
  --mode repeat --out reports/paired_repeat
```

The deterministic stratified subset uses shapes/splits, never latency or prediction errors. It is 77/609 shapes for the standard plan and 4/32 for smoke. It does not refit a target model. The same physical device is verified only when UUID evidence is available; hardware/software/math-policy changes cannot silently be called a plain repeat.

For another GPU, collect the source manifest there with `collect`, then use:

```bash
python -m linear_perf compare --source runs/paired_s01 --target runs/other_gpu \
  --mode gpu --out reports/gpu_transfer
```

The source model and its chosen configuration remain unchanged. Target SM geometry is supplied where available; unknown target hardware is explicitly flagged. Results report zero-retraining transfer, not target calibration or universal cross-GPU accuracy. Both datasets retain their quality qualifications.

## 7. Optional decoder-layer check

In the same GPU/software runtime as the source run:

```bash
python -m linear_perf layer --source runs/paired_s01 --out runs/layer_check \
  --model-id qwen2_5_0_5b --dtype float16 --batch-size 1 --sequence-length 128
```

The randomly initialized prefill decoder includes two RMSNorms, Q/K/V/O projections, rotary embeddings, causal grouped-query attention, SwiGLU and residual additions. The predictor supplies seven linear GEMM estimates; all other component costs are measured independently. Their hybrid sum is compared against actual full-layer CUDA-event timing. A separate component-additivity gap helps distinguish GEMM prediction error from invalid cost summation.

This is not a fully predicted end-to-end model: attention/non-GEMM times are measured inputs. It excludes token embeddings, KV-cache decoding, distributed work and serving. Component shapes can overlap source training/test shapes. Layer results never enter predictor training. A preflight memory skip keeps original model dimensions and records the reason.

## 8. Interpretation boundaries

The [completed A100 experiment](RESULTS.md) supplies measured evidence for the
current workflow in one hardware/software context. Its ordinary-test weighted
mean APE is 7.71–10.32%; warm-reuse challenge and irregular stress results are
substantially weaker. Timing-quality flags remain attached to the data. A lower
error on the historical answers is retrospective. Repeated measurements of
known shapes test repeatability, not new-shape generalization. CV scores select
a model and should not be quoted as its independent test accuracy. Report mean,
median, P90 and worst errors, frequency-weighted and unweighted, with workload
and timing conditions. The current held-out results have now informed failure
analysis; future model revisions need new final evaluation cases for fresh
generalization claims.
