# Run a Fresh Experiment — v1.1.0

This is the complete fresh GPU experiment kit for release **1.1.0**. Use the notebook from this kit for the experiment below.

The experiment collects new GPU measurements, fits new predictors, evaluates the new held-out cases, and checks a short repeat subset. It needs no previous result ZIP, old timing CSV, or previously trained predictor. Historical results included elsewhere in the project are not inputs to this run.

The repository includes a [completed A100 experiment](RESULTS.md) with its [original results archive and review](../experiments/a100_20261106/README.md), alongside the implementation and CPU verification records. To inspect that evidence, download and extract `experiments/a100_20261106/results.zip` and open `run/report.html`; no GPU is needed. Running the notebook below creates a separate experiment with a new ID and new measurements. CPU checks and simulations do not establish those future hardware results.

The notebook saves verified checkpoint copies to Google Drive so that completed snapshots can be retrieved after a Colab runtime is lost.

## 1. Open the experiment in Colab

1. Extract the project ZIP on your computer.
2. Upload [`Run_Experiment.ipynb`](../notebooks/Run_Experiment.ipynb) to Google Colab. The notebook embeds its source code and model configuration files; uploading the rest of the ZIP is unnecessary.
3. Select a **GPU** runtime in Colab's runtime settings. An available A100 is a suitable choice. The standard experiment requires a CUDA-capable NVIDIA GPU with native support for both FP16 and BF16. The program records the assigned GPU and software versions.
4. In the notebook's settings cell, keep the automatically generated `EXPERIMENT_ID` and the standard measured-experiment defaults: `INCLUDE_REPEAT = True`, `INCLUDE_LAYER = False`, `SKIP_PROFILES = False`, and `SIMULATE = False`.
5. Run the notebook's cells in order, or use **Run all** once. When the Drive mount cell requests authorization, choose the Google account in which you want to keep the experiment backups and approve the mount.
6. Check the displayed backup directory and experiment ID. The initial checkpoint must be copied and verified before GPU collection begins. If backup setup fails, resolve that error before starting another attempt.

A Mac/CPU is sufficient to upload the notebook and later review the report. GPU measurements run in the Colab runtime. The notebook retains that runtime's existing PyTorch/CUDA installation and prepares the separate analysis dependencies.

The default backup directory is:

```text
/content/drive/MyDrive/Transformer_Linear_Layer_Performance_Predictor/experiments
```

Find the corresponding folder in **Google Drive > My Drive > Transformer_Linear_Layer_Performance_Predictor > experiments**. Each experiment has its own folder. Keep the experiment ID printed by the notebook so you can select the correct folder later.

**Saving the notebook to Drive does not, by itself, save the experiment files.** The notebook's code and displayed output are separate from the files created in Colab's runtime. This experiment explicitly copies checkpoints and the final archive to the mounted Drive. Colab runtimes can be deleted after inactivity or when their maximum lifetime is reached; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

The assigned device, memory availability, profiler support and configured budgets determine what can complete. Time limits are caps, not predictions of duration or cost. A GPU without the required precision support produces a clear error; it does not silently generate another precision or simulated measurements.

By default, `MAX_WALL_SECONDS = 7200` applies separately to primary collection and repeat collection. The combined experiment can therefore exceed two hours; setup, analysis and export also take time. The notebook displays the collection budgets and the separate worker limit before execution.

## 2. What runs automatically

| Stage | Work performed | Result |
|---|---|---|
| Full dataset | Measure the standard 609 shapes for FP16 and BF16, with interleaved precision order | New raw samples and shape-level timing targets |
| Model selection | Group training cases by `(N,K)`, run cross-validation, select configurations and a model family, then refit on all training rows | Fitted predictors and recorded selection evidence |
| Held-out evaluation | Evaluate test, challenge and irregular-stress cases separately | Predictions, average and tail errors, and subgroup reports |
| Dataset audit | Recompute raw-event divisors, visit aggregation and record consistency on CPU | Auditable links from timing records to model targets |
| Checkpoint copies | Save snapshots at full timing-visit boundaries, before profiling, after successful collection, and after major experiment stages | Verified archives in the selected Google Drive folder |
| Short repeat | Measure a deterministic 77-of-609 subset on the same GPU/runtime | Additional observations for repeatability, without target refitting |
| Repeat comparison | Apply the source models unchanged and compare paired timings | Repeatability and source-model transfer reports |
| Combined report and export | Summarize stage completion, selected-model errors and quality qualifications | One complete results ZIP with an entry report |

Each precision has its own models. Warm buffer reuse and eligible four-buffer rotation also remain separate contexts. Rotation is collected only where the declared L2 and memory conditions permit it; its coverage and any unavailable/skipped variants are recorded. Rotation does not guarantee cold-cache execution.

The [primary workload manifest](../configs/workloads.json) has these fixed splits per precision:

| Split | Shapes | Purpose |
|---|---:|---|
| Train | 311 | Cross-validation, model selection and final training |
| Test | 134 | Unseen M values within training `(N,K)` families |
| Challenge | 118 | Entire `(N,K)` families absent from training |
| Stress | 46 | Separate irregular-dimension evaluation |

All primary splits are measured before fitting begins. **There is no freeze-before-measurement requirement.** Test, challenge and stress labels are excluded from model selection. The existing published baseline shapes are excluded from new held-out cases.

Measurements and trace files are written locally during execution. Checkpoint packing and copying occur at declared boundaries outside CUDA event timing intervals. The copies can add wall time and pauses between visits; those pauses are recorded, and warmup, control timings and telemetry remain relevant when assessing timing quality. A completed checkpoint is a snapshot of the work finished at that boundary, not a claim that later stages completed.

Profiling still runs separately after timing collection, with visible capture progress. A `609/609` timing progress line means the scheduled primary timing pass has finished; it does not establish that profiling, analysis, the repeat or final export has finished.

## 3. Download the complete results

When execution ends, use the notebook's download cell to save the **single complete results ZIP**. The notebook also attempts to copy and verify the final archive in the experiment's Drive folder; check the separately returned backup outcome. Keep the ZIP intact, including any failure records. Completion of the final archive copy and completion of the experiment are recorded separately: a successfully saved archive can contain a failed or incomplete experiment.

If the connection or Python state is lost, use [`Download_Saved_Results.ipynb`](../notebooks/Download_Saved_Results.ipynb) to mount the same Drive account, list saved experiments and select the exact archive to download. It needs no GPU and does not require the previous `result` variable. If no final archive exists, download the latest verified checkpoint for inspection. See [COLAB_RECOVERY.md](COLAB_RECOVERY.md) for the recovery procedure and the limits of partial snapshots.

After extraction, open **`run/report.html`** first. It links to the primary model report, the repeat comparison and the underlying evidence.

| Path inside the results ZIP | What it contains |
|---|---|
| `launcher_status.json`, `worker.log` | Notebook launcher outcome and process log |
| `run/report.html` | Combined experiment report |
| `run/status.json`, `run/summary.json` | Stage completion, experiment scope and qualifications |
| `run/workload_manifest.json` | Exact workload, splits and measurement settings |
| `run/selected_model_metrics.csv` | Concise held-out metrics for the selected models |
| `run/primary/dataset/` | New raw visits, aggregates, controls, telemetry, profiles and actual environments |
| `run/primary/analysis/` | Cross-validation records, fitted models, all-candidate metrics and per-case predictions |
| `run/primary/audit.json`, `run/repeat/audit.json` | CPU consistency audits of the retained measurement records |
| `run/repeat/dataset/` | New timing records for the fixed repeat subset |
| `run/repeat_comparison/` | Paired timing changes and source-model evaluations without retraining |

The notebook's final ZIP is the file to share for review. If execution stopped early, share the saved checkpoint and identify it as a checkpoint. A screenshot of an error table alone is insufficient to inspect timing quality, context or the data-to-model chain.

## 4. Interpret the outcome

**Completion and measurement quality are different questions.** A completed experiment can still have timing instability or limited rotation/profile coverage. The report keeps those qualifications visible; slow observations are not silently removed to improve prediction scores.

| Outcome | Interpretation and next step |
|---|---|
| `completed`, with measured CUDA data | Required stages finished. Inspect timing quality and the held-out error distributions before drawing conclusions. |
| Completion with quality qualifications | Keep and inspect the flags, control timings and raw samples. A successful consistency audit alone does not establish stable hardware timing. |
| `incomplete` | A requested optional layer stage did not finish. Inspect `required_stages_completed` and `all_requested_stages_completed`; the primary and repeat results can still be complete and remain available. |
| `failed` or `interrupted` | Download the retained archive. Inspect the failed stage and logs. A completed primary stage remains useful even if a later stage failed; the entire experiment must not be described as complete. |
| `simulated_complete` | Software validation only. It is not a GPU latency or prediction-accuracy result. |

For model quality, examine weighted and unweighted mean APE, median APE, P90 and worst-case errors. Keep FP16/BF16, cache policies and evaluation splits separate. Cross-validation scores were used for selection; final held-out scores answer a different question.

The repeat subset measures stability of already selected shapes. It does not turn their answers into new unseen cases. Replays within one session are also not independent GPU sessions. One completed experiment supports conclusions for its recorded GPU and workload, not all architectures or a production serving system.

The [published A100 run](RESULTS.md) is a concrete completed example with quality qualifications: 6 of 2,256 primary aggregate rows and one repeat control group exceed their thresholds. Its ordinary-test weighted mean errors are 7.71%–10.32%, while warm challenge and irregular stress are less accurate. Those outcomes remain visible alongside successful completion; the original archive is preserved without deleting flagged observations.

## 5. If the run does not finish

Keep the failed archive and read the error before retrying. An unsupported device, memory limit, time limit or execution error needs the corresponding fix. If a new attempt is needed, choose a new experiment ID; existing output is preserved rather than overwritten. There are no automatic retries that replace the evidence from a failed attempt.

For a missing `result` variable or a replaced runtime, open the dedicated recovery notebook and retrieve the saved archive from Drive. Rerunning the settings and main experiment cells starts a new measurement attempt; it does not resume the previous one.

Checkpoints preserve evidence, but they do not guarantee that an interrupted run can pass the normal completed-dataset audit. A checkpoint taken before profiling or before all visits finish may lack required records. Keep its recorded status intact. Do not change it to `completed`, fill missing values, or combine visits from a different GPU to make analysis proceed.

An abrupt loss can occur before the first measurement checkpoint finishes, and a Drive or network failure can prevent a copy from completing. Only already completed, verified copies are recovery candidates. If the old runtime is gone and no result archive or checkpoint was saved elsewhere, its raw measurements cannot be reconstructed from notebook output.

## 6. Optional follow-up

The decoder-layer check is optional and off by default. It compares predicted linear GEMM costs plus independently measured other operations with an actual random-weight prefill decoder layer. This is a **hybrid estimate**; it does not establish fully predicted end-to-end LLM or serving latency. Its measurements never enter predictor training.

The standard primary and repeat experiment is complete without that optional check. Testing another GPU is a separate follow-up, after the current device's result is understood.

For measurement definitions and model details, see [PROTOCOL.md](PROTOCOL.md), [MODELS.md](MODELS.md), and [VALIDATION.md](VALIDATION.md).
