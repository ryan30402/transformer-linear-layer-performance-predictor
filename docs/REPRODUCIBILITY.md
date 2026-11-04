# Reproducing and Extending the Project

There are three distinct tasks: verify the checked-in evidence, generate new GPU observations, and exercise the software with synthetic inputs. They have different hardware requirements and output identities.

| Task | Hardware | Entry point |
|---|---|---|
| Verify the published A100 snapshot | CPU, including a Mac | `python scripts/verify_published_results.py` |
| Run a complete synthetic workflow | CPU | `python scripts/run_demo.py --out runs/synthetic_demo` |
| Collect a new measured session | CUDA-capable NVIDIA GPU | `python -m linear_perf run ...` |
| Audit a new native run | CPU | `python -m linear_perf audit --run ...` |

Run all commands below from the repository root. Keep generated runs and reports separate from the checked-in `results/` snapshot.

## 1. Install the analysis environment

Python 3.11 or newer is required. The recorded A100 sessions used Python 3.12.13. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis-lock.txt
python -m pip install --no-deps -e .
```

The lock file pins NumPy, SciPy, and Matplotlib. It deliberately does not install PyTorch or select a CUDA build. For GPU collection, use a compatible NVIDIA driver and CUDA-enabled PyTorch environment, then record the actual versions with `doctor` and the run's environment artifact. The bundled results use PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, and driver `580.82.07`; a different stack is a different measurement environment.

The console command `linear-perf` and `python -m linear_perf` expose the same CLI. Inspect available commands with:

```bash
python -m linear_perf --help
```

## 2. Verify the published results on a CPU

```bash
python scripts/verify_published_results.py
```

The verifier checks the publication manifest and numerical consistency of the exported evidence. Its detailed scope and the recorded-session layout are documented in [`results/README.md`](../results/README.md). It performs no new GPU measurement and does not replace the recorded model-selection decisions.

Regenerate the README chart directly from the published CSVs with:

```bash
python scripts/plot_results.py
```

The main inspection paths are:

- [`results/summary/selected_pipeline.csv`](../results/summary/selected_pipeline.csv): official development-selected outcomes.
- [`results/summary/model_metrics.csv`](../results/summary/model_metrics.csv): all registered model candidates.
- [`results/summary/quality_and_audits.csv`](../results/summary/quality_and_audits.csv): recorded audit outcomes and timing qualifications.
- [`results/a100/`](../results/a100/): per-session timing, model, freeze, diagnostic, and numerical evidence.
- [`results/MANIFEST.json`](../results/MANIFEST.json): checksums for the publication export.

### Published evidence and source provenance

The checked-in dataset is an English publication export of completed measurement runs. The export preserves measurement values, numerical witnesses, predictions, and scientific decisions. Labels and path metadata are normalized for the standalone release, and the export receives its own file checksums.

Recorded source/archive digests identify the capture artifacts. Publication checksums identify the exported files. Renaming and packaging the code produces a different source identity; it does not rerun the experiment or make the publication source byte-identical to the capture source. Recorded digests remain provenance records rather than being presented as checksums of rewritten bytes.

Reconstructing the original byte-level capture seals requires the original capture archives. The independent checks available from the public export are those implemented by its verifier; they are separate from the 5,468 original-run checks reported in the summary.

Use the publication verifier for this snapshot. Use the native campaign audit for new run directories produced by this release. Keeping these checks separate prevents a presentation transformation from silently replacing a recorded scientific claim.

## 3. Exercise the software without a GPU

```bash
python scripts/run_demo.py --out runs/synthetic_demo --dtype fp16
```

The demo generates a complete synthetic session and audits its software artifacts. The outputs are explicitly marked simulated. Simulation exercises data boundaries, selection, serialization, and audit behavior; it supplies no CUDA performance or numerical-execution evidence.

To run the CPU test suite:

```bash
python -m unittest discover -s tests -v
```

The repository includes a [CPU validation workflow](../.github/workflows/cpu-validation.yml). A successful local test run and a successful GitHub Actions run are separate observations; inspect the actual workflow result after pushing.

## 4. Collect a native GPU run

Run one benchmark process at a time on an otherwise idle GPU. Preflight records the actual device and checks that the selected dtype and precision policy are available:

```bash
python -m linear_perf doctor --config configs/fp16_graph.json
python -m linear_perf run \
  --config configs/fp16_graph.json \
  --out runs/fp16_s01 \
  --session-id fp16_s01
python -m linear_perf audit --run runs/fp16_s01
```

For BF16:

```bash
python -m linear_perf doctor --config configs/bf16_graph.json
python -m linear_perf run \
  --config configs/bf16_graph.json \
  --out runs/bf16_s01 \
  --session-id bf16_s01
python -m linear_perf audit --run runs/bf16_s01
```

BF16 requires native hardware support. The runner does not silently convert a requested BF16 campaign into FP16. Each run requires a fresh output directory; keep failed or interrupted artifacts for inspection instead of overwriting them.

Use `diagnose` in place of `run` to collect only the separate diagnostics. To render a complete audited native run again, write to a new report directory:

```bash
python -m linear_perf report \
  --run runs/fp16_s01 \
  --out reports/fp16_s01
```

Changing the GPU, dtype, precision settings, library version, cache policy, timing contract, or shape plan changes the experiment. Compare those results through an explicitly declared analysis; do not silently combine them into one training set.

## 5. Use Colab

Open [`Run_Benchmark.ipynb`](../notebooks/Run_Benchmark.ipynb) in Colab, choose a GPU runtime, set the dtype/session parameters, and run its cells in order. The notebook includes the packaged source and creates a downloadable native result archive. Inspect the reported GPU and software versions rather than assuming a requested GPU was assigned. [Colab documents that resource availability varies](REFERENCES.md#colab).

Download each completed result archive before releasing the runtime. The companion [`Review_Results.ipynb`](../notebooks/Review_Results.ipynb) handles native run archives produced by this release. It is separate from verification of the checked-in publication snapshot.

To build source/launcher artifacts and regenerate the notebooks after a deliberate code change:

```bash
python release_tools/build_release.py --out dist --notebook-dir notebooks
```

Retain the exact source bundle with new experimental results so that a later audit can identify the implementation used to collect them.

## 6. Compare new sessions

After collecting and auditing three compatible native FP16 runs:

```bash
python -m linear_perf study \
  --runs runs/fp16_s01 runs/fp16_s02 runs/fp16_s03 \
  --out reports/fp16_study
```

The study requires three complete compatible runs. It reports each run's own fit and transfers the first supplied run's frozen models to the other two. Input order therefore determines the source; choose it before inspecting held-out outcomes.

For a compatible same-device precision pair:

```bash
python -m linear_perf compare-dtypes \
  --runs runs/fp16_s01 runs/bf16_s01 \
  --out reports/precision_pair
```

For an explicitly declared transfer analysis:

```bash
python -m linear_perf transfer \
  --source runs/fp16_s01 \
  --target runs/fp16_s02 \
  --mode session \
  --out reports/fp16_transfer
```

`--mode gpu` permits a different hardware configuration under the other compatibility checks. The transfer report includes all declared zero/four/eight-target-calibration-label budgets. A single scale correction cannot necessarily repair shape-dependent hardware changes. The bundled release makes no cross-architecture accuracy claim.

## 7. Preserve the first-release baseline

Treat the committed dataset and its manifest as the fixed result of this experiment. New work should write to new run/report directories and use separate commits. Preserve the official choices and failed predictions in the baseline, even when a later model performs better.

Once published test labels guide a new feature, hyperparameter range, or selector, they are development evidence for that change. Register a new evaluation set before collecting its labels if the goal is to test fresh predictive generalization. Repeating the existing shapes on another session is useful for repeatability, but it does not make their shape-level answers unseen again.
