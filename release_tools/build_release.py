"""Build a deterministic source archive and self-contained English Colab notebooks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

VERSION = "1.1.0"
NAME = "transformer-linear-layer-performance-predictor"
TITLE = "Transformer Linear Layer Performance Predictor"
ARCHIVE_ROOT = f"{NAME}-{VERSION}"


def source_paths(root: Path):
    """Keep generated notebooks and recorded data out of the executable payload."""
    directories = {"linear_perf", "configs", "docs", "release_tools", "scripts", "tests"}
    top_files = {"README.md", "pyproject.toml", "requirements-analysis-lock.txt"}
    excluded = {"__pycache__", ".git", ".pytest_cache", ".venv", "node_modules"}
    paths = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink() or any(p in excluded for p in relative.parts):
            continue
        if path.suffix == ".pyc":
            continue
        if relative.parts[0] in directories or relative.as_posix() in top_files:
            paths.append(path)
    return paths


def _zip(paths, destination, root):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            entry = zipfile.ZipInfo(ARCHIVE_ROOT + "/" + path.relative_to(root).as_posix(),
                                    date_time=(2026, 11, 4, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())


def _cell(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        compile(source, "<notebook_cell>", "exec")
        cell.update(execution_count=None, outputs=[])
    return cell


def _notebook(path, cells, *, gpu=False):
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
                "colab": {"name": path.name, "provenance": []}}
    if gpu:
        metadata["accelerator"] = "GPU"
    for index, cell in enumerate(cells):
        cell["id"] = f"cell_{index:02d}"
    path.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": metadata,
                                "cells": cells}, indent=1) + "\n", encoding="utf-8")


def build(root, output, notebook_dir=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook_dir = Path(notebook_dir).resolve() if notebook_dir else output
    notebook_dir.mkdir(parents=True, exist_ok=True)
    paths = source_paths(root)
    source_zip = output / f"{ARCHIVE_ROOT}.zip"
    _zip(paths, source_zip, root)
    raw = source_zip.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    template = (root / "release_tools/colab_launcher_template.py").read_text(encoding="utf-8")
    launcher = template.replace("__SOURCE_SHA256__", source_sha).replace(
        "__SOURCE_B64__", base64.b64encode(raw).decode("ascii"))
    compile(launcher, "linear_perf_colab.py", "exec")
    (output / "linear_perf_colab.py").write_text(launcher, encoding="utf-8")
    encoded = base64.b64encode(launcher.encode()).decode("ascii")
    launcher_sha = hashlib.sha256(launcher.encode()).hexdigest()
    bootstrap = (
        "# Verify and load the embedded release; this cell performs no GPU measurement.\n"
        "import base64, hashlib, types\n"
        f"_LAUNCHER_B64 = {encoded!r}\n"
        "_launcher_source = base64.b64decode(_LAUNCHER_B64).decode('utf-8')\n"
        f"assert hashlib.sha256(_launcher_source.encode()).hexdigest() == {launcher_sha!r}, 'Launcher integrity mismatch'\n"
        "launcher = types.ModuleType('linear_perf_colab')\n"
        "exec(compile(_launcher_source, 'linear_perf_colab.py', 'exec'), launcher.__dict__)\n"
        "print(launcher.self_check())\n"
    )
    recovery_source = (root / "release_tools/colab_recovery.py").read_text(encoding="utf-8")
    mount_backup = """from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
_my_drive = Path("/content/drive/MyDrive")
if not _my_drive.is_dir():
    raise RuntimeError("Google Drive is not mounted. Complete authorization before measuring.")
BACKUP_DIRECTORY = _my_drive / "Transformer_Linear_Layer_Performance_Predictor" / "experiments"
BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)
print("Backup directory:", BACKUP_DIRECTORY)
print("The experiment verifies its initial backup before starting GPU measurement.")
"""
    download_code = recovery_source + """

from google.colab import files

# Leave these empty for the current result, or for the only saved experiment.
# If multiple experiments are saved, enter one exact ID from the printed list.
RECOVERY_EXPERIMENT_ID = ""
RECOVERY_ARCHIVE_NAME = ""
_backup_root = globals().get("BACKUP_DIRECTORY", DEFAULT_BACKUP_DIRECTORY)
_current_result = globals().get("result")
if not isinstance(_current_result, dict):
    print("Saved experiments:")
    for item in list_saved_experiments(_backup_root):
        print(item["experiment_id"], "|", item["archive_count"], "archive(s)")

saved_result = locate_saved_result(
    _backup_root, experiment_id=RECOVERY_EXPERIMENT_ID,
    archive_name=RECOVERY_ARCHIVE_NAME, local_result=_current_result,
)
print("Archive:", saved_result["path"])
print("Kind:", saved_result["kind"], "| Recorded status:", saved_result["status"])
print(saved_result["note"])
print("Archive SHA256:", saved_result["sha256"])
files.download(saved_result["path"])
"""
    main = [
        _cell("markdown", f"""# {TITLE}

## Fresh GPU experiment — v{VERSION}

Select **Runtime > Change runtime type > GPU**, then **Runtime > Run all**.
Authorize Google Drive when prompted. Verified evidence checkpoints are saved
to your Drive before and during the experiment; active measurement files stay
on the Colab VM. Saving the notebook alone does not save those measurements.
This notebook contains the complete source. No previous ZIP, timing CSV,
trained predictor, model weights, or repository checkout is needed.

The default experiment performs:

1. Fresh FP16/BF16 collection on the **609-shape** model-derived workload.
2. Training-only grouped cross-validation, configuration/model selection,
   full-training refit, and separate test/challenge/stress evaluation.
3. A deterministic **77-shape repeat** on the same GPU, without refitting.
4. Repeat comparison and one combined report with raw-record and timing-quality checks.
5. One complete result ZIP, also copied and verified in Google Drive. Earlier
   checkpoints retain evidence if the runtime disappears before the final export.

Every split is measured before fitting. Test labels may exist on disk, but
never enter fitting or model selection. There is no freeze-before-measurement step.

The primary target is preallocated `Y = X @ W.T` under CUDA Graph replay.
Warm reuse and eligible four-buffer rotation are separate contexts. GPU memory
reads/writes are inside the interval; host transfers, graph capture and serving
queues are outside it. Rotation does not guarantee cold-cache execution.
"""),
        _cell("code", """from datetime import datetime, timezone
import uuid

EXPERIMENT_ID = "fresh_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
PROFILE = "standard"                 # 609 shapes/dtype; "smoke" is a 32-shape software/hardware check.
DTYPES = ("float16", "bfloat16")       # Both interleaved; each precision/cache context is modeled separately.
SIMULATE = False                      # Keep False for actual GPU data.
INCLUDE_REPEAT = True                 # Required by the default complete experiment; 77/609 shapes.
INCLUDE_LAYER = False                 # Optional random-weight decoder-layer hybrid estimate.
SKIP_PROFILES = False                 # Separate diagnostic profiling after timing.
MAX_WALL_SECONDS = 7200               # Maximum per collection stage, not a predicted duration.
PIN_ANALYSIS_DEPS = True

_collection_stages = 2 if INCLUDE_REPEAT else 1
_worker_limit = MAX_WALL_SECONDS * _collection_stages + 900 + (900 if INCLUDE_LAYER else 0)
print("Experiment:", EXPERIMENT_ID)
print("Collection stages:", _collection_stages, "| Per-stage cap:", MAX_WALL_SECONDS, "seconds")
print("Combined worker cap:", _worker_limit, "seconds; setup and download are additional.")
print("These are upper limits, not runtime or cost predictions.")
"""),
        _cell("markdown", """## Mount backup storage

Authorize your Google Drive connection. Backups are stored under
`MyDrive/Transformer_Linear_Layer_Performance_Predictor/experiments/`, in a
new folder for this experiment. Existing experiment folders are never reused.

The first verified checkpoint is required before any GPU measurements. Later
checkpoints run synchronously between complete timing visits, before profiling,
and after major stages. They are outside CUDA event timing intervals. Their
pauses are recorded and count toward the collection wall-time budget; they can
affect thermal state, so controls and timing-quality checks remain necessary.

A backup is confirmed only after file-copy readback verification. Drive errors
stop the required checkpoint path; no automatic measurement retry occurs.
"""),
        _cell("code", mount_backup),
        _cell("markdown", """## Load the included experiment

This verifies the embedded source and compiles Python files without measuring
GPU work. The payload contains no historical measured result files. Its analysis
stack may be installed; the runtime's existing PyTorch/CUDA is retained.
"""),
        _cell("code", bootstrap),
        _cell("markdown", """## Run the complete experiment

Progress reports primary collection/analysis, repeat collection, and comparison.
The 7,200-second cap applies separately to primary and repeat collection, so the
complete experiment can exceed two hours. Analysis and export add time.

Failures are retained and exported when the process can finish. A verified
checkpoint before profiling protects completed timing records from a later
runtime loss. Profiling reports progress separately; 609/609 timing shapes does
not yet mean that model evaluation and the short repeat have completed.

An abrupt loss before a checkpoint finishes can still lose the latest work.
Checkpoints are preserved evidence, not automatic resume points or proof of a
complete experiment. Never silently append measurements from a replacement GPU.
"""),
        _cell("code", """result = launcher.run_experiment(
    experiment_id=EXPERIMENT_ID, profile=PROFILE, dtypes=DTYPES,
    simulate=SIMULATE, include_repeat=INCLUDE_REPEAT, include_layer=INCLUDE_LAYER,
    skip_profiles=SKIP_PROFILES, max_wall_seconds=MAX_WALL_SECONDS,
    pin_analysis_deps=PIN_ANALYSIS_DEPS, backup_directory=BACKUP_DIRECTORY,
)
print("Launcher status:", result["status"])
print("Experiment status:", result.get("experiment_status", "Inspect launcher_status.json"))
print("Result archive:", result["archive"])
print("Final backup status:", result.get("backup_status", "unknown"))
if result.get("backup_archive"):
    print("Saved backup:", result["backup_archive"])
if result.get("backup_status") == "failed":
    print("Download the local archive immediately. Final Drive copy failed:", result.get("backup_error"))
if result["status"] != "worker_completed":
    print("Download the archive below. Completed primary results may still be available; inspect stage statuses.")
"""),
        _cell("markdown", """## Download the single complete result

After extraction, open **`run/report.html`** first. It links to the primary
prediction report, repeat comparison, selected-model metrics and quality records.
Raw data are under `run/primary/dataset/` and `run/repeat/dataset/`.

If the download does not start, rerun only the download cell. It can also find
saved archives when the in-memory `result` variable is unavailable: mount Drive
first, then run this cell. If multiple experiments exist, select the exact ID
printed below. A final archive is preferred; otherwise a clearly labeled
checkpoint is offered. Re-running the settings and experiment cells starts a
new measurement attempt.

After losing the entire runtime, the separate **Download_Saved_Results.ipynb**
can retrieve saved evidence on CPU without starting another experiment.
"""),
        _cell("code", download_code),
        _cell("markdown", """## How to interpret this run

- A completed run can still have timing-quality qualifications. Inspect controls,
  drift and unstable rows alongside mean, median, P90 and worst-case errors.
- CV scores select a model; the held-out scores evaluate that selection separately.
- The repeat subset checks stability on known shapes. It does not create a second
  independent new-shape test set and it never refits the source model.
- Compare model families on the same new test data and measurement context.
  An aggregate over this workload is not directly comparable to a different old workload.
- If `INCLUDE_LAYER = True`, the optional layer stage uses predicted linear GEMM
  costs plus independently measured other-operation costs. It is a hybrid estimate,
  not a fully predicted LLM/serving benchmark.
- Another GPU is an additional experiment; this single-GPU run does not establish
  cross-architecture accuracy. No GPU is rented automatically.

The source and result metadata retain release **1.1.0** plus exact source hashes.
CPU simulation is explicitly labeled and never presented as GPU evidence.
"""),
    ]
    recovery = [
        _cell("markdown", f"""# {TITLE}

## Download saved results — v{VERSION}

Use a **CPU runtime**. This notebook mounts your Drive, lists saved experiment
archives, verifies the selected archive, and downloads it. It does not start or
resume GPU measurement, train a model, or execute code from an uploaded archive.

It can retrieve only files previously saved in your Drive. It cannot reconstruct
measurements that disappeared with an earlier VM before any backup was created.
"""),
        _cell("code", """from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
BACKUP_DIRECTORY = Path("/content/drive/MyDrive/Transformer_Linear_Layer_Performance_Predictor/experiments")
print("Reading saved evidence from:", BACKUP_DIRECTORY)
"""),
        _cell("markdown", """## List saved experiments

Copy the exact experiment ID into the next cell if there is more than one.
The list includes only ZIP files with completed checksum receipts; partial
uploads are not offered as verified backups.
"""),
        _cell("code", recovery_source + """

saved_experiments = list_saved_experiments(BACKUP_DIRECTORY)
if not saved_experiments:
    print("No saved experiments found in this directory.")
for item in saved_experiments:
    print("Experiment ID:", item["experiment_id"])
    for name in item["archives"]:
        print("  ", name)
"""),
        _cell("markdown", """## Select and download

Leave the experiment ID empty only if one saved experiment exists. Leave the
archive name empty to prefer that experiment's final result; otherwise the
latest checkpoint is selected. Enter an exact listed archive name to choose an
earlier checkpoint. A checksum failure is reported instead of silently choosing
another file.

Read the printed **kind** and **recorded status**. A checkpoint can contain
partial timing records or an unfinished later stage; it is not automatically
eligible for the normal completed-dataset analysis. Preserve the ZIP unchanged
for review. No files are rewritten and no measurement is resumed.
"""),
        _cell("code", """from google.colab import files

RECOVERY_EXPERIMENT_ID = ""
RECOVERY_ARCHIVE_NAME = ""

saved_result = locate_saved_result(
    BACKUP_DIRECTORY, experiment_id=RECOVERY_EXPERIMENT_ID,
    archive_name=RECOVERY_ARCHIVE_NAME,
)
print("Archive:", saved_result["path"])
print("Kind:", saved_result["kind"], "| Recorded status:", saved_result["status"])
print(saved_result["note"])
print("Archive SHA256:", saved_result["sha256"])
files.download(saved_result["path"])
"""),
    ]
    review = [
        _cell("markdown", f"""# {TITLE}

## Analyze already collected data on CPU

This notebook embeds release **{VERSION}**. No GPU is required. Input archives
are checked against their own inventory and source hashes; compatible dataset
schemas may be analyzed with later code. Uploaded source code is never executed.

A fresh experiment ZIP already contains its complete report and repeat comparison.
Choose `analyze` to reanalyze its primary dataset on CPU; the reviewer resolves
`run/primary/` automatically. For advanced two-archive comparisons, `repeat` accepts
a source and a separately exported repeat dataset, and `gpu` compares source and
target primary datasets. Comparisons transfer the source's
already selected model; target labels never refit it. The first input is the source.

The older published A100 snapshot has a separate verifier:
`python scripts/verify_published_results.py` in the repository.
"""),
        _cell("code", """REVIEW_MODE = "analyze"       # "analyze" (1 ZIP), "repeat" (2), or "gpu" (2).
REVIEW_ID = "review_01"        # New ID for each analysis.
RESULT_ORDER = []             # For comparisons, list exact filenames: [source, target].
PIN_ANALYSIS_DEPS = True
"""),
        _cell("code", bootstrap),
        _cell("markdown", "## Upload complete result ZIPs\nKeep original inputs with any derived analysis."),
        _cell("code", """from google.colab import files
uploaded = files.upload()
review_result = launcher.review(uploaded, review_mode=REVIEW_MODE,
                                result_order=RESULT_ORDER, review_id=REVIEW_ID,
                                pin_analysis_deps=PIN_ANALYSIS_DEPS)
"""),
        _cell("markdown", """## Download the analysis

Open `review/report.html` for analysis or comparison. `review/summary.json` and
CSV/JSON files retain detailed metrics. On failure, inspect the status and log.
"""),
        _cell("code", """files.download(review_result["archive"])
files.download(review_result["archive"] + ".sha256")
"""),
        _cell("markdown", """## Interpretation

Inspect unweighted and occurrence-weighted mean, median, P90 and worst errors,
not only the selector's winning score. CV scores are tuning evidence; held-out
scores are separate evaluations. Keep cache policies and precisions separate.
Replays are not independent GPU sessions. Repeated known shapes test stability,
not new-shape generalization. File checksums establish internal consistency;
they do not attest the physical hardware or replace measurement quality checks.
"""),
    ]
    _notebook(notebook_dir / "Run_Experiment.ipynb", main, gpu=True)
    _notebook(notebook_dir / "Run_Benchmark.ipynb", main, gpu=True)
    _notebook(notebook_dir / "Review_Results.ipynb", review)
    _notebook(notebook_dir / "Download_Saved_Results.ipynb", recovery)
    (output / "SOURCE_SHA256.txt").write_text(source_sha + "  " + source_zip.name + "\n", encoding="ascii")
    return {"version": VERSION, "source_archive": str(source_zip), "source_sha256": source_sha,
            "source_files": len(paths), "notebooks": [str(notebook_dir / name) for name in
                                                       ("Run_Experiment.ipynb", "Run_Benchmark.ipynb", "Review_Results.ipynb", "Download_Saved_Results.ipynb")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--notebook-dir", help="Write notebooks here instead of the output directory")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.out, args.notebook_dir), indent=2))
