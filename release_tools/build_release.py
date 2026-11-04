"""Build a deterministic source archive and self-contained English Colab notebooks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zipfile

VERSION = "1.0.0"
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
    main = [
        _cell("markdown", f"""# {TITLE}

## Run a benchmark session

This notebook embeds release **{VERSION}**, including the fixed experiment plans.
Select **Runtime > Change runtime type > GPU** before a measured run. The recorded
hardware identity, rather than the requested allocation, determines the device used.

The workload is preallocated `Y = X @ W.T`. The primary label is a warm-reuse
CUDA Graph replay interval divided by its GEMM count. GPU reads and writes are
inside the interval. Host transfers, graph capture, bias, activation, and serving
queues are outside it. This is not an end-to-end language-model benchmark.

Keep every result archive, including failed or qualified runs. The launcher makes
one attempt. The campaign has a 30-minute budget and the worker a 32-minute timeout;
dependency setup and export are additional. It retains the runtime's PyTorch/CUDA.
"""),
        _cell("code", """# Choose a new session ID for each attempt; existing output is never overwritten.
ACTION = "run"               # "run" or "diagnose"; diagnose does not open held-out data.
DTYPE = "float16"            # "float16" or "bfloat16"; fit each dtype separately.
SESSION_ID = "fp16_s01"
SIMULATE = False             # True exercises software on CPU; it is not a GPU measurement.
PIN_ANALYSIS_DEPS = True     # Pin the recorded NumPy/SciPy/Matplotlib analysis stack.
"""),
        _cell("markdown", "## Load the release\nThis verifies the embedded source and Python syntax."),
        _cell("code", bootstrap),
        _cell("markdown", """## Execute one session

Progress and quality flags are retained. Download the archive as soon as the run
finishes: a complete Colab runtime loss can prevent any program from exporting.
"""),
        _cell("code", """result = launcher.run(session_id=SESSION_ID, dtype=DTYPE, action=ACTION,
                      simulate=SIMULATE, pin_analysis_deps=PIN_ANALYSIS_DEPS)
print("Result archive:", result["archive"])
"""),
        _cell("markdown", """## Download the complete result

For a completed run, open `run/report.html` after extraction. For diagnostics or
failures, read `launcher_status.json`, `worker.log`, and `run/status.json`.
"""),
        _cell("code", """from google.colab import files
files.download(result["archive"])
files.download(result["archive"] + ".sha256")
"""),
        _cell("markdown", """## Plan repeated sessions before looking at test scores

- A paired dtype comparison requires the same recorded physical GPU UUID. Run
  `bfloat16` with `bf16_s01` in the same runtime, then download both archives.
- Repeat the registered procedure with session IDs ending in `s02` and `s03`.
  Separate processes do not establish separate physical devices.
- Use `Review_Results.ipynb` for CPU-only auditing and comparisons of archives
  produced by this exact embedded release. Fix the transfer source in advance.
- Inspect the measurement kind, formal selected model, test and challenge errors,
  tail errors, controls, and quality flags. Synthetic data are software checks only.

The repository's published A100 snapshot has its own offline verification command:
`python scripts/verify_published_results.py`. It is not an input archive for this launcher.
"""),
    ]
    review = [
        _cell("markdown", f"""# {TITLE}

## Review result archives on CPU

Upload complete `RESULTS.zip` archives produced by this exact embedded release
**{VERSION}**. The reviewer checks archive inventory, source identity, sealed data,
models, and chronology before analysis. It does not measure GPU work or change inputs.

`study` requires three compatible sessions. `transfer` requires two and treats
the first as the source. `compare-dtypes` requires FP16 and BF16 on the same physical
GPU. Choose input order from the session plan, not from test scores.

For the repository's published A100 snapshot, use
`python scripts/verify_published_results.py` in a local checkout instead.
"""),
        _cell("code", """REVIEW_MODE = "study"        # "study", "transfer", or "compare-dtypes"
TRANSFER_MODE = "gpu"        # "gpu" or "session", used only with transfer.
REVIEW_ID = "review_01"
RESULT_ORDER = []             # Exact uploaded filenames in planned order; [] sorts by filename.
PIN_ANALYSIS_DEPS = True
"""),
        _cell("code", bootstrap),
        _cell("markdown", "## Upload and review\nKeep the original archives with the derived report."),
        _cell("code", """from google.colab import files
uploaded = files.upload()
review_result = launcher.review(uploaded, review_mode=REVIEW_MODE,
                                transfer_mode=TRANSFER_MODE, result_order=RESULT_ORDER,
                                review_id=REVIEW_ID, pin_analysis_deps=PIN_ANALYSIS_DEPS)
"""),
        _cell("markdown", """## Download the review

Open `review/report.html` after extraction. On failure, inspect the status and log;
environment and evidence checks must pass before a comparison can be interpreted.
"""),
        _cell("code", """files.download(review_result["archive"])
files.download(review_result["archive"] + ".sha256")
"""),
        _cell("markdown", """## Interpretation

Keep every session and quality qualification. Zero-shot transfer and target-label
calibration are distinct analyses. Replays within a session are not independent
GPU sessions, and acquisition seeds are not additional hardware trials. File hashes
establish internal consistency, not external hardware attestation.
"""),
    ]
    _notebook(notebook_dir / "Run_Benchmark.ipynb", main, gpu=True)
    _notebook(notebook_dir / "Review_Results.ipynb", review)
    (output / "SOURCE_SHA256.txt").write_text(source_sha + "  " + source_zip.name + "\n", encoding="ascii")
    return {"version": VERSION, "source_archive": str(source_zip), "source_sha256": source_sha,
            "source_files": len(paths), "notebooks": [str(notebook_dir / name) for name in
                                                       ("Run_Benchmark.ipynb", "Review_Results.ipynb")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--notebook-dir", help="Write notebooks here instead of the output directory")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.out, args.notebook_dir), indent=2))
