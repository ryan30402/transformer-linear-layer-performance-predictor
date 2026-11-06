"""Read and verify saved experiment archives without the original notebook state.

This standalone helper performs no measurement, fitting, extraction, or source
execution. It can be embedded in a separate CPU-only recovery notebook.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile


DEFAULT_BACKUP_DIRECTORY = "/content/drive/MyDrive/Transformer_Linear_Layer_Performance_Predictor/experiments"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Experiment ID must contain only letters, numbers, underscores, and hyphens")
    return value


def list_saved_experiments(backup_directory):
    root = Path(backup_directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError("Backup folder is unavailable. Mount Google Drive first, or check BACKUP_DIRECTORY.")
    experiments = []
    for folder in sorted(root.glob("linear-perf_*")):
        if not folder.is_dir() or folder.is_symlink():
            continue
        experiment_id = folder.name.removeprefix("linear-perf_")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment_id):
            continue
        archives = [p.name for p in sorted(folder.glob("*.zip"))
                    if p.is_file() and not p.is_symlink() and p.with_name(p.name + ".sha256").is_file()]
        experiments.append({"experiment_id": experiment_id, "directory": str(folder),
                            "archives": archives, "archive_count": len(archives)})
    return experiments


def _inspect_saved_archive(path, expected_sha256=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("The selected archive is unavailable or is a link: " + str(path))
    if expected_sha256 is None:
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ValueError("No completed checksum receipt exists for this archive: " + path.name)
        fields = sidecar.read_text(encoding="ascii").strip().split(maxsplit=1)
        if len(fields) != 2 or fields[1].lstrip("*") != path.name:
            raise ValueError("Archive checksum receipt names a different file")
        expected_sha256 = fields[0]
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Invalid SHA256 receipt")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError("Archive checksum mismatch. Preserve this file; select another verified backup.")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        def small_json(name):
            if name not in names:
                return {}
            if archive.getinfo(name).file_size > 4 * 1024 * 1024:
                raise ValueError("Archive metadata exceeds the inspection limit")
            value = json.loads(archive.read(name))
            if not isinstance(value, dict):
                raise ValueError("Archive metadata must be a JSON object")
            return value
        checkpoint = small_json("checkpoint.json")
        run = small_json("run/status.json") or small_json("run/summary.json")
        launch = small_json("launcher_status.json")
    kind = "checkpoint" if checkpoint else "results"
    status = run.get("status", "unknown")
    return {"path": str(path), "sha256": actual, "kind": kind, "status": status,
            "bytes": path.stat().st_size, "launcher_status": launch.get("status", "unknown"),
            "checkpoint": checkpoint, "measurement_kind": run.get("measurement_kind", "unknown"),
            "all_requested_stages_completed": run.get("all_requested_stages_completed", False),
            "note": ("This is a preserved checkpoint, not proof of a completed experiment. No work was resumed."
                     if kind == "checkpoint" else
                     "An exported archive can contain a failed or incomplete run. Inspect its recorded status.")}


def locate_saved_result(backup_directory, *, experiment_id="", archive_name="", local_result=None):
    """Select one existing archive and verify it before offering a download.

    An explicit experiment/filename takes precedence. Without a selection, a
    surviving launcher result can resolve the local archive; otherwise exactly
    one saved experiment is required. No arbitrary latest-experiment selection.
    """
    if experiment_id:
        _valid_id(experiment_id)
    if archive_name and (Path(archive_name).name != archive_name or "\\" in archive_name
                         or not archive_name.endswith(".zip")):
        raise ValueError("ARCHIVE_NAME must be a ZIP filename, not a path")
    if not experiment_id and not archive_name and isinstance(local_result, dict):
        local = local_result.get("archive")
        expected = local_result.get("result_archive_sha256")
        if local and expected and Path(local).is_file():
            return _inspect_saved_archive(local, expected)
    experiments = list_saved_experiments(backup_directory)
    if experiment_id:
        experiments = [item for item in experiments if item["experiment_id"] == experiment_id]
        if not experiments:
            raise FileNotFoundError("No saved folder for the selected experiment: " + experiment_id)
    if len(experiments) != 1:
        ids = [item["experiment_id"] for item in experiments]
        raise ValueError("Set RECOVERY_EXPERIMENT_ID to one exact listed ID. Available IDs: " + repr(ids))
    experiment = experiments[0]
    folder = Path(experiment["directory"])
    experiment_id = experiment["experiment_id"]
    if archive_name:
        selected = folder / archive_name
    else:
        final = f"linear-perf_{experiment_id}_RESULTS.zip"
        if final in experiment["archives"]:
            selected = folder / final
        else:
            prefix = f"linear-perf_{experiment_id}_CHECKPOINT_"
            checkpoints = [name for name in experiment["archives"] if name.startswith(prefix)]
            if not checkpoints:
                raise FileNotFoundError("No completed archive/checksum pair is saved for this experiment yet")
            selected = folder / sorted(checkpoints)[-1]
    return _inspect_saved_archive(selected)
