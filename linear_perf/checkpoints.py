"""Verified, immutable evidence snapshots outside GPU timing intervals.

This module deliberately uses only the Python standard library so the Colab
launcher can load it by file path before importing the analysis stack. A
checkpoint preserves the recorded state; it never upgrades an incomplete
dataset to completed, reconstructs missing measurements, or resumes GPU work.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from datetime import datetime, timezone
import uuid
import zipfile


class CheckpointError(ValueError):
    """A required evidence backup could not be verified."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _safe_filename(value):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or Path(value).name != value or "/" in value or "\\" in value
            or "\x00" in value or ":" in value):
        raise CheckpointError("Backup filename must be one ordinary filename")
    return value


def _install_without_overwrite(partial, destination):
    """Reserve the final name exclusively immediately before the rename.

    The reservation prevents a competing publication from being overwritten.
    An abrupt process loss during this tiny window can leave an empty final
    name; consumers must require the checksum sidecar and verify its digest.
    """
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        identity = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = destination.lstat()
        if (current.st_dev, current.st_ino, current.st_size) != (identity.st_dev, identity.st_ino, 0):
            raise CheckpointError("Backup destination changed before publication")
        os.replace(partial, destination)
    except BaseException:
        # Remove only our own still-empty reservation, never an existing file.
        try:
            current = destination.lstat()
            if (current.st_dev, current.st_ino, current.st_size) == (identity.st_dev, identity.st_ino, 0):
                destination.unlink()
        except FileNotFoundError:
            pass
        raise


def publish_verified_file(source, directory, filename=None):
    """Copy one closed file, verify readback, and publish without replacement.

    A unique ``.partial`` file is flushed and fsynced, then read back before
    publication. The returned receipt confirms filesystem readback; it cannot
    attest a remote provider's internal replication or recover an earlier VM.
    Existing archives and checksum sidecars are never overwritten. A failure
    leaves the source and earlier verified backups intact.
    """
    original = Path(source)
    if original.is_symlink() or not original.is_file():
        raise CheckpointError("Backup source must be an existing regular file")
    original = original.resolve()
    destination_dir = Path(directory).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(filename if filename is not None else original.name)
    destination = destination_dir / name
    sidecar = destination.with_name(destination.name + ".sha256")
    if destination.exists() or destination.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise CheckpointError("Backup already exists; refusing to overwrite: " + str(destination))
    partial = destination_dir / ("." + name + "." + uuid.uuid4().hex + ".partial")
    sidecar_partial = destination_dir / ("." + name + "." + uuid.uuid4().hex + ".sha256.partial")
    before = original.stat()
    copied, digest = 0, hashlib.sha256()
    try:
        with original.open("rb") as reader, partial.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = original.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or copied != before.st_size:
            raise CheckpointError("Backup source changed during publication")
        expected = digest.hexdigest()
        if partial.stat().st_size != copied or _hash_file(partial) != expected:
            raise CheckpointError("Backup readback checksum mismatch")
        _install_without_overwrite(partial, destination)
        with sidecar_partial.open("x", encoding="ascii") as handle:
            handle.write(expected + "  " + name + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _install_without_overwrite(sidecar_partial, sidecar)
        if sidecar.read_text(encoding="ascii") != expected + "  " + name + "\n":
            raise CheckpointError("Backup checksum sidecar readback mismatch")
        return {"path": str(destination), "sha256": expected, "bytes": copied,
                "sidecar": str(sidecar), "verified": True}
    finally:
        # Incomplete files are not advertised as backups; all earlier snapshots
        # and any successfully published file remain untouched.
        for leftover in (partial, sidecar_partial):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass


class DurableCheckpoints:
    """Publish synchronous snapshots of one experiment's current evidence."""

    def __init__(self, output, directory, *, session_id, source_sha256=None, progress=print):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session_id):
            raise CheckpointError("Checkpoint session ID must be a short alphanumeric identifier")
        if source_sha256 is not None and (not isinstance(source_sha256, str)
                                          or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)):
            raise CheckpointError("Checkpoint source SHA256 must be a lowercase hexadecimal digest")
        if progress is not None and not callable(progress):
            raise CheckpointError("Checkpoint progress must be callable or null")
        if Path(output).is_symlink():
            raise CheckpointError("Checkpoint output cannot be a symbolic link")
        self.output, self.directory = Path(output).resolve(), Path(directory).resolve()
        if self.directory == self.output or self.directory.is_relative_to(self.output):
            raise CheckpointError("Backups must be outside the experiment output tree")
        self.session_id, self.source_sha256, self.progress = session_id, source_sha256, progress
        self.sequence = 0

    def _log(self, record):
        with (self.output / "checkpoints.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def stage(self, summary):
        event = summary.get("checkpoint_event") or ("stage_completed_" + str(summary.get("active_stage")))
        return self._snapshot(event, {"experiment": copy.deepcopy(summary)})

    def collection(self, event):
        child = event.get("session_id", "dataset")
        role = child[len(self.session_id) + 1:] if child.startswith(self.session_id + "_") else "dataset"
        label = str(role) + "_" + str(event.get("checkpoint_event", "collection"))
        if event.get("visit_index") is not None:
            label += "_" + str(event["visit_index"])
        return self._snapshot(label, {"collection": copy.deepcopy(event)})

    def _files(self):
        if self.output.is_symlink() or not self.output.is_dir():
            raise CheckpointError("Experiment output is unavailable for checkpointing")
        result = []
        for path in sorted(self.output.rglob("*")):
            if path.is_symlink() or not path.resolve().is_relative_to(self.output):
                raise CheckpointError("Checkpoint refuses linked or outside evidence: " + str(path))
            if path.is_file() and "__pycache__" not in path.parts:
                result.append((path, "run/" + path.relative_to(self.output).as_posix()))
        for name in ("source_archive.zip", "source_archive.sha256", "launcher_status.json", "worker.log"):
            path = self.output.parent / name
            if path.is_symlink():
                raise CheckpointError("Checkpoint refuses linked source evidence: " + str(path))
            if path.is_file():
                result.append((path, name))
        return result

    def _snapshot(self, event, details):
        event = re.sub(r"[^A-Za-z0-9_-]", "_", str(event))[:96]
        if not event:
            raise CheckpointError("Checkpoint event name cannot be empty")
        self.sequence += 1
        name = f"linear-perf_{self.session_id}_CHECKPOINT_{self.sequence:04d}_{event}.zip"
        started, started_at = time.monotonic(), _utc_now()
        attempt = {"sequence": self.sequence, "event": event, "started_at_utc": started_at,
                   "destination": str(self.directory / name), "status": "started",
                   "timing_scope": "Synchronous boundary outside CUDA event timing intervals"}
        self._log(attempt)
        if self.progress:
            self.progress(f"[{self.session_id}] checkpoint {self.sequence} {event}: saving evidence")
        local = None
        try:
            entries = self._files()
            source_file = self.output.parent / "source_archive.zip"
            actual_source = _hash_file(source_file) if source_file.is_file() else None
            if self.source_sha256 is not None and actual_source != self.source_sha256:
                raise CheckpointError("Checkpoint source archive is missing or its checksum differs")
            source_sha = self.source_sha256 or actual_source
            source_sidecar = self.output.parent / "source_archive.sha256"
            if source_sidecar.is_file():
                recorded = source_sidecar.read_text(encoding="ascii").strip().split()
                if not recorded or actual_source != recorded[0]:
                    raise CheckpointError("Checkpoint source archive checksum sidecar differs")
            snapshot = {
                "schema_version": 1, "artifact_type": "experiment_checkpoint", "protocol_version": "1.1.0",
                "session_id": self.session_id, "checkpoint_event": event, "sequence": self.sequence,
                "created_at_utc": started_at, "source_sha256": source_sha,
                "evidence_status": "Snapshot of recorded state; completion must be read from run/status.json",
                "automatic_resume": False, "original_evidence_modified": False,
                "scope": "A checkpoint preserves existing files; it is not a completed experiment or missing-data reconstruction.",
                **details,
            }
            descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint_", suffix=".zip", dir=self.output.parent)
            os.close(descriptor)
            local = Path(temporary)
            inventory, file_stats = {}, {}
            with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path, relative in entries:
                    before = path.stat()
                    if not stat.S_ISREG(before.st_mode):
                        raise CheckpointError("Checkpoint evidence must be regular files")
                    digest, size = hashlib.sha256(), 0
                    with path.open("rb") as reader, archive.open(relative, "w", force_zip64=True) as writer:
                        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                            writer.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
                        raise CheckpointError("Evidence changed during a checkpoint boundary: " + relative)
                    inventory[relative] = digest.hexdigest()
                    file_stats[relative] = {"bytes": size, "source_mtime_ns": before.st_mtime_ns}
                snapshot["files"] = file_stats
                raw = (json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                archive.writestr("checkpoint.json", raw)
                inventory["checkpoint.json"] = hashlib.sha256(raw).hexdigest()
                manifest = {"source_sha256": source_sha, "files": inventory,
                            "scope": "Checkpoint inventory; internal consistency, not hardware or remote durability attestation"}
                archive.writestr("export_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            with local.open("rb") as handle:
                os.fsync(handle.fileno())
            receipt = publish_verified_file(local, self.directory, filename=name)
            completed = {**attempt, "status": "verified", "finished_at_utc": _utc_now(),
                         "elapsed_seconds": time.monotonic() - started, **receipt}
            self._log(completed)
            if self.progress:
                self.progress(f"[{self.session_id}] checkpoint {self.sequence} verified: {name} ({completed['elapsed_seconds']:.1f}s)")
            return completed
        except BaseException as exc:
            failure = {**attempt, "status": "failed", "finished_at_utc": _utc_now(),
                       "elapsed_seconds": time.monotonic() - started,
                       "error_type": type(exc).__name__, "error": str(exc)}
            self._log(failure)
            if self.progress:
                self.progress(f"[{self.session_id}] checkpoint {self.sequence} failed; retained local evidence is unchanged: {exc}")
            raise
        finally:
            if local is not None:
                try:
                    local.unlink()
                except FileNotFoundError:
                    pass
