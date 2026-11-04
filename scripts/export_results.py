#!/usr/bin/env python3
"""Build the publication snapshot from locally retained acquisition archives.

This is a presentation export, not a new experiment. Measured numbers, model
parameters, predictions, timestamps, and scientific decisions are preserved.
The original acquisition archives are read only. Their recorded seals are not
reissued for the exported representation; MANIFEST.json authenticates that
representation separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Any


PROJECT = "Transformer Linear Layer Performance Predictor"
PUBLICATION_VERSION = "1.0.0"
SESSION_IDS = tuple(f"{dtype}_s{number:02d}" for dtype in ("fp16", "bf16") for number in (1, 2, 3))
EXCLUDED_RUN_FILES = {
    "diagnostics_summary.json",  # Byte-identical duplicate of diagnostics/diagnostics.json.
    "manifest.json",  # Its acquisition seal is represented separately below.
    "source_identity.json",  # Keep its opaque identity without a source-file inventory.
    "report.html",  # Replaced by the English repository report.
}
EXCLUDED_KEYS = {"protocol_version", "history_registry", "live_llm_called", "live_llm_used"}
KEY_RENAMES = {
    "historical_shape": "diagnostic_shape",
    "max_agent_rounds": "max_acquisition_rounds",
}
CSV_FILES = (
    "selected_pipeline.csv", "model_metrics.csv", "quality_and_audits.csv",
    "case_latencies.csv", "cross_round_latency_spread.csv", "paired_dtype_latencies.csv",
    "frozen_selected_transfer.csv", "frozen_krr_candidate_transfer.csv",
    "acquisition_all_budgets.csv", "offline_batching.csv",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def neutral_key(key: str) -> str:
    if re.fullmatch(r"v[0-9]+", key):
        return "study"
    return re.sub(r"^v[0-9]+_", "", KEY_RENAMES.get(key, key))


def normalize(value: Any) -> Any:
    """Normalize presentation metadata while leaving numeric leaves untouched."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in EXCLUDED_KEYS:
                continue
            if key == "filename" and isinstance(item, str) and item.endswith("_RESULTS.zip"):
                match = re.search(r"(fp16|bf16)_s[0-9]+", item)
                if match:
                    # Some summary records already declare this same identity.
                    # It may be redundant, but must never silently disagree.
                    if value.get("session_id", match.group(0)) != match.group(0):
                        raise ValueError("Archive label and declared session identity disagree")
                    if "session_id" not in value:
                        if "session_id" in result:
                            raise ValueError("Normalized session identity key collision")
                        result["session_id"] = match.group(0)
                    continue
            public_key = neutral_key(key)
            if public_key in result:
                raise ValueError(f"Normalized metadata key collision: {public_key}")
            result[public_key] = normalize(item)
        return result
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        result = value.replace("historical_diagnostic_not_generalization", "diagnostic_not_generalization")
        result = re.sub(r"(?:_|-)v[0-9]+(?=[_\-]|$)", "", result)
        result = re.sub(r"^v[0-9]+_", "", result)
        result = result.replace("_linear_perf_registered_study", "_registered_study")
        # Colab paths are execution-local, not a portable location of evidence.
        result = re.sub(r"/content/[^\s\"']+?/run(?=/|$)", ".", result)
        return result
    return value


def numeric_leaves(value: Any, path: tuple = ()) -> list:
    """Identify every retained number by its normalized JSON path.

    Only omitted presentation-only metadata is outside this comparison.
    Booleans are intentionally not treated as numbers.
    """
    if isinstance(value, dict):
        output = []
        for key, item in value.items():
            if key not in EXCLUDED_KEYS:
                output.extend(numeric_leaves(item, (*path, neutral_key(key))))
        return output
    if isinstance(value, list):
        output = []
        for index, item in enumerate(value):
            output.extend(numeric_leaves(item, (*path, index)))
        return output
    if type(value) in (int, float):
        return [[list(path), value]]
    return []


def numeric_digest(value: Any) -> str:
    # JSON paths can mix string keys and integer array positions.
    leaves = sorted(numeric_leaves(value), key=lambda row: json.dumps(row[0]))
    return sha256(json.dumps(leaves, separators=(",", ":"), allow_nan=False).encode())


def reject_symlink_components(path: Path) -> None:
    """Do not let a chosen output path traverse an existing symbolic link."""
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Symbolic links are not permitted in an export path: {component}")


def validate_archive_members(archive: zipfile.ZipFile) -> None:
    """Reject unsafe or ambiguous member names before any member is exported."""
    seen = set()
    for info in archive.infolist():
        name = info.filename
        parts = name.removesuffix("/").split("/")
        if (not name or name.startswith("/") or "\\" in name or "\x00" in name or
                re.match(r"^[A-Za-z]:", name) or any(part in {"", ".", ".."} for part in parts)):
            raise ValueError(f"Unsafe archive member path: {name!r}")
        canonical = "/".join(parts)
        if canonical in seen:
            raise ValueError(f"Duplicate archive member path: {name!r}")
        seen.add(canonical)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"Symbolic-link archive members are not supported: {name!r}")


def export(evidence_root: Path, output: Path) -> dict:
    # Refuse to mutate an existing baseline. Export to a fresh directory and
    # review the new snapshot separately when a new experiment is released.
    output = output.absolute()
    reject_symlink_components(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Export requires a new or empty output directory; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "project": PROJECT,
        "publication_version": PUBLICATION_VERSION,
        "dataset_kind": "publication_snapshot_of_recorded_cuda_measurements",
        "scope": (
            "File integrity and reproducible CPU checks of recorded evidence. "
            "No independent attestation of GPU execution, no new measurements, "
            "and no assertion that publication source bytes equal acquisition source bytes."
        ),
        "normalization": [
            "English publication labels and portable relative paths replace execution-local labels and paths.",
            "Protocol revision labels and development-only shape registries are omitted from data presentation.",
            "Project source inventories, generated reports, launcher logs, and duplicate diagnostic summaries are not included.",
            "Every retained numeric JSON leaf is checked for exact equality during export; binary numerical witnesses are byte-identical.",
            "Embedded hashes, signatures, and environment IDs remain recorded acquisition metadata; they are not recomputed to resemble original seals.",
            "This manifest's SHA-256 fields authenticate the exported bytes. Original SHA-256 fields identify retained input artifacts separately.",
        ],
        "recorded_inputs": {},
        "files": {},
    }

    def save(relative: str, data: bytes, *, original: bytes | None = None,
             source_artifact: str | None = None, numeric: str | None = None,
             transformation: str | None = None) -> None:
        path = output / relative
        reject_symlink_components(path)
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError(f"Export path escapes the chosen directory: {relative}")
        if path.exists():
            raise ValueError(f"Export would overwrite an existing file: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entry = {"sha256": sha256(data), "bytes": len(data)}
        if original is not None:
            entry["original_sha256"] = sha256(original)
            entry["transformation"] = transformation or (
                "byte_identical" if original == data else "presentation_metadata_normalized"
            )
        if source_artifact:
            entry["source_artifact"] = source_artifact
        if numeric:
            entry["retained_numeric_payload_sha256"] = numeric
        manifest["files"][relative] = entry

    observed_sessions = set()
    for archive_path in sorted((evidence_root / "raw_results").glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            validate_archive_members(archive)
            if "run/environment.json" not in archive.namelist():
                continue
            environment = json.loads(archive.read("run/environment.json"))
            session_id = environment["session_id"]
            if session_id not in SESSION_IDS or session_id in observed_sessions:
                raise ValueError(f"Unexpected or duplicate session: {session_id}")
            observed_sessions.add(session_id)
            source_identity_bytes = archive.read("run/source_identity.json")
            source_identity = json.loads(source_identity_bytes)
            seal_bytes = archive.read("run/manifest.json")
            seal = json.loads(seal_bytes)
            original_source_sha = archive.read("source_archive.sha256").decode().strip().split()[0]
            if sha256(archive.read("source_archive.zip")) != original_source_sha:
                raise ValueError(f"Acquisition source archive checksum mismatch: {session_id}")
            for member, expected_hash in seal["files"].items():
                if sha256(archive.read("run/" + member)) != expected_hash:
                    raise ValueError(f"Recorded acquisition seal mismatch: {session_id}/{member}")
            manifest["recorded_inputs"][session_id] = {
                "archive_sha256": sha256(archive_path.read_bytes()),
                "acquisition_source_archive_sha256": original_source_sha,
                "acquisition_source_identity_sha256": sha256(source_identity_bytes),
                "acquisition_run_seal_sha256": sha256(seal_bytes),
                "original_acquisition_seal_checked_during_export": True,
            }

            for member in archive.namelist():
                if not member.startswith("run/") or member.endswith("/"):
                    continue
                relative = member.removeprefix("run/")
                if relative in EXCLUDED_RUN_FILES or relative.startswith("figures/"):
                    continue
                original = archive.read(member)
                numeric = None
                if relative.endswith((".json", ".jsonl")):
                    if relative.endswith(".jsonl"):
                        parsed = [json.loads(line) for line in original.decode().splitlines() if line.strip()]
                        normalized = normalize(parsed)
                        data = b"".join(
                            (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
                            for row in normalized
                        )
                    else:
                        parsed = json.loads(original)
                        normalized = normalize(parsed)
                        data = encode_json(normalized)
                    if numeric_leaves(parsed) != numeric_leaves(normalized):
                        raise ValueError(f"A retained numeric value changed: {session_id}/{relative}")
                    numeric = numeric_digest(normalized)
                else:
                    data = original
                if not relative.endswith(".npz"):
                    text = data.decode()
                    if re.search(r"[\u3400-\u9fff]", text):
                        raise ValueError(f"Non-English presentation in {session_id}/{relative}")
                save(f"a100/{session_id}/{relative}", data, original=original,
                     source_artifact=member, numeric=numeric)

            # Preserve recorded acquisition provenance without pretending that
            # rewritten metadata is byte-identical to the acquisition record.
            recorded_identity = {
                "source_hash": source_identity["source_hash"],
                "analysis_versions": source_identity["analysis_versions"],
                "source_archive_sha256": original_source_sha,
                "scope": "Recorded acquisition source identity; not the publication source identity.",
            }
            save(f"a100/{session_id}/recorded_source_identity.json", encode_json(recorded_identity),
                 original=source_identity_bytes, source_artifact="run/source_identity.json",
                 transformation="recorded_identity_projection_without_source_inventory")
            recorded_seal = {
                key: normalize(value) for key, value in seal.items()
                if key not in {"protocol_version", "implementation", "files", "scope"}
            }
            recorded_seal["recorded_file_sha256"] = seal["files"]
            recorded_seal["source_hash"] = source_identity["source_hash"]
            recorded_seal["scope"] = (
                "Recorded acquisition hashes; they refer to original acquisition bytes, including omitted artifacts. "
                "Use ../../MANIFEST.json for the normalized publication bytes."
            )
            save(f"a100/{session_id}/recorded_acquisition_seal.json", encode_json(recorded_seal),
                 original=seal_bytes, source_artifact="run/manifest.json",
                 transformation="recorded_seal_projection_without_source_inventory")

    if observed_sessions != set(SESSION_IDS):
        raise ValueError(f"Expected all six measured sessions, found {sorted(observed_sessions)}")

    for name in CSV_FILES:
        original = (evidence_root / "analysis" / name).read_bytes()
        save(f"summary/{name}", original, original=original, source_artifact=f"analysis/{name}")

    summary_sources = {
        "summary.json": evidence_root / "FINAL_SUMMARY.json",
        "cross_session_summary.json": evidence_root / "analysis" / "six_session_summary.json",
    }
    for name, source in summary_sources.items():
        original = source.read_bytes()
        parsed = json.loads(original)
        normalized = normalize(parsed)
        if name == "summary.json":
            normalized["note"] = (
                "Acquisition statuses and conservative flags are retained. "
                "This normalized snapshot and its CPU verification are separate from acquisition records and do not certify hardware."
            )
        else:
            normalized["limitations"][3] = (
                "Recorded acquisition audit outcomes are summarized separately. "
                "The publication verifier checks exported-file integrity and reproducible numeric evidence; "
                "it does not reconstruct omitted original source bytes or original full-archive seals."
            )
        if numeric_leaves(parsed) != numeric_leaves(normalized):
            raise ValueError(f"A summary number changed: {name}")
        save(f"summary/{name}", encode_json(normalized), original=original,
             source_artifact=source.name, numeric=numeric_digest(normalized))

    manifest_path = output / "MANIFEST.json"
    reject_symlink_components(manifest_path)
    if manifest_path.exists():
        raise ValueError("Export would overwrite an existing manifest")
    manifest_path.write_bytes(encode_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True,
                        help="Local retained evidence directory with raw_results/, analysis/, and FINAL_SUMMARY.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    manifest = export(args.evidence_root.resolve(), args.output.absolute())
    print(json.dumps({"sessions": len(manifest["recorded_inputs"]), "files": len(manifest["files"]),
                      "bytes": sum(item["bytes"] for item in manifest["files"].values()),
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
