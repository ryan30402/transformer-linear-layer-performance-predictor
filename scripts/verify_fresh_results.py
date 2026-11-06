#!/usr/bin/env python3
"""Verify the published 1.1.0 A100 result archive on a CPU.

This checks fixed archive/source provenance, all export-manifest hashes, the
readable CSV copies, dataset records, saved-model predictions and metrics, CV
record consistency, and paired repeats. It never executes the embedded source,
fits a model, collects GPU data, changes quality thresholds, or removes rows.
The preserved timing-quality qualifications are expected evidence, not a
reason for this integrity/consistency command to fail.

The full repository checkout includes the archive under experiments/. Embedded
Colab source bundles omit experiments/ to avoid duplicating measured data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SESSION_ID = "fresh_20261106T065308_c5e789"
ARCHIVE_SHA256 = "a601e14bb3d1d2b661f06021432b461104850452ddcc1c695c55006db29374b1"
SOURCE_SHA256 = "1f748883a23d0e9706856493eb011eec88d5e3f8926fafd3874c5d23d2c8345a"
MANIFEST_FILES = 1444
EXACT_COPIES = {
    "selected_model_metrics.csv": "run/selected_model_metrics.csv",
    "metrics.csv": "run/primary/analysis/metrics.csv",
}
DEFAULT_PUBLICATION = ROOT / "experiments" / "a100_20261106"


class Verification:
    def __init__(self):
        self.checks = 0

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def close(self, actual, expected, message):
        self.check(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
                   math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10), message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    """Require a canonical relative POSIX file name, also safe on Windows."""
    path = PurePosixPath(name)
    if (not name or "\\" in name or "\x00" in name or path.is_absolute() or
            any(part in {".", ".."} or ":" in part for part in path.parts) or
            path.as_posix() != name):
        raise ValueError(f"Unsafe publication/archive path: {name!r}")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_asset_checksums(directory, checks):
    """Check the publication's files without accepting symlink/path escapes."""
    root = Path(directory).resolve()
    expected = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        safe_name(name)
        checks.check(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                     f"Invalid asset checksum: {name}")
        checks.check(name not in expected, f"Duplicate asset checksum: {name}")
        expected[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
              and path.relative_to(root).as_posix() != "SHA256SUMS.txt"}
    checks.check(set(expected) == actual, "Publication checksum coverage mismatch")
    for name, digest in expected.items():
        path = root / name
        checks.check(path.resolve().is_relative_to(root) and not path.is_symlink(),
                     f"Unsafe asset location: {name}")
        checks.check(sha256_file(path) == digest, f"Publication asset checksum mismatch: {name}")
    return len(expected)


def verify_archive(path, destination, checks, *, expected_sha256=ARCHIVE_SHA256,
                   expected_source_sha256=SOURCE_SHA256, expected_manifest_files=MANIFEST_FILES):
    """Hash every member; materialize only run/ records in a temporary directory.

    Expected hashes are fixed for the published run. Keyword overrides support
    small local test fixtures, not CLI acceptance of a different experiment.
    """
    checks.check(sha256_file(path) == expected_sha256, "Result archive SHA-256 mismatch")
    target_root = Path(destination).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    checks.check(not any(target_root.iterdir()), "Archive verification needs an empty temporary destination")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        checks.check(len(entries) <= 5000, "Archive member limit exceeded")
        checks.check(sum(info.file_size for info in entries) <= 512 * 1024**2,
                     "Archive expanded-size limit exceeded")
        names = []
        for info in entries:
            safe_name(info.filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            checks.check(not info.is_dir() and mode in {0, stat.S_IFREG},
                         f"Archive has a non-regular file: {info.filename}")
            checks.check(not info.flag_bits & 1, "Encrypted archive member is unsupported")
            checks.check(info.file_size <= 128 * 1024**2, "Archive member size limit exceeded")
            names.append(info.filename)
        checks.check(len(names) == len(set(names)), "Duplicate archive member")
        checks.check("export_manifest.json" in names, "Missing export manifest")
        manifest = json.loads(archive.read("export_manifest.json"))
        members = manifest["files"]
        checks.check(isinstance(members, dict), "Export manifest files must be a mapping")
        checks.check(len(members) == expected_manifest_files, "Unexpected export-manifest member count")
        checks.check(set(members) == set(names) - {"export_manifest.json"}, "Export manifest coverage mismatch")
        checks.check(manifest["source_sha256"] == expected_source_sha256,
                     "Export-manifest source provenance mismatch")
        for name, digest in members.items():
            content = archive.read(name)  # ZipFile also verifies the stored CRC.
            checks.check(hashlib.sha256(content).hexdigest() == digest, f"Export member hash mismatch: {name}")
            if name.startswith("run/"):
                target = target_root / name
                checks.check(target.resolve().is_relative_to(target_root), "Archive extraction would escape its root")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        checks.check(hashlib.sha256(archive.read("source_archive.zip")).hexdigest() == expected_source_sha256,
                     "Embedded source archive SHA-256 mismatch")
        checks.check(archive.read("source_archive.sha256").decode().strip() == expected_source_sha256,
                     "Embedded source checksum record mismatch")
        launcher = json.loads(archive.read("launcher_status.json"))
    return manifest, launcher


def verify_exact_copies(archive_path, directory, copies, checks):
    root = Path(directory).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for filename, member in copies.items():
            safe_name(filename)
            safe_name(member)
            path = root / filename
            checks.check(path.resolve().is_relative_to(root) and not path.is_symlink(),
                         f"Unsafe CSV copy: {filename}")
            checks.check(path.read_bytes() == archive.read(member), f"Readable CSV copy differs from archive: {filename}")


def numeric_fields_equal(recorded, computed, checks, label):
    for key, value in computed.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            checks.check(key in recorded, f"Missing metric {label}/{key}")
            checks.close(recorded[key], value, f"Metric mismatch {label}/{key}")


def verify_models(run, primary_rows, checks):
    """Replay the saved models through trusted local code, without fitting."""
    import numpy as np
    from linear_perf.experiment import _source_suite
    from linear_perf.selection import evaluate_cv_suite, feature_matrix, regression_metrics

    analysis = run / "primary" / "analysis"
    summary = read_json(analysis / "summary.json")
    checks.check(summary["status"] == "completed" and not summary["heldout_labels_used_for_selection"],
                 "Analysis completion/selection status mismatch")
    metrics = read_csv(analysis / "metrics.csv")
    selected = read_csv(run / "selected_model_metrics.csv")
    predictions = read_csv(analysis / "predictions.csv")
    by_metric = {(row["group_id"], row["split"], row["model_id"]): row for row in metrics}
    by_prediction = {(row["group_id"], row["split"], row["model_id"], row["case_id"]): row for row in predictions}
    checks.check(len(by_metric) == len(metrics) == 60, "Candidate metric coverage mismatch")
    checks.check(len(by_prediction) == len(predictions) == 5510, "Per-case prediction coverage mismatch")
    checks.check(selected == [row for row in metrics if row["selected_by_cv"] == "True"] and len(selected) == 12,
                 "Selected metric rows differ from the CV-selected subset")
    evaluated_count = 0
    replayed_count = 0
    for group in summary["groups"]:
        gid = group["group_id"]
        context = (group["environment_id"], group["dtype"], group["cache_policy"])
        rows = [row for row in primary_rows if tuple(row[k] for k in ("environment_id", "dtype", "cache_policy")) == context]
        train = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["case_id"])
        train_by_id = {row["case_id"]: row for row in train}
        suite = _source_suite(analysis, group, primary_rows)
        checks.check(set(suite["train_case_ids"]) == set(train_by_id), f"{gid}: training IDs differ")
        checks.check(suite["selected_model_id"] == group["selected_model_id"], f"{gid}: selected model differs")
        cv = suite["cv"]
        checks.check(cv["fold_count"] == len(cv["folds"]) == 6, f"{gid}: fold count differs")
        seen_validation = Counter()
        for fold in cv["folds"]:
            train_ids, validation_ids = set(fold["train_case_ids"]), set(fold["validation_case_ids"])
            checks.check(len(train_ids) == len(fold["train_case_ids"]) == fold["n_train"] and
                         len(validation_ids) == len(fold["validation_case_ids"]) == fold["n_validation"],
                         f"{gid}: duplicate or missing fold cases")
            checks.check(not train_ids & validation_ids and train_ids | validation_ids == set(train_by_id),
                         f"{gid}: fold overlaps or includes nontraining labels")
            train_families = {(train_by_id[cid]["n"], train_by_id[cid]["k"]) for cid in train_ids}
            validation_families = {(train_by_id[cid]["n"], train_by_id[cid]["k"]) for cid in validation_ids}
            checks.check(not train_families & validation_families, f"{gid}: N/K families overlap within fold")
            seen_validation.update(validation_ids)
            x = feature_matrix([train_by_id[cid] for cid in fold["train_case_ids"]], suite["dtype"], sm_count=suite["sm_count"])
            mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
            scale = np.where(scale < 1e-10, 1.0, scale)
            checks.check(np.allclose(mean, fold["feature_mean"], rtol=1e-10, atol=1e-10) and
                         np.allclose(scale, fold["feature_scale"], rtol=1e-10, atol=1e-10),
                         f"{gid}: standardization is not fold-training-only")
        checks.check(seen_validation == Counter({cid: 1 for cid in train_by_id}),
                     f"{gid}: incomplete out-of-fold coverage")
        order = suite["model_order"]
        score, tail = suite["selection_score_key"], suite["selection_secondary_key"]
        for model_id in order:
            candidate = cv["models"][model_id]
            trials = [trial for trial in cv["trials"] if trial["model_id"] == model_id]
            best = min(trials, key=lambda item: (item["oof_metrics"][score], item["oof_metrics"][tail], item["trial_index"]))
            checks.check(candidate["trial_index"] == best["trial_index"] and
                         candidate["configuration"] == suite["models"][model_id]["configuration"],
                         f"{gid}/{model_id}: configuration selection mismatch")
            oof = {item["case_id"]: item["predicted_ms"] for item in candidate["oof_predictions"]}
            checks.check(len(oof) == len(candidate["oof_predictions"]) and set(oof) == set(train_by_id),
                         f"{gid}/{model_id}: out-of-fold prediction coverage mismatch")
            computed_oof = regression_metrics(train, [oof[row["case_id"]] for row in train], include_predictions=False)
            for key, value in candidate["oof_metrics"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    checks.close(value, computed_oof[key], f"{gid}/{model_id}: OOF {key}")
        winner = min(order, key=lambda name: (cv["models"][name]["oof_metrics"][score],
                                              cv["models"][name]["oof_metrics"][tail], order.index(name)))
        checks.check(winner == suite["selected_model_id"], f"{gid}: saved CV winner differs")
        for split in ("test", "challenge", "stress"):
            targets = [row for row in rows if row["split"] == split]
            evaluation = evaluate_cv_suite(suite, targets, sm_count=suite["sm_count"])
            saved = read_json(analysis / group["evaluation_files"][split])
            checks.check(saved["refitted_during_evaluation"] is False and
                         saved["suite_hash"] == suite["suite_hash"], f"{gid}/{split}: evaluation source differs")
            for model_id, computed in evaluation["models"].items():
                key = (gid, split, model_id)
                metric_row = by_metric[key]
                checks.check(metric_row["selected_by_cv"] == str(model_id == winner), f"{key}: selection annotation mismatch")
                numeric_fields_equal(metric_row, computed, checks, str(key))
                numeric_fields_equal(saved["models"][model_id], computed, checks, f"saved/{key}")
                values = computed["predictions"]
                actual = np.asarray([p["actual_ms"] for p in values])
                predicted = np.asarray([p["predicted_ms"] for p in values])
                weights = np.asarray([p["weight"] for p in values])
                ape = np.abs(predicted - actual) / actual * 100
                independent = {"mean_ape_pct": np.mean(ape), "median_ape_pct": np.median(ape),
                               "p90_ape_pct": np.percentile(ape, 90), "max_ape_pct": np.max(ape),
                               "weighted_mean_ape_pct": np.sum(weights * ape) / np.sum(weights)}
                for name, value in independent.items():
                    checks.close(metric_row[name], value, f"{key}: independently recomputed {name}")
                saved_predictions = {p["case_id"]: p for p in saved["models"][model_id]["predictions"]}
                checks.check(len(saved_predictions) == len(values), f"{key}: saved prediction coverage mismatch")
                for prediction in values:
                    cid = prediction["case_id"]
                    recorded = by_prediction[(*key, cid)]
                    checks.check(recorded["selected_by_cv"] == str(model_id == winner), f"{key}/{cid}: selection differs")
                    numeric_fields_equal(recorded, prediction, checks, f"CSV/{key}/{cid}")
                    numeric_fields_equal(saved_predictions[cid], prediction, checks, f"JSON/{key}/{cid}")
                    replayed_count += 1
                evaluated_count += 1
    checks.check(evaluated_count == len(metrics) and replayed_count == len(predictions), "Evaluation coverage incomplete")
    return {"contexts": len(summary["groups"]), "candidate_metric_rows": evaluated_count,
            "selected_metric_rows": len(selected), "replayed_predictions": replayed_count,
            "model_fitting_performed": False, "cv_scope": "Validate stored trial rankings and selected OOF records; no hyperparameter refitting."}


def verify_repeat(run, primary_rows, repeat_rows, checks):
    import numpy as np
    summary = read_json(run / "repeat_comparison" / "summary.json")
    checks.check(summary["status"] == "completed" and summary["target_labels_used_for_refit"] is False,
                 "Repeat completion/refit status mismatch")
    devices = [read_json(run / stage / "dataset" / "environments.json")["device"] for stage in ("primary", "repeat")]
    checks.check(devices[0] == devices[1] and devices[0].get("uuid") and
                 summary["same_physical_gpu_verified"] is True, "Recorded repeat hardware context differs")
    observed = []
    for group in summary["groups"]:
        def index(rows):
            return {row["case_id"]: row for row in rows if row["dtype"] == group["dtype"] and
                    row["cache_policy"] == group["cache_policy"]}
        source, target = index(primary_rows), index(repeat_rows)
        records = read_csv(run / "repeat_comparison" / group["repeat_file"])
        checks.check({row["case_id"] for row in records} == set(target) and len(records) == len(target),
                     "Repeat comparison case coverage mismatch")
        changes = []
        for record in records:
            a, b = source[record["case_id"]], target[record["case_id"]]
            ratio = b["median_ms"] / a["median_ms"]
            change = abs(ratio - 1) * 100
            checks.check(a["split"] == b["split"] == record["split"], "Repeat split differs")
            for key, value in {"source_ms": a["median_ms"], "target_ms": b["median_ms"],
                               "target_over_source": ratio, "absolute_change_pct": change}.items():
                checks.close(record[key], value, f"Repeat/{record['case_id']}: {key}")
            changes.append(change)
        checks.close(group["median_absolute_change_pct"], np.median(changes), "Repeat median differs")
        checks.close(group["maximum_absolute_change_pct"], max(changes), "Repeat maximum differs")
        checks.check(group["paired_shapes"] == len(changes), "Repeat pair count differs")
        observed.append({"dtype": group["dtype"], "cache_policy": group["cache_policy"], "pairs": len(changes),
                         "median_absolute_change_pct": float(np.median(changes)),
                         "maximum_absolute_change_pct": max(changes)})
    return observed


def verify_publication(directory=DEFAULT_PUBLICATION):
    from linear_perf.experiment import audit_dataset

    root = Path(directory).resolve()
    checks = Verification()
    assets = verify_asset_checksums(root, checks)
    publication = read_json(root / "publication.json")
    checks.check(publication["protocol_version"] == "1.1.0" and publication["session_id"] == SESSION_ID,
                 "Publication identity differs from the registered run")
    checks.check(publication["archive"]["file"] == "results.zip" and
                 publication["archive"]["sha256"] == ARCHIVE_SHA256 and
                 publication["measurement_source"]["sha256"] == SOURCE_SHA256,
                 "Publication provenance differs from the fixed archive/source identity")
    checks.check(publication["exact_copies"] == EXACT_COPIES, "Publication CSV mapping differs")
    archive_path = root / "results.zip"
    checks.check(publication["archive"]["bytes"] == archive_path.stat().st_size, "Archive byte count differs")
    with tempfile.TemporaryDirectory(prefix="linear_perf_publication_") as temporary:
        manifest, launcher = verify_archive(archive_path, temporary, checks)
        verify_exact_copies(archive_path, root, EXACT_COPIES, checks)
        checks.check(publication["archive"]["manifest_file_count"] == len(manifest["files"]),
                     "Manifest size differs from publication metadata")
        checks.check(launcher["source_sha256"] == SOURCE_SHA256 and launcher["session_id"] == SESSION_ID and
                     launcher["simulate"] is False and launcher["worker"]["returncode"] == 0,
                     "Launcher provenance/completion differs")
        run = Path(temporary) / "run"
        summary = read_json(run / "summary.json")
        checks.check(summary["session_id"] == SESSION_ID and summary["status"] == "completed" and
                     summary["measurement_kind"] == "cuda_measured" and summary["required_stages_completed"] is True and
                     summary["all_requested_stages_completed"] is True, "Required experiment stages are not complete")
        checks.check(publication["completion"] == {key: summary[key] for key in publication["completion"]},
                     "Published completion/quality status differs from evidence")
        audits = {}
        datasets = {}
        expected_counts = {"primary": (609, 2256, 6), "repeat": (77, 280, 0)}
        for stage, (shapes, rows, flags) in expected_counts.items():
            audit = audit_dataset(run / stage)
            checks.check(audit["passed"] and audit["measurement_kind"] == "cuda_measured", f"{stage}: dataset audit failed")
            checks.check((audit["distinct_shapes"], audit["measurement_rows"]) == (shapes, rows), f"{stage}: dataset coverage differs")
            quality = audit["measurement_quality"]
            checks.check(quality == summary["measurement_quality"][stage], f"{stage}: quality summary differs")
            checks.check(len(quality["flagged_rows"]) == flags, f"{stage}: retained quality flags differ")
            audits[stage] = {"passed": audit["passed"], "record_validation": audit["record_validation"],
                             "quality_passed": quality["passed"], "flagged_rows": len(quality["flagged_rows"]),
                             "failing_control_groups": sum(not item["passed"] for item in quality["controls"])}
            datasets[stage] = read_jsonl(run / stage / "dataset" / "measurements.jsonl")
        checks.check(summary["measurement_quality_passed"] is False and audits["repeat"]["failing_control_groups"] == 1,
                     "Declared timing-quality qualifications were lost")
        models = verify_models(run, datasets["primary"], checks)
        repeat = verify_repeat(run, datasets["primary"], datasets["repeat"], checks)
    return {"status": "passed", "passed": True, "session_id": SESSION_ID, "checks": checks.checks,
            "archive_sha256": ARCHIVE_SHA256, "source_sha256": SOURCE_SHA256,
            "asset_files_verified": assets, "archive_members_verified": len(manifest["files"]),
            "dataset_audits": audits, "saved_models": models, "paired_repeat": repeat,
            "measurement_quality_passed": False, "embedded_source_executed": False,
            "scope": "Offline integrity, record consistency, saved-model replay and repeat comparison; no new GPU measurement, fitting, or hardware attestation."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION,
                        help="Directory containing this exact published result archive and readable summaries")
    args = parser.parse_args(argv)
    try:
        result = verify_publication(args.publication)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "failed", "passed": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
