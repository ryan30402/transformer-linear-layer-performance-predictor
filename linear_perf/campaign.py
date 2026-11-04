"""Preregistered sessions, an immutable holdout boundary, and evidence auditing.

Run `python -m linear_perf --help`. CUDA runs never fall back to
CPU or simulation. Simulation is an explicit software-validation mode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
import traceback
from pathlib import Path

from .common import (LabError, digest, read_json, read_jsonl, utc_now,
                     validate_measurements, write_json, write_jsonl)
from .protocol import validate_study_plan, protocol_signature


def implementation_identity() -> dict:
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob("*.py"))}
    versions = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"source_hash": digest(files), "sources": files,
            "analysis_versions": versions}


def _fresh_output(path: str | Path) -> Path:
    out = Path(path).resolve()
    if out.exists():
        raise LabError("Refusing to overwrite an existing session. Preserve it and use a new registered session ID.")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(out: Path, *, plan: dict, environment: dict, status: dict,
         implementation: dict, stage: str = "linear_perf_session") -> dict:
    if implementation_identity() != implementation:
        raise LabError("Program source or analysis dependencies changed during the session")
    out = Path(out).resolve()
    files = {}
    for path in sorted(out.rglob("*")):
        if path == out / "manifest.json":
            continue
        if path.is_symlink():
            raise LabError("Evidence cannot be sealed through a symbolic link")
        if path.is_file():
            if not path.resolve().is_relative_to(out):
                raise LabError("Evidence path escapes the session directory")
            files[path.relative_to(out).as_posix()] = _sha(path)
    manifest = {"schema_version": 1, "protocol_version": "1.0.0", "stage": stage,
                "sealed_at_utc": utc_now(), "plan_hash": digest(plan),
                "protocol_signature": protocol_signature(plan),
                "environment_id": environment.get("environment_id"),
                "measurement_kind": status["measurement_kind"],
                "session_id": status["session_id"], "status": status["status"],
                "implementation": implementation, "files": files,
                "scope": "Internal integrity and chronology; not independent hardware attestation"}
    write_json(out / "manifest.json", manifest, overwrite=False)
    return manifest


def _new_claim(path: Path, payload: dict) -> None:
    # A durable one-way boundary is recorded before any protected measurement.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _control_summary(records: list[dict], threshold: float) -> dict:
    groups = {}
    for record in records:
        row = record.get("row", record)
        cid, median = row.get("case_id"), row.get("median_ms")
        if cid is not None and isinstance(median, (int, float)) and math.isfinite(median) and median > 0:
            groups.setdefault(cid, []).append(float(median))
    cases = []
    for cid, values in sorted(groups.items()):
        ratio = max(values) / min(values)
        cases.append({"case_id": cid, "visits": len(values), "median_ms_by_visit": values,
                      "max_over_min": ratio, "passed": ratio <= threshold})
    return {"passed": bool(cases) and all(c["passed"] for c in cases),
            "cases": cases, "threshold": threshold,
            "interpretation": "Descriptive whole-session control variation; never used to rescale labels or select runs"}


def _frozen_shape_predictions(suite: dict, cases: list[dict]) -> dict:
    from .models import predict_model
    models = {}
    for name, model in suite["models"].items():
        values = predict_model(model, cases, suite["dtype"])
        models[name] = [{"case_id": case["id"], "predicted_ms": float(value)}
                        for case, value in zip(cases, values)]
        if len(models[name]) != len(cases):
            raise LabError("Prediction count differs from registered heldout shapes")
    return {"created_at_utc": utc_now(), "source_suite_hash": digest(suite),
            "selected_model_id": suite["selected_model_id"], "models": models,
            "labels_available": False, "case_ids": [c["id"] for c in cases]}


def _check_frozen_predictions(frozen: dict, evaluations: dict) -> None:
    for split_result in evaluations.values():
        for name, metrics in split_result["models"].items():
            expected = {r["case_id"]: r["predicted_ms"] for r in frozen["models"][name]}
            for row in metrics["predictions"]:
                if row["case_id"] not in expected or not math.isclose(
                    row["predicted_ms"], expected[row["case_id"]], rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise LabError("Evaluation predictions differ from the pre-holdout frozen predictions")


def _write_predictions(path: Path, evaluations: dict) -> None:
    rows = []
    for split, result in evaluations.items():
        for model_id, metrics in result["models"].items():
            for prediction in metrics["predictions"]:
                rows.append({"evaluation_split": split, "model_id": model_id,
                             "selected_on_development": model_id == result["selected_model_id"],
                             **prediction})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run_campaign(config: str | Path | dict, output: str | Path, *, session_id: str,
                 simulate: bool = False, device: str = "cuda:0", progress=print) -> dict:
    """Complete one bounded session. Independent sessions are explicit user runs.

    All model choices and offline acquisition checkpoints are frozen before the
    test/challenge stage. Every session is retained regardless of its scores.
    """
    from .measurement import make_environment, collect_stage, run_diagnostics
    from .models import fit_suite, evaluate_suite, fit_acquisition, evaluate_acquisition
    from .report import render_session_report

    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session_id):
        raise LabError("Use a unique 1–64 character alphanumeric/dash/underscore session ID")
    plan = validate_study_plan(copy.deepcopy(config) if isinstance(config, dict) else read_json(config))
    out = _fresh_output(output)
    started = time.monotonic()
    deadline = started + plan["budget"]["max_wall_seconds"]
    implementation = implementation_identity()
    status = {"protocol_version": "1.0.0", "session_id": session_id,
              "status": "started", "stage": "preflight", "started_at_utc": utc_now(),
              "measurement_kind": "simulated" if simulate else "cuda_measured",
              "test_opened": False, "automatic_retries": 0, "samples_removed": 0,
              "expected_independent_sessions": plan["study"]["expected_sessions"],
              "hardware_accuracy_validated": False, "industry_certification": False}
    write_json(out / "plan.json", plan, overwrite=False)
    write_json(out / "source_identity.json", implementation, overwrite=False)
    write_json(out / "status.json", status)
    journal = []
    environment = {"measurement_kind": status["measurement_kind"], "environment_id": None}

    def stage(name):
        if time.monotonic() >= deadline:
            raise LabError("Session deadline exhausted; retained data must not be silently retried")
        if implementation_identity() != implementation:
            raise LabError("Source/dependencies changed after the session was registered")
        status.update(stage=name, elapsed_seconds=time.monotonic() - started)
        journal.append({"stage": name, "at_utc": utc_now(), "monotonic_seconds": time.monotonic()})
        write_json(out / "journal.json", journal)
        write_json(out / "status.json", status)
        progress(f"[{session_id}] {name}")

    try:
        environment = make_environment(plan, session_id, simulate=simulate, device=device)
        write_json(out / "environment.json", environment, overwrite=False)
        stage("diagnostics_no_heldout")
        diagnostics = run_diagnostics(plan, environment, out / "diagnostics", session_id=session_id,
                                      device=device, deadline_monotonic=deadline, progress=progress)
        write_json(out / "diagnostics_summary.json", diagnostics, overwrite=False)

        stage("calibration_and_development")
        cases = [c for c in plan["cases"] if c["split"] in {"calibration", "development"}]
        initial = collect_stage(plan, cases, environment, out / "initial", session_id=session_id,
                                device=device, deadline_monotonic=deadline, progress=progress)
        initial_rows = initial["rows"]
        validate_measurements(initial_rows, plan, allowed_splits={"calibration", "development"})
        write_jsonl(out / "initial_measurements.jsonl", initial_rows, overwrite=False)
        train = [r for r in initial_rows if r["split"] == "calibration"]
        dev = [r for r in initial_rows if r["split"] == "development"]

        stage("fit_and_development_only_model_selection")
        suite = fit_suite(train, dev, plan, environment=environment)
        write_json(out / "model_suite.json", suite, overwrite=False)

        stage("fixed_candidate_pool_collection")
        pool_ids = set(plan["study"]["candidate_pool_ids"])
        pool = collect_stage(plan, [c for c in plan["cases"] if c["id"] in pool_ids], environment,
                             out / "candidate_pool", session_id=session_id, device=device,
                             deadline_monotonic=deadline, progress=progress)
        validate_measurements(pool["rows"], plan, allowed_splits={"candidate"})
        write_jsonl(out / "candidate_measurements.jsonl", pool["rows"], overwrite=False)
        stage("offline_acquisition_development_only")
        acquisition = fit_acquisition(pool["rows"], train, dev, plan, environment=environment)
        write_json(out / "acquisition_frozen.json", acquisition, overwrite=False)

        stage("freeze_models_and_predictions_before_heldout")
        heldout_cases = [c for c in plan["cases"] if c["split"] in {"test", "challenge"}]
        frozen_predictions = _frozen_shape_predictions(suite, heldout_cases)
        write_json(out / "frozen_predictions.json", frozen_predictions, overwrite=False)
        freeze = {"frozen_at_utc": utc_now(), "frozen_at_monotonic": time.monotonic(),
                  "plan_hash": digest(plan), "source_identity_hash": digest(implementation),
                  "model_suite_hash": digest(suite), "acquisition_hash": digest(acquisition),
                  "frozen_predictions_hash": digest(frozen_predictions),
                  "test_labels_available": False, "development_selection_closed": True}
        write_json(out / "freeze.json", freeze, overwrite=False)
        claim = {"claimed_at_utc": utc_now(), "claimed_at_monotonic": time.monotonic(),
                 "freeze_hash": digest(freeze), "case_ids": [c["id"] for c in heldout_cases],
                 "policy": "one fixed heldout stage; no test-driven fitting, selection, or retry"}
        _new_claim(out / "heldout_claim.json", claim)
        status["test_opened"] = True

        stage("frozen_test_and_challenge_collection")
        heldout = collect_stage(plan, heldout_cases, environment, out / "heldout", session_id=session_id,
                                device=device, deadline_monotonic=deadline, progress=progress)
        heldout_rows = heldout["rows"]
        validate_measurements(heldout_rows, plan, allowed_splits={"test", "challenge"})
        write_jsonl(out / "heldout_measurements.jsonl", heldout_rows, overwrite=False)
        stage("frozen_evaluation_no_retuning")
        evaluations = {split: evaluate_suite(suite, [r for r in heldout_rows if r["split"] == split],
                                             plan, target_environment=environment)
                       for split in ("test", "challenge")}
        _check_frozen_predictions(frozen_predictions, evaluations)
        acquisition_evaluations = {
            split: evaluate_acquisition(acquisition, [r for r in heldout_rows if r["split"] == split],
                                        plan, target_environment=environment)
            for split in ("test", "challenge")}
        write_json(out / "acquisition_results.json", acquisition_evaluations, overwrite=False)

        stage("separate_fixed_work_batching")
        if simulate:
            batching = {"measurement_kind": "simulated", "status": "skipped_no_gpu",
                        "reason": "Software validation does not invent a measured batching speedup"}
        else:
            from .benchmark import benchmark_batching
            batching = benchmark_batching(plan, device=device, deadline_monotonic=deadline)
        write_json(out / "batching.json", batching, overwrite=False)

        controls = initial.get("controls", []) + pool.get("controls", []) + heldout.get("controls", [])
        write_json(out / "controls.json", controls, overwrite=False)
        global_control = _control_summary(controls, plan["study"]["quality"]["control_ratio"])
        quality = {"initial": initial["quality"], "candidate_pool": pool["quality"],
                   "heldout": heldout["quality"], "whole_session_controls": global_control,
                   "rule": "Disclose all timing flags; never remove data or select a favorable session"}
        quality["passed"] = all(quality[k].get("passed") is True
                                for k in ("initial", "candidate_pool", "heldout", "whole_session_controls"))
        write_json(out / "quality.json", quality, overwrite=False)
        results = {"schema_version": 1, "protocol_version": "1.0.0", "session_id": session_id,
                   "measurement_kind": status["measurement_kind"], "plan_hash": digest(plan),
                   "environment_id": environment["environment_id"],
                   "selected_model_id": suite["selected_model_id"], "evaluations": evaluations,
                   "quality_passed": quality["passed"], "independent_sessions_observed": 1,
                   "claims": {"gpu_executed": not simulate, "accuracy_improvement_requires_new_gpu_data": simulate,
                              "cross_session_validated": False, "cross_gpu_validated": False,
                              "mlperf_compliant": False},
                   "evidence_counts": {"calibration": len(train), "development": len(dev),
                                       "candidate_pool": len(pool["rows"]), "heldout": len(heldout_rows),
                                       "registered_shapes": len(plan["cases"]), "control_visits": len(controls)},
                   "interpretation": "All models share training labels and heldout targets. Selection used development only. Separate acquisition comparisons use explicitly revealed candidate labels."}
        write_json(out / "results.json", results, overwrite=False)
        _write_predictions(out / "predictions.csv", evaluations)
        stage("report_and_integrity_review")
        status.update(status="simulated_complete" if simulate else (
            "completed_pending_review" if quality["passed"] else "completed_with_timing_qualifications"),
            all_timing_criteria_passed=quality["passed"], completed_at_utc=utc_now(),
            elapsed_seconds=time.monotonic() - started, stage="completed",
            hardware_accuracy_validated=False,
            result_scope="single session; perform planned independent sessions before generalizing")
        write_json(out / "status.json", status)
        render_session_report(out)
        seal(out, plan=plan, environment=environment, status=status, implementation=implementation)
        audit = audit_run(out)
        if not audit["passed"]:
            raise LabError("Post-seal audit rejected the session: " + "; ".join(audit["errors"]))
        progress(f"[{session_id}] complete; evidence audit passed; CUDA execution recorded={not simulate}; hardware accuracy claim still requires review")
        return {**status, "output": str(out), "audit": audit}
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc), failed_at_utc=utc_now(),
                      elapsed_seconds=time.monotonic() - started)
        # Do not mutate an already sealed result. An audit failure is saved next
        # to it so that its original file hashes remain reviewable.
        failure = {**status, "traceback": traceback.format_exc(),
                   "partial_evidence_retained": True, "automatic_retry": False}
        if (out / "manifest.json").exists():
            write_json(out.with_name(out.name + "-postseal-failure.json"), failure, overwrite=False)
        else:
            write_json(out / "status.json", status)
            write_json(out / "FAILED.json", failure, overwrite=False)
            if not (out / "environment.json").exists():
                write_json(out / "environment.json", environment, overwrite=False)
            try:
                seal(out, plan=plan, environment=environment, status=status, implementation=implementation)
            except Exception as seal_error:
                write_json(out / "SEAL_ERROR.json", {"error": str(seal_error)})
        progress(f"[{session_id}] stopped: {type(exc).__name__}: {exc}; partial evidence retained")
        return {**status, "output": str(out)}


def audit_run(path: str | Path) -> dict:
    """Fail-closed evidence audit, with CPU recomputation but no GPU attestation."""
    out = Path(path).resolve()
    errors, checks = [], 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    def equivalent(a, b):
        # Cross-platform floating-point recomputation may differ by roundoff.
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(equivalent(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
        if type(a) in (int, float) and type(b) in (int, float):
            return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
        return type(a) is type(b) and a == b

    def finite_time(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def visit_bounds(row, label, *, after=None):
        begin, end = (row.get(k) for k in ("visit_started_monotonic_seconds", "visit_finished_monotonic_seconds"))
        if not finite_time(begin) or not finite_time(end):
            raise LabError("Missing/invalid visit chronology: " + label)
        check(end >= begin, "Visit finished before it started: " + label)
        if after is not None:
            check(begin >= after, "Heldout visit preceded its frozen claim: " + label)
        check(row.get("session_id") == status["session_id"], "Visit/session mismatch: " + label)
        return begin, end

    def audit_witnesses(summary, *, require_complete):
        from .numerical_witness import audit_witness_npz
        from .collection import _checked_correctness
        folder = out / "diagnostics" / "numerical_witnesses"
        actual_summary = read_json(folder / "witnesses.json")
        check(summary == actual_summary, "Embedded/file numerical witness summaries differ")
        for key, value in (("plan_hash", digest(plan)), ("source_environment_id", env.get("environment_id")),
                           ("source_measurement_kind", status["measurement_kind"]),
                           ("measurement_kind", "numerical_witness"), ("excluded_from_model", True)):
            check(summary.get(key) == value, "Numerical witness provenance mismatch: " + key)
        records = summary.get("records")
        if not isinstance(records, list):
            raise LabError("Numerical witness records must be a list")
        names = [r.get("name") for r in records]
        check(len(names) == len(set(names)), "Duplicate numerical witnesses")
        witness_begin = summary.get("started_monotonic_seconds")
        witness_end = summary.get("completed_monotonic_seconds")
        if require_complete:
            check(summary.get("session_id") == status["session_id"] and summary.get("input_seed") == plan["seed"],
                  "Numerical witness session/seed identity differs")
            if not finite_time(witness_begin) or not finite_time(witness_end):
                raise LabError("Missing numerical witness completion chronology")
            check(witness_end >= witness_begin, "Numerical witness stage completion precedes its start")
        if status["measurement_kind"] == "simulated":
            check(summary.get("status") == "skipped" and summary.get("reason") == "simulation_no_GPU_witness"
                  and summary.get("arrays_archived") is False and records == [],
                  "Simulation must explicitly skip GPU numerical witnesses")
        elif require_complete:
            check(summary.get("status") == "completed" and summary.get("arrays_archived") is True
                  and set(names) == {"zero", "identity", "alternating", "random"},
                  "Completed CUDA diagnostics require four archived numerical witnesses")
        expected_files = set()
        for record in records:
            name, filename = record.get("name"), record.get("file")
            if name not in {"zero", "identity", "alternating", "random"} or filename != name + ".npz":
                raise LabError("Invalid numerical witness filename")
            candidate = folder / filename
            expected_files.add(filename)
            check(candidate.is_file() and _sha(candidate) == record.get("sha256"), "Numerical witness archive hash mismatch: " + name)
            recomputed = audit_witness_npz(candidate)
            check(recomputed["witness_name"] == name and recomputed["dtype"] == plan["dtype"],
                  "Numerical witness shape/precision identity differs: " + name)
            check(equivalent(recomputed, record.get("independent_cpu_recomputation")),
                  "Numerical witness stored CPU recomputation differs: " + name)
            _checked_correctness(record.get("gpu_check"))
            if require_complete:
                begin, end = (record.get(k) for k in ("visit_started_monotonic_seconds", "visit_finished_monotonic_seconds"))
                check(finite_time(begin) and finite_time(end) and witness_begin <= begin <= end <= witness_end,
                      "Numerical witness visit chronology differs: " + name)
        check({p.name for p in folder.glob("*.npz")} == expected_files, "Numerical witness archive inventory differs")

    def audit_diagnostics(*, require_complete):
        from .diagnostics import _diagnostic_plan
        diagnostics_path = out / "diagnostics" / "diagnostics.json"
        if not diagnostics_path.exists():
            check(not require_complete, "Missing completed diagnostic bundle")
            witness_file = out / "diagnostics" / "numerical_witnesses" / "witnesses.json"
            if witness_file.exists():
                audit_witnesses(read_json(witness_file), require_complete=False)
            return
        bundle = read_json(diagnostics_path)
        if require_complete:
            check(bundle.get("status") == "completed", "Diagnostic stage was not completed")
            check(read_json(out / "diagnostics_summary.json") == bundle, "Diagnostic summary differs from raw bundle")
        for key, value in (("session_id", status["session_id"]), ("source_plan_hash", digest(plan)),
                           ("source_environment_id", env.get("environment_id")),
                           ("source_measurement_kind", status["measurement_kind"]),
                           ("excluded_from_model", True), ("automatic_retries", 0),
                           ("main_timing_policy_changed_from_diagnostic_results", False)):
            check(bundle.get(key) == value, "Diagnostic provenance mismatch: " + key)
        for field, hash_field in (("records", "records_hash"), ("rotation_records", "rotation_records_hash"), ("schedule", "schedule_hash")):
            check(digest(bundle[field]) == bundle.get(hash_field), "Diagnostic internal hash mismatch: " + hash_field)
        check(bundle.get("schedule_declared_before_measurement") is True, "Diagnostic schedule was not declared")
        expected_variants = {f"{mode}_{iterations}": _diagnostic_plan(plan, mode, iterations)
                             for mode in ("eager", "cuda_graph") for iterations in plan["study"]["diagnostics"]["iterations"]}
        check(set(bundle["variants"]) == set(expected_variants), "Diagnostic variant coverage changed")
        for name, variant in expected_variants.items():
            saved = bundle["variants"][name]
            check(saved["plan"] == variant and saved["plan_hash"] == digest(variant), "Diagnostic variant plan/hash mismatch: " + name)
        expected_tasks = {(i, c, v) for i in range(1, plan["study"]["diagnostics"]["passes"] + 1)
                          for c in plan["study"]["diagnostic_case_ids"] for v in expected_variants}
        observed_tasks = [(r["pass_index"], r["case_id"], r["variant"]) for r in bundle["schedule"]]
        check(len(observed_tasks) == len(expected_tasks) and set(observed_tasks) == expected_tasks, "Diagnostic schedule is incomplete or duplicated")
        records = bundle["records"]
        check(len(records) <= len(bundle["schedule"]), "More diagnostic records than scheduled visits")
        if require_complete:
            check(len(records) == len(bundle["schedule"]), "Incomplete diagnostic sensitivity records")
        previous_end = None
        for index, record in enumerate(records):
            task = bundle["schedule"][index]
            check(all(record.get(k) == v for k, v in task.items()), "Diagnostic record does not follow its schedule")
            check(record.get("excluded_from_model") is True and record.get("source_plan_hash") == digest(plan), "Diagnostic record exclusion/source changed")
            check(record.get("source_environment_id") == env.get("environment_id"), "Diagnostic record source environment changed")
            variant = expected_variants[task["variant"]]
            variant_env = bundle["variant_environments"][task["variant"]]
            check(record.get("environment_id") == variant_env.get("environment_id")
                  and variant_env.get("plan_hash") == digest(variant)
                  and variant_env.get("session_id") == status["session_id"], "Diagnostic variant environment mismatch")
            restored = copy.deepcopy(record)
            restored["measurement_kind"] = restored["source_measurement_kind"]
            check(restored["measurement_kind"] == status["measurement_kind"], "Diagnostic measurement kind mismatch")
            validate_measurements([restored], variant, allowed_splits={"candidate"})
            begin, end = visit_bounds(record, "diagnostic/" + str(index))
            if previous_end is not None:
                check(begin >= previous_end, "Diagnostic visit timestamps overlap or reverse order")
            previous_end = end
        if require_complete:
            expected_rotation = {(i, c, b) for i in range(1, plan["study"]["diagnostics"]["passes"] + 1)
                                 for c in plan["study"]["diagnostics"]["rotation_case_ids"]
                                 for b in plan["study"]["diagnostics"]["rotation_buffer_counts"]}
            actual_rotation = [(r["pass_index"], r["case_id"], r["buffer_sets"]) for r in bundle["rotation_records"]]
            check(len(actual_rotation) == len(expected_rotation) and set(actual_rotation) == expected_rotation,
                  "Working-set rotation coverage changed")
            for row in bundle["rotation_records"]:
                check(row.get("excluded_from_model") is True, "Rotation measurement entered main model scope")
                visit_bounds(row, "rotation/" + row["case_id"])
            profiles = bundle["profiles"]
            check(len(profiles) == len(plan["study"]["diagnostics"]["profile_case_ids"])
                  and {p["case_id"] for p in profiles} == set(plan["study"]["diagnostics"]["profile_case_ids"]),
                  "Graph profiler case coverage changed")
            check(all(p.get("excluded_from_model") is True and p.get("execution_mode") == "cuda_graph" for p in profiles),
                  "Profiler scope/exclusion differs from declared CUDA Graph diagnostics")
            for profile in profiles:
                case_id = profile["case_id"]
                saved = read_json(out / "diagnostics" / "profiles" / case_id / "profile.json")
                check(saved == {k: v for k, v in profile.items() if k != "artifact_directory"},
                      "Embedded/file profiler records differ: " + case_id)
                check(profile.get("source_environment_id") == env.get("environment_id")
                      and profile.get("source_measurement_kind") == status["measurement_kind"]
                      and profile.get("plan_hash") == digest(plan)
                      and profile.get("timing_excluded_from_model") is True,
                      "Profiler source/timing provenance mismatch: " + case_id)
                visit_bounds(profile, "profile/" + case_id)
        if "numerical_witnesses" in bundle:
            audit_witnesses(bundle["numerical_witnesses"], require_complete=require_complete)
        elif require_complete:
            check(False, "Missing numerical witness completion/skip evidence")
        else:
            witness_file = out / "diagnostics" / "numerical_witnesses" / "witnesses.json"
            if witness_file.exists():
                audit_witnesses(read_json(witness_file), require_complete=False)

    def audit_collection(folder, rows, expected_cases, *, after=None):
        from .measurement import _schedule, quality_summary
        from .collection import _checked_correctness
        raw = read_json(out / folder / "raw_visits.json")
        for key, value in (("status", "completed"), ("session_id", status["session_id"]),
                           ("plan_hash", digest(plan)), ("environment_id", env["environment_id"]),
                           ("measurement_kind", status["measurement_kind"]),
                           ("raw_records_excluded_from_model", True), ("automatic_retries", 0),
                           ("schedule_declared_before_measurement", True)):
            check(raw.get(key) == value, f"{folder} raw provenance mismatch: {key}")
        orders, schedule = _schedule(plan, expected_cases, status["session_id"])
        check(raw["block_orders"] == orders and raw["schedule"] == schedule, folder + " schedule differs from registered seeded order")
        check(raw["planned_case_ids"] == [c["id"] for c in expected_cases], folder + " planned case IDs differ")
        for field in ("rows", "controls", "schedule"):
            check(digest(raw[field]) == raw.get(field + "_hash"), folder + " raw internal hash mismatch: " + field)
        check(raw["completed_visit_count"] == len(schedule), folder + " scheduled visit count differs")
        visits = raw["rows"] + raw["controls"]
        indexed = {r["visit_index"]: r for r in visits}
        check(len(visits) == len(schedule) and len(indexed) == len(visits)
              and set(indexed) == set(range(1, len(schedule) + 1)), folder + " missing/duplicate raw visit indices")
        first_begin, previous_end = None, None
        for task in schedule:
            row = indexed[task["visit_index"]]
            label = folder + "/" + str(task["visit_index"])
            check(row["case_id"] == task["case_id"] and row["environment_id"] == env["environment_id"], "Scheduled visit case/environment mismatch: " + label)
            begin, end = visit_bounds(row, label, after=after)
            if first_begin is None:
                first_begin = begin
            if previous_end is not None:
                check(begin >= previous_end, "Raw visit timestamps overlap or reverse order: " + label)
            previous_end = end
            if task["role"] == "control":
                check(row.get("excluded_from_model") is True and row.get("diagnostic_kind") == "interleaved_control"
                      and row.get("case_id") in plan["study"]["control_case_ids"], "Control role/identity mismatch: " + label)
                restored = copy.deepcopy(row)
                restored["measurement_kind"] = row["source_measurement_kind"]
                check(restored["measurement_kind"] == status["measurement_kind"], "Control measurement kind mismatch")
                validate_measurements([restored], plan, allowed_splits={"candidate"})
                _checked_correctness(restored["correctness"])
            else:
                check(row.get("block_index") == task["block_index"] and row.get("case_position") == task["position"]
                      and row.get("block_order_seed") == task["order_seed"], "Measurement block schedule mismatch: " + label)
        raw_blocks = {(r["case_id"], r["block_index"]): r for r in raw["rows"]}
        check(len(raw_blocks) == len(expected_cases) * plan["collection"]["blocks"], folder + " raw block coverage differs")
        for row in rows:
            check(row.get("session_id") == status["session_id"], "Aggregate/session mismatch: " + row["case_id"])
            blocks = row["measurement_blocks"]
            for block in blocks:
                check(block == raw_blocks.get((row["case_id"], block["block_index"])), "Aggregate block differs from retained raw evidence: " + row["case_id"])
                visit_bounds(block, folder + "/" + row["case_id"], after=after)
            bounds = row.get("visit_time_bounds")
            expected_bounds = {"first_started_monotonic_seconds": min(b["visit_started_monotonic_seconds"] for b in blocks),
                               "last_finished_monotonic_seconds": max(b["visit_finished_monotonic_seconds"] for b in blocks)}
            check(bounds == expected_bounds, "Aggregate visit bounds differ from raw blocks: " + row["case_id"])
        target_hash = digest([{"case_id": r["case_id"], "median_ms": r["median_ms"],
                              "measurement_blocks_hash": r["measurement_blocks_hash"]} for r in rows])
        check(raw.get("aggregate_targets_hash") == target_hash, folder + " aggregate target hash mismatch")
        quality = quality_summary(plan, rows, raw["controls"])
        check(equivalent(read_json(out / folder / "quality.json"), quality), folder + " quality summary differs from raw records")
        return raw["controls"], quality, (first_begin, previous_end)

    try:
        manifest = read_json(out / "manifest.json")
        check(manifest.get("protocol_version") == "1.0.0", "Not a registered artifact")
        listed = manifest.get("files")
        if not isinstance(listed, dict):
            raise LabError("Missing manifest file inventory")
        paths = [p for p in out.rglob("*") if p != out / "manifest.json"]
        check(not any(p.is_symlink() for p in paths), "Symbolic links are not sealed evidence")
        actual = {p.relative_to(out).as_posix() for p in paths if p.is_file()}
        check(actual == set(listed), "File inventory differs from the sealed manifest")
        for name, expected in listed.items():
            candidate = (out / name).resolve()
            check(candidate.is_relative_to(out) and candidate.is_file() and _sha(candidate) == expected,
                  "Artifact hash mismatch: " + name)
        plan = validate_study_plan(read_json(out / "plan.json"))
        env = read_json(out / "environment.json")
        status = read_json(out / "status.json")
        implementation = read_json(out / "source_identity.json")
        check(digest(plan) == manifest["plan_hash"] and protocol_signature(plan) == manifest.get("protocol_signature"), "Plan/protocol identity mismatch")
        check(env.get("environment_id") == manifest.get("environment_id"), "Environment identity mismatch")
        check(status["measurement_kind"] == manifest["measurement_kind"] == env.get("measurement_kind"), "Measurement kind mismatch")
        check(status["session_id"] == manifest.get("session_id"), "Status/manifest session mismatch")
        check(status["status"] == manifest.get("status"), "Status/manifest completion mismatch")
        check(implementation == manifest["implementation"], "Source identity mismatch")
        check(digest(implementation["sources"]) == implementation.get("source_hash"), "Internal source identity hash mismatch")
        complete = status["status"] in {"simulated_complete", "completed_pending_review", "completed_with_timing_qualifications"}
        diagnostic_complete = status["status"] == "diagnostics_completed"
        check(manifest.get("stage") in {"linear_perf_session", "linear_perf_diagnostics"}, "Unknown evidence stage")
        if diagnostic_complete:
            check(manifest.get("stage") == "linear_perf_diagnostics" and status.get("test_opened") is False,
                  "Diagnostic-only artifact opened a heldout/model stage")
        if complete:
            check(manifest.get("stage") == "linear_perf_session", "Main session has a diagnostic-only manifest")
        if complete or diagnostic_complete:
            check(env.get("session_id") == status["session_id"] and env.get("plan_hash") == digest(plan), "Environment/session/plan mismatch")
            identity = env.get("identity")
            if not isinstance(identity, dict):
                raise LabError("Completed session lacks its stable environment identity")
            expected_env = digest(identity)
            if status["measurement_kind"] == "simulated":
                expected_env = "simulated:" + expected_env
            check(env["environment_id"] == expected_env, "Internal environment identity hash mismatch")
            audit_diagnostics(require_complete=True)
        else:
            check(status["status"] in {"failed", "interrupted"}, "Unrecognized or unfinished sealed status")
            audit_diagnostics(require_complete=False)
        if complete:
            from . import models as model_api
            initial = read_jsonl(out / "initial_measurements.jsonl")
            pool = read_jsonl(out / "candidate_measurements.jsonl")
            heldout = read_jsonl(out / "heldout_measurements.jsonl")
            expected = {
                "initial": [c for c in plan["cases"] if c["split"] in {"calibration", "development"}],
                "candidate_pool": [c for c in plan["cases"] if c["id"] in set(plan["study"]["candidate_pool_ids"])],
                "heldout": [c for c in plan["cases"] if c["split"] in {"test", "challenge"}],
            }
            for name, rows, splits in (("initial", initial, {"calibration", "development"}),
                                       ("candidate_pool", pool, {"candidate"}), ("heldout", heldout, {"test", "challenge"})):
                validate_measurements(rows, plan, allowed_splits=splits)
                check({r["case_id"] for r in rows} == {c["id"] for c in expected[name]}, name + " case IDs differ from complete registered set")
                check(all(r["environment_id"] == env["environment_id"] and r.get("session_id") == status["session_id"] for r in rows), "Row environment/session mismatch: " + name)
            suite = read_json(out / "model_suite.json")
            acquisition = read_json(out / "acquisition_frozen.json")
            freeze, claim = read_json(out / "freeze.json"), read_json(out / "heldout_claim.json")
            frozen, results = read_json(out / "frozen_predictions.json"), read_json(out / "results.json")
            check(digest(suite) == freeze["model_suite_hash"], "Frozen model changed")
            check(digest(acquisition) == freeze["acquisition_hash"], "Frozen policy checkpoints changed")
            check(digest(frozen) == freeze["frozen_predictions_hash"], "Frozen predictions changed")
            check(digest(freeze) == claim["freeze_hash"], "Heldout claim/freeze mismatch")
            check(freeze["plan_hash"] == digest(plan) and freeze["source_identity_hash"] == digest(implementation), "Freeze provenance mismatch")
            if not finite_time(freeze.get("frozen_at_monotonic")) or not finite_time(claim.get("claimed_at_monotonic")):
                raise LabError("Missing/invalid freeze/claim chronology")
            check(freeze["frozen_at_monotonic"] <= claim["claimed_at_monotonic"], "Holdout preceded freeze")
            check(freeze.get("test_labels_available") is False and frozen.get("labels_available") is False
                  and freeze.get("development_selection_closed") is True, "Freeze does not preserve unavailable labels / closed selection")
            expected_heldout = {c["id"] for c in expected["heldout"]}
            check(set(claim["case_ids"]) == set(frozen["case_ids"]) == expected_heldout
                  and len(claim["case_ids"]) == len(frozen["case_ids"]) == len(expected_heldout), "Frozen/claimed heldout coverage differs")
            check(set(frozen["models"]) == set(suite["models"]), "Frozen prediction model coverage differs")
            for model_id, predictions in frozen["models"].items():
                check(len(predictions) == len(expected_heldout)
                      and {r["case_id"] for r in predictions} == expected_heldout,
                      "Frozen prediction case coverage differs: " + model_id)
            controls, stage_quality, times = [], {}, {}
            for name, rows in (("initial", initial), ("candidate_pool", pool), ("heldout", heldout)):
                control_rows, quality, bounds = audit_collection(name, rows, expected[name], after=claim["claimed_at_monotonic"] if name == "heldout" else None)
                controls.extend(control_rows)
                stage_quality[name], times[name] = quality, bounds
            check(times["initial"][1] <= times["candidate_pool"][0], "Candidate collection preceded initial collection completion")
            check(times["candidate_pool"][1] <= freeze["frozen_at_monotonic"], "Model freeze preceded candidate-pool collection completion")
            train = sorted((r for r in initial if r["split"] == "calibration"), key=lambda r: r["case_id"])
            dev = sorted((r for r in initial if r["split"] == "development"), key=lambda r: r["case_id"])
            train_hash, dev_hash, pool_hash = (digest(model_api._training_evidence(rows)) for rows in (train, dev, pool))
            check(set(suite["train_case_ids"]) == {r["case_id"] for r in train}, "Main model training IDs differ from calibration")
            check(set(suite["development_case_ids"]) == {r["case_id"] for r in dev}, "Main model development IDs differ")
            check(suite["training_data_hash"] == train_hash and suite["development_data_hash"] == dev_hash, "Suite data hashes differ from retained labels")
            check(suite["source_environment"] == env and acquisition["source_environment"] == env, "Model source environment differs from session")
            check(suite["selected_model_id"] == results["selected_model_id"] == frozen["selected_model_id"], "Selected model changed after freeze")
            check(frozen["source_suite_hash"] == digest(suite), "Frozen predictions refer to different model suite")
            dev_scores = {name: model_api._score(model, dev, plan["dtype"]) for name, model in suite["models"].items()}
            check(equivalent(dev_scores, suite["development_metrics"]), "Saved development metrics differ from recomputation")
            selected = min(model_api.MODEL_ORDER, key=lambda name: model_api._rank_key(dev_scores[name], model_api.MODEL_ORDER.index(name)))
            check(suite["selected_model_id"] == selected, "Model selection differs from declared development rule")
            recalculated = {split: model_api.evaluate_suite(suite, [r for r in heldout if r["split"] == split], plan, target_environment=env) for split in ("test", "challenge")}
            _check_frozen_predictions(frozen, recalculated)
            check(equivalent(recalculated, results["evaluations"]), "Saved heldout metrics/groups differ from full recomputation")
            model_api._check_seal(acquisition, "acquisition_hash")
            for key, value in (("training_data_hash", train_hash), ("development_data_hash", dev_hash), ("candidate_pool_data_hash", pool_hash)):
                check(acquisition.get(key) == value, "Acquisition retained data hash mismatch: " + key)
            all_fit_rows = {r["case_id"]: r for r in train + pool}

            def check_acquisition_model(model):
                model_api._check_seal(model, "model_hash")
                ids = model["train_case_ids"]
                check(len(ids) == len(set(ids)) and set(ids) <= set(all_fit_rows), "Acquisition model contains unknown/duplicate training IDs")
                check(model["training_data_hash"] == digest(model_api._training_evidence([all_fit_rows[i] for i in ids])), "Acquisition model training hash differs from revealed labels")
                return model_api._score(model, dev, plan["dtype"])

            initial_score = check_acquisition_model(acquisition["initial_model"])
            check(equivalent(initial_score, acquisition["initial_development_metrics"]), "Acquisition baseline development score differs")
            for run in acquisition["runs"]:
                best, score = acquisition["initial_model"], initial_score
                check(len(run["trials"]) == max(acquisition["budgets"]) // acquisition["batch_size"],
                      "Acquisition round log is incomplete or duplicated")
                for index, trial in enumerate(run["trials"], 1):
                    count = index * acquisition["batch_size"]
                    revealed = run["label_revelation_order"][:count]
                    check(trial["round"] == index and trial["acquired_count"] == count
                          and trial["revealed_case_ids"] == revealed
                          and trial["selected_case_ids"] == revealed[-acquisition["batch_size"]:], "Acquisition trial/revelation chronology differs")
                    candidate = trial["candidate_model"]
                    check(set(candidate["train_case_ids"]) == {r["case_id"] for r in train} | set(revealed), "Acquisition trial did not use exactly its permitted labels")
                    candidate_score = check_acquisition_model(candidate)
                    check(equivalent(candidate_score, trial["development_metrics"]), "Acquisition trial saved score differs")
                    accepted = candidate_score["median_ape_pct"] < score["median_ape_pct"]
                    check(trial["accepted"] is accepted and equivalent(trial["previous_best_development_median_ape_pct"], score["median_ape_pct"]), "Acquisition acceptance violates frozen development rule")
                    if accepted:
                        best, score = candidate, candidate_score
                    for checkpoint in (c for c in run["checkpoints"] if c["budget"] == count):
                        check(checkpoint["accepted_model"] == best and equivalent(checkpoint["development_metrics"], score), "Acquisition checkpoint differs from prior accepted model")
                        check_acquisition_model(checkpoint["accepted_model"])
            acquisition_recomputed = {split: model_api.evaluate_acquisition(acquisition, [r for r in heldout if r["split"] == split], plan, target_environment=env) for split in ("test", "challenge")}
            check(equivalent(acquisition_recomputed, read_json(out / "acquisition_results.json")), "Saved acquisition results differ from frozen-checkpoint recomputation")
            check(read_json(out / "controls.json") == controls, "Whole-session controls differ from stage raw controls")
            global_control = _control_summary(controls, plan["study"]["quality"]["control_ratio"])
            quality = read_json(out / "quality.json")
            for name in stage_quality:
                check(equivalent(quality[name], stage_quality[name]), "Whole-session/stage quality differs: " + name)
            check(equivalent(quality["whole_session_controls"], global_control), "Whole-session control summary differs")
            quality_passed = all(q["passed"] for q in stage_quality.values()) and global_control["passed"]
            check(quality.get("passed") is quality_passed and results.get("quality_passed") is quality_passed
                  and status.get("all_timing_criteria_passed") is quality_passed, "Quality qualification differs from retained measurements")
            check(status.get("automatic_retries") == 0 and status.get("samples_removed") == 0 and status.get("test_opened") is True, "Unexpected retry/deletion/holdout status")
            check(results["session_id"] == status["session_id"] and results["environment_id"] == env["environment_id"]
                  and results["plan_hash"] == digest(plan) and results["measurement_kind"] == status["measurement_kind"], "Result provenance mismatch")
            check(results["claims"]["gpu_executed"] == (status["measurement_kind"] == "cuda_measured"), "Simulation/GPU claims mismatch")
        return {"passed": not errors, "checks": checks, "errors": errors,
                "complete_session": complete, "diagnostics_complete": diagnostic_complete,
                "measurement_kind": status["measurement_kind"],
                "scope": "Sealed files, complete case coverage, raw aggregation, visit chronology, frozen models, acquisition and CPU numerical witness recomputation; no independent GPU attestation"}
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
        return {"passed": False, "checks": checks, "errors": errors, "complete_session": False, "diagnostics_complete": False}


def run_diagnostic_only(config, output, *, session_id, simulate=False, device="cuda:0", progress=print):
    from .measurement import make_environment, run_diagnostics
    plan = validate_study_plan(read_json(config))
    out = _fresh_output(output)
    implementation = implementation_identity()
    status = {"status": "started", "session_id": session_id,
              "measurement_kind": "simulated" if simulate else "cuda_measured", "test_opened": False}
    environment = {"environment_id": None, "measurement_kind": status["measurement_kind"]}
    write_json(out / "plan.json", plan)
    write_json(out / "source_identity.json", implementation)
    try:
        environment = make_environment(plan, session_id, simulate=simulate, device=device)
        summary = run_diagnostics(plan, environment, out / "diagnostics", session_id=session_id,
                                  device=device, deadline_monotonic=time.monotonic() + plan["budget"]["max_wall_seconds"],
                                  progress=progress)
        write_json(out / "diagnostics_summary.json", summary)
        status["status"] = "diagnostics_completed"
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    write_json(out / "environment.json", environment)
    write_json(out / "status.json", status)
    seal(out, plan=plan, environment=environment, status=status, implementation=implementation,
         stage="linear_perf_diagnostics")
    return {**status, "output": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/fp16_graph.json")
        p.add_argument("--out", required=True)
        p.add_argument("--session-id", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--simulate", action="store_true", help="Explicit synthetic software validation; no GPU conclusions")
    p = sub.add_parser("audit")
    p.add_argument("--run", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--config", default="configs/fp16_graph.json")
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("study")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("transfer")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("session", "gpu"), required=True)
    p = sub.add_parser("compare-dtypes")
    p.add_argument("--runs", nargs=2, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True, help="Separate report directory; sealed runs are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "diagnose"}:
            function = run_campaign if args.command == "run" else run_diagnostic_only
            result = function(args.config, args.out, session_id=args.session_id,
                              simulate=args.simulate, device=args.device)
        elif args.command == "audit":
            result = audit_run(args.run)
        elif args.command == "doctor":
            from .measurement import make_environment
            plan = validate_study_plan(read_json(args.config))
            result = make_environment(plan, "preflight", device=args.device)
        elif args.command == "report":
            from .report import render_session_report
            audit = audit_run(args.run)
            if not audit["passed"] or not audit["complete_session"]:
                raise LabError("Only audited complete sessions can be re-reported")
            destination = _fresh_output(args.out)
            render_session_report(Path(args.run), destination=destination)
            result = {"status": "report_created", "output": str(destination)}
        else:
            from .study import compare_sessions, compare_transfer, compare_dtypes
            if args.command == "study":
                result = compare_sessions(args.runs, args.out)
            elif args.command == "transfer":
                result = compare_transfer(args.source, args.target, args.out, mode=args.mode)
            else:
                result = compare_dtypes(args.runs, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("status") in {"failed", "interrupted"} or result.get("passed") is False else 0
    except (LabError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
