#!/usr/bin/env python3
"""Verify the released measured-data snapshot on a CPU.

Checks exported-file integrity, raw-repeat aggregation, registered coverage,
recorded freeze chronology, frozen predictions, development-only selection,
reported errors, control drift, numerical witnesses, acquisition checkpoints,
and the cross-session summary tables. No GPU execution or fitting is performed.

Recorded acquisition hashes are provenance, not hashes of normalized metadata.
This verifier does not reconstruct omitted original source/archive bytes and
does not claim independent attestation of GPU execution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from linear_perf import models  # noqa: E402
from linear_perf.common import digest  # noqa: E402
from linear_perf.numerical_witness import audit_witness_npz  # noqa: E402

from export_results import SESSION_IDS, numeric_digest  # noqa: E402


class Verification:
    def __init__(self) -> None:
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def close(self, observed, expected, message: str, *, atol=1e-10, rtol=1e-9) -> None:
        self.check(bool(np.allclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False)), message)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def replay_envelope(recorded_model: dict) -> dict:
    """Make a temporary runtime envelope around unchanged recorded parameters.

    Publication metadata differs from acquisition metadata. A fresh in-memory
    checksum permits the current prediction API to replay these parameters;
    it is never saved, never substituted for a recorded freeze seal, and never
    used as evidence of pre-measurement chronology. This function does not fit.
    """
    replay = copy.deepcopy(recorded_model)
    if "base_model" in replay:
        replay["base_model"] = replay_envelope(replay["base_model"])
    replay["protocol_version"] = models.VERSION
    replay.pop("model_hash", None)
    replay["model_hash"] = digest(replay)
    return replay


METRICS = (
    "n", "median_ape_pct", "p90_ape_pct", "max_ape_pct", "mean_ape_pct",
    "rmse_ms", "rmse_us", "mae_ms", "mae_us", "signed_bias_ms", "signed_bias_us",
    "log_rmse", "geometric_mean_multiplicative_error", "pairwise_ordering_accuracy",
    "ranking_pairs", "excluded_actual_ties",
)


def verify_metrics(v: Verification, recorded: dict, rows: list[dict], predictions: list[float], label: str) -> dict:
    actual = np.asarray([row["median_ms"] for row in rows])
    predicted = np.asarray(predictions)
    v.check(bool(np.isfinite(actual).all() and np.isfinite(predicted).all() and
                 (actual > 0).all() and (predicted > 0).all()), f"{label}: invalid times")
    computed = models.evaluate_predictions(rows, predictions)
    # Independently recompute headline errors from the observed/frozen numbers.
    ape = np.abs(predicted - actual) / actual * 100
    v.close(computed["median_ape_pct"], np.median(ape), f"{label}: independent median APE")
    v.close(computed["p90_ape_pct"], np.percentile(ape, 90), f"{label}: independent P90 APE")
    v.close(computed["rmse_ms"], np.sqrt(np.mean((predicted - actual) ** 2)), f"{label}: independent RMSE")
    v.check(len(recorded["predictions"]) == len(rows), f"{label}: prediction count")
    recorded_by_id = {row["case_id"]: row for row in recorded["predictions"]}
    v.check(len(recorded_by_id) == len(rows), f"{label}: duplicate prediction")
    for row, prediction in zip(rows, predictions):
        item = recorded_by_id[row["case_id"]]
        v.close(item["actual_ms"], row["median_ms"], f"{label}/{row['case_id']}: measured label")
        v.close(item["predicted_ms"], prediction, f"{label}/{row['case_id']}: predicted label")
        v.close(item["ape_pct"], abs(prediction - row["median_ms"]) / row["median_ms"] * 100,
                f"{label}/{row['case_id']}: APE")
    for metric in METRICS:
        v.close(recorded[metric], computed[metric], f"{label}: {metric}")
    return computed


def check_repeat_row(v: Verification, row: dict, label: str, *, main: bool) -> None:
    samples = row["samples_ms"]
    v.check(len(samples) == 9 and all(math.isfinite(x) and x > 0 for x in samples), f"{label}: 9 positive event means")
    v.close(row["median_ms"], np.median(samples), f"{label}: repeat median")
    v.check(row.get("correctness", row.get("correctness_before", {})).get("passed") is True,
            f"{label}: recorded correctness")
    v.check(row["visit_started_monotonic_seconds"] < row["visit_finished_monotonic_seconds"],
            f"{label}: visit chronology")
    if main:
        v.check(row["timing"]["iterations"] == 50 and row["execution_mode"] == "cuda_graph",
                f"{label}: graph timing contract")
        v.check(row["warmup"]["completed_replays"] >= 5 and row["warmup"]["actual_wall_seconds"] >= 0.25,
                f"{label}: duration/replay warmup contract")


def verify_session(v: Verification, directory: Path) -> dict:
    sid = directory.name
    start = v.checks
    plan = read_json(directory / "plan.json")
    env = read_json(directory / "environment.json")
    suite = read_json(directory / "model_suite.json")
    frozen = read_json(directory / "frozen_predictions.json")
    freeze = read_json(directory / "freeze.json")
    claim = read_json(directory / "heldout_claim.json")
    results = read_json(directory / "results.json")
    quality = read_json(directory / "quality.json")
    controls = read_json(directory / "controls.json")
    study = plan["study"]
    dtype = plan["dtype"]
    cases = {case["id"]: case for case in plan["cases"]}
    diagnostic_ids = set(study["diagnostic_case_ids"])
    main_cases = {key: value for key, value in cases.items() if key not in diagnostic_ids}
    v.check(len(cases) == 104 and len(main_cases) == 96 and len(diagnostic_ids) == 8, f"{sid}: registered coverage")
    v.check(Counter(item["split"] for item in main_cases.values()) ==
            Counter(calibration=24, development=12, candidate=24, test=24, challenge=12), f"{sid}: split sizes")
    v.check(len({(case["m"], case["n"], case["k"]) for case in cases.values()}) == 104,
            f"{sid}: unique registered dimensions")
    v.check(env["measurement_kind"] == "cuda_measured" and env["session_id"] == sid,
            f"{sid}: measured session identity")
    rows = []
    raw_repeat_count = 0
    max_cv = 0.0
    max_block_ratio = 0.0
    heldout_first_start = math.inf
    for stage, filename in (("initial", "initial_measurements.jsonl"),
                            ("candidate_pool", "candidate_measurements.jsonl"),
                            ("heldout", "heldout_measurements.jsonl")):
        aggregate = read_jsonl(directory / filename)
        raw = read_json(directory / stage / "raw_visits.json")
        v.check(raw["schedule_declared_before_measurement"] is True, f"{sid}/{stage}: declared schedule")
        expected_splits = {"initial": {"calibration", "development"}, "candidate_pool": {"candidate"},
                           "heldout": {"test", "challenge"}}[stage]
        expected_ids = {cid for cid, case in main_cases.items() if case["split"] in expected_splits}
        v.check({row["case_id"] for row in aggregate} == expected_ids and len(aggregate) == len(expected_ids),
                f"{sid}/{stage}: exact aggregate coverage")
        raw_by_case = defaultdict(list)
        for row in raw["rows"]:
            raw_by_case[row["case_id"]].append(row)
            check_repeat_row(v, row, f"{sid}/{stage}/{row['case_id']}", main=True)
            raw_repeat_count += len(row["samples_ms"])
            cv = float(np.std(row["samples_ms"], ddof=1) / np.mean(row["samples_ms"]) * 100)
            max_cv = max(max_cv, cv)
            if stage == "heldout":
                heldout_first_start = min(heldout_first_start, row["visit_started_monotonic_seconds"])
        v.check(len(raw["rows"]) == 3 * len(expected_ids), f"{sid}/{stage}: three visits per shape")
        for row in aggregate:
            cid = row["case_id"]
            blocks = row["measurement_blocks"]
            v.check(len(blocks) == 3 and len(raw_by_case[cid]) == 3, f"{sid}/{cid}: three blocks")
            v.check(all(block in raw_by_case[cid] for block in blocks), f"{sid}/{cid}: raw block linkage")
            medians = [block["median_ms"] for block in blocks]
            v.close(row["samples_ms"], medians, f"{sid}/{cid}: block medians")
            v.close(row["median_ms"], np.median(medians), f"{sid}/{cid}: median of block medians")
            max_block_ratio = max(max_block_ratio, max(medians) / min(medians))
            v.check(row["sample_unit"] == "block_median", f"{sid}/{cid}: target sampling unit")
            v.check(row["correctness"]["passed"] is True, f"{sid}/{cid}: all blocks correct")
            v.check(all(row[key] == cases[cid][key] for key in ("m", "n", "k", "split")), f"{sid}/{cid}: shape registration")
            q = quality[stage]["per_case"][cid]
            v.close(q["block_medians_ms"], medians, f"{sid}/{cid}: quality medians")
            v.close(q["max_over_min_block_median"], max(medians) / min(medians), f"{sid}/{cid}: quality span")
        for control in raw["controls"]:
            v.check(control in controls, f"{sid}/{stage}: control retained in whole-session record")
        rows.extend(aggregate)
    by_id = {row["case_id"]: row for row in rows}
    v.check(len(rows) == len(by_id) == 96 and raw_repeat_count == 2592, f"{sid}: complete measured labels/event means")

    control_by_id = defaultdict(list)
    for control in controls:
        check_repeat_row(v, control, f"{sid}/control/{control['case_id']}", main=True)
        control_by_id[control["case_id"]].append(control["median_ms"])
        v.check(control["excluded_from_model"] is True, f"{sid}: control exclusion")
    v.check(len(controls) == 120 and set(control_by_id) == set(study["control_case_ids"]), f"{sid}: control coverage")
    control_ratios = {cid: max(values) / min(values) for cid, values in control_by_id.items()}
    for item in quality["whole_session_controls"]["cases"]:
        v.close(item["max_over_min"], control_ratios[item["case_id"]], f"{sid}: recorded control span")
    v.check(max_cv <= 10 and max_block_ratio <= 1.2 and max(control_ratios.values()) <= 1.2,
            f"{sid}: registered timing criteria")
    v.check(quality["passed"] is True and results["quality_passed"] is True, f"{sid}: recorded quality outcome")

    train_ids = {cid for cid, row in by_id.items() if row["split"] == "calibration"}
    dev_ids = {cid for cid, row in by_id.items() if row["split"] == "development"}
    heldout_ids = {cid for cid, row in by_id.items() if row["split"] in {"test", "challenge"}}
    v.check(set(suite["train_case_ids"]) == train_ids and set(suite["development_case_ids"]) == dev_ids,
            f"{sid}: fitting/selection isolation")
    v.check(set(suite["models"]) == set(models.MODEL_ORDER), f"{sid}: five registered models")
    v.check(set(frozen["case_ids"]) == heldout_ids and set(claim["case_ids"]) == heldout_ids, f"{sid}: frozen coverage")
    v.check(frozen["labels_available"] is False and freeze["test_labels_available"] is False and
            freeze["development_selection_closed"] is True, f"{sid}: recorded freeze policy")
    v.check(freeze["frozen_at_monotonic"] < claim["claimed_at_monotonic"] < heldout_first_start,
            f"{sid}: monotonic freeze before heldout work")
    v.check(frozen["created_at_utc"] <= freeze["frozen_at_utc"] < claim["claimed_at_utc"], f"{sid}: UTC freeze chronology")
    selected = min(models.MODEL_ORDER, key=lambda mid: (
        suite["development_metrics"][mid]["median_ape_pct"],
        suite["development_metrics"][mid]["p90_ape_pct"], models.MODEL_ORDER.index(mid)))
    v.check(selected == suite["selected_model_id"] == frozen["selected_model_id"] == results["selected_model_id"],
            f"{sid}: development-only model choice retained")
    metrics = {}
    for model_id, recorded_model in suite["models"].items():
        v.check(set(recorded_model["train_case_ids"]) == train_ids, f"{sid}/{model_id}: common training labels")
        replay_model = replay_envelope(recorded_model)
        for split in ("development", "test", "challenge"):
            split_rows = sorted((row for row in rows if row["split"] == split), key=lambda row: row["case_id"])
            predicted = models.predict_model(replay_model, split_rows, dtype)
            if split == "development":
                recorded_metrics = suite["development_metrics"][model_id]
            else:
                frozen_by_id = {item["case_id"]: item["predicted_ms"] for item in frozen["models"][model_id]}
                v.check(set(frozen_by_id) == heldout_ids, f"{sid}/{model_id}: frozen shape set")
                v.close(predicted, [frozen_by_id[row["case_id"]] for row in split_rows],
                        f"{sid}/{model_id}/{split}: parameters reproduce frozen predictions")
                recorded_metrics = results["evaluations"][split]["models"][model_id]
            metrics[(split, model_id)] = verify_metrics(v, recorded_metrics, split_rows, predicted,
                                                       f"{sid}/{split}/{model_id}")

    # Family-cluster resampling uses one error per shape, never event repeats.
    for split in ("test", "challenge"):
        evaluation = results["evaluations"][split]
        v.check(evaluation["heldout_used_for_fitting_or_selection"] is False, f"{sid}/{split}: frozen evaluation")
        a = evaluation["models"][selected]
        b = evaluation["models"]["roofline_3p"]
        groups = defaultdict(list)
        for index, row in enumerate(a["predictions"]):
            groups[f"nk_{row['n']}_{row['k']}"].append(index)
        labels = sorted(groups)
        recorded = evaluation["selected_vs_roofline_3p"]
        rng = np.random.default_rng(recorded["seed"])
        ape_a = np.asarray([row["ape_pct"] for row in a["predictions"]])
        ape_b = np.asarray([row["ape_pct"] for row in b["predictions"]])
        differences = []
        for sample in rng.integers(0, len(labels), size=(1000, len(labels))):
            indexes = np.concatenate([groups[labels[index]] for index in sample])
            differences.append(float(np.median(ape_a[indexes]) - np.median(ape_b[indexes])))
        v.close(recorded["percentile_95_interval_pp"], np.percentile(differences, [2.5, 97.5]), f"{sid}/{split}: family bootstrap")
        v.check(recorded["n_shape_families"] == len(labels) and recorded["resamples"] == 1000,
                f"{sid}/{split}: bootstrap sampling units")

    diagnostics = read_json(directory / "diagnostics" / "diagnostics.json")
    v.check(len(diagnostics["records"]) == 96 and len(diagnostics["rotation_records"]) == 8 and
            len(diagnostics["profiles"]) == 2, f"{sid}: diagnostic coverage")
    v.check(diagnostics["excluded_from_model"] is True and
            diagnostics["main_timing_policy_changed_from_diagnostic_results"] is False, f"{sid}: diagnostic isolation")
    for row in diagnostics["records"]:
        check_repeat_row(v, row, f"{sid}/diagnostic/{row['case_id']}", main=False)
    witnesses = read_json(directory / "diagnostics" / "numerical_witnesses" / "witnesses.json")
    v.check({row["name"] for row in witnesses["records"]} == {"zero", "identity", "alternating", "random"},
            f"{sid}: four witness patterns")
    for record in witnesses["records"]:
        path = directory / "diagnostics" / "numerical_witnesses" / record["file"]
        v.check(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], f"{sid}: witness bytes")
        checked = audit_witness_npz(path)
        v.check(checked["passed"] is True, f"{sid}/{record['name']}: independent FP64 recomputation")
        v.close(checked["fp64_relative_l2_error"], record["independent_cpu_recomputation"]["fp64_relative_l2_error"],
                f"{sid}/{record['name']}: witness error record")

    acquisition = read_json(directory / "acquisition_frozen.json")
    acquisition_results = read_json(directory / "acquisition_results.json")
    v.check(len(acquisition["runs"]) == 20 and acquisition["budgets"] == [2, 4, 8, 12], f"{sid}: acquisition menu")
    v.check(acquisition["protocol"]["test_labels_used"] is False and
            acquisition["protocol"]["live_gpu_closed_loop"] is False, f"{sid}: offline acquisition scope")
    for run in acquisition["runs"]:
        order = run["label_revelation_order"]
        v.check(len(order) == len(set(order)) == 12 and set(order) <= set(study["candidate_pool_ids"]),
                f"{sid}/{run['policy']}: candidate revelation order")
        for trial in run["trials"]:
            expected_acceptance = (trial["development_metrics"]["median_ape_pct"] <
                                   trial["previous_best_development_median_ape_pct"])
            v.check(trial["accepted"] == expected_acceptance, f"{sid}/{run['policy']}: development acceptance")
        for checkpoint in run["checkpoints"]:
            budget = checkpoint["budget"]
            v.check(checkpoint["revealed_case_ids"] == order[:budget], f"{sid}/{run['policy']}: checkpoint label budget")
            model = replay_envelope(checkpoint["accepted_model"])
            v.check(set(model["train_case_ids"]) <= train_ids | set(order[:budget]), f"{sid}: acquisition fitting isolation")
            for split in ("test", "challenge"):
                split_rows = sorted((row for row in rows if row["split"] == split), key=lambda row: row["case_id"])
                recorded = next(item for item in acquisition_results[split]["runs"]
                                if item["policy"] == run["policy"] and item["seed"] == run["seed"] and item["budget"] == budget)
                computed = models.evaluate_predictions(split_rows, models.predict_model(model, split_rows, dtype))
                for metric in ("median_ape_pct", "p90_ape_pct", "rmse_us", "pairwise_ordering_accuracy"):
                    v.close(recorded["metrics"][split][metric], computed[metric], f"{sid}/{run['policy']}/{budget}/{split}: {metric}")

    batching = read_json(directory / "batching.json")
    for variant in ("small_gemms", "one_large_gemm"):
        v.close(batching[variant]["median_ms"], np.median(batching[variant]["samples_ms"]), f"{sid}: offline batching median")
        v.check(batching[variant]["correctness"]["passed"] is True, f"{sid}: offline batching correctness")
    v.close(batching["speedup_small_over_large"], batching["small_gemms"]["median_ms"] / batching["one_large_gemm"]["median_ms"],
            f"{sid}: offline equal-work ratio")
    return {"session_id": sid, "checks": v.checks - start, "rows": rows, "suite": suite, "metrics": metrics,
            "environment": env, "selected": selected, "quality": quality, "acquisition_results": acquisition_results,
            "max_cv": max_cv, "max_block_ratio": max_block_ratio,
            "control_ratios": control_ratios, "batching": batching, "raw_event_means": raw_repeat_count}


def verify_summaries(v: Verification, results_dir: Path, sessions: dict) -> None:
    summary_dir = results_dir / "summary"
    selected_rows = csv_rows(summary_dir / "selected_pipeline.csv")
    v.check(len(selected_rows) == 6, "Summary: six official selections")
    for row in selected_rows:
        session = sessions[row["session_id"]]
        selected = session["selected"]
        v.check(row["selected_model_id"] == selected, "Summary: official model selection")
        for split in ("development", "test", "challenge"):
            v.close(float(row[f"{split}_median_ape_pct"]), session["metrics"][(split, selected)]["median_ape_pct"],
                    "Summary: selected median APE")
    rows_by_session = {sid: {row["case_id"]: row for row in data["rows"]} for sid, data in sessions.items()}
    latency_table = csv_rows(summary_dir / "case_latencies.csv")
    v.check(len(latency_table) == 576, "Summary: six times 96 shape labels")
    for row in latency_table:
        measured = rows_by_session[row["session_id"]][row["case_id"]]
        v.close(float(row["median_latency_us"]), measured["median_ms"] * 1000, "Summary: case latency")
        v.check(all(int(row[axis]) == measured[axis] for axis in ("m", "n", "k")), "Summary: case shape")
    metric_table = csv_rows(summary_dir / "model_metrics.csv")
    v.check(len(metric_table) == 90, "Summary: five models by three splits by six sessions")
    for row in metric_table:
        session = sessions[row["session_id"]]
        metric = session["metrics"][(row["split"], row["model_id"])]
        v.check((row["selected_by_development"] == "True") == (session["selected"] == row["model_id"]),
                "Summary: selected candidate flag")
        for key in ("median_ape_pct", "p90_ape_pct", "max_ape_pct", "rmse_us", "pairwise_ordering_accuracy"):
            v.close(float(row[key]), metric[key], f"Summary: model {key}")
    for row in csv_rows(summary_dir / "quality_and_audits.csv"):
        session = sessions[row["session_id"]]
        v.close(float(row["max_main_within_visit_cv_pct"]), session["max_cv"], "Summary: within-visit CV")
        v.close(float(row["max_main_block_median_ratio"]), session["max_block_ratio"], "Summary: pass span")
        v.close(float(row["max_control_ratio"]), max(session["control_ratios"].values()), "Summary: control span")
        # The acquisition audit count is a recorded outcome, not this verifier's count.
        v.check(int(row["audit_checks"]) == 5468 and row["audit_passed"] == "True", "Summary: recorded acquisition audit outcome")
    for row in csv_rows(summary_dir / "cross_round_latency_spread.csv"):
        times = [rows_by_session[f"{row['dtype']}_s{index:02d}"][row["case_id"]]["median_ms"] for index in (1, 2, 3)]
        v.close(float(row["max_over_min_across_three_rounds"]), max(times) / min(times), "Summary: cross-round span")
        for index, value in enumerate(times, 1):
            v.close(float(row[f"s{index:02d}_latency_us"]), value * 1000, "Summary: cross-round observation")
    for row in csv_rows(summary_dir / "paired_dtype_latencies.csv"):
        fp = rows_by_session[f"fp16_{row['round']}"][row["case_id"]]["median_ms"]
        bf = rows_by_session[f"bf16_{row['round']}"][row["case_id"]]["median_ms"]
        v.close(float(row["fp16_time_over_bf16_time"]), fp / bf, "Summary: paired precision ratio")
    for filename in ("frozen_selected_transfer.csv", "frozen_krr_candidate_transfer.csv"):
        for row in csv_rows(summary_dir / filename):
            source, target = sessions[row["source"]], sessions[row["target"]]
            v.check(row["source"].endswith("s01") and row["target"].endswith(("s02", "s03")), "Summary: fixed source session")
            if filename == "frozen_selected_transfer.csv":
                v.check(row["model"] == source["selected"], "Summary: transfer of official source winner")
            model = replay_envelope(source["suite"]["models"][row["model"]])
            for split in ("test", "challenge"):
                targets = sorted((item for item in target["rows"] if item["split"] == split), key=lambda item: item["case_id"])
                metric = models.evaluate_predictions(targets, models.predict_model(model, targets, model["dtype"]))
                v.close(float(row[f"{split}_median_ape_pct"]), metric["median_ape_pct"], "Summary: frozen cross-session error")
    for row in csv_rows(summary_dir / "acquisition_all_budgets.csv"):
        session = sessions[row["session_id"]]
        acquisition = session["acquisition_results"][row["split"]]
        curve = next(item for item in acquisition["curves"] if item["policy"] == row["policy"] and item["budget"] == int(row["budget"]))
        v.close(float(row["median_of_seed_median_ape_pct"]), curve["median_median_ape_pct"], "Summary: acquisition curve")
    for row in csv_rows(summary_dir / "offline_batching.csv"):
        batching = sessions[row["session_id"]]["batching"]
        v.close(float(row["offline_equal_work_ratio"]), batching["speedup_small_over_large"], "Summary: offline batching ratio")
    uuids = {data["environment"]["identity"]["device"]["uuid"] for data in sessions.values()}
    v.check(len(uuids) == 3, "Summary: three recorded physical GPU identities")
    for number in (1, 2, 3):
        fp = sessions[f"fp16_s{number:02d}"]["environment"]
        bf = sessions[f"bf16_s{number:02d}"]["environment"]
        v.check(fp["identity"]["device"]["uuid"] == bf["identity"]["device"]["uuid"], "Summary: paired precision device")
        v.check(fp["captured_at_utc"] < bf["captured_at_utc"], "Summary: recorded precision order")
    shapes = [{(row["case_id"], row["m"], row["n"], row["k"], row["split"]) for row in data["rows"]} for data in sessions.values()]
    v.check(all(shape == shapes[0] for shape in shapes), "Summary: identical 96-shape set across sessions")


def verify(results_dir: Path) -> dict:
    v = Verification()
    manifest = read_json(results_dir / "MANIFEST.json")
    v.check(set(manifest["recorded_inputs"]) == set(SESSION_IDS), "Manifest: six source sessions")
    total_bytes = 0
    for relative, entry in manifest["files"].items():
        path = (results_dir / relative).resolve()
        v.check(path.is_relative_to(results_dir.resolve()), "Manifest: path containment")
        data = path.read_bytes()
        v.check(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"Manifest: {relative}")
        total_bytes += len(data)
        if "retained_numeric_payload_sha256" in entry:
            parsed = [json.loads(line) for line in data.decode().splitlines() if line.strip()] if relative.endswith(".jsonl") else json.loads(data)
            v.check(numeric_digest(parsed) == entry["retained_numeric_payload_sha256"], f"Numeric payload: {relative}")
    sessions = {}
    for sid in SESSION_IDS:
        sessions[sid] = verify_session(v, results_dir / "a100" / sid)
    verify_summaries(v, results_dir, sessions)
    return {
        "status": "passed",
        "verification_kind": "publication_snapshot_cpu_verification",
        "checks": v.checks,
        "files_verified": len(manifest["files"]),
        "bytes_verified": total_bytes,
        "sessions": 6,
        "distinct_shapes": 96,
        "shape_session_labels": 576,
        "primary_event_repeat_means": sum(session["raw_event_means"] for session in sessions.values()),
        "archived_numerical_witnesses_recomputed": 24,
        "model_refits_performed": 0,
        "gpu_operations_performed": 0,
        "per_session_checks": {sid: session["checks"] for sid, session in sessions.items()},
        "selected_models": {sid: session["selected"] for sid, session in sessions.items()},
        "scope": "Exported-file integrity and CPU recomputation of retained evidence; not independent GPU attestation or reconstruction of omitted original archive bytes.",
        "recorded_acquisition_audit_count_is_separate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="Optional path for a new verification report; does not modify recorded results")
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(args.results.resolve()):
        parser.error("Save verification reports outside the recorded results directory")
    result = verify(args.results.resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
