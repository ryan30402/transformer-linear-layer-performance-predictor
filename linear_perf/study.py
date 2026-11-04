"""Audited comparisons of sealed sessions, devices and precisions.

These are offline analyses of already collected sessions. They never train on
target development/test/challenge labels, discard a bad session, or manufacture
independent GPUs from repeated event samples. Every source manifest and source
implementation identity is retained in the derived artifact.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import html
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import LabError, digest, read_json, read_jsonl, utc_now, write_json
from . import models as models
from .protocol import validate_study_plan


_COMPLETE = {"simulated_complete", "completed_pending_review", "completed_with_timing_qualifications"}
_SOFTWARE = ("runner_version", "python_version", "pytorch_version", "cuda_runtime_version",
             "driver_versions", "math_policy", "execution")
_METRICS = ("median_ape_pct", "p90_ape_pct", "max_ape_pct", "rmse_us", "mae_us",
            "signed_bias_us", "log_rmse", "geometric_mean_multiplicative_error", "pairwise_ordering_accuracy")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_run(path: str | Path) -> dict:
    from .campaign import audit_run
    root = Path(path).resolve()
    audit = audit_run(root)
    if not audit.get("passed") or not audit.get("complete_session"):
        raise LabError("Comparison requires an audited complete session: " + str(root) + "; " + "; ".join(audit.get("errors", [])))
    plan = validate_study_plan(read_json(root / "plan.json"))
    status = read_json(root / "status.json")
    if status.get("status") not in _COMPLETE:
        raise LabError("Failed, interrupted or diagnostic-only runs cannot be compared as complete sessions")
    result = {
        "root": root, "plan": plan, "status": status,
        "environment": read_json(root / "environment.json"),
        "suite": read_json(root / "model_suite.json"),
        "initial": read_jsonl(root / "initial_measurements.jsonl"),
        "pool": read_jsonl(root / "candidate_measurements.jsonl"),
        "heldout": read_jsonl(root / "heldout_measurements.jsonl"),
        "results": read_json(root / "results.json"), "quality": read_json(root / "quality.json"),
        "manifest": read_json(root / "manifest.json"), "manifest_sha256": _sha(root / "manifest.json"),
        "source_identity": read_json(root / "source_identity.json"), "audit": audit,
    }
    if result["source_identity"] != result["manifest"].get("implementation"):
        raise LabError("Source implementation differs from its sealed manifest")
    meta = models._rows(result["initial"], plan, {"calibration", "development"})
    models._environment(result["environment"], meta, plan, required=True)
    return result


def _source_snapshot(run: dict) -> dict:
    env = run["environment"]
    return {"path": str(run["root"]), "manifest_sha256": run["manifest_sha256"],
            "session_id": run["status"]["session_id"], "status": run["status"]["status"],
            "plan_hash": digest(run["plan"]), "environment_id": env["environment_id"],
            "measurement_kind": env["measurement_kind"], "dtype": run["plan"]["dtype"],
            "suite_hash": run["suite"]["suite_hash"], "source_identity": run["source_identity"],
            "source_identity_sha256": _sha(run["root"] / "source_identity.json"),
            "recorded_device": env.get("identity", {}).get("device", {"name": env.get("device_name")}),
            "quality": run["quality"], "audit": run["audit"]}


def _guard_sources(runs: list[dict], *, same_plan: bool = True, same_gpu: bool = True) -> None:
    if len({str(r["root"]) for r in runs}) != len(runs):
        raise LabError("The same source directory cannot count as multiple sessions")
    if len({r["status"]["session_id"] for r in runs}) != len(runs):
        raise LabError("Comparisons require distinct declared session IDs")
    first = runs[0]
    for run in runs[1:]:
        if run["environment"]["measurement_kind"] != first["environment"]["measurement_kind"]:
            raise LabError("Measured and simulated sources cannot be combined")
        if run["source_identity"] != first["source_identity"]:
            raise LabError("Source programs or analysis dependency versions differ between sessions")
        if same_plan and run["plan"] != first["plan"]:
            raise LabError("Session/device comparison requires the identical registered plan")
        if same_gpu and models._hardware_identity(run["environment"]) != models._hardware_identity(first["environment"]):
            raise LabError("Session/precision comparison requires the same GPU configuration")
        a, b = first["environment"].get("identity", {}), run["environment"].get("identity", {})
        mismatch = [key for key in _SOFTWARE if a.get(key) != b.get(key)]
        if mismatch:
            raise LabError("Software, math or execution contract differs: " + ", ".join(mismatch))


def _fresh_output(output: str | Path, runs: list[dict]) -> Path:
    out = Path(output).resolve()
    if out.exists():
        raise LabError("Comparison output already exists; preserve it and choose a new destination")
    if any(out.is_relative_to(run["root"]) for run in runs):
        raise LabError("A derived comparison must not modify a sealed source directory")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _summary_base(kind: str, runs: list[dict]) -> dict:
    measured = runs[0]["environment"]["measurement_kind"] == "cuda_measured"
    uuids = sorted({r["environment"].get("identity", {}).get("device", {}).get("uuid")
                    for r in runs if r["environment"].get("identity", {}).get("device", {}).get("uuid")}) if measured else []
    return {"schema_version": 1, "protocol_version": models.VERSION, "comparison_kind": kind,
            "created_at_utc": utc_now(), "measurement_kind": runs[0]["environment"]["measurement_kind"],
            "sources": [_source_snapshot(run) for run in runs], "source_count": len(runs),
            "recorded_physical_gpu_uuids": uuids,
            "recorded_unique_gpu_uuid_count": len(uuids) if measured else None,
            "all_source_timing_criteria_passed": all(r["quality"].get("passed") is True for r in runs),
            "claims": {"actual_cuda_source_evidence": measured, "gpu_execution_by_this_analysis": False,
                       "vendor_or_mlperf_certification": False, "session_count_implies_gpu_count": False,
                       "test_based_model_selection": False, "bad_sessions_removed": False},
            "limitations": [
                "An offline comparison of already sealed sessions; it is not a new prospective blinded GPU experiment.",
                "Repeated sessions, GPU UUIDs, shape families and event repeats are different replication units.",
                "Timing qualifications are retained for every source; no favorable-session filtering is performed.",
                "No causal hardware or precision conclusion follows from timing ratios alone when state or run order differs.",
            ] + ([] if measured else ["SIMULATED software validation only. No GPU accuracy, stability or speedup was measured."])}


def _metric_rows(evaluations: dict, *, evaluation: str, session_id: str, budget: int = 0, dtype: str = "") -> list[dict]:
    output = []
    for split, result in evaluations.items():
        for name, score in result["models"].items():
            output.append({"evaluation": evaluation, "session_id": session_id, "dtype": dtype,
                           "calibration_budget": budget, "split": split, "model_id": name,
                           "selected_on_source_development": name == result["selected_model_id"],
                           "n": score["n"], **{key: score.get(key) for key in _METRICS}})
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value if not isinstance(value, (dict, list)) else str(value) for key, value in row.items()})


def _report(summary: dict, rows: list[dict]) -> str:
    title = {"sessions": "Three Sessions: Per-session Fits and Frozen Transfer", "transfer": "Frozen Model Transfer with 4 / 8 Calibration Labels",
             "dtypes": "FP16 / BF16: Paired Comparison with Separate Models"}[summary["comparison_kind"]]
    fields = ("evaluation", "session_id", "dtype", "calibration_budget", "split", "model_id",
              "median_ape_pct", "p90_ape_pct", "rmse_us", "pairwise_ordering_accuracy")
    header = "".join("<th>" + html.escape(field) + "</th>" for field in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            value = f"{value:.5g}" if isinstance(value, float) else "—" if value is None else str(value)
            cells.append("<td>" + html.escape(value) + "</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    notes = "".join("<li>" + html.escape(item) + "</li>" for item in summary["limitations"])
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>" + html.escape(title) + "</title><style>body{font:16px system-ui;line-height:1.6;max-width:1250px;margin:32px auto;padding:0 24px;color:#17212f}"
            "table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ccd3db;text-align:left}"
            "th{background:#e9f0f6}.scroll{overflow:auto}.kind{padding:12px;background:#fff2cd}h1{font-size:30px}</style>"
            "<h1>" + html.escape(title) + "</h1><p class='kind'>Evidence: " + html.escape(summary["measurement_kind"]) +
            " · source sessions: " + str(summary["source_count"]) + " · all timing criteria passed: " +
            str(summary["all_source_timing_criteria_passed"]) + "</p><p>See summary.json for sources, per-case predictions, limitations and hashes; metrics.csv contains the reproducible tables.</p>"
            "<ul>" + notes + "</ul><div class='scroll'><table><thead><tr>" + header + "</tr></thead><tbody>" +
            "".join(body) + "</tbody></table></div></html>")


def _persist(out: Path, summary: dict, rows: list[dict], runs: list[dict], implementation: dict) -> dict:
    from .campaign import audit_run, implementation_identity
    # Audit again after deriving results: preserve both input integrity and the
    # analysis implementation used while the comparison was running.
    for run in runs:
        if _sha(run["root"] / "manifest.json") != run["manifest_sha256"]:
            raise LabError("A source manifest changed while deriving the comparison")
        audit = audit_run(run["root"])
        if not audit.get("passed") or not audit.get("complete_session"):
            raise LabError("A sealed source changed during comparison")
    if implementation_identity() != implementation:
        raise LabError("Analysis source or dependencies changed during comparison")
    summary["analysis_source_identity"] = implementation
    summary["status"] = "comparison_complete"
    summary["summary_hash"] = digest(summary)
    write_json(out / "summary.json", summary, overwrite=False)
    _write_csv(out / "metrics.csv", rows)
    (out / "report.html").write_text(_report(summary, rows), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "manifest.json", {
        "schema_version": 1, "protocol_version": models.VERSION, "stage": "linear_perf_comparison",
        "comparison_kind": summary["comparison_kind"], "measurement_kind": summary["measurement_kind"],
        "sealed_at_utc": utc_now(), "summary_hash": summary["summary_hash"], "files": files,
        "sources": [{"path": str(r["root"]), "manifest_sha256": r["manifest_sha256"],
                     "source_identity": r["source_identity"]} for r in runs],
        "analysis_source_identity": implementation,
        "scope": "Derived artifact integrity; not a new GPU experiment or independent attestation",
    }, overwrite=False)
    return {"status": "comparison_complete", "comparison_kind": summary["comparison_kind"],
            "measurement_kind": summary["measurement_kind"], "output": str(out), "source_count": len(runs),
            "all_source_timing_criteria_passed": summary["all_source_timing_criteria_passed"],
            "summary_hash": summary["summary_hash"]}


def compare_sessions(runs: list[str | Path], out: str | Path) -> dict:
    from .campaign import implementation_identity
    if len(runs) != 3:
        raise LabError("The preregistered session study requires exactly three complete sessions")
    loaded = [_load_run(path) for path in runs]
    _guard_sources(loaded)
    output = _fresh_output(out, loaded)
    implementation = implementation_identity()
    summary = _summary_base("sessions", loaded)
    summary["source_session_rule"] = "The first supplied session is the fixed transfer source; no test-score ordering is used."
    summary["self_fit_evaluations"] = [{"session_id": r["status"]["session_id"],
        "selected_model_id": r["suite"]["selected_model_id"], "evaluations": r["results"]["evaluations"]} for r in loaded]
    rows = []
    distributions = defaultdict(list)
    for run in loaded:
        own = _metric_rows(run["results"]["evaluations"], evaluation="independently_self_fit",
                           session_id=run["status"]["session_id"], dtype=run["plan"]["dtype"])
        rows.extend(own)
        for row in own:
            distributions[(row["split"], row["model_id"])].append(row["median_ape_pct"])
    summary["session_variation"] = [{"split": split, "model_id": model, "sessions": len(values),
        "mean_of_session_median_ape_pct": float(np.mean(values)), "min_session_median_ape_pct": min(values),
        "max_session_median_ape_pct": max(values), "sample_sd_session_median_ape_pct": float(np.std(values, ddof=1)),
        "interpretation": "Descriptive across three sessions; no normality claim or confidence interval."}
        for (split, model), values in sorted(distributions.items())]
    source = loaded[0]
    transfers = []
    for target in loaded[1:]:
        evaluations = {split: models.evaluate_suite(source["suite"], [r for r in target["heldout"] if r["split"] == split],
                         target["plan"], target_environment=target["environment"], transfer_mode="session")
                       for split in ("test", "challenge")}
        transfers.append({"source_session_id": source["status"]["session_id"],
                          "target_session_id": target["status"]["session_id"], "evaluations": evaluations})
        rows.extend(_metric_rows(evaluations, evaluation="first_session_frozen_transfer",
                                 session_id=target["status"]["session_id"], dtype=target["plan"]["dtype"]))
    summary["frozen_source_transfers"] = transfers
    return _persist(output, summary, rows, loaded, implementation)


def _calibration_order(plan: dict) -> list[str]:
    """Fixed nested 4/8 subset by shape-only farthest-point coverage."""
    cases = sorted([c for c in plan["cases"] if c["split"] == "calibration"], key=lambda c: c["id"])
    selected = [cases.pop(0)]
    def nearest(case):
        return min(sum(abs(math.log2(case[axis] / prior[axis])) for axis in ("m", "n", "k")) for prior in selected)
    while cases:
        next_case = min(cases, key=lambda c: (-nearest(c), c["id"]))
        selected.append(next_case)
        cases.remove(next_case)
    return [case["id"] for case in selected]


def _fit_scale(model: dict, calibration_rows: list[dict], plan: dict, target_environment: dict) -> dict:
    """One-parameter log-space least squares; only calibration rows accepted."""
    meta = models._rows(calibration_rows, plan, {"calibration"})
    models._environment(target_environment, meta, plan, required=True)
    if len(calibration_rows) not in (4, 8):
        raise LabError("Transfer correction permits only the fixed four/eight calibration labels")
    budget = len(calibration_rows)
    expected_ids = _calibration_order(plan)[:budget]
    if [r["case_id"] for r in calibration_rows] != expected_ids:
        raise LabError("Target calibration subset differs from the fixed shape-only ordering")
    predicted = models.predict_model(model, calibration_rows, plan["dtype"])
    log_ratios = [math.log(row["median_ms"]) - math.log(p) for row, p in zip(calibration_rows, predicted)]
    log_scale = float(np.mean(log_ratios))
    scale = math.exp(log_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise LabError("Invalid transfer correction scale")
    result = {"model_id": model["model_id"], "source_model_hash": model["model_hash"],
              "target_environment_id": meta["environment_id"], "plan_hash": digest(plan),
              "calibration_budget": budget, "calibration_case_ids": expected_ids,
              "calibration_data_hash": digest(models._training_evidence(calibration_rows)),
              "log_scale": log_scale, "multiplicative_scale": scale, "fitted_parameter_count": 1,
              "objective": "Mean squared natural-log latency error; log_scale = mean(log(target_calibration / source_prediction)).",
              "subset_rule": "Start at lexicographically first calibration ID; greedily maximize nearest selected L1 log2(M,N,K) distance; ties by ID. Four labels are a prefix of eight.",
              "protected_labels_used": False,
              "calibration_observations": [{"case_id": row["case_id"], "actual_ms": row["median_ms"],
                                            "source_predicted_ms": p, "log_ratio": ratio}
                                           for row, p, ratio in zip(calibration_rows, predicted, log_ratios)]}
    return models._sealed(result, "calibration_hash")


def _scaled_evaluation(source: dict, target: dict, corrections: dict, split: str, transfer_mode: str) -> dict:
    rows = [r for r in target["heldout"] if r["split"] == split]
    context = models._evaluation_context(source["suite"], rows, target["plan"], target["environment"], transfer_mode)
    metrics, groups = {}, {}
    for name, model in source["suite"]["models"].items():
        correction = corrections[name]
        models._check_seal(correction, "calibration_hash")
        if correction["source_model_hash"] != model["model_hash"] or correction["target_environment_id"] != target["environment"]["environment_id"]:
            raise LabError("Frozen correction does not match source model and target environment")
        predicted = [p * correction["multiplicative_scale"] for p in models.predict_model(model, rows, target["plan"]["dtype"])]
        score = models.evaluate_predictions(rows, predicted)
        for row, record in zip(rows, score["predictions"]):
            record["ood"] = models._ood(row, model["training_domain"])
        metrics[name], groups[name] = score, models._groups(rows, predicted, target["plan"])
    return {"models": metrics, "groups": groups, "selected_model_id": source["suite"]["selected_model_id"],
            "transfer": {**context, "refitted_on_target": True, "target_fit_scope": "one multiplicative factor per fixed source model; target calibration labels only"},
            "target_test_or_development_used_for_fitting": False,
            "evaluation_data_hash": digest(models._training_evidence(rows)),
            "selection_source": "source_session_development_only; no target selection"}


def compare_transfer(source: str | Path, target: str | Path, out: str | Path, *, mode: str = "session") -> dict:
    from .campaign import implementation_identity
    if mode not in ("session", "gpu"):
        raise LabError("Transfer mode must be 'session' or 'gpu'")
    loaded = [_load_run(source), _load_run(target)]
    _guard_sources(loaded, same_gpu=mode == "session")
    a, b = loaded
    # Guard metadata before any calibration parameter is fitted.
    models._evaluation_context(a["suite"], b["heldout"], b["plan"], b["environment"], mode)
    output = _fresh_output(out, loaded)
    implementation = implementation_identity()
    summary = _summary_base("transfer", loaded)
    summary.update(transfer_mode=mode, source_session_id=a["status"]["session_id"], target_session_id=b["status"]["session_id"],
                   selected_model_id=a["suite"]["selected_model_id"], calibration_order=_calibration_order(b["plan"]),
                   target_model_selection=False)
    lookup = {r["case_id"]: r for r in b["initial"] if r["split"] == "calibration"}
    corrections = {str(budget): {name: _fit_scale(model, [lookup[cid] for cid in summary["calibration_order"][:budget]],
                                                       b["plan"], b["environment"])
                                  for name, model in a["suite"]["models"].items()} for budget in (4, 8)}
    write_json(output / "frozen_calibration.json", {"source_suite_hash": a["suite"]["suite_hash"],
        "corrections": corrections, "frozen_at_utc": utc_now(),
        "scope": "Corrections frozen before evaluating them below; source sessions were already audited and their heldout labels existed."}, overwrite=False)
    zero = {split: models.evaluate_suite(a["suite"], [r for r in b["heldout"] if r["split"] == split], b["plan"],
                    target_environment=b["environment"], transfer_mode=mode) for split in ("test", "challenge")}
    summary["zero_recalibration"] = zero
    rows = _metric_rows(zero, evaluation="frozen_zero_recalibration", session_id=b["status"]["session_id"], dtype=b["plan"]["dtype"])
    summary["calibrated"] = {}
    for budget in (4, 8):
        evaluated = {split: _scaled_evaluation(a, b, corrections[str(budget)], split, mode) for split in ("test", "challenge")}
        summary["calibrated"][str(budget)] = {"corrections": corrections[str(budget)], "evaluations": evaluated}
        rows.extend(_metric_rows(evaluated, evaluation="fixed_target_calibration_scale", session_id=b["status"]["session_id"],
                                 budget=budget, dtype=b["plan"]["dtype"]))
    summary["limitations"].append("A single multiplicative correction cannot fix shape-dependent compute/memory regime changes; both 4 and 8 label budgets are reported without choosing one using test error.")
    return _persist(output, summary, rows, loaded, implementation)


def _dtype_plan_contract(plan: dict) -> dict:
    value = copy.deepcopy(plan)
    dtype = value["dtype"]
    value["dtype"] = "<dtype>"
    value["name"] = value["name"].replace(dtype, "<dtype>")
    value["study"]["study_id"] = value["study"]["study_id"].replace(dtype, "<dtype>")
    return value


def compare_dtypes(runs: list[str | Path], out: str | Path) -> dict:
    from .campaign import implementation_identity
    if len(runs) != 2:
        raise LabError("Precision comparison requires one FP16 and one BF16 session")
    loaded = [_load_run(path) for path in runs]
    _guard_sources(loaded, same_plan=False, same_gpu=True)
    if {run["plan"]["dtype"] for run in loaded} != {"float16", "bfloat16"}:
        raise LabError("Precision comparison requires exactly FP16 and BF16")
    if _dtype_plan_contract(loaded[0]["plan"]) != _dtype_plan_contract(loaded[1]["plan"]):
        raise LabError("Precision plans differ beyond dtype and its name/study identifier")
    by_dtype = {run["plan"]["dtype"]: run for run in loaded}
    fp16, bf16 = by_dtype["float16"], by_dtype["bfloat16"]
    if fp16["environment"]["measurement_kind"] == "cuda_measured":
        devices = [r["environment"]["identity"]["device"] for r in loaded]
        if not devices[0].get("uuid") or devices[0].get("uuid") != devices[1].get("uuid"):
            raise LabError("Precision attribution requires the same recorded physical GPU UUID")
        from .benchmark import _correctness_limits
        for run in loaded:
            for row in run["initial"] + run["pool"] + run["heldout"]:
                if row["correctness"].get("limits") != _correctness_limits(run["plan"]["dtype"]):
                    raise LabError("Precision evidence must retain its registered dtype-specific numerical limits")
    output = _fresh_output(out, loaded)
    implementation = implementation_identity()
    summary = _summary_base("dtypes", loaded)
    summary["models_fitted_separately"] = True
    summary["cross_dtype_parameter_transfer"] = False
    summary["self_fit_evaluations"] = {dtype: run["results"]["evaluations"] for dtype, run in by_dtype.items()}
    rows = []
    for dtype, run in by_dtype.items():
        rows.extend(_metric_rows(run["results"]["evaluations"], evaluation="separate_dtype_self_fit",
                                 session_id=run["status"]["session_id"], dtype=dtype))
    left = {row["case_id"]: row for row in fp16["initial"] + fp16["pool"] + fp16["heldout"]}
    right = {row["case_id"]: row for row in bf16["initial"] + bf16["pool"] + bf16["heldout"]}
    if set(left) != set(right):
        raise LabError("Precision comparisons require identical measured shape IDs")
    pairs = []
    split_ratios = defaultdict(list)
    for cid in sorted(left):
        a, b = left[cid], right[cid]
        if any(a[key] != b[key] for key in ("m", "n", "k", "split")):
            raise LabError("Precision shape pair has inconsistent dimensions or split")
        ratio = a["median_ms"] / b["median_ms"]
        pairs.append({"case_id": cid, "m": a["m"], "n": a["n"], "k": a["k"], "split": a["split"],
                      "float16_median_ms": a["median_ms"], "bfloat16_median_ms": b["median_ms"],
                      "float16_time_over_bfloat16_time": ratio,
                      "float16_relative_l2_error": a["correctness"]["relative_l2_error"],
                      "bfloat16_relative_l2_error": b["correctness"]["relative_l2_error"]})
        split_ratios[a["split"]].append(ratio)
    summary["paired_latency_cases"] = pairs
    summary["paired_latency_by_split"] = {split: {"n": len(values), "median_fp16_time_over_bf16_time": float(np.median(values)),
        "geometric_mean_fp16_time_over_bf16_time": math.exp(float(np.mean(np.log(values)))),
        "min_ratio": min(values), "max_ratio": max(values)} for split, values in split_ratios.items()}
    summary["latency_ratio_definition"] = "FP16 median latency divided by BF16 median latency; above 1 means BF16 was faster in these recorded visits. Separate-session order/state may confound attribution."
    summary["numerical_comparison_scope"] = "Each dtype is checked against FP32 on its own represented inputs; this does not measure downstream model accuracy or error against common unrounded inputs."
    _write_csv(output / "paired_latencies.csv", pairs)
    return _persist(output, summary, rows, loaded, implementation)
