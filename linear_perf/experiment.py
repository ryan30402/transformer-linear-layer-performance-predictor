"""Collect an offline dataset, select on training folds, then evaluate held-out rows.

Measurement and model selection are separate operations. Test labels may already
exist on disk; they are never supplied to the fitting function. Historical
release data remain immutable and use their original verifier.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from .common import LabError, digest, read_json, read_jsonl, utc_now, write_json

VERSION = "1.1.0"


def fresh_directory(path):
    root = Path(path).resolve()
    if root.exists():
        raise LabError(f"Output already exists: {root}. Use a new output directory.")
    root.mkdir(parents=True)
    return root


def dataset_directory(path):
    root = Path(path).resolve()
    if (root / "dataset" / "measurements.jsonl").is_file():
        return root / "dataset"
    if (root / "measurements.jsonl").is_file():
        return root
    raise LabError(f"No collected dataset found in {root}")


def _case_id(row):
    return row.get("case_id", row.get("id"))


def _context(row):
    return str(row["environment_id"]), str(row["dtype"]), str(row["cache_policy"])


def _write_csv(path, records):
    records = list(records)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def load_dataset(path):
    """Validate record membership before any model sees data.

    This is ordinary data validation, not a chronological model-freeze check.
    The collector keeps detailed raw samples; this reader checks their positive
    targets and the shape/split/weight contract used by modeling.
    """
    from .workloads import validate_workload_manifest
    root = dataset_directory(path)
    manifest = read_json(root / "manifest.json")
    validate_workload_manifest(manifest)
    status = read_json(root / "status.json")
    if status.get("status") not in {"completed", "complete", "cuda_complete", "simulated_complete", "collected"}:
        raise LabError("Dataset collection did not complete; retain the partial data and inspect status.json")
    rows = read_jsonl(root / "measurements.jsonl")
    if not rows:
        raise LabError("Dataset has no measurements")
    lookup = {case["id"]: case for case in manifest["cases"]}
    seen, kinds = set(), set()
    for row in rows:
        cid = _case_id(row)
        if cid not in lookup:
            raise LabError("Measurement case is absent from its workload manifest")
        case = lookup[cid]
        if any(row.get(key) != case[key] for key in ("m", "n", "k", "split")):
            raise LabError(f"Measurement shape or split differs from manifest: {cid}")
        expected_weight = float(case.get("weight", 1.0))
        if not math.isclose(float(row.get("weight", 1.0)), expected_weight, rel_tol=1e-12):
            raise LabError(f"Measurement frequency weight differs from manifest: {cid}")
        if row.get("dtype") not in manifest["dtypes"]:
            raise LabError("Measurement precision is not declared in the manifest")
        target = row.get("median_ms")
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
            raise LabError("Latency targets must be positive finite milliseconds")
        if not row.get("environment_id") or not row.get("cache_policy"):
            raise LabError("Measurement context is missing")
        key = (_context(row), cid)
        if key in seen:
            raise LabError("Duplicate shape target within one measurement context")
        seen.add(key)
        kinds.add(row.get("measurement_kind"))
    if len(kinds) != 1 or not kinds <= {"simulated", "cuda_measured"}:
        raise LabError("Do not mix measured data with software simulation")
    from .dataset_measurement import validate_collected_dataset
    validation = validate_collected_dataset(root)
    if not validation["passed"]:
        raise LabError("Dataset record validation failed: " + validation.get("error", "unknown error"))
    environments = read_json(root / "environments.json")
    status = {**status, "validation": validation}
    return root, manifest, rows, environments, status


def _sm_count(rows, environments=None):
    values = {row.get("sm_count") for row in rows if row.get("sm_count") is not None}
    if len(values) > 1:
        raise LabError("One model group has inconsistent SM counts")
    if values:
        value = next(iter(values))
        if type(value) is not int or value <= 0:
            raise LabError("SM count must be a positive integer")
        return value
    return None


def analyze_dataset(dataset, output, *, progress=print):
    """Tune using train folds and refit on all train rows before evaluating.

    This function intentionally loads an already measured dataset. It only
    passes split='train' to fit_cv_suite. Test/challenge/stress are evaluation
    inputs after selection, not model-selection data.
    """
    from .selection import fit_cv_suite, evaluate_cv_suite
    root, manifest, rows, environments, collection_status = load_dataset(dataset)
    destination = fresh_directory(output)
    groups = defaultdict(list)
    for row in rows:
        groups[_context(row)].append(row)
    summary = {
        "protocol_version": VERSION, "artifact_type": "offline_dataset_analysis",
        "created_at_utc": utc_now(), "dataset_path": str(root), "manifest_hash": digest(manifest),
        "measurement_kind": rows[0]["measurement_kind"],
        "workflow": "collect_all_then_grouped_cv_refit_and_evaluate",
        "selection_metric": "weighted_mean_ape_pct", "groups": [],
        "test_labels_measured_before_fitting": True,
        "heldout_labels_used_for_selection": False,
        "cv_score_is_selection_evidence_not_unbiased_test_performance": True,
        "weights": "Declared model operation occurrences and M sampling, not observed production traffic",
        "accuracy_claim": "No new GPU accuracy is established by simulation or retrospective reanalysis.",
        "measurement_quality": collection_status.get("quality", {}),
        "record_validation": collection_status["validation"],
    }
    csv_metrics, csv_predictions = [], []
    try:
        for number, (context, records) in enumerate(sorted(groups.items()), 1):
            environment_id, dtype, cache_policy = context
            gid = f"group_{number:02d}_{dtype}_{re.sub('[^A-Za-z0-9_-]', '_', cache_policy)}"
            folder = destination / gid
            folder.mkdir()
            train = [copy.deepcopy(row) for row in records if row["split"] == "train"]
            families = {(row["n"], row["k"]) for row in train}
            item = {"group_id": gid, "environment_id": environment_id, "dtype": dtype,
                    "cache_policy": cache_policy, "train_count": len(train), "train_family_count": len(families)}
            if len(families) < 3 or len(train) < 6:
                item.update(status="insufficient_training_coverage",
                            reason="Grouped model selection needs at least three training families and six rows")
                summary["groups"].append(item)
                continue
            if progress:
                progress(f"[{gid}] grouped cross-validation on {len(train)} training shapes")
            suite = fit_cv_suite(train, sm_count=_sm_count(records, environments))
            write_json(folder / "model_suite.json", suite)
            item.update(status="analyzed", selected_model_id=suite["selected_model_id"],
                        model_suite_file=f"{gid}/model_suite.json", evaluation_files={})
            # The model is now fitted and selected. Existing held-out labels are
            # evaluated without being passed through a fitter or a selector.
            for split in ("test", "challenge", "stress"):
                targets = [copy.deepcopy(row) for row in records if row["split"] == split]
                if not targets:
                    continue
                evaluation = evaluate_cv_suite(suite, targets, sm_count=_sm_count(records, environments))
                write_json(folder / f"{split}.json", evaluation)
                item["evaluation_files"][split] = f"{gid}/{split}.json"
                for model_id, metrics in evaluation["models"].items():
                    numeric = {key: value for key, value in metrics.items()
                               if isinstance(value, (int, float)) and not isinstance(value, bool)}
                    csv_metrics.append({"group_id": gid, "dtype": dtype, "cache_policy": cache_policy,
                                        "split": split, "model_id": model_id,
                                        "selected_by_cv": model_id == suite["selected_model_id"], **numeric})
                    for prediction in metrics.get("predictions", []):
                        csv_predictions.append({"group_id": gid, "split": split, "model_id": model_id,
                                                "selected_by_cv": model_id == suite["selected_model_id"], **prediction})
            summary["groups"].append(item)
        summary["analyzed_group_count"] = sum(item["status"] == "analyzed" for item in summary["groups"])
        if not summary["analyzed_group_count"]:
            raise LabError("No context has enough training-family coverage for model selection; use a full dataset")
        summary["status"] = "completed"
        _write_csv(destination / "metrics.csv", csv_metrics)
        _write_csv(destination / "predictions.csv", csv_predictions)
        write_json(destination / "summary.json", summary)
        render_analysis_report(destination)
    except BaseException as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(destination / "summary.json", summary)
        raise
    return {**summary, "output": str(destination)}


def render_analysis_report(directory):
    root = Path(directory)
    summary = read_json(root / "summary.json")
    rows = []
    if (root / "metrics.csv").is_file():
        with (root / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    columns = ("dtype", "cache_policy", "split", "model_id", "selected_by_cv",
               "weighted_mean_ape_pct", "mean_ape_pct", "median_ape_pct", "p90_ape_pct", "max_ape_pct")
    head = "".join(f"<th>{html.escape(key)}</th>" for key in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in columns) + "</tr>"
                   for row in rows)
    kind = html.escape(summary["measurement_kind"])
    message = "SIMULATED SOFTWARE CHECK — these are not GPU performance results." if kind == "simulated" else (
        "Measured dataset analysis. Interpret each precision and cache policy separately.")
    quality = summary.get("measurement_quality", {})
    quality_message = ("The declared timing stability checks passed." if quality.get("passed") else
                       "Timing stability checks have qualifications. Inspect measurement_quality in the metadata before interpreting prediction errors.")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transformer Linear Layer Performance Predictor — Dataset Analysis</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#17212e}}h1{{line-height:1.2}}table{{border-collapse:collapse;font-size:13px;min-width:850px}}td,th{{padding:9px;border:1px solid #cbd2d9;text-align:left}}th{{background:#eaf0f5}}.scroll{{overflow-x:auto}}a{{color:#1758a0}}</style>
<h1>Transformer Linear Layer Performance Predictor</h1><h2>Offline dataset analysis</h2>
<p><strong>{message}</strong></p>
<p>{quality_message} Every observation is retained; stability flags do not silently remove slow cases.</p>
<p>All measurement splits were collected before fitting. Model and hyperparameter selection use grouped cross-validation on training data only. Selected configurations are refitted on all training rows; test, challenge and irregular stress data are evaluated afterwards.</p>
<p>The selection objective is weighted mean absolute percentage error. The weights describe model operation occurrences under a declared shape plan; they are not a production traffic trace. Cross-validation scores are tuning scores, not independent test estimates.</p>
<p><a href="summary.json">Analysis metadata</a> · <a href="metrics.csv">All metrics CSV</a> · <a href="predictions.csv">Per-shape predictions CSV</a></p>
<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<p>Each group directory contains the model payload, cross-validation results and all candidate scores. Unavailable groups remain in the metadata. Warm reuse and rotating buffers describe separate measurement contracts; rotation is not guaranteed cold-cache execution.</p></html>"""
    (root / "report.html").write_text(page, encoding="utf-8")
    return root / "report.html"


def run_experiment(manifest, output, *, session_id, simulate=False, device="cuda:0",
                   analyze=True, repeat_ids=None, progress=print, collection_checkpoint=None,
                   checkpoint_required=False):
    from .dataset_measurement import collect_dataset
    from .workloads import validate_workload_manifest
    plan = copy.deepcopy(read_json(manifest) if isinstance(manifest, (str, Path)) else manifest)
    validate_workload_manifest(plan)
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session_id):
        raise LabError("Use a short alphanumeric session ID")
    if collection_checkpoint is not None and not callable(collection_checkpoint):
        raise LabError("collection_checkpoint must be callable or null")
    if type(checkpoint_required) is not bool or (checkpoint_required and collection_checkpoint is None):
        raise LabError("Required checkpointing needs a callable collection checkpoint")
    out = fresh_directory(output)
    status = {"protocol_version": VERSION, "session_id": session_id, "status": "collecting",
              "created_at_utc": utc_now(), "measurement_kind": "simulated" if simulate else "cuda_measured",
              "workflow": "collect_all_then_grouped_cv_refit_and_evaluate", "repeat_only": repeat_ids is not None}
    write_json(out / "status.json", status)
    write_json(out / "workload_manifest.json", plan)
    try:
        if checkpoint_required:
            collection_checkpoint({"checkpoint_event": "experiment_initialized", "session_id": session_id,
                                   "dataset_directory": str(out / "dataset"), "recorded_status": copy.deepcopy(status)})
        collected = collect_dataset(plan, out / "dataset", session_id=session_id, simulate=simulate,
                                    device=device, progress=progress, case_ids=repeat_ids,
                                    **({"checkpoint": collection_checkpoint} if collection_checkpoint is not None else {}))
        status["collection"] = collected
        # No predictor exists during collection. This is an ordinary offline
        # train/test boundary; there is no freeze-before-measurement requirement.
        load_dataset(out)
        if analyze and repeat_ids is None:
            status["status"] = "analyzing"
            write_json(out / "status.json", status)
            result = analyze_dataset(out, out / "analysis", progress=progress)
            status["analysis_output"] = result["output"]
        status.update(status="simulated_complete" if simulate else "completed", finished_at_utc=utc_now())
        write_json(out / "status.json", status)
        if checkpoint_required:
            collection_checkpoint({"checkpoint_event": "experiment_finished", "session_id": session_id,
                                   "dataset_directory": str(out / "dataset"), "recorded_status": copy.deepcopy(status)})
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc), finished_at_utc=utc_now())
        write_json(out / "status.json", status)
        raise
    write_json(out / "status.json", status)
    return {**status, "output": str(out)}


def audit_dataset(path):
    """Check dataset membership and raw sample aggregation without refitting."""
    root, manifest, rows, environments, status = load_dataset(path)
    return {"status": "passed", "passed": True, "protocol_version": VERSION,
            "dataset": str(root), "measurement_kind": rows[0]["measurement_kind"],
            "measurement_rows": len(rows), "distinct_shapes": len({_case_id(row) for row in rows}),
            "contexts": len({_context(row) for row in rows}), "manifest_hash": digest(manifest),
            "record_validation": status["validation"], "measurement_quality": status.get("quality", {}),
            "scope": "Dataset membership, raw event divisors, visit aggregation, numerical-check records, coverage and context; no new GPU work",
            "freeze_before_measurement_required": False}


def _source_analysis(dataset, manifest, rows):
    folder = Path(dataset).parent / "analysis"
    if not (folder / "summary.json").is_file():
        raise LabError("Source needs its completed analysis in the sibling analysis directory")
    summary = read_json(folder / "summary.json")
    if summary.get("status") != "completed" or summary.get("manifest_hash") != digest(manifest):
        raise LabError("Source analysis is incomplete or belongs to another workload")
    if summary.get("measurement_kind") != rows[0]["measurement_kind"]:
        raise LabError("Source analysis measurement kind differs from its dataset")
    return folder, summary


def _source_suite(folder, item, rows):
    from .selection import _training_evidence, _validate_rows, _check_suite
    path = (folder / item["model_suite_file"]).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise LabError("Source model path escapes its analysis directory")
    suite = read_json(path)
    _check_suite(suite)
    train = [row for row in rows if row["split"] == "train" and
             _context(row) == (item["environment_id"], item["dtype"], item["cache_policy"])]
    clean, context = _validate_rows(train, fitting=True)
    if digest(_training_evidence(clean)) != suite["training_data_hash"] or any(suite[key] != value for key, value in context.items()):
        raise LabError("Source model does not match its recorded training rows or context")
    return suite


def compare_datasets(source, target, output, *, mode="repeat", progress=print):
    """Compare repeats and transfer already-selected source models without refit."""
    from .selection import evaluate_cv_suite
    if mode not in {"repeat", "gpu"}:
        raise LabError("Comparison mode must be repeat or gpu")
    source_root, source_plan, a, source_env, source_status = load_dataset(source)
    target_root, target_plan, b, target_env, target_status = load_dataset(target)
    if digest(source_plan) != digest(target_plan):
        raise LabError("Comparisons require the same shape and measurement plan; use the source workload manifest")
    if a[0]["measurement_kind"] != b[0]["measurement_kind"]:
        raise LabError("Cannot compare simulation with hardware observations")
    source_analysis, source_summary = _source_analysis(source_root, source_plan, a)
    device_a, device_b = source_env["device"], target_env["device"]
    uuids = (device_a.get("uuid"), device_b.get("uuid"))
    same_uuid = bool(all(uuids) and uuids[0] == uuids[1])
    if mode == "repeat":
        hardware_keys = ("name", "multiprocessor_count", "compute_capability", "total_memory_bytes")
        if any(device_a.get(key) != device_b.get(key) for key in hardware_keys) or (all(uuids) and not same_uuid):
            raise LabError("Different GPU models or physical UUIDs require --mode gpu")
    destination = fresh_directory(output)
    result = {"protocol_version": VERSION, "created_at_utc": utc_now(), "mode": mode,
              "measurement_kind": a[0]["measurement_kind"], "source": str(source_root), "target": str(target_root),
              "target_labels_used_for_refit": False, "groups": [], "status": "completed",
              "source_device": device_a, "target_device": device_b,
              "same_physical_gpu_verified": same_uuid,
              "hardware_identity_note": "Physical identity is unverified when a UUID is unavailable; simulation has no physical GPU identity.",
              "source_quality": source_status.get("quality", {}), "target_quality": target_status.get("quality", {}),
              "scope": "Same-shape repeatability and zero-retraining transfer; not new-shape generalization"}
    result["status"] = "comparing"
    write_json(destination / "summary.json", result)
    try:
        for item in source_summary["groups"]:
            if item["status"] != "analyzed":
                continue
            suite = _source_suite(source_analysis, item, a)
            matching = [r for r in b if r["dtype"] == item["dtype"] and r["cache_policy"] == item["cache_policy"]]
            grouped_targets = defaultdict(list)
            for row in matching:
                grouped_targets[row["environment_id"]].append(row)
            source_rows = {_case_id(r): r for r in a if _context(r) == (item["environment_id"], item["dtype"], item["cache_policy"])}
            for target_id, records in sorted(grouped_targets.items()):
                # Hardware identity is inspected through metadata, not inferred from
                # the opaque environment hash (which can include session metadata).
                source_context = source_env["contexts"][f"{item['dtype']}/{item['cache_policy']}"]["identity"]
                target_context = target_env["contexts"][f"{item['dtype']}/{item['cache_policy']}"]["identity"]
                comparison_keys = ("pytorch_version", "cuda_runtime_version", "driver_versions", "math_policy")
                differences = [key for key in comparison_keys if source_context.get(key) != target_context.get(key)]
                if mode == "repeat" and differences:
                    raise LabError("Repeatability requires the same software and math policy: " + ", ".join(differences))
                pairs = []
                for row in records:
                    cid = _case_id(row)
                    if cid not in source_rows:
                        continue
                    before = source_rows[cid]["median_ms"]
                    after = row["median_ms"]
                    pairs.append({"case_id": cid, "split": row["split"], "source_ms": before,
                                  "target_ms": after, "target_over_source": after / before,
                                  "absolute_change_pct": abs(after - before) / before * 100})
                gid = f"comparison_{len(result['groups']) + 1:02d}"
                _write_csv(destination / f"{gid}_repeat.csv", pairs)
                scores = {}
                for split in ("test", "challenge", "stress"):
                    targets = [r for r in records if r["split"] == split]
                    if targets:
                        scores[split] = evaluate_cv_suite(suite, targets, sm_count=_sm_count(records),
                                                          allow_environment_transfer=True)
                write_json(destination / f"{gid}_transfer.json", scores)
                changes = [p["absolute_change_pct"] for p in pairs]
                result["groups"].append({"group_id": gid, "dtype": item["dtype"], "cache_policy": item["cache_policy"],
                                         "source_model_id": suite["selected_model_id"], "target_environment_id": target_id,
                                         "paired_shapes": len(pairs),
                                         "software_context_differences": differences,
                                         "median_absolute_change_pct": statistics.median(changes) if changes else None,
                                         "maximum_absolute_change_pct": max(changes) if changes else None,
                                         "repeat_file": f"{gid}_repeat.csv", "transfer_file": f"{gid}_transfer.json"})
        if not result["groups"]:
            raise LabError("No compatible source/target precision and cache-policy groups were found")
        result["status"] = "completed"
        write_json(destination / "summary.json", result)
        render_comparison_report(destination)
        return {**result, "output": str(destination)}
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(destination / "summary.json", result)
        raise


def render_comparison_report(directory):
    root = Path(directory)
    summary = read_json(root / "summary.json")
    body = []
    for group in summary["groups"]:
        evaluations = read_json(root / group["transfer_file"])
        for split, evaluation in evaluations.items():
            metrics = evaluation["models"][group["source_model_id"]]
            values = (group["dtype"], group["cache_policy"], split, group["source_model_id"],
                      metrics["n"], metrics["mean_ape_pct"], metrics["p90_ape_pct"], metrics["max_ape_pct"])
            body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    message = "SIMULATED SOFTWARE CHECK" if summary["measurement_kind"] == "simulated" else "Measured source-to-target comparison"
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transformer Linear Layer Performance Predictor — Comparison</title>
<style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:8px}}</style>
<h1>Source-to-target comparison</h1><p><strong>{message}</strong></p>
<p>Source model selection is unchanged. Target labels are used only for evaluation, without retraining. Same-shape repetition measures stability; it does not establish new-shape generalization.</p>
<p>Same physical GPU verified by UUID: {summary["same_physical_gpu_verified"]}. Missing UUIDs and simulation cannot establish physical identity. Inspect source and target quality flags and context differences.</p>
<p><a href="summary.json">Full comparison metadata and per-group file links</a></p>
<div style="overflow-x:auto"><table><tr><th>Precision</th><th>Cache policy</th><th>Split</th><th>Source model</th><th>Shapes</th><th>Mean APE %</th><th>P90 APE %</th><th>Worst APE %</th></tr>{''.join(body)}</table></div></html>'''
    (root / "report.html").write_text(page, encoding="utf-8")


def validate_layer(source, output, *, model_id="qwen2_5_0_5b", dtype="float16",
                   batch_size=1, sequence_length=128, device="cuda:0", progress=print):
    """Compose source predictions and separately measured non-GEMM layer costs."""
    from .dataset_measurement import inspect_environment
    from .layer_validation import benchmark_layer, compose_layer_estimate, layer_gemm_cases
    from .selection import predict_suite
    from .workloads import load_model_specs
    root, manifest, rows, environments, status = load_dataset(source)
    if rows[0]["measurement_kind"] != "cuda_measured":
        raise LabError("Layer validation needs a measured GPU source, not simulated timing labels")
    folder, summary = _source_analysis(root, manifest, rows)
    groups = [item for item in summary["groups"] if item["status"] == "analyzed" and
              item["dtype"] == dtype and item["cache_policy"] == "warm_reuse"]
    if len(groups) != 1:
        raise LabError("Layer validation needs exactly one selected warm-reuse model for this precision")
    item = groups[0]
    suite = _source_suite(folder, item, rows)
    sources = {source["id"]: source for source in load_model_specs()}
    if model_id not in sources:
        raise LabError("Unknown model ID; choose from " + ", ".join(sorted(sources)))
    spec = sources[model_id]["spec"]
    cases = layer_gemm_cases(spec, batch_size=batch_size, sequence_length=sequence_length)
    current = inspect_environment(manifest, device=device)
    current_context = current["contexts"][f"{dtype}/warm_reuse"]
    if current_context["environment_id"] != suite["environment_id"]:
        raise LabError("Layer validation requires the source GPU, software and math-policy context; collect a matching source run first")
    values = predict_suite(suite, cases, sm_count=current_context["sm_count"])[suite["selected_model_id"]]
    predictions = {case["id"]: value for case, value in zip(cases, values)}
    destination = fresh_directory(output)
    result = {"protocol_version": VERSION, "artifact_type": "hybrid_decoder_layer_comparison",
              "status": "measuring", "source_environment_id": suite["environment_id"],
              "source_model_id": suite["selected_model_id"], "source_suite_hash": suite["suite_hash"],
              "model_source": sources[model_id], "prediction_cases": cases, "predicted_gemm_ms": predictions,
              "measurement_kind": "cuda_measured", "excluded_from_predictor_training": True,
              "same_physical_gpu_verified": bool(environments["device"].get("uuid") and
                  environments["device"].get("uuid") == current["device"].get("uuid")),
              "hardware_identity_note": "Matching context metadata without a UUID does not establish physical device identity.",
              "source_quality": status.get("quality", {}),
              "scope": "Downstream composition check; component shapes may overlap source training/test data. Not a new held-out shape test."}
    write_json(destination / "model_suite.json", suite)
    write_json(destination / "summary.json", result)
    try:
        record = benchmark_layer(spec, batch_size=batch_size, sequence_length=sequence_length,
                                 dtype=dtype, device=device, output_dir=destination / "measurements", progress=progress)
        result["status"] = record["status"]
        if record["status"] == "completed":
            result["comparison"] = compose_layer_estimate(record, predictions)
            result["layer_quality_flags"] = record.get("quality_flags", {})
        else:
            result["reason"] = record.get("reason", "Layer measurement did not complete")
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(destination / "summary.json", result)
        raise
    write_json(destination / "summary.json", result)
    return {**result, "output": str(destination)}
