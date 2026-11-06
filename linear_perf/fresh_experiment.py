"""One fresh GPU experiment with training-only modeling and a short repeat.

The primary and repeat datasets are collected by the current implementation.
This module does not load historical measurement archives or retrospectively
relabel their results. Explicit simulation is only a CPU software check.
"""
from __future__ import annotations

import copy
import csv
import html
import re
from collections import Counter
from pathlib import Path

from . import experiment
from .common import LabError, digest, read_json, utc_now, write_json
from .dataset_measurement import repeat_subset
from .workloads import validate_workload_manifest

VERSION = "1.1.0"
_STAGES = ("primary", "repeat", "repeat_comparison", "layer")
_LABELS = {
    "primary": "Fresh full dataset, training-only CV, refit and held-out evaluation",
    "repeat": "Fresh deterministic repeat subset, without refitting",
    "repeat_comparison": "Repeatability and unchanged source-model evaluation",
    "layer": "Optional random-weight decoder-layer hybrid estimate",
}


def _write_selected_metrics(source, destination):
    """Retain the selected-model rows without changing their measured results."""
    with Path(source).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if row.get("selected_by_cv", "").lower() == "true"]
    if not fields or not rows:
        raise LabError("The completed primary analysis has no selected-model evaluation rows")
    with Path(destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _success_kind(result, simulate):
    expected = "simulated_complete" if simulate else "completed"
    if result.get("status") != expected:
        raise LabError("Collection did not complete with the requested measurement kind")
    if result.get("measurement_kind") != ("simulated" if simulate else "cuda_measured"):
        raise LabError("Collection returned a different measured/simulated identity")


def _audit_stage(path, simulate):
    audit = experiment.audit_dataset(path)
    if audit.get("passed") is not True:
        raise LabError("Collected dataset record validation failed")
    if audit.get("measurement_kind") != ("simulated" if simulate else "cuda_measured"):
        raise LabError("Dataset audit returned a different measurement identity")
    write_json(Path(path) / "audit.json", audit)
    return audit


def render_fresh_report(directory):
    """Render the combined report from retained metadata, including failures."""
    root = Path(directory)
    summary = read_json(root / "summary.json")

    def esc(value):
        return html.escape(str(value), quote=True)

    def number(value):
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value) if value is not None else "Unavailable"

    def link(relative, label):
        if (root / relative).is_file():
            return f'<a href="{esc(relative)}">{esc(label)}</a>'
        return f"{esc(label)} (not available)"

    simulated = summary["measurement_kind"] == "simulated"
    banner = ("SIMULATED SOFTWARE CHECK — no GPU performance evidence" if simulated else
              "Fresh GPU experiment — interpret the recorded completion and quality status")
    stage_rows = []
    stage_links = {
        "primary": ("primary/analysis/report.html", "All candidates and predictions"),
        "repeat": ("repeat/audit.json", "Repeat audit"),
        "repeat_comparison": ("repeat_comparison/report.html", "Repeat comparison"),
        "layer": ("layer/summary.json", "Layer comparison"),
    }
    for name in _STAGES:
        stage = summary["stages"][name]
        destination, label = stage_links[name]
        detail = stage.get("reason", stage.get("error", ""))
        stage_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in
                          (_LABELS[name], stage["status"], detail)) +
                          f"<td>{link(destination, label)}</td></tr>")

    quality_rows = []
    for name in ("primary", "repeat"):
        stage = summary["stages"][name]
        audit = stage.get("audit", {})
        quality = summary["measurement_quality"].get(name, {})
        state = ("Passed declared thresholds" if quality.get("passed") is True else
                 "Qualified; inspect retained flags" if quality.get("passed") is False else "Unavailable")
        controls = quality.get("controls", [])
        quality_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in (
            name, audit.get("checks", "Unavailable"), state,
            len(quality.get("flagged_rows", [])) if quality else "Unavailable",
            sum(control.get("passed") is False for control in controls) if controls else "Unavailable",
        )) + "</tr>")

    metrics = []
    if (root / "selected_model_metrics.csv").is_file():
        with (root / "selected_model_metrics.csv").open(newline="", encoding="utf-8") as stream:
            metrics = list(csv.DictReader(stream))
    metric_rows = []
    for row in metrics:
        values = [row.get(key, "") for key in ("dtype", "cache_policy", "split", "model_id", "n")]
        values += [number(row.get(key)) for key in
                   ("weighted_mean_ape_pct", "mean_ape_pct", "median_ape_pct", "p90_ape_pct", "max_ape_pct")]
        metric_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in values) + "</tr>")

    repeat_rows = []
    for group in summary.get("repeat_comparison", {}).get("groups", []):
        values = (group["dtype"], group["cache_policy"], group["paired_shapes"],
                  number(group["median_absolute_change_pct"]), number(group["maximum_absolute_change_pct"]))
        repeat_rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in values) + "</tr>")
    quality_note = ("Timing quality has retained qualifications. Read the controls and flagged rows before interpreting error scores."
                    if summary.get("measurement_quality_passed") is False else
                    "Record validation and timing quality are distinct. Replays are not independent GPU sessions.")
    error_note = f'<p class="warning">{esc(summary.get("error", ""))}</p>' if summary.get("error") else ""
    checkpoints = "".join(f"<li>{esc(item['error'])}</li>" for item in summary.get("checkpoint_warnings", []))
    checkpoint_note = f"<h2>Checkpoint export notes</h2><ul>{checkpoints}</ul>" if checkpoints else ""
    split_text = ", ".join(f"{key}: {value}" for key, value in summary["workload"]["split_counts"].items())
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transformer Linear Layer Performance Predictor — Fresh Experiment</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1320px;margin:36px auto;padding:0 24px;color:#182432}}h1{{line-height:1.15}}h2{{margin-top:34px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd5df;padding:9px;text-align:left;vertical-align:top}}th{{background:#edf2f7}}.scroll{{overflow-x:auto}}.banner{{background:#e9f1fc;padding:16px;border-left:5px solid #285da8}}.warning{{background:#fff2d9;padding:12px}}a{{color:#1758a0}}code{{background:#edf2f7;padding:2px 5px}}</style>
<h1>Transformer Linear Layer Performance Predictor</h1>
<p>Version {esc(summary['protocol_version'])} · Fresh experiment {esc(summary['session_id'])}</p>
<p class="banner"><strong>{esc(banner)}</strong><br>Overall status: <strong>{esc(summary['status'])}</strong><br>Required stages completed: {esc(summary['required_stages_completed'])}. All requested stages completed: {esc(summary['all_requested_stages_completed'])}.</p>
{error_note}
<p>This experiment collects new data from the declared workload. Historical measurement archives are not inputs. All primary splits are measured before fitting; grouped cross-validation, preprocessing and model selection use training data only, followed by full-training refit and held-out evaluation. No freeze-before-measurement gate is used.</p>
<p>Planned shapes per precision: {esc(summary['workload']['shape_count'])}. {esc(split_text)}. Precisions: {esc(', '.join(summary['workload']['dtypes']))}. Workload weights are architectural occurrence assumptions, not production traffic.</p>
<p>{link('summary.json', 'Complete experiment metadata')} · {link('workload_manifest.json', 'Workload manifest')} · {link('selected_model_metrics.csv', 'Selected-model metrics CSV')}</p>
<h2>Stage completion</h2><div class="scroll"><table><tr><th>Stage</th><th>Status</th><th>Details</th><th>Evidence</th></tr>{''.join(stage_rows)}</table></div>
<p>Completed primary measurements and predictions remain usable for inspection if a later stage fails. An incomplete requested workflow is not labeled complete. No automatic retry or deletion of slow observations occurs.</p>
<h2>Measurement quality</h2><p>{esc(quality_note)}</p>
<div class="scroll"><table><tr><th>Dataset</th><th>Record checks</th><th>Timing quality</th><th>Flagged targets</th><th>Flagged controls</th></tr>{''.join(quality_rows)}</table></div>
<p>{link('primary/dataset/quality.json', 'Primary quality details')} · {link('primary/dataset/controls.jsonl', 'Primary controls')} · {link('primary/dataset/profiles.jsonl', 'Separate kernel diagnostics')} · {link('repeat/dataset/quality.json', 'Repeat quality details')}</p>
<h2>Primary held-out results</h2>
<p>Only the training-CV-selected model is summarized here. The primary report retains every candidate, subgroup and per-case prediction. CV scores are tuning evidence. Test, family-challenge and irregular-stress errors describe different scopes and are not pooled into one score.</p>
<div class="scroll"><table><tr><th>Precision</th><th>Cache policy</th><th>Split</th><th>Selected model</th><th>Shapes</th><th>Weighted mean APE %</th><th>Mean APE %</th><th>Median APE %</th><th>P90 APE %</th><th>Worst APE %</th></tr>{''.join(metric_rows)}</table></div>
<h2>Repeatability</h2>
<p>The repeat subset is chosen from shapes and split metadata without reading latencies or prediction errors. It uses the same manifest and does not fit another model. The repeat comparison evaluates the unchanged primary predictor and same-shape timing differences; it is not a second independent new-shape test.</p>
<div class="scroll"><table><tr><th>Precision</th><th>Cache policy</th><th>Paired shapes</th><th>Median absolute timing change %</th><th>Maximum absolute timing change %</th></tr>{''.join(repeat_rows)}</table></div>
<p>Same physical GPU verified by UUID: {esc(summary.get('repeat_comparison', {}).get('same_physical_gpu_verified', 'Unavailable'))}. Missing UUIDs and simulation cannot establish physical identity. Warm reuse and rotating buffers remain separate contracts; rotation does not guarantee cold caches.</p>
<h2>Interpretation</h2><p>New-workload errors are not a controlled improvement percentage over a different old workload. One GPU and a short repeat do not establish cross-architecture accuracy or serving latency. If these held-out results guide future feature or hyperparameter changes, they become development evidence for those changes.</p>
<p>The optional layer check is a hybrid estimate: predicted linear GEMMs plus separately measured other operations. It is not a fully predicted end-to-end model. No layer or repeat labels enter predictor fitting.</p>
{checkpoint_note}</html>'''
    (root / "report.html").write_text(page, encoding="utf-8")
    return root / "report.html"


def run_fresh_experiment(manifest, output, *, session_id, simulate=False, device="cuda:0",
                         include_repeat=True, include_layer=False, layer_model_id="qwen2_5_0_5b",
                         layer_dtype=None, progress=print, checkpoint=None,
                         collection_checkpoint=None, checkpoint_required=False):
    """Collect a fresh primary, short repeat and combined report in one call.

    Required-stage errors are persisted and re-raised, preserving completed
    earlier stages. A requested optional layer that cannot finish returns an
    explicit ``incomplete`` status. Timing-quality flags do not discard data or
    change model selection. ``checkpoint(summary)`` may export completed stages.
    Required checkpointing additionally exports initialized and final states;
    any backup failure stops follow-ups without repeating GPU work. Collector
    callbacks preserve raw evidence at complete-visit boundaries before profiling.
    """
    plan = copy.deepcopy(read_json(manifest) if isinstance(manifest, (str, Path)) else manifest)
    validate_workload_manifest(plan)
    for name, value in (("simulate", simulate), ("include_repeat", include_repeat), ("include_layer", include_layer)):
        if type(value) is not bool:
            raise LabError(name + " must be boolean")
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,55}", session_id):
        raise LabError("Use a 1–56 character alphanumeric session ID; stage suffixes reserve the remaining characters")
    if checkpoint is not None and not callable(checkpoint):
        raise LabError("checkpoint must be callable or null")
    if collection_checkpoint is not None and not callable(collection_checkpoint):
        raise LabError("collection_checkpoint must be callable or null")
    if type(checkpoint_required) is not bool or (checkpoint_required and checkpoint is None):
        raise LabError("Required checkpointing needs a callable stage checkpoint")
    if layer_dtype is None:
        layer_dtype = plan["dtypes"][0]
    if include_layer and layer_dtype not in plan["dtypes"]:
        raise LabError("Layer precision must be present in the primary workload")
    repeated = repeat_subset(plan) if include_repeat else []
    out = experiment.fresh_directory(output)
    kind = "simulated" if simulate else "cuda_measured"
    required = ["primary"] + (["repeat", "repeat_comparison"] if include_repeat else [])
    requested = required + (["layer"] if include_layer else [])
    summary = {
        "protocol_version": VERSION, "artifact_type": "fresh_experiment", "session_id": session_id,
        "measurement_kind": kind, "status": "running", "active_stage": None, "output": str(out),
        "created_at_utc": utc_now(), "manifest_hash": digest(plan),
        "workflow": "fresh_primary_collection_then_train_only_cv_refit_evaluation_and_short_repeat",
        "historical_measurements_loaded": False, "freeze_before_measurement_required": False,
        "heldout_labels_used_for_selection": False, "repeat_labels_used_for_refit": False,
        "primary_evaluation_available": False, "primary_data_retained": False,
        "required_stages": required, "requested_stages": requested,
        "required_stages_completed": False, "all_requested_stages_completed": False,
        "stages": {name: {"requested": name in requested, "required": name in required,
                           "status": "pending" if name in requested else "not_requested", "path": name}
                   for name in _STAGES},
        "workload": {"shape_count": len(plan["cases"]), "dtypes": list(plan["dtypes"]),
                     "split_counts": dict(sorted(Counter(case["split"] for case in plan["cases"]).items()))},
        "repeat_plan": {"enabled": include_repeat, "fraction": 0.125,
                        "selected_shapes": len(repeated), "case_ids": repeated,
                        "selection_uses_latency_or_prediction_errors": False,
                        "same_measurement_manifest": True},
        "measurement_quality": {}, "measurement_quality_passed": None, "checkpoint_warnings": [],
        "scope": "Fresh measurements under one declared workload and environment; no historical replay, cross-GPU or serving claim.",
        "simulation_scope": "Explicit software checks do not establish CUDA execution, numerical correctness or GPU predictive accuracy.",
    }
    write_json(out / "workload_manifest.json", plan, overwrite=False)

    def persist():
        summary["required_stages_completed"] = all(summary["stages"][name]["status"] == "completed" for name in required)
        summary["all_requested_stages_completed"] = all(summary["stages"][name]["status"] == "completed" for name in requested)
        expected_quality = ["primary"] + (["repeat"] if include_repeat else [])
        observed = [summary["measurement_quality"].get(name, {}).get("passed") for name in expected_quality]
        summary["measurement_quality_passed"] = (False if False in observed else True if all(value is True for value in observed) else None)
        raw_path = out / "primary" / "dataset" / "raw_visits.jsonl"
        summary["primary_data_retained"] = raw_path.is_file() and raw_path.stat().st_size > 0
        summary["updated_at_utc"] = utc_now()
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {key: summary[key] for key in (
            "protocol_version", "session_id", "measurement_kind", "status", "active_stage", "updated_at_utc",
            "required_stages_completed", "all_requested_stages_completed", "primary_evaluation_available",
            "primary_data_retained", "measurement_quality_passed", "stages",
        )} | {key: summary[key] for key in ("error_type", "error", "finished_at_utc") if key in summary})
        render_fresh_report(out)

    def start(name):
        summary["active_stage"] = name
        summary["stages"][name].update(status="running", started_at_utc=utc_now())
        persist()
        if progress:
            progress(f"[{session_id}] stage {name}: {_LABELS[name]}")

    def save_checkpoint(event):
        if checkpoint is not None:
            try:
                state = copy.deepcopy(summary)
                state["checkpoint_event"] = event
                checkpoint(state)
            except Exception as exc:
                summary["checkpoint_warnings"].append({"stage": summary["active_stage"], "event": event,
                                                       "required": checkpoint_required,
                                                       "error_type": type(exc).__name__, "error": str(exc)})
                persist()
                if progress:
                    progress(f"[{session_id}] checkpoint export failed at {event}; retained evidence is unchanged: {exc}")
                if checkpoint_required:
                    raise

    def finish(name, details):
        summary["stages"][name].update(details, status="completed", finished_at_utc=utc_now())
        persist()
        save_checkpoint("stage_completed_" + name)

    persist()
    try:
        if checkpoint_required:
            save_checkpoint("experiment_initialized")
        start("primary")
        primary = experiment.run_experiment(plan, out / "primary", session_id=session_id + "_primary",
                                            simulate=simulate, device=device, analyze=True, progress=progress,
                                            **({"collection_checkpoint": collection_checkpoint}
                                               if collection_checkpoint is not None else {}))
        _success_kind(primary, simulate)
        primary_audit = _audit_stage(out / "primary", simulate)
        analysis = read_json(out / "primary" / "analysis" / "summary.json")
        if analysis.get("status") != "completed" or analysis.get("measurement_kind") != kind:
            raise LabError("Primary analysis did not complete with the requested measurement identity")
        summary["primary_evaluation_available"] = True
        summary["measurement_quality"]["primary"] = primary_audit.get("measurement_quality", {})
        count = _write_selected_metrics(out / "primary" / "analysis" / "metrics.csv", out / "selected_model_metrics.csv")
        finish("primary", {"audit": {"passed": True, "checks": primary_audit["record_validation"]["checks"],
                                     "shapes": primary_audit["distinct_shapes"], "rows": primary_audit["measurement_rows"]},
                           "analyzed_groups": analysis["analyzed_group_count"], "selected_metric_rows": count,
                           "analysis_status": analysis["status"], "measurement_kind": kind})

        if include_repeat:
            start("repeat")
            repeat = experiment.run_experiment(plan, out / "repeat", session_id=session_id + "_repeat",
                                               simulate=simulate, device=device, analyze=False,
                                               repeat_ids=repeated, progress=progress,
                                               **({"collection_checkpoint": collection_checkpoint}
                                                  if collection_checkpoint is not None else {}))
            _success_kind(repeat, simulate)
            repeat_audit = _audit_stage(out / "repeat", simulate)
            summary["measurement_quality"]["repeat"] = repeat_audit.get("measurement_quality", {})
            finish("repeat", {"audit": {"passed": True, "checks": repeat_audit["record_validation"]["checks"],
                                        "shapes": repeat_audit["distinct_shapes"], "rows": repeat_audit["measurement_rows"]},
                              "refitted": False, "measurement_kind": kind})
            start("repeat_comparison")
            comparison = experiment.compare_datasets(out / "primary", out / "repeat", out / "repeat_comparison",
                                                     mode="repeat", progress=progress)
            if comparison.get("status") != "completed" or comparison.get("measurement_kind") != kind:
                raise LabError("Repeat comparison did not complete with the requested measurement identity")
            summary["repeat_comparison"] = {key: comparison[key] for key in (
                "groups", "same_physical_gpu_verified", "source_device", "target_device", "target_labels_used_for_refit",
            )}
            finish("repeat_comparison", {"compared_groups": len(comparison["groups"]), "target_refitted": False,
                                         "same_physical_gpu_verified": comparison["same_physical_gpu_verified"]})

        if include_layer:
            start("layer")
            if simulate:
                summary["stages"]["layer"].update(status="unavailable_simulation", finished_at_utc=utc_now(),
                    reason="A decoder-layer check requires actual CUDA measurements; it is never replaced with a simulated layer score")
            else:
                try:
                    layer = experiment.validate_layer(out / "primary", out / "layer", model_id=layer_model_id,
                                                      dtype=layer_dtype, device=device, progress=progress)
                    if layer.get("status") == "completed":
                        summary["layer_comparison"] = layer.get("comparison", {})
                        finish("layer", {"model_id": layer_model_id, "dtype": layer_dtype,
                                         "quality_flags": layer.get("layer_quality_flags", {})})
                    else:
                        summary["stages"]["layer"].update(status=layer.get("status", "incomplete"),
                            finished_at_utc=utc_now(), reason=layer.get("reason", "Requested layer check did not finish"))
                except Exception as exc:
                    if checkpoint_required and summary["stages"]["layer"]["status"] == "completed":
                        # The measured layer finished; a required storage
                        # failure must propagate without relabeling its data.
                        raise
                    summary["stages"]["layer"].update(status="failed", finished_at_utc=utc_now(),
                                                       error_type=type(exc).__name__, error=str(exc))
            if summary["stages"]["layer"]["status"] != "completed":
                summary["status"] = "incomplete"
                summary["error"] = "The primary workflow completed, but the requested optional layer check did not complete."

        if summary["status"] != "incomplete":
            summary["status"] = "simulated_complete" if simulate else "completed"
        summary.update(active_stage=None, finished_at_utc=utc_now())
        persist()
        if checkpoint_required:
            save_checkpoint("experiment_finished")
        return copy.deepcopy(summary)
    except BaseException as exc:
        name = summary.get("active_stage")
        interrupted = isinstance(exc, KeyboardInterrupt)
        summary.update(status="interrupted" if interrupted else "failed", error_type=type(exc).__name__,
                       error=str(exc) or type(exc).__name__, finished_at_utc=utc_now())
        if name is not None and summary["stages"][name]["status"] != "completed":
            summary["stages"][name].update(status=summary["status"], finished_at_utc=summary["finished_at_utc"],
                                           error_type=type(exc).__name__, error=summary["error"])
        for name in requested:
            if summary["stages"][name]["status"] == "pending":
                summary["stages"][name].update(status="not_run", reason="A preceding required stage did not finish")
        persist()
        raise
