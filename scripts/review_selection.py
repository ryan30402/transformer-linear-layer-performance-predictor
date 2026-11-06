#!/usr/bin/env python3
"""Recompute the design-review claims without changing the published snapshot.

Panel A changes only the development-set selection criterion over the five
already fitted, previously median-tuned candidates. Panel B runs the new grouped
CV procedure and refits every candidate on all 36 calibration/development rows.
The old test labels were known during this methodological revision: neither
panel is a fresh independent test of the new procedure. This is CPU replay and
refitting of existing GPU measurements, not a new GPU benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from linear_perf import models as recorded_models  # noqa: E402
from linear_perf import selection  # noqa: E402
from linear_perf.common import LabError, digest  # noqa: E402

from export_results import SESSION_IDS  # noqa: E402
from verify_published_results import read_json, read_jsonl, replay_envelope, verify  # noqa: E402

METRIC_KEYS = ("n", "mean_ape_pct", "median_ape_pct", "p90_ape_pct", "max_ape_pct", "mae_ms", "rmse_ms")
CRITERIA = ("median_ape_pct", "mean_ape_pct", "p90_ape_pct", "max_ape_pct")
SCOPE = "Retrospective analysis of previously published, already inspected test labels; no new held-out accuracy claim."


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise LabError("Cannot write an empty review table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _snapshot_hashes(results: Path) -> dict[str, str]:
    """Check that the complete recorded directory is untouched by this replay."""
    return {str(path.relative_to(results)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(results.rglob("*")) if path.is_file()}


def _offline_row(row: dict, *, training: bool) -> dict:
    """Explicit, narrow adapter from the historical protocol to the new one."""
    allowed = {"calibration", "development"} if training else {"test", "challenge"}
    if row["split"] not in allowed:
        raise LabError("Only historical calibration/development labels may be adapted to train")
    # Full raw-evidence verification occurs before this adapter. The new model
    # receives the retained shape target, not traces, controls or candidate data.
    return {
        **{key: row[key] for key in ("case_id", "m", "n", "k", "median_ms", "dtype",
                                     "cache_policy", "environment_id", "measurement_kind")},
        "split": "train" if training else row["split"],
        "original_split": row["split"], "weight": 1.0,
        "status": row["status"], "correctness": {"passed": row["correctness"]["passed"]},
    }


def _headline(metrics: dict) -> dict:
    return {key: metrics[key] for key in METRIC_KEYS}


def _old_winner(candidate_metrics: dict, criterion: str) -> str:
    # Median reproduces the historical selector exactly. For the alternate
    # tail-first criterion, mean is the second tie-breaker; all ties are fixed.
    secondary = "mean_ape_pct" if criterion == "p90_ape_pct" else "p90_ape_pct"
    return min(recorded_models.MODEL_ORDER, key=lambda name: (
        candidate_metrics[name][criterion], candidate_metrics[name][secondary],
        recorded_models.MODEL_ORDER.index(name),
    ))


def analyze_session(directory: Path, output: Path, *, include_details: bool = True) -> tuple[dict, list[dict], list[dict], list[dict]]:
    sid = directory.name
    saved = read_json(directory / "model_suite.json")
    environment = read_json(directory / "environment.json")
    initial = read_jsonl(directory / "initial_measurements.jsonl")
    heldout = read_jsonl(directory / "heldout_measurements.jsonl")
    counts = Counter(row["split"] for row in initial + heldout)
    if counts != Counter(calibration=24, development=12, test=24, challenge=12):
        raise LabError("This review adapter expects the published 24/12/24/12 split")
    dev = sorted((row for row in initial if row["split"] == "development"), key=lambda row: row["case_id"])
    split_rows = {"development": dev, **{
        split: sorted((row for row in heldout if row["split"] == split), key=lambda row: row["case_id"])
        for split in ("test", "challenge")
    }}
    old_scores, development_table = {}, []
    for name in recorded_models.MODEL_ORDER:
        saved_model = saved["models"][name]
        if len(saved_model["train_case_ids"]) != 24:
            raise LabError("Criterion-only replay must retain the original 24-label candidate fit")
        model = replay_envelope(saved_model)
        old_scores[name] = {}
        for split, rows in split_rows.items():
            predicted = recorded_models.predict_model(model, rows, saved["dtype"])
            score = recorded_models.evaluate_predictions(rows, predicted)
            old_scores[name][split] = score
            if split == "development":
                for key in METRIC_KEYS:
                    if not np.isclose(score[key], saved["development_metrics"][name][key], rtol=1e-9, atol=1e-10):
                        raise LabError("Stored development score does not replay")
        score = old_scores[name]["development"]
        errors = {row["case_id"]: row["ape_pct"] for row in score["predictions"]}
        severe = [row["case_id"] for row in dev if errors[row["case_id"]] > 50]
        irregular = [row["case_id"] for row in dev
                     if (row["n"] % 16 or row["k"] % 16) and errors[row["case_id"]] > 50]
        development_table.append({
            "session_id": sid, "dtype": saved["dtype"], "model_id": name,
            "n_fit": 24, **_headline(score), "ape_gt_50_count": len(severe),
            "ape_gt_50_case_ids": " ".join(severe),
            "irregular_ape_gt_50_count": len(irregular),
            "irregular_ape_gt_50_case_ids": " ".join(irregular),
            "scope": "Criterion-only replay: candidate hyperparameters and fit remain unchanged.",
        })
    dev_scores = {name: values["development"] for name, values in old_scores.items()}
    choices, criterion_table = {}, []
    for criterion in CRITERIA:
        winner = _old_winner(dev_scores, criterion)
        choices[criterion] = winner
        row = {"session_id": sid, "dtype": saved["dtype"], "criterion": criterion,
               "selected_model_id": winner, "n_fit": 24, "n_development": 12,
               "original_selection": saved["selected_model_id"], "retrospective": True}
        for split in split_rows:
            row.update({f"{split}_{key}": value for key, value in _headline(old_scores[winner][split]).items()})
        criterion_table.append(row)
    if choices["median_ape_pct"] != saved["selected_model_id"]:
        raise LabError("Historical median/P90 selection does not replay")

    train = [_offline_row(row, training=True) for row in initial]
    test = [_offline_row(row, training=False) for row in heldout]
    sm_count = environment["identity"]["device"].get("multiprocessor_count")
    # This data set has no production-frequency measurements. Uniform weights
    # preserve the released equal-shape estimand, and do not fabricate traffic.
    suite = selection.fit_cv_suite(train, sm_count=sm_count)
    evaluation = selection.evaluate_cv_suite(suite, test)
    cv_table = []
    for name in selection.MODEL_ORDER:
        cv_record = suite["cv"]["models"][name]
        for split, score in (("train_oof_tuning", cv_record["oof_metrics"]),
                             ("test", evaluation["group_metrics"]["split:test"][name]),
                             ("challenge", evaluation["group_metrics"]["split:challenge"][name])):
            cv_table.append({
                "session_id": sid, "dtype": saved["dtype"], "model_id": name,
                "split": split, "selected_by_grouped_cv": name == suite["selected_model_id"],
                "n_refit": len(train), "cv_families": suite["cv"]["family_count"],
                "cv_folds": suite["cv"]["fold_count"], "sm_count": sm_count,
                "configuration": json.dumps(suite["models"][name]["configuration"], sort_keys=True),
                **_headline(score), "weighted_mean_ape_pct": score["weighted_mean_ape_pct"],
                "weighted_p90_ape_pct": score["weighted_p90_ape_pct"],
                "cv_fold_primary_mean": cv_record["fold_score_mean"],
                "cv_fold_primary_std": cv_record["fold_score_std"], "retrospective": True,
            })
    if include_details:
        _write_json(output / sid / "cv_refit_suite.json", suite)
        _write_json(output / sid / "cv_refit_evaluation.json", {"retrospective_scope": SCOPE, **evaluation})
    report = {
        "session_id": sid, "dtype": saved["dtype"], "sm_count": sm_count,
        "scope": SCOPE, "source_environment_id": environment["environment_id"],
        "criterion_only": {
            "n_fit": 24, "n_development": 12, "choices": choices,
            "candidate_scores": old_scores,
            "hyperparameters_changed": False, "models_refitted": False,
            "interpretation": "Only the final selector changes; previous median-based hyperparameter tuning remains embedded in the candidates.",
        },
        "grouped_cv_refit": {
            "n_train": len(train), "training_split_adapter": {"calibration": "train", "development": "train"},
            "candidate_pool_used": False, "test_or_challenge_used_for_fit_or_selection": False,
            "selected_model_id": suite["selected_model_id"], "suite_hash": suite["suite_hash"],
            "selection_score_key": suite["selection_score_key"],
            "cv_family_count": suite["cv"]["family_count"], "cv_fold_count": suite["cv"]["fold_count"],
            "all_candidates": cv_table,
            "interpretation": "This panel changes the CV/refit protocol, shared features and robust fitting together. It is not a causal ablation of the selection metric alone.",
        },
        "weight_policy": "Uniform weight per shape: the historical set has no measured production workload frequencies.",
    }
    if include_details:
        _write_json(output / sid / "review.json", report)
    return report, criterion_table, development_table, cv_table


def _markdown_report(summary: dict, criteria: list[dict], cv: list[dict]) -> str:
    lines = [
        "# Retrospective selection review", "",
        "These tables recompute already published A100 measurements. The old test and challenge labels were known during the methodology revision. The numbers describe known data; they are **not a fresh independent validation of the revised procedure**. No GPU measurements were added.", "",
        "## Two distinct comparisons", "",
        "1. **Criterion-only replay:** keep the original five fitted candidates, their 24 calibration labels, and their previously median-tuned hyperparameters unchanged. Change only the final selection criterion on the 12 development cases. This comparison does not retune hyperparameters.",
        "2. **Grouped CV and refit:** explicitly map the 24 calibration plus 12 development rows to 36 training rows. Select hyperparameters and a model family using leave-one-(N,K)-family-out CV across six families, then refit every family's chosen configuration on all 36 rows. Evaluate the old test and challenge rows afterwards. Candidate-pool labels are excluded.", "",
        "The second comparison also changes shared geometry/wave-proxy features and robust physical fitting. Its differences cannot be attributed to one design change. All historical rows use warm buffer reuse, and each shape has weight 1. These weights represent the released equal-shape benchmark, not production traffic.", "",
        "## Check of the review's selection claim", "",
        "| Development selection criterion | Sessions selecting residual KRR |", "|---|---:|",
    ]
    for key in CRITERIA:
        lines.append(f"| {key} | {summary['criterion_only_krr_selection_counts'][key]} / {summary['sessions']} |")
    lines.extend([
        "", "In each BF16 session, the saved three-parameter roofline candidate has 5 of 12 development cases above 50% APE; four of those cases have N or K not divisible by 16. Its development median is approximately 13%, while its mean is approximately 30%. The unchanged-candidate replay supports the criticism that median selection concealed a consequential tail in this data. It does not establish that the selector is the sole cause of the original errors.", "",
        "## Old test results, with the comparison separated", "",
        "All errors below are percentages. The original and criterion-only columns retain the 24-label fits. The CV/refit columns use 36 training labels. Conventional median and P90 are unweighted NumPy percentiles.", "",
        "| Session | Original median | Criterion-only mean-selector median | CV/refit mean | CV/refit median | CV/refit P90 | CV/refit worst |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for sid in SESSION_IDS:
        original = next(row for row in criteria if row["session_id"] == sid and row["criterion"] == "median_ape_pct")
        alternate = next(row for row in criteria if row["session_id"] == sid and row["criterion"] == "mean_ape_pct")
        revised = next(row for row in cv if row["session_id"] == sid and row["split"] == "test" and row["selected_by_grouped_cv"])
        values = [original["test_median_ape_pct"], alternate["test_median_ape_pct"],
                  revised["mean_ape_pct"], revised["median_ape_pct"], revised["p90_ape_pct"], revised["max_ape_pct"]]
        lines.append(f"| {sid} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines.extend([
        "", "## Old challenge results after CV/refit", "",
        "| Session | Selected family | Mean APE | Median APE | P90 APE | Worst APE |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for sid in SESSION_IDS:
        row = next(row for row in cv if row["session_id"] == sid and row["split"] == "challenge" and row["selected_by_grouped_cv"])
        lines.append(f"| {sid} | {row['model_id']} | " + " | ".join(f"{row[key]:.2f}" for key in
                      ("mean_ape_pct", "median_ape_pct", "p90_ape_pct", "max_ape_pct")) + " |")
    lines.extend([
        "", "The revised selector chooses residual KRR in all six sessions. Test mean and median errors decrease in this retrospective comparison, but the BF16 test worst case remains approximately 65%; BF16 challenge worst cases remain approximately 58%. Lower typical error therefore does not establish reliable tail behavior or deployment readiness.", "",
        "## Interpretation and provenance", "",
        "The CV metric is a tuning score, not an unbiased winner accuracy estimate. `cv_refit_all_models.csv` retains every candidate, pooled out-of-fold scores, fold mean/sample standard deviation, and full test/challenge tails. Fold spread is descriptive and is not a confidence interval. Repeated sessions use the same shapes, so they are not independent workload samples.", "",
        f"The existing CPU snapshot verifier passed {summary['source_verification']['checks']:,} checks. Every file under `results/` remained byte-identical before and after replay. The verification checks retained evidence and does not independently attest to GPU execution.", "",
        "The revised experiment still needs newly collected workload-derived cases and prospective validation under its declared measurement/cache contract. Neither the old public test set nor this retrospective report can supply that evidence.", "",
        "## Reproduce", "",
        "Run from the repository root with the project's NumPy/SciPy dependencies:", "",
        "```bash", "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/review_selection.py --compact --output runs/review_analysis", "```", "",
        "Omit `--compact` to retain each refitted suite and detailed per-case evaluation in the output directory. The default output is ignored by Git; the concise reviewed copy is stored here. The script rejects output paths inside the immutable recorded-results tree.", "",
        "Files: `criterion_only_selection.csv`, `development_candidate_metrics.csv`, `cv_refit_all_models.csv`, `summary.json`, and `source_verification.json`.", "",
    ])
    return "\n".join(lines)


def analyze(results: Path, output: Path, *, include_details: bool = True) -> dict:
    results, output = results.resolve(), output.resolve()
    if output.is_relative_to(results) or results.is_relative_to(output):
        raise LabError("Place review output outside the recorded results tree and its ancestors")
    before = _snapshot_hashes(results)
    verification = verify(results)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "source_verification.json", verification)
    reports, criteria, development, cv = [], [], [], []
    for sid in SESSION_IDS:
        report, a, b, c = analyze_session(results / "a100" / sid, output, include_details=include_details)
        reports.append(report)
        criteria.extend(a)
        development.extend(b)
        cv.extend(c)
        print(f"{sid}: stored={report['criterion_only']['choices']['median_ape_pct']}, "
              f"dev-mean={report['criterion_only']['choices']['mean_ape_pct']}, "
              f"CV-refit={report['grouped_cv_refit']['selected_model_id']}", flush=True)
    after = _snapshot_hashes(results)
    if before != after:
        raise LabError("Published result bytes changed during retrospective replay")
    _write_csv(output / "criterion_only_selection.csv", criteria)
    _write_csv(output / "development_candidate_metrics.csv", development)
    _write_csv(output / "cv_refit_all_models.csv", cv)
    summary = {
        "schema_version": 1, "analysis_type": "retrospective_design_review_recomputation",
        "scope": SCOPE, "sessions": len(reports), "gpu_operations_performed": 0,
        "recorded_results_unchanged": True, "recorded_results_hash": digest(before),
        "source_verification": verification,
        "criterion_only_krr_selection_counts": {
            key: sum(report["criterion_only"]["choices"][key] == "roofline_residual_krr" for report in reports)
            for key in CRITERIA
        },
        "bf16_roofline_3p_development_tail": [row for row in development
                                             if row["dtype"] == "bfloat16" and row["model_id"] == "roofline_3p"],
        "cv_refit_selected_models": {report["session_id"]: report["grouped_cv_refit"]["selected_model_id"]
                                     for report in reports},
        "interpretation_limits": [
            "Old test/challenge labels were already inspected before the modeling revision; scores describe known data.",
            "Criterion-only comparisons reuse originally median-tuned candidates and do not retune their hyperparameters.",
            "New grouped CV/refitting changes several design choices, so accuracy differences cannot be assigned to one cause.",
            "CV winner scores are tuning scores; fold spread is not a confidence interval for deployment error.",
            "Three sessions per dtype reuse the same shapes and are not six independent workload samples.",
            "All historical data in this analysis use warm buffer reuse; no rotation-cache accuracy claim is made.",
        ],
        "source_paths": ["results/a100/" + sid for sid in SESSION_IDS],
    }
    _write_json(output / "summary.json", summary)
    (output / "README.md").write_text(_markdown_report(summary, criteria, cv), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "review_analysis")
    parser.add_argument("--compact", action="store_true", help="Write tables and summary without per-session suite/evaluation copies")
    args = parser.parse_args()
    result = analyze(args.results, args.output, include_details=not args.compact)
    print(json.dumps({"status": "completed", "output": str(args.output),
                      "criterion_only_krr_selection_counts": result["criterion_only_krr_selection_counts"],
                      "cv_refit_selected_models": result["cv_refit_selected_models"], "scope": SCOPE}, indent=2))


if __name__ == "__main__":
    main()
