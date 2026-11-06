"""Small, complete CPU workflow checks using explicitly simulated observations.

The source suites are really fitted once per test class. Only comparison-time
fitters are mocked, so the no-refit assertion cannot pass with an empty model
fixture. No CUDA execution or hardware performance claim is involved.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from linear_perf import benchmark, dataset_measurement, experiment, selection
from linear_perf.common import LabError, read_json, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def csv_records(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def tree_hashes(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in Path(path).rglob("*") if file.is_file()}


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


class OfflineExperimentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.storage.cleanup)
        cls.root = Path(cls.storage.name)
        cls.plan = read_json(ROOT / "configs/workloads_smoke.json")
        cls.source, cls.target = cls.root / "source", cls.root / "repeat"
        cls.repeat_ids = dataset_measurement.repeat_subset(cls.plan)
        with patch.object(benchmark, "_load_torch", side_effect=AssertionError("CPU workflow imported CUDA")):
            cls.source_result = experiment.run_experiment(
                cls.plan, cls.source, session_id="integration_source", simulate=True, progress=None,
            )
            cls.target_result = experiment.run_experiment(
                cls.plan, cls.target, session_id="integration_repeat", simulate=True,
                analyze=False, repeat_ids=cls.repeat_ids, progress=None,
            )
        cls.summary = read_json(cls.source / "analysis/summary.json")

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.workspace.cleanup)
        self.work = Path(self.workspace.name)

    def copy_source(self):
        return Path(shutil.copytree(self.source, self.work / "source"))

    def test_complete_source_and_repeat_keep_precision_cache_and_split_contracts(self):
        source_audit = experiment.audit_dataset(self.source)
        target_audit = experiment.audit_dataset(self.target)
        self.assertTrue(source_audit["passed"] and target_audit["passed"])
        self.assertEqual(source_audit["measurement_kind"], "simulated")
        self.assertFalse(source_audit["freeze_before_measurement_required"])
        self.assertEqual(source_audit["distinct_shapes"], len(self.plan["cases"]))
        self.assertEqual(target_audit["distinct_shapes"], math.ceil(len(self.plan["cases"]) * 0.125))
        self.assertEqual(read_json(self.source / "dataset/manifest.json"), read_json(self.target / "dataset/manifest.json"))
        status = read_json(self.target / "dataset/status.json")
        self.assertEqual(set(status["selected_case_ids"]), set(self.repeat_ids))
        self.assertFalse(status["model_fitting_performed"])
        self.assertFalse((self.target / "analysis").exists())
        groups = self.summary["groups"]
        self.assertEqual({(item["dtype"], item["cache_policy"]) for item in groups},
                         {(dtype, policy) for dtype in ("float16", "bfloat16")
                          for policy in ("warm_reuse", "rotating_buffers")})
        self.assertTrue(all(item["status"] == "analyzed" for item in groups))
        self.assertTrue(self.summary["test_labels_measured_before_fitting"])
        self.assertFalse(self.summary["heldout_labels_used_for_selection"])

    def test_csv_and_html_report_agree_with_independently_recomputed_prediction_errors(self):
        folder = self.source / "analysis"
        metrics, predictions = csv_records(folder / "metrics.csv"), csv_records(folder / "predictions.csv")
        self.assertTrue(metrics and predictions)
        by_group = {}
        for prediction in predictions:
            key = tuple(prediction[field] for field in ("group_id", "split", "model_id"))
            by_group.setdefault(key, []).append(prediction)
        for row in metrics:
            key = tuple(row[field] for field in ("group_id", "split", "model_id"))
            records = by_group[key]
            actual = [float(item["actual_ms"]) for item in records]
            predicted = [float(item["predicted_ms"]) for item in records]
            errors = [abs(p / a - 1) * 100 for a, p in zip(actual, predicted)]
            weights = [float(item["weight"]) for item in records]
            self.assertEqual(int(row["n"]), len(records))
            self.assertAlmostEqual(float(row["mean_ape_pct"]), math.fsum(errors) / len(errors), places=10)
            self.assertAlmostEqual(float(row["weighted_mean_ape_pct"]),
                                   math.fsum(w * error for w, error in zip(weights, errors)) / math.fsum(weights), places=10)
            self.assertAlmostEqual(float(row["max_ape_pct"]), max(errors), places=10)
            self.assertAlmostEqual(float(row["rmse_ms"]),
                                   math.sqrt(math.fsum((p - a)**2 for a, p in zip(actual, predicted)) / len(actual)), places=12)
        html = (folder / "report.html").read_text(encoding="utf-8")
        table = TableReader()
        table.feed(html)
        self.assertIn("SIMULATED SOFTWARE CHECK", html)
        self.assertEqual(len(table.rows) - 1, len(metrics))
        columns = table.rows[0]
        self.assertEqual(table.rows[1:], [[row[column] for column in columns] for row in metrics])

    def test_reanalysis_of_existing_measurements_reproduces_all_fits_and_predictions(self):
        original_data = tree_hashes(self.source / "dataset")
        output = self.work / "reanalysis"
        result = experiment.analyze_dataset(self.source, output, progress=None)
        self.assertEqual(result["groups"], self.summary["groups"])
        for item in self.summary["groups"]:
            relative = item["model_suite_file"]
            self.assertEqual((output / relative).read_bytes(), (self.source / "analysis" / relative).read_bytes())
        for name in ("metrics.csv", "predictions.csv"):
            self.assertEqual((output / name).read_bytes(), (self.source / "analysis" / name).read_bytes())
        self.assertEqual(tree_hashes(self.source / "dataset"), original_data)

    def test_stale_or_incomplete_source_analysis_is_rejected_before_comparison(self):
        for field, value in (("manifest_hash", "0" * 64), ("status", "failed")):
            source = Path(shutil.copytree(self.source, self.work / field))
            summary = read_json(source / "analysis/summary.json")
            summary[field] = value
            write_json(source / "analysis/summary.json", summary)
            output = self.work / (field + "_comparison")
            with self.subTest(field=field), self.assertRaisesRegex(LabError, "incomplete or belongs to another workload"):
                experiment.compare_datasets(source, self.target, output, progress=None)
            self.assertFalse(output.exists())

    def test_edited_aggregate_cannot_be_audited_or_used_for_training(self):
        source = self.copy_source()
        path = source / "dataset/measurements.jsonl"
        rows = read_jsonl(path)
        rows[0]["median_ms"] *= 2
        write_jsonl(path, rows)
        with self.assertRaisesRegex(LabError, "Aggregate does not reproduce"):
            experiment.audit_dataset(source)
        output = self.work / "invalid_analysis"
        with patch.object(selection, "fit_cv_suite", side_effect=AssertionError("fitting invalid data")):
            with self.assertRaisesRegex(LabError, "Aggregate does not reproduce"):
                experiment.analyze_dataset(source, output, progress=None)
        self.assertFalse(output.exists())

    def test_changed_raw_operation_divisor_is_rejected(self):
        source = self.copy_source()
        path = source / "dataset/raw_visits.jsonl"
        rows = read_jsonl(path)
        original = rows[0]["captured_gemm_count"]
        rows[0]["captured_gemm_count"] = original // 2 if original >= 8 else original * 2
        write_jsonl(path, rows)
        with self.assertRaisesRegex(LabError, "Event duration was not divided by its recorded operation count"):
            experiment.audit_dataset(source)

    def test_validly_serialized_suite_from_different_training_targets_is_rejected(self):
        source = self.copy_source()
        item = next(item for item in self.summary["groups"] if item["status"] == "analyzed")
        rows = read_jsonl(source / "dataset/measurements.jsonl")
        train = [copy.deepcopy(row) for row in rows if row["split"] == "train" and
                 (row["environment_id"], row["dtype"], row["cache_policy"]) ==
                 (item["environment_id"], item["dtype"], item["cache_policy"])]
        for row in train:
            row["median_ms"] *= 1.1
        foreign_suite = selection.fit_cv_suite(train, sm_count=train[0]["sm_count"])
        selection._check_suite(foreign_suite)  # This is a complete valid fit, not a damaged JSON checksum.
        write_json(source / "analysis" / item["model_suite_file"], foreign_suite)
        with self.assertRaisesRegex(LabError, "does not match its recorded training rows"):
            experiment.compare_datasets(source, self.target, self.work / "foreign_comparison", progress=None)

    def test_comparison_reuses_real_source_suites_and_never_refits_on_target_labels(self):
        before_source, before_target = tree_hashes(self.source), tree_hashes(self.target)
        output = self.work / "comparison"
        with ExitStack() as stack:
            for name in ("fit_cv_suite", "_fit_physical", "_fit_learned"):
                stack.enter_context(patch.object(selection, name, side_effect=AssertionError("target-time refit")))
            result = experiment.compare_datasets(self.source, self.target, output, mode="repeat", progress=None)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["target_labels_used_for_refit"])
        self.assertFalse(result["same_physical_gpu_verified"])
        self.assertTrue(result["groups"])
        source_items = {(item["dtype"], item["cache_policy"]): item for item in self.summary["groups"]}
        for group in result["groups"]:
            item = source_items[group["dtype"], group["cache_policy"]]
            suite = read_json(self.source / "analysis" / item["model_suite_file"])
            self.assertEqual(group["source_model_id"], suite["selected_model_id"])
            pairs = csv_records(output / group["repeat_file"])
            self.assertEqual(len(pairs), len(self.repeat_ids))
            for pair in pairs:
                self.assertAlmostEqual(float(pair["target_over_source"]),
                                       float(pair["target_ms"]) / float(pair["source_ms"]), places=12)
            scores = read_json(output / group["transfer_file"])
            self.assertTrue(scores)
            for evaluation in scores.values():
                self.assertEqual(evaluation["suite_hash"], suite["suite_hash"])
                self.assertFalse(evaluation["refitted_during_evaluation"])
                self.assertEqual(evaluation["selected_model_id"], suite["selected_model_id"])
        self.assertIn("SIMULATED SOFTWARE CHECK", (output / "report.html").read_text(encoding="utf-8"))
        self.assertEqual(tree_hashes(self.source), before_source)
        self.assertEqual(tree_hashes(self.target), before_target)


if __name__ == "__main__":
    unittest.main()
