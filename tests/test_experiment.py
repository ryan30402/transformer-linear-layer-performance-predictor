"""Offline workflow tests: acquisition order is independent of data permissions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from linear_perf import cli, experiment
from linear_perf.common import LabError, read_json, read_jsonl, write_json, write_jsonl


class ExperimentWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        from linear_perf.workloads import build_workload_manifest
        plan = build_workload_manifest(profile="smoke", dtypes=("float16",))
        plan.setdefault("measurement", {}).update(profile=False, rotation_enabled=False)
        return plan

    def collect(self):
        from linear_perf.dataset_measurement import collect_dataset
        out = self.root / "dataset"
        collect_dataset(self.manifest(), out, session_id="workflow_fixture", simulate=True, progress=None)
        return out

    def test_existing_test_labels_are_permitted_but_never_supplied_to_fitter(self):
        data = self.collect()
        records = read_jsonl(data / "measurements.jsonl")
        self.assertTrue(any(r["split"] == "test" for r in records))
        received = []

        def fitter(train, **kwargs):
            self.assertTrue((data / "measurements.jsonl").is_file())
            self.assertEqual({row["split"] for row in train}, {"train"})
            received.append({row["case_id"] for row in train})
            return {"selected_model_id": "fixture", "models": {"fixture": {}}, "fixture_only": True}

        def evaluate(suite, targets, **kwargs):
            self.assertTrue(all(r["split"] in {"test", "challenge", "stress"} for r in targets))
            return {"selected_model_id": "fixture", "models": {"fixture": {"mean_ape_pct": 0.0, "predictions": []}}}

        with patch("linear_perf.selection.fit_cv_suite", side_effect=fitter), \
                patch("linear_perf.selection.evaluate_cv_suite", side_effect=evaluate):
            result = experiment.analyze_dataset(data, self.root / "analysis", progress=None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0], {r["case_id"] for r in records if r["split"] == "train"})
        self.assertFalse((data / "freeze.json").exists())
        self.assertTrue((self.root / "analysis/report.html").is_file())

    def test_split_tampering_is_rejected_before_fitting(self):
        data = self.collect()
        rows = read_jsonl(data / "measurements.jsonl")
        held = next(row for row in rows if row["split"] == "test")
        held["split"] = "train"
        write_jsonl(data / "measurements.jsonl", rows)
        with patch("linear_perf.selection.fit_cv_suite") as fitter, self.assertRaises(LabError):
            experiment.analyze_dataset(data, self.root / "blocked", progress=None)
        fitter.assert_not_called()

    def test_failed_collection_is_not_analyzable(self):
        data = self.collect()
        status = read_json(data / "status.json")
        status["status"] = "failed"
        write_json(data / "status.json", status)
        with self.assertRaisesRegex(LabError, "did not complete"):
            experiment.load_dataset(data)

    def test_default_cli_run_uses_dataset_first_workflow(self):
        config = self.root / "plan.json"
        write_json(config, self.manifest())
        with patch("linear_perf.experiment.run_experiment", return_value={"status": "completed"}) as run, \
                patch("linear_perf.campaign.run_campaign", side_effect=AssertionError("Legacy acquisition order is not the default")):
            code = cli.main(["run", "--config", str(config), "--out", str(self.root / "run"),
                             "--session-id", "default", "--simulate"])
        self.assertEqual(code, 0)
        self.assertTrue(run.call_args.kwargs["analyze"])

    def test_completed_dataset_audit_requires_no_fitter_or_gpu(self):
        data = self.collect()
        with patch("linear_perf.selection.fit_cv_suite", side_effect=AssertionError("Audit must not refit")), \
                patch("linear_perf.benchmark._load_torch", side_effect=AssertionError("Audit must not execute CUDA")):
            checked = experiment.audit_dataset(data)
        self.assertTrue(checked["passed"])
        self.assertFalse(checked["freeze_before_measurement_required"])


if __name__ == "__main__":
    unittest.main()
