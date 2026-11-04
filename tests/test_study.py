"""Offline study integrity tests using sealed, explicitly simulated sessions."""

import copy
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from linear_perf import campaign as campaign
from linear_perf import models as models
from linear_perf import study as study
from linear_perf.common import LabError, read_json
from linear_perf.protocol import build_plan
from linear_perf.simulation import simulated_environment
from tests.test_models import aggregate_fixture, scaled_fixture


class CalibrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan()
        cls.env = simulated_environment(cls.plan)
        all_rows = aggregate_fixture(cls.plan)
        cls.train = [row for row in all_rows if row["split"] == "calibration"]
        cls.dev = [row for row in all_rows if row["split"] == "development"]
        cls.test = [row for row in all_rows if row["split"] == "test"]
        cls.suite = models.fit_suite(cls.train, cls.dev, cls.plan, environment=cls.env)

    def test_shape_only_order_is_stable_nested_and_without_duplicate_labels(self):
        order = study._calibration_order(self.plan)
        reversed_plan = copy.deepcopy(self.plan)
        reversed_plan["cases"].reverse()
        self.assertEqual(order, study._calibration_order(reversed_plan))
        self.assertEqual(len(order), 24)
        self.assertEqual(len(set(order)), 24)
        self.assertEqual(order[0], min(row["case_id"] for row in self.train))
        self.assertTrue(set(order[:4]) < set(order[:8]))

    def test_single_log_scale_recovers_known_factor_from_only_four_or_eight_labels(self):
        model = self.suite["models"]["roofline_2p"]
        lookup = {row["case_id"]: row for row in self.train}
        order = study._calibration_order(self.plan)
        for budget in (4, 8):
            rows = [lookup[cid] for cid in order[:budget]]
            predictions = models.predict_model(model, rows, self.plan["dtype"])
            scaled = [scaled_fixture([row], (2.5 * predicted) / row["median_ms"])[0]
                      for row, predicted in zip(rows, predictions)]
            with patch.object(models, "fit_suite", side_effect=AssertionError("must not refit source suite")):
                result = study._fit_scale(model, scaled, self.plan, self.env)
            self.assertAlmostEqual(result["multiplicative_scale"], 2.5)
            self.assertAlmostEqual(result["log_scale"], math.log(2.5))
            self.assertEqual(result["fitted_parameter_count"], 1)
            self.assertFalse(result["protected_labels_used"])
            self.assertEqual(result["calibration_case_ids"], order[:budget])

    def test_scale_rejects_protected_labels_incorrect_budget_or_changed_subset(self):
        model = self.suite["models"]["roofline_2p"]
        for rows in (self.dev[:4], self.test[:4], self.train[:3], self.train[:4]):
            with self.subTest(ids=[r["case_id"] for r in rows]), self.assertRaises(LabError):
                study._fit_scale(model, rows, self.plan, self.env)


class SealedStudyIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.runs = []
        for session, dtype in (("s1", "float16"), ("s2", "float16"), ("s3", "float16"), ("bf16", "bfloat16")):
            output = cls.root / session
            result = campaign.run_campaign(build_plan(dtype), output, session_id=session,
                                           simulate=True, progress=lambda _: None)
            if result.get("status") != "simulated_complete":
                raise AssertionError("CPU simulation fixture did not complete: " + str(result))
            cls.runs.append(output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _out(self, name):
        return self.root / (self._testMethodName + "_" + name)

    def test_three_session_comparison_keeps_all_sessions_and_frozen_source(self):
        output = self._out("study")
        result = study.compare_sessions(self.runs[:3], output)
        self.assertEqual(result["status"], "comparison_complete")
        summary = read_json(output / "summary.json")
        self.assertEqual(summary["measurement_kind"], "simulated")
        self.assertEqual(summary["source_count"], 3)
        self.assertFalse(summary["claims"]["actual_cuda_source_evidence"])
        self.assertFalse(summary["claims"]["session_count_implies_gpu_count"])
        self.assertIsNone(summary["recorded_unique_gpu_uuid_count"])
        self.assertEqual(len(summary["self_fit_evaluations"]), 3)
        self.assertEqual(len(summary["session_variation"]), 10)
        self.assertEqual(len(summary["frozen_source_transfers"]), 2)
        for transfer in summary["frozen_source_transfers"]:
            self.assertEqual(transfer["source_session_id"], "s1")
            self.assertFalse(transfer["evaluations"]["test"]["transfer"]["refitted_on_target"])
        for source in summary["sources"]:
            self.assertEqual(len(source["manifest_sha256"]), 64)
            self.assertIn("source_hash", source["source_identity"])
        for filename in ("manifest.json", "metrics.csv", "report.html"):
            self.assertTrue((output / filename).is_file())

    def test_transfer_keeps_zero_four_and_eight_and_never_selects_target_model(self):
        output = self._out("transfer")
        before = study._sha(self.runs[0] / "model_suite.json")
        with patch.object(models, "fit_suite", side_effect=AssertionError("No post-heldout model fitting")), \
             patch.object(models.predictor_api, "fit_model", side_effect=AssertionError("No source refit")):
            result = study.compare_transfer(self.runs[0], self.runs[1], output, mode="session")
        self.assertEqual(result["status"], "comparison_complete")
        self.assertEqual(study._sha(self.runs[0] / "model_suite.json"), before)
        summary = read_json(output / "summary.json")
        self.assertEqual(set(summary["calibrated"]), {"4", "8"})
        self.assertFalse(summary["target_model_selection"])
        calibration_ids = {c["id"] for c in build_plan()["cases"] if c["split"] == "calibration"}
        source_model = read_json(self.runs[0] / "model_suite.json")["selected_model_id"]
        for budget, payload in summary["calibrated"].items():
            self.assertEqual(payload["evaluations"]["test"]["selected_model_id"], source_model)
            for name, correction in payload["corrections"].items():
                self.assertEqual(correction["calibration_budget"], int(budget))
                self.assertEqual(len(correction["calibration_case_ids"]), int(budget))
                self.assertTrue(set(correction["calibration_case_ids"]) <= calibration_ids)
                baseline = summary["zero_recalibration"]["test"]["models"][name]["predictions"]
                adjusted = payload["evaluations"]["test"]["models"][name]["predictions"]
                for a, b in zip(baseline, adjusted):
                    self.assertAlmostEqual(b["predicted_ms"], a["predicted_ms"] * correction["multiplicative_scale"])
        self.assertTrue((output / "frozen_calibration.json").is_file())

    def test_explicit_gpu_mode_is_still_marked_simulation_when_inputs_are_synthetic(self):
        output = self._out("gpu")
        study.compare_transfer(self.runs[0], self.runs[1], output, mode="gpu")
        summary = read_json(output / "summary.json")
        self.assertEqual(summary["zero_recalibration"]["test"]["transfer"]["mode"], "cross_gpu_zero_recalibration")
        self.assertEqual(summary["measurement_kind"], "simulated")
        self.assertFalse(summary["claims"]["actual_cuda_source_evidence"])

    def test_dtype_comparison_has_separate_suites_and_paired_shape_ratios(self):
        output = self._out("dtypes")
        study.compare_dtypes([self.runs[0], self.runs[3]], output)
        summary = read_json(output / "summary.json")
        self.assertTrue(summary["models_fitted_separately"])
        self.assertFalse(summary["cross_dtype_parameter_transfer"])
        self.assertEqual(set(summary["self_fit_evaluations"]), {"float16", "bfloat16"})
        self.assertEqual(len(summary["paired_latency_cases"]), 96)
        self.assertTrue(all(row["float16_time_over_bfloat16_time"] > 0 for row in summary["paired_latency_cases"]))
        self.assertIn("represented inputs", summary["numerical_comparison_scope"])
        self.assertTrue((output / "paired_latencies.csv").is_file())

    def test_failed_audit_stops_before_output_or_reading_other_files(self):
        output = self._out("untrusted")
        with patch.object(campaign, "audit_run", return_value={"passed": False, "complete_session": False, "errors": ["corrupt"]}):
            with self.assertRaisesRegex(LabError, "audited complete session"):
                study.compare_sessions(self.runs[:3], output)
        self.assertFalse(output.exists())

    def test_count_duplicate_and_existing_output_guards(self):
        with self.assertRaisesRegex(LabError, "exactly three"):
            study.compare_sessions(self.runs[:2], self._out("two"))
        with self.assertRaisesRegex(LabError, "same source directory"):
            study.compare_sessions([self.runs[0], self.runs[0], self.runs[2]], self._out("duplicate"))
        with self.assertRaisesRegex(LabError, "exactly FP16 and BF16"):
            study.compare_dtypes(self.runs[:2], self._out("same_precision"))
        output = self._out("exists")
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
        with self.assertRaisesRegex(LabError, "already exists"):
            study.compare_sessions(self.runs[:3], output)
        self.assertEqual((output / "keep.txt").read_text(), "preserve")
        with self.assertRaisesRegex(LabError, "sealed source directory"):
            study.compare_sessions(self.runs[:3], self.runs[0] / "forbidden_comparison")

    def test_same_plan_and_implementation_are_required_without_hiding_quality_flags(self):
        a, b = (study._load_run(path) for path in self.runs[:2])
        changed = copy.deepcopy(b)
        changed["source_identity"]["analysis_versions"]["numpy"] = "changed"
        with self.assertRaisesRegex(LabError, "analysis dependency"):
            study._guard_sources([a, changed])
        changed = copy.deepcopy(b)
        changed["plan"]["seed"] += 1
        with self.assertRaisesRegex(LabError, "identical registered plan"):
            study._guard_sources([a, changed])
        changed = copy.deepcopy(b)
        changed["quality"]["passed"] = False
        summary = study._summary_base("sessions", [a, changed])
        self.assertFalse(summary["all_source_timing_criteria_passed"])
        self.assertFalse(summary["claims"]["bad_sessions_removed"])

    def test_source_modified_during_derivation_cannot_receive_comparison_manifest(self):
        copied = self._out("source_copy")
        shutil.copytree(self.runs[0], copied)
        output = self._out("changed_source")
        original = study._persist
        def modify_then_persist(out, summary, rows, sources, implementation):
            (copied / "quality.json").write_text('{"tampered": true}\n')
            return original(out, summary, rows, sources, implementation)
        with patch.object(study, "_persist", side_effect=modify_then_persist):
            with self.assertRaisesRegex(LabError, "source changed during comparison"):
                study.compare_sessions([copied, self.runs[1], self.runs[2]], output)
        self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
