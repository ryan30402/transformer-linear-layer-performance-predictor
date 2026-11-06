"""Fresh experiment integration and failure-retention checks on a CPU.

One explicitly simulated source/repeat pair is really collected and modeled.
Failure tests reuse that fixture to exercise orchestration without claiming a
second hardware measurement or spending time refitting identical examples.
"""
from __future__ import annotations

import builtins
import copy
import csv
import hashlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from linear_perf import benchmark, experiment, fresh_experiment, selection
from linear_perf.common import LabError, read_json, read_jsonl, write_json
from linear_perf.dataset_measurement import repeat_subset

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    return {file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in Path(path).rglob("*") if file.is_file()}


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class FreshExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.storage.cleanup)
        cls.root = Path(cls.storage.name)
        cls.plan = read_json(ROOT / "configs/workloads_smoke.json")
        cls.full = cls.root / "complete"
        cls.fit_records = []
        actual_fit = selection.fit_cv_suite
        original_open, original_io_open = builtins.open, io.open
        historical = (ROOT / "results").resolve()

        def reject_historical_read(original, file, *args, **kwargs):
            if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(historical):
                raise AssertionError("Fresh workflow read historical measurements: " + str(file))
            return original(file, *args, **kwargs)

        def fit_training_only(rows, **kwargs):
            cls.fit_records.append(copy.deepcopy(rows))
            if {row["split"] for row in rows} != {"train"}:
                raise AssertionError("Held-out labels reached a fitter")
            return actual_fit(rows, **kwargs)

        with patch.object(benchmark, "_load_torch", side_effect=AssertionError("CPU simulation must not load CUDA")), \
                patch.object(selection, "fit_cv_suite", side_effect=fit_training_only), \
                patch("builtins.open", side_effect=lambda file, *a, **k: reject_historical_read(original_open, file, *a, **k)), \
                patch("io.open", side_effect=lambda file, *a, **k: reject_historical_read(original_io_open, file, *a, **k)):
            cls.completed = fresh_experiment.run_fresh_experiment(
                cls.plan, cls.full, session_id="fresh_cpu", simulate=True, progress=None,
            )

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.workspace.cleanup)
        self.work = Path(self.workspace.name)
        self.out = self.work / "run"

    def cached_collection(self, manifest, output, *, session_id, simulate=False, device="cuda:0",
                          analyze=True, repeat_ids=None, progress=None):
        """Reuse a real validated software fixture in orchestration-only tests."""
        name = "primary" if analyze else "repeat"
        if manifest != self.plan or not simulate:
            raise AssertionError("Fixture is restricted to the declared simulated smoke workload")
        if name == "repeat" and set(repeat_ids or []) != set(repeat_subset(manifest)):
            raise AssertionError("Repeat request changed the declared subset")
        shutil.copytree(self.full / name, output)
        return {**read_json(Path(output) / "status.json"), "output": str(output)}

    def test_complete_new_measurement_flow_has_no_historical_input_or_repeat_refit(self):
        summary = self.completed
        self.assertEqual(summary["status"], "simulated_complete")
        self.assertEqual(summary["measurement_kind"], "simulated")
        self.assertTrue(summary["required_stages_completed"] and summary["all_requested_stages_completed"])
        self.assertFalse(summary["historical_measurements_loaded"])
        self.assertFalse(summary["freeze_before_measurement_required"])
        self.assertFalse(summary["heldout_labels_used_for_selection"] or summary["repeat_labels_used_for_refit"])
        self.assertTrue(summary["primary_evaluation_available"] and summary["primary_data_retained"])
        self.assertEqual(summary["repeat_plan"]["selected_shapes"], 4)
        self.assertEqual(len(repeat_subset(read_json(ROOT / "configs/workloads.json"))), 77)
        self.assertEqual(summary["stages"]["layer"]["status"], "not_requested")
        self.assertEqual(len(self.fit_records), 4)  # One fit suite for each primary precision/cache context.
        self.assertTrue(all({row["split"] for row in rows} == {"train"} for rows in self.fit_records))
        primary, repeat = self.full / "primary", self.full / "repeat"
        self.assertEqual(read_json(primary / "dataset/manifest.json"), read_json(repeat / "dataset/manifest.json"))
        self.assertFalse((repeat / "analysis").exists())
        selected = set(summary["repeat_plan"]["case_ids"])
        self.assertEqual({row["case_id"] for row in read_jsonl(repeat / "dataset/measurements.jsonl")}, selected)
        self.assertFalse(summary["repeat_comparison"]["same_physical_gpu_verified"])
        self.assertFalse(summary["repeat_comparison"]["target_labels_used_for_refit"])

    def test_combined_report_preserves_selected_scores_and_quality_separately(self):
        original = [row for row in csv_rows(self.full / "primary/analysis/metrics.csv") if row["selected_by_cv"] == "True"]
        self.assertEqual(csv_rows(self.full / "selected_model_metrics.csv"), original)
        report = (self.full / "report.html").read_text(encoding="utf-8")
        self.assertIn("SIMULATED SOFTWARE CHECK", report)
        self.assertIn("Mean APE %", report)
        self.assertIn("P90 APE %", report)
        self.assertIn("Worst APE %", report)
        self.assertIn("primary/analysis/report.html", report)
        self.assertIn("repeat_comparison/report.html", report)
        self.assertIn("selected_model_metrics.csv", report)
        for name in ("primary", "repeat"):
            audit = read_json(self.full / name / "audit.json")
            self.assertEqual(self.completed["measurement_quality"][name], audit["measurement_quality"])
            self.assertEqual(self.completed["stages"][name]["audit"]["checks"], audit["record_validation"]["checks"])

    def test_primary_failure_stops_followups_and_keeps_partial_evidence(self):
        def fail_primary(manifest, output, **kwargs):
            folder = Path(output) / "dataset"
            folder.mkdir(parents=True)
            (folder / "raw_visits.jsonl").write_text('{"fixture": "partial failed attempt"}\n')
            raise LabError("Primary failed after a retained visit")

        with patch.object(experiment, "run_experiment", side_effect=fail_primary) as runner, \
                patch.object(experiment, "compare_datasets") as comparison, \
                patch.object(experiment, "validate_layer") as layer:
            with self.assertRaisesRegex(LabError, "retained visit"):
                fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="primary_failure",
                                                     simulate=True, include_layer=True, progress=None)
        self.assertEqual(runner.call_count, 1)
        comparison.assert_not_called()
        layer.assert_not_called()
        summary = read_json(self.out / "summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["primary_data_retained"])
        self.assertFalse(summary["primary_evaluation_available"])
        self.assertEqual(summary["stages"]["repeat"]["status"], "not_run")
        self.assertFalse(summary["required_stages_completed"])
        self.assertIn("Primary failed after a retained visit", (self.out / "report.html").read_text())

    def test_repeat_failure_keeps_complete_primary_and_does_not_compare_partial_target(self):
        before = {}

        def collect(manifest, output, **kwargs):
            if kwargs["analyze"]:
                return self.cached_collection(manifest, output, **kwargs)
            before.update(hashes(self.out / "primary"))
            Path(output).mkdir()
            (Path(output) / "partial.json").write_text('{"status":"partial"}\n')
            raise LabError("Repeat interrupted by a fixture GPU error")

        with patch.object(experiment, "run_experiment", side_effect=collect) as runner, \
                patch.object(experiment, "compare_datasets") as comparison:
            with self.assertRaisesRegex(LabError, "Repeat interrupted"):
                fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="repeat_failure",
                                                     simulate=True, progress=None)
        self.assertEqual(runner.call_count, 2)
        comparison.assert_not_called()
        summary = read_json(self.out / "summary.json")
        self.assertTrue(summary["primary_evaluation_available"])
        self.assertEqual(summary["stages"]["primary"]["status"], "completed")
        self.assertEqual(summary["stages"]["repeat"]["status"], "failed")
        self.assertEqual(summary["stages"]["repeat_comparison"]["status"], "not_run")
        self.assertEqual(hashes(self.out / "primary"), before)
        self.assertTrue((self.out / "repeat/partial.json").is_file())
        self.assertFalse(summary["all_requested_stages_completed"])

    def test_comparison_failure_retains_both_completed_datasets(self):
        with patch.object(experiment, "run_experiment", side_effect=self.cached_collection), \
                patch.object(experiment, "compare_datasets", side_effect=LabError("Comparison context mismatch")):
            with self.assertRaisesRegex(LabError, "context mismatch"):
                fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="comparison_failure",
                                                     simulate=True, progress=None)
        summary = read_json(self.out / "summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stages"]["primary"]["status"], "completed")
        self.assertEqual(summary["stages"]["repeat"]["status"], "completed")
        self.assertEqual(summary["stages"]["repeat_comparison"]["status"], "failed")
        self.assertTrue((self.out / "primary/analysis/report.html").is_file())
        self.assertTrue((self.out / "repeat/dataset/raw_visits.jsonl").is_file())

    def test_requested_layer_is_not_faked_or_marked_complete_in_simulation(self):
        with patch.object(experiment, "run_experiment", side_effect=self.cached_collection), \
                patch.object(experiment, "validate_layer", side_effect=AssertionError("Layer must not be simulated")) as layer:
            result = fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="layer_unavailable",
                                                         simulate=True, include_repeat=False, include_layer=True,
                                                         progress=None)
        layer.assert_not_called()
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(result["required_stages_completed"])
        self.assertFalse(result["all_requested_stages_completed"])
        self.assertEqual(result["stages"]["layer"]["status"], "unavailable_simulation")
        self.assertTrue(result["primary_evaluation_available"])
        self.assertFalse((self.out / "layer").exists())

    def test_quality_qualifications_do_not_silently_drop_primary_data(self):
        real_audit = experiment.audit_dataset

        def qualified(path):
            result = real_audit(path)
            result["measurement_quality"] = {"passed": False, "flagged_rows": [{"case_id": "fixture_qualified_case"}],
                                             "controls": [{"passed": False}]}
            return result

        with patch.object(experiment, "run_experiment", side_effect=self.cached_collection), \
                patch.object(experiment, "audit_dataset", side_effect=qualified):
            result = fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="qualified",
                                                         simulate=True, include_repeat=False, progress=None)
        self.assertEqual(result["status"], "simulated_complete")
        self.assertTrue(result["all_requested_stages_completed"])
        self.assertFalse(result["measurement_quality_passed"])
        self.assertEqual(csv_rows(self.out / "selected_model_metrics.csv"), csv_rows(self.full / "selected_model_metrics.csv"))
        self.assertEqual(hashes(self.out / "primary/dataset"), hashes(self.full / "primary/dataset"))
        self.assertIn("Timing quality has retained qualifications", (self.out / "report.html").read_text())

    def test_checkpoint_export_failure_does_not_repeat_measurements_or_mutate_state(self):
        checkpoints = []

        def checkpoint(state):
            checkpoints.append(state["active_stage"])
            state["stages"]["primary"]["status"] = "corrupted external copy"
            raise OSError("Fixture checkpoint disk is unavailable")

        with patch.object(experiment, "run_experiment", side_effect=self.cached_collection) as runner:
            result = fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="export_failure",
                                                         simulate=True, checkpoint=checkpoint, progress=None)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(checkpoints, ["primary", "repeat", "repeat_comparison"])
        self.assertEqual(result["status"], "simulated_complete")
        self.assertEqual(result["stages"]["primary"]["status"], "completed")
        self.assertEqual(len(result["checkpoint_warnings"]), 3)
        self.assertTrue(result["all_requested_stages_completed"])

    def test_existing_output_and_overlong_session_are_rejected_before_collection(self):
        self.out.mkdir()
        sentinel = self.out / "keep.txt"
        sentinel.write_text("Keep this existing experiment")
        with patch.object(experiment, "run_experiment") as runner:
            with self.assertRaisesRegex(LabError, "Output already exists"):
                fresh_experiment.run_fresh_experiment(self.plan, self.out, session_id="existing", simulate=True, progress=None)
            with self.assertRaisesRegex(LabError, "stage suffixes"):
                fresh_experiment.run_fresh_experiment(self.plan, self.work / "too_long", session_id="a" * 57,
                                                     simulate=True, progress=None)
        runner.assert_not_called()
        self.assertEqual(sentinel.read_text(), "Keep this existing experiment")
        self.assertFalse((self.work / "too_long").exists())

    def test_required_initial_backup_fails_before_any_gpu_collection(self):
        with patch.object(experiment, "run_experiment") as runner:
            with self.assertRaisesRegex(OSError, "Storage unavailable"):
                fresh_experiment.run_fresh_experiment(
                    self.plan, self.out, session_id="initial_backup_failure", simulate=True,
                    progress=None, checkpoint_required=True,
                    checkpoint=lambda state: (_ for _ in ()).throw(OSError("Storage unavailable")),
                )
        runner.assert_not_called()
        self.assertFalse((self.out / "primary").exists())
        result = read_json(self.out / "summary.json")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["checkpoint_warnings"][0]["event"], "experiment_initialized")
        self.assertTrue(result["checkpoint_warnings"][0]["required"])

    def test_required_stage_backup_failure_keeps_completed_primary_and_stops_repeat(self):
        def checkpoint(state):
            if state["checkpoint_event"] == "stage_completed_primary":
                raise OSError("Primary backup could not be verified")

        with patch.object(experiment, "run_experiment", side_effect=self.cached_collection) as runner:
            with self.assertRaisesRegex(OSError, "could not be verified"):
                fresh_experiment.run_fresh_experiment(
                    self.plan, self.out, session_id="required_stage_failure", simulate=True,
                    checkpoint_required=True, checkpoint=checkpoint, progress=None,
                )
        self.assertEqual(runner.call_count, 1)
        summary = read_json(self.out / "summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stages"]["primary"]["status"], "completed")
        self.assertTrue(summary["primary_evaluation_available"])
        self.assertEqual(summary["stages"]["repeat"]["status"], "not_run")
        self.assertTrue((self.out / "primary/dataset/raw_visits.jsonl").is_file())

    def test_required_snapshots_include_initialized_and_final_completion_states(self):
        from linear_perf.checkpoints import DurableCheckpoints
        import json
        import zipfile

        destination = self.work / "persistent"
        writer = DurableCheckpoints(self.out, destination, session_id="states", progress=None)
        with patch.object(experiment, "run_experiment", side_effect=self.cached_collection):
            result = fresh_experiment.run_fresh_experiment(
                self.plan, self.out, session_id="states", simulate=True, include_repeat=False,
                checkpoint_required=True, checkpoint=writer.stage, progress=None,
            )
        self.assertEqual(result["status"], "simulated_complete")
        snapshots = sorted(destination.glob("*.zip"))
        self.assertEqual(len(snapshots), 3)
        with zipfile.ZipFile(snapshots[0]) as archive:
            initial = json.loads(archive.read("run/status.json"))
            self.assertEqual(initial["status"], "running")
            self.assertFalse(any(name.startswith("run/primary/") for name in archive.namelist()))
        with zipfile.ZipFile(snapshots[-1]) as archive:
            final = json.loads(archive.read("run/status.json"))
            self.assertEqual(final["status"], "simulated_complete")
            self.assertTrue(final["all_requested_stages_completed"])
            self.assertEqual(json.loads(archive.read("checkpoint.json"))["checkpoint_event"], "experiment_finished")


if __name__ == "__main__":
    unittest.main()
