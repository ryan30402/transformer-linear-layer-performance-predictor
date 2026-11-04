"""CPU lifecycle tests, separate from manifest/numerical audit tests.

Collection fixtures contain valid three-pass synthetic rows; they never claim
GPU execution. Fitting, report rendering and the final audit are mocked to
isolate disclosure boundaries and failure behavior without fitting 80 models.
"""

from __future__ import annotations

import copy
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from linear_perf import campaign as campaign
from linear_perf.collection import _aggregate, _block_row
from linear_perf.common import LabError, digest, read_json, write_json
from linear_perf.measurement import (
    MeasurementError, _visit, make_environment,
)
from linear_perf.protocol import build_plan


class CampaignLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "session"
        self.plan = build_plan()
        self.environment = make_environment(self.plan, "lifecycle", simulate=True)
        self.events = []
        self.deadlines = []
        self.opened = False
        self.fail_at = None
        self.flag_timing = False
        self.change_evaluation_predictions = False
        self.fixture_identity = {
            "source_hash": "TEST_FIXED_SOURCE_IDENTITY",
            "sources": {"test_fixture": "NOT_HARDWARE_EVIDENCE"},
            "analysis_versions": {},
        }
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(
            campaign, "implementation_identity", return_value=self.fixture_identity))
        self.env_mock = self.patches.enter_context(patch(
            "linear_perf.measurement.make_environment", return_value=self.environment))
        self.patches.enter_context(patch(
            "linear_perf.measurement.run_diagnostics", side_effect=self.diagnostics))
        self.patches.enter_context(patch(
            "linear_perf.measurement.collect_stage", side_effect=self.collect))
        self.patches.enter_context(patch(
            "linear_perf.models.fit_suite", side_effect=self.fit_suite))
        self.patches.enter_context(patch(
            "linear_perf.models.fit_acquisition", side_effect=self.fit_acquisition))
        self.patches.enter_context(patch(
            "linear_perf.models.predict_model", side_effect=self.predict))
        self.patches.enter_context(patch(
            "linear_perf.models.evaluate_suite", side_effect=self.evaluate_suite))
        self.patches.enter_context(patch(
            "linear_perf.models.evaluate_acquisition", side_effect=self.evaluate_acquisition))
        self.batch_mock = self.patches.enter_context(patch(
            "linear_perf.benchmark.benchmark_batching",
            side_effect=AssertionError("Simulation must not request GPU batching")))
        self.report_mock = self.patches.enter_context(patch(
            "linear_perf.report.render_session_report", side_effect=self.report))
        self.seal_mock = self.patches.enter_context(patch.object(
            campaign, "seal", side_effect=self.seal))
        self.audit_mock = self.patches.enter_context(patch.object(
            campaign, "audit_run", return_value={"passed": True, "errors": [], "fixture_only": True}))

    def tearDown(self):
        self.temp.cleanup()

    def run_session(self, **kwargs):
        return campaign.run_campaign(
            self.plan, self.out, session_id="lifecycle", simulate=kwargs.pop("simulate", True),
            progress=lambda message: None, **kwargs,
        )

    def assert_pre_holdout(self):
        self.assertFalse(self.opened)
        self.assertFalse((self.out / "heldout_claim.json").exists())
        self.assertFalse((self.out / "heldout").exists())

    def diagnostics(self, plan, environment, output, **kwargs):
        self.events.append("diagnostics")
        self.assert_pre_holdout()
        self.deadlines.append(kwargs["deadline_monotonic"])
        Path(output).mkdir()
        value = {"status": "completed", "measurement_kind": "diagnostic_bundle",
                 "excluded_from_model": True, "fixture_only": True}
        write_json(Path(output) / "fixture_diagnostics.json", value)
        return value

    def fixture_rows(self, cases):
        rows = []
        for position, case in enumerate(cases, 1):
            blocks = []
            for block in range(1, 4):
                raw = _visit(
                    self.plan, case, self.environment, device="cuda:0",
                    deadline=time.monotonic() + 60,
                    token={"purpose": "CPU_lifecycle_fixture", "pass": block},
                )
                raw["visit_index"] = (block - 1) * len(cases) + position
                blocks.append(_block_row(
                    raw, plan=self.plan, case=case, env=self.environment,
                    block_index=block, position=position, order_seed=123 + block,
                ))
            aggregate = _aggregate(blocks, expected_blocks=3)
            aggregate["session_id"] = "lifecycle"
            rows.append(aggregate)
        return rows

    def collect(self, plan, cases, environment, output, **kwargs):
        folder = Path(output)
        stage = folder.name
        self.events.append(stage)
        self.deadlines.append(kwargs["deadline_monotonic"])
        self.assertEqual(environment, self.environment)
        splits = {case["split"] for case in cases}
        if stage == "initial":
            self.assert_pre_holdout()
            self.assertEqual(splits, {"calibration", "development"})
        elif stage == "candidate_pool":
            self.assert_pre_holdout()
            self.assertEqual(splits, {"candidate"})
            self.assertEqual({case["id"] for case in cases}, set(plan["study"]["candidate_pool_ids"]))
        elif stage == "heldout":
            self.assertEqual(splits, {"test", "challenge"})
            self.assertTrue((self.out / "model_suite.json").is_file())
            self.assertTrue((self.out / "acquisition_frozen.json").is_file())
            freeze = read_json(self.out / "freeze.json")
            claim = read_json(self.out / "heldout_claim.json")
            frozen = read_json(self.out / "frozen_predictions.json")
            self.assertEqual(claim["freeze_hash"], digest(freeze))
            self.assertEqual(freeze["frozen_predictions_hash"], digest(frozen))
            self.assertLessEqual(freeze["frozen_at_monotonic"], claim["claimed_at_monotonic"])
            self.assertLessEqual(claim["claimed_at_monotonic"], time.monotonic())
            self.assertFalse(freeze["test_labels_available"])
            self.assertTrue(freeze["development_selection_closed"])
            self.assertEqual(set(claim["case_ids"]), {case["id"] for case in cases})
            self.opened = True
        else:
            self.fail("Unexpected collection stage: " + stage)
        folder.mkdir()
        chosen = cases[:2] if self.fail_at == stage else cases
        rows = self.fixture_rows(chosen)
        raw = folder / "fixture_partial_evidence.json"
        write_json(raw, {"fixture_only": True, "rows": rows})
        if self.fail_at == stage:
            raise MeasurementError("injected hard numerical failure",
                                      completed_rows=rows, raw_path=str(raw))
        controls = [
            {"case_id": cid, "median_ms": 0.05, "measurement_kind": "simulated_diagnostic",
             "excluded_from_model": True, "fixture_only": True}
            for cid in plan["study"]["control_case_ids"]
        ]
        return {"rows": rows, "controls": controls,
                "quality": {"passed": not (self.flag_timing and stage == "initial"),
                            "fixture_only": True}}

    def fit_suite(self, train, dev, plan, **kwargs):
        self.events.append("fit_suite")
        self.assert_pre_holdout()
        self.assertEqual({row["split"] for row in train}, {"calibration"})
        self.assertEqual({row["split"] for row in dev}, {"development"})
        return {"dtype": plan["dtype"], "selected_model_id": "fixture",
                "models": {"fixture": {"fixture_only": True, "offset": 0.01}},
                "fixture_only": True}

    def fit_acquisition(self, pool, train, dev, plan, **kwargs):
        self.events.append("fit_acquisition")
        self.assert_pre_holdout()
        self.assertEqual({row["case_id"] for row in pool}, set(plan["study"]["candidate_pool_ids"]))
        self.assertEqual({row["split"] for row in train}, {"calibration"})
        self.assertEqual({row["split"] for row in dev}, {"development"})
        return {"fixture_only": True, "frozen": True, "budgets": [2, 4, 8, 12]}

    @staticmethod
    def predict(model, cases, dtype):
        return [model["offset"] + case["m"] * 0.00001 for case in cases]

    def evaluate_suite(self, suite, rows, plan, **kwargs):
        self.events.append("evaluate_" + rows[0]["split"])
        self.assertTrue(self.opened)
        values = self.predict(suite["models"]["fixture"], rows, plan["dtype"])
        if self.change_evaluation_predictions:
            values = [value * 2 for value in values]
        predictions = [
            {"case_id": row["case_id"], "actual_ms": row["median_ms"],
             "predicted_ms": value, "split": row["split"]}
            for row, value in zip(rows, values)
        ]
        return {"selected_model_id": "fixture",
                "models": {"fixture": {"predictions": predictions, "n": len(rows)}}}

    def evaluate_acquisition(self, acquisition, rows, plan, **kwargs):
        self.events.append("evaluate_acquisition_" + rows[0]["split"])
        self.assertTrue(self.opened)
        self.assertTrue(acquisition["frozen"])
        return {"fixture_only": True, "split": rows[0]["split"]}

    def report(self, out):
        self.events.append("report")
        self.assertTrue((Path(out) / "results.json").is_file())
        return {"fixture_only": True}

    def seal(self, out, **kwargs):
        write_json(Path(out) / "manifest.json",
                   {"fixture_only": True, "status": kwargs["status"]["status"]}, overwrite=False)
        return {"fixture_only": True}

    def test_frozen_models_claim_and_predictions_precede_every_heldout_label(self):
        result = self.run_session()
        self.assertEqual(result["status"], "simulated_complete")
        self.assertEqual(self.events, [
            "diagnostics", "initial", "fit_suite", "candidate_pool", "fit_acquisition",
            "heldout", "evaluate_test", "evaluate_challenge",
            "evaluate_acquisition_test", "evaluate_acquisition_challenge", "report",
        ])
        self.assertEqual(len(set(self.deadlines)), 1, "All GPU stages share one outer deadline")
        self.assertTrue(result["test_opened"])
        self.assertFalse(result["hardware_accuracy_validated"])
        self.assertEqual(result["automatic_retries"], 0)
        self.audit_mock.assert_called_once_with(self.out.resolve())
        self.seal_mock.assert_called_once()
        journal = read_json(self.out / "journal.json")
        stages = [row["stage"] for row in journal]
        self.assertLess(stages.index("freeze_models_and_predictions_before_heldout"),
                        stages.index("frozen_test_and_challenge_collection"))

    def test_simulation_never_invents_GPU_batching(self):
        self.run_session()
        self.batch_mock.assert_not_called()
        batching = read_json(self.out / "batching.json")
        self.assertEqual(batching["status"], "skipped_no_gpu")
        self.assertEqual(batching["measurement_kind"], "simulated")
        self.assertNotIn("speedup", batching)
        self.assertTrue(self.env_mock.call_args.kwargs["simulate"])
        self.assertFalse(read_json(self.out / "results.json")["claims"]["gpu_executed"])

    def test_existing_session_cannot_be_overwritten_or_resumed_implicitly(self):
        self.run_session()
        status_bytes = (self.out / "status.json").read_bytes()
        calls_before = copy.deepcopy(self.events)
        with self.assertRaisesRegex(LabError, "overwrite"):
            self.run_session()
        self.assertEqual(self.events, calls_before)
        self.assertEqual((self.out / "status.json").read_bytes(), status_bytes)

    def test_numerical_failure_before_holdout_preserves_partial_data_and_stops(self):
        self.fail_at = "initial"
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.events, ["diagnostics", "initial"])
        self.assertFalse(result["test_opened"])
        self.assertFalse((self.out / "heldout_claim.json").exists())
        self.assertFalse((self.out / "candidate_pool").exists())
        evidence = read_json(self.out / "initial" / "fixture_partial_evidence.json")
        self.assertEqual(len(evidence["rows"]), 2)
        failure = read_json(self.out / "FAILED.json")
        self.assertFalse(failure["automatic_retry"])
        self.assertTrue(failure["partial_evidence_retained"])
        self.report_mock.assert_not_called()
        self.audit_mock.assert_not_called()

    def test_failure_during_holdout_keeps_consumed_claim_and_never_retunes(self):
        self.fail_at = "heldout"
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["test_opened"])
        self.assertEqual(self.events[-1], "heldout")
        self.assertEqual(self.events.count("heldout"), 1)
        self.assertEqual(self.events.count("fit_suite"), 1)
        self.assertEqual(self.events.count("fit_acquisition"), 1)
        self.assertTrue((self.out / "heldout_claim.json").is_file())
        self.assertTrue((self.out / "freeze.json").is_file())
        self.assertTrue((self.out / "heldout" / "fixture_partial_evidence.json").is_file())
        self.assertFalse((self.out / "heldout_measurements.jsonl").exists())
        self.batch_mock.assert_not_called()

    def test_disclosed_timing_flags_do_not_trigger_rerun_or_label_removal(self):
        self.flag_timing = True
        result = self.run_session()
        self.assertEqual(result["status"], "simulated_complete")
        self.assertFalse(result["all_timing_criteria_passed"])
        self.assertEqual(self.events.count("initial"), 1)
        self.assertEqual(self.events.count("heldout"), 1)
        counts = read_json(self.out / "results.json")["evidence_counts"]
        self.assertEqual(counts["calibration"], 24)
        self.assertEqual(counts["development"], 12)
        self.assertEqual(counts["heldout"], 36)
        self.assertEqual(result["samples_removed"], 0)

    def test_evaluation_cannot_substitute_predictions_after_test_is_open(self):
        self.change_evaluation_predictions = True
        result = self.run_session()
        self.assertEqual(result["status"], "failed")
        self.assertIn("frozen predictions", result["error"])
        self.assertEqual(self.events.count("fit_suite"), 1)
        self.assertNotIn("evaluate_acquisition_test", self.events)
        self.assertFalse((self.out / "batching.json").exists())

    def test_real_CUDA_preflight_failure_does_not_turn_into_simulation(self):
        self.env_mock.side_effect = LabError("CUDA unavailable; no CPU fallback")
        result = self.run_session(simulate=False)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["measurement_kind"], "cuda_measured")
        self.assertFalse(result["test_opened"])
        self.assertFalse(self.events)
        self.assertFalse(self.env_mock.call_args.kwargs["simulate"])
        self.assertEqual(self.env_mock.call_count, 1)
        self.batch_mock.assert_not_called()

    def test_claim_creation_is_exclusive_and_prior_claim_survives(self):
        claim = self.root / "exclusive_claim.json"
        campaign._new_claim(claim, {"claim": "first"})
        original = claim.read_bytes()
        with self.assertRaises(FileExistsError):
            campaign._new_claim(claim, {"claim": "second"})
        self.assertEqual(claim.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
