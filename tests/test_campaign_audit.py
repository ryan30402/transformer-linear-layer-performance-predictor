"""CPU-only evidence tampering tests; all timing fixtures are explicit simulation."""

import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np

from linear_perf import campaign as campaign
from linear_perf.common import LabError, digest, read_json, read_jsonl, write_json, write_jsonl
from linear_perf.measurement import collect_stage, make_environment, run_diagnostics
from linear_perf.models import evaluate_acquisition, evaluate_suite, fit_acquisition, fit_suite
from linear_perf.numerical_witness import audit_witness_npz, witness_inputs
from linear_perf.protocol import build_plan


def _seal_existing(out):
    (out / "manifest.json").unlink(missing_ok=True)
    implementation = campaign.implementation_identity()
    write_json(out / "source_identity.json", implementation)
    # Keep a legitimate fixture's source-identity chain synchronized if a test
    # intentionally re-seals it after modifying an artifact.
    if (out / "freeze.json").exists():
        freeze = read_json(out / "freeze.json")
        freeze["source_identity_hash"] = digest(implementation)
        write_json(out / "freeze.json", freeze)
        claim = read_json(out / "heldout_claim.json")
        claim["freeze_hash"] = digest(freeze)
        write_json(out / "heldout_claim.json", claim)
    status = read_json(out / "status.json")
    stage = "linear_perf_diagnostics" if status["status"] == "diagnostics_completed" else "linear_perf_session"
    return campaign.seal(out, plan=read_json(out / "plan.json"), environment=read_json(out / "environment.json"),
                         status=status, implementation=implementation, stage=stage)


def _make_simulated_fixture(out):
    """Use actual collectors/models to exercise the audit; never invent GPU data."""
    out.mkdir()
    plan, sid = build_plan(), "audit_software_fixture"
    env = make_environment(plan, sid, simulate=True)
    write_json(out / "plan.json", plan)
    write_json(out / "environment.json", env)
    deadline = time.monotonic() + 1800
    diag = run_diagnostics(plan, env, out / "diagnostics", session_id=sid, deadline_monotonic=deadline, progress=None)
    write_json(out / "diagnostics_summary.json", diag)
    initial_cases = [c for c in plan["cases"] if c["split"] in {"calibration", "development"}]
    initial = collect_stage(plan, initial_cases, env, out / "initial", session_id=sid, deadline_monotonic=deadline, progress=None)
    train = [r for r in initial["rows"] if r["split"] == "calibration"]
    dev = [r for r in initial["rows"] if r["split"] == "development"]
    suite = fit_suite(train, dev, plan, environment=env)
    pool = collect_stage(plan, [c for c in plan["cases"] if c["id"] in plan["study"]["candidate_pool_ids"]],
                         env, out / "candidate_pool", session_id=sid, deadline_monotonic=deadline, progress=None)
    acquisition = fit_acquisition(pool["rows"], train, dev, plan, environment=env)
    heldout_cases = [c for c in plan["cases"] if c["split"] in {"test", "challenge"}]
    frozen = campaign._frozen_shape_predictions(suite, heldout_cases)
    implementation = campaign.implementation_identity()
    freeze = {"frozen_at_monotonic": time.monotonic(), "model_suite_hash": digest(suite),
              "acquisition_hash": digest(acquisition), "frozen_predictions_hash": digest(frozen),
              "plan_hash": digest(plan), "source_identity_hash": digest(implementation),
              "test_labels_available": False, "development_selection_closed": True}
    claim = {"claimed_at_monotonic": time.monotonic(), "freeze_hash": digest(freeze),
             "case_ids": [c["id"] for c in heldout_cases]}
    heldout = collect_stage(plan, heldout_cases, env, out / "heldout", session_id=sid, deadline_monotonic=deadline, progress=None)
    evaluations = {split: evaluate_suite(suite, [r for r in heldout["rows"] if r["split"] == split], plan, target_environment=env)
                   for split in ("test", "challenge")}
    acquisition_results = {split: evaluate_acquisition(acquisition, [r for r in heldout["rows"] if r["split"] == split], plan, target_environment=env)
                           for split in ("test", "challenge")}
    controls = initial["controls"] + pool["controls"] + heldout["controls"]
    global_controls = campaign._control_summary(controls, plan["study"]["quality"]["control_ratio"])
    quality = {"initial": initial["quality"], "candidate_pool": pool["quality"], "heldout": heldout["quality"],
               "whole_session_controls": global_controls}
    quality["passed"] = all(q["passed"] for q in quality.values())
    status = {"status": "simulated_complete", "session_id": sid, "measurement_kind": "simulated",
              "automatic_retries": 0, "samples_removed": 0, "test_opened": True,
              "all_timing_criteria_passed": quality["passed"]}
    results = {"session_id": sid, "environment_id": env["environment_id"], "plan_hash": digest(plan),
               "measurement_kind": "simulated", "selected_model_id": suite["selected_model_id"],
               "evaluations": evaluations, "quality_passed": quality["passed"], "claims": {"gpu_executed": False}}
    for filename, value in (("model_suite.json", suite), ("acquisition_frozen.json", acquisition),
                            ("frozen_predictions.json", frozen), ("freeze.json", freeze), ("heldout_claim.json", claim),
                            ("acquisition_results.json", acquisition_results), ("controls.json", controls),
                            ("quality.json", quality), ("status.json", status), ("results.json", results)):
        write_json(out / filename, value)
    for filename, rows in (("initial_measurements.jsonl", initial["rows"]),
                           ("candidate_measurements.jsonl", pool["rows"]), ("heldout_measurements.jsonl", heldout["rows"])):
        write_jsonl(out / filename, rows)
    _seal_existing(out)


class CampaignAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.shared.name) / "complete_simulated_session"
        _make_simulated_fixture(cls.fixture)
        result = campaign.audit_run(cls.fixture)
        if not result["passed"]:
            raise AssertionError("Legitimate fixture failed audit: " + repr(result))

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name) / "session"
        shutil.copytree(self.fixture, self.out)

    def tearDown(self):
        self.temp.cleanup()

    def assert_rejected(self, fragment=None):
        result = campaign.audit_run(self.out)
        self.assertFalse(result["passed"], result)
        if fragment:
            self.assertIn(fragment, "\n".join(result["errors"]), result)
        return result

    def test_complete_simulation_passes_without_claiming_gpu_or_witnesses(self):
        result = campaign.audit_run(self.out)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["complete_session"])
        self.assertEqual(result["measurement_kind"], "simulated")
        self.assertFalse(list((self.out / "diagnostics" / "numerical_witnesses").glob("*.npz")))

    def test_nested_manifest_is_sealed_and_tampering_is_detected(self):
        nested = self.out / "nested" / "manifest.json"
        write_json(nested, {"nested_evidence": 1})
        manifest = _seal_existing(self.out)
        self.assertIn("nested/manifest.json", manifest["files"])
        write_json(nested, {"nested_evidence": 2})
        self.assert_rejected("Artifact hash mismatch: nested/manifest.json")

    def test_symbolic_link_evidence_cannot_be_sealed(self):
        target = Path(self.temp.name) / "outside.json"
        write_json(target, {"outside": True})
        (self.out / "outside-link.json").symlink_to(target)
        with self.assertRaises(LabError):
            _seal_existing(self.out)

    def test_complete_initial_coverage_required_even_after_resealing(self):
        path = self.out / "initial_measurements.jsonl"
        rows = read_jsonl(path)
        write_jsonl(path, rows[1:])
        _seal_existing(self.out)
        self.assert_rejected("initial case IDs differ")

    def test_complete_pool_coverage_required(self):
        path = self.out / "candidate_measurements.jsonl"
        write_jsonl(path, read_jsonl(path)[1:])
        _seal_existing(self.out)
        self.assert_rejected("candidate_pool case IDs differ")

    def test_complete_heldout_coverage_required(self):
        path = self.out / "heldout_measurements.jsonl"
        write_jsonl(path, read_jsonl(path)[1:])
        _seal_existing(self.out)
        self.assert_rejected("heldout case IDs differ")

    def test_heldout_raw_visit_must_follow_claim_not_just_freeze(self):
        claim = read_json(self.out / "heldout_claim.json")
        raw = read_json(self.out / "heldout" / "raw_visits.json")
        # Move the claim later without altering valid raw timing/aggregation.
        claim["claimed_at_monotonic"] = max(r["visit_finished_monotonic_seconds"] for r in raw["rows"]) + 1
        write_json(self.out / "heldout_claim.json", claim)
        _seal_existing(self.out)
        self.assert_rejected("Heldout visit preceded its frozen claim")

    def test_environment_and_manifest_session_are_both_checked(self):
        env = read_json(self.out / "environment.json")
        env["session_id"] = "another_session_same_hardware"
        write_json(self.out / "environment.json", env)
        _seal_existing(self.out)
        self.assert_rejected("Environment/session/plan mismatch")

    def test_raw_block_session_cannot_be_changed_behind_valid_hashes(self):
        path = self.out / "heldout_measurements.jsonl"
        rows = read_jsonl(path)
        rows[0]["measurement_blocks"][0]["session_id"] = "other_session"
        rows[0]["measurement_blocks_hash"] = digest(rows[0]["measurement_blocks"])
        write_jsonl(path, rows)
        _seal_existing(self.out)
        self.assert_rejected("Visit/session mismatch")

    def test_resealed_raw_visit_cannot_disagree_with_aggregate(self):
        path = self.out / "heldout" / "raw_visits.json"
        raw = read_json(path)
        raw["rows"][0]["visit_started_at_utc"] = "changed_without_updating_aggregate"
        raw["rows_hash"] = digest(raw["rows"])
        write_json(path, raw)
        _seal_existing(self.out)
        self.assert_rejected("Aggregate block differs from retained raw evidence")

    def test_saved_acquisition_result_is_recomputed(self):
        path = self.out / "acquisition_results.json"
        result = read_json(path)
        result["test"]["curves"][0]["mean_median_ape_pct"] += 0.5
        write_json(path, result)
        _seal_existing(self.out)
        self.assert_rejected("Saved acquisition results differ")

    def test_nested_acquisition_model_hash_is_checked_even_if_outer_hashes_updated(self):
        path = self.out / "acquisition_frozen.json"
        result = read_json(path)
        result["runs"][0]["trials"][0]["candidate_model"]["params"]["compute_tflops"] *= 2
        result["acquisition_hash"] = digest({k: v for k, v in result.items() if k != "acquisition_hash"})
        write_json(path, result)
        freeze = read_json(self.out / "freeze.json")
        freeze["acquisition_hash"] = digest(result)
        write_json(self.out / "freeze.json", freeze)
        _seal_existing(self.out)
        self.assert_rejected("Payload digest mismatch: model_hash")

    def test_development_acceptance_decision_cannot_be_flipped(self):
        path = self.out / "acquisition_frozen.json"
        result = read_json(path)
        trial = result["runs"][0]["trials"][0]
        trial["accepted"] = not trial["accepted"]
        result["acquisition_hash"] = digest({k: v for k, v in result.items() if k != "acquisition_hash"})
        write_json(path, result)
        freeze = read_json(self.out / "freeze.json")
        freeze["acquisition_hash"] = digest(result)
        write_json(self.out / "freeze.json", freeze)
        _seal_existing(self.out)
        self.assert_rejected("Acquisition acceptance violates")

    def test_complete_acquisition_trial_log_is_required(self):
        path = self.out / "acquisition_frozen.json"
        result = read_json(path)
        result["runs"][0]["trials"] = []
        result["acquisition_hash"] = digest({k: v for k, v in result.items() if k != "acquisition_hash"})
        write_json(path, result)
        freeze = read_json(self.out / "freeze.json")
        freeze["acquisition_hash"] = digest(result)
        write_json(self.out / "freeze.json", freeze)
        _seal_existing(self.out)
        self.assert_rejected("Acquisition round log is incomplete")

    def test_duplicate_frozen_prediction_cannot_hide_behind_dictionary_lookup(self):
        path = self.out / "frozen_predictions.json"
        frozen = read_json(path)
        first = next(iter(frozen["models"]))
        frozen["models"][first].append(copy.deepcopy(frozen["models"][first][0]))
        write_json(path, frozen)
        freeze = read_json(self.out / "freeze.json")
        freeze["frozen_predictions_hash"] = digest(frozen)
        write_json(self.out / "freeze.json", freeze)
        _seal_existing(self.out)
        self.assert_rejected("Frozen prediction case coverage differs")

    def test_diagnostic_only_completion_has_valid_distinct_audit_scope(self):
        for path in self.out.iterdir():
            if path.name not in {"plan.json", "environment.json", "source_identity.json", "diagnostics", "diagnostics_summary.json"}:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        env = read_json(self.out / "environment.json")
        write_json(self.out / "status.json", {"status": "diagnostics_completed", "session_id": env["session_id"],
                                             "measurement_kind": "simulated", "test_opened": False})
        _seal_existing(self.out)
        result = campaign.audit_run(self.out)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["diagnostics_complete"])
        self.assertFalse(result["complete_session"])


class NumericalWitnessAuditIntegrationTests(unittest.TestCase):
    """Only hand-built numerical-array fixtures, with no CUDA execution claim."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name) / "failed_numerical_fixture"
        self.out.mkdir()
        plan = build_plan()
        # A failed-stage fixture permits partial witness evidence. The string
        # names the schema branch under test; it does not represent hardware.
        env = {"measurement_kind": "cuda_measured", "environment_id": "TEST_FIXTURE_NOT_HARDWARE", "session_id": "cpu_array_fixture"}
        status = {"status": "failed", "measurement_kind": "cuda_measured", "session_id": "cpu_array_fixture"}
        write_json(self.out / "plan.json", plan)
        write_json(self.out / "environment.json", env)
        write_json(self.out / "status.json", status)
        folder = self.out / "diagnostics" / "numerical_witnesses"
        folder.mkdir(parents=True)
        x, w = witness_inputs("identity")
        x, w = x.astype(np.float16).astype(np.float32), w.astype(np.float16).astype(np.float32)
        self.archive = folder / "identity.npz"
        np.savez_compressed(self.archive, input_x=x, weights_nk=w, output_y=x.copy(),
                            reference_fp32=x.copy(), input_dtype=np.array("float16"), witness_name=np.array("identity"))
        self.summary = {"plan_hash": digest(plan), "source_environment_id": env["environment_id"],
                        "source_measurement_kind": "cuda_measured", "measurement_kind": "numerical_witness",
                        "excluded_from_model": True, "status": "failed", "arrays_archived": True,
                        "records": [{"name": "identity", "file": "identity.npz",
                                     "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                                     "gpu_check": {"passed": True, "finite": True, "relative_l2_error": 0.0, "max_abs_error": 0.0},
                                     "independent_cpu_recomputation": audit_witness_npz(self.archive)}]}
        write_json(folder / "witnesses.json", self.summary)
        _seal_existing(self.out)

    def tearDown(self):
        self.temp.cleanup()

    def test_partial_witness_is_independently_recomputed_and_not_hardware_attestation(self):
        result = campaign.audit_run(self.out)
        self.assertTrue(result["passed"], result)
        self.assertFalse(result["complete_session"])
        self.assertIn("no independent GPU attestation", result["scope"])

    def test_changed_npz_output_is_rejected_even_after_file_and_manifest_hashes_updated(self):
        with np.load(self.archive, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays["output_y"].fill(1.0)
        np.savez_compressed(self.archive, **arrays)
        self.summary["records"][0]["sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        write_json(self.archive.parent / "witnesses.json", self.summary)
        _seal_existing(self.out)
        result = campaign.audit_run(self.out)
        self.assertFalse(result["passed"], result)
        self.assertIn("small-witness check", "\n".join(result["errors"]))

    def test_stored_cpu_recomputation_cannot_be_fabricated(self):
        self.summary["records"][0]["independent_cpu_recomputation"]["fp64_relative_l2_error"] = 0.5
        write_json(self.archive.parent / "witnesses.json", self.summary)
        _seal_existing(self.out)
        result = campaign.audit_run(self.out)
        self.assertFalse(result["passed"], result)
        self.assertIn("stored CPU recomputation differs", "\n".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
