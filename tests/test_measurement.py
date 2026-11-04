"""Contract, corruption and sequencing checks; no GPU speed claim is made."""

from __future__ import annotations

import copy
import json
import statistics
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from linear_perf import benchmark
from linear_perf.collection import validate_aggregate_row
from linear_perf.common import LabError, digest, validate_measurements
from linear_perf.diagnostics import profile_graph
from linear_perf.measurement import (
    MeasurementError, TelemetryRecorder, collect_stage, make_environment,
    quality_summary, run_diagnostics,
)
from linear_perf.numerical_witness import audit_witness_npz, witness_inputs
from linear_perf.protocol import build_plan


class MeasurementContracts(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan()
        self.environment = make_environment(self.plan, "unit-session", simulate=True)
        self.cases = [c for c in self.plan["cases"] if c["split"] == "calibration"][:4]
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def collect(self, folder="stage"):
        return collect_stage(
            self.plan, self.cases, self.environment, self.root / folder,
            session_id="unit-session", progress=None,
        )

    def test_complete_stage_retains_all_nested_data_and_schedule(self):
        result = self.collect()
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(len(result["controls"]), 9)  # Start, after 8 visits, end.
        state = json.loads(Path(result["raw_path"]).read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["rows"]), 12)
        self.assertEqual(len(state["schedule"]), 21)
        self.assertEqual(state["completed_visit_count"], len(state["schedule"]))
        self.assertEqual(state["schedule_hash"], digest(state["schedule"]))
        self.assertEqual(state["rows_hash"], digest(state["rows"]))
        self.assertTrue(state["schedule_declared_before_measurement"])
        self.assertIsInstance(result["quality"]["passed"], bool)
        for row in result["rows"]:
            validate_aggregate_row(row, expected_blocks=3, expected_repeats=9)
            self.assertEqual(row["measurement_kind"], "simulated")
            self.assertEqual(len(row["measurement_blocks"]), 3)
            self.assertEqual(row["samples_ms"], [x["median_ms"] for x in row["measurement_blocks"]])
            for raw in row["measurement_blocks"]:
                self.assertEqual(len(raw["samples_ms"]), 9)
                self.assertLessEqual(raw["visit_started_monotonic_seconds"], raw["visit_finished_monotonic_seconds"])
                self.assertIn("visit_started_at_utc", raw)
        validate_measurements(result["rows"], self.plan)

    def test_controls_cannot_be_accepted_as_model_labels(self):
        result = self.collect()
        for control in result["controls"]:
            self.assertTrue(control["excluded_from_model"])
            self.assertEqual(control["measurement_kind"], "simulated_diagnostic")
            self.assertIn("case_id", control)
            self.assertIn("median_ms", control)
            with self.assertRaises(LabError):
                validate_measurements([control], self.plan)

    def test_diagnostic_cases_cannot_be_requested_as_primary_rows(self):
        case = next(c for c in self.plan["cases"] if c["id"] in self.plan["study"]["diagnostic_case_ids"])
        with self.assertRaisesRegex(LabError, "Diagnostic"):
            collect_stage(
                self.plan, [case], self.environment, self.root / "bad",
                session_id="unit-session", progress=None,
            )
        self.assertFalse((self.root / "bad").exists())

    def test_session_and_plan_identity_are_not_relaxed(self):
        changed_identity = copy.deepcopy(self.environment)
        changed_identity["identity"]["dtype"] = "bfloat16"
        for env in (
            {**self.environment, "session_id": "different"},
            {**self.environment, "plan_hash": "wrong"},
            {**self.environment, "measurement_kind": "cuda_diagnostic"},
            changed_identity,
        ):
            with self.assertRaises(LabError):
                collect_stage(self.plan, self.cases, env, self.root / "bad",
                              session_id="unit-session", progress=None)
        self.assertFalse((self.root / "bad").exists())

    def test_no_overwrite_or_automatic_rerun(self):
        self.collect()
        original = (self.root / "stage" / "raw_visits.json").read_bytes()
        with self.assertRaises(LabError):
            self.collect()
        self.assertEqual(original, (self.root / "stage" / "raw_visits.json").read_bytes())

    def test_corrupt_nested_sample_is_rejected(self):
        result = self.collect()
        row = copy.deepcopy(result["rows"][0])
        row["measurement_blocks"][0]["samples_ms"][0] *= 1.5
        with self.assertRaises(LabError):
            validate_aggregate_row(row)

    def test_failure_retains_prior_raw_and_full_planned_order(self):
        from linear_perf import measurement as module
        actual = module._raw_visit
        calls = [0]

        def failing(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 6:
                raise LabError("injected numerical failure")
            return actual(*args, **kwargs)

        with patch.object(module, "_raw_visit", side_effect=failing):
            with self.assertRaises(MeasurementError) as caught:
                self.collect()
        state = json.loads((self.root / "stage" / "raw_visits.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed_visit_count"], 5)
        self.assertEqual(len(state["controls"]), 3)
        self.assertEqual(len(state["rows"]), 2)
        self.assertEqual(len(state["schedule"]), 21)
        self.assertEqual(len(caught.exception.raw_rows), 2)
        self.assertEqual(calls[0], 6)
        self.assertEqual(state["automatic_retries"], 0)

    def test_expired_deadline_stops_before_collection(self):
        with self.assertRaises(LabError):
            collect_stage(
                self.plan, self.cases, self.environment, self.root / "expired",
                session_id="unit-session", progress=None,
                deadline_monotonic=time.monotonic() - 1,
            )
        self.assertFalse((self.root / "expired").exists())

    def test_real_mode_cannot_fall_back_to_simulated_or_CPU(self):
        no_cuda = SimpleNamespace(version=SimpleNamespace(cuda=None, hip=None))
        with patch.object(benchmark, "_load_torch", return_value=no_cuda):
            with self.assertRaisesRegex(LabError, "CUDA"):
                make_environment(self.plan, "real-request", simulate=False)

    def test_detect_control_drift_without_correcting_or_deleting_labels(self):
        result = self.collect()
        controls = copy.deepcopy(result["controls"])
        case_id = controls[0]["case_id"]
        target = next(row for row in reversed(controls) if row["case_id"] == case_id)
        target["samples_ms"] = [value * 3 for value in target["samples_ms"]]
        target["median_ms"] = statistics.median(target["samples_ms"])
        original = copy.deepcopy(result["rows"])
        quality = quality_summary(self.plan, result["rows"], controls)
        self.assertFalse(quality["passed"])
        self.assertTrue(any(f["reason"] == "control_drift" for f in quality["flags"]))
        self.assertEqual(original, result["rows"])
        self.assertEqual(len(controls), len(result["controls"]))

    def test_missing_registered_controls_cannot_pass_quality(self):
        result = self.collect()
        retained = [c for c in result["controls"]
                    if c["case_id"] != self.plan["study"]["control_case_ids"][0]]
        summary = quality_summary(self.plan, result["rows"], retained)
        self.assertFalse(summary["passed"])
        self.assertTrue(any(flag["reason"] == "missing_registered_control"
                            for flag in summary["flags"]))

    def test_telemetry_is_rate_limited_even_when_force_requested(self):
        from linear_perf import measurement as module
        clock = [100.0]
        snapshot = {
            "observed_at_utc": "unit", "monotonic_seconds": 100.0,
            "status": "unavailable", "hardware_state_observed": False,
        }
        with patch.object(module.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(module, "_telemetry_snapshot", return_value=snapshot.copy()) as query:
            recorder = TelemetryRecorder(
                self.root / "telemetry.jsonl", interval_seconds=1.0,
                simulate=True, device="cuda:0", deadline=200,
            )
            try:
                recorder.sample("start", force=True)
                clock[0] = 100.25
                self.assertIsNone(recorder.sample("too_soon", force=True))
                clock[0] = 101.25
                recorder.sample("later")
            finally:
                recorder.close()
        self.assertEqual(query.call_count, 2)
        rows = [json.loads(line) for line in (self.root / "telemetry.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["observation_scope"] == "case_boundary_outside_CUDA_event_intervals" for r in rows))

    def test_diagnostics_are_separate_balanced_and_never_fake_GPU_witnesses(self):
        result = run_diagnostics(
            self.plan, self.environment, self.root / "diagnostics",
            session_id="unit-session", progress=None,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["records"]), 8 * 6 * 2)
        self.assertEqual(len(result["rotation_records"]), 2 * 2 * 2)
        self.assertFalse(result["main_timing_policy_changed_from_diagnostic_results"])
        self.assertEqual(result["numerical_witnesses"]["status"], "skipped")
        self.assertFalse(list((self.root / "diagnostics" / "numerical_witnesses").glob("*.npz")))
        for case_id in self.plan["study"]["diagnostic_case_ids"]:
            first = [t["variant"] for t in result["schedule"] if t["case_id"] == case_id and t["pass_index"] == 1]
            second = [t["variant"] for t in result["schedule"] if t["case_id"] == case_id and t["pass_index"] == 2]
            self.assertEqual(first, list(reversed(second)))
        for record in result["records"]:
            self.assertTrue(record["excluded_from_model"])
            self.assertEqual(record["measurement_kind"], "simulated_diagnostic")
            self.assertIn("visit_started_monotonic_seconds", record)
        for record in result["profiles"]:
            self.assertEqual(record["execution_mode"], "cuda_graph")
            self.assertEqual(record["status"], "unavailable")
            self.assertIsNone(record["torch_trace_path"])


class WitnessAudits(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save_witness(self, name="random", *, change_output=False, change_reference=False, unrepresented=False):
        x, w = witness_inputs(name)
        x = x.astype(np.float16).astype(np.float32)
        w = w.astype(np.float16).astype(np.float32)
        reference = (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)
        output = reference.astype(np.float16).astype(np.float32)
        if change_output:
            output[0, 0] += 8
        if change_reference:
            reference[0, 0] += 1
        if unrepresented:
            x[0, 0] = np.float32(0.1234567)
        path = self.root / (name + ".npz")
        np.savez_compressed(
            path, input_x=x, weights_nk=w, output_y=output,
            reference_fp32=reference, input_dtype=np.array("float16"), witness_name=np.array(name),
        )
        return path

    def test_independent_float64_audit_all_small_fixtures(self):
        for name in ("zero", "identity", "alternating", "random"):
            audit = audit_witness_npz(self.save_witness(name))
            self.assertTrue(audit["passed"])
            self.assertIn("NumPy float64", audit["comparison"])

    def test_wrong_output_rejected(self):
        with self.assertRaises(LabError):
            audit_witness_npz(self.save_witness(change_output=True))

    def test_wrong_reference_rejected(self):
        with self.assertRaises(LabError):
            audit_witness_npz(self.save_witness(change_reference=True))

    def test_input_not_representable_in_claimed_dtype_rejected(self):
        with self.assertRaisesRegex(LabError, "representable"):
            audit_witness_npz(self.save_witness(unrepresented=True))

    def test_bfloat16_represented_values_are_auditable_without_torch(self):
        # Fixture construction alone emulates BF16 rounding. It is explicitly
        # not GPU output and is never included in run_campaign artifacts.
        def represented(array):
            bits = np.asarray(array, dtype=np.float32).view(np.uint32)
            rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
            return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

        x, w = witness_inputs("random")
        x, w = represented(x), represented(w)
        reference = (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)
        y = represented(reference)
        path = self.root / "bf16_fixture.npz"
        np.savez_compressed(
            path, input_x=x, weights_nk=w, output_y=y, reference_fp32=reference,
            input_dtype=np.array("bfloat16"), witness_name=np.array("random"),
        )
        result = audit_witness_npz(path)
        self.assertTrue(result["passed"])
        self.assertEqual(result["dtype"], "bfloat16")

    def test_witness_name_cannot_relabel_a_different_input_pattern(self):
        path = self.save_witness("zero")
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        arrays["witness_name"] = np.array("alternating")
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(LabError, "sign pattern"):
            audit_witness_npz(path)


class GraphProfileSequencing(unittest.TestCase):
    """A call-order test, explicitly not a CUDA functional/performance test."""

    def test_profile_uses_graph_replay_and_reports_missing_CUDA_activity(self):
        plan = build_plan()
        case = next(c for c in plan["cases"] if c["id"] == plan["study"]["diagnostics"]["profile_case_ids"][0])
        environment = {"environment_id": "gpu-test", "identity": {"test": "mock"},
                       "measurement_kind": "cuda_measured"}
        tensor = SimpleNamespace(float=lambda: tensor, fill_=lambda value: None)
        counts = {"mm": 0, "replay": 0, "checks": 0}

        def mm(*args, **kwargs):
            counts["mm"] += 1
            return tensor

        def replay():
            counts["replay"] += 1

        class Profiler:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def export_chrome_trace(self, path):
                Path(path).write_text(json.dumps({"traceEvents": [
                    {"cat": "cpu_op", "name": "graph replay wrapper", "ph": "X", "dur": 1},
                ]}))

        torch = SimpleNamespace(
            cuda=SimpleNamespace(device=lambda x: nullcontext(), synchronize=lambda x: None),
            inference_mode=lambda: nullcontext(), autocast=lambda **kwargs: nullcontext(),
            mm=mm,
            profiler=SimpleNamespace(
                profile=lambda **kwargs: Profiler(),
                record_function=lambda label: nullcontext(),
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
            ),
        )
        passed = {"passed": True, "finite": True}
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(benchmark, "_load_torch", return_value=torch), \
                patch.object(benchmark, "_require_cuda", return_value="cuda:0"), \
                patch.object(benchmark, "_precision_policy", return_value=nullcontext({})), \
                patch.object(benchmark, "_collect_environment", return_value=environment), \
                patch.object(benchmark, "_check_memory", return_value={}), \
                patch.object(benchmark, "_allocate_inputs", return_value=(tensor, tensor, tensor)), \
                patch.object(benchmark, "_correctness", return_value=passed) as checks, \
                patch.object(benchmark, "_capture_operation_graph", return_value=SimpleNamespace(replay=replay)), \
                patch.object(benchmark, "_warm_graph", return_value={"test": "mock"}):
            record = profile_graph(
                plan, case, environment, Path(folder) / "profile",
                device="cuda:0", deadline=time.monotonic() + 20,
            )
            self.assertEqual(checks.call_count, 2)
        self.assertEqual(counts["replay"], 1)
        self.assertEqual(counts["mm"], 2)  # FP32 reference + precheck, no eager substitute in profile.
        self.assertEqual(record["status"], "unavailable")
        self.assertEqual(record["reason"], "trace_has_no_CUDA_kernel_activity")
        self.assertEqual(record["cuda_kernel_event_count"], 0)
        self.assertTrue(record["timing_excluded_from_model"])


if __name__ == "__main__":
    unittest.main()
