"""CPU checks of the offline collector contract; no CUDA execution is claimed."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from linear_perf import benchmark as bench
from linear_perf import dataset_measurement as dm
from linear_perf.common import LabError, read_json, read_jsonl, write_json, write_jsonl
from tests.test_graph import FakeEvent, FakeGraph, FakeTensor, FakeTorch


def tiny_manifest():
    return {
        "schema_version": 2, "seed": 71, "dtypes": ["float16", "bfloat16"],
        "cases": [{"id": split, "m": index + 1, "n": 64 + index * 16,
                   "k": 64, "split": split, "weight": index + 0.5}
                  for index, split in enumerate(("train", "test", "challenge", "stress"))],
        "measurement": {"visits": 2, "repeats": 3, "warmup_min_seconds": 0.01},
    }


class CollectorSettingsTests(unittest.TestCase):
    def test_adaptive_interval_targets_and_rotation_use_complete_buffer_cycles(self):
        self.assertEqual(dm.choose_operation_count(0.01, target_replay_ms=2, minimum=1, maximum=256), 200)
        self.assertEqual(dm.choose_operation_count(3, target_replay_ms=2, minimum=1, maximum=256), 1)
        self.assertEqual(dm.choose_operation_count(0.00001, target_replay_ms=2, minimum=1, maximum=256), 256)
        self.assertEqual(dm.choose_operation_count(0.3, target_replay_ms=2, minimum=1, maximum=256, buffer_sets=4), 8)
        self.assertEqual(dm.choose_operation_count(3, target_replay_ms=2, minimum=1, maximum=255, buffer_sets=4), 4)
        self.assertEqual(dm.choose_operation_count(0.00001, target_replay_ms=2, minimum=1, maximum=255, buffer_sets=4), 252)
        with self.assertRaises(LabError):
            dm.choose_operation_count(1, target_replay_ms=2, minimum=5, maximum=7, buffer_sets=4)

    def test_divide_whole_event_interval_by_actual_captured_gemm_count(self):
        self.assertEqual(dm.normalize_event_intervals([2.0, 2.4, 1.6], 8), [0.25, 0.3, 0.2])
        for intervals, count in (([1], 0), ([], 1), ([float("nan")], 4), ([-1], 4), ([1], True)):
            with self.subTest(intervals=intervals, count=count), self.assertRaises(LabError):
                dm.normalize_event_intervals(intervals, count)

    def test_invalid_resource_settings_fail_before_cuda_or_directory_creation(self):
        for update in ({"rotation_buffers": 4, "max_operations": 3},
                       {"min_operations": 5, "max_operations": 7, "rotation_buffers": 4},
                       {"warmup_min_seconds": 20}, {"memory_fraction": 0.99},
                       {"max_flops": -1}, {"l2_cache_bytes": "A100"}, {"unknown": 1},
                       *({"max_gemm_calls": value} for value in (0, -1, True, 1.5, "unlimited"))):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as tmp:
                manifest = tiny_manifest()
                manifest["measurement"].update(update)
                with patch.object(bench, "_load_torch") as load:
                    with self.assertRaises(LabError):
                        dm.collect_dataset(manifest, Path(tmp) / "invalid", session_id="test")
                    load.assert_not_called()
                    self.assertFalse((Path(tmp) / "invalid").exists())

    def test_call_accounting_has_no_default_cap_but_explicit_limits_still_apply(self):
        settings = dm.measurement_settings(tiny_manifest())
        self.assertIsNone(settings["max_gemm_calls"])
        budget = dm._WorkBudget(settings)
        # Arithmetic-only accounting of a plausible long warmup campaign;
        # no actual or simulated GPU work is performed by this check.
        budget.charge(100_000_001, 2)
        self.assertEqual(budget.summary()["gemm_calls_accounted"], 100_000_001)
        self.assertIsNone(budget.summary()["max_gemm_calls"])

        capped = dm._WorkBudget({**settings, "max_gemm_calls": 10})
        capped.charge(10, 2)
        with self.assertRaisesRegex(LabError, "budget exhausted"):
            capped.charge(1, 2)
        self.assertEqual(capped.summary()["gemm_calls_accounted"], 10)

        flop_cap = dm._WorkBudget({**settings, "max_flops": 8})
        flop_cap.charge(2, 4)
        with self.assertRaisesRegex(LabError, "budget exhausted"):
            flop_cap.charge(1, 4)

    def test_schedule_covers_every_shape_dtype_and_balances_first_precision(self):
        manifest = tiny_manifest()
        manifest["cases"].append({"id": "extra", "m": 5, "n": 64, "k": 64, "split": "train"})
        schedule = dm.make_schedule(manifest["cases"], manifest["dtypes"], 4, manifest["seed"])
        self.assertEqual(schedule, dm.make_schedule(manifest["cases"], manifest["dtypes"], 4, manifest["seed"]))
        self.assertNotEqual(schedule, dm.make_schedule(manifest["cases"], manifest["dtypes"], 4, manifest["seed"] + 1))
        starts = Counter()
        for visit in range(1, 5):
            tasks = [task for task in schedule if task["visit_index"] == visit]
            self.assertEqual({task["case_id"] for task in tasks}, {case["id"] for case in manifest["cases"]})
            count = Counter(task["dtype_order"][0] for task in tasks)
            self.assertEqual(abs(count["float16"] - count["bfloat16"]), 1)
            starts.update(count)
            for task in tasks:
                self.assertEqual(set(task["dtype_order"]), set(manifest["dtypes"]))
                self.assertEqual(set(task["cache_policy_order"]), {"warm_reuse", "rotating_buffers"})
        self.assertEqual(starts["float16"], starts["bfloat16"])

    def test_repeat_subset_has_exact_stratified_budget_without_latency_inputs(self):
        manifest = tiny_manifest()
        manifest["cases"] = [{"id": f"case_{i:03}", "m": i + 1,
                              "n": 64 + i % 2, "k": 64,
                              "split": ("train", "test", "challenge", "stress")[i % 4], "weight": 1}
                             for i in range(100)]
        selected = dm.repeat_subset(manifest)
        self.assertEqual(len(selected), math.ceil(100 * 0.125))
        other = copy.deepcopy(manifest)
        other["cases"].reverse()
        for case in other["cases"]:
            case.update(median_ms=999999, prediction_error=1000)
        self.assertEqual(selected, dm.repeat_subset(other))
        lookup = {case["id"]: case for case in manifest["cases"]}
        self.assertEqual({lookup[cid]["split"] for cid in selected}, {"train", "test", "challenge", "stress"})

    def test_l2_property_and_current_cuda_context_query_do_not_guess_device_name(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(l2_cache_size=12 * 1024**2)))
        with patch.object(dm.ctypes, "CDLL") as loader:
            self.assertEqual(dm._l2_cache_size(torch, "cuda:0")["bytes"], 12 * 1024**2)
            loader.assert_not_called()

        calls = []

        class Function:
            def __init__(self, callback):
                self.callback = callback

            def __call__(self, *args):
                return self.callback(*args)

        def current(pointer):
            pointer._obj.value = 7  # Current context can differ from logical index zero.
            return 0

        def attribute(pointer, enum, selected):
            calls.append((enum, selected))
            pointer._obj.value = 18 * 1024**2
            return 0

        driver = SimpleNamespace(cuCtxGetDevice=Function(current), cuDeviceGetAttribute=Function(attribute))
        torch.cuda = SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(), current_stream=lambda device: None)
        with patch.object(dm.ctypes.util, "find_library", return_value="fake_cuda"), patch.object(dm.ctypes, "CDLL", return_value=driver):
            result = dm._l2_cache_size(torch, "cuda:0")
        self.assertEqual(result["bytes"], 18 * 1024**2)
        self.assertEqual(calls, [(38, 7)])
        with patch.object(dm.ctypes, "CDLL", side_effect=OSError("unavailable")):
            self.assertIsNone(dm._l2_cache_size(torch, "cuda:0")["bytes"])

    def test_unknown_l2_skips_rotation_but_weight_threshold_is_inclusive(self):
        case, settings = tiny_manifest()["cases"][0], dm.measurement_settings(tiny_manifest())
        environment = {"device": {"l2_cache_bytes": None}}
        self.assertFalse(dm.rotation_eligibility(case, "float16", environment, settings)["eligible"])
        environment["device"]["l2_cache_bytes"] = 4096
        self.assertTrue(dm.rotation_eligibility(case, "float16", environment, settings)["eligible"])
        environment["device"]["l2_cache_bytes"] = 4095
        self.assertFalse(dm.rotation_eligibility(case, "float16", environment, settings)["eligible"])

    def test_observed_l2_wins_and_explicit_override_only_fills_unavailable_query(self):
        manifest = tiny_manifest()
        manifest["measurement"]["l2_cache_bytes"] = 4 * 1024**2
        torch = FakeTorch()
        torch.__version__, torch.version = "fake", SimpleNamespace(cuda="fake")
        for observed, expected, applied in ((8 * 1024**2, 8 * 1024**2, False), (None, 4 * 1024**2, True)):
            with self.subTest(observed=observed), patch.object(bench, "_load_torch", return_value=torch), \
                    patch.object(bench, "_require_cuda", return_value="cuda:0"), \
                    patch.object(bench, "_precision_policy", return_value=nullcontext({"mode": "fake"})), \
                    patch.object(bench, "_driver_versions", return_value=[]), \
                    patch.object(bench, "_gpu_metadata", return_value={"name": "fake", "multiprocessor_count": 4}), \
                    patch.object(dm, "_l2_cache_size", return_value={"bytes": observed, "source": "fake_query"}):
                with dm._CudaBackend(manifest, dm.measurement_settings(manifest), "cuda:0", float("inf")) as backend:
                    device = backend.environments["device"]
                    self.assertEqual(device["l2_cache_bytes"], expected)
                    self.assertEqual(device["l2_cache_override_applied"], applied)


class DatasetCollectionTests(unittest.TestCase):
    def collect(self, path, manifest=None, **kwargs):
        return dm.collect_dataset(manifest or tiny_manifest(), path, session_id="test_session", simulate=True,
                                  progress=lambda message: None, **kwargs)

    def test_collect_all_splits_before_profiles_without_importing_torch_or_models(self):
        manifest, events = tiny_manifest(), []
        original_measure, original_profile = dm._SimulatedBackend.measure, dm._SimulatedBackend.profile

        def measure(backend, case, dtype, policy):
            events.append(("timing", case["id"], dtype, policy))
            return original_measure(backend, case, dtype, policy)

        def profile(backend, case, dtype, operations, destination):
            events.append(("profile", case["id"], dtype))
            result = original_profile(backend, case, dtype, operations, destination)
            result["diagnostic_duration_ms"] = 9999999.0
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, "_load_torch", side_effect=AssertionError("no torch")), \
                patch.object(dm._SimulatedBackend, "measure", measure), patch.object(dm._SimulatedBackend, "profile", profile):
            path = Path(tmp) / "dataset"
            result = self.collect(path, manifest)
            self.assertEqual(result["status"], "simulated_complete", result.get("error"))
            self.assertTrue(result["validation"]["passed"])
            self.assertFalse(result["hardware_measurements_collected"])
            self.assertFalse(result["predictive_accuracy_established"])
            rows = read_jsonl(path / "measurements.jsonl")
            self.assertEqual(len(rows), 4 * 2 * 2)
            self.assertEqual({row["split"] for row in rows}, {"train", "test", "challenge", "stress"})
            self.assertEqual(len({row["environment_id"] for row in rows}), 4)
            self.assertTrue(all(row["measurement_kind"] == "simulated" and not row["numeric_execution_checked"] for row in rows))
            self.assertTrue(all(row["median_ms"] < 9999999 for row in rows))
            self.assertFalse(result["heldout_measurement_requires_frozen_model"])
            first_profile = next(i for i, event in enumerate(events) if event[0] == "profile")
            self.assertTrue(all(event[0] == "timing" for event in events[:first_profile]))
            self.assertTrue(all(event[0] == "profile" for event in events[first_profile:]))
            visits = {visit["visit_id"]: visit for visit in read_jsonl(path / "raw_visits.jsonl")}
            for row in rows:
                self.assertEqual(row["samples_ms"], [visits[visit_id]["median_ms"] for visit_id in row["raw_visit_ids"]])

    def test_optional_call_cap_round_trips_through_environment_status_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = tiny_manifest()
            manifest["measurement"]["max_gemm_calls"] = None
            path = Path(tmp) / "dataset"
            result = self.collect(path, manifest)
            self.assertEqual(result["status"], "simulated_complete", result.get("error"))
            self.assertIsNone(result["work_budget"]["max_gemm_calls"])
            self.assertIsNone(read_json(path / "measurement_settings.json")["max_gemm_calls"])
            for context in read_json(path / "environments.json")["contexts"].values():
                self.assertIsNone(context["identity"]["measurement_settings"]["max_gemm_calls"])
            self.assertTrue(dm.validate_collected_dataset(path)["passed"])

    def test_profiles_do_not_change_primary_timing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            with_profile = tiny_manifest()
            without_profile = copy.deepcopy(with_profile)
            without_profile["measurement"]["profile"] = False
            self.collect(Path(tmp) / "on", with_profile)
            self.collect(Path(tmp) / "off", without_profile)
            on, off = [read_jsonl(Path(tmp) / name / "measurements.jsonl") for name in ("on", "off")]
            fields = ("case_id", "dtype", "cache_policy", "median_ms", "samples_ms", "raw_samples_ms_by_visit")
            self.assertEqual([{field: row[field] for field in fields} for row in on],
                             [{field: row[field] for field in fields} for row in off])

    def test_repeat_subset_preserves_original_manifest_and_requires_no_training_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, manifest = Path(tmp) / "repeat", tiny_manifest()
            result = self.collect(path, manifest, case_ids=["challenge", "stress"])
            self.assertEqual(result["status"], "simulated_complete", result.get("error"))
            self.assertEqual(read_json(path / "manifest.json"), manifest)
            self.assertEqual(result["selected_case_ids"], ["challenge", "stress"])
            self.assertTrue(result["repeat_subset"])
            self.assertTrue(dm.validate_collected_dataset(path)["passed"])
            self.assertEqual({row["split"] for row in read_jsonl(path / "measurements.jsonl")}, {"challenge", "stress"})

    def test_raw_operation_divisor_and_aggregate_edits_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset"
            self.collect(path)
            raw = read_jsonl(path / "raw_visits.jsonl")
            original = copy.deepcopy(raw)
            raw[0]["event_elapsed_ms"][0] *= 2
            write_jsonl(path / "raw_visits.jsonl", raw)
            checked = dm.validate_collected_dataset(path)
            self.assertFalse(checked["passed"])
            self.assertIn("operation count", checked["error"])
            write_jsonl(path / "raw_visits.jsonl", original)
            rows = read_jsonl(path / "measurements.jsonl")
            rows[0]["median_ms"] *= 2
            write_jsonl(path / "measurements.jsonl", rows)
            self.assertIn("Aggregate", dm.validate_collected_dataset(path)["error"])

    def test_deleting_eligible_rotation_cannot_pass_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset"
            self.collect(path)
            rows = read_jsonl(path / "measurements.jsonl")
            rows = [row for row in rows if not (row["case_id"] == "train" and row["dtype"] == "float16" and row["cache_policy"] == "rotating_buffers")]
            write_jsonl(path / "measurements.jsonl", rows)
            result = dm.validate_collected_dataset(path)
            self.assertFalse(result["passed"])
            self.assertIn("rotation coverage", result["error"])

    def test_partial_failure_retains_completed_visits_and_active_attempt(self):
        original = dm._SimulatedBackend.measure
        counter = [0]

        def fail(backend, case, dtype, policy):
            if not case["id"].startswith("control_"):
                counter[0] += 1
                if counter[0] == 5:
                    raise RuntimeError("controlled collection failure")
            return original(backend, case, dtype, policy)

        with tempfile.TemporaryDirectory() as tmp, patch.object(dm._SimulatedBackend, "measure", fail):
            path = Path(tmp) / "dataset"
            result = self.collect(path)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(read_jsonl(path / "raw_visits.jsonl")), 4)
            failure = read_json(path / "failure.json")
            self.assertEqual(failure["error"], "controlled collection failure")
            self.assertIsNotNone(failure["active_variant"])
            self.assertTrue(failure["failed_attempt_is_not_a_training_label"])
            self.assertFalse(dm.validate_collected_dataset(path)["passed"])
            with self.assertRaisesRegex(LabError, "already exists"):
                self.collect(path)

    def test_optional_rotation_memory_skip_retains_primary_and_partial_rotations(self):
        original, seen = dm._SimulatedBackend.measure, Counter()

        def guard(backend, case, dtype, policy):
            key = (case["id"], dtype, policy)
            seen[key] += 1
            if key == ("train", "float16", "rotating_buffers") and seen[key] == 2:
                raise dm._MemoryGuardError({"passed": False, "estimated_peak_bytes": 200,
                                            "available_bytes": 100, "memory_fraction": 0.65})
            return original(backend, case, dtype, policy)

        with tempfile.TemporaryDirectory() as tmp, patch.object(dm._SimulatedBackend, "measure", guard):
            path = Path(tmp) / "dataset"
            result = self.collect(path)
            self.assertEqual(result["status"], "simulated_complete", result.get("error"))
            self.assertEqual(result["completed_rows"], 15)
            skips = read_jsonl(path / "skipped_variants.jsonl")
            self.assertEqual(skips[0]["incomplete_visits_retained"], 1)
            self.assertTrue(dm.validate_collected_dataset(path)["passed"])

    def test_primary_memory_guard_and_unavailable_cuda_never_fall_back_to_simulation(self):
        original = dm._SimulatedBackend.measure

        def guard(backend, case, dtype, policy):
            if case["id"] == "train" and policy == "warm_reuse":
                raise dm._MemoryGuardError({"passed": False})
            return original(backend, case, dtype, policy)

        with tempfile.TemporaryDirectory() as tmp, patch.object(dm._SimulatedBackend, "measure", guard):
            self.assertEqual(self.collect(Path(tmp) / "guard")["status"], "failed")
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench, "_load_torch", side_effect=LabError("CUDA is unavailable")):
            path = Path(tmp) / "no_cuda"
            result = dm.collect_dataset(tiny_manifest(), path, session_id="test", progress=lambda message: None)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["measurement_kind"], "cuda_measured")
            self.assertFalse(result["hardware_measurements_collected"])
            self.assertEqual(read_jsonl(path / "measurements.jsonl"), [])


class TrackedOutput(FakeTensor):
    def __init__(self, index):
        self.index, self.computed = index, False

    def fill_(self, value):
        self.computed = False
        return self


class TrackedGraph(FakeGraph):
    def __init__(self, cuda):
        super().__init__(cuda)
        self.outputs = []

    def replay(self):
        super().replay()
        for output in self.outputs:
            output.computed = True


class TrackedTorch(FakeTorch):
    def __init__(self):
        super().__init__()

        def graph():
            result = TrackedGraph(self.cuda)
            self.cuda.graphs.append(result)
            return result

        self.cuda.CUDAGraph = graph

    def mm(self, x, w, *, out=None):
        result = super().mm(x, w, out=out)
        if out is not None:
            if self.cuda.capturing is None:
                out.computed = True
            else:
                self.cuda.capturing.outputs.append(out)
        return result


class FakeCudaControlFlowTests(unittest.TestCase):
    def make_backend(self):
        manifest = tiny_manifest()
        settings = dm.measurement_settings(manifest)
        backend = dm._CudaBackend(manifest, settings, "cuda:0", float("inf"))
        backend.torch, backend.selected = TrackedTorch(), "cuda:0"
        return backend

    def test_pilot_recapture_normalization_and_every_rotating_output_replayed(self):
        backend, outputs = self.make_backend(), []

        def allocate(*args):
            output = TrackedOutput(len(outputs))
            outputs.append(output)
            return FakeTensor(), FakeTensor(), output

        def correctness(torch, output, reference, dtype):
            return {"passed": output.computed, "finite": output.computed}

        tick = [0.0]

        def clock():
            tick[0] += 0.01
            return tick[0]

        with patch.object(backend, "_guard", return_value={"passed": True}), \
                patch.object(bench, "_allocate_inputs", allocate), patch.object(bench, "_correctness", correctness), \
                patch.object(FakeEvent, "elapsed_time", return_value=1.0), patch.object(dm.time, "monotonic", clock):
            raw = backend.measure(tiny_manifest()["cases"][0], "float16", "rotating_buffers")
        self.assertEqual(raw["pilot"]["operations"], 8)
        self.assertEqual(raw["captured_gemm_count"], 16)  # 1 ms / 8 => 16 calls for a 2 ms target.
        self.assertEqual(raw["event_elapsed_ms"], [1.0] * 3)
        self.assertEqual(raw["samples_ms"], [1 / 16] * 3)
        self.assertEqual([graph.nodes for graph in backend.torch.cuda.graphs], [8, 16])
        for graph in backend.torch.cuda.graphs:
            exposure = Counter(output.index for output in graph.outputs)
            self.assertEqual(set(exposure), {0, 1, 2, 3})
            self.assertEqual(len(set(exposure.values())), 1)
        self.assertTrue(all(output.computed for output in outputs))
        self.assertTrue(all(check["passed"] for check in raw["correctness_after"]))
        self.assertEqual(backend.last_attempt["stage"], "completed")

    def test_partial_event_intervals_are_retained_when_later_event_fails(self):
        backend = self.make_backend()
        graph = TrackedGraph(backend.torch.cuda)
        graph.nodes = 4
        retained = []
        with patch.object(bench, "_event_sample", side_effect=[1.5, 2.0, RuntimeError("event failure")]):
            with self.assertRaisesRegex(RuntimeError, "event failure"):
                backend._intervals(graph, tiny_manifest()["cases"][0], 4, 3, retained=retained)
        self.assertEqual(retained, [1.5, 2.0])

    def test_work_budget_blocks_excess_graph_work_and_negative_charges(self):
        backend = self.make_backend()
        backend.work.limit_calls = 1
        with self.assertRaisesRegex(LabError, "budget exhausted"):
            backend._capture(backend.torch.cuda.operation, tiny_manifest()["cases"][0], 8)
        self.assertEqual(backend.torch.cuda.graphs, [])
        with self.assertRaisesRegex(LabError, "nonnegative"):
            backend.work.charge(-1, 10)

    def test_profiler_budget_exhaustion_is_fatal_not_disguised_as_missing_profiler(self):
        backend = self.make_backend()
        backend.work.limit_calls = 1
        graph = TrackedGraph(backend.torch.cuda)
        graph.nodes = 4
        with patch.object(backend, "_prepare", return_value=([], [], [], lambda: None, {})), \
                patch.object(backend, "_capture", return_value=graph), patch.object(backend, "_warm", return_value={}):
            with self.assertRaisesRegex(LabError, "budget exhausted"):
                backend.profile(tiny_manifest()["cases"][0], "float16", 4, Path("unused.json"))


if __name__ == "__main__":
    unittest.main()
