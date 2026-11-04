"""CPU-only contracts; these tests deliberately do not claim GPU validation."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from linear_perf import benchmark as bench
from linear_perf.common import LabError, digest, validate_plan


def tiny_plan() -> dict:
    return {
        "schema_version": 1,
        "seed": 20261031,
        "dtype": "bfloat16",
        "cache_policy": "warm_reuse",
        "timing": {"warmup": 2, "iterations": 3, "repeats": 3},
        "budget": {"max_acquisition_rounds": 2, "max_new_cases": 4, "max_wall_seconds": 30},
        "cases": [
            {"id": "cal", "m": 2, "n": 8, "k": 8, "split": "calibration"},
            {"id": "dev", "m": 3, "n": 8, "k": 8, "split": "development"},
            {"id": "test", "m": 4, "n": 8, "k": 8, "split": "test"},
        ],
    }


class _LegacyMatmul:
    def __init__(self) -> None:
        self.allow_tf32 = True
        self.allow_fp16_reduced_precision_reduction = True
        self.allow_bf16_reduced_precision_reduction = True


class _ModernMatmul:
    def __init__(self) -> None:
        self.fp32_precision = "tf32"
        self.allow_fp16_reduced_precision_reduction = True
        self.allow_bf16_reduced_precision_reduction = True
        self.allow_fp16_accumulation = True

    @property
    def allow_tf32(self) -> bool:
        raise AssertionError("The old precision API must not be read on the new API path")

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        raise AssertionError("The old precision API must not be written on the new API path")


def _math_only_torch(matmul: object) -> object:
    return types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=matmul)))


class TestLazyImportsAndGuards(unittest.TestCase):
    def test_import_does_not_load_torch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import linear_perf.benchmark; print('torch' in sys.modules)"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_torch_import_failure_is_actionable(self) -> None:
        with patch.object(importlib, "import_module", side_effect=ModuleNotFoundError("torch")):
            with self.assertRaisesRegex(LabError, "CUDA-enabled PyTorch"):
                bench._load_torch()

    def test_broken_torch_runtime_is_not_cpu_fallback(self) -> None:
        with patch.object(importlib, "import_module", side_effect=OSError("missing CUDA library")):
            with self.assertRaisesRegex(LabError, "runtime libraries"):
                bench._load_torch()

    def test_doctor_succeeds_without_torch(self) -> None:
        with patch.object(bench, "_load_torch", side_effect=LabError("PyTorch is not installed")), patch.object(bench, "_command_version", return_value=None), patch.object(bench.shutil, "which", return_value=None):
            result = bench.doctor()
        self.assertFalse(result["pytorch_available"])
        self.assertFalse(result["benchmark_ready"])
        self.assertEqual(result["gpus"], [])
        self.assertFalse(result["ncu_available"])
        self.assertIn("python_version", result)
        self.assertNotIn("environment_variables", result)
        json.dumps(result, allow_nan=False)

    def test_cuda_unavailable_guard(self) -> None:
        torch = types.SimpleNamespace(
            version=types.SimpleNamespace(cuda="12.8", hip=None),
            cuda=types.SimpleNamespace(is_available=lambda: False),
        )
        with self.assertRaisesRegex(LabError, "no CPU timing was collected"):
            bench._require_cuda(torch, "float32", "cuda:0")

    def test_rocm_is_not_labeled_nvidia_cuda(self) -> None:
        torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None, hip="6.0"))
        with self.assertRaisesRegex(LabError, "ROCm"):
            bench._require_cuda(torch, "float32", "cuda:0")

    def test_device_parser_rejects_cpu_and_ambiguous_inputs(self) -> None:
        for value in ("cpu", "mps", "cuda:-1", "cuda:0;echo", 0, None, "cuda:1.0"):
            with self.subTest(value=value), self.assertRaises(LabError):
                bench._device_index(value)
        self.assertIsNone(bench._device_index("cuda"))
        self.assertEqual(bench._device_index("cuda:12"), 12)

    def test_public_runner_rejects_cpu_before_import(self) -> None:
        plan = tiny_plan()
        with patch.object(bench, "_load_torch") as loader:
            with self.assertRaisesRegex(LabError, "CPU fallback"):
                bench.benchmark_cases(plan, [plan["cases"][0]], device="cpu")
            loader.assert_not_called()

    def test_public_gpu_apis_fail_honestly_without_torch(self) -> None:
        plan = tiny_plan()
        calls = [
            lambda: bench.cuda_environment(plan),
            lambda: bench.benchmark_cases(plan, [plan["cases"][0]]),
            lambda: bench.benchmark_batching(plan),
            lambda: bench.profile_case(plan, plan["cases"][0], "/tmp/unused-linear-perf-trace"),
        ]
        for call in calls:
            with self.subTest(call=call), patch.object(bench, "_load_torch", side_effect=LabError("PyTorch is not installed")):
                with self.assertRaisesRegex(LabError, "not installed"):
                    call()

    def test_environment_must_be_object(self) -> None:
        plan = tiny_plan()
        with patch.object(bench, "_load_torch") as loader:
            with self.assertRaisesRegex(LabError, "environment must be"):
                bench.benchmark_cases(plan, [plan["cases"][0]], environment=[])
            loader.assert_not_called()


class TestCaseAndResourceContracts(unittest.TestCase):
    def test_registered_subset_is_accepted(self) -> None:
        plan = tiny_plan()
        cases = [copy.deepcopy(plan["cases"][1])]
        self.assertEqual(bench._validate_requested_cases(plan, cases), cases)

    def test_unknown_modified_duplicate_and_empty_cases_rejected(self) -> None:
        plan = tiny_plan()
        modified = {**plan["cases"][0], "m": 99}
        unknown = {**plan["cases"][0], "id": "unknown"}
        for cases in ([], [modified], [unknown], [plan["cases"][0]] * 2, ["cal"]):
            with self.subTest(cases=cases), self.assertRaises(LabError):
                bench._validate_requested_cases(plan, cases)

    def test_invalid_plan_rejected_before_torch_import(self) -> None:
        plan = tiny_plan()
        plan["timing"]["repeats"] = 1
        with patch.object(bench, "_load_torch") as loader:
            with self.assertRaisesRegex(LabError, "timing.repeats"):
                bench.benchmark_cases(plan, plan["cases"])
            loader.assert_not_called()

    def test_oversized_launch_budget_rejected(self) -> None:
        plan = tiny_plan()
        plan["timing"] = {"warmup": 100, "iterations": 500, "repeats": 30}
        for index in range(4, 11):
            plan["cases"].append({"id": f"candidate_{index}", "m": index + 1, "n": 8, "k": 8, "split": "candidate"})
        validate_plan(plan)
        with self.assertRaisesRegex(LabError, "per-session safety bound"):
            bench._validate_work_budget(plan, plan["cases"])

    def test_typical_plan_has_bounded_work(self) -> None:
        plan = tiny_plan()
        self.assertIsNone(bench._validate_work_budget(plan, plan["cases"]))

    def test_peak_memory_estimate_accounts_for_reference_and_extra_output(self) -> None:
        case = {"id": "shape", "m": 8, "n": 16, "k": 32}
        normal = bench._estimated_peak_bytes(case, "bfloat16")
        extra = bench._estimated_peak_bytes(case, "bfloat16", extra_output=True)
        self.assertEqual(extra - normal, 2 * 8 * 16)
        self.assertGreater(normal, bench.ideal_bytes(case, "bfloat16") + bench.WORKSPACE_ALLOWANCE_BYTES)

    def test_memory_guard_rejects_before_allocation(self) -> None:
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                mem_get_info=lambda _: (128 * 1024**2, 1024**3),
                memory_reserved=lambda _: 0,
                memory_allocated=lambda _: 0,
            )
        )
        with self.assertRaisesRegex(LabError, "safety limit"):
            bench._check_memory(torch, "cuda:0", tiny_plan()["cases"][0], "bfloat16")

    def test_memory_guard_includes_reusable_allocator_cache(self) -> None:
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                mem_get_info=lambda _: (256 * 1024**2, 1024**3),
                memory_reserved=lambda _: 512 * 1024**2,
                memory_allocated=lambda _: 128 * 1024**2,
            )
        )
        result = bench._check_memory(torch, "cuda:0", tiny_plan()["cases"][0], "bfloat16")
        self.assertEqual(result["available_bytes_before_case"], 640 * 1024**2)
        self.assertTrue(result["estimate_is_not_measured_peak"])

    def test_deadline_check_uses_monotonic_clock(self) -> None:
        with patch.object(bench.time, "monotonic", return_value=100.0):
            self.assertIsNone(bench._check_deadline(101.0))
            with self.assertRaisesRegex(LabError, "wall-time budget"):
                bench._check_deadline(100.0)

    def test_command_deadline_can_shorten_but_not_extend_local_budget(self) -> None:
        plan = tiny_plan()
        with patch.object(bench.time, "monotonic", return_value=100.0):
            self.assertEqual(bench._session_deadline(plan), 130.0)
            self.assertEqual(bench._session_deadline(plan, 115.0), 115.0)
            self.assertEqual(bench._session_deadline(plan, 1000.0), 130.0)
            self.assertEqual(bench._session_deadline(plan, 90.0), 90.0)
        for value in (float("nan"), float("inf"), True, "130"):
            with self.subTest(value=value), self.assertRaises(LabError):
                bench._session_deadline(plan, value)

    def test_expired_command_budget_fails_before_cuda_import(self) -> None:
        plan = tiny_plan()
        with patch.object(bench.time, "monotonic", return_value=100.0), patch.object(bench, "_load_torch") as loader:
            with self.assertRaisesRegex(LabError, "wall-time budget"):
                bench.benchmark_cases(plan, plan["cases"], deadline_monotonic=99.0)
            with self.assertRaisesRegex(LabError, "wall-time budget"):
                bench.benchmark_batching(plan, deadline_monotonic=99.0)
            loader.assert_not_called()

    def test_failed_run_keeps_prior_successful_rows(self) -> None:
        rows = [{"case_id": "completed"}]
        error = bench.BenchmarkRunError("stopped", completed_rows=rows, failed_case_id="next", environment={"environment_id": "e"})
        rows.append({"case_id": "later"})
        self.assertEqual(error.completed_rows, [{"case_id": "completed"}])
        self.assertEqual(error.failed_case_id, "next")
        self.assertEqual(error.environment["environment_id"], "e")

    def test_profile_refuses_to_overwrite_existing_trace(self) -> None:
        plan = tiny_plan()
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "torch_trace.json"
            trace.write_text("existing evidence", encoding="utf-8")
            with patch.object(bench, "_load_torch") as loader:
                with self.assertRaisesRegex(LabError, "Refusing to overwrite"):
                    bench.profile_case(plan, plan["cases"][0], temporary, torch_trace=True)
                loader.assert_not_called()
            self.assertEqual(trace.read_text(encoding="utf-8"), "existing evidence")


class TestPrecisionPolicy(unittest.TestCase):
    def test_older_precision_is_restored_after_normal_exit(self) -> None:
        flags = _LegacyMatmul()
        with bench._precision_policy(_math_only_torch(flags)) as policy:
            self.assertFalse(flags.allow_tf32)
            self.assertFalse(flags.allow_bf16_reduced_precision_reduction)
            self.assertEqual(policy["tf32_control_api"], "allow_tf32")
            self.assertEqual(policy["allow_fp16_accumulation"], "unavailable")
        self.assertTrue(flags.allow_tf32)
        self.assertTrue(flags.allow_bf16_reduced_precision_reduction)

    def test_modern_precision_never_touches_older_api(self) -> None:
        flags = _ModernMatmul()
        with bench._precision_policy(_math_only_torch(flags)) as policy:
            self.assertEqual(flags.fp32_precision, "ieee")
            self.assertFalse(flags.allow_fp16_accumulation)
            self.assertEqual(policy["tf32_control_api"], "fp32_precision")
            self.assertNotIn("allow_tf32", policy)
        self.assertEqual(flags.fp32_precision, "tf32")
        self.assertTrue(flags.allow_fp16_accumulation)

    def test_precision_restored_when_operation_raises(self) -> None:
        flags = _ModernMatmul()
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with bench._precision_policy(_math_only_torch(flags)):
                raise RuntimeError("injected failure")
        self.assertEqual(flags.fp32_precision, "tf32")
        self.assertTrue(flags.allow_fp16_reduced_precision_reduction)
        self.assertTrue(flags.allow_bf16_reduced_precision_reduction)

    def test_case_seeds_are_deterministic_and_distinct(self) -> None:
        self.assertEqual(bench._case_seed(42, "case"), bench._case_seed(42, "case"))
        self.assertNotEqual(bench._case_seed(42, "case"), bench._case_seed(42, "other"))
        self.assertGreaterEqual(bench._case_seed(-42, "case"), 0)
        self.assertLess(bench._case_seed(-42, "case"), 2**63)

    def test_correctness_thresholds_are_explicit_by_dtype(self) -> None:
        bf16 = bench._correctness_limits("bfloat16")
        fp16 = bench._correctness_limits("float16")
        fp32 = bench._correctness_limits("float32")
        self.assertGreater(bf16["relative_l2"], fp16["relative_l2"])
        self.assertGreater(fp16["relative_l2"], fp32["relative_l2"])
        bf16["relative_l2"] = 99
        self.assertLess(bench._correctness_limits("bfloat16")["relative_l2"], 1)


class TestStableMetadata(unittest.TestCase):
    def _metadata(self, plan: dict, timestamp: str) -> dict:
        torch = types.SimpleNamespace(__version__="test-version", version=types.SimpleNamespace(cuda="12.8"))
        gpu = {"logical_device_index": 0, "name": "metadata-fixture", "total_memory_bytes": 8 * 1024**3}
        with patch.object(bench, "_driver_versions", return_value=["570.1"]), patch.object(bench, "_gpu_metadata", return_value=gpu), patch.object(bench, "utc_now", return_value=timestamp):
            return bench._collect_environment(torch, plan, "cuda:0", {"fp32_precision": "ieee"})

    def test_timestamp_does_not_change_environment_identity(self) -> None:
        plan = tiny_plan()
        first = self._metadata(plan, "first")
        second = self._metadata(plan, "second")
        self.assertNotEqual(first["captured_at_utc"], second["captured_at_utc"])
        self.assertEqual(first["environment_id"], second["environment_id"])
        self.assertEqual(first["environment_id"], digest(first["identity"]))
        self.assertEqual(first["plan_hash"], digest(plan))
        self.assertNotIn("captured_at_utc", first["identity"])

    def test_dtype_and_timing_change_environment_identity(self) -> None:
        plan = tiny_plan()
        first = self._metadata(plan, "same")
        changed = copy.deepcopy(plan)
        changed["dtype"] = "float16"
        self.assertNotEqual(first["environment_id"], self._metadata(changed, "same")["environment_id"])
        changed = copy.deepcopy(plan)
        changed["timing"]["iterations"] += 1
        self.assertNotEqual(first["environment_id"], self._metadata(changed, "same")["environment_id"])

    def test_new_plan_hash_does_not_pretend_hardware_changed(self) -> None:
        plan = tiny_plan()
        first = self._metadata(plan, "same")
        changed = copy.deepcopy(plan)
        changed["seed"] += 1
        second = self._metadata(changed, "same")
        self.assertNotEqual(first["plan_hash"], second["plan_hash"])
        self.assertEqual(first["environment_id"], second["environment_id"])

    def test_metadata_is_json_serializable(self) -> None:
        result = self._metadata(tiny_plan(), "timestamp")
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["identity"]["execution"]["mode"], "eager")
        self.assertIn("host_submission_gaps", result["identity"]["execution"]["timer_scope"])
        self.assertEqual(result["measurement_kind"], "cuda_measured")
        self.assertIn("weight_NK", result["identity"]["execution"]["layout"])
        self.assertIn("transposed_weight_view_KN", result["identity"]["execution"]["layout"])
        self.assertIn("W.t() prepared before timing", result["identity"]["execution"]["operation"])

    def test_driver_probe_only_retains_versions(self) -> None:
        with patch.object(bench, "_command_version", return_value="570.1\n570.1\n571.2\nunexpected-name\n"):
            self.assertEqual(bench._driver_versions(), ["570.1", "571.2"])


if __name__ == "__main__":
    unittest.main()
