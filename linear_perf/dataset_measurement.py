"""Collect an ordinary offline latency dataset before fitting any model.

The collector does not import the model-selection code and does not impose a
model-freeze-before-measurement rule. Split labels are retained for the later
training/evaluation boundary. GPU execution remains explicit: CPU simulation
exercises software only, and is never a substitute for CUDA measurements.

Timing is a CUDA Graph replay interval divided by its recorded GEMM count.
Rotating buffers are a second, separately labeled working-set contract, not a
cache flush. Torch profiler captures run after all primary timing and never
provide training labels or predictor features.
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.util
import json
import math
import os
import platform
import random
import re
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from . import benchmark
from .common import DTYPE_BYTES, GRAPH_TIMER_SCOPE, LabError, digest, flops, ideal_bytes, read_json, read_jsonl, utc_now, write_json
from .measurement import TelemetryRecorder


DATASET_SCHEMA_VERSION = 2
RUNNER_VERSION = "dataset-cuda-events-1.1.0"
_SPLITS = {"train", "test", "challenge", "stress"}
_DTYPES = {"float16", "bfloat16"}
_CACHE_POLICIES = {"warm_reuse", "rotating_buffers"}
_DEFAULTS = {
    "visits": 3,
    "repeats": 9,
    "warmup_eager": 10,
    "warmup_min_seconds": 0.2,
    "warmup_min_replays": 3,
    "warmup_max_replays": 2048,
    "pilot_operations": 8,
    "pilot_repeats": 3,
    "target_replay_ms": 2.0,
    "min_operations": 1,
    "max_operations": 256,
    "max_wall_seconds": 7200,
    "rotation_enabled": True,
    "rotation_buffers": 4,
    "rotation_l2_multiple": 2.0,
    "l2_cache_bytes": None,
    "profile": True,
    "control_every_cases": 32,
    "telemetry_interval_seconds": 1.0,
    "max_case_bytes": 512 * 1024**2,
    "memory_fraction": 0.65,
    "quality_cv_pct": 10.0,
    "quality_visit_ratio": 1.2,
    "quality_control_ratio": 1.2,
    # Duration-based warmup legitimately submits more GEMMs on faster shapes.
    # The full workload is bounded by wall time, replay counts and memory;
    # an optional caller cap must not impose an unrelated default stop point.
    "max_gemm_calls": None,
    "max_flops": 2_000_000_000_000_000_000,
}
_CONTROLS = (
    {"id": "control_small", "m": 1, "n": 1024, "k": 1024},
    {"id": "control_weight", "m": 1, "n": 4096, "k": 4096},
    {"id": "control_large", "m": 1024, "n": 8192, "k": 2048},
)


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise LabError(f"{label} must be positive and finite")
    return float(value)


def measurement_settings(manifest: dict) -> dict:
    """Return a validated copy of the declared measurement settings."""
    supplied = manifest.get("measurement", {})
    if not isinstance(supplied, dict):
        raise LabError("manifest.measurement must be an object")
    unknown = set(supplied) - set(_DEFAULTS)
    if unknown:
        raise LabError("Unknown measurement settings: " + ", ".join(sorted(unknown)))
    settings = {**_DEFAULTS, **copy.deepcopy(supplied)}
    bounds = {
        "visits": (1, 10), "repeats": (3, 31), "warmup_eager": (1, 100),
        "warmup_min_replays": (1, 100), "warmup_max_replays": (1, 10000),
        "pilot_operations": (1, 1024), "pilot_repeats": (1, 9),
        "min_operations": (1, 1024), "max_operations": (1, 1024),
        "rotation_buffers": (2, 16), "control_every_cases": (1, 1024),
        "max_case_bytes": (1024, 4 * 1024**3),
        "max_flops": (1, 10**19),
    }
    for name, (lower, upper) in bounds.items():
        value = settings[name]
        if type(value) is not int or not lower <= value <= upper:
            raise LabError(f"measurement.{name} must be an integer in [{lower}, {upper}]")
    calls = settings["max_gemm_calls"]
    if calls is not None and (type(calls) is not int or calls < 1):
        raise LabError("measurement.max_gemm_calls must be a positive integer or null")
    for name in ("rotation_enabled", "profile"):
        if type(settings[name]) is not bool:
            raise LabError(f"measurement.{name} must be boolean")
    for name in ("warmup_min_seconds", "target_replay_ms", "max_wall_seconds",
                 "rotation_l2_multiple", "telemetry_interval_seconds", "memory_fraction",
                 "quality_cv_pct", "quality_visit_ratio", "quality_control_ratio"):
        _positive(settings[name], "measurement." + name)
    if not 0.01 <= settings["warmup_min_seconds"] <= 5:
        raise LabError("warmup_min_seconds must be between 0.01 and 5 seconds")
    if not 0.05 <= settings["target_replay_ms"] <= 100:
        raise LabError("target_replay_ms must be between 0.05 and 100 ms")
    if not 1 <= settings["max_wall_seconds"] <= 24 * 3600:
        raise LabError("max_wall_seconds must be between 1 second and 24 hours")
    if not 0 < settings["memory_fraction"] <= 0.8:
        raise LabError("memory_fraction must be in (0, 0.8]")
    if settings["telemetry_interval_seconds"] < 0.5:
        raise LabError("Telemetry snapshots must be at least 0.5 seconds apart")
    if settings["min_operations"] > settings["max_operations"]:
        raise LabError("min_operations exceeds max_operations")
    if settings["warmup_min_replays"] > settings["warmup_max_replays"]:
        raise LabError("Minimum warmup replay count exceeds its maximum")
    if settings["quality_visit_ratio"] < 1 or settings["quality_control_ratio"] < 1:
        raise LabError("Max/min ratio thresholds cannot be below one")
    if settings["rotation_enabled"] and settings["max_operations"] < settings["rotation_buffers"]:
        raise LabError("A rotating graph must be able to visit every buffer")
    if settings["rotation_enabled"] and math.ceil(settings["min_operations"] / settings["rotation_buffers"]) > settings["max_operations"] // settings["rotation_buffers"]:
        raise LabError("Operation bounds must include a complete rotating buffer cycle")
    l2 = settings["l2_cache_bytes"]
    if l2 is not None and (type(l2) is not int or not 1024 <= l2 <= 2 * 1024**3):
        raise LabError("l2_cache_bytes must be an explicit positive byte count or null")
    return settings


def _validated_cases(manifest: dict, case_ids: list[str] | None = None) -> list[dict]:
    if not isinstance(manifest, dict):
        raise LabError("A workload manifest must be an object")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10000:
        raise LabError("A workload manifest requires 1 to 10,000 cases")
    dtypes = manifest.get("dtypes")
    if not isinstance(dtypes, list) or not dtypes or not all(isinstance(dtype, str) for dtype in dtypes) or len(set(dtypes)) != len(dtypes) or not set(dtypes) <= _DTYPES:
        raise LabError("dtypes must contain unique float16 and/or bfloat16 entries")
    if type(manifest.get("seed")) is not int or not 0 <= manifest["seed"] < 2**31:
        raise LabError("A nonnegative integer seed below 2**31 is required")
    settings = measurement_settings(manifest)
    ids, shapes = set(), set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", case["id"]):
            raise LabError("Each case needs a filesystem-safe id")
        if case["id"] in ids:
            raise LabError("Duplicate workload case id")
        ids.add(case["id"])
        for dimension in ("m", "n", "k"):
            if type(case.get(dimension)) is not int or not 1 <= case[dimension] <= 2**20:
                raise LabError("Matrix dimensions must be integers between 1 and 2**20")
        shape = tuple(case[key] for key in ("m", "n", "k"))
        if shape in shapes:
            raise LabError("Duplicate matrix shapes must be merged before collecting a dataset")
        shapes.add(shape)
        if case.get("split") not in _SPLITS:
            raise LabError("Case split must be train, test, challenge, or stress")
        weight = case.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise LabError("Case weight must be a finite nonnegative frequency")
        if any(ideal_bytes(case, dtype) > settings["max_case_bytes"] for dtype in dtypes):
            raise LabError(f"Case {case['id']} exceeds the declared ideal-tensor-byte budget")
    if case_ids is None:
        return copy.deepcopy(cases)
    if not isinstance(case_ids, list) or not case_ids or not all(isinstance(cid, str) for cid in case_ids) or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= ids:
        raise LabError("case_ids must be a nonempty, unique subset of manifest case IDs")
    selected = set(case_ids)
    return copy.deepcopy([case for case in cases if case["id"] in selected])


def repeat_subset(manifest: dict, fraction: float = 0.125, *, seed: int | None = None) -> list[str]:
    """Choose an exact-size, label-free stratified subset for a second session.

    Strata are split, small/large M and regular/irregular dimensions. The target
    size is ceil(fraction * case count); largest-remainder allocation avoids
    accidentally selecting one case per tiny family and overspending the budget.
    """
    cases = _validated_cases(manifest)
    if not 0 < _positive(fraction, "repeat fraction") <= 1:
        raise LabError("repeat fraction must be in (0, 1]")
    if seed is None:
        seed = manifest["seed"]
    if type(seed) is not int:
        raise LabError("Repeat subset seed must be an integer")
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for case in cases:
        strata[(case["split"], case["m"] <= 256, case["n"] % 16 == 0 and case["k"] % 16 == 0)].append(case)
    target = min(len(cases), max(1, math.ceil(len(cases) * fraction)))
    allocations, remainders = {}, []
    for key in sorted(strata):
        quota = target * len(strata[key]) / len(cases)
        allocations[key] = math.floor(quota)
        remainders.append((quota - math.floor(quota), key))
    for _, key in sorted(remainders, key=lambda x: (-x[0], x[1]))[:target - sum(allocations.values())]:
        allocations[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda case: case["id"])
        rng.shuffle(candidates)
        selected.extend(case["id"] for case in candidates[:allocations[key]])
    return sorted(selected)


def make_schedule(cases: list[dict], dtypes: list[str], visits: int, seed: int) -> list[dict]:
    """Shuffle shapes and counterbalance the first dtype within each visit."""
    rng = random.Random(seed)
    schedule = []
    for visit in range(1, visits + 1):
        ordered = sorted(cases, key=lambda case: case["id"])
        rng.shuffle(ordered)
        first_positions = [(i + visit - 1) % len(dtypes) for i in range(len(ordered))]
        # A seeded shuffle gives approximately equal first-position counts even
        # for an odd number of cases. The other precision follows immediately.
        rng.shuffle(first_positions)
        for position, (case, first) in enumerate(zip(ordered, first_positions)):
            order = dtypes[first:] + dtypes[:first]
            policies = ["warm_reuse", "rotating_buffers"]
            rng.shuffle(policies)
            schedule.append({"schedule_index": len(schedule), "visit_index": visit,
                             "case_position": position, "case_id": case["id"],
                             "dtype_order": order, "cache_policy_order": policies})
    return schedule


def choose_operation_count(pilot_per_gemm_ms: float, *, target_replay_ms: float,
                           minimum: int, maximum: int, buffer_sets: int = 1) -> int:
    """Amortize short timing intervals without searching for a favorable score."""
    _positive(pilot_per_gemm_ms, "pilot per-GEMM time")
    _positive(target_replay_ms, "target replay duration")
    if type(minimum) is not int or type(maximum) is not int or type(buffer_sets) is not int or minimum < 1 or maximum < minimum or buffer_sets < 1:
        raise LabError("Invalid operation-count bounds")
    lower = math.ceil(max(minimum, buffer_sets) / buffer_sets) * buffer_sets
    upper = (maximum // buffer_sets) * buffer_sets
    if lower > upper:
        raise LabError("Operation-count bounds cannot represent a full buffer cycle")
    desired = math.ceil(target_replay_ms / pilot_per_gemm_ms)
    rounded = math.ceil(desired / buffer_sets) * buffer_sets
    return max(lower, min(upper, rounded))


def normalize_event_intervals(intervals_ms: list[float], operations: int) -> list[float]:
    if type(operations) is not int or operations < 1 or not intervals_ms:
        raise LabError("Each interval requires its positive captured-operation divisor")
    return [_positive(value, "CUDA event elapsed time") / operations for value in intervals_ms]


class _WorkBudget:
    def __init__(self, settings: dict):
        self.limit_calls, self.limit_flops = settings["max_gemm_calls"], settings["max_flops"]
        self.calls, self.flops = 0, 0

    def charge(self, calls: int, per_call_flops: int) -> None:
        if type(calls) is not int or type(per_call_flops) is not int or calls < 0 or per_call_flops < 0:
            raise LabError("Work accounting requires nonnegative integer calls and FLOPs")
        if ((self.limit_calls is not None and self.calls + calls > self.limit_calls)
                or self.flops + calls * per_call_flops > self.limit_flops):
            raise LabError("Declared dataset GEMM/FLOP budget exhausted; partial records are retained")
        self.calls += calls
        self.flops += calls * per_call_flops

    def summary(self) -> dict:
        return {"gemm_calls_accounted": self.calls, "flops_accounted": self.flops,
                "max_gemm_calls": self.limit_calls, "max_flops": self.limit_flops,
                "scope": "Conservative submitted/captured GEMM work including references and warmup; not hardware counters"}


def _l2_cache_size(torch: Any, selected: Any) -> dict:
    props = torch.cuda.get_device_properties(selected)
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(props, name, None)
        if isinstance(value, int) and value > 0:
            return {"bytes": value, "source": "torch.cuda.device_properties." + name}
    # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38, in bytes (CUDA Driver API cuda.h).
    # Query the already selected current context, rather than guessing how
    # CUDA_VISIBLE_DEVICES maps a logical PyTorch index to a physical device.
    try:
        torch.cuda.current_stream(selected)  # initialize the selected context
        library = ctypes.util.find_library("cuda") or ("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        driver = ctypes.CDLL(library)
        driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuCtxGetDevice.restype = ctypes.c_int
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        current, size = ctypes.c_int(), ctypes.c_int()
        if driver.cuCtxGetDevice(ctypes.byref(current)) == 0 and driver.cuDeviceGetAttribute(ctypes.byref(size), 38, current.value) == 0 and size.value > 0:
            return {"bytes": int(size.value), "source": "CUDA_driver_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE"}
    except (OSError, AttributeError, TypeError, RuntimeError):
        pass
    return {"bytes": None, "source": "unavailable", "reason": "L2 size query unavailable; no model-name lookup or guessed capacity"}


def _environments(base: dict, manifest: dict, settings: dict, *, simulate: bool) -> dict:
    contexts = {}
    for dtype in manifest["dtypes"]:
        for policy in sorted(_CACHE_POLICIES):
            identity = {**copy.deepcopy(base), "dtype": dtype, "cache_policy": policy,
                        "buffer_sets": 1 if policy == "warm_reuse" else settings["rotation_buffers"],
                        "measurement_settings": copy.deepcopy(settings),
                        "timer_scope": GRAPH_TIMER_SCOPE}
            eid = ("simulated:" if simulate else "") + digest(identity)
            contexts[f"{dtype}/{policy}"] = {
                "environment_id": eid, "identity": identity,
                "measurement_kind": "simulated" if simulate else "cuda_measured",
                "sm_count": base["device"].get("multiprocessor_count"),
            }
    return {"schema_version": DATASET_SCHEMA_VERSION, "runner_version": RUNNER_VERSION,
            "captured_at_utc": utc_now(), "measurement_kind": "simulated" if simulate else "cuda_measured",
            "device": copy.deepcopy(base["device"]), "contexts": contexts,
            "limitations": ["Environment identity does not lock clocks, power, thermals, or other processes.",
                            "Rotating buffers do not certify cache eviction or measured HBM traffic.",
                            "An offline dataset is collected before model fitting; split isolation is enforced by the analysis stage."]}


class _CudaBackend:
    def __init__(self, manifest: dict, settings: dict, device: str, deadline: float):
        self.manifest, self.settings, self.device, self.deadline = manifest, settings, device, deadline
        self.stack = ExitStack()
        self.work = _WorkBudget(settings)

    def __enter__(self):
        try:
            self.torch = benchmark._load_torch()
            self.selected = benchmark._require_cuda(self.torch, self.manifest["dtypes"][0], self.device)
            for dtype in self.manifest["dtypes"]:
                benchmark._require_cuda(self.torch, dtype, self.device)
            self.stack.enter_context(self.torch.cuda.device(self.selected))
            policy = self.stack.enter_context(benchmark._precision_policy(self.torch))
            self.stack.enter_context(self.torch.inference_mode())
            self.stack.enter_context(self.torch.autocast(device_type="cuda", enabled=False))
            gpu = benchmark._gpu_metadata(self.torch, self.selected)
            observed_l2 = _l2_cache_size(self.torch, self.selected)
            override = self.settings["l2_cache_bytes"]
            known_l2 = observed_l2["bytes"] is not None
            gpu.update(l2_cache_bytes=observed_l2["bytes"] if known_l2 else override,
                       l2_cache_source=observed_l2["source"] if known_l2 or override is None else "explicit_override_when_query_unavailable",
                       l2_cache_observation=observed_l2, l2_cache_override_requested=override,
                       l2_cache_override_applied=not known_l2 and override is not None)
            if known_l2 and override is not None and override != observed_l2["bytes"]:
                gpu["l2_cache_override_note"] = "Ignored conflicting override because an observed device L2 size is available"
            base = {"runner_version": RUNNER_VERSION, "device": gpu,
                    "python_version": platform.python_version(), "pytorch_version": str(self.torch.__version__),
                    "cuda_runtime_version": str(self.torch.version.cuda),
                    "driver_versions": benchmark._driver_versions(), "math_policy": copy.deepcopy(policy),
                    "execution": "preallocated torch.mm(X, W.T, out=Y), inference mode, autocast disabled"}
            self.environments = _environments(base, self.manifest, self.settings, simulate=False)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _guard(self, case: dict, dtype: str, buffers: int) -> dict:
        free, total = self.torch.cuda.mem_get_info(self.selected)
        reusable = max(0, self.torch.cuda.memory_reserved(self.selected) - self.torch.cuda.memory_allocated(self.selected))
        available = min(int(total), int(free) + int(reusable))
        estimate = buffers * benchmark._estimated_peak_bytes(case, dtype)
        return {"estimated_peak_bytes": estimate, "available_bytes": available,
                "memory_fraction": self.settings["memory_fraction"],
                "passed": estimate <= self.settings["memory_fraction"] * available,
                "estimate_is_not_measured_peak": True}

    def _prepare(self, case: dict, dtype: str, buffers: int):
        benchmark._check_deadline(self.deadline)
        guard = self._guard(case, dtype, buffers)
        if not guard["passed"]:
            raise _MemoryGuardError(guard)
        inputs, references, checks = [], [], []
        for index in range(buffers):
            x, w, y = benchmark._allocate_inputs(self.torch, self.selected, case, dtype, self.manifest["seed"] + index)
            self.work.charge(2, flops(case))
            reference = self.torch.mm(x.float(), w.float())
            self.torch.mm(x, w, out=y)
            checked = benchmark._correctness(self.torch, y, reference, dtype)
            benchmark._assert_correctness(checked, case["id"])
            inputs.append((x, w, y))
            references.append(reference)
            checks.append(checked)
        pointer = [0]

        def operation():
            x, w, y = inputs[pointer[0] % buffers]
            pointer[0] += 1
            self.torch.mm(x, w, out=y)

        return inputs, references, checks, operation, guard

    def _capture(self, operation, case: dict, operations: int):
        self.work.charge(self.settings["warmup_eager"] + operations, flops(case))
        return benchmark._capture_operation_graph(
            self.torch, self.selected, operation,
            {"warmup": self.settings["warmup_eager"], "iterations": operations}, self.deadline)

    def _warm(self, graph, case: dict, operations: int, *, duration: float) -> dict:
        self.torch.cuda.synchronize(self.selected)
        started, count = time.monotonic(), 0
        while count < self.settings["warmup_min_replays"] or time.monotonic() - started < duration:
            benchmark._check_deadline(self.deadline)
            if count >= self.settings["warmup_max_replays"]:
                raise LabError("Graph warmup reached its declared cap before completing its duration")
            self.work.charge(operations, flops(case))
            graph.replay()
            self.torch.cuda.synchronize(self.selected)
            count += 1
        return {"completed_replays": count, "actual_wall_seconds": time.monotonic() - started,
                "requested_min_seconds": duration, "minimum_replays": self.settings["warmup_min_replays"],
                "operations_per_replay": operations, "excluded_from_labels": True,
                "stop_rule": "minimum completed replays and wall duration, with a fixed cap; no quality-score stopping"}

    def _intervals(self, graph, case: dict, operations: int, repeats: int, *, retained: list | None = None) -> list[float]:
        events = benchmark._event_pair(self.torch)
        intervals = retained if retained is not None else []
        for _ in range(repeats):
            benchmark._check_deadline(self.deadline)
            self.work.charge(operations, flops(case))
            # iterations=1 here means one complete graph, not one GEMM.
            intervals.append(benchmark._event_sample(graph.replay, events, 1))
        return intervals

    def measure(self, case: dict, dtype: str, cache_policy: str) -> dict:
        attempt = {"case_id": case["id"], "dtype": dtype, "cache_policy": cache_policy,
                   "stage": "input_allocation_and_correctness", "label_eligible": False}
        self.last_attempt = attempt
        buffers = 1 if cache_policy == "warm_reuse" else self.settings["rotation_buffers"]
        inputs, references, before, operation, guard = self._prepare(case, dtype, buffers)
        pilot_count = max(buffers, math.ceil(self.settings["pilot_operations"] / buffers) * buffers)
        pilot_count = min(pilot_count, (self.settings["max_operations"] // buffers) * buffers)
        graph = self._capture(operation, case, pilot_count)
        pilot_warmup = self._warm(graph, case, pilot_count, duration=0.0)
        attempt.update(stage="pilot_timing", pilot_operations=pilot_count, pilot_event_elapsed_ms=[])
        pilot_intervals = self._intervals(graph, case, pilot_count, self.settings["pilot_repeats"], retained=attempt["pilot_event_elapsed_ms"])
        per_call = statistics.median(normalize_event_intervals(pilot_intervals, pilot_count))
        operations = choose_operation_count(
            per_call, target_replay_ms=self.settings["target_replay_ms"],
            minimum=self.settings["min_operations"], maximum=self.settings["max_operations"], buffer_sets=buffers)
        if operations != pilot_count:
            # Release the pilot graph before constructing its replacement.
            del graph
            graph = self._capture(operation, case, operations)
        for _, _, output in inputs:
            output.fill_(float("nan"))
        attempt.update(stage="final_graph_warmup", captured_gemm_count=operations)
        warmup = self._warm(graph, case, operations, duration=self.settings["warmup_min_seconds"])
        attempt.update(stage="timing", event_elapsed_ms=[])
        intervals = self._intervals(graph, case, operations, self.settings["repeats"], retained=attempt["event_elapsed_ms"])
        after = [benchmark._correctness(self.torch, triple[2], reference, dtype)
                 for triple, reference in zip(inputs, references)]
        attempt.update(stage="post_timing_correctness", correctness_after=after)
        for check in after:
            benchmark._assert_correctness(check, case["id"])
        attempt.update(stage="completed", label_eligible=True)
        return {"buffer_sets": buffers, "captured_gemm_count": operations,
                "event_elapsed_ms": intervals, "samples_ms": normalize_event_intervals(intervals, operations),
                "pilot": {"operations": pilot_count, "event_elapsed_ms": pilot_intervals,
                          "per_gemm_median_ms": per_call, "selected_operations": operations,
                          "target_replay_ms": self.settings["target_replay_ms"],
                          "warmup": pilot_warmup, "excluded_from_labels": True},
                "warmup": warmup, "memory_guard": guard,
                "correctness_before": before, "correctness_after": after,
                "numeric_execution_checked": True,
                "declared_working_set_bytes": buffers * ideal_bytes(case, dtype)}

    def profile(self, case: dict, dtype: str, operations: int, destination: Path) -> dict:
        self.last_attempt = {"case_id": case["id"], "dtype": dtype, "cache_policy": "warm_reuse",
                             "stage": "separate_profiler_capture", "label_eligible": False}
        inputs, references, before, operation, guard = self._prepare(case, dtype, 1)
        graph = self._capture(operation, case, operations)
        self._warm(graph, case, operations, duration=0.0)
        record = {"status": "unavailable", "cache_policy": "warm_reuse", "dtype": dtype,
                  "captured_gemm_count": operations, "excluded_from_labels": True,
                  "excluded_from_predictor_features": True, "hardware_counters": "not_collected",
                  "memory_guard": guard, "correctness_before": before,
                  "profile_scope": "Separate warmed graph capture after all timing labels have been collected"}
        self.work.charge(operations, flops(case))
        try:
            with self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU, self.torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=False, with_stack=False,
            ) as profiler:
                graph.replay()
                self.torch.cuda.synchronize(self.selected)
            destination.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(destination))
            events = read_json(destination).get("traceEvents", [])
            kernels = [event for event in events if isinstance(event, dict) and event.get("ph") == "X"
                       and "kernel" in str(event.get("cat", "")).lower()]
            record.update(status="ok" if kernels else "unavailable",
                          reason=None if kernels else "trace_contains_no_CUDA_kernel_events",
                          trace_path=str(destination.name), cuda_kernel_event_count=len(kernels),
                          kernel_names=sorted({str(event.get("name", "")) for event in kernels}))
        except (RuntimeError, ImportError, AttributeError, OSError, ValueError) as exc:
            record.update(reason="profiler_unavailable", error_type=type(exc).__name__, error=str(exc))
        # An optional profiler error cannot hide a subsequent CUDA or arithmetic
        # failure. The ordinary timing labels are never replaced by trace times.
        self.torch.cuda.synchronize(self.selected)
        after = [benchmark._correctness(self.torch, triple[2], reference, dtype)
                 for triple, reference in zip(inputs, references)]
        for check in after:
            benchmark._assert_correctness(check, case["id"])
        record["correctness_after"] = after
        benchmark._check_deadline(self.deadline)
        return record


class _MemoryGuardError(LabError):
    def __init__(self, guard: dict):
        super().__init__("Estimated tensor/reference/scratch storage exceeds the available-memory budget")
        self.guard = guard


class _SimulatedBackend:
    """Deterministic software fixture. Every artifact explicitly says simulated."""
    def __init__(self, manifest: dict, settings: dict, device: str, deadline: float):
        self.manifest, self.settings, self.deadline = manifest, settings, deadline
        self.counter = 0
        self.work = _WorkBudget(settings)
        gpu = {"name": "synthetic_device_fixture_not_GPU_hardware", "uuid": None,
               "multiprocessor_count": 16,
               "l2_cache_bytes": settings["l2_cache_bytes"] or 16 * 1024**2,
               "l2_cache_source": "synthetic_test_fixture_no_hardware_observation"}
        self.environments = _environments(
            {"runner_version": RUNNER_VERSION, "device": gpu, "python_version": platform.python_version(),
             "pytorch_version": None, "cuda_runtime_version": None, "driver_versions": [],
             "math_policy": "not_executed_CPU_software_fixture"}, manifest, settings, simulate=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def measure(self, case: dict, dtype: str, cache_policy: str) -> dict:
        self.last_attempt = {"case_id": case["id"], "dtype": dtype, "cache_policy": cache_policy,
                             "stage": "synthetic_software_fixture", "label_eligible": False}
        benchmark._check_deadline(self.deadline)
        buffers = 1 if cache_policy == "warm_reuse" else self.settings["rotation_buffers"]
        # Arbitrary test-fixture constants are deliberately not GPU ratings.
        latency = 0.004 + max(flops(case) / 1.4e11, ideal_bytes(case, dtype) / 7.0e8)
        latency *= 1.02 if dtype == "bfloat16" else 1.0
        latency *= 1.3 if cache_policy == "rotating_buffers" else 1.0
        latency *= 1.0 + (0.08 if case["n"] % 16 else 0) + (0.05 if case["k"] % 16 else 0)
        operations = choose_operation_count(latency, target_replay_ms=self.settings["target_replay_ms"],
                                            minimum=self.settings["min_operations"],
                                            maximum=self.settings["max_operations"], buffer_sets=buffers)
        token = {"case_id": case["id"], "dtype": dtype, "policy": cache_policy,
                 "seed": self.manifest["seed"], "fixture_visit": self.counter}
        rng = random.Random(int(digest(token)[:16], 16))
        self.counter += 1
        samples = [latency * (1 + rng.uniform(-0.01, 0.01)) for _ in range(self.settings["repeats"])]
        return {"buffer_sets": buffers, "captured_gemm_count": operations,
                "event_elapsed_ms": [value * operations for value in samples], "samples_ms": samples,
                "pilot": {"status": "synthetic_fixture", "excluded_from_labels": True,
                          "selected_operations": operations},
                "warmup": {"status": "not_executed_simulation"},
                "memory_guard": {"status": "not_executed_simulation"},
                "correctness_before": [], "correctness_after": [], "numeric_execution_checked": False,
                "declared_working_set_bytes": buffers * ideal_bytes(case, dtype),
                "simulation_notice": "Generated software fixture; no CUDA event, graph, memory traffic, or arithmetic was measured"}

    def profile(self, case: dict, dtype: str, operations: int, destination: Path) -> dict:
        return {"status": "unavailable", "reason": "simulation_no_GPU_profile", "dtype": dtype,
                "cache_policy": "warm_reuse", "captured_gemm_count": operations,
                "excluded_from_labels": True, "excluded_from_predictor_features": True,
                "hardware_counters": "not_collected", "kernel_names": []}


def inspect_environment(manifest: dict, *, simulate: bool = False, device: str = "cuda:0") -> dict:
    _validated_cases(manifest)
    settings = measurement_settings(manifest)
    backend_type = _SimulatedBackend if simulate else _CudaBackend
    with backend_type(manifest, settings, device, time.monotonic() + settings["max_wall_seconds"]) as backend:
        return copy.deepcopy(backend.environments)


def rotation_eligibility(case: dict, dtype: str, environments: dict, settings: dict) -> dict:
    weight_bytes = DTYPE_BYTES[dtype] * case["n"] * case["k"]
    l2 = environments["device"].get("l2_cache_bytes")
    enabled = settings["rotation_enabled"]
    reason = "eligible"
    if not enabled:
        reason = "disabled_in_manifest"
    elif not isinstance(l2, int) or l2 <= 0:
        reason = "L2_size_unavailable_set_explicit_l2_cache_bytes_if_known"
    elif weight_bytes > settings["rotation_l2_multiple"] * l2:
        reason = "weight_exceeds_declared_L2_multiple"
    return {"case_id": case["id"], "dtype": dtype, "eligible": reason == "eligible", "reason": reason,
            "weight_bytes": weight_bytes, "l2_cache_bytes": l2,
            "l2_cache_source": environments["device"].get("l2_cache_source"),
            "l2_multiple": settings["rotation_l2_multiple"], "buffer_sets": settings["rotation_buffers"],
            "interpretation": "Working-set sensitivity only; even four buffer sets do not guarantee cache eviction"}


def _append(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _cv_pct(values: list[float]) -> float:
    return 100 * statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def _aggregate(visits: list[dict], settings: dict) -> dict:
    visits = sorted(visits, key=lambda row: row["visit_index"])
    first = visits[0]
    values = [row["median_ms"] for row in visits]
    ratio = max(values) / min(values)
    cv_max = max(row["quality"]["cv_pct"] for row in visits)
    return {key: copy.deepcopy(first[key]) for key in (
        "schema_version", "case_id", "m", "n", "k", "split", "weight", "dtype", "cache_policy",
        "buffer_sets", "measurement_kind", "environment_id", "manifest_hash", "session_id", "sm_count", "gpu_name",
        "timer_scope", "declared_working_set_bytes")} | {
        "status": "ok", "sample_unit": "visit_median", "aggregation": "median_of_visit_medians",
        "median_ms": statistics.median(values), "samples_ms": values,
        "raw_samples_ms_by_visit": [row["samples_ms"] for row in visits],
        "event_elapsed_ms_by_visit": [row["event_elapsed_ms"] for row in visits],
        "captured_operations_by_visit": [row["captured_gemm_count"] for row in visits],
        "raw_visit_ids": [row["visit_id"] for row in visits], "completed_visits": len(visits),
        "flops": flops(first), "ideal_bytes": ideal_bytes(first, first["dtype"]),
        "quality": {"max_within_visit_cv_pct": cv_max, "max_over_min_visit_median": ratio,
                    "passed": cv_max <= settings["quality_cv_pct"] and ratio <= settings["quality_visit_ratio"]},
        "numeric_execution_checked": all(row["numeric_execution_checked"] for row in visits),
        "correctness_after_by_visit": [row["correctness_after"] for row in visits],
        "cache_interpretation": "Repeated same buffers" if first["cache_policy"] == "warm_reuse" else "Fixed rotating preallocated buffers; not certified cold cache",
    }


def _quality_summary(rows: list[dict], controls: list[dict], settings: dict) -> dict:
    buckets = defaultdict(list)
    for control in controls:
        buckets[(control["case_id"], control["dtype"])].append(control["median_ms"])
    control_records = [{"case_id": key[0], "dtype": key[1], "visits": len(values),
                        "max_over_min": max(values) / min(values),
                        "passed": max(values) / min(values) <= settings["quality_control_ratio"]}
                       for key, values in sorted(buckets.items())]
    return {"passed": bool(rows) and all(row["quality"]["passed"] for row in rows) and all(row["passed"] for row in control_records),
            "flagged_rows": [{"case_id": row["case_id"], "dtype": row["dtype"], "cache_policy": row["cache_policy"],
                              **row["quality"]} for row in rows if not row["quality"]["passed"]],
            "controls": control_records,
            "thresholds": {key: settings[key] for key in ("quality_cv_pct", "quality_visit_ratio", "quality_control_ratio")},
            "interpretation": "Descriptive declared thresholds; flags are retained, not sample deletion or certification"}


def collect_dataset(manifest: dict, output: str | Path, *, session_id: str,
                    simulate: bool = False, device: str = "cuda:0", progress=print,
                    case_ids: list[str] | None = None, checkpoint=None) -> dict:
    """Collect every selected split, then profile; no model is fitted here.

    A fresh directory is required. Exceptions during collection produce a
    failed status while retaining completed visits and aggregate rows. A repeat
    subset keeps the full original manifest and records its selected case IDs.
    Optional checkpoints run synchronously between complete visits and after
    timing, never inside event intervals. A checkpoint error stops further GPU
    work and retains the measured records; there is no automatic retry.
    """
    cases = _validated_cases(manifest, case_ids)
    settings = measurement_settings(manifest)
    if progress is None:
        progress = lambda message: None
    elif not callable(progress):
        raise LabError("progress must be a callable or null")
    if checkpoint is not None and not callable(checkpoint):
        raise LabError("checkpoint must be a callable or null")
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session_id):
        raise LabError("session_id must be 1 to 64 alphanumeric/dash/underscore characters")
    out = Path(output).resolve()
    if out.exists():
        raise LabError("Dataset output already exists; use a new directory to preserve the previous attempt")
    out.mkdir(parents=True)
    manifest_hash = digest(manifest)
    started = time.monotonic()
    deadline = started + settings["max_wall_seconds"]
    kind = "simulated" if simulate else "cuda_measured"
    selected_ids = [case["id"] for case in cases]
    schedule = make_schedule(cases, manifest["dtypes"], settings["visits"], manifest["seed"])
    status = {"schema_version": DATASET_SCHEMA_VERSION, "status": "started", "stage": "preflight",
              "session_id": session_id, "measurement_kind": kind, "manifest_hash": manifest_hash,
              "started_at_utc": utc_now(), "selected_case_ids": selected_ids,
              "repeat_subset": len(cases) != len(manifest["cases"]),
              "workflow": "collect_all_splits_then_fit_on_training_data_only",
              "model_fitting_performed": False, "heldout_measurement_requires_frozen_model": False,
              "completed_rows": 0, "completed_visits": 0, "samples_removed": 0, "automatic_retries": 0}
    write_json(out / "manifest.json", manifest, overwrite=False)
    write_json(out / "measurement_settings.json", settings, overwrite=False)
    write_json(out / "schedule.json", schedule, overwrite=False)
    write_json(out / "status.json", status)
    rows, controls, profiles, skipped = [], [], [], []
    accumulated: dict[tuple, list[dict]] = defaultdict(list)
    environments = {"measurement_kind": kind, "contexts": {}, "device": {}}
    telemetry = None
    backend = None

    def update(stage: str, **extra):
        status.update(stage=stage, elapsed_seconds=time.monotonic() - started,
                      completed_rows=len(rows), **extra)
        write_json(out / "status.json", status)

    def save_checkpoint(event, *, visit_index=None):
        if checkpoint is None:
            return
        # Every JSONL append is already flushed/fsynced. This call runs in the
        # same thread as collection while no timed measurement is in flight.
        record = {"checkpoint_event": event, "dataset_directory": str(out),
                  "session_id": session_id, "visit_index": visit_index,
                  "recorded_status": copy.deepcopy(status)}
        try:
            checkpoint(record)
        except BaseException as exc:
            status["checkpoint_failure"] = {"event": event, "visit_index": visit_index,
                                             "error_type": type(exc).__name__, "error": str(exc)}
            write_json(out / "status.json", status)
            raise

    def make_visit(case: dict, dtype: str, policy: str, *, visit_index: int, schedule_index: int, control: bool = False):
        began = utc_now()
        status["active_variant"] = {"case_id": case["id"], "dtype": dtype, "cache_policy": policy,
                                    "visit_index": visit_index, "schedule_index": schedule_index,
                                    "diagnostic_control": control}
        raw = backend.measure(case, dtype, policy)
        samples = normalize_event_intervals(raw["event_elapsed_ms"], raw["captured_gemm_count"])
        if len(samples) != settings["repeats"] or len(raw["samples_ms"]) != len(samples) or any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(samples, raw["samples_ms"])):
            raise LabError("Recorded timing samples disagree with their operation-count normalization")
        context = environments["contexts"][f"{dtype}/{policy}"]
        record = {**raw, "schema_version": DATASET_SCHEMA_VERSION, "case_id": case["id"],
                  **{key: case[key] for key in ("m", "n", "k")},
                  "split": "diagnostic" if control else case["split"], "weight": 0.0 if control else float(case.get("weight", 1.0)),
                  "dtype": dtype, "cache_policy": policy, "measurement_kind": kind,
                  "session_id": session_id, "environment_id": context["environment_id"],
                  "manifest_hash": manifest_hash, "visit_index": visit_index, "schedule_index": schedule_index,
                  "visit_id": f"{'control' if control else 'case'}:{case['id']}:{dtype}:{policy}:{visit_index}",
                  "sample_unit": "graph_replay_mean", "median_ms": statistics.median(samples),
                  "timer_scope": GRAPH_TIMER_SCOPE,
                  "sm_count": environments["device"].get("multiprocessor_count"),
                  "gpu_name": environments["device"].get("name"),
                  "started_at_utc": began, "finished_at_utc": utc_now(),
                  "quality": {"cv_pct": _cv_pct(samples), "passed": _cv_pct(samples) <= settings["quality_cv_pct"]}}
        return record

    try:
        with ExitStack() as files:
            handles = {name: files.enter_context((out / f"{name}.jsonl").open("x", encoding="utf-8"))
                       for name in ("raw_visits", "measurements", "controls", "profiles", "skipped_variants")}
            backend_type = _SimulatedBackend if simulate else _CudaBackend
            backend = files.enter_context(backend_type(manifest, settings, device, deadline))
            environments = backend.environments
            write_json(out / "environments.json", environments, overwrite=False)
            telemetry = TelemetryRecorder(out / "telemetry.jsonl", interval_seconds=settings["telemetry_interval_seconds"],
                                          simulate=simulate, device=device, deadline=deadline)
            files.callback(telemetry.close)
            telemetry.sample("start", force=True)
            eligibility = {(case["id"], dtype): rotation_eligibility(case, dtype, environments, settings)
                           for case in cases for dtype in manifest["dtypes"]}
            write_json(out / "rotation_eligibility.json", list(eligibility.values()), overwrite=False)
            for item in eligibility.values():
                if not item["eligible"]:
                    record = {**item, "cache_policy": "rotating_buffers", "stage": "eligibility"}
                    skipped.append(record)
                    _append(handles["skipped_variants"], record)
            lookup = {case["id"]: case for case in cases}
            blocked_variants = set()
            control_round = 0

            def collect_controls(boundary: str, schedule_index: int):
                nonlocal control_round
                control_round += 1
                dtype_order = list(manifest["dtypes"])
                if control_round % 2 == 0:
                    dtype_order.reverse()
                for control_case in _CONTROLS:
                    for dtype in dtype_order:
                        raw = make_visit(control_case, dtype, "warm_reuse", visit_index=control_round,
                                         schedule_index=schedule_index, control=True)
                        raw["boundary"] = boundary
                        controls.append(raw)
                        _append(handles["controls"], raw)

            update("collecting_all_splits")
            progress(f"[{session_id}] collecting {len(cases)} shapes across {len(manifest['dtypes'])} precision(s); labels precede fitting")
            for task in schedule:
                benchmark._check_deadline(deadline)
                if task["case_position"] == 0:
                    collect_controls("visit_start", task["schedule_index"])
                case = lookup[task["case_id"]]
                for policy in task["cache_policy_order"]:
                    for dtype in task["dtype_order"]:
                        key = (case["id"], dtype, policy)
                        if key in blocked_variants or (policy == "rotating_buffers" and not eligibility[(case["id"], dtype)]["eligible"]):
                            continue
                        telemetry.sample("before_case", task["schedule_index"])
                        try:
                            raw = make_visit(case, dtype, policy, visit_index=task["visit_index"], schedule_index=task["schedule_index"])
                        except _MemoryGuardError as exc:
                            if policy != "rotating_buffers":
                                raise
                            record = {"case_id": case["id"], "dtype": dtype, "cache_policy": policy,
                                      "stage": "memory_guard", "reason": str(exc), "memory_guard": exc.guard,
                                      "incomplete_visits_retained": len(accumulated[key])}
                            blocked_variants.add(key)
                            skipped.append(record)
                            _append(handles["skipped_variants"], record)
                            continue
                        _append(handles["raw_visits"], raw)
                        accumulated[key].append(raw)
                        status["completed_visits"] += 1
                        if len(accumulated[key]) == settings["visits"]:
                            aggregate = _aggregate(accumulated[key], settings)
                            rows.append(aggregate)
                            _append(handles["measurements"], aggregate)
                position = task["case_position"] + 1
                if position % settings["control_every_cases"] == 0 or position == len(cases):
                    collect_controls("visit_end" if position == len(cases) else "periodic", task["schedule_index"])
                update("collecting_all_splits", last_case_id=case["id"], last_visit_index=task["visit_index"])
                if position % 25 == 0 or position == len(cases):
                    progress(f"[{session_id}] visit {task['visit_index']}/{settings['visits']}: {position}/{len(cases)} shapes")
                if position == len(cases):
                    save_checkpoint("timing_visit_completed", visit_index=task["visit_index"])
            update("timing_collection_completed", timing_completed_at_utc=utc_now())
            save_checkpoint("timing_completed", visit_index=settings["visits"])
            if settings["profile"]:
                update("separate_profiler_captures")
                warm_rows = {(row["case_id"], row["dtype"]): row for row in rows if row["cache_policy"] == "warm_reuse"}
                expected_profiles = len(cases) * len(manifest["dtypes"])
                progress(f"[{session_id}] profiling: 0/{expected_profiles} separate captures; timing records already retained")
                for case in cases:
                    for dtype in manifest["dtypes"]:
                        benchmark._check_deadline(deadline)
                        row = warm_rows[(case["id"], dtype)]
                        destination = out / "traces" / f"{case['id']}_{dtype}.json"
                        record = backend.profile(case, dtype, row["captured_operations_by_visit"][0], destination)
                        record.update(case_id=case["id"], measurement_kind=kind,
                                      environment_id=row["environment_id"], captured_at_utc=utc_now())
                        if destination.exists():
                            record["trace_path"] = destination.relative_to(out).as_posix()
                        profiles.append(record)
                        _append(handles["profiles"], record)
                        if len(profiles) == 1 or len(profiles) % 25 == 0 or len(profiles) == expected_profiles:
                            progress(f"[{session_id}] profiling: {len(profiles)}/{expected_profiles} separate captures")
                update("profiling_completed")
            telemetry.sample("finish")
            status["status"] = "simulated_complete" if simulate else "completed"
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, _MemoryGuardError):
            status["memory_guard"] = exc.guard
        write_json(out / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                   "stage": status["stage"], "active_variant": status.get("active_variant"),
                   "latest_backend_attempt": getattr(backend, "last_attempt", None),
                   "failed_attempt_is_not_a_training_label": True})
        progress(f"[{session_id}] {status['status']}: {exc}; collected records remain in {out}")
    finally:
        if not (out / "environments.json").exists():
            write_json(out / "environments.json", environments, overwrite=False)
        quality = _quality_summary(rows, controls, settings)
        write_json(out / "quality.json", quality)
        status.update(completed_rows=len(rows), finished_at_utc=utc_now(), elapsed_seconds=time.monotonic() - started,
                      quality=quality, profile_captures=len(profiles),
                      profiles_with_kernel_events=sum(record.get("status") == "ok" for record in profiles),
                      skipped_variants=len(skipped),
                      work_budget=backend.work.summary() if backend is not None else None,
                      telemetry=telemetry.summary() if telemetry is not None else None,
                      hardware_measurements_collected=kind == "cuda_measured" and bool(rows),
                      predictive_accuracy_established=False)
        write_json(out / "status.json", status)
    if status["status"] in {"completed", "simulated_complete"}:
        validation = validate_collected_dataset(out)
        status["validation"] = validation
        if not validation["passed"]:
            status.update(status="failed", stage="record_consistency_validation",
                          error_type="LabError", error=validation["error"])
        write_json(out / "status.json", status)
    if status["status"] in {"completed", "simulated_complete"}:
        # Validation has finished and the CUDA backend is closed. A subsequent
        # backup failure does not rewrite this completed dataset as incomplete,
        # but propagates to the surrounding experiment and stops follow-ups.
        save_checkpoint("collection_completed", visit_index=settings["visits"])
    return {**status, "output": str(out), "measurements_path": str(out / "measurements.jsonl"),
            "raw_visits_path": str(out / "raw_visits.jsonl"), "environments_path": str(out / "environments.json"),
            "required_followup_gpu_validation": kind == "simulated"}


def validate_collected_dataset(path: str | Path) -> dict:
    """Recompute raw-to-aggregate consistency on a CPU, without fitting models."""
    root = Path(path)
    checks = 0

    def check(condition: bool, message: str):
        nonlocal checks
        checks += 1
        if not condition:
            raise LabError(message)

    def close(actual, expected, message):
        check(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual)
              and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), message)

    try:
        manifest, status = read_json(root / "manifest.json"), read_json(root / "status.json")
        settings = measurement_settings(manifest)
        cases = _validated_cases(manifest, status.get("selected_case_ids"))
        case_map = {case["id"]: case for case in cases}
        check(status.get("status") in {"completed", "simulated_complete"}, "Dataset collection is incomplete")
        check(status.get("manifest_hash") == digest(manifest), "Manifest identity differs from collection status")
        check(status.get("model_fitting_performed") is False, "Collector status incorrectly includes model fitting")
        check(read_json(root / "measurement_settings.json") == settings, "Stored measurement settings differ from manifest")
        kind = status.get("measurement_kind")
        check(kind in {"cuda_measured", "simulated"}, "Dataset measurement kind is missing")
        check(status["status"] == ("simulated_complete" if kind == "simulated" else "completed"), "Dataset completion label disagrees with measurement kind")
        environments = read_json(root / "environments.json")
        check(environments.get("measurement_kind") == kind, "Environment measurement kind mismatch")
        contexts = environments.get("contexts", {})
        check(set(contexts) == {f"{dtype}/{policy}" for dtype in manifest["dtypes"] for policy in _CACHE_POLICIES}, "Environment contexts are incomplete")
        for key, context in contexts.items():
            expected = ("simulated:" if kind == "simulated" else "") + digest(context["identity"])
            check(context.get("environment_id") == expected, "Environment identity mismatch: " + key)
            identity = context["identity"]
            check(identity["device"] == environments["device"] and identity["measurement_settings"] == settings,
                  "Environment hardware/settings differ between contexts")
            check(key == identity["dtype"] + "/" + identity["cache_policy"] and identity["timer_scope"] == GRAPH_TIMER_SCOPE,
                  "Environment context precision/cache/timer mismatch")
            check(context.get("measurement_kind") == kind, "Environment context kind mismatch")
        schedule = make_schedule(cases, manifest["dtypes"], settings["visits"], manifest["seed"])
        check(read_json(root / "schedule.json") == schedule, "Recorded schedule differs from declared seeded ordering")
        tasks = {(task["case_id"], task["visit_index"]): task for task in schedule}

        def validate_visit(visit, case, *, control=False):
            cid = case["id"]
            for field in ("m", "n", "k", "split"):
                expected = "diagnostic" if field == "split" and control else case[field]
                check(visit.get(field) == expected, "Raw visit changed declared case field: " + field)
            close(visit.get("weight"), 0.0 if control else float(case.get("weight", 1.0)), "Raw visit changed declared case weight")
            key = (cid, visit.get("dtype"), visit.get("cache_policy"))
            check(key[1] in manifest["dtypes"] and key[2] in _CACHE_POLICIES, "Undeclared precision or cache contract")
            check(not control or key[2] == "warm_reuse", "Controls must use the primary cache contract")
            context = contexts[f"{key[1]}/{key[2]}"]
            check(visit.get("environment_id") == context["environment_id"], "Raw visit environment mismatch")
            check(visit.get("measurement_kind") == kind and visit.get("manifest_hash") == digest(manifest), "Raw visit provenance mismatch")
            check(visit.get("session_id") == status["session_id"], "Raw visit session mismatch")
            check(visit.get("sm_count") == environments["device"].get("multiprocessor_count") and visit.get("gpu_name") == environments["device"].get("name"), "Raw visit hardware metadata mismatch")
            check(visit.get("timer_scope") == GRAPH_TIMER_SCOPE and visit.get("sample_unit") == "graph_replay_mean", "Raw visit timing scope mismatch")
            check(visit.get("visit_id") == f"{'control' if control else 'case'}:{cid}:{key[1]}:{key[2]}:{visit['visit_index']}", "Raw visit ID disagrees with its fields")
            operations, buffers = visit.get("captured_gemm_count"), visit.get("buffer_sets")
            check(type(operations) is int and settings["min_operations"] <= operations <= settings["max_operations"], "Invalid captured operation count")
            check(buffers == (1 if key[2] == "warm_reuse" else settings["rotation_buffers"]), "Buffer count differs from cache contract")
            check(operations >= buffers and operations % buffers == 0, "Graph does not cover equal complete buffer cycles")
            check(visit.get("declared_working_set_bytes") == buffers * ideal_bytes(case, key[1]), "Working-set byte calculation mismatch")
            samples = normalize_event_intervals(visit.get("event_elapsed_ms", []), operations)
            check(len(samples) == settings["repeats"] == len(visit.get("samples_ms", [])), "Raw repeat count differs from settings")
            for observed, expected in zip(visit["samples_ms"], samples):
                close(observed, expected, "Event duration was not divided by its recorded operation count")
            close(visit.get("median_ms"), statistics.median(samples), "Raw visit median mismatch")
            close(visit.get("quality", {}).get("cv_pct"), _cv_pct(samples), "Raw visit variation mismatch")
            check(visit["quality"]["passed"] == (_cv_pct(samples) <= settings["quality_cv_pct"]), "Raw visit quality flag mismatch")
            check(visit["pilot"].get("excluded_from_labels") is True and visit["pilot"].get("selected_operations") == operations, "Pilot must remain separate from timing labels")
            if kind == "cuda_measured":
                pilot = visit["pilot"]
                pilot_operations = pilot["operations"]
                check(type(pilot_operations) is int and buffers <= pilot_operations <= settings["max_operations"] and pilot_operations % buffers == 0, "Invalid pilot graph operation count")
                pilot_samples = normalize_event_intervals(pilot["event_elapsed_ms"], pilot_operations)
                check(len(pilot_samples) == settings["pilot_repeats"], "Pilot repeat count mismatch")
                close(pilot["per_gemm_median_ms"], statistics.median(pilot_samples), "Pilot interval normalization mismatch")
                chosen = choose_operation_count(statistics.median(pilot_samples), target_replay_ms=settings["target_replay_ms"], minimum=settings["min_operations"], maximum=settings["max_operations"], buffer_sets=buffers)
                check(chosen == operations, "Captured graph count differs from the recorded adaptive pilot")
                warmup = visit["warmup"]
                check(settings["warmup_min_replays"] <= warmup["completed_replays"] <= settings["warmup_max_replays"] and warmup["actual_wall_seconds"] >= settings["warmup_min_seconds"], "Warmup did not reach its declared minimum within the cap")
                check(warmup["excluded_from_labels"] is True and warmup["operations_per_replay"] == operations, "Warmup timing scope mismatch")
                check(visit.get("numeric_execution_checked") is True, "CUDA visit lacks numerical checks")
                for name in ("correctness_before", "correctness_after"):
                    check(len(visit.get(name, [])) == buffers, "Every buffer requires numerical verification")
                    for numeric in visit[name]:
                        check(numeric.get("passed") is True and numeric.get("finite") is True, "Failed numerical correctness check")
                        check(numeric["limits"] == benchmark._correctness_limits(key[1]), "Numerical tolerance differs from the executed precision policy")
                        close(numeric["max_abs_error_limit"], numeric["limits"]["absolute_atol"] + numeric["limits"]["absolute_rtol"] * numeric["reference_max_abs"], "Numerical absolute-error limit mismatch")
                        check(0 <= numeric["relative_l2_error"] <= numeric["limits"]["relative_l2"], "Invalid relative numerical error")
                        check(0 <= numeric["max_abs_error"] <= numeric["max_abs_error_limit"], "Invalid maximum numerical error")
            else:
                check(visit.get("numeric_execution_checked") is False, "Simulation cannot claim CUDA numerical verification")
            return key

        raw = read_jsonl(root / "raw_visits.jsonl")
        raw_map, by_variant = {}, defaultdict(list)
        for visit in raw:
            cid = visit.get("case_id")
            check(cid in case_map, "Raw visit contains an undeclared case")
            key = validate_visit(visit, case_map[cid])
            check(visit.get("visit_id") not in raw_map, "Duplicate raw visit identity")
            task = tasks[(cid, visit["visit_index"])]
            check(visit.get("schedule_index") == task["schedule_index"], "Raw visit ordering differs from the schedule")
            raw_map[visit["visit_id"]] = visit
            by_variant[key].append(visit)
        rows = read_jsonl(root / "measurements.jsonl")
        seen = set()
        for row in rows:
            key = (row["case_id"], row["dtype"], row["cache_policy"])
            check(key not in seen and key in by_variant, "Duplicate or unsupported aggregate row")
            seen.add(key)
            visits = by_variant[key]
            check(len(visits) == settings["visits"], "Aggregate missing a declared visit")
            check(sorted(visit["visit_index"] for visit in visits) == list(range(1, settings["visits"] + 1)), "Duplicate or missing visit index")
            expected = _aggregate(visits, settings)
            check(row == expected, "Aggregate does not reproduce from retained raw visits")
        for case in cases:
            for dtype in manifest["dtypes"]:
                check((case["id"], dtype, "warm_reuse") in seen, "Primary precision/shape coverage is incomplete")
        eligibility = [rotation_eligibility(case, dtype, environments, settings) for case in cases for dtype in manifest["dtypes"]]
        check(read_json(root / "rotation_eligibility.json") == eligibility, "Rotation eligibility differs from observed/declared L2 and shape metadata")
        eligible_keys = {(item["case_id"], item["dtype"], "rotating_buffers") for item in eligibility if item["eligible"]}
        ineligible_keys = {(item["case_id"], item["dtype"], "rotating_buffers") for item in eligibility if not item["eligible"]}
        skipped = read_jsonl(root / "skipped_variants.jsonl")
        skipped_eligibility, skipped_memory = set(), set()
        for record in skipped:
            key = (record["case_id"], record["dtype"], record["cache_policy"])
            if record["stage"] == "eligibility":
                check(key in ineligible_keys and key not in skipped_eligibility, "Incorrect or duplicated eligibility skip")
                skipped_eligibility.add(key)
            else:
                check(record["stage"] == "memory_guard" and key in eligible_keys and key not in skipped_memory, "Incorrect or duplicated memory skip")
                check(record["memory_guard"]["passed"] is False, "Memory skip lacks a failed guard")
                check(record["incomplete_visits_retained"] == len(by_variant[key]) < settings["visits"], "Skipped rotation partial-visit count mismatch")
                skipped_memory.add(key)
        check(skipped_eligibility == ineligible_keys, "Rotation ineligibility skips are incomplete")
        check({key for key in seen if key[2] == "rotating_buffers"} == eligible_keys - skipped_memory, "Eligible rotation coverage is incomplete")
        check(not (set(by_variant) & ineligible_keys), "Ineligible rotating buffers were measured")
        check(set(by_variant) - seen <= skipped_memory, "Unexplained partial visits")
        check(status.get("completed_rows") == len(rows) and status.get("completed_visits") == len(raw), "Status coverage counts mismatch")
        controls = read_jsonl(root / "controls.jsonl")
        control_map = {case["id"]: case for case in _CONTROLS}
        control_boundaries = []
        for task in schedule:
            if task["case_position"] == 0:
                control_boundaries.append((task["schedule_index"], "visit_start"))
            position = task["case_position"] + 1
            if position % settings["control_every_cases"] == 0 or position == len(cases):
                control_boundaries.append((task["schedule_index"], "visit_end" if position == len(cases) else "periodic"))
        control_seen = set()
        for record in controls:
            check(record["case_id"] in control_map, "Undeclared control shape")
            validate_visit(record, control_map[record["case_id"]], control=True)
            index = record["visit_index"]
            check(type(index) is int and 1 <= index <= len(control_boundaries), "Control round index mismatch")
            check((record["schedule_index"], record["boundary"]) == control_boundaries[index - 1], "Control boundary mismatch")
            key = (record["case_id"], record["dtype"], index)
            check(key not in control_seen, "Duplicated control visit")
            control_seen.add(key)
        check(len(control_seen) == len(control_boundaries) * len(_CONTROLS) * len(manifest["dtypes"]), "Control coverage is incomplete")
        quality = _quality_summary(rows, controls, settings)
        check(read_json(root / "quality.json") == quality == status["quality"], "Quality summaries do not reproduce from retained visits")
        profiles = read_jsonl(root / "profiles.jsonl")
        if settings["profile"]:
            expected = {(case["id"], dtype) for case in cases for dtype in manifest["dtypes"]}
            check({(record["case_id"], record["dtype"]) for record in profiles} == expected and len(profiles) == len(expected), "Profile coverage differs from selected cases")
        else:
            check(not profiles, "Profiles were collected although disabled")
        for profile in profiles:
            check(profile.get("excluded_from_labels") is True and profile.get("excluded_from_predictor_features") is True,
                  "Profiler diagnostics must not be predictor labels or input features")
            check(profile["measurement_kind"] == kind and profile["cache_policy"] == "warm_reuse" and profile["environment_id"] == contexts[f"{profile['dtype']}/warm_reuse"]["environment_id"], "Profiler context mismatch")
            check(profile["captured_at_utc"] >= status["timing_completed_at_utc"], "Profiler capture preceded timing completion")
        check(status["profile_captures"] == len(profiles) and status["skipped_variants"] == len(skipped), "Profile/skip status counts mismatch")
        return {"passed": True, "complete": True, "checks": checks, "measurement_kind": kind,
                "rows": len(rows), "raw_visits": len(raw), "control_visits": len(controls), "selected_shapes": len(cases),
                "scope": "Manifest, environment, coverage, event normalization, arithmetic checks, and aggregation; no new GPU execution"}
    except (LabError, OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        return {"passed": False, "complete": False, "checks": checks, "error": f"{type(exc).__name__}: {exc}"}
