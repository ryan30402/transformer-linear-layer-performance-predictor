"""Bounded CUDA experiments with explicit provenance and no CPU fallback.

Importing this module does not import PyTorch. CUDA measurements use
``torch.mm(X, W_T, out=Y)`` with contiguous preallocated X/Y and a precomputed
transpose view of contiguous linear-layer weights stored as W(N, K). The eager
protocol measures repeated Python submissions. The explicit cuda_graph protocol
captures a fixed number of operations, times one replay per sample, and divides
by the captured operation count. Neither protocol measures kernel-only latency.

The only cache policy implemented here reuses the same buffers after warmup.
``ideal_bytes`` is an arithmetic lower-bound count, not measured HBM traffic.
Profiler captures are a separate artifact and never become benchmark samples.
"""

from __future__ import annotations

import importlib
import math
import platform
import random
import re
import shutil
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator

from .common import (
    DTYPE_BYTES,
    EAGER_TIMER_SCOPE,
    GRAPH_TIMER_SCOPE,
    SCHEMA_VERSION,
    LabError,
    case_lookup,
    digest,
    flops,
    ideal_bytes,
    utc_now,
    validate_measurements,
    validate_plan,
)


RUNNER_VERSION = "cuda-events-1.0.0"
MAX_SESSION_GEMM_CALLS = 100_000
MAX_SESSION_FLOPS = 20_000_000_000_000_000
MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS = 2_000_000
MAX_CAPTURED_GEMM_NODES = 8192
MEMORY_SAFETY_FRACTION = 0.65
WORKSPACE_ALLOWANCE_BYTES = 256 * 1024**2
NVTX_RANGE = "linear_perf_measure"
_MATH_LOCK = threading.RLock()
_REFERENCE = (
    "FP32 torch.mm(X, W_T) of the same already-represented inputs, with CUDA matmul "
    "FP32 precision set to IEEE; this is not a float64 oracle"
)


class BenchmarkRunError(LabError):
    """A failed run with successful prior rows available for caller persistence."""

    def __init__(
        self,
        message: str,
        *,
        completed_rows: list[dict] | None = None,
        failed_case_id: str | None = None,
        environment: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.completed_rows = list(completed_rows or [])
        self.failed_case_id = failed_case_id
        self.environment = environment


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise LabError(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build on "
            "an NVIDIA GPU machine using https://pytorch.org/get-started/locally/. "
            "This runner does not substitute CPU timings for CUDA measurements."
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise LabError(
            "PyTorch could not load its runtime libraries. Repair the PyTorch/CUDA "
            "installation before collecting measurements."
        ) from exc


def _device_index(device: str) -> int | None:
    if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise LabError("device must be 'cuda' or 'cuda:N'; CPU fallback is forbidden")
    return int(device.split(":", 1)[1]) if ":" in device else None


def _require_cuda(torch: Any, dtype: str, device: str) -> Any:
    index = _device_index(device)
    if getattr(torch.version, "hip", None):
        raise LabError("This NVIDIA experiment requires a CUDA build, not a ROCm build")
    if not getattr(torch.version, "cuda", None) or not torch.cuda.is_available():
        raise LabError(
            "CUDA is unavailable. Run doctor on an NVIDIA GPU machine with a "
            "compatible driver and CUDA-enabled PyTorch; no CPU timing was collected."
        )
    index = torch.cuda.current_device() if index is None else index
    if index >= torch.cuda.device_count():
        raise LabError(f"Requested CUDA device index {index} is not visible")
    selected = torch.device(f"cuda:{index}")
    if dtype == "bfloat16":
        with torch.cuda.device(selected):
            # Older PyTorch versions do not expose including_emulation.  Requiring
            # Ampere or later avoids accepting an emulated BF16 performance path.
            checker = getattr(torch.cuda, "is_bf16_supported", None)
            if checker is None:
                supported = False
            else:
                try:
                    supported = bool(checker(including_emulation=False))
                except TypeError:
                    supported = bool(checker()) and torch.cuda.get_device_capability(selected)[0] >= 8
            if not supported:
                raise LabError(
                    "Native BF16 is unavailable on this GPU/build. Use a supported "
                    "GPU or register a new float16/float32 plan; do not relabel existing data."
                )
    return selected


def _command_version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()[:2000] if result.returncode == 0 else None


def _driver_versions() -> list[str]:
    output = _command_version(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if output is None:
        return []
    # Only retain version numbers, not arbitrary command output or device names.
    return sorted({line.strip() for line in output.splitlines() if re.fullmatch(r"[0-9.]+", line.strip())})


def _gpu_metadata(torch: Any, device: Any) -> dict:
    props = torch.cuda.get_device_properties(device)
    result = {
        "logical_device_index": int(device.index),
        "name": str(props.name),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(props.total_memory),
        "multiprocessor_count": int(props.multi_processor_count),
    }
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        result["uuid"] = str(uuid)
    return result


def doctor() -> dict:
    """Report relevant runtime capabilities without collecting environment secrets."""
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "pytorch_available": False,
        "pytorch_version": None,
        "cuda_runtime_version": None,
        "cuda_available": False,
        "gpus": [],
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "driver_versions": _driver_versions(),
        "ncu_available": shutil.which("ncu") is not None,
        "ncu_version": _command_version(["ncu", "--version"]),
        "benchmark_ready": False,
        "notes": [],
    }
    try:
        torch = _load_torch()
    except LabError as exc:
        result["notes"].append(str(exc))
        return result
    result["pytorch_available"] = True
    result["pytorch_version"] = str(torch.__version__)
    result["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"] and not getattr(torch.version, "hip", None):
            result["gpus"] = [
                _gpu_metadata(torch, torch.device(f"cuda:{index}"))
                for index in range(torch.cuda.device_count())
            ]
            result["benchmark_ready"] = bool(result["gpus"])
        else:
            result["notes"].append("An NVIDIA CUDA device is required for real measurements.")
    except (RuntimeError, OSError) as exc:
        result["notes"].append(f"CUDA inspection failed ({type(exc).__name__}); check the driver/runtime.")
    result["notes"].append(
        "benchmark_ready checks CUDA presence; each run separately checks dtype support, "
        "available memory, correctness, and its budget."
    )
    return result


@contextmanager
def _precision_policy(torch: Any) -> Iterator[dict]:
    """Select one TF32 API family and restore every changed setting on exit."""
    with _MATH_LOCK:
        matmul = torch.backends.cuda.matmul
        saved: list[tuple[Any, str, Any]] = []
        policy: dict[str, Any] = {"fp32_reference_precision": "ieee"}

        def change(owner: Any, name: str, value: Any, previous: Any) -> None:
            saved.append((owner, name, previous))
            setattr(owner, name, value)

        try:
            try:
                previous = getattr(matmul, "fp32_precision")
            except AttributeError:
                previous = getattr(matmul, "allow_tf32")
                change(matmul, "allow_tf32", False, previous)
                policy["tf32_control_api"] = "allow_tf32"
                policy["allow_tf32"] = False
            else:
                # Do not even read allow_tf32 on the new API path: PyTorch does
                # not support mixing the old and new precision-control APIs.
                change(matmul, "fp32_precision", "ieee", previous)
                policy["tf32_control_api"] = "fp32_precision"
                policy["fp32_precision"] = "ieee"
            for name in (
                "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction",
                "allow_fp16_accumulation",
            ):
                try:
                    previous = getattr(matmul, name)
                except AttributeError:
                    policy[name] = "unavailable"
                    continue
                change(matmul, name, False, previous)
                policy[name] = False
            yield policy
        finally:
            for owner, name, previous in reversed(saved):
                setattr(owner, name, previous)


def _environment_identity(payload: dict) -> str:
    """Hash an explicitly stable payload; timestamps/free memory live outside it."""
    return digest(payload)


def _execution_metadata(timing: dict) -> dict:
    mode = timing.get("mode", "eager")
    result = {
        "operation": "torch.mm(X, W_T, out=Y); W_T=W.t() prepared before timing",
        "mode": mode,
        "layout": "contiguous_X_MK_weight_NK_Y_MN; transposed_weight_view_KN",
        "autocast": "disabled",
        "grad": "inference_mode",
        "input_distribution": "normal; X scaled by 1/sqrt(K) before dtype cast",
        "timer": "CUDA_events_mean_per_call_within_each_repeat",
        "timer_scope": EAGER_TIMER_SCOPE,
    }
    if mode == "cuda_graph":
        result.update(
            timer="CUDA_events_one_graph_replay_divided_by_captured_operations",
            timer_scope=GRAPH_TIMER_SCOPE,
            captured_operations_per_graph=timing["iterations"],
            graph_warmup_replays=timing["graph_warmup_replays"],
            capture_and_warmup_excluded_from_timing=True,
            output_poisoned_after_capture_before_replay=True,
            setup="eager warmup and capture on a joined side stream; static input/output addresses",
        )
        if "graph_warmup_min_seconds" in timing:
            result.update(
                graph_warmup_min_seconds=timing["graph_warmup_min_seconds"],
                graph_warmup_max_replays=timing["graph_warmup_max_replays"],
                graph_warmup_stop_rule="fixed minimum wall duration and minimum replay count; never sample-quality adaptive",
                graph_warmup_scope="target graph before each case visit; batching prepares each variant before every timed sample",
            )
    return result


def _collect_environment(torch: Any, plan: dict, selected: Any, policy: dict) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "pytorch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "driver_versions": _driver_versions(),
        "device": _gpu_metadata(torch, selected),
        "dtype": plan["dtype"],
        "cache_policy": plan["cache_policy"],
        "timing": dict(plan["timing"]),
        "collection": dict(plan.get("collection", {
            "blocks": 1, "order": "caller_order", "aggregation": "median_of_repeat_means",
        })),
        "math_policy": dict(policy),
        "execution": _execution_metadata(plan["timing"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "measurement_kind": "cuda_measured",
        "environment_id": _environment_identity(payload),
        "plan_hash": digest(plan),
        "identity": payload,
        "limitations": [
            "Clock, thermal, power, and concurrent-load differences are not controlled by this identity.",
            "warm_reuse does not flush caches or measure HBM bytes.",
            "Run one experiment process at a time on an otherwise idle GPU.",
        ],
    }


def cuda_environment(plan: dict, device: str = "cuda:0") -> dict:
    validate_plan(plan)
    _device_index(device)
    torch = _load_torch()
    selected = _require_cuda(torch, plan["dtype"], device)
    with torch.cuda.device(selected), _precision_policy(torch) as policy:
        return _collect_environment(torch, plan, selected, policy)


def _validate_requested_cases(plan: dict, cases: list[dict]) -> list[dict]:
    validate_plan(plan)
    if not isinstance(cases, list) or not cases:
        raise LabError("Provide a nonempty list of registered experiment cases")
    lookup = case_lookup(plan)
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise LabError("Requested cases must be registered case objects")
        case_id = case["id"]
        if case_id in seen or case_id not in lookup or case != lookup[case_id]:
            raise LabError("Requested cases contain duplicate, unknown, or modified plan cases")
        seen.add(case_id)
    return cases


def _validate_work_budget(plan: dict, cases: list[dict], *, call_multiplier: int = 1) -> None:
    timing = plan["timing"]
    calls_per_case = _bounded_operation_count(timing)
    if timing.get("mode", "eager") == "cuda_graph":
        _check_graph_size(timing["iterations"] * call_multiplier)
    calls = len(cases) * calls_per_case * call_multiplier
    work = sum(flops(case) for case in cases) * calls_per_case * call_multiplier
    if calls > MAX_SESSION_GEMM_CALLS or work > MAX_SESSION_FLOPS:
        raise LabError(
            "Requested work exceeds this runner's per-session safety bound. Register "
            "a smaller timing plan or collect fewer cases in this invocation."
        )


def _bounded_operation_count(timing: dict) -> int:
    """Conservative fixed-work bound, including capture and minimum warmups.

    Captured nodes are recorded rather than executed during capture. Counting
    them here additionally bounds setup work; this is not an observed call count.
    Extra duration-conditioning replays are charged by _ConditioningBudget.
    """
    extra = 1 + timing["graph_warmup_replays"] if timing.get("mode", "eager") == "cuda_graph" else 0
    return timing["warmup"] + timing["iterations"] * (timing["repeats"] + extra) + 2


def _check_graph_size(captured_gemm_nodes: int) -> None:
    if captured_gemm_nodes > MAX_CAPTURED_GEMM_NODES:
        raise LabError("CUDA graph exceeds the captured GEMM-node safety limit; reduce timing.iterations")


class _ConditioningBudget:
    """Additional duration warmup has its own explicit, shared call budget.

    The existing 100,000-call preflight still bounds the fixed protocol work.
    Its conservative FLOP reservation includes minimum graph warmups, capture
    records, timed work, and reference checks. Extra replay work is charged
    before submission and shares the existing total FLOP limit with that fixed
    reservation. One instance must span the whole benchmark invocation.
    """

    def __init__(self, fixed_reserved_flops: int) -> None:
        if type(fixed_reserved_flops) is not int or not 0 <= fixed_reserved_flops <= MAX_SESSION_FLOPS:
            raise LabError("Invalid fixed-work FLOP reservation for duration warmup")
        self.fixed_reserved_flops = fixed_reserved_flops
        self.additional_gemm_calls = 0
        self.additional_flops = 0

    def charge(self, gemm_calls: int, operation_flops: int) -> None:
        if type(gemm_calls) is not int or gemm_calls < 1 or type(operation_flops) is not int or operation_flops < 1:
            raise LabError("Duration warmup requires explicit positive GEMM/FLOP accounting")
        calls = self.additional_gemm_calls + gemm_calls
        work = self.additional_flops + operation_flops
        if calls > MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS:
            raise LabError(f"Additional graph conditioning exceeds its {MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS:,}-GEMM-call per-invocation limit")
        if self.fixed_reserved_flops + work > MAX_SESSION_FLOPS:
            raise LabError("Fixed work plus additional graph conditioning exceeds the per-invocation FLOP limit")
        self.additional_gemm_calls = calls
        self.additional_flops = work

    def metadata(self) -> dict:
        return {
            "scope": "shared across this benchmark invocation; counters never reset per case or batching variant",
            "fixed_work_gemm_call_limit": MAX_SESSION_GEMM_CALLS,
            "fixed_reserved_flops": self.fixed_reserved_flops,
            "additional_conditioning_gemm_call_limit": MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS,
            "additional_conditioning_gemm_calls": self.additional_gemm_calls,
            "additional_conditioning_flops": self.additional_flops,
            "total_flop_limit": MAX_SESSION_FLOPS,
            "interpretation": "Conservative operation accounting, not measured hardware counters; extra replays are charged before submission.",
        }


def _estimated_peak_bytes(case: dict, dtype: str, *, extra_output: bool = False) -> int:
    mk, kn, mn = case["m"] * case["k"], case["k"] * case["n"], case["m"] * case["n"]
    # Inputs + their FP32 reference copies + low-precision output, FP32 reference,
    # converted output, difference/absolute-value scratch, and finite masks.
    # This is a conservative guard estimate, not a measured allocator peak.
    total = (DTYPE_BYTES[dtype] + 4) * (mk + kn)
    total += (DTYPE_BYTES[dtype] + 4 * 5 + 2) * mn
    if extra_output:
        total += DTYPE_BYTES[dtype] * mn
    return total + WORKSPACE_ALLOWANCE_BYTES


def _check_memory(torch: Any, selected: Any, case: dict, dtype: str, *, extra_output: bool = False) -> dict:
    free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
    reusable = max(0, torch.cuda.memory_reserved(selected) - torch.cuda.memory_allocated(selected))
    available = min(int(total_bytes), int(free_bytes) + int(reusable))
    estimated = _estimated_peak_bytes(case, dtype, extra_output=extra_output)
    if estimated > MEMORY_SAFETY_FRACTION * available:
        raise LabError(
            f"Case {case['id']} needs an estimated {estimated / 1024**2:.0f} MiB "
            f"including reference/scratch allowance, exceeding the {MEMORY_SAFETY_FRACTION:.0%} "
            f"safety limit on {available / 1024**2:.0f} MiB currently available."
        )
    return {
        "estimated_peak_bytes": estimated,
        "available_bytes_before_case": available,
        "safety_fraction": MEMORY_SAFETY_FRACTION,
        "estimate_is_not_measured_peak": True,
    }


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise LabError(
            "Session wall-time budget exhausted. Completed rows remain available; "
            "a running CUDA kernel cannot be preempted by this Python guard."
        )


def _session_deadline(plan: dict, command_deadline: float | None = None) -> float:
    """An outer command may shorten, but never extend, this invocation's budget."""
    local_deadline = time.monotonic() + plan["budget"]["max_wall_seconds"]
    if command_deadline is None:
        return local_deadline
    if (
        isinstance(command_deadline, bool)
        or not isinstance(command_deadline, (int, float))
        or not math.isfinite(command_deadline)
    ):
        raise LabError("deadline_monotonic must be a finite monotonic-clock deadline")
    return min(local_deadline, command_deadline)


def _case_seed(seed: int, case_id: str) -> int:
    return int(digest({"seed": seed, "case_id": case_id})[:16], 16) % (2**63 - 1)


def _allocate_inputs(torch: Any, selected: Any, case: dict, dtype: str, seed: int) -> tuple[Any, Any, Any]:
    """Return X, a precomputed W.T view, and output; W uses nn.Linear storage."""
    generator = torch.Generator(device=selected)
    generator.manual_seed(_case_seed(seed, case["id"]))
    target_dtype = getattr(torch, dtype)
    x = torch.randn((case["m"], case["k"]), device=selected, dtype=torch.float32, generator=generator)
    x.mul_(1 / math.sqrt(case["k"]))
    x = x.to(dtype=target_dtype)
    w = torch.randn((case["n"], case["k"]), device=selected, dtype=torch.float32, generator=generator)
    w = w.to(dtype=target_dtype)
    w_transposed = w.t()
    y = torch.empty((case["m"], case["n"]), device=selected, dtype=target_dtype)
    return x, w_transposed, y


def _correctness_limits(dtype: str) -> dict:
    return {
        "bfloat16": {"relative_l2": 0.01, "absolute_atol": 0.01, "absolute_rtol": 0.01},
        "float16": {"relative_l2": 0.002, "absolute_atol": 0.002, "absolute_rtol": 0.002},
        "float32": {"relative_l2": 0.0001, "absolute_atol": 0.00005, "absolute_rtol": 0.00005},
    }[dtype].copy()


def _correctness(torch: Any, output: Any, reference: Any, dtype: str) -> dict:
    limits = _correctness_limits(dtype)
    finite = bool(torch.isfinite(output).all().item() and torch.isfinite(reference).all().item())
    result = {
        "passed": False,
        "finite": finite,
        "relative_l2_error": None,
        "max_abs_error": None,
        "reference": _REFERENCE,
        "limits": limits,
    }
    if not finite:
        return result
    difference = output.float() - reference
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    relative_error = difference_norm / max(reference_norm, 1e-30)
    max_error = float(difference.abs().max().item())
    reference_max = float(reference.abs().max().item())
    allowed_max = limits["absolute_atol"] + limits["absolute_rtol"] * reference_max
    result.update(
        relative_l2_error=relative_error,
        max_abs_error=max_error,
        reference_max_abs=reference_max,
        max_abs_error_limit=allowed_max,
        passed=bool(
            math.isfinite(relative_error)
            and math.isfinite(max_error)
            and relative_error <= limits["relative_l2"]
            and max_error <= allowed_max
        ),
    )
    return result


def _assert_correctness(result: dict, label: str) -> None:
    if not result["passed"]:
        raise LabError(f"Correctness failed for {label}: {result}")


def _event_pair(torch: Any) -> tuple[Any, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # CUDA events initialize lazily. Prime both outside the measured interval.
    start.record()
    end.record()
    end.synchronize()
    return start, end


def _event_sample(operation: Callable[[], None], events: tuple[Any, Any], iterations: int) -> float:
    start, end = events
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    sample = float(start.elapsed_time(end)) / iterations
    if not math.isfinite(sample) or sample <= 0:
        raise LabError("CUDA events returned nonpositive/nonfinite elapsed time; increase iterations")
    return sample


def _capture_operation_graph(torch: Any, selected: Any, operation: Callable[[], None], timing: dict, deadline: float) -> Any:
    """Capture repeated work with all tensor references owned by the caller.

    Warmup initializes library/workspace state on the same side stream used for
    capture. Synchronization and deadline checks remain outside the capture body.
    The returned graph must stay alive until all replays have completed. The
    caller retains its preallocated input/output tensors through correctness
    checks. No eager fallback is allowed if capture fails.
    """
    _check_deadline(deadline)
    current = torch.cuda.current_stream(selected)
    capture_stream = torch.cuda.Stream(device=selected)
    capture_stream.wait_stream(current)
    with torch.cuda.stream(capture_stream):
        for _ in range(timing["warmup"]):
            _check_deadline(deadline)
            operation()
    current.wait_stream(capture_stream)
    torch.cuda.synchronize(selected)
    _check_deadline(deadline)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(timing["iterations"]):
            operation()
    current.wait_stream(capture_stream)
    torch.cuda.synchronize(selected)
    _check_deadline(deadline)
    return graph


def _warm_graph(
    torch: Any, selected: Any, graph: Any, timing: dict, deadline: float, *,
    budget: _ConditioningBudget | None,
    gemms_per_operation: int = 1,
    flops_per_operation: int | None = None,
    fixed_reserved_replays: int | None = None,
) -> dict:
    """Warm the target graph for a predeclared duration, never until CV passes.

    Duration mode synchronizes each replay, so the reported elapsed wall time
    covers completed work rather than an unbounded queue. The minimum replay
    count is already reserved by the static work check; subsequent batching
    warmups pass zero reserved replays and charge every replay to the shared
    additional-conditioning budget. No measured sample is inspected here.
    """
    minimum = timing["graph_warmup_replays"]
    target = timing.get("graph_warmup_min_seconds", 0.0)
    maximum = timing.get("graph_warmup_max_replays", minimum)
    reserved = minimum if fixed_reserved_replays is None else fixed_reserved_replays
    if type(reserved) is not int or not 0 <= reserved <= minimum:
        raise LabError("Invalid graph warmup reservation")
    if type(gemms_per_operation) is not int or gemms_per_operation < 1:
        raise LabError("Graph warmup requires positive GEMM accounting")
    if target > 0 and (budget is None or type(flops_per_operation) is not int or flops_per_operation <= 0):
        raise LabError("Duration graph warmup requires an explicit conditioning budget and FLOP count")
    calls_per_replay = timing["iterations"] * gemms_per_operation
    work_per_replay = timing["iterations"] * flops_per_operation if flops_per_operation is not None else None
    _check_deadline(deadline)
    if target > 0:
        torch.cuda.synchronize(selected)
    started = time.monotonic()
    completed, elapsed = 0, 0.0
    while completed < minimum or elapsed < target:
        _check_deadline(deadline)
        if completed >= maximum:
            raise LabError(
                f"Graph warmup reached its {maximum}-replay cap after {elapsed:.6f} seconds "
                f"before satisfying {target:.6f} seconds and {minimum} replays"
            )
        if completed >= reserved:
            if budget is None or work_per_replay is None:
                raise LabError("Additional graph warmup requires explicit work accounting")
            budget.charge(calls_per_replay, work_per_replay)
        graph.replay()
        completed += 1
        if target > 0:
            torch.cuda.synchronize(selected)
        elapsed = time.monotonic() - started
    torch.cuda.synchronize(selected)
    elapsed = time.monotonic() - started
    _check_deadline(deadline)
    return {
        "policy": "fixed_minimum_duration_and_replay_count" if target > 0 else "fixed_replay_count",
        "requested_min_seconds": target,
        "actual_wall_seconds": elapsed,
        "required_min_replays": minimum,
        "max_replays": maximum,
        "completed_replays": completed,
        "captured_operations_per_replay": timing["iterations"],
        "gemms_per_operation": gemms_per_operation,
        "gemm_calls": completed * calls_per_replay,
        "total_gemm_flops": completed * work_per_replay if work_per_replay is not None else None,
        "fixed_reserved_replays": reserved,
        "additional_conditioning_gemm_calls": max(0, completed - reserved) * calls_per_replay,
        "additional_conditioning_flops": max(0, completed - reserved) * work_per_replay if work_per_replay is not None else None,
        "excluded_from_timing_samples": True,
        "sample_quality_used_for_stopping": False,
        "interpretation": "Wall duration of completed target-graph warmup; not proof of stable clocks or GPU busy time.",
    }


def _measure_operation(
    torch: Any, selected: Any, operation: Callable[[], None], timing: dict, deadline: float, *,
    before_graph_replay: Callable[[], None] | None = None,
    conditioning_budget: _ConditioningBudget | None = None,
    flops_per_operation: int | None = None,
    warmup_metadata: dict | None = None,
) -> list[float]:
    _check_deadline(deadline)
    if timing.get("mode", "eager") == "cuda_graph":
        graph = _capture_operation_graph(torch, selected, operation, timing, deadline)
        if before_graph_replay is not None:
            before_graph_replay()
        prepared = _warm_graph(torch, selected, graph, timing, deadline,
                               budget=conditioning_budget, flops_per_operation=flops_per_operation)
        if warmup_metadata is not None:
            warmup_metadata.update(prepared)
        events = _event_pair(torch)
        samples = []
        for _ in range(timing["repeats"]):
            _check_deadline(deadline)
            samples.append(_event_sample(graph.replay, events, 1) / timing["iterations"])
        return samples
    for _ in range(timing["warmup"]):
        _check_deadline(deadline)
        operation()
    torch.cuda.synchronize(selected)
    events = _event_pair(torch)
    samples = []
    for _ in range(timing["repeats"]):
        _check_deadline(deadline)
        samples.append(_event_sample(operation, events, timing["iterations"]))
    return samples


def benchmark_cases(
    plan: dict,
    cases: list[dict],
    *,
    device: str = "cuda:0",
    environment: dict | None = None,
    deadline_monotonic: float | None = None,
) -> list[dict]:
    """Measure registered cases; failures preserve prior rows in BenchmarkRunError."""
    cases = _validate_requested_cases(plan, cases)
    _validate_work_budget(plan, cases)
    conditioning_budget = _ConditioningBudget(
        sum(flops(case) for case in cases) * _bounded_operation_count(plan["timing"])
    )
    _device_index(device)
    if environment is not None and not isinstance(environment, dict):
        raise LabError("environment must be the object returned by cuda_environment")
    deadline = _session_deadline(plan, deadline_monotonic)
    _check_deadline(deadline)
    torch = _load_torch()
    selected = _require_cuda(torch, plan["dtype"], device)
    rows: list[dict] = []
    with torch.cuda.device(selected), _precision_policy(torch) as policy, torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        current_environment = _collect_environment(torch, plan, selected, policy)
        if environment is not None:
            if (
                environment.get("environment_id") != current_environment["environment_id"]
                or environment.get("plan_hash") != digest(plan)
                or environment.get("identity") != current_environment["identity"]
            ):
                raise LabError("Provided environment differs from the current device/software/timing policy or plan")
        environment = current_environment
        for case in cases:
            try:
                _check_deadline(deadline)
                memory_guard = _check_memory(torch, selected, case, plan["dtype"])
                x, w, y = _allocate_inputs(torch, selected, case, plan["dtype"], plan["seed"])
                reference = torch.mm(x.float(), w.float())

                def operation() -> None:
                    torch.mm(x, w, out=y)

                operation()
                correctness = _correctness(torch, y, reference, plan["dtype"])
                _assert_correctness(correctness, case["id"])
                warmup_metadata: dict = {}
                samples = _measure_operation(
                    torch, selected, operation, plan["timing"], deadline,
                    before_graph_replay=lambda: y.fill_(float("nan")),
                    conditioning_budget=conditioning_budget,
                    flops_per_operation=flops(case),
                    warmup_metadata=warmup_metadata,
                )
                # Validate the final overwritten output too; checking happens
                # after timing and cannot contribute to any event sample.
                correctness = _correctness(torch, y, reference, plan["dtype"])
                _assert_correctness(correctness, case["id"])
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "case_id": case["id"],
                    **{key: case[key] for key in ("m", "n", "k", "split")},
                    "dtype": plan["dtype"],
                    "cache_policy": plan["cache_policy"],
                    "plan_hash": digest(plan),
                    "environment_id": environment["environment_id"],
                    "measurement_kind": "cuda_measured",
                    "status": "ok",
                    "measured_at_utc": utc_now(),
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "flops": flops(case),
                    "ideal_bytes": ideal_bytes(case, plan["dtype"]),
                    "correctness": correctness,
                    "memory_guard": memory_guard,
                    "timing": dict(plan["timing"]),
                    "execution_mode": plan["timing"].get("mode", "eager"),
                    "timer_scope": _execution_metadata(plan["timing"])["timer_scope"],
                    "warmup": warmup_metadata,
                    "conditioning_budget": conditioning_budget.metadata(),
                }
                validate_measurements([row], plan)
                rows.append(row)
                del x, w, y, reference
            except (RuntimeError, LabError, MemoryError) as exc:
                raise BenchmarkRunError(
                    f"CUDA benchmark stopped at {case['id']}: {exc}",
                    completed_rows=rows,
                    failed_case_id=case["id"],
                    environment=environment,
                ) from exc
    validate_measurements(rows, plan)
    return rows


def benchmark_batching(
    plan: dict,
    *,
    device: str = "cuda:0",
    deadline_monotonic: float | None = None,
) -> dict:
    """Compare equal offline work: 64 small GEMMs versus one large GEMM."""
    validate_plan(plan)
    _device_index(device)
    total_rows, chunk_rows, n, k = 1024, 16, 1024, 1024
    chunks = total_rows // chunk_rows
    case = {"id": "offline_batching", "m": total_rows, "n": n, "k": k}
    timing = plan["timing"]
    calls_per_variant = _bounded_operation_count(timing)
    graph_mode = timing.get("mode", "eager") == "cuda_graph"
    duration_mode = graph_mode and timing.get("graph_warmup_min_seconds", 0.0) > 0
    if graph_mode:
        _check_graph_size(chunks * timing["iterations"])
    if (chunks + 1) * calls_per_variant > MAX_SESSION_GEMM_CALLS:
        raise LabError("Batching experiment exceeds the per-session GEMM-call limit; reduce the registered timing plan")
    if 2 * flops(case) * calls_per_variant > MAX_SESSION_FLOPS:
        raise LabError("Batching experiment exceeds the per-session FLOP limit")
    conditioning_budget = _ConditioningBudget(2 * flops(case) * calls_per_variant)
    deadline = _session_deadline(plan, deadline_monotonic)
    _check_deadline(deadline)
    torch = _load_torch()
    selected = _require_cuda(torch, plan["dtype"], device)
    with torch.cuda.device(selected), _precision_policy(torch) as policy, torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        environment = _collect_environment(torch, plan, selected, policy)
        _check_deadline(deadline)
        memory_guard = _check_memory(torch, selected, case, plan["dtype"], extra_output=True)
        x, w, y_large = _allocate_inputs(torch, selected, case, plan["dtype"], plan["seed"])
        y_small = torch.empty_like(y_large)
        # Views and output buffers are prepared before any timed operation.
        input_chunks = tuple(x[start:start + chunk_rows] for start in range(0, total_rows, chunk_rows))
        output_chunks = tuple(y_small[start:start + chunk_rows] for start in range(0, total_rows, chunk_rows))
        pairs = tuple(zip(input_chunks, output_chunks))
        reference = torch.mm(x.float(), w.float())

        def small() -> None:
            for x_chunk, y_chunk in pairs:
                torch.mm(x_chunk, w, out=y_chunk)

        def large() -> None:
            torch.mm(x, w, out=y_large)

        operations = {"small_gemms": small, "one_large_gemm": large}
        outputs = {"small_gemms": y_small, "one_large_gemm": y_large}
        for label, operation in operations.items():
            operation()
            _assert_correctness(_correctness(torch, outputs[label], reference, plan["dtype"]), label)
        graphs = {}
        if graph_mode:
            for label, operation in operations.items():
                graphs[label] = _capture_operation_graph(torch, selected, operation, timing, deadline)
            # Capture does not execute nodes. Poison the eager outputs so the
            # later correctness checks require replay to have written them.
            for output in outputs.values():
                output.fill_(float("nan"))
            replay_operations = {label: graph.replay for label, graph in graphs.items()}
            if not duration_mode:
                # Use the fixed-count preparation for this mode. Duration
                # mode instead prepares each variant immediately before its
                # timed sample, since switching variants changes the workload.
                for index in range(timing["graph_warmup_replays"]):
                    order = list(operations) if index % 2 == 0 else list(reversed(operations))
                    for label in order:
                        _check_deadline(deadline)
                        replay_operations[label]()
        else:
            for index in range(timing["warmup"]):
                _check_deadline(deadline)
                order = (small, large) if index % 2 == 0 else (large, small)
                for operation in order:
                    operation()
        torch.cuda.synchronize(selected)
        events = _event_pair(torch)
        samples: dict[str, list[float]] = {label: [] for label in operations}
        warmup_blocks: dict[str, list[dict]] = {label: [] for label in operations}
        orders = []
        rng = random.Random(_case_seed(plan["seed"], "batching_order"))
        first = list(operations)
        rng.shuffle(first)
        for repeat in range(timing["repeats"]):
            # Balanced alternating order reduces always-first thermal/cache bias.
            # Cache residency still differs inherently between these workloads.
            order = first if repeat % 2 == 0 else list(reversed(first))
            orders.append(list(order))
            for label in order:
                _check_deadline(deadline)
                if graph_mode:
                    if duration_mode:
                        warmed = _warm_graph(
                            torch, selected, graphs[label], timing, deadline,
                            budget=conditioning_budget,
                            gemms_per_operation=chunks if label == "small_gemms" else 1,
                            flops_per_operation=flops(case),
                            fixed_reserved_replays=timing["graph_warmup_replays"] if repeat == 0 else 0,
                        )
                        warmed["repeat_index"] = repeat
                        warmed["variant"] = label
                        warmup_blocks[label].append(warmed)
                    # Each graph contains iterations complete offline workloads.
                    # The denominator is workloads, not the number of GEMM nodes.
                    value = _event_sample(replay_operations[label], events, 1) / timing["iterations"]
                else:
                    value = _event_sample(operations[label], events, timing["iterations"])
                samples[label].append(value)
        variants = {}
        for label, output in outputs.items():
            correctness = _correctness(torch, output, reference, plan["dtype"])
            _assert_correctness(correctness, label)
            median = statistics.median(samples[label])
            variants[label] = {
                "gemms_per_total_work": chunks if label == "small_gemms" else 1,
                "median_ms": median,
                "samples_ms": samples[label],
                "rows_per_second": total_rows * 1000 / median,
                "correctness": correctness,
                "warmup": {
                    "scope": "target variant immediately before each timed sample" if duration_mode else "fixed-count preparation before the repeat loop",
                    "blocks": warmup_blocks[label],
                },
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "measurement_kind": "cuda_measured_offline_batching",
            "status": "ok",
            "scope": "single_gpu_offline_total_work",
            "measured_at_utc": utc_now(),
            "plan_hash": digest(plan),
            "environment_id": environment["environment_id"],
            "environment": environment,
            "dtype": plan["dtype"],
            "cache_policy": plan["cache_policy"],
            "timing": dict(timing),
            "conditioning_budget": conditioning_budget.metadata(),
            "execution_mode": timing.get("mode", "eager"),
            "timer_scope": _execution_metadata(timing)["timer_scope"],
            "captured_workloads_per_graph": timing["iterations"] if graph_mode else None,
            "captured_gemm_nodes_per_variant": {"small_gemms": chunks * timing["iterations"], "one_large_gemm": timing["iterations"]} if graph_mode else None,
            "total_rows": total_rows,
            "chunk_rows": chunk_rows,
            "n": n,
            "k": k,
            "flops_per_total_work_per_variant": flops(case),
            **variants,
            "correctness": {
                "passed": all(variant["correctness"]["passed"] for variant in variants.values()),
                "reference": _REFERENCE,
                "details": "Both variants are checked independently against the same FP32 reference.",
            },
            "speedup_small_over_large": variants["small_gemms"]["median_ms"] / variants["one_large_gemm"]["median_ms"],
            "variant_order_per_repeat": orders,
            "memory_guard": memory_guard,
            "interpretation": (
                "Equal total rows and matrix products, with identical input/weight values. "
                + ("Times are CUDA-event elapsed time for one graph replay divided by the number of captured whole offline workloads. Graph launch/inter-kernel gaps remain included. Capture and replay warmups are excluded. " if graph_mode else "Times are CUDA-event stream elapsed time per whole offline workload, including possible host submission gaps. ")
                + "Input allocation and slicing are excluded. This does not measure online arrival, queueing, or tail latency."
                + (" Each batching variant receives its own fixed-duration target warmup immediately before every timed sample. Model GEMM rows instead warm once per case visit before their repeat block; batching is a separate offline workload contract, not model-training data." if duration_mode else "")
            ),
        }


def profile_case(
    plan: dict,
    case: dict,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    torch_trace: bool = False,
) -> dict:
    """Emit one warmed-up NVTX-marked GEMM, optionally with a Chrome trace."""
    _validate_requested_cases(plan, [case])
    if plan["timing"].get("mode", "eager") == "cuda_graph":
        raise LabError("profile_case currently profiles eager execution only. Register a separate eager diagnostic plan or use a dedicated graph profiler; an eager trace must not be labeled as cuda_graph evidence.")
    _device_index(device)
    output_dir = Path(output_dir)
    trace_path = output_dir / "torch_trace.json"
    if torch_trace and trace_path.exists():
        raise LabError(f"Refusing to overwrite existing profiling evidence: {trace_path}")
    deadline = time.monotonic() + plan["budget"]["max_wall_seconds"]
    torch = _load_torch()
    selected = _require_cuda(torch, plan["dtype"], device)
    with torch.cuda.device(selected), _precision_policy(torch) as policy, torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        environment = _collect_environment(torch, plan, selected, policy)
        _check_deadline(deadline)
        memory_guard = _check_memory(torch, selected, case, plan["dtype"])
        x, w, y = _allocate_inputs(torch, selected, case, plan["dtype"], plan["seed"])
        reference = torch.mm(x.float(), w.float())
        for _ in range(plan["timing"]["warmup"]):
            torch.mm(x, w, out=y)
        torch.cuda.synchronize(selected)
        correctness = _correctness(torch, y, reference, plan["dtype"])
        _assert_correctness(correctness, case["id"])
        _check_deadline(deadline)
        profiler = None
        if torch_trace:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            )
        with profiler if profiler is not None else nullcontext():
            torch.cuda.nvtx.range_push(NVTX_RANGE)
            try:
                torch.mm(x, w, out=y)
            finally:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(selected)
        if profiler is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
        return {
            "schema_version": SCHEMA_VERSION,
            "measurement_kind": "profile_capture",
            "status": "ok",
            "case_id": case["id"],
            "plan_hash": digest(plan),
            "environment_id": environment["environment_id"],
            "environment": environment,
            "captured_at_utc": utc_now(),
            "nvtx_range": NVTX_RANGE,
            "profiled_iterations": 1,
            "torch_trace_path": str(trace_path) if profiler is not None else None,
            "timing_excluded_from_model": True,
            "correctness": correctness,
            "memory_guard": memory_guard,
            "notes": [
                "External profiling can replay kernels, serialize execution, and alter cache behavior.",
                "Use an ordinary benchmark run for latency; no profile timing is stored as a sample.",
            ],
        }
