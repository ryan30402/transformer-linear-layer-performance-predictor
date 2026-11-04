"""Bounded diagnostics, all excluded from latency-model training.

Sensitivity data compare declared execution/preparation contracts rather than
claiming an isolated causal launch overhead. Profile captures use the actual
CUDA Graph, not a substituted eager trace. Rotating-buffer experiments vary the
working set; they do not prove DRAM traffic, cold caches or full cache eviction.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from . import benchmark
from .common import LabError, case_lookup, digest, flops, ideal_bytes, utc_now, validate_measurements, validate_plan, write_json
from .measurement import (
    MeasurementError, TelemetryRecorder, _check_environment, _session_id,
    _simulate_visit, _validate, _visit,
)
from .numerical_witness import collect_numerical_witnesses


def _diagnostic_plan(plan, mode, iterations):
    derived = copy.deepcopy(plan)
    timing = {
        "mode": mode, "warmup": plan["timing"]["warmup"],
        "iterations": iterations, "repeats": plan["timing"]["repeats"],
    }
    if mode == "cuda_graph":
        for key in ("graph_warmup_replays", "graph_warmup_min_seconds", "graph_warmup_max_replays"):
            timing[key] = plan["timing"][key]
    derived["timing"] = timing
    derived["diagnostic_contract"] = {
        "excluded_from_model": True,
        "source_plan_hash": digest(plan),
        "purpose": "replay_count_and_execution_mode_sensitivity",
        "matched_minimum_warmup_seconds": plan["timing"]["graph_warmup_min_seconds"],
        "matched_minimum_complete_chunks": plan["timing"]["graph_warmup_replays"],
        "eager_chunk_calls": iterations,
        "timing_scope": benchmark._execution_metadata(timing)["timer_scope"],
        "interpretation": "Paired execution contracts; not an isolated causal kernel-launch cost.",
    }
    validate_plan(derived)
    return derived


def _diagnostic_environment(derived_plan, source, device):
    if source["measurement_kind"] == "simulated":
        identity = copy.deepcopy(source["identity"])
        identity["timing"] = copy.deepcopy(derived_plan["timing"])
        identity["execution"] = benchmark._execution_metadata(derived_plan["timing"])
        result = {
            "schema_version": 1, "measurement_kind": "simulated",
            "identity": identity, "environment_id": "simulated:" + digest(identity),
            "plan_hash": digest(derived_plan), "captured_at_utc": utc_now(),
        }
    else:
        result = benchmark.cuda_environment(derived_plan, device)
        # Timing is deliberately changed. All other stable hardware/software/
        # numerical fields must stay authenticated against the parent.
        excluded = {"timing", "execution"}
        before = {k: v for k, v in source["identity"].items() if k not in excluded}
        after = {k: v for k, v in result["identity"].items() if k not in excluded}
        if before != after:
            raise LabError("Hardware/software/numerical environment changed during diagnostics")
    result["session_id"] = source["session_id"]
    result["source_environment_id"] = source["environment_id"]
    return result


def _eager_matched_visit(plan, case, environment, *, device, deadline):
    """Eager sensitivity visit with the same declared completed-work warmup duration."""
    benchmark._validate_requested_cases(plan, [case])
    benchmark._validate_work_budget(plan, [case])
    torch = benchmark._load_torch()
    selected = benchmark._require_cuda(torch, plan["dtype"], device)
    started, utc = time.monotonic(), utc_now()
    preparation = plan["diagnostic_contract"]
    with torch.cuda.device(selected), benchmark._precision_policy(torch) as policy, \
            torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        current = benchmark._collect_environment(torch, plan, selected, policy)
        if current["environment_id"] != environment["environment_id"] or current["identity"] != environment["identity"]:
            raise LabError("Eager diagnostic environment changed")
        memory_guard = benchmark._check_memory(torch, selected, case, plan["dtype"])
        x, w, y = benchmark._allocate_inputs(torch, selected, case, plan["dtype"], plan["seed"])
        reference = torch.mm(x.float(), w.float())
        operation = lambda: torch.mm(x, w, out=y)
        operation()
        before = benchmark._correctness(torch, y, reference, plan["dtype"])
        benchmark._assert_correctness(before, case["id"])
        for _ in range(plan["timing"]["warmup"]):
            benchmark._check_deadline(deadline)
            operation()
        torch.cuda.synchronize(selected)
        budget = benchmark._ConditioningBudget(flops(case) * benchmark._bounded_operation_count(plan["timing"]))
        begin = time.monotonic()
        chunks = 0
        maximum = plan["study"]["diagnostics"].get("maximum_warmup_chunks", 4096)
        while chunks < preparation["matched_minimum_complete_chunks"] or time.monotonic() - begin < preparation["matched_minimum_warmup_seconds"]:
            benchmark._check_deadline(deadline)
            if chunks >= maximum:
                raise LabError("Eager diagnostic warmup hit the declared chunk cap")
            calls = plan["timing"]["iterations"]
            budget.charge(calls, calls * flops(case))
            for _ in range(calls):
                operation()
            torch.cuda.synchronize(selected)
            chunks += 1
        elapsed = time.monotonic() - begin
        events = benchmark._event_pair(torch)
        samples = []
        for _ in range(plan["timing"]["repeats"]):
            benchmark._check_deadline(deadline)
            samples.append(benchmark._event_sample(operation, events, plan["timing"]["iterations"]))
        after = benchmark._correctness(torch, y, reference, plan["dtype"])
        benchmark._assert_correctness(after, case["id"])
        result = {
            "schema_version": 1, "case_id": case["id"],
            **{key: case[key] for key in ("m", "n", "k", "split")},
            "dtype": plan["dtype"], "cache_policy": plan["cache_policy"],
            "plan_hash": digest(plan), "environment_id": environment["environment_id"],
            "measurement_kind": "cuda_measured", "status": "ok",
            "median_ms": statistics.median(samples), "samples_ms": samples,
            "flops": flops(case), "ideal_bytes": ideal_bytes(case, plan["dtype"]),
            "correctness": after, "correctness_before": before,
            "timing": copy.deepcopy(plan["timing"]), "execution_mode": "eager",
            "timer_scope": benchmark._execution_metadata(plan["timing"])["timer_scope"],
            "warmup": {
                "policy": "fixed_duration_completed_eager_chunks",
                "completed_chunks": chunks, "actual_wall_seconds": elapsed,
                "requested_min_seconds": preparation["matched_minimum_warmup_seconds"],
                "sample_quality_used_for_stopping": False,
            },
            "conditioning_budget": budget.metadata(), "memory_guard": memory_guard,
            "measured_at_utc": utc_now(), "session_id": environment["session_id"],
            "visit_started_monotonic_seconds": started,
            "visit_finished_monotonic_seconds": time.monotonic(),
            "visit_started_at_utc": utc, "visit_finished_at_utc": utc_now(),
        }
        validate_measurements([result], plan)
        return result


def profile_graph(plan, case, source_environment, output_dir, *, device, deadline):
    """Capture a genuine warmed CUDA Graph replay in a separate Torch trace."""
    started, started_utc = time.monotonic(), utc_now()
    out = Path(output_dir)
    if out.exists():
        raise LabError("Graph profile output already exists")
    out.mkdir(parents=True)
    record = {
        "schema_version": 1, "measurement_kind": "profile_capture",
        "status": "running", "visit_started_at_utc": started_utc,
        "visit_started_monotonic_seconds": started,
        "session_id": source_environment.get("session_id"),
        "case_id": case["id"], "dtype": plan["dtype"], "plan_hash": digest(plan),
        "source_environment_id": source_environment["environment_id"],
        "source_measurement_kind": source_environment["measurement_kind"],
        "execution_mode": "cuda_graph", "excluded_from_model": True,
        "timing_excluded_from_model": True,
        "captured_gemm_calls": plan["timing"]["iterations"],
        "hardware_counters": {
            "status": "not_collected",
            "reason": "Torch timeline is not a DRAM-bandwidth/SM-occupancy counter measurement.",
            "ncu_executable_found": bool(shutil.which("ncu")),
            "nsys_executable_found": bool(shutil.which("nsys")),
            "tool_presence_does_not_establish_counter_permissions": True,
        },
        "notes": [
            "Profiler activity can perturb timing and replay/cache behavior.",
            "No profiler duration is substituted for a normal CUDA-event label.",
            "Graph replay does not fuse captured GEMMs into a single kernel.",
        ],
    }
    if source_environment["measurement_kind"] == "simulated":
        record.update(status="unavailable", reason="simulation_no_GPU_profile", torch_trace_path=None,
                      visit_finished_at_utc=utc_now(), visit_finished_monotonic_seconds=time.monotonic())
        write_json(out / "profile.json", record)
        return record
    write_json(out / "profile.json", record)
    torch = benchmark._load_torch()
    selected = benchmark._require_cuda(torch, plan["dtype"], device)
    benchmark._validate_work_budget(plan, [case])
    trace = out / "graph_trace.json"
    with torch.cuda.device(selected), benchmark._precision_policy(torch) as policy, \
            torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        environment = benchmark._collect_environment(torch, plan, selected, policy)
        if environment["environment_id"] != source_environment["environment_id"] or environment["identity"] != source_environment["identity"]:
            raise LabError("Graph profiler environment changed")
        benchmark._check_deadline(deadline)
        guard = benchmark._check_memory(torch, selected, case, plan["dtype"])
        x, w, y = benchmark._allocate_inputs(torch, selected, case, plan["dtype"], plan["seed"])
        reference = torch.mm(x.float(), w.float())
        operation = lambda: torch.mm(x, w, out=y)
        operation()
        before = benchmark._correctness(torch, y, reference, plan["dtype"])
        benchmark._assert_correctness(before, case["id"])
        graph = benchmark._capture_operation_graph(torch, selected, operation, plan["timing"], deadline)
        y.fill_(float("nan"))
        budget = benchmark._ConditioningBudget(flops(case) * benchmark._bounded_operation_count(plan["timing"]))
        warmup = benchmark._warm_graph(
            torch, selected, graph, plan["timing"], deadline,
            budget=budget, flops_per_operation=flops(case),
        )
        record.update(correctness_before=before, memory_guard=guard, warmup=warmup,
                      conditioning_budget=budget.metadata())
        write_json(out / "profile.json", record)
        try:
            benchmark._check_deadline(deadline)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, profile_memory=False, with_stack=False,
            ) as profiler:
                with torch.profiler.record_function("linear_perf_graph_replay"):
                    graph.replay()
                    torch.cuda.synchronize(selected)
            profiler.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text(encoding="utf-8")).get("traceEvents", [])
            kernels = [
                event for event in events
                if isinstance(event, dict) and "kernel" in str(event.get("cat", "")).lower()
                and event.get("ph") == "X"
            ]
            record.update(
                status="ok" if kernels else "unavailable",
                reason=None if kernels else "trace_has_no_CUDA_kernel_activity",
                torch_trace_path=trace.name, cuda_kernel_event_count=len(kernels),
                kernel_names=sorted({str(event.get("name", "")) for event in kernels}),
                profiler_kind="torch_profiler_CPU_CUDA_timeline",
            )
        except (RuntimeError, ImportError, AttributeError, OSError, ValueError) as exc:
            # A capability failure is retained. A subsequent CUDA sync and
            # numerical check must still succeed; actual device failures are
            # not silently reclassified as harmless profiler limitations.
            record.update(
                status="unavailable", reason="profiler_capture_unavailable",
                error_type=type(exc).__name__, error=str(exc),
                torch_trace_path=trace.name if trace.exists() else None,
            )
        torch.cuda.synchronize(selected)
        after = benchmark._correctness(torch, y, reference, plan["dtype"])
        benchmark._assert_correctness(after, case["id"])
        benchmark._check_deadline(deadline)
        record.update(
            correctness_before=before, correctness_after=after, memory_guard=guard,
            warmup=warmup, conditioning_budget=budget.metadata(), captured_at_utc=utc_now(),
            visit_finished_at_utc=utc_now(), visit_finished_monotonic_seconds=time.monotonic(),
        )
    write_json(out / "profile.json", record)
    return record


def _rotation_visit(plan, case, environment, buffers, *, device, deadline, token):
    if type(buffers) is not int or buffers not in {1, 4}:
        raise LabError("Registered rotation diagnostic permits one or four buffer sets")
    if environment["measurement_kind"] == "simulated":
        raw = _simulate_visit(plan, case, environment, token={"rotation": buffers, **token})
        raw.update(
            measurement_kind="simulated_diagnostic", excluded_from_model=True,
            diagnostic_kind="working_set_rotation", buffer_sets=buffers,
            declared_working_set_ideal_bytes=buffers * ideal_bytes(case, plan["dtype"]),
            cache_interpretation="Synthetic contract only; no cache or GPU was exercised.",
        )
        return raw
    torch = benchmark._load_torch()
    selected = benchmark._require_cuda(torch, plan["dtype"], device)
    benchmark._validate_work_budget(plan, [case])
    with torch.cuda.device(selected), benchmark._precision_policy(torch) as policy, \
            torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        current = benchmark._collect_environment(torch, plan, selected, policy)
        if current["environment_id"] != environment["environment_id"] or current["identity"] != environment["identity"]:
            raise LabError("Rotation diagnostic environment changed")
        benchmark._check_deadline(deadline)
        free, total = torch.cuda.mem_get_info(selected)
        reusable = max(0, torch.cuda.memory_reserved(selected) - torch.cuda.memory_allocated(selected))
        available = min(int(total), int(free) + int(reusable))
        estimate = buffers * benchmark._estimated_peak_bytes(case, plan["dtype"])
        if estimate > benchmark.MEMORY_SAFETY_FRACTION * available:
            raise LabError("Rotating working set exceeds the declared conservative memory guard")
        inputs, references = [], []
        for index in range(buffers):
            x, w, y = benchmark._allocate_inputs(torch, selected, case, plan["dtype"], plan["seed"] + index)
            inputs.append((x, w, y))
            references.append(torch.mm(x.float(), w.float()))
        before = []
        for (x, w, y), reference in zip(inputs, references):
            torch.mm(x, w, out=y)
            check = benchmark._correctness(torch, y, reference, plan["dtype"])
            benchmark._assert_correctness(check, case["id"])
            before.append(check)
        pointer = [0]

        def operation():
            x, w, y = inputs[pointer[0] % buffers]
            pointer[0] += 1
            torch.mm(x, w, out=y)

        # The recorded graph cycles through fixed preallocated sets. Each replay
        # reuses the same graph order; this is not fresh allocation per sample.
        graph = benchmark._capture_operation_graph(torch, selected, operation, plan["timing"], deadline)
        for _, _, output in inputs:
            output.fill_(float("nan"))
        budget = benchmark._ConditioningBudget(
            flops(case) * (benchmark._bounded_operation_count(plan["timing"]) + 2 * buffers)
        )
        warmup = benchmark._warm_graph(
            torch, selected, graph, plan["timing"], deadline,
            budget=budget, flops_per_operation=flops(case),
        )
        events = benchmark._event_pair(torch)
        samples = []
        for _ in range(plan["timing"]["repeats"]):
            benchmark._check_deadline(deadline)
            samples.append(benchmark._event_sample(graph.replay, events, 1) / plan["timing"]["iterations"])
        after = []
        for (_, _, output), reference in zip(inputs, references):
            check = benchmark._correctness(torch, output, reference, plan["dtype"])
            benchmark._assert_correctness(check, case["id"])
            after.append(check)
        return {
            "schema_version": 1, "measurement_kind": "cuda_diagnostic",
            "source_measurement_kind": "cuda_measured", "status": "ok",
            "case_id": case["id"], **{key: case[key] for key in ("m", "n", "k", "split")},
            "dtype": plan["dtype"], "source_plan_hash": digest(plan),
            "source_environment_id": environment["environment_id"],
            "excluded_from_model": True, "diagnostic_kind": "working_set_rotation",
            "buffer_sets": buffers, "declared_working_set_ideal_bytes": buffers * ideal_bytes(case, plan["dtype"]),
            "median_ms": statistics.median(samples), "samples_ms": samples,
            "timing": copy.deepcopy(plan["timing"]),
            "timer_scope": "one_fixed_rotating_buffer_graph_replay_divided_by_captured_GEMMs",
            "cache_interpretation": "Repeated rotating preallocated sets; no proof of cold caches, eviction or actual DRAM traffic.",
            "memory_guard": {"estimated_peak_bytes": estimate, "available_bytes": available,
                             "safety_fraction": benchmark.MEMORY_SAFETY_FRACTION},
            "correctness_before": before, "correctness_after": after,
            "warmup": warmup, "conditioning_budget": budget.metadata(),
            "measured_at_utc": utc_now(),
        }


def _sensitivity_summary(records):
    buckets = {}
    for record in records:
        key = (record["case_id"], record["execution_mode"], record["timing"]["iterations"])
        buckets.setdefault(key, []).append(record["median_ms"])
    summaries = [
        {"case_id": key[0], "mode": key[1], "iterations": key[2],
         "visit_medians_ms": values, "median_ms": statistics.median(values),
         "max_over_min_visit_median": max(values) / min(values)}
        for key, values in sorted(buckets.items())
    ]
    sensitivities = []
    for case_id, mode in sorted({(x["case_id"], x["mode"]) for x in summaries}):
        values = [x for x in summaries if x["case_id"] == case_id and x["mode"] == mode]
        latencies = [x["median_ms"] for x in values]
        sensitivities.append({
            "case_id": case_id, "mode": mode,
            "max_over_min_across_iteration_counts": max(latencies) / min(latencies),
            "iterations": [x["iterations"] for x in values],
            "per_call_medians_ms": latencies,
        })
    return {
        "groups": summaries, "iteration_sensitivity": sensitivities,
        "interpretation": "Descriptive diagnostic values; no automatic fastest-setting selection or correction of main labels.",
    }


def run_diagnostics(plan, environment, output_dir, *, session_id,
                    device="cuda:0", deadline_monotonic=None, progress=print):
    _validate(plan)
    _session_id(session_id)
    _check_environment(plan, environment, session_id)
    deadline = benchmark._session_deadline(plan, deadline_monotonic)
    benchmark._check_deadline(deadline)
    out = Path(output_dir).resolve()
    if out.exists():
        raise LabError("Diagnostic output already exists; refusing automatic retries")
    out.mkdir(parents=True)
    state_path = out / "diagnostics.json"
    diag = plan["study"]["diagnostics"]
    lookup = case_lookup(plan)
    variants = {
        f"{mode}_{iterations}": _diagnostic_plan(plan, mode, iterations)
        for mode in ("eager", "cuda_graph") for iterations in diag["iterations"]
    }
    case_order = list(plan["study"]["diagnostic_case_ids"])
    random.Random(plan["seed"]).shuffle(case_order)
    variant_order = {}
    for case_id in case_order:
        keys = list(variants)
        seed = int(digest({"seed": plan["seed"], "case_id": case_id, "purpose": "sensitivity_order"})[:16], 16)
        random.Random(seed).shuffle(keys)
        variant_order[case_id] = keys
    schedule = []
    for block in range(diag["passes"]):
        order = case_order if block % 2 == 0 else list(reversed(case_order))
        for case_id in order:
            keys = variant_order[case_id] if block % 2 == 0 else list(reversed(variant_order[case_id]))
            schedule.extend({"pass_index": block + 1, "case_id": case_id, "variant": key} for key in keys)
    for index, task in enumerate(schedule, 1):
        task["visit_index"] = index
    rotation_schedule = []
    for block in range(diag["passes"]):
        case_ids = diag["rotation_case_ids"] if block % 2 == 0 else list(reversed(diag["rotation_case_ids"]))
        counts = diag["rotation_buffer_counts"] if block % 2 == 0 else list(reversed(diag["rotation_buffer_counts"]))
        rotation_schedule.extend(
            {"pass_index": block + 1, "case_id": case_id, "buffer_sets": count}
            for case_id in case_ids for count in counts
        )
    state = {
        "schema_version": 1, "protocol_version": "1.0.0", "status": "running",
        "session_id": session_id, "source_plan_hash": digest(plan),
        "source_environment_id": environment["environment_id"],
        "source_measurement_kind": environment["measurement_kind"],
        "measurement_kind": "diagnostic_bundle", "excluded_from_model": True,
        "schedule": schedule, "rotation_schedule": rotation_schedule,
        "schedule_declared_before_measurement": True,
        "variants": {
            key: {"plan": value, "plan_hash": digest(value)}
            for key, value in variants.items()
        },
        "records": [], "profiles": [], "rotation_records": [],
        "started_at_utc": utc_now(), "automatic_retries": 0,
        "main_timing_policy_changed_from_diagnostic_results": False,
    }
    telemetry = TelemetryRecorder(
        out / "telemetry.jsonl", interval_seconds=plan["study"]["telemetry_interval_seconds"],
        simulate=environment["measurement_kind"] == "simulated", device=device, deadline=deadline,
    )

    def persist():
        state["records_hash"] = digest(state["records"])
        state["rotation_records_hash"] = digest(state["rotation_records"])
        state["schedule_hash"] = digest(state["schedule"])
        state["telemetry"] = telemetry.summary()
        write_json(state_path, state)

    persist()
    active = None
    try:
        telemetry.sample("diagnostics_start", force=True)
        envs = {key: _diagnostic_environment(value, environment, device) for key, value in variants.items()}
        state["variant_environments"] = envs
        persist()
        for task in schedule:
            active = task
            benchmark._check_deadline(deadline)
            key = task["variant"]
            derived = variants[key]
            case = lookup[task["case_id"]]
            telemetry.sample("before_sensitivity", task["visit_index"])
            if environment["measurement_kind"] == "cuda_measured" and derived["timing"]["mode"] == "eager":
                record = _eager_matched_visit(derived, case, envs[key], device=device, deadline=deadline)
            else:
                record = _visit(
                    derived, case, envs[key], device=device, deadline=deadline, token=task,
                )
            record.update(
                source_measurement_kind=record["measurement_kind"],
                measurement_kind="simulated_diagnostic" if environment["measurement_kind"] == "simulated" else "cuda_diagnostic",
                excluded_from_model=True, diagnostic_kind="iteration_mode_sensitivity",
                source_plan_hash=digest(plan), source_environment_id=environment["environment_id"],
                diagnostic_contract=derived["diagnostic_contract"], **task,
            )
            state["records"].append(record)
            persist()
            telemetry.sample("after_sensitivity", task["visit_index"])
            if progress is not None and len(state["records"]) % 12 == 0:
                progress(f"Diagnostics: {len(state['records'])}/{len(schedule)} fixed sensitivity visits retained")
        state["sensitivity"] = _sensitivity_summary(state["records"])
        persist()
        for case_id in diag["profile_case_ids"]:
            active = {"stage": "graph_profile", "case_id": case_id}
            benchmark._check_deadline(deadline)
            record = profile_graph(
                plan, lookup[case_id], environment, out / "profiles" / case_id,
                device=device, deadline=deadline,
            )
            record["artifact_directory"] = "profiles/" + case_id
            state["profiles"].append(record)
            persist()
        for index, task in enumerate(rotation_schedule, 1):
            active = {"stage": "working_set_rotation", **task}
            benchmark._check_deadline(deadline)
            telemetry.sample("before_rotation", index)
            started, started_utc = time.monotonic(), utc_now()
            record = _rotation_visit(
                plan, lookup[task["case_id"]], environment, task["buffer_sets"],
                device=device, deadline=deadline, token=task,
            )
            record.update(
                **task, session_id=session_id,
                visit_started_monotonic_seconds=started,
                visit_finished_monotonic_seconds=time.monotonic(),
                visit_started_at_utc=started_utc, visit_finished_at_utc=utc_now(),
            )
            state["rotation_records"].append(record)
            persist()
        active = {"stage": "numerical_witness"}
        state["numerical_witnesses"] = collect_numerical_witnesses(
            plan, environment, out / "numerical_witnesses",
            device=device, deadline_monotonic=deadline,
        )
        state.update(status="completed", completed_at_utc=utc_now())
        telemetry.sample("diagnostics_end")
        persist()
        return state
    except Exception as exc:
        state.update(
            status="failed", failed_at_utc=utc_now(), failed_task=active,
            error_type=type(exc).__name__, error=str(exc),
        )
        persist()
        raise MeasurementError(
            "Diagnostics stopped; all completed evidence is retained: " + str(exc),
            raw_rows=copy.deepcopy(state["records"]), environment=environment,
            raw_path=str(state_path), telemetry_path=str(telemetry.path),
        ) from exc
    finally:
        telemetry.close()
