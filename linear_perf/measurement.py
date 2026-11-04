"""Repeated measurement with scheduled controls and auditable timing.

CUDA labels estimate warm-reuse graph replay time per operation. The collector
uses scheduled control visits, coarse environmental observations and immediate
evidence retention. None of these proves stable clocks. Telemetry is
sampled at visit boundaries, outside CUDA event intervals, at no more than the
declared rate. It cannot establish the state inside a microsecond event.

The explicit simulator exercises contracts only. It never imports torch,
executes a matrix multiplication or produces NVIDIA performance evidence.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

from . import benchmark
from .collection import _aggregate, _block_row, replica_dispersion
from .common import (
    LabError, case_lookup, digest, flops, ideal_bytes, utc_now,
    validate_measurements, validate_plan, write_json,
)


class MeasurementError(LabError):
    """Failure with locations of all retained observations; never a retry."""

    def __init__(self, message, *, completed_rows=None, raw_rows=None,
                 controls=None, environment=None, raw_path=None,
                 telemetry_path=None):
        super().__init__(message)
        self.completed_rows = list(completed_rows or [])
        self.raw_rows = list(raw_rows or [])
        self.controls = list(controls or [])
        self.environment = environment
        self.raw_path = raw_path
        self.telemetry_path = telemetry_path


def _validate(plan):
    from .protocol import validate_study_plan
    return validate_study_plan(plan)


def _session_id(session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
        raise LabError("session_id must be a 1–80 character filesystem-safe identifier")
    return session_id


def _check_environment(plan, environment, session_id):
    if not isinstance(environment, dict):
        raise LabError("An explicit measured or simulated environment is required")
    if environment.get("measurement_kind") not in {"cuda_measured", "simulated"}:
        raise LabError("Unsupported source measurement kind")
    if environment.get("plan_hash") != digest(plan):
        raise LabError("Environment does not match the full registered plan")
    if not isinstance(environment.get("environment_id"), str) or not environment["environment_id"]:
        raise LabError("Missing environment identity")
    identity = environment.get("identity")
    if not isinstance(identity, dict):
        raise LabError("Environment identity fields are required")
    expected_id = digest(identity)
    if environment["measurement_kind"] == "simulated":
        expected_id = "simulated:" + expected_id
    if environment["environment_id"] != expected_id:
        raise LabError("Environment identity hash is inconsistent")
    for key in ("dtype", "timing", "cache_policy", "collection"):
        if identity.get(key) != plan.get(key):
            raise LabError("Environment differs from the registered contract: " + key)
    if environment.get("session_id") != session_id:
        raise LabError("The requested session differs from the environment session")


def make_environment(plan, session_id, *, simulate=False, device="cuda:0"):
    """Authenticate the CUDA environment, or explicitly create a synthetic one.

    Real hardware identity is not varied using the session ID. The simulation
    uses a clearly non-hardware UUID per synthetic session, so transfer guards
    are exercised without attributing artificial data to any real GPU.
    """
    _validate(plan)
    _session_id(session_id)
    if simulate:
        identity = {
            "schema_version": 1,
            "runner_version": "SIMULATED_NO_GPU",
            "python_version": "synthetic-contract",
            "pytorch_version": "NOT_USED",
            "cuda_runtime_version": "NOT_USED",
            "driver_versions": [],
            "device": {
                "name": "SIMULATED_NO_GPU",
                "uuid": "SIMULATED-" + session_id,
                "index": 0,
                "total_memory_bytes": None,
            },
            "dtype": plan["dtype"],
            "cache_policy": plan["cache_policy"],
            "timing": copy.deepcopy(plan["timing"]),
            "collection": copy.deepcopy(plan["collection"]),
            "math_policy": {"reference": "synthetic_contract_no_matmul"},
            "execution": benchmark._execution_metadata(plan["timing"]),
        }
        result = {
            "schema_version": 1, "measurement_kind": "simulated",
            "environment_id": "simulated:" + digest(identity),
            "identity": identity, "plan_hash": digest(plan),
            "captured_at_utc": utc_now(),
            "limitations": [
                "SIMULATED software validation; no GPU, timings or numerical output was measured.",
                "Synthetic sessions test record handling, not statistical hardware reproducibility.",
            ],
        }
    else:
        result = benchmark.cuda_environment(plan, device)
    result["session_id"] = session_id
    result["session_metadata_is_not_hardware_identity"] = True
    return result


_TELEMETRY_FIELDS = [
    "timestamp", "name", "uuid", "pstate", "temperature.gpu",
    "power.draw", "power.limit", "clocks.sm", "clocks.mem",
    "utilization.gpu", "memory.used", "memory.total",
]


def _telemetry_snapshot(*, simulate, device, deadline):
    base = {"observed_at_utc": utc_now(), "monotonic_seconds": time.monotonic()}
    if simulate:
        return {
            **base, "status": "unavailable", "reason": "simulation_no_gpu",
            "devices": [], "hardware_state_observed": False,
        }
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            **base, "status": "unavailable", "reason": "nvidia-smi_not_installed",
            "devices": [], "hardware_state_observed": False,
        }
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        benchmark._check_deadline(deadline)
    query = [
        executable, "--query-gpu=" + ",".join(_TELEMETRY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    # Query all visible rows and retain UUID/name. CUDA_VISIBLE_DEVICES can
    # remap cuda:0, so the physical nvidia-smi row index is not guessed.
    try:
        call = subprocess.run(
            query, capture_output=True, text=True, check=False,
            timeout=max(0.01, min(2.0, remaining)),
        )
        if call.returncode:
            return {
                **base, "status": "unavailable",
                "reason": "nvidia-smi_query_failed",
                "returncode": call.returncode, "error": call.stderr.strip()[:2000],
                "devices": [], "hardware_state_observed": False,
            }
        records = []
        for row in csv.reader(io.StringIO(call.stdout)):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) != len(_TELEMETRY_FIELDS):
                records.append({"unparsed_row": row})
            else:
                records.append({k: v.strip() for k, v in zip(_TELEMETRY_FIELDS, row)})
        return {
            **base, "status": "observed" if records else "unavailable",
            "reason": None if records else "empty_nvidia-smi_output",
            "requested_cuda_device": device,
            "devices": records, "hardware_state_observed": bool(records),
            "matching_policy": "Match device UUID to authenticated CUDA environment; do not assume physical row 0.",
            "units": {
                "temperature.gpu": "C", "power.draw": "W", "power.limit": "W",
                "clocks.sm": "MHz", "clocks.mem": "MHz",
                "utilization.gpu": "percent", "memory.used": "MiB", "memory.total": "MiB",
            },
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            **base, "status": "unavailable", "reason": type(exc).__name__,
            "error": str(exc)[:2000], "devices": [], "hardware_state_observed": False,
        }


class TelemetryRecorder:
    """Bounded-rate snapshots between visits, never in an event interval."""

    def __init__(self, path, *, interval_seconds, simulate, device, deadline):
        if not isinstance(interval_seconds, (int, float)) or isinstance(interval_seconds, bool):
            raise LabError("Telemetry interval must be numeric")
        if not math.isfinite(interval_seconds) or interval_seconds < 0.5:
            raise LabError("Telemetry must be low-frequency (interval >= 0.5 s)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        self.interval = float(interval_seconds)
        self.simulate = simulate
        self.device = device
        self.deadline = deadline
        self.last_sample = None
        self.count = 0
        self.observed_count = 0
        self.closed = False

    def sample(self, boundary, visit_index=None, *, force=False):
        now = time.monotonic()
        # "force" is intended for the first observation only. It never defeats
        # the low-frequency bound for subsequent requests.
        if self.last_sample is not None and now - self.last_sample < self.interval:
            return None
        record = _telemetry_snapshot(
            simulate=self.simulate, device=self.device, deadline=self.deadline,
        )
        record.update(
            boundary=boundary, visit_index=visit_index,
            observation_scope="case_boundary_outside_CUDA_event_intervals",
            interpretation="Coarse state snapshot; not clocks or utilization inside each timed event.",
        )
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.last_sample = time.monotonic()
        self.count += 1
        self.observed_count += int(record["hardware_state_observed"])
        return record

    def close(self):
        if not self.closed:
            self.handle.close()
            self.closed = True

    def summary(self):
        return {
            "path": str(self.path), "count": self.count,
            "hardware_observations": self.observed_count,
            "minimum_interval_seconds": self.interval,
            "scope": "visit boundaries only; no per-event clock inference",
        }


def _simulate_visit(plan, case, environment, *, token):
    """One explicitly synthetic visit with deterministic hierarchical variation."""
    f = flops(case)
    q = ideal_bytes(case, plan["dtype"])
    # These arbitrary numbers are a test fixture, never GPU specifications.
    compute = f / (140.0 * 1e9)
    memory = q / (600.0 * 1e6)
    shape_factor = 1.0 + 0.14 * math.exp(-case["m"] / 45.0)
    if case["n"] % 16 or case["k"] % 16:
        shape_factor += 0.28
    session_seed = int(digest({"session": environment["session_id"]})[:8], 16)
    session_scale = 0.96 + (session_seed % 101) / 1000.0
    truth = (0.008 + max(compute, memory) * shape_factor) * session_scale
    seed = int(digest({
        "seed": plan["seed"], "session": environment["session_id"],
        "case_id": case["id"], "token": token,
    })[:16], 16)
    rng = random.Random(seed)
    visit_scale = 1 + rng.uniform(-0.012, 0.012)
    values = [
        truth * visit_scale * (1 + rng.uniform(-0.008, 0.008))
        for _ in range(plan["timing"]["repeats"])
    ]
    return {
        "schema_version": 1, "case_id": case["id"],
        **{key: case[key] for key in ("m", "n", "k", "split")},
        "dtype": plan["dtype"], "cache_policy": plan["cache_policy"],
        "plan_hash": digest(plan), "environment_id": environment["environment_id"],
        "measurement_kind": "simulated", "status": "ok",
        "median_ms": statistics.median(values), "samples_ms": values,
        "flops": f, "ideal_bytes": q, "measured_at_utc": utc_now(),
        "timing": copy.deepcopy(plan["timing"]),
        "execution_mode": plan["timing"].get("mode", "eager"),
        "timer_scope": benchmark._execution_metadata(plan["timing"])["timer_scope"],
        "correctness": {
            "passed": True, "finite": True, "relative_l2_error": 0.0,
            "max_abs_error": 0.0,
            "reference": "synthetic_contract_only_no_matmul_executed",
            "check_kind": "synthetic_contract",
        },
        "synthetic_fixture": True,
        "warning": "No CUDA timing or numeric matrix output has been measured.",
    }


def _raw_visit(plan, case, environment, *, device, deadline, token):
    benchmark._check_deadline(deadline)
    if environment["measurement_kind"] == "simulated":
        return _simulate_visit(plan, case, environment, token=token)
    rows = benchmark.benchmark_cases(
        plan, [case], device=device, environment=environment,
        deadline_monotonic=deadline,
    )
    if len(rows) != 1:
        raise LabError("One case visit must produce exactly one raw observation")
    return rows[0]


def _visit(plan, case, environment, *, device, deadline, token):
    started_mono, started_utc = time.monotonic(), utc_now()
    row = _raw_visit(
        plan, case, environment, device=device, deadline=deadline, token=token,
    )
    row.update(
        visit_started_monotonic_seconds=started_mono,
        visit_finished_monotonic_seconds=time.monotonic(),
        visit_started_at_utc=started_utc, visit_finished_at_utc=utc_now(),
        session_id=environment["session_id"],
    )
    validate_measurements([row], plan)
    return row


def _schedule(plan, cases, session_id):
    lookup = case_lookup(plan)
    controls = plan["study"]["control_case_ids"]
    spacing = plan["study"]["control_every_visits"]
    orders, tasks = [], []
    measured = 0
    control_sequence = 0

    def add_controls(reason, block_index):
        nonlocal control_sequence
        # Rotate the three controls so one is not always immediately adjacent
        # to the first or last measurement in each control group.
        offset = control_sequence % len(controls)
        sequence = controls[offset:] + controls[:offset]
        for case_id in sequence:
            tasks.append({
                "role": "control", "case_id": case_id, "reason": reason,
                "block_index": block_index, "control_sequence": control_sequence,
            })
        control_sequence += 1

    add_controls("stage_start", 0)
    for block_index in range(1, plan["collection"]["blocks"] + 1):
        seed = int(digest({
            "seed": plan["seed"], "session_id": session_id,
            "case_ids": sorted(case["id"] for case in cases),
            "block_index": block_index, "order": "seeded_shuffle",
        })[:16], 16)
        order = sorted(case["id"] for case in cases)
        random.Random(seed).shuffle(order)
        orders.append({"block_index": block_index, "seed": seed, "case_ids": order})
        for position, case_id in enumerate(order, 1):
            if case_id not in lookup:
                raise LabError("Unregistered scheduled case")
            tasks.append({
                "role": "measurement", "case_id": case_id, "block_index": block_index,
                "position": position, "order_seed": seed,
            })
            measured += 1
            if measured % spacing == 0:
                add_controls("periodic", block_index)
    if tasks[-1]["role"] == "control":
        for task in tasks[-len(controls):]:
            task["reason"] = "periodic_and_stage_end"
    else:
        add_controls("stage_end", plan["collection"]["blocks"])
    for index, task in enumerate(tasks, 1):
        task["visit_index"] = index
    return orders, tasks


def quality_summary(plan, rows, controls):
    limits = plan["study"]["quality"]
    per_case, flags = {}, []
    for row in rows:
        item = replica_dispersion(row)
        per_case[row["case_id"]] = item
        if item["max_within_block_sample_cv_pct"] > limits["within_visit_cv_pct"]:
            flags.append({"case_id": row["case_id"], "reason": "within_visit_cv",
                          "observed": item["max_within_block_sample_cv_pct"],
                          "limit": limits["within_visit_cv_pct"]})
        if item["max_over_min_block_median"] > limits["pass_ratio"]:
            flags.append({"case_id": row["case_id"], "reason": "between_pass_ratio",
                          "observed": item["max_over_min_block_median"],
                          "limit": limits["pass_ratio"]})
    control_summary = {}
    for case_id in plan["study"]["control_case_ids"]:
        visits = [row for row in controls if row["case_id"] == case_id]
        if not visits:
            flags.append({"case_id": case_id, "reason": "missing_registered_control"})
            continue
        values = [row["median_ms"] for row in visits]
        ratio = max(values) / min(values)
        cvs = [
            100 * statistics.stdev(row["samples_ms"]) / statistics.mean(row["samples_ms"])
            for row in visits
        ]
        control_summary[case_id] = {
            "visits": len(visits), "medians_ms": values,
            "visit_indices": [row["visit_index"] for row in visits],
            "monotonic_seconds": [row["visit_started_monotonic_seconds"] for row in visits],
            "max_over_min": ratio,
            "ratios_to_first": [value / values[0] for value in values],
            "within_visit_cv_pct": cvs,
        }
        if ratio > limits["control_ratio"]:
            flags.append({"case_id": case_id, "reason": "control_drift",
                          "observed": ratio, "limit": limits["control_ratio"]})
        if max(cvs) > limits["within_visit_cv_pct"]:
            flags.append({"case_id": case_id, "reason": "control_within_visit_cv",
                          "observed": max(cvs), "limit": limits["within_visit_cv_pct"]})
    return {
        "criteria": copy.deepcopy(limits), "per_case": per_case,
        "controls": control_summary, "flags": flags,
        "passed": not flags,
        "all_timing_criteria_passed": not flags,
        "interpretation": (
            "Prospectively declared descriptive engineering criteria, not significance tests. "
            "Retain all data; flags qualify claims and never trigger best-pass selection or retries."
        ),
    }


def collect_stage(plan, cases, environment, output_dir, *, session_id,
                  device="cuda:0", deadline_monotonic=None, progress=print):
    """Collect three passes plus prospectively interleaved diagnostic controls.

    This function does not decide whether heldout is authorized: campaign
    owns the freeze and single-use claim boundary. It does authenticate every
    requested case, disallow diagnostic cases as fitting rows, preserve all raw
    repeats and stop on numerical, provenance or resource errors.
    """
    _validate(plan)
    _session_id(session_id)
    _check_environment(plan, environment, session_id)
    benchmark._validate_requested_cases(plan, cases)
    diagnostic_ids = set(plan["study"]["diagnostic_case_ids"])
    if any(case["id"] in diagnostic_ids for case in cases):
        raise LabError("Diagnostic cases cannot become primary modeling rows")
    deadline = benchmark._session_deadline(plan, deadline_monotonic)
    benchmark._check_deadline(deadline)
    out = Path(output_dir).resolve()
    if out.exists():
        raise LabError("Collection output already exists; evidence cannot be overwritten")
    out.mkdir(parents=True)
    raw_path, telemetry_path = out / "raw_visits.json", out / "telemetry.jsonl"
    orders, tasks = _schedule(plan, cases, session_id)
    state = {
        "schema_version": 1, "protocol_version": "1.0.0",
        "status": "running", "stage": out.name, "session_id": session_id,
        "measurement_kind": environment["measurement_kind"],
        "environment_id": environment["environment_id"], "plan_hash": digest(plan),
        "started_at_utc": utc_now(), "block_orders": orders, "schedule": tasks,
        "schedule_declared_before_measurement": True,
        "planned_case_ids": [case["id"] for case in cases],
        "rows": [], "controls": [], "completed_visit_count": 0,
        "sample_unit": "repeat_mean", "raw_records_excluded_from_model": True,
        "automatic_retries": 0,
    }
    lookup = case_lookup(plan)
    telemetry = TelemetryRecorder(
        telemetry_path, interval_seconds=plan["study"]["telemetry_interval_seconds"],
        simulate=environment["measurement_kind"] == "simulated",
        device=device, deadline=deadline,
    )

    def persist():
        state["rows_hash"] = digest(state["rows"])
        state["controls_hash"] = digest(state["controls"])
        state["schedule_hash"] = digest(state["schedule"])
        write_json(raw_path, state)

    def completed():
        result = []
        for case in cases:
            blocks = [row for row in state["rows"] if row["case_id"] == case["id"]]
            if len(blocks) == plan["collection"]["blocks"]:
                row = _aggregate(blocks, expected_blocks=plan["collection"]["blocks"])
                row["session_id"] = session_id
                row["visit_time_bounds"] = {
                    "first_started_monotonic_seconds": min(x["visit_started_monotonic_seconds"] for x in blocks),
                    "last_finished_monotonic_seconds": max(x["visit_finished_monotonic_seconds"] for x in blocks),
                }
                result.append(row)
        return result

    persist()
    active_task = None
    try:
        telemetry.sample("stage_start", force=True)
        for task in tasks:
            active_task = task
            benchmark._check_deadline(deadline)
            case = lookup[task["case_id"]]
            telemetry.sample("before_visit", task["visit_index"])
            row = _visit(
                plan, case, environment, device=device, deadline=deadline,
                token={"stage": out.name, **task},
            )
            row["visit_index"] = task["visit_index"]
            if task["role"] == "measurement":
                tagged = _block_row(
                    row, plan=plan, case=case, env=environment,
                    block_index=task["block_index"], position=task["position"],
                    order_seed=task["order_seed"],
                )
                tagged["visit_role"] = "primary_measurement"
                state["rows"].append(tagged)
            else:
                row.update(
                    source_measurement_kind=row["measurement_kind"],
                    measurement_kind="simulated_diagnostic" if environment["measurement_kind"] == "simulated" else "cuda_diagnostic",
                    excluded_from_model=True, diagnostic_kind="interleaved_control",
                    visit_role="control", control_reason=task["reason"],
                    control_sequence=task["control_sequence"],
                    associated_block_index=task["block_index"],
                )
                state["controls"].append(row)
            state["completed_visit_count"] += 1
            persist()
            telemetry.sample("after_visit", task["visit_index"])
            if progress is not None and task["role"] == "measurement" and task["position"] == len(cases):
                progress(f"{out.name}: pass {task['block_index']}/3 collected; raw visits and controls retained")
        rows = completed()
        if len(rows) != len(cases):
            raise LabError("A completed collection must include all registered case aggregates")
        validate_measurements(rows, plan)
        quality = quality_summary(plan, rows, state["controls"])
        state.update(
            status="completed", completed_at_utc=utc_now(),
            aggregate_targets_hash=digest([
                {"case_id": r["case_id"], "median_ms": r["median_ms"],
                 "measurement_blocks_hash": r["measurement_blocks_hash"]} for r in rows
            ]),
        )
        telemetry.sample("stage_end")
        state["telemetry"] = telemetry.summary()
        persist()
        write_json(out / "quality.json", quality)
        return {
            "rows": rows, "controls": copy.deepcopy(state["controls"]),
            "quality": quality, "raw_path": str(raw_path),
            "telemetry_path": str(telemetry_path), "environment": environment,
            "schedule_hash": state["schedule_hash"],
            "telemetry": telemetry.summary(),
        }
    except Exception as exc:
        state.update(
            status="failed", failed_at_utc=utc_now(),
            failed_task=active_task, error_type=type(exc).__name__, error=str(exc),
            telemetry=telemetry.summary(),
        )
        persist()
        raise MeasurementError(
            "Collection stopped; completed observations are retained: " + str(exc),
            completed_rows=completed(), raw_rows=copy.deepcopy(state["rows"]),
            controls=copy.deepcopy(state["controls"]), environment=environment,
            raw_path=str(raw_path), telemetry_path=str(telemetry_path),
        ) from exc
    finally:
        telemetry.close()


def run_diagnostics(plan, environment, output_dir, *, session_id,
                    device="cuda:0", deadline_monotonic=None, progress=print):
    """Run separately labelled sensitivity/profile/working-set diagnostics."""
    from .diagnostics import run_diagnostics as run
    return run(
        plan, environment, output_dir, session_id=session_id, device=device,
        deadline_monotonic=deadline_monotonic, progress=progress,
    )
