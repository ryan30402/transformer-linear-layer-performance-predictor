"""Small shared contracts. No PyTorch import is needed for analysis or validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SPLITS = {"calibration", "development", "test", "challenge", "candidate"}
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
EAGER_TIMER_SCOPE = "stream_elapsed_including_possible_host_submission_gaps"
GRAPH_TIMER_SCOPE = "graph_replay_elapsed_including_graph_launch_and_inter_kernel_gaps_per_operation"


class LabError(ValueError):
    """Actionable validation or experiment error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda x: (_ for _ in ()).throw(LabError(f"Non-finite JSON constant: {x}")))


def _sync_directory(path: Path) -> None:
    """Persist a completed rename when the platform supports directory fsync."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise LabError(f"Refusing to overwrite existing evidence: {path}")
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_jsonl(path: str | Path, rows: list[dict], *, overwrite: bool = True) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise LabError(f"Refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(LabError(f"Non-finite JSON constant: {x}")))
                for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise LabError("JSONL measurements must be objects")
    return rows


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise LabError(f"{label} must be a positive finite number")
    return float(value)


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA_VERSION:
        raise LabError("Unsupported experiment plan schema")
    if plan.get("dtype") not in DTYPE_BYTES:
        raise LabError("dtype must be bfloat16, float16 or float32")
    if plan.get("cache_policy") != "warm_reuse":
        raise LabError("Only explicitly labeled warm_reuse measurements are implemented")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > 128:
        raise LabError("Plan must contain 1–128 cases")
    seen_ids, seen_shapes = set(), set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "m", "n", "k", "split"}:
            raise LabError("Each case requires exactly id, m, n, k, split")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case["id"]):
            raise LabError("Case id must be a short filesystem-safe identifier")
        if case["id"] in seen_ids:
            raise LabError("Duplicate case id")
        seen_ids.add(case["id"])
        if case["split"] not in SPLITS:
            raise LabError("Unknown data split")
        for dim in ("m", "n", "k"):
            if type(case[dim]) is not int or not 1 <= case[dim] <= 16384:
                raise LabError("Matrix dimensions must be integers in [1, 16384]")
        shape = (case["m"], case["n"], case["k"])
        if shape in seen_shapes:
            raise LabError("Duplicate matrix shapes across splits are not permitted")
        seen_shapes.add(shape)
        if ideal_bytes(case, plan["dtype"]) > 512 * 1024**2:
            raise LabError("One case exceeds the 512 MiB ideal-tensor-byte limit")
    for split in ("calibration", "development", "test"):
        if not any(case["split"] == split for case in cases):
            raise LabError(f"Plan lacks required {split} cases")
    timing = plan.get("timing", {})
    if not isinstance(timing, dict):
        raise LabError("timing must be an object")
    mode = timing.get("mode", "eager")
    if not isinstance(mode, str) or mode not in {"eager", "cuda_graph"}:
        raise LabError("timing.mode must be eager or cuda_graph")
    allowed_timing_keys = {"warmup", "iterations", "repeats", "mode"}
    if mode == "cuda_graph":
        allowed_timing_keys.add("graph_warmup_replays")
        value = timing.get("graph_warmup_replays")
        if type(value) is not int or not 1 <= value <= 20:
            raise LabError("timing.graph_warmup_replays must be an integer in [1, 20] for cuda_graph mode")
        duration_keys = {"graph_warmup_min_seconds", "graph_warmup_max_replays"}
        if duration_keys & set(timing):
            if not duration_keys.issubset(timing):
                raise LabError("Duration warmup requires both duration and maximum replay count")
            allowed_timing_keys.update(duration_keys)
            duration = positive_number(timing["graph_warmup_min_seconds"], "graph_warmup_min_seconds")
            if not 0.05 <= duration <= 2.0:
                raise LabError("graph_warmup_min_seconds must be in [0.05, 2.0]")
            maximum = timing["graph_warmup_max_replays"]
            if type(maximum) is not int or not value <= maximum <= 4096:
                raise LabError("graph_warmup_max_replays must be between the minimum replay count and 4096")
    if set(timing) - allowed_timing_keys:
        raise LabError("Unknown timing settings or graph-only settings in eager mode")
    for name, lower, upper in (("warmup", 1, 100), ("iterations", 1, 500), ("repeats", 3, 30)):
        value = timing.get(name)
        if type(value) is not int or not lower <= value <= upper:
            raise LabError(f"timing.{name} must be an integer in [{lower}, {upper}]")
    if type(plan.get("seed")) is not int:
        raise LabError("seed must be an integer")
    collection = plan.get("collection")
    if collection is not None:
        if not isinstance(collection, dict) or set(collection) != {"blocks", "order", "aggregation"}:
            raise LabError("collection requires exactly blocks, order and aggregation")
        if type(collection["blocks"]) is not int or collection["blocks"] != 3:
            raise LabError("Replicated collection currently requires exactly three blocks")
        if collection["order"] != "seeded_shuffle_per_block" or collection["aggregation"] != "median_of_block_medians":
            raise LabError("Unsupported replicated collection order or aggregation")
    budget = plan.get("budget", {})
    registered = plan.get("protocol_version") == "1.0.0"
    for name, lower, upper in (("max_acquisition_rounds", 1, 12 if registered else 5),
                               ("max_new_cases", 1, 32 if registered else 8),
                               ("max_wall_seconds", 30, 3600)):
        value = budget.get(name)
        if type(value) is not int or not lower <= value <= upper:
            raise LabError(f"budget.{name} must be an integer in [{lower}, {upper}]")
    return plan


def flops(case: dict) -> int:
    return 2 * case["m"] * case["n"] * case["k"]


def ideal_bytes(case: dict, dtype: str) -> int:
    return DTYPE_BYTES[dtype] * (case["m"] * case["k"] + case["k"] * case["n"] + case["m"] * case["n"])


def case_lookup(plan: dict) -> dict[str, dict]:
    return {case["id"]: case for case in plan["cases"]}


def validate_measurements(rows: list[dict], plan: dict, *, allowed_splits: set[str] | None = None) -> None:
    lookup = case_lookup(plan)
    if not rows:
        raise LabError("No measurements were provided")
    ids, kinds, envs = set(), set(), set()
    for row in rows:
        case_id = row.get("case_id")
        if case_id not in lookup or case_id in ids:
            raise LabError("Measurements contain an unknown or duplicate case id")
        ids.add(case_id)
        case = lookup[case_id]
        for key in ("m", "n", "k", "split"):
            if row.get(key) != case[key]:
                raise LabError(f"Measurement does not match the registered plan: {case_id}.{key}")
        if allowed_splits is not None and row["split"] not in allowed_splits:
            raise LabError("A protected data split was supplied to this operation")
        if row.get("dtype") != plan["dtype"] or row.get("cache_policy") != plan["cache_policy"]:
            raise LabError("Measurement precision/cache policy differs from the registered plan")
        if row.get("plan_hash") != digest(plan):
            raise LabError("Measurement plan hash mismatch")
        if row.get("measurement_kind") not in {"cuda_measured", "simulated"}:
            raise LabError("Measurement kind must distinguish CUDA data from simulation")
        if row.get("measurement_kind") == "cuda_measured":
            mode = plan["timing"].get("mode", "eager")
            expected_scope = GRAPH_TIMER_SCOPE if mode == "cuda_graph" else EAGER_TIMER_SCOPE
            if row.get("execution_mode", "eager") != mode:
                raise LabError("Measurement execution mode differs from the registered timing plan")
            if row.get("timing") != plan["timing"] or row.get("timer_scope") != expected_scope:
                raise LabError("Measurement timing settings or timer scope differ from the registered plan")
        if row.get("status") != "ok" or row.get("correctness", {}).get("passed") is not True:
            raise LabError(f"Failed or unchecked case cannot enter modeling: {case_id}")
        if not isinstance(row.get("environment_id"), str) or not row["environment_id"]:
            raise LabError("Missing measurement environment identity")
        positive_number(row.get("median_ms"), "median_ms")
        samples = row.get("samples_ms")
        if not isinstance(samples, list) or len(samples) < 3:
            raise LabError("At least three measured repeat samples are required")
        for sample in samples:
            positive_number(sample, "sample_ms")
        expected_median = sorted(samples)[len(samples)//2] if len(samples) % 2 else sum(sorted(samples)[len(samples)//2-1:len(samples)//2+1])/2
        if not math.isclose(row["median_ms"], expected_median, rel_tol=1e-8, abs_tol=1e-12):
            raise LabError("Reported median does not match raw repeat samples")
        unit = row.get("sample_unit", "repeat_mean")
        if unit not in {"repeat_mean", "block_median"}:
            raise LabError("Unknown timing sample unit")
        if unit == "block_median":
            if not plan.get("collection"):
                raise LabError("Aggregated block medians require a registered collection plan")
            from .collection import validate_aggregate_row
            validate_aggregate_row(row, expected_blocks=plan["collection"]["blocks"])
        if row.get("flops") != flops(case) or row.get("ideal_bytes") != ideal_bytes(case, plan["dtype"]):
            raise LabError("Operation count or ideal-byte count mismatch")
        kinds.add(row["measurement_kind"])
        envs.add(row["environment_id"])
    if len(kinds) != 1 or len(envs) != 1:
        raise LabError("Cannot mix measurement kinds or environments")

