"""Replicated collection with explicit aggregation and retained raw evidence.

Every case is visited once in every predeclared shuffled pass. A model row's
samples are pass medians, never mislabeled individual CUDA event observations.
Raw block rows use a distinct kind and cannot enter the model fitter directly.
"""

from __future__ import annotations

import copy
import math
import random
import statistics
import time
from pathlib import Path
from typing import Callable

from .common import (SCHEMA_VERSION, LabError, case_lookup, digest, flops, ideal_bytes,
                     positive_number, utc_now, validate_plan, write_json)


_MATCH_FIELDS = (
    "schema_version", "case_id", "m", "n", "k", "split", "dtype",
    "cache_policy", "plan_hash", "environment_id", "flops", "ideal_bytes",
    "execution_mode", "timer_scope", "timing",
)


class ReplicatedCollectionError(LabError):
    """Only complete aggregate rows appear in completed_rows on failure."""

    def __init__(self, message: str, *, completed_rows: list[dict], raw_rows: list[dict],
                 raw_evidence_path: str | None, environment: dict,
                 failed_case_id: str | None = None):
        super().__init__(message)
        self.completed_rows = completed_rows
        self.raw_completed_rows = raw_rows
        self.raw_evidence_path = raw_evidence_path
        self.environment = environment
        self.failed_case_id = failed_case_id


def _repeat_median(row: dict, *, expected_repeats: int | None = None) -> float:
    samples = row.get("samples_ms")
    if not isinstance(samples, list) or len(samples) < 3:
        raise LabError("A raw block requires at least three repeat means")
    if expected_repeats is not None and len(samples) != expected_repeats:
        raise LabError("Raw block repeat count differs from the registered plan")
    values = [positive_number(x, "raw repeat mean") for x in samples]
    actual = positive_number(row.get("median_ms"), "raw block median")
    median = statistics.median(values)
    if not math.isclose(actual, median, rel_tol=1e-8, abs_tol=1e-12):
        raise LabError("Raw block median does not match all retained repeat means")
    return median


def _checked_correctness(value: dict) -> None:
    if not isinstance(value, dict) or value.get("passed") is not True or value.get("finite") is not True:
        raise LabError("Every replicated block must pass finite numerical checks")
    for key in ("relative_l2_error", "max_abs_error"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise LabError("Numerical error summaries must be finite and nonnegative")
    limits = value.get("limits")
    if limits is not None:
        if not isinstance(limits, dict):
            raise LabError("Malformed numerical limits in a replicated block")
        threshold = positive_number(limits.get("relative_l2"), "relative L2 limit")
        if value["relative_l2_error"] > threshold:
            raise LabError("Replicated block exceeds its relative L2 limit")
    if "max_abs_error_limit" in value:
        limit = positive_number(value["max_abs_error_limit"], "absolute error limit")
        if value["max_abs_error"] > limit:
            raise LabError("Replicated block exceeds its absolute error limit")


def _correctness_summary(blocks: list[dict]) -> dict:
    checks = [block["correctness"] for block in blocks]
    for check in checks:
        _checked_correctness(check)
    first = checks[0]
    for check in checks[1:]:
        for key in ("reference", "limits", "check_kind"):
            if check.get(key) != first.get(key):
                raise LabError("Replicated blocks use different correctness contracts")
    summary = copy.deepcopy(first)
    summary.update(passed=True, finite=True, checked_blocks=len(checks),
                   aggregation="all blocks passed; maximum recorded errors retained")
    for key in ("relative_l2_error", "max_abs_error", "reference_max_abs"):
        if key in first:
            values = [check.get(key) for check in checks]
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
                raise LabError("Malformed error/reference summary across blocks")
            summary[key] = max(values)
    if "max_abs_error_limit" in first:
        summary["max_abs_error_limit"] = min(check["max_abs_error_limit"] for check in checks)
    return summary


def validate_aggregate_row(row: dict, expected_blocks: int | None = None,
                           expected_repeats: int | None = None) -> None:
    """Validate the full raw-to-aggregate chain without importing CUDA.

    This checks internal consistency, not independent hardware attestation.
    The caller separately authenticates the case against its registered plan.
    """
    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
        raise LabError("Unsupported aggregate row schema")
    if row.get("sample_unit") != "block_median" or row.get("aggregation") != "median_of_block_medians":
        raise LabError("Replicated model rows must declare block-median aggregation")
    source_kind = row.get("measurement_kind")
    if source_kind not in {"cuda_measured", "simulated"}:
        raise LabError("An aggregate must retain its measured/simulated source kind")
    blocks = row.get("measurement_blocks")
    if not isinstance(blocks, list) or len(blocks) < 3:
        raise LabError("An aggregate requires at least three complete raw blocks")
    if expected_blocks is not None and len(blocks) != expected_blocks:
        raise LabError("Aggregate block count differs from the registered plan")
    try:
        observed_hash = digest(blocks)
    except (ValueError, TypeError) as exc:
        raise LabError("Raw-block evidence must be finite serializable JSON") from exc
    if row.get("measurement_blocks_hash") != observed_hash:
        raise LabError("Aggregate raw-block evidence hash mismatch")
    repeats = row.get("raw_repeat_count")
    if type(repeats) is not int or repeats < 3:
        raise LabError("Aggregate must declare the raw repeat count per block")
    if expected_repeats is not None and repeats != expected_repeats:
        raise LabError("Aggregate raw-repeat count differs from the registered plan")
    if isinstance(row.get("timing"), dict) and row["timing"].get("repeats") != repeats:
        raise LabError("Aggregate raw-repeat count differs from timing metadata")
    medians = []
    for index, block in enumerate(blocks, 1):
        if not isinstance(block, dict) or block.get("block_index") != index or type(block.get("block_index")) is not int:
            raise LabError("Raw blocks must have unique consecutive ordered block indices")
        if (block.get("measurement_kind") != source_kind + "_block"
                or block.get("source_measurement_kind") != source_kind
                or block.get("excluded_from_model") is not True
                or block.get("sample_unit") != "repeat_mean"):
            raise LabError("Raw block kind/source/sample-unit exclusion metadata is invalid")
        if block.get("aggregation") not in (None, "none") or "measurement_blocks" in block:
            raise LabError("Raw blocks cannot contain nested aggregate evidence")
        for field in _MATCH_FIELDS:
            if block.get(field) != row.get(field):
                raise LabError("Replicated block metadata mismatch: " + field)
        if block.get("status") != "ok":
            raise LabError("Failed raw block cannot enter an aggregate")
        medians.append(_repeat_median(block, expected_repeats=repeats))
        _checked_correctness(block.get("correctness"))
        if block.get("flops") != flops(block) or block.get("ideal_bytes") != ideal_bytes(block, block["dtype"]):
            raise LabError("Replicated block operation accounting is inconsistent")
    samples = row.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != len(medians):
        raise LabError("Aggregate samples must contain every block median exactly once")
    if any(not math.isclose(positive_number(a, "aggregate sample"), b, rel_tol=1e-8, abs_tol=1e-12)
           for a, b in zip(samples, medians)):
        raise LabError("Aggregate samples differ from the raw block medians")
    if not math.isclose(positive_number(row.get("median_ms"), "aggregate median"), statistics.median(medians),
                        rel_tol=1e-8, abs_tol=1e-12):
        raise LabError("Aggregate target is not the median of every block median")
    if row.get("status") != "ok" or row.get("correctness") != _correctness_summary(blocks):
        raise LabError("Aggregate correctness summary does not match all raw blocks")


def _aggregate(blocks: list[dict], *, expected_blocks: int) -> dict:
    ordered = sorted(copy.deepcopy(blocks), key=lambda row: row["block_index"])
    row = copy.deepcopy(ordered[0])
    source_kind = row.pop("source_measurement_kind")
    for key in ("block_index", "excluded_from_model", "case_position", "block_order_seed",
                "warmup", "conditioning_budget", "memory_guard"):
        row.pop(key, None)
    row.update(measurement_kind=source_kind, sample_unit="block_median",
               aggregation="median_of_block_medians", measurement_blocks=ordered,
               measurement_blocks_hash=digest(ordered), raw_repeat_count=len(ordered[0]["samples_ms"]),
               samples_ms=[block["median_ms"] for block in ordered],
               median_ms=statistics.median(block["median_ms"] for block in ordered),
               correctness=_correctness_summary(ordered), aggregated_at_utc=utc_now(),
               per_visit_resource_metadata_location="measurement_blocks")
    if "measured_at_utc" in ordered[-1]:
        row["measured_at_utc"] = ordered[-1]["measured_at_utc"]
    validate_aggregate_row(row, expected_blocks=expected_blocks)
    return row


def replica_dispersion(row: dict) -> dict:
    """Descriptive variability only; no confidence or prediction interval."""
    validate_aggregate_row(row)
    medians = row["samples_ms"]
    within = [100 * statistics.stdev(block["samples_ms"]) / statistics.mean(block["samples_ms"])
              for block in row["measurement_blocks"]]
    return {
        "blocks": len(medians), "raw_repeat_means_per_block": row["raw_repeat_count"],
        "block_medians_ms": list(medians),
        "block_median_sample_cv_pct": 100 * statistics.stdev(medians) / statistics.mean(medians),
        "max_over_min_block_median": max(medians) / min(medians),
        "within_block_sample_cv_pct": within,
        "max_within_block_sample_cv_pct": max(within),
        "observed_block_median_min_ms": min(medians), "observed_block_median_max_ms": max(medians),
        "absolute_block_median_range_us": (max(medians) - min(medians)) * 1000,
        "interpretation": "Descriptive observed variation; not a confidence or prediction interval. Blocks may be temporally dependent.",
    }


def _block_row(raw: dict, *, plan: dict, case: dict, env: dict,
               block_index: int, position: int, order_seed: int) -> dict:
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise LabError("Raw collector returned an unsupported measurement schema")
    row = copy.deepcopy(raw)
    kind = env["measurement_kind"]
    if row.get("measurement_kind") != kind or kind not in {"cuda_measured", "simulated"}:
        raise LabError("Unexpected measured/simulated kind returned by a block collector")
    if row.get("sample_unit", "repeat_mean") != "repeat_mean" or "measurement_blocks" in row:
        raise LabError("Collector returned an aggregate instead of raw repeat means")
    # The common validator authenticates shape, split, plan hash, precision and
    # actual CUDA mode/scope before we change the raw row's evidence kind.
    # Its aggregate-specific path is not entered for these repeat-mean rows.
    from .common import validate_measurements
    validate_measurements([row], plan)
    for field in ("m", "n", "k", "split"):
        if row.get(field) != case[field]:
            raise LabError("Raw collector case metadata differs from its registered shape")
    if row.get("case_id") != case["id"] or row.get("status") != "ok":
        raise LabError("Raw collector did not return the requested successful case")
    if (row.get("plan_hash") != digest(plan) or row.get("environment_id") != env["environment_id"]
            or row.get("dtype") != plan["dtype"] or row.get("cache_policy") != plan["cache_policy"]):
        raise LabError("Raw collector environment or protocol metadata differs")
    _repeat_median(row, expected_repeats=plan["timing"]["repeats"])
    _checked_correctness(row.get("correctness"))
    if row.get("flops") != flops(case) or row.get("ideal_bytes") != ideal_bytes(case, plan["dtype"]):
        raise LabError("Raw collector operation accounting differs from its shape")
    if kind == "cuda_measured" and row.get("timing") != plan["timing"]:
        raise LabError("Raw CUDA timing metadata differs from the registered plan")
    row.setdefault("timing", copy.deepcopy(plan["timing"]))
    row.update(measurement_kind=kind + "_block", source_measurement_kind=kind,
               excluded_from_model=True, sample_unit="repeat_mean", block_index=block_index,
               case_position=position, block_order_seed=order_seed)
    return row


def collect_replicated(plan: dict, cases: list[dict], env: dict, device: str,
                       *, deadline_monotonic: float | None = None,
                       raw_output: str | Path | None = None,
                       progress: Callable[[str], None] | None = None) -> list[dict]:
    validate_plan(plan)
    collection = plan.get("collection")
    if not isinstance(collection, dict) or collection != {
        "blocks": 3, "order": "seeded_shuffle_per_block", "aggregation": "median_of_block_medians",
    }:
        raise LabError("Replicated collector requires the registered three-block protocol")
    lookup = case_lookup(plan)
    if not cases or len({case.get("id") for case in cases}) != len(cases):
        raise LabError("Replicated collector requires distinct registered cases")
    if any(case.get("id") not in lookup or lookup[case["id"]] != case for case in cases):
        raise LabError("Replicated collector received an unknown or modified case")
    if env.get("measurement_kind") not in {"cuda_measured", "simulated"} or not env.get("environment_id"):
        raise LabError("Replicated collector requires an explicit source environment")
    if env.get("plan_hash") != digest(plan):
        raise LabError("Replicated collection environment plan hash mismatch")
    path = Path(raw_output).resolve() if raw_output is not None else None
    if path is not None and path.exists():
        raise LabError("Raw collection evidence already exists; refusing to overwrite it")
    orders = []
    for index in range(1, collection["blocks"] + 1):
        seed = int(digest({"seed": plan["seed"], "case_ids": sorted(case["id"] for case in cases),
                           "block_index": index, "order": collection["order"]})[:16], 16)
        order = sorted(cases, key=lambda case: case["id"])
        random.Random(seed).shuffle(order)
        orders.append({"block_index": index, "seed": seed, "case_ids": [case["id"] for case in order]})
    state = {
        "schema_version": 1, "stage": "replicated_collection_raw", "status": "running",
        "started_at_utc": utc_now(), "plan_hash": digest(plan), "environment_id": env["environment_id"],
        "source_measurement_kind": env["measurement_kind"], "excluded_from_model": True,
        "sample_unit": "repeat_mean", "collection": copy.deepcopy(collection),
        "planned_case_ids": [case["id"] for case in cases], "block_orders": orders,
        "completed_blocks": 0, "rows": [],
    }

    def persist():
        state["raw_rows_hash"] = digest(state["rows"])
        if path is not None:
            write_json(path, state)

    persist()
    current_case, block_index, position, seed = None, 0, 0, 0
    try:
        for order in orders:
            block_index, seed = order["block_index"], order["seed"]
            if progress is not None:
                progress(f"Replicated collection: block {block_index}/3, {len(cases)} cases; all raw repeats retained")
            for position, case_id in enumerate(order["case_ids"]):
                current_case = lookup[case_id]
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    raise LabError("Campaign wall-time budget exhausted during replicated collection")
                if env["measurement_kind"] == "simulated":
                    from .simulation import simulate_cases
                    rows = simulate_cases(plan, [current_case], environment=env)
                else:
                    from .benchmark import benchmark_cases
                    rows = benchmark_cases(plan, [current_case], device=device, environment=env,
                                           deadline_monotonic=deadline_monotonic)
                if not isinstance(rows, list) or len(rows) != 1:
                    raise LabError("A replicated case visit must return exactly one raw row")
                tagged = _block_row(rows[0], plan=plan, case=current_case, env=env,
                                    block_index=block_index, position=position, order_seed=seed)
                state["rows"].append(tagged)
                persist()
            state["completed_blocks"] = block_index
            persist()
        aggregates = [_aggregate([row for row in state["rows"] if row["case_id"] == case["id"]],
                                  expected_blocks=collection["blocks"]) for case in cases]
        state.update(status="completed", completed_at_utc=utc_now(),
                     aggregate_targets_hash=digest([{"case_id": row["case_id"], "median_ms": row["median_ms"],
                                                    "measurement_blocks_hash": row["measurement_blocks_hash"]}
                                                   for row in aggregates]))
        persist()
        return aggregates
    except Exception as exc:
        rejected = []
        for raw in getattr(exc, "completed_rows", None) or []:
            try:
                if current_case is None:
                    raise LabError("No active registered case for partial evidence")
                tagged = _block_row(raw, plan=plan, case=current_case, env=env,
                                    block_index=block_index, position=position, order_seed=seed)
                if any(row["case_id"] == tagged["case_id"] and row["block_index"] == block_index for row in state["rows"]):
                    raise LabError("Duplicate partial row")
                state["rows"].append(tagged)
            except Exception as partial_error:
                rejected.append({"reason": str(partial_error), "raw_repr": repr(raw)})
        if rejected:
            state["rejected_partial_evidence"] = rejected
        state.update(status="failed", failed_at_utc=utc_now(), error_type=type(exc).__name__,
                     error=str(exc), failed_case_id=current_case["id"] if current_case else None,
                     failed_block_index=block_index)
        persist()
        completed = []
        for case in cases:
            raw_blocks = [row for row in state["rows"] if row["case_id"] == case["id"]]
            if len(raw_blocks) == collection["blocks"]:
                completed.append(_aggregate(raw_blocks, expected_blocks=collection["blocks"]))
        raise ReplicatedCollectionError(
            "Replicated collection stopped; all completed raw evidence retained: " + str(exc),
            completed_rows=completed, raw_rows=copy.deepcopy(state["rows"]),
            raw_evidence_path=str(path) if path is not None else None, environment=env,
            failed_case_id=state["failed_case_id"],
        ) from exc
