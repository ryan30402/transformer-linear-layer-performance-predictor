"""Predeclared shape coverage and evidence contracts.

This module imports no GPU library.  The case manifest is deterministic and
contains no measured latency. Main and acquisition cases are disjoint from the
eight shapes reserved for diagnostics. Holding out entire (N, K) families is separate from
holding out M values inside known families.

These counts and thresholds are project design choices, not vendor standards
or a statistical guarantee.  Do not silently modify a frozen plan to improve
test scores.  A changed design requires a new study/plan hash and new evaluation.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Iterable

from .common import LabError, digest, ideal_bytes, validate_plan


PROTOCOL_VERSION = "1.0.0"
PRIMARY_SPLITS = frozenset({"calibration", "development", "test", "challenge"})
MAIN_COUNTS = {"calibration": 24, "development": 12, "test": 24, "challenge": 12}
_TIMING = {
    "mode": "cuda_graph", "warmup": 20, "iterations": 50, "repeats": 9,
    "graph_warmup_replays": 5, "graph_warmup_min_seconds": 0.25,
    "graph_warmup_max_replays": 4096,
}
_COLLECTION = {
    "blocks": 3, "order": "seeded_shuffle_per_block",
    "aggregation": "median_of_block_medians",
}
_QUALITY = {"within_visit_cv_pct": 10.0, "pass_ratio": 1.2, "control_ratio": 1.2}

_SEEN_FAMILIES = (
    (1024, 768), (1536, 1024), (4096, 2048), (8192, 6144),
    (1537, 1025), (4097, 2049),
)
_CHALLENGE_FAMILIES = ((2048, 1536), (6144, 4096), (2049, 1537), (6145, 4097))
_M_BY_SPLIT = {
    "calibration": (1, 16, 128, 1024),
    "development": (8, 256),
    "test": (3, 48, 384, 1536),
    "candidate": (2, 32, 96, 768),
    "challenge": (5, 192, 1152),
}
_DIAGNOSTIC_SHAPES = (
    (1, 1024, 1024), (4, 1024, 1024), (1, 4096, 4096), (8, 4096, 4096),
    (64, 4096, 1024), (1024, 8192, 2048), (7, 1023, 1025), (1025, 8193, 2049),
)


def diagnostic_shapes() -> set[tuple[int, int, int]]:
    """Return an independent copy of the eight diagnostic-only shapes."""
    return set(_DIAGNOSTIC_SHAPES)


def _shape(case: dict) -> tuple[int, int, int]:
    return case["m"], case["n"], case["k"]


def _family(case: dict) -> str:
    return f"nk_{case['n']}_{case['k']}"


def _m_bucket(m: int) -> str:
    return "small_m_le_16" if m <= 16 else "medium_m_17_to_256" if m <= 256 else "large_m_gt_256"


def _metadata(case: dict, role: str, dtype: str) -> dict:
    family_seen = (case["n"], case["k"]) in _SEEN_FAMILIES
    if role == "diagnostic":
        coverage = "diagnostic_not_generalization"
    elif not family_seen:
        coverage = "heldout_NK_family"
    elif case["m"] > max(_M_BY_SPLIT["calibration"]):
        coverage = "M_extrapolation_in_seen_NK_family"
    else:
        coverage = "M_within_training_range_in_seen_NK_family"
    return {
        "role": role,
        "family": _family(case),
        "m_bucket": _m_bucket(case["m"]),
        "alignment": "NK_multiple_of_16" if case["n"] % 16 == 0 and case["k"] % 16 == 0 else "NK_irregular",
        "coverage": coverage,
        "NK_family_seen_in_calibration": family_seen,
        "diagnostic_shape": _shape(case) in _DIAGNOSTIC_SHAPES,
        "ideal_tensor_bytes": ideal_bytes(case, dtype),
        "workload_origin": "registered_transformer_style_dense_linear_shapes_not_a_full_model_trace",
        "alignment_interpretation": "shape feature only; does not identify a kernel tile or prove Tensor Core usage",
    }


def build_plan(dtype: str = "float16", seed: int = 20261101) -> dict:
    """Construct one fresh, latency-free FP16 or BF16 study manifest.

    A seed changes predeclared order/acquisition RNGs, not the shape membership.
    The primary FP16 and optional BF16 plans are independently hashed contracts.
    """
    if dtype not in {"float16", "bfloat16"}:
        raise LabError("Plans support separate float16 and bfloat16 registrations")
    if type(seed) is not int or not 0 <= seed < 2**31 - 10:
        raise LabError("Study seed must be an integer in [0, 2**31-10)")
    cases, metadata = [], {}
    prefixes = {"calibration": "c", "development": "d", "test": "t", "challenge": "h", "candidate": "a"}
    for split in ("calibration", "development", "test", "challenge", "candidate"):
        families = _CHALLENGE_FAMILIES if split == "challenge" else _SEEN_FAMILIES
        counter = 0
        for n, k in families:
            for m in _M_BY_SPLIT[split]:
                counter += 1
                case = {"id": f"{prefixes[split]}{counter:02d}", "m": m, "n": n, "k": k, "split": split}
                role = "candidate_pool" if split == "candidate" else "primary"
                cases.append(case)
                metadata[case["id"]] = _metadata(case, role, dtype)
    for index, (m, n, k) in enumerate(_DIAGNOSTIC_SHAPES, 1):
        case = {"id": f"x{index:02d}", "m": m, "n": n, "k": k, "split": "candidate"}
        cases.append(case)
        metadata[case["id"]] = _metadata(case, "diagnostic", dtype)
    plan = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "name": f"{dtype}_registered_study",
        "seed": seed,
        "dtype": dtype,
        "cache_policy": "warm_reuse",
        "timing": copy.deepcopy(_TIMING),
        "collection": copy.deepcopy(_COLLECTION),
        "budget": {"max_acquisition_rounds": 6, "max_new_cases": 12, "max_wall_seconds": 1800},
        "cases": cases,
        "study": {
            "study_id": f"linear-perf-{seed}-{dtype}",
            "case_metadata": metadata,
            "diagnostic_case_ids": [f"x{i:02d}" for i in range(1, 9)],
            "control_case_ids": ["x01", "x03", "x06"],
            "candidate_pool_ids": [f"a{i:02d}" for i in range(1, 25)],
            "expected_sessions": 3,
            "control_every_visits": 8,
            "telemetry_interval_seconds": 1.0,
            "quality": copy.deepcopy(_QUALITY),
            "diagnostics": {
                "iterations": [25, 50, 100], "passes": 2,
                "profile_case_ids": ["x02", "x06"],
                "rotation_case_ids": ["x02", "x03"],
                "rotation_buffer_counts": [1, 4],
            },
            "acquisition": {"budgets": [2, 4, 8, 12], "seeds": list(range(seed, seed + 10)), "batch_size": 2},
            "primary_metric": "median_ape_pct",
            "diagnostic_registry": {
                "source": "registered_diagnostic_shapes",
                "shape_order": ["m", "n", "k"],
                "shape_count": len(_DIAGNOSTIC_SHAPES),
                "sha256": digest(sorted(_DIAGNOSTIC_SHAPES)),
                "policy": "diagnostic shapes are excluded from primary and acquisition data",
            },
            "generalization_contract": {
                "ordinary_test": "new M values inside known NK families; M=1536 additionally extrapolates beyond initial calibration M",
                "challenge": "four entirely heldout NK families, two regular and two irregular; no family occurs in train/dev/candidate pool",
                "training_coverage": "six NK families including four regular and two irregular; small/medium/large M",
                "cross_session": "frozen source model and per-session refit are separate evaluations; no claim that event repeats are independent sessions",
                "cross_gpu": "explicitly declared transfer only; no fitting that mixes environment identities",
                "bf16": "separate plan and fit; never pooled with FP16 observations",
            },
            "interpretation": {
                "diagnostics_excluded_from_model": True,
                "profiler_times_excluded_from_model": True,
                "rotation_is_cache_flush": False,
                "quality_thresholds_are_vendor_standards": False,
                "accuracy_improvement_is_established": False,
                "benchmark_certification": "none; not MLPerf or vendor certification",
                "shape_metadata_contains_latency": False,
            },
        },
    }
    return validate_study_plan(plan)


def _unique_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(x, str) for x in value):
        raise LabError(f"{label} must be a nonempty list of case IDs")
    if len(set(value)) != len(value):
        raise LabError(f"{label} contains duplicate IDs")
    return value


def validate_study_plan(plan: dict) -> dict:
    """Validate the complete preregistration, including roles and holdout families.

    The base validator checks matrix bounds, dtype, distinct shapes,
    budgets, and numerical timing settings.  This additional layer prevents
    relabeling diagnostics or challenge families as training.
    """
    validate_plan(plan)
    if plan.get("protocol_version") != PROTOCOL_VERSION:
        raise LabError("Expected a 1.0.0 registered plan")
    if plan.get("dtype") not in {"float16", "bfloat16"}:
        raise LabError("Primary plans require float16 or bfloat16")
    if plan.get("timing") != _TIMING or plan.get("collection") != _COLLECTION:
        raise LabError("Primary timing/collection contract changed; create a separately declared protocol")
    if plan.get("budget") != {"max_acquisition_rounds": 6, "max_new_cases": 12, "max_wall_seconds": 1800}:
        raise LabError("The offline policy budget is six two-case rounds / twelve cases / 1800 seconds")
    if type(plan.get("seed")) is not int or not 0 <= plan["seed"] < 2**31 - 10:
        raise LabError("Invalid study seed")
    v = plan.get("study")
    if not isinstance(v, dict):
        raise LabError("Missing study settings")
    if not isinstance(v.get("study_id"), str) or not v["study_id"].strip():
        raise LabError("Missing study identity")
    cases = plan["cases"]
    if len(cases) != 104:
        raise LabError("The registered study requires 72 primary, 24 pool and 8 diagnostic cases")
    lookup = {c["id"]: c for c in cases}
    meta = v.get("case_metadata")
    if not isinstance(meta, dict) or set(meta) != set(lookup):
        raise LabError("Every registered case needs exactly one metadata entry")
    diagnostics = _unique_ids(v.get("diagnostic_case_ids"), "diagnostic_case_ids")
    controls = _unique_ids(v.get("control_case_ids"), "control_case_ids")
    pool = _unique_ids(v.get("candidate_pool_ids"), "candidate_pool_ids")
    if len(diagnostics) != 8 or len(pool) != 24 or len(controls) != 3:
        raise LabError("The study requires eight diagnostic, three control and 24 candidate-pool IDs")
    if not set(controls) <= set(diagnostics) or set(pool) & set(diagnostics):
        raise LabError("Controls must be diagnostics, and candidate pool must not contain diagnostics")
    if not set(pool + diagnostics) <= set(lookup):
        raise LabError("Unknown pool or diagnostic ID")
    counts = Counter()
    for case in cases:
        role = "diagnostic" if case["id"] in diagnostics else "candidate_pool" if case["id"] in pool else "primary"
        expected = _metadata(case, role, plan["dtype"])
        if meta[case["id"]] != expected:
            raise LabError("Case role/family/coverage metadata differs from registered shape: " + case["id"])
        if role == "primary":
            if case["split"] not in PRIMARY_SPLITS:
                raise LabError("Primary role has a non-primary split")
            counts[case["split"]] += 1
        elif case["split"] != "candidate":
            raise LabError("Diagnostic/pool cases require the candidate transport split")
        if role != "diagnostic" and _shape(case) in _DIAGNOSTIC_SHAPES:
            raise LabError("Diagnostic shapes cannot be primary or acquisition data")
        if role == "diagnostic" and _shape(case) not in _DIAGNOSTIC_SHAPES:
            raise LabError("Registered diagnostic cases must match the declared diagnostic registry")
    if dict(counts) != MAIN_COUNTS:
        raise LabError("Incorrect primary split sizes")
    families = {
        split: {(c["n"], c["k"]) for c in cases if c["split"] == split and meta[c["id"]]["role"] != "diagnostic"}
        for split in (*PRIMARY_SPLITS, "candidate")
    }
    if any(families[s] != set(_SEEN_FAMILIES) for s in ("calibration", "development", "test", "candidate")):
        raise LabError("Train/dev/test/pool must use the six preregistered NK families")
    if families["challenge"] != set(_CHALLENGE_FAMILIES) or families["challenge"] & families["calibration"]:
        raise LabError("Challenge NK families must be fully held out")
    for split, expected_m in _M_BY_SPLIT.items():
        for family in families[split]:
            observed = {c["m"] for c in cases if c["split"] == split and (c["n"], c["k"]) == family and meta[c["id"]]["role"] != "diagnostic"}
            if observed != set(expected_m):
                raise LabError("Predeclared M values changed for " + split + "/" + str(family))
    if v.get("expected_sessions") != 3 or type(v.get("expected_sessions")) is not int:
        raise LabError("The study declares three separately identified sessions")
    if type(v.get("control_every_visits")) is not int or v["control_every_visits"] != 8:
        raise LabError("Control insertion frequency must be eight primary visits")
    interval = v.get("telemetry_interval_seconds")
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval != 1.0:
        raise LabError("The telemetry interval is one second, outside timed events")
    if v.get("quality") != _QUALITY:
        raise LabError("Quality qualifications differ from the preregistered values")
    diag = v.get("diagnostics")
    if not isinstance(diag, dict) or diag.get("iterations") != [25, 50, 100] or type(diag.get("passes")) is not int or diag["passes"] != 2:
        raise LabError("Diagnostic replay-count sweep must be 25/50/100 over two declared passes")
    for name in ("profile_case_ids", "rotation_case_ids"):
        ids = _unique_ids(diag.get(name), "diagnostics." + name)
        if len(ids) != 2 or not set(ids) <= set(diagnostics):
            raise LabError(name + " must contain two diagnostic-only cases")
    if diag.get("rotation_buffer_counts") != [1, 4]:
        raise LabError("Working-set rotation comparison must retain one and four buffer sets")
    acquisition = v.get("acquisition")
    if not isinstance(acquisition, dict) or acquisition.get("budgets") != [2, 4, 8, 12] or type(acquisition.get("batch_size")) is not int or acquisition["batch_size"] != 2:
        raise LabError("Invalid offline acquisition learning-curve contract")
    seeds = acquisition.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 10 or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds) or len(set(seeds)) != 10:
        raise LabError("Acquisition requires ten distinct declared integer seeds")
    if v.get("primary_metric") != "median_ape_pct":
        raise LabError("Primary model-selection metric must remain median APE")
    registry = v.get("diagnostic_registry", {})
    if not isinstance(registry, dict) or registry.get("shape_count") != 8 or registry.get("sha256") != digest(sorted(_DIAGNOSTIC_SHAPES)) or registry.get("shape_order") != ["m", "n", "k"]:
        raise LabError("Diagnostic exclusion registry mismatch")
    interpretation = v.get("interpretation", {})
    if not isinstance(interpretation, dict) or interpretation.get("diagnostics_excluded_from_model") is not True or interpretation.get("profiler_times_excluded_from_model") is not True:
        raise LabError("Diagnostic and profiler measurements must remain excluded from fitting")
    if interpretation.get("rotation_is_cache_flush") is not False or interpretation.get("accuracy_improvement_is_established") is not False:
        raise LabError("Unmeasured cache/accuracy claims are not part of a preregistered plan")
    if interpretation.get("quality_thresholds_are_vendor_standards") is not False or interpretation.get("benchmark_certification") != "none; not MLPerf or vendor certification":
        raise LabError("Project quality gates must not be presented as vendor/MLPerf certification")
    if interpretation.get("shape_metadata_contains_latency") is not False:
        raise LabError("Shape metadata is declared before any latency observation")
    return plan


def case_groups(case: dict, plan: dict) -> dict[str, str]:
    """Return transparent shape groups for reporting, without using latency."""
    case_id = case.get("case_id", case.get("id"))
    lookup = {c["id"]: c for c in plan.get("cases", [])}
    if case_id not in lookup:
        raise LabError("Unknown case for group reporting")
    registered = lookup[case_id]
    if any(case.get(k) != registered[k] for k in ("m", "n", "k")):
        raise LabError("Group-reporting case differs from its registered shape")
    role = plan["study"]["case_metadata"][case_id]["role"]
    meta = _metadata(registered, role, plan["dtype"])
    return {key: str(meta[key]) for key in ("m_bucket", "alignment", "family", "role", "coverage")}


def protocol_signature(plan: dict) -> str:
    """Hash the workload/timer contract; no device, UUID, session or telemetry.

    Full plan hashes additionally bind split membership, models' acquisition
    budgets, and case identities.  Matching this signature alone never grants
    permission to fit across devices or evaluate arbitrary foreign rows.
    """
    return digest({
        "protocol_version": plan.get("protocol_version"),
        "dtype": plan.get("dtype"), "cache_policy": plan.get("cache_policy"),
        "timing": plan.get("timing"), "collection": plan.get("collection"),
        "operation": "torch.mm(X[M,K],W[N,K].T,out=Y[M,N]);no_bias_no_activation",
        "layout": "contiguous_X_MK_weight_NK_Y_MN;precomputed_transpose_view_KN",
        "reference": "IEEE_FP32_matmul_of_the_same_already_represented_inputs",
        "inference_mode": True, "autocast": "disabled",
    })


def primary_cases(plan: dict, splits: Iterable[str] | None = None) -> list[dict]:
    """Return primary registry entries; never include diagnostic candidate rows."""
    wanted = PRIMARY_SPLITS if splits is None else set(splits)
    if not wanted <= PRIMARY_SPLITS:
        raise LabError("primary_cases accepts only calibration/development/test/challenge")
    return [copy.deepcopy(c) for c in plan["cases"]
            if c["split"] in wanted and plan["study"]["case_metadata"][c["id"]]["role"] == "primary"]


def diagnostic_cases(plan: dict) -> list[dict]:
    lookup = {c["id"]: c for c in plan["cases"]}
    return [copy.deepcopy(lookup[case_id]) for case_id in plan["study"]["diagnostic_case_ids"]]
