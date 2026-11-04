"""Interpretable, positive, single-environment linear-layer latency models.

No GPU library is imported here.  ``bandwidth_gbps`` and ``compute_tflops``
are effective regression parameters; neither is a hardware measurement.
All times in the public interface are milliseconds, and GB/Tera use base 10.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .common import DTYPE_BYTES, SCHEMA_VERSION, SPLITS, LabError, digest, flops, ideal_bytes, positive_number


_MODEL_TYPE = "calibrated_roofline_v1"
_METADATA = ("measurement_kind", "environment_id", "dtype", "cache_policy", "plan_hash", "sample_unit")
_FIT_SPLITS = {"calibration", "candidate"}
_PARAMETER_NAMES = ("overhead_ms", "compute_tflops", "bandwidth_gbps")
_ASSUMPTIONS = [
    "Single device/environment, dtype, registered plan and warm_reuse cache policy.",
    "Y = X @ W.T; dense forward linear layer only, without bias or an activation epilogue.",
    "FLOPs = 2*M*N*K; ideal bytes = dtype_bytes*(M*K + K*N + M*N).",
    "Ideal bytes are an accounting proxy, not measured DRAM traffic; caching and transactions are not modeled.",
    "Latency_ms = overhead_ms + max(FLOPs/(compute_tflops*1e9), ideal_bytes/(bandwidth_gbps*1e6)).",
    "The fitted overhead is a fixed latency component, not a direct measurement of kernel launch overhead.",
    "compute_tflops and bandwidth_gbps are fitted effective parameters, not device peak throughput or measured DRAM bandwidth.",
    "Regime labels are hypotheses from the fitted model; hardware counters are needed to establish actual bottlenecks.",
    "Alignment, occupancy, algorithm selection, cache capacity and kernel switching are not explicit features.",
    "Only calibration and explicitly admitted candidate measurements can influence fitted parameters.",
]


def _dimensions(case: dict) -> None:
    if not isinstance(case, dict):
        raise LabError("A case must be an object")
    for name in ("m", "n", "k"):
        value = case.get(name)
        if type(value) is not int or not 1 <= value <= 16384:
            raise LabError(f"{name} must be an integer in [1, 16384]")


def _metadata(source: dict) -> dict:
    if source.get("measurement_kind") not in {"cuda_measured", "simulated"}:
        raise LabError("measurement_kind must distinguish cuda_measured from simulated")
    if source.get("dtype") not in DTYPE_BYTES:
        raise LabError("Unsupported model/measurement dtype")
    if source.get("cache_policy") != "warm_reuse":
        raise LabError("Only the explicitly labeled warm_reuse cache policy is supported")
    for name in ("environment_id", "plan_hash"):
        if not isinstance(source.get(name), str) or not source[name].strip():
            raise LabError(f"Missing {name}")
    unit = source.get("sample_unit", "repeat_mean")
    if unit not in {"repeat_mean", "block_median"}:
        raise LabError("Unsupported timing sample unit")
    return {name: (unit if name == "sample_unit" else source[name]) for name in _METADATA}


def _validate_rows(rows: list[dict], *, fitting: bool = False) -> dict:
    """Check records without requiring a GPU or trusting the caller's filtering.

    The caller must additionally use common.validate_measurements with the full
    registered plan to authenticate IDs/shapes against the plan itself.  This
    module only receives records, so it can enforce hash agreement, not recover
    or authenticate the plan from its hash.
    """
    if not isinstance(rows, list) or not rows:
        raise LabError("No measurement records were provided")
    expected = None
    ids: set[str] = set()
    shapes: set[tuple[int, int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise LabError("Measurement records must be objects")
        _dimensions(row)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise LabError("Missing or duplicate measurement case_id")
        ids.add(case_id)
        shape = (row["m"], row["n"], row["k"])
        if shape in shapes:
            raise LabError("Duplicate matrix shapes cannot enter one modeling operation")
        shapes.add(shape)
        split = row.get("split")
        if split not in SPLITS:
            raise LabError("Unknown measurement split")
        if fitting and split not in _FIT_SPLITS:
            raise LabError("Protected split cannot enter fit_model: only calibration/candidate are allowed")
        meta = _metadata(row)
        if expected is None:
            expected = meta
        elif meta != expected:
            mismatch = [name for name in _METADATA if meta[name] != expected[name]]
            raise LabError(f"Cannot mix measurement metadata: {', '.join(mismatch)}")
        correctness = row.get("correctness")
        if row.get("status") != "ok" or not isinstance(correctness, dict) or correctness.get("passed") is not True:
            raise LabError(f"Failed or unchecked measurement cannot enter modeling: {case_id}")
        actual = positive_number(row.get("median_ms"), "median_ms")
        samples = row.get("samples_ms")
        if not isinstance(samples, list) or len(samples) < 3:
            raise LabError("At least three repeat samples are required")
        ordered = sorted(positive_number(sample, "sample_ms") for sample in samples)
        midpoint = len(ordered) // 2
        median = ordered[midpoint] if len(ordered) % 2 else ordered[midpoint - 1] / 2 + ordered[midpoint] / 2
        if not math.isclose(actual, median, rel_tol=1e-8, abs_tol=1e-12):
            raise LabError("Reported median_ms does not match repeat samples")
        if meta["sample_unit"] == "block_median":
            from .collection import validate_aggregate_row
            validate_aggregate_row(row, expected_blocks=3)
        if row.get("flops") != flops(row) or row.get("ideal_bytes") != ideal_bytes(row, meta["dtype"]):
            raise LabError("Operation count or ideal-byte count does not match shape/dtype")
    if fitting and len(rows) < 4:
        raise LabError("At least four distinct calibration/candidate shapes are required")
    assert expected is not None
    return expected


def _validate_model(model: dict) -> dict:
    if not isinstance(model, dict) or model.get("schema_version") != SCHEMA_VERSION or model.get("model_type") != _MODEL_TYPE:
        raise LabError("Unsupported calibrated roofline model schema/type")
    meta = _metadata(model)
    params = model.get("params")
    if not isinstance(params, dict):
        raise LabError("Model parameters are missing")
    for name in _PARAMETER_NAMES:
        positive_number(params.get(name), name)
    baseline = model.get("flop_only_baseline")
    if baseline is not None:
        if not isinstance(baseline, dict) or baseline.get("model_type") != "flop_only_v1" or not isinstance(baseline.get("params"), dict):
            raise LabError("Malformed FLOP-only baseline")
        for name in ("overhead_ms", "compute_tflops"):
            positive_number(baseline["params"].get(name), f"baseline.{name}")
    return meta


def _components(params: dict, case: dict, dtype: str) -> tuple[float, float, float]:
    fixed = float(params["overhead_ms"])
    compute = flops(case) / (float(params["compute_tflops"]) * 1e9)
    memory = ideal_bytes(case, dtype) / (float(params["bandwidth_gbps"]) * 1e6)
    if not all(math.isfinite(value) and value > 0 for value in (fixed, compute, memory)):
        raise LabError("Model produces non-finite or underflowed latency components")
    if not math.isfinite(fixed + max(compute, memory)):
        raise LabError("Model produces non-finite latency")
    return fixed, compute, memory


def _regime(fixed: float, compute: float, memory: float) -> str:
    if fixed >= max(compute, memory):
        return "overhead"
    return "compute" if compute >= memory else "memory"


def _fit_positive(flops_array: np.ndarray, bytes_array: np.ndarray, actual: np.ndarray, *, roofline: bool) -> tuple[dict, dict]:
    """Fit positive log parameters after scaling the terms to comparable units.

    In scaled coordinates a + max(c*f, b*q), c and b are inverse
    throughput parameters.  Their logs differ from log(P) and log(B) only by
    a sign and a constant; this is the same positive-parameter model.
    """
    time_scale = float(np.median(actual))
    flop_scale = float(np.max(flops_array))
    byte_scale = float(np.max(bytes_array))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        y, f, q = actual / time_scale, flops_array / flop_scale, bytes_array / byte_scale
        minimum = float(np.min(y))
        c_ref = float(np.min(y / f))
        b_ref = float(np.min(y / q))
        lower = np.array([minimum * 1e-8, c_ref * 1e-6, b_ref * 1e-6] if roofline else [minimum * 1e-8, c_ref * 1e-6])
        upper = np.array([minimum, c_ref * 100, b_ref * 100] if roofline else [minimum, c_ref * 100])
    if not np.all(np.isfinite(lower)) or not np.all(lower > 0) or not np.all(np.isfinite(upper)):
        raise LabError("Timing dynamic range is too extreme for a stable positive fit")
    log_bounds = (np.log(lower), np.log(upper))

    def terms(log_coefficients: np.ndarray) -> np.ndarray:
        coefficients = np.exp(log_coefficients)
        kernel = np.maximum(f * coefficients[1], q * coefficients[2]) if roofline else f * coefficients[1]
        return coefficients[0] + kernel

    def residuals(log_coefficients: np.ndarray) -> np.ndarray:
        return np.log(terms(log_coefficients)) - np.log(y)

    solutions = []
    attempts = 0
    # Different crossover positions avoid accepting one arbitrary inactive term.
    factors = ((1.0, 1.0), (0.3, 1.5), (1.5, 0.3)) if roofline else ((1.0, 1.0),)
    for fraction in (0.01, 0.20, 0.50, 0.80, 0.95):
        a = minimum * fraction
        for c_factor, b_factor in factors:
            c = float(np.min((y - a) / f)) * c_factor
            b = float(np.min((y - a) / q)) * b_factor
            initial = np.array([a, c, b] if roofline else [a, c])
            initial = np.clip(initial, lower * (1 + 1e-6), upper / (1 + 1e-6))
            result = least_squares(
                residuals, np.log(initial), bounds=log_bounds,
                loss="soft_l1", f_scale=0.05, max_nfev=2000,
                ftol=1e-11, xtol=1e-11, gtol=1e-11,
            )
            attempts += 1
            if result.success and np.isfinite(result.cost) and np.all(np.isfinite(result.x)):
                solutions.append(result)
    if not solutions:
        raise LabError("Positive latency fit failed to converge from every starting point")
    best = min(solutions, key=lambda result: result.cost)
    coefficients = np.exp(best.x)
    params = {
        "overhead_ms": float(coefficients[0] * time_scale),
        "compute_tflops": float(flop_scale / time_scale / coefficients[1] / 1e9),
    }
    if roofline:
        params["bandwidth_gbps"] = float(byte_scale / time_scale / coefficients[2] / 1e6)
    for name, value in params.items():
        positive_number(value, name)
    fitted = terms(best.x) * time_scale
    if not np.all(np.isfinite(fitted)):
        raise LabError("Fitted latency overflowed")

    # Unweighted log-residual Jacobian: sensitivity to relative parameter changes.
    compute = f * coefficients[1]
    memory = q * coefficients[2] if roofline else np.zeros_like(compute)
    scaled_pred = terms(best.x)
    compute_active = compute >= memory
    jacobian = np.column_stack([np.full_like(y, coefficients[0]), np.where(compute_active, compute, 0)])
    if roofline:
        jacobian = np.column_stack([jacobian, np.where(compute_active, 0, memory)])
    jacobian /= scaled_pred[:, None]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-6))
    condition = float(singular[0] / singular[-1]) if singular[-1] > singular[0] * 1e-12 else None
    parameter_names = list(params)
    near_bounds = []
    for index, name in enumerate(parameter_names):
        span = log_bounds[1][index] - log_bounds[0][index]
        if min(best.x[index] - log_bounds[0][index], log_bounds[1][index] - best.x[index]) < span * 1e-4:
            near_bounds.append(name)
    warnings = []
    if rank < len(params):
        warnings.append("Parameter sensitivity matrix is rank deficient; not all fitted parameters are identifiable.")
    elif condition is not None and condition > 1e4:
        warnings.append("Parameter sensitivities are ill-conditioned; fitted parameters may be unstable.")
    if near_bounds:
        warnings.append("Fit reached or approached an artificial numerical boundary: " + ", ".join(near_bounds) + ".")
    if roofline and not bool(np.any(compute_active)):
        warnings.append("No compute-active calibration case; compute_tflops is not identifiable from this fit.")
    if roofline and bool(np.all(compute_active)):
        warnings.append("No memory-active calibration case; bandwidth_gbps is not identifiable from this fit.")
    if not bool(np.any(np.maximum(compute, memory) >= 5 * coefficients[0])):
        warnings.append("No calibration case has a modeled kernel term >= 5 times the fixed term; throughput saturation is not established.")
    if float(np.max(coefficients[0] / scaled_pred)) < 1e-4:
        warnings.append("The fixed term is negligible across calibration cases; overhead_ms is weakly constrained.")
    near_optimal = [result for result in solutions if result.cost <= best.cost + max(1e-9, abs(best.cost) * 0.01)]
    equivalent = np.exp(np.array([result.x for result in near_optimal]))
    spread = np.max(equivalent, axis=0) / np.min(equivalent, axis=0)
    if bool(np.any(spread > 10)):
        warnings.append("Near-equivalent multi-start fits differ by > 10x in at least one parameter; do not interpret those parameters as hardware measurements.")
    error_pct = np.abs(fitted / actual - 1) * 100
    diagnostics = {
        "fit_method": "multistart least_squares on positive log parameters",
        "objective": "soft_l1(log(predicted_ms) - log(actual_ms)); f_scale=0.05",
        "n_train": len(actual),
        "starts_attempted": attempts,
        "starts_converged": len(solutions),
        "robust_cost": float(best.cost),
        "train_median_ape_pct": float(np.median(error_pct)),
        "train_p90_ape_pct": float(np.percentile(error_pct, 90)),
        "sensitivity_rank": rank,
        "parameter_count": len(params),
        "sensitivity_singular_values": singular.tolist(),
        "sensitivity_condition_number": condition,
        "near_boundary_parameters": near_bounds,
        "near_equivalent_parameter_spread_ratio": {name: float(spread[index]) for index, name in enumerate(parameter_names)},
        "warnings": warnings,
    }
    if roofline:
        labels = [_regime(coefficients[0], float(c), float(b)) for c, b in zip(compute, memory)]
        counts = Counter(labels)
        diagnostics["regime_hypothesis_counts"] = {name: counts[name] for name in ("overhead", "compute", "memory")}
        diagnostics["kernel_term_hypothesis_counts"] = {"compute": int(np.sum(compute_active)), "memory": int(np.sum(~compute_active))}
    return params, diagnostics


def fit_model(rows: list[dict]) -> dict:
    """Fit one calibrated roofline plus a separately fitted FLOP-only baseline.

    Development, test and challenge data are rejected, even when mixed into an
    otherwise valid calibration list.  At least four distinct shapes are
    required for the three-parameter model; that alone does not ensure that
    every parameter is identifiable, so callers must inspect diagnostics.
    """
    meta = _validate_rows(rows, fitting=True)
    f = np.array([flops(row) for row in rows], dtype=np.float64)
    q = np.array([ideal_bytes(row, meta["dtype"]) for row in rows], dtype=np.float64)
    y = np.array([row["median_ms"] for row in rows], dtype=np.float64)
    params, diagnostics = _fit_positive(f, q, y, roofline=True)
    baseline_params, baseline_diagnostics = _fit_positive(f, q, y, roofline=False)
    evidence_fields = ("case_id", "m", "n", "k", "split", "median_ms", "samples_ms", "flops", "ideal_bytes", *_METADATA)
    evidence = []
    for row in rows:
        item = {key: (row.get("sample_unit", "repeat_mean") if key == "sample_unit" else row[key])
                for key in evidence_fields}
        if row.get("sample_unit") == "block_median":
            item.update(aggregation=row["aggregation"], measurement_blocks_hash=row["measurement_blocks_hash"])
        evidence.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": _MODEL_TYPE,
        "params": params,
        **meta,
        "train_case_ids": [row["case_id"] for row in rows],
        "training_data_hash": digest(evidence),
        "assumptions": list(_ASSUMPTIONS),
        "fit_diagnostics": diagnostics,
        "flop_only_baseline": {
            "model_type": "flop_only_v1",
            "params": baseline_params,
            "formula": "latency_ms = overhead_ms + FLOPs/(compute_tflops*1e9)",
            "fit_diagnostics": baseline_diagnostics,
        },
    }


def explain_prediction(model: dict, case: dict, dtype: str) -> dict:
    """Explain a registered shape or an unmeasured shape, without a target time.

    A raw shape has no device provenance.  This is conditional prediction for
    the fitted environment, never validation on a new GPU.  If provenance is
    supplied in ``case``, it must agree with the model.
    """
    meta = _validate_model(model)
    _dimensions(case)
    if dtype != meta["dtype"]:
        raise LabError("Prediction dtype differs from the fitted dtype")
    for key, value in meta.items():
        if key in case and case[key] != value:
            raise LabError(f"Prediction {key} differs from the model")
    fixed, compute, memory = _components(model["params"], case, dtype)
    return {
        "case_id": case.get("case_id", case.get("id")),
        "m": case["m"], "n": case["n"], "k": case["k"],
        "dtype": dtype,
        "flops": flops(case),
        "ideal_bytes": ideal_bytes(case, dtype),
        "arithmetic_intensity_flops_per_byte": flops(case) / ideal_bytes(case, dtype),
        "overhead_ms": fixed,
        "compute_ms": compute,
        "memory_ms": memory,
        "predicted_ms": fixed + max(compute, memory),
        "regime_hypothesis": _regime(fixed, compute, memory),
        "interpretation": "Conditional fitted-model hypothesis; ideal bytes are not measured DRAM traffic and effective rates are not device peaks.",
    }


def _prediction_record(row: dict, predicted: float, regime: str) -> dict:
    actual = float(row["median_ms"])
    error = predicted - actual
    ape = abs(error / actual) * 100
    if not all(math.isfinite(value) for value in (predicted, error, ape)) or predicted <= 0:
        raise LabError("Prediction or percentage error is non-finite")
    return {
        "case_id": row["case_id"],
        "m": row["m"], "n": row["n"], "k": row["k"],
        "split": row["split"],
        "actual_ms": actual,
        "predicted_ms": predicted,
        "ape_pct": ape,
        "error_ms": error,
        "regime_hypothesis": regime,
    }


def predict_rows(model: dict, rows: list[dict]) -> list[dict]:
    """Predict checked measurement records without updating the fitted model."""
    model_meta = _validate_model(model)
    row_meta = _validate_rows(rows)
    if model_meta != row_meta:
        mismatch = [key for key in _METADATA if model_meta[key] != row_meta[key]]
        raise LabError(f"Model and prediction rows differ: {', '.join(mismatch)}")
    predictions = []
    for row in rows:
        fixed, compute, memory = _components(model["params"], row, model_meta["dtype"])
        predictions.append(_prediction_record(row, fixed + max(compute, memory), _regime(fixed, compute, memory)))
    return predictions


def _metrics(predictions: list[dict]) -> dict:
    ape = np.array([row["ape_pct"] for row in predictions])
    error = np.array([row["error_ms"] for row in predictions])
    # Scaling avoids avoidable overflow from squaring a large finite residual.
    error_scale = float(np.max(np.abs(error)))
    rmse = 0.0 if error_scale == 0 else float(error_scale * np.sqrt(np.mean((error / error_scale) ** 2)))
    rank_score, pairs, excluded = 0.0, 0, 0
    for left, right in combinations(predictions, 2):
        actual_difference = left["actual_ms"] - right["actual_ms"]
        # Near-identical observed times need not have a reliably known order.
        if abs(actual_difference) <= 1e-12 + 0.01 * max(left["actual_ms"], right["actual_ms"]):
            excluded += 1
            continue
        pairs += 1
        predicted_difference = left["predicted_ms"] - right["predicted_ms"]
        if math.isclose(left["predicted_ms"], right["predicted_ms"], rel_tol=1e-12, abs_tol=0):
            rank_score += 0.5
        elif (actual_difference > 0) == (predicted_difference > 0):
            rank_score += 1.0
    return {
        "n": len(predictions),
        "median_ape_pct": float(np.median(ape)),
        "p90_ape_pct": float(np.percentile(ape, 90)),
        "max_ape_pct": float(np.max(ape)),
        "rmse_ms": rmse,
        "pairwise_ordering_accuracy": rank_score / pairs if pairs else None,
        "ranking_pairs": pairs,
        "excluded_actual_ties": excluded,
        "pairwise_actual_tie_rel_tol": 0.01,
        "predicted_tie_score": 0.5,
        "predictions": predictions,
    }


def evaluate_model(model: dict, rows: list[dict]) -> dict:
    """Return errors and ranking on supplied data, plus a separate baseline.

    Calling this function never changes parameters or training provenance.
    Percentages use positive measured latency as denominator; exact zero
    residuals legitimately produce zero APE and zero RMSE.
    """
    predictions = predict_rows(model, rows)
    result = _metrics(predictions)
    result["model_type"] = _MODEL_TYPE
    baseline = model.get("flop_only_baseline")
    if baseline is not None:
        params = baseline["params"]
        baseline_predictions = []
        for row in rows:
            compute = flops(row) / (params["compute_tflops"] * 1e9)
            fixed = params["overhead_ms"]
            baseline_predictions.append(_prediction_record(row, fixed + compute, "overhead" if fixed >= compute else "compute"))
        result["flop_only_baseline"] = {"model_type": "flop_only_v1", **_metrics(baseline_predictions)}
    return result
