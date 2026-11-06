"""Offline, family-grouped model selection for an already measured dataset.

Only records explicitly assigned to ``train`` can enter fitting.  Hyperparameters
and model families are selected with grouped out-of-fold predictions, then every
family's chosen configuration is refitted on all training records.  Test labels
may already exist on disk; they are never arguments to ``fit_cv_suite``.

The out-of-fold scores are tuning scores, not an unbiased estimate of the winner's
accuracy.  Geometry-derived wave features are declared proxies, not kernel tile
observations.  No profiler output or observed target time is a prediction input.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares

from .common import DTYPE_BYTES, LabError, digest, flops, ideal_bytes


VERSION = "1.1.0"
MODEL_ORDER = (
    "flop_only", "roofline_2p", "roofline_3p", "shape_log_ridge", "roofline_residual_krr",
)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
KERNEL_ALPHAS = (0.01, 0.1, 1.0)
KERNEL_LENGTHS = (1.0, 2.0, 4.0, 8.0)
TILE_PROXIES = ((64, 64), (128, 128))
FEATURE_NAMES = (
    "log2_m", "log2_n", "log2_k", "log2_arithmetic_intensity", "log2_ideal_bytes",
    "log2_n_over_k", "small_m_exp_32", "small_m_exp_128", "n_unaligned_8",
    "k_unaligned_8", "n_unaligned_16", "k_unaligned_16", "m_tile128_waste_proxy",
    "n_tile128_waste_proxy", "k_tile128_waste_proxy",
    "log2_cta_count_64x64_proxy", "log2_wave_count_64x64_proxy", "wave_fill_64x64_proxy",
    "log2_cta_count_128x128_proxy", "log2_wave_count_128x128_proxy", "wave_fill_128x128_proxy",
    "sm_count_known",
)
_CONTEXT_FIELDS = ("measurement_kind", "environment_id", "dtype", "cache_policy")
_HELDOUT_SPLITS = frozenset({"test", "challenge", "stress"})
_SELECTION_METRICS = frozenset({"mean_ape_pct", "p90_ape_pct", "max_ape_pct"})


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise LabError(f"{label} must be positive and finite")
    return float(value)


def _case_id(row: dict) -> str:
    value = row.get("case_id", row.get("id"))
    if not isinstance(value, str) or not value.strip():
        raise LabError("Every record needs a nonempty case_id or id")
    if "case_id" in row and "id" in row and row["case_id"] != row["id"]:
        raise LabError("case_id and id disagree")
    return value


def _shape(row: dict) -> tuple[int, int, int]:
    if not isinstance(row, dict):
        raise LabError("A shape record must be an object")
    for key in ("m", "n", "k"):
        if type(row.get(key)) is not int or not 1 <= row[key] <= 2**31 - 1:
            raise LabError(f"{key} must be an integer in [1, 2**31-1]")
    return row["m"], row["n"], row["k"]


def _sm_count(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise LabError("sm_count must be a positive integer or None")
    return value


def _row_weight(row: dict) -> float:
    value = row.get("weight", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise LabError("Case weights must be finite and nonnegative")
    return float(value)


def _weights(rows: list[dict], weighted: bool) -> np.ndarray:
    weights = np.array([_row_weight(row) if weighted else 1.0 for row in rows], dtype=float)
    if not len(weights) or float(np.max(weights)) <= 0:
        raise LabError("Records must have positive total weight")
    # Normalize without overflowing a sum of individually finite weights.
    weights /= float(np.max(weights))
    return weights / float(np.sum(weights))


def _validate_rows(rows: list[dict], *, fitting: bool = False) -> tuple[list[dict], dict]:
    if not isinstance(rows, list) or not rows:
        raise LabError("No measurement rows supplied")
    clean, ids, shapes = [], set(), set()
    context = None
    for row in rows:
        shape, cid = _shape(row), _case_id(row)
        if cid in ids or shape in shapes:
            raise LabError("One modeling dataset must contain unique case IDs and matrix shapes")
        ids.add(cid)
        shapes.add(shape)
        if row.get("split") not in {"train", *_HELDOUT_SPLITS}:
            raise LabError("Offline model records require train/test/challenge/stress split labels")
        if fitting and row["split"] != "train":
            raise LabError("Held-out labels cannot enter fitting; fit_cv_suite accepts only train records")
        if row.get("measurement_kind") not in {"cuda_measured", "simulated"}:
            raise LabError("measurement_kind must distinguish CUDA measurements from simulation")
        if row.get("dtype") not in DTYPE_BYTES:
            raise LabError("Unsupported dtype")
        for key in ("environment_id", "cache_policy"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise LabError(f"Missing {key}")
        current = {key: row[key] for key in _CONTEXT_FIELDS}
        if context is None:
            context = current
        elif context != current:
            changed = [key for key in _CONTEXT_FIELDS if context[key] != current[key]]
            raise LabError("Cannot mix modeling contexts: " + ", ".join(changed))
        if row.get("status", "ok") != "ok" or row.get("correctness", {}).get("passed", True) is not True:
            raise LabError("Failed measurements cannot enter modeling")
        target = _positive(row.get("median_ms"), "median_ms")
        weight = _row_weight(row)
        clean.append({**row, "case_id": cid, "median_ms": target, "weight": weight})
    return sorted(clean, key=lambda row: row["case_id"]), context


def feature_matrix(cases: list[dict], dtype: str, *, sm_count: int | None = None) -> np.ndarray:
    """Features known from dimensions and declared hardware, without execution.

    Wave proxies assume one tile per SM per wave for two fixed tile geometries.
    They do not identify an actual cuBLAS/CUTLASS kernel, residency, or tile size.
    If SM count is unavailable, the SM-dependent proxies are zero and the final
    availability flag is zero; no A100 SM count is silently substituted.
    """
    if dtype not in DTYPE_BYTES or not isinstance(cases, list) or not cases:
        raise LabError("Features require a supported dtype and nonempty shape list")
    sm_count = _sm_count(sm_count)
    output = []
    for case in cases:
        m, n, k = _shape(case)
        q, f = ideal_bytes(case, dtype), flops(case)
        values = [
            math.log2(m), math.log2(n), math.log2(k), math.log2(f / q), math.log2(q),
            math.log2(n / k), math.exp(-m / 32), math.exp(-m / 128),
            float(n % 8 != 0), float(k % 8 != 0), float(n % 16 != 0), float(k % 16 != 0),
            1 - m / (128 * math.ceil(m / 128)),
            1 - n / (128 * math.ceil(n / 128)),
            1 - k / (128 * math.ceil(k / 128)),
        ]
        for tm, tn in TILE_PROXIES:
            tiles = math.ceil(m / tm) * math.ceil(n / tn)
            waves = math.ceil(tiles / sm_count) if sm_count is not None else None
            values.extend((math.log2(tiles), math.log2(waves) if waves is not None else 0.0,
                           tiles / (waves * sm_count) if waves is not None else 0.0))
        values.append(float(sm_count is not None))
        output.append(values)
    result = np.asarray(output, dtype=float)
    if result.shape != (len(cases), len(FEATURE_NAMES)) or not np.all(np.isfinite(result)):
        raise LabError("Non-finite shape features")
    return result


def _standardize_train(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _family(row: dict) -> tuple[int, int]:
    return row["n"], row["k"]


def grouped_folds(rows: list[dict], *, max_folds: int = 6, weighted: bool = True) -> list[dict]:
    """Deterministic folds with no (N,K) family on both sides of a fold.

    At most six folds are used by default. With six or fewer families this is
    leave-one-family-out. Larger sets are greedily balanced using only family
    membership and declared weights, never measured latency.
    """
    if type(max_folds) is not int or not 2 <= max_folds <= 32:
        raise LabError("max_folds must be an integer in [2, 32]")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        _shape(row)
        groups[_family(row)].append(index)
    if len(groups) < 3:
        raise LabError("Grouped cross-validation requires at least three N/K families")
    weights = _weights(rows, weighted)
    count = min(max_folds, len(groups))
    assignment = [[] for _ in range(count)]
    loads = [0.0] * count
    group_order = sorted(groups, key=lambda group: (-float(np.sum(weights[groups[group]])), group))
    for group in group_order:
        position = min(range(count), key=lambda i: (loads[i], len(assignment[i]), i))
        assignment[position].append(group)
        loads[position] += float(np.sum(weights[groups[group]]))
    folds = []
    all_indices = set(range(len(rows)))
    for index, families in enumerate(assignment):
        valid = sorted(i for family in families for i in groups[family])
        train = sorted(all_indices - set(valid))
        if len(train) < 4 or sum(weights[i] > 0 for i in train) < 4:
            raise LabError("Each CV training fold needs at least four positive-weight shapes")
        if not valid or float(np.sum(weights[valid])) <= 0:
            raise LabError("Each CV validation fold needs positive total weight")
        folds.append({
            "fold_index": index, "train_indices": train, "validation_indices": valid,
            "train_case_ids": [rows[i]["case_id"] for i in train],
            "validation_case_ids": [rows[i]["case_id"] for i in valid],
            "train_families": [list(family) for family in sorted({_family(rows[i]) for i in train})],
            "validation_families": [list(family) for family in sorted(families)],
        })
    return folds


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    positive = weights > 0
    indices = np.argsort(values[positive], kind="stable")
    ordered, mass = values[positive][indices], weights[positive][indices]
    cutoff = probability * float(np.sum(mass))
    position = min(int(np.searchsorted(np.cumsum(mass), cutoff, side="left")), len(ordered) - 1)
    return float(ordered[position])


def regression_metrics(rows: list[dict], predicted_ms: list[float], *, include_predictions: bool = True) -> dict:
    """Report ordinary and explicitly weighted errors on distinct shape targets.

    Weighted quantiles use the inverse weighted empirical CDF. Unweighted
    percentiles use NumPy's linear interpolation. Weights describe the declared
    workload scenario; they are not an assertion about production traffic.
    """
    if not rows or len(rows) != len(predicted_ms):
        raise LabError("Metrics require one prediction per nonempty row")
    actual = np.asarray([_positive(row.get("median_ms"), "median_ms") for row in rows])
    predicted = np.asarray([_positive(value, "predicted_ms") for value in predicted_ms])
    weights = _weights(rows, True)
    error = predicted - actual
    ape = np.abs(error / actual) * 100
    log_ratio = np.log(predicted) - np.log(actual)
    if not np.all(np.isfinite(ape)):
        raise LabError("Prediction error overflows finite arithmetic")
    error_scale = float(np.max(np.abs(error)))
    squared = np.zeros_like(error) if error_scale == 0 else (error / error_scale) ** 2
    metrics = {
        "n": len(rows), "n_positive_weight": int(np.sum(weights > 0)),
        "mean_ape_pct": float(np.mean(ape)), "median_ape_pct": float(np.median(ape)),
        "p90_ape_pct": float(np.percentile(ape, 90)), "max_ape_pct": float(np.max(ape)),
        "weighted_mean_ape_pct": float(weights @ ape),
        "weighted_median_ape_pct": _weighted_quantile(ape, weights, 0.5),
        "weighted_p90_ape_pct": _weighted_quantile(ape, weights, 0.9),
        "weighted_max_ape_pct": float(np.max(ape[weights > 0])),
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(error_scale * np.sqrt(np.mean(squared))),
        "weighted_mae_ms": float(weights @ np.abs(error)),
        "weighted_rmse_ms": float(error_scale * np.sqrt(weights @ squared)),
        "signed_bias_ms": float(np.mean(error)), "weighted_signed_bias_ms": float(weights @ error),
        "log_rmse": float(np.sqrt(np.mean(log_ratio**2))),
        "weighted_log_rmse": float(np.sqrt(weights @ (log_ratio**2))),
        "sample_unit": "distinct_shape_target",
        "weighted_quantile_method": "inverse_weighted_empirical_cdf",
        "unweighted_quantile_method": "numpy_linear_interpolation",
    }
    if include_predictions:
        records = [{
            "case_id": _case_id(row), **{axis: row[axis] for axis in ("m", "n", "k")},
            "split": row.get("split"), "weight": _row_weight(row),
            "actual_ms": float(actual[i]), "predicted_ms": float(predicted[i]),
            "ape_pct": float(ape[i]), "error_ms": float(error[i]),
        } for i, row in enumerate(rows)]
        metrics["predictions"] = records
        # Across shapes, not across competing implementations of one shape.
        left, right = np.triu_indices(len(rows), 1)
        difference = actual[left] - actual[right]
        keep = np.abs(difference) > 0.01 * np.maximum(actual[left], actual[right]) + 1e-12
        predicted_difference = predicted[left] - predicted[right]
        score = np.where(np.isclose(predicted[left], predicted[right], rtol=1e-12, atol=0), 0.5,
                         (np.sign(difference) == np.sign(predicted_difference)).astype(float))
        metrics.update(
            pairwise_ordering_accuracy=float(np.mean(score[keep])) if np.any(keep) else None,
            ranking_pairs=int(np.sum(keep)), excluded_actual_ties=int(np.sum(~keep)),
            pairwise_actual_tie_rel_tol=0.01,
            ranking_scope="Across different shapes; not kernel-selection accuracy or decision regret.",
        )
    return metrics


def _fit_physical(rows: list[dict], model_id: str, *, weighted: bool) -> dict:
    """Positive effective-rate regression; these coefficients are not counters."""
    dtype = rows[0]["dtype"]
    f = np.asarray([flops(row) for row in rows], dtype=float)
    q = np.asarray([ideal_bytes(row, dtype) for row in rows], dtype=float)
    target = np.asarray([row["median_ms"] for row in rows], dtype=float)
    weights = _weights(rows, weighted) * len(rows)
    fs, qs, ts = float(np.max(f)), float(np.max(q)), float(np.median(target))
    fn, qn, yn = f / fs, q / qs, target / ts
    has_offset, has_memory = model_id != "roofline_2p", model_id != "flop_only"
    names = (["overhead_ms"] if has_offset else []) + ["compute_tflops"] + (["bandwidth_gbps"] if has_memory else [])
    cref, bref = float(np.min(yn / fn)), float(np.min(yn / qn))
    lower = ([float(np.min(yn)) * 1e-8] if has_offset else []) + [cref * 1e-6] + ([bref * 1e-6] if has_memory else [])
    upper = ([float(np.max(yn)) * 10] if has_offset else []) + [cref * 100] + ([bref * 100] if has_memory else [])
    bounds = np.log(lower), np.log(upper)

    def terms(log_coefficients):
        coefficients = np.exp(log_coefficients)
        start = int(has_offset)
        compute = fn * coefficients[start]
        kernel = np.maximum(compute, qn * coefficients[start + 1]) if has_memory else compute
        return (coefficients[0] if has_offset else 0.0) + kernel

    def residual(log_coefficients):
        return np.log(terms(log_coefficients)) - np.log(yn)

    def weighted_soft_l1(z):
        root = np.sqrt(1 + z)
        return np.vstack((2 * (root - 1) * weights, weights / root, -0.5 * weights / root**3))

    solutions = []
    starts = ((1.0, 1.0), (0.3, 1.5), (1.5, 0.3)) if has_memory else ((1.0, 1.0),)
    offsets = (0.1, 0.8) if has_offset else (0.0,)
    for fraction in offsets:
        for cf, bf in starts:
            initial = ([float(np.min(yn)) * fraction] if has_offset else []) + [cref * cf] + ([bref * bf] if has_memory else [])
            initial = np.clip(np.log(initial), bounds[0] + 1e-6, bounds[1] - 1e-6)
            fit = least_squares(residual, initial, bounds=bounds, loss=weighted_soft_l1,
                                f_scale=0.05, max_nfev=800, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            if fit.success and math.isfinite(float(fit.cost)) and np.all(np.isfinite(fit.x)):
                solutions.append(fit)
    if not solutions:
        raise LabError(f"{model_id} failed to converge")
    best = min(solutions, key=lambda item: item.cost)
    coefficients = np.exp(best.x)
    start = int(has_offset)
    params = {"overhead_ms": float(coefficients[0] * ts) if has_offset else 0.0,
              "compute_tflops": float(fs / (ts * coefficients[start] * 1e9))}
    if has_memory:
        params["bandwidth_gbps"] = float(qs / (ts * coefficients[start + 1] * 1e6))
    near = [name for i, name in enumerate(names)
            if min(best.x[i] - bounds[0][i], bounds[1][i] - best.x[i]) < 1e-4 * (bounds[1][i] - bounds[0][i])]
    singular = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.sum(singular > max(float(singular[0]) * 1e-6, 1e-15)))
    warnings = []
    if rank < len(names):
        warnings.append("Some effective coefficients are weakly identifiable under this training data.")
    if near:
        warnings.append("Effective fit approaches artificial bounds: " + ", ".join(near))
    return {"model_id": model_id, "params": params, "fit_diagnostics": {
        "objective": "weighted soft_l1(log(prediction / target)), f_scale=0.05",
        "n_train": len(rows), "starts_attempted": len(offsets) * len(starts),
        "starts_converged": len(solutions), "cost": float(best.cost),
        "sensitivity_rank": rank, "parameter_count": len(names),
        "near_boundary_parameters": near, "warnings": warnings,
        "interpretation": "Effective regression coefficients, not measured peak FLOPs, DRAM bandwidth, or launch time.",
    }}


def _rbf(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    # Avoid an N x M x feature-count temporary for the larger workload dataset.
    distance = np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :] - 2 * a @ b.T, 0)
    return np.exp(-distance / (2 * length**2))


def _predict_model(model: dict, cases: list[dict], dtype: str, sm_count: int | None) -> list[float]:
    name = model["model_id"]
    if name in MODEL_ORDER[:3]:
        params = model["params"]
        values = []
        for row in cases:
            _shape(row)
            compute = flops(row) / (params["compute_tflops"] * 1e9)
            term = compute if name == "flop_only" else max(compute, ideal_bytes(row, dtype) / (params["bandwidth_gbps"] * 1e6))
            values.append(params["overhead_ms"] + term)
    else:
        x = (feature_matrix(cases, dtype, sm_count=sm_count) - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
        if name == "shape_log_ridge":
            values = np.exp(np.clip(model["intercept"] + x @ np.asarray(model["coefficients"]),
                                    *model["log_prediction_bounds"])).tolist()
        elif name == "roofline_residual_krr":
            correction = _rbf(x, np.asarray(model["feature_train"]), model["length_scale"]) @ np.asarray(model["kernel_weights"])
            base = np.asarray(_predict_model(model["base_model"], cases, dtype, sm_count))
            values = (base * np.exp(np.clip(correction, *model["correction_log_bounds"]))).tolist()
        else:
            raise LabError("Unknown latency model")
    return [_positive(float(value), "predicted_ms") for value in values]


def _fit_learned(rows: list[dict], model_id: str, configuration: dict, *, sm_count: int | None,
                 weighted: bool, base_model: dict, standardized: tuple | None = None) -> dict:
    dtype = rows[0]["dtype"]
    z, mean, scale = standardized or _standardize_train(feature_matrix(rows, dtype, sm_count=sm_count))
    weights = _weights(rows, weighted) * len(rows)
    y = np.log(np.asarray([row["median_ms"] for row in rows]))
    model = {"model_id": model_id, "feature_names": list(FEATURE_NAMES),
             "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
             "standardization": "unweighted training-only feature mean and population standard deviation",
             "alpha": configuration["alpha"]}
    if model_id == "shape_log_ridge":
        mass = weights / np.sum(weights)
        z_mean, y_mean = mass @ z, float(mass @ y)
        centered = z - z_mean
        coefficient = solve(centered.T @ (weights[:, None] * centered) + configuration["alpha"] * np.eye(z.shape[1]),
                            centered.T @ (weights * (y - y_mean)), assume_a="pos")
        model.update(coefficients=coefficient.tolist(), intercept=float(y_mean - z_mean @ coefficient),
                     log_prediction_bounds=[float(np.min(y) - math.log(100)), float(np.max(y) + math.log(100))])
    else:
        residual = y - np.log(_predict_model(base_model, rows, dtype, sm_count))
        kernel = _rbf(z, z, configuration["length_scale"])
        root_weights = np.sqrt(weights)
        dual = solve(root_weights[:, None] * kernel * root_weights[None, :] + configuration["alpha"] * np.eye(len(rows)),
                     root_weights * residual, assume_a="pos")
        model.update(base_model=copy.deepcopy(base_model), feature_train=z.tolist(),
                     kernel_weights=(root_weights * dual).tolist(), length_scale=configuration["length_scale"],
                     correction_log_bounds=[-math.log(4), math.log(4)],
                     residual_baseline="roofline_2p",
                     residual_baseline_rationale="Use the lower-parameter physical baseline consistently across folds; 3p remains a separate comparator.")
    return model


def _trials() -> list[dict]:
    return ([{"model_id": name, "configuration": {}} for name in MODEL_ORDER[:3]] +
            [{"model_id": "shape_log_ridge", "configuration": {"alpha": alpha}} for alpha in RIDGE_ALPHAS] +
            [{"model_id": "roofline_residual_krr", "configuration": {"alpha": alpha, "length_scale": length}}
             for alpha in KERNEL_ALPHAS for length in KERNEL_LENGTHS])


def _training_evidence(rows: list[dict]) -> list[dict]:
    fields = ("case_id", "m", "n", "k", "split", "median_ms", "weight", *_CONTEXT_FIELDS)
    return [{key: row[key] for key in fields} for row in rows]


def _domain(rows: list[dict]) -> dict:
    shapes = np.asarray([_shape(row) for row in rows], dtype=float)
    return {"train_shapes": shapes.astype(int).tolist(), "axis_min": np.min(shapes, axis=0).astype(int).tolist(),
            "axis_max": np.max(shapes, axis=0).astype(int).tolist(),
            "nk_families": [list(family) for family in sorted({_family(row) for row in rows})]}


def _rank_key(metrics: dict, score_key: str, secondary_key: str, order: int = 0) -> tuple:
    return metrics[score_key], metrics[secondary_key], order


def fit_cv_suite(train_rows: list[dict], *, sm_count: int | None = None,
                 selection_metric: str = "mean_ape_pct", weighted: bool = True, max_folds: int = 6) -> dict:
    """Select with grouped CV and refit on all train rows, never held-out labels."""
    rows, context = _validate_rows(train_rows, fitting=True)
    sm_count = _sm_count(sm_count)
    if type(weighted) is not bool or selection_metric not in _SELECTION_METRICS:
        raise LabError("Use a supported selection metric and a boolean weighted flag")
    folds = grouped_folds(rows, max_folds=max_folds, weighted=weighted)
    prefix = "weighted_" if weighted else ""
    score_key = prefix + selection_metric
    secondary_key = prefix + ("p90_ape_pct" if selection_metric == "mean_ape_pct" else "mean_ape_pct")
    trials = _trials()
    oof = [np.full(len(rows), np.nan) for _ in trials]
    trial_fold_scores = [[] for _ in trials]
    fold_records = []
    for fold in folds:
        train = [rows[i] for i in fold["train_indices"]]
        validation = [rows[i] for i in fold["validation_indices"]]
        physical = {name: _fit_physical(train, name, weighted=weighted) for name in MODEL_ORDER[:3]}
        standardized = _standardize_train(feature_matrix(train, context["dtype"], sm_count=sm_count))
        fold_records.append({**{key: value for key, value in fold.items() if not key.endswith("indices")},
                             "n_train": len(train), "n_validation": len(validation),
                             "feature_mean": standardized[1].tolist(), "feature_scale": standardized[2].tolist()})
        for index, trial in enumerate(trials):
            name = trial["model_id"]
            model = physical[name] if name in physical else _fit_learned(
                train, name, trial["configuration"], sm_count=sm_count, weighted=weighted,
                base_model=physical["roofline_2p"], standardized=standardized,
            )
            predicted = _predict_model(model, validation, context["dtype"], sm_count)
            oof[index][fold["validation_indices"]] = predicted
            fold_score = regression_metrics(validation, predicted, include_predictions=False)
            trial_fold_scores[index].append({"fold_index": fold["fold_index"], **fold_score})
    trial_reports = []
    for index, trial in enumerate(trials):
        if not np.all(np.isfinite(oof[index])):
            raise LabError("Every training shape needs exactly one out-of-fold prediction")
        metrics = regression_metrics(rows, oof[index].tolist(), include_predictions=False)
        scores = [fold[score_key] for fold in trial_fold_scores[index]]
        trial_reports.append({**trial, "trial_index": index, "oof_metrics": metrics,
                              "fold_scores": trial_fold_scores[index],
                              "fold_score_mean": float(np.mean(scores)),
                              "fold_score_std": float(np.std(scores, ddof=1)),
                              "fold_score_std_definition": "Sample standard deviation across CV folds; not a confidence interval."})
    best = {}
    for name in MODEL_ORDER:
        indices = [i for i, trial in enumerate(trials) if trial["model_id"] == name]
        best[name] = min(indices, key=lambda i: _rank_key(trial_reports[i]["oof_metrics"], score_key, secondary_key, i))
    selected = min(MODEL_ORDER, key=lambda name: _rank_key(
        trial_reports[best[name]]["oof_metrics"], score_key, secondary_key, MODEL_ORDER.index(name)))
    physical = {name: _fit_physical(rows, name, weighted=weighted) for name in MODEL_ORDER[:3]}
    standardized = _standardize_train(feature_matrix(rows, context["dtype"], sm_count=sm_count))
    train_hash = digest(_training_evidence(rows))
    fitted = {}
    summaries = {}
    for name in MODEL_ORDER:
        trial = trial_reports[best[name]]
        model = copy.deepcopy(physical[name]) if name in physical else _fit_learned(
            rows, name, trial["configuration"], sm_count=sm_count, weighted=weighted,
            base_model=physical["roofline_2p"], standardized=standardized,
        )
        model.update(protocol_version=VERSION, dtype=context["dtype"], sm_count=sm_count,
                     train_case_ids=[row["case_id"] for row in rows], training_data_hash=train_hash,
                     configuration=trial["configuration"], weighted_fit=weighted)
        model["model_hash"] = digest(model)
        fitted[name] = model
        summaries[name] = {**trial, "selected_configuration": True,
                           "oof_predictions": [{"case_id": row["case_id"], "predicted_ms": float(oof[best[name]][i])}
                                               for i, row in enumerate(rows)]}
    suite = {
        "schema_version": 2, "protocol_version": VERSION, "artifact_type": "offline_grouped_cv_suite",
        **context, "sm_count": sm_count, "train_case_ids": [row["case_id"] for row in rows],
        "training_data_hash": train_hash, "training_domain": _domain(rows),
        "selection_metric": selection_metric, "selection_score_key": score_key,
        "selection_secondary_key": secondary_key, "weighted_selection": weighted,
        "selection_rule": "Minimum grouped out-of-fold primary error, then tail/mean error, then fixed order. Refit on all training records.",
        "selected_model_id": selected, "model_order": list(MODEL_ORDER), "models": fitted,
        "cv": {"method": "leave_one_NK_family_out" if len(folds) == len({_family(row) for row in rows}) else "balanced_group_k_fold_by_NK_family",
               "fold_count": len(folds), "family_count": len({_family(row) for row in rows}),
               "folds": fold_records, "trials": trial_reports, "models": summaries,
               "score_scope": "Model/hyperparameter tuning on training data; not an unbiased final accuracy estimate.",
               "selection_aggregation": "Pooled out-of-fold case errors with declared weights; fold means/std are descriptive.",
               "few_family_warning": len({_family(row) for row in rows}) < 6},
        "feature_contract": {"names": list(FEATURE_NAMES), "shared_by": list(MODEL_ORDER[3:]),
                             "tile_proxies": [list(value) for value in TILE_PROXIES],
                             "wave_assumption": "One tile per SM per proxy wave; not observed kernel tiling or occupancy.",
                             "unknown_sm_policy": "Zero SM-dependent proxies and availability flag; no assumed GPU model.",
                             "kernel_profile_fields_used": False, "standardization_scope": "Each fold's training rows only; full training rows for refit."},
        "weight_scope": "Declared operation-occurrence/scenario weights, not an empirical production request distribution.",
        "measurement_chronology_requirement": "None: all data may be measured before training; held-out labels remain excluded from fit and selection.",
        "prediction_interval_available": False,
    }
    suite["suite_hash"] = digest(suite)
    return suite


def _check_suite(suite: dict) -> None:
    if not isinstance(suite, dict) or suite.get("artifact_type") != "offline_grouped_cv_suite" or suite.get("protocol_version") != VERSION:
        raise LabError("Unsupported offline CV suite")
    if suite.get("suite_hash") != digest({key: value for key, value in suite.items() if key != "suite_hash"}):
        raise LabError("Model suite checksum mismatch")
    if set(suite.get("models", {})) != set(MODEL_ORDER) or suite.get("selected_model_id") not in MODEL_ORDER:
        raise LabError("Incomplete model suite")
    for name, model in suite["models"].items():
        if model.get("model_hash") != digest({key: value for key, value in model.items() if key != "model_hash"}):
            raise LabError("Model checksum mismatch")
        if model.get("model_id") != name or model.get("training_data_hash") != suite["training_data_hash"]:
            raise LabError("Model training identity differs from its suite")


def predict_suite(suite: dict, cases: list[dict], *, sm_count: int | None = None) -> dict[str, list[float]]:
    """Conditional predictions using shapes only; no evaluation or refitting."""
    _check_suite(suite)
    if not isinstance(cases, list) or not cases:
        raise LabError("Prediction needs nonempty cases")
    target_sm = suite["sm_count"] if sm_count is None else _sm_count(sm_count)
    for case in cases:
        _shape(case)
        if "dtype" in case and case["dtype"] != suite["dtype"]:
            raise LabError("Prediction dtype differs from the model")
        if "cache_policy" in case and case["cache_policy"] != suite["cache_policy"]:
            raise LabError("Prediction cache policy differs from the model")
    return {name: _predict_model(model, cases, suite["dtype"], target_sm) for name, model in suite["models"].items()}


def evaluate_cv_suite(suite: dict, rows: list[dict], *, sm_count: int | None = None,
                      allow_environment_transfer: bool = False) -> dict:
    """Evaluate held-out data without changing the already selected model.

    An explicit environment transfer permits a different GPU environment only;
    dtype, cache policy, and measured/simulated status still must agree.
    """
    _check_suite(suite)
    clean, context = _validate_rows(rows)
    if any(row["split"] not in _HELDOUT_SPLITS for row in clean):
        raise LabError("Final evaluation accepts test/challenge/stress records only")
    if type(allow_environment_transfer) is not bool:
        raise LabError("allow_environment_transfer must be boolean")
    for key in _CONTEXT_FIELDS:
        if key == "environment_id" and allow_environment_transfer:
            continue
        if context[key] != suite[key]:
            raise LabError("Evaluation context differs from fitted context: " + key)
    train_ids = set(suite["train_case_ids"])
    train_shapes = {tuple(shape) for shape in suite["training_domain"]["train_shapes"]}
    if any(row["case_id"] in train_ids or _shape(row) in train_shapes for row in clean):
        raise LabError("Held-out evaluation overlaps the training shapes or case IDs")
    transfer = context["environment_id"] != suite["environment_id"]
    target_sm = (None if transfer else suite["sm_count"]) if sm_count is None else _sm_count(sm_count)
    if not allow_environment_transfer and target_sm != suite["sm_count"]:
        raise LabError("Different SM count requires explicit environment transfer")
    # A different environment with unknown SM count must not silently inherit
    # the source GPU. Context and shapes have already been validated above.
    predictions = {name: _predict_model(model, clean, suite["dtype"], target_sm)
                   for name, model in suite["models"].items()}
    evaluated = {name: regression_metrics(clean, values) for name, values in predictions.items()}
    groups = defaultdict(list)
    for index, row in enumerate(clean):
        groups["split:" + row["split"]].append(index)
        groups["alignment:" + ("multiple_of_16" if row["n"] % 16 == 0 and row["k"] % 16 == 0 else "irregular")].append(index)
        groups[f"family:{row['n']}x{row['k']}"].append(index)
    grouped = {}
    for key, indices in sorted(groups.items()):
        subset = [clean[i] for i in indices]
        if sum(_row_weight(row) for row in subset) <= 0:
            grouped[key] = {"status": "unavailable_zero_weight", "n": len(subset)}
            continue
        grouped[key] = {name: regression_metrics(subset, [values[i] for i in indices], include_predictions=False)
                        for name, values in predictions.items()}
    domain = suite["training_domain"]
    seen = {tuple(value) for value in domain["nk_families"]}
    warnings = []
    for row in clean:
        flags = []
        if _family(row) not in seen:
            flags.append("unseen_NK_family")
        if any(value < low or value > high for value, low, high in zip(_shape(row), domain["axis_min"], domain["axis_max"])):
            flags.append("outside_training_axis_range")
        warnings.append({"case_id": row["case_id"], "flags": flags})
    return {"schema_version": 2, "suite_hash": suite["suite_hash"],
            "selected_model_id": suite["selected_model_id"], "selection_score_key": suite["selection_score_key"],
            "source_environment_id": suite["environment_id"], "target_environment_id": context["environment_id"],
            "environment_transfer": transfer,
            "source_sm_count": suite["sm_count"], "target_sm_count": target_sm,
            "target_sm_unknown": target_sm is None,
            "measurement_kind": context["measurement_kind"], "dtype": context["dtype"], "cache_policy": context["cache_policy"],
            "models": evaluated, "group_metrics": grouped, "domain_warnings": warnings,
            "domain_warning_scope": "Shape coverage diagnostics, not calibrated failure probabilities or prediction intervals.",
            "evaluation_case_ids": [row["case_id"] for row in clean],
            "refitted_during_evaluation": False,
            "accuracy_claim_scope": "Only the supplied held-out observations and declared measurement contract."}
