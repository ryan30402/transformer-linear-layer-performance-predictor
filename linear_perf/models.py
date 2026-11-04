"""Frozen, JSON-serializable latency models with explicit evaluation boundaries.

This module never executes GPU work.  Measured rows are authenticated against
their registered plan and retained raw blocks before fitting.  Development
labels select hyperparameters; test/challenge labels can only be evaluated.
Effective roofline coefficients are regression quantities, not GPU counters.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .common import LabError, digest, flops, ideal_bytes, positive_number, utc_now, validate_measurements
from . import predictor as predictor_api


VERSION = "1.0.0"
MODEL_ORDER = ("flop_only", "roofline_2p", "roofline_3p", "shape_log_ridge", "roofline_residual_krr")
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
KERNEL_ALPHAS = (0.01, 0.1, 1.0)
KERNEL_LENGTHS = (0.5, 1.0, 2.0, 4.0)
FEATURE_NAMES = (
    "log2_m", "log2_n", "log2_k", "log2_arithmetic_intensity", "log2_ideal_bytes",
    "log2_n_over_k", "small_m_exp_32", "small_m_exp_128", "n_unaligned_8",
    "k_unaligned_8", "n_unaligned_16", "k_unaligned_16", "m_tile128_waste_proxy",
    "n_tile128_waste_proxy", "k_tile128_waste_proxy",
)
KERNEL_FEATURE_NAMES = ("log2_m", "log2_n", "log2_k", "n_unaligned_16", "k_unaligned_16")
_META = ("measurement_kind", "environment_id", "dtype", "cache_policy", "plan_hash", "sample_unit")


def _plan(plan: dict) -> dict:
    from .protocol import validate_study_plan
    return validate_study_plan(plan)


def _signature(plan: dict) -> str:
    from .protocol import protocol_signature
    return protocol_signature(plan)


def _sealed(value: dict, field: str) -> dict:
    result = copy.deepcopy(value)
    result[field] = digest(result)
    return result


def _check_seal(value: dict, field: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get(field), str):
        raise LabError("Missing immutable payload digest: " + field)
    payload = {key: item for key, item in value.items() if key != field}
    if digest(payload) != value[field]:
        raise LabError("Payload digest mismatch: " + field)


def _rows(rows: list[dict], plan: dict, splits: set[str]) -> dict:
    _plan(plan)
    validate_measurements(rows, plan, allowed_splits=splits)
    meta = predictor_api._validate_rows(rows, fitting=False)
    if meta["sample_unit"] != "block_median":
        raise LabError("Modeling requires complete three-pass aggregate evidence")
    excluded = set(plan["study"]["diagnostic_case_ids"])
    if any(row["case_id"] in excluded for row in rows):
        raise LabError("Diagnostic/control records cannot enter modeling or evaluation")
    return meta


def _train_and_dev(train: list[dict], dev: list[dict], plan: dict) -> dict:
    meta = _rows(train, plan, {"calibration", "candidate"})
    predictor_api._validate_rows(train, fitting=True)
    dev_meta = _rows(dev, plan, {"development"})
    if meta != dev_meta:
        raise LabError("Training and development provenance differs")
    permitted_candidates = set(plan["study"]["candidate_pool_ids"])
    if any(row["split"] == "candidate" and row["case_id"] not in permitted_candidates for row in train):
        raise LabError("Only registered acquisition-pool candidate labels may enter fitting")
    expected_dev = {case["id"] for case in plan["cases"] if case["split"] == "development"}
    if {row["case_id"] for row in dev} != expected_dev:
        raise LabError("Selection requires the complete predeclared development set")
    return meta


def _environment(env: dict | None, meta: dict, plan: dict, *, required: bool = False) -> dict | None:
    if env is None:
        if required:
            raise LabError("Explicit cross-environment evaluation requires source and target environments")
        return None
    if not isinstance(env, dict):
        raise LabError("Environment must be an object")
    for key in ("environment_id", "measurement_kind", "plan_hash"):
        expected = digest(plan) if key == "plan_hash" else meta[key]
        if env.get(key) != expected:
            raise LabError("Environment does not match the validated rows: " + key)
    if meta["measurement_kind"] == "cuda_measured":
        identity = env.get("identity")
        if not isinstance(identity, dict) or digest(identity) != env["environment_id"]:
            raise LabError("CUDA environment identity hash is inconsistent")
        for key in ("dtype", "cache_policy", "timing", "collection"):
            if identity.get(key) != plan.get(key):
                raise LabError("Environment timing/precision contract mismatch: " + key)
        if identity.get("math_policy", {}).get("fp32_reference_precision") != "ieee":
            raise LabError("Cross-run evidence requires the declared IEEE FP32 reference policy")
    return copy.deepcopy(env)


def _training_evidence(rows: list[dict]) -> list[dict]:
    return [{key: row.get(key) for key in (
        "case_id", "m", "n", "k", "split", "median_ms", "samples_ms",
        "measurement_blocks_hash", *_META,
    )} for row in sorted(rows, key=lambda r: r["case_id"])]


def _domain(rows: list[dict]) -> dict:
    shapes = np.asarray([[row[axis] for axis in ("m", "n", "k")] for row in rows], dtype=float)
    log_shapes = np.log2(shapes)
    distances = np.sum(np.abs(log_shapes[:, None, :] - log_shapes[None, :, :]), axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    threshold = max(float(np.percentile(nearest, 95)), 1e-6)
    return {
        "train_shapes": shapes.astype(int).tolist(),
        "axis_min": np.min(shapes, axis=0).astype(int).tolist(),
        "axis_max": np.max(shapes, axis=0).astype(int).tolist(),
        "nk_pairs": sorted({(row["n"], row["k"]) for row in rows}),
        "loo_nearest_log2_l1_p95": threshold,
        "interpretation": "Training-shape distance diagnostics only; not a probability of failure or accuracy guarantee.",
    }


def _ood(case: dict, domain: dict) -> dict:
    shape = np.array([case[axis] for axis in ("m", "n", "k")], dtype=float)
    training = np.array(domain["train_shapes"], dtype=float)
    distance = float(np.min(np.sum(np.abs(np.log2(training) - np.log2(shape)), axis=1)))
    flags = []
    if np.any(shape < np.array(domain["axis_min"])) or np.any(shape > np.array(domain["axis_max"])):
        flags.append("outside_training_axis_range")
    if (int(case["n"]), int(case["k"])) not in {tuple(p) for p in domain["nk_pairs"]}:
        flags.append("unseen_nk_family")
    if distance > domain["loo_nearest_log2_l1_p95"]:
        flags.append("distance_above_training_loo_p95")
    return {"flags": flags, "nearest_training_log2_l1_distance": distance,
            "distance_warning_threshold": domain["loo_nearest_log2_l1_p95"]}


def _features(cases: list[dict], dtype: str, *, kernel: bool = False) -> np.ndarray:
    output = []
    for case in cases:
        predictor_api._dimensions(case)
        m, n, k = (case[axis] for axis in ("m", "n", "k"))
        if kernel:
            output.append([math.log2(m), math.log2(n), math.log2(k), float(n % 16 != 0), float(k % 16 != 0)])
        else:
            q = ideal_bytes(case, dtype)
            output.append([
                math.log2(m), math.log2(n), math.log2(k), math.log2(flops(case) / q), math.log2(q),
                math.log2(n / k), math.exp(-m / 32), math.exp(-m / 128),
                float(n % 8 != 0), float(k % 8 != 0), float(n % 16 != 0), float(k % 16 != 0),
                1 - m / (128 * math.ceil(m / 128)),
                1 - n / (128 * math.ceil(n / 128)),
                1 - k / (128 * math.ceil(k / 128)),
            ])
    return np.asarray(output, dtype=np.float64)


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _base_model(model_id: str, meta: dict, train: list[dict], **payload: Any) -> dict:
    return _sealed({
        "schema_version": 1, "protocol_version": VERSION, "model_id": model_id,
        "model_type": model_id, **meta,
        "train_case_ids": [row["case_id"] for row in train],
        "training_data_hash": digest(_training_evidence(train)),
        "training_domain": _domain(train), **payload,
    }, "model_hash")


def _fit_two_parameter(train: list[dict], meta: dict) -> dict:
    f = np.array([flops(row) for row in train], dtype=float)
    q = np.array([ideal_bytes(row, meta["dtype"]) for row in train], dtype=float)
    y = np.array([row["median_ms"] for row in train], dtype=float)
    scale_y, scale_f, scale_q = float(np.median(y)), float(np.max(f)), float(np.max(q))
    fn, qn, yn = f / scale_f, q / scale_q, y / scale_y
    refs = np.array([np.min(yn / fn), np.min(yn / qn)])
    bounds = (np.log(refs * 1e-6), np.log(refs * 100))

    def residual(z):
        p = np.exp(z)
        return np.log(np.maximum(p[0] * fn, p[1] * qn)) - np.log(yn)

    solutions = []
    for multipliers in ((1, 1), (0.3, 1.5), (1.5, 0.3)):
        initial = np.log(refs * np.array(multipliers))
        result = least_squares(residual, initial, bounds=bounds, loss="soft_l1", f_scale=0.05,
                               max_nfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if result.success and np.all(np.isfinite(result.x)) and math.isfinite(float(result.cost)):
            solutions.append(result)
    if not solutions:
        raise LabError("Two-parameter roofline did not converge")
    best = min(solutions, key=lambda solution: solution.cost)
    coefficient = np.exp(best.x)
    compute = coefficient[0] * fn
    memory = coefficient[1] * qn
    active = compute >= memory
    jacobian = np.column_stack([active.astype(float), (~active).astype(float)])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-6))
    warnings = [] if rank == 2 else ["Only one resource term is active; the other coefficient is not identifiable."]
    return _base_model("roofline_2p", meta, train,
        params={"overhead_ms": 0.0,
                "compute_tflops": float(scale_f / scale_y / coefficient[0] / 1e9),
                "bandwidth_gbps": float(scale_q / scale_y / coefficient[1] / 1e6)},
        fit_diagnostics={"parameter_count": 2, "sensitivity_rank": rank,
                         "starts_attempted": 3, "starts_converged": len(solutions),
                         "robust_cost": float(best.cost), "warnings": warnings},
        interpretation="Explicit zero-offset statistical model; zero offset does not assert zero physical launch overhead.")


def _ridge(train: list[dict], meta: dict, alpha: float) -> dict:
    x, mean, scale = _standardize(_features(train, meta["dtype"]))
    y = np.log(np.array([row["median_ms"] for row in train]))
    intercept = float(np.mean(y))
    beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - intercept))
    return _base_model("shape_log_ridge", meta, train,
        feature_names=list(FEATURE_NAMES), feature_mean=mean.tolist(), feature_scale=scale.tolist(),
        coefficients=beta.tolist(), intercept=intercept, alpha=alpha,
        log_prediction_bounds=[float(np.min(y) - math.log(100)), float(np.max(y) + math.log(100))],
        interpretation="Regularized log-latency regression. Alignment/waste proxies do not identify actual kernel tile dimensions.")


def _kernel(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    distance = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return np.exp(-distance / (2 * length * length))


def _residual_kernel(train: list[dict], meta: dict, base: dict, alpha: float, length: float) -> dict:
    x, mean, scale = _standardize(_features(train, meta["dtype"], kernel=True))
    reference = np.array(predict_model(base, train, meta["dtype"]))
    residuals = np.log(np.array([row["median_ms"] for row in train]) / reference)
    weights = np.linalg.solve(_kernel(x, x, length) + alpha * np.eye(len(train)), residuals)
    return _base_model("roofline_residual_krr", meta, train,
        base_model=copy.deepcopy(base), feature_names=list(KERNEL_FEATURE_NAMES),
        feature_mean=mean.tolist(), feature_scale=scale.tolist(), feature_train=x.tolist(),
        kernel_weights=weights.tolist(), alpha=alpha, length_scale=length,
        correction_log_bounds=[-math.log(4), math.log(4)],
        interpretation="RBF-kernel ridge correction to a two-parameter roofline prior; corrections tend to zero far from training features. Bounds are predeclared numerical safeguards, not confidence limits.")


def predict_model(model: dict, cases: list[dict], dtype: str) -> list[float]:
    """Predict from dimensions only. Never reads observed time or a target split."""
    _check_seal(model, "model_hash")
    if model.get("protocol_version") != VERSION or dtype != model.get("dtype"):
        raise LabError("Model version or dtype mismatch")
    if not isinstance(cases, list) or not cases:
        raise LabError("Prediction requires at least one case")
    for case in cases:
        predictor_api._dimensions(case)
    model_type = model["model_type"]
    if model_type in {"flop_only", "roofline_2p", "roofline_3p"}:
        p = model["params"]
        compute_rate = positive_number(p.get("compute_tflops"), "compute_tflops")
        offset = p.get("overhead_ms")
        if type(offset) not in (int, float) or not math.isfinite(offset) or offset < 0:
            raise LabError("Invalid model offset")
        if model_type != "roofline_2p" and offset <= 0:
            raise LabError("Only the explicit two-parameter model permits zero offset")
        if model_type == "roofline_2p" and offset != 0:
            raise LabError("Two-parameter roofline must have a fixed zero offset")
        bandwidth = None if model_type == "flop_only" else positive_number(p.get("bandwidth_gbps"), "bandwidth_gbps")
        output = []
        for case in cases:
            compute = flops(case) / (compute_rate * 1e9)
            value = offset + (compute if bandwidth is None else max(compute, ideal_bytes(case, dtype) / (bandwidth * 1e6)))
            output.append(value)
    elif model_type == "shape_log_ridge":
        x = (_features(cases, dtype) - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
        log_y = model["intercept"] + x @ np.asarray(model["coefficients"])
        output = np.exp(np.clip(log_y, *model["log_prediction_bounds"])).tolist()
    elif model_type == "roofline_residual_krr":
        x = (_features(cases, dtype, kernel=True) - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
        correction = _kernel(x, np.asarray(model["feature_train"]), model["length_scale"]) @ np.asarray(model["kernel_weights"])
        base = np.asarray(predict_model(model["base_model"], cases, dtype))
        output = (base * np.exp(np.clip(correction, *model["correction_log_bounds"]))).tolist()
    else:
        raise LabError("Unknown model type")
    return [positive_number(float(value), "predicted_ms") for value in output]


def evaluate_predictions(rows: list[dict], predicted_ms: list[float]) -> dict:
    """Metrics over distinct shape targets, never over individual event repeats."""
    if not rows or len(rows) != len(predicted_ms):
        raise LabError("Metric rows and predictions must have identical nonzero length")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise LabError("Metrics require one target per distinct case")
    predictions = []
    for row, value in zip(rows, predicted_ms):
        positive_number(row.get("median_ms"), "observed_ms")
        prediction = predictor_api._prediction_record(row, positive_number(value, "predicted_ms"), "model_hypothesis_only")
        samples = row.get("samples_ms", [])
        if samples and row.get("sample_unit") == "block_median":
            lo, hi = min(samples), max(samples)
            prediction["observed_pass_range_ms"] = [lo, hi]
            prediction["prediction_inside_observed_pass_range"] = lo <= value <= hi
        predictions.append(prediction)
    result = predictor_api._metrics(predictions)
    error = np.array([row["error_ms"] for row in predictions])
    log_ratio = np.array([math.log(row["predicted_ms"]) - math.log(row["actual_ms"]) for row in predictions])
    mean_abs_log_ratio = float(np.mean(np.abs(log_ratio)))
    if mean_abs_log_ratio > math.log(np.finfo(float).max):
        raise LabError("Multiplicative error exceeds finite floating-point range")
    result.update(rmse_us=result["rmse_ms"] * 1000,
                  mae_ms=float(np.mean(np.abs(error))), mae_us=float(np.mean(np.abs(error))) * 1000,
                  signed_bias_ms=float(np.mean(error)), signed_bias_us=float(np.mean(error)) * 1000,
                  mean_ape_pct=float(np.mean([row["ape_pct"] for row in predictions])),
                  log_rmse=float(np.sqrt(np.mean(log_ratio ** 2))),
                  geometric_mean_multiplicative_error=math.exp(mean_abs_log_ratio),
                  log_metric_definitions={
                      "log_rmse": "sqrt(mean(log(predicted_ms / actual_ms)^2)); natural logarithm",
                      "geometric_mean_multiplicative_error": "exp(mean(abs(log(predicted_ms / actual_ms)))); 1 is exact, 2 means geometric mean factor-of-two error; not signed bias",
                  },
                  sample_unit="distinct_shape_aggregate_target",
                  uncertainty_note="Pass ranges are descriptive observed variation, not confidence or prediction intervals.")
    return result


def _score(model: dict, dev: list[dict], dtype: str) -> dict:
    return evaluate_predictions(dev, predict_model(model, dev, dtype))


def _rank_key(metrics: dict, order: int = 0) -> tuple[float, float, int]:
    return (metrics["median_ape_pct"], metrics["p90_ape_pct"], order)


def fit_suite(train_rows: list[dict], development_rows: list[dict], plan: dict,
              *, environment: dict | None = None) -> dict:
    """Fit a fixed model menu; tune and choose exclusively on development data."""
    meta = _train_and_dev(train_rows, development_rows, plan)
    env = _environment(environment, meta, plan)
    train = sorted(copy.deepcopy(train_rows), key=lambda row: row["case_id"])
    dev = sorted(copy.deepcopy(development_rows), key=lambda row: row["case_id"])
    original = predictor_api.fit_model(train)
    models = {
        "flop_only": _base_model("flop_only", meta, train,
            params=original["flop_only_baseline"]["params"], fit_diagnostics=original["flop_only_baseline"]["fit_diagnostics"]),
        "roofline_2p": _fit_two_parameter(train, meta),
        "roofline_3p": _base_model("roofline_3p", meta, train,
            params=original["params"], fit_diagnostics=original["fit_diagnostics"]),
    }
    searches = {}
    ridge_trials = [(_ridge(train, meta, alpha), {"alpha": alpha}) for alpha in RIDGE_ALPHAS]
    kernel_trials = [(_residual_kernel(train, meta, models["roofline_2p"], alpha, length),
                      {"alpha": alpha, "length_scale": length})
                     for alpha in KERNEL_ALPHAS for length in KERNEL_LENGTHS]
    for family, trials in (("shape_log_ridge", ridge_trials), ("roofline_residual_krr", kernel_trials)):
        evaluated = [(model, configuration, _score(model, dev, meta["dtype"])) for model, configuration in trials]
        best_index = min(range(len(evaluated)), key=lambda i: _rank_key(evaluated[i][2], i))
        models[family] = evaluated[best_index][0]
        searches[family] = [{"configuration": config, "median_ape_pct": score["median_ape_pct"],
                            "p90_ape_pct": score["p90_ape_pct"], "selected": index == best_index,
                            "model_hash": model["model_hash"]}
                           for index, (model, config, score) in enumerate(evaluated)]
    dev_metrics = {name: _score(model, dev, meta["dtype"]) for name, model in models.items()}
    selected = min(MODEL_ORDER, key=lambda name: _rank_key(dev_metrics[name], MODEL_ORDER.index(name)))
    return _sealed({
        "schema_version": 1, "artifact_type": "linear_perf_suite", "protocol_version": VERSION,
        "created_at_utc": utc_now(), **meta, "protocol_signature": _signature(plan),
        "models": models, "selected_model_id": selected, "development_metrics": dev_metrics,
        "hyperparameter_search": searches,
        "selection_rule": "Minimum development median APE; ties break by development P90 APE then the fixed MODEL_ORDER. No test/challenge labels are available to this function.",
        "model_order": list(MODEL_ORDER), "train_case_ids": [row["case_id"] for row in train],
        "development_case_ids": [row["case_id"] for row in dev],
        "training_data_hash": digest(_training_evidence(train)),
        "development_data_hash": digest(_training_evidence(dev)),
        "source_environment": env,
        "accuracy_claim": "No hardware accuracy improvement is established by fitting or software tests; frozen held-out measured results decide.",
    }, "suite_hash")


def _hardware_identity(env: dict) -> dict:
    identity = env.get("identity", {})
    if env.get("measurement_kind") == "simulated":
        return {"name": identity.get("device", {}).get("name", env.get("device_name", "SIMULATED")),
                "simulated": True}
    device = identity.get("device", {})
    return {key: device.get(key) for key in ("name", "compute_capability", "total_memory_bytes", "multiprocessor_count")}


def _evaluation_context(suite: dict, rows: list[dict], plan: dict, target_environment: dict | None,
                        transfer_mode: str | None) -> dict:
    if transfer_mode not in (None, "session", "gpu"):
        raise LabError("transfer_mode must be None, 'session', or 'gpu'")
    meta = _rows(rows, plan, {"development", "test", "challenge"})
    if any(suite.get(key) != meta[key] for key in _META if key != "environment_id"):
        raise LabError("Frozen model and evaluation precision/plan/sample contract differ")
    if suite.get("protocol_signature") != _signature(plan):
        raise LabError("Frozen model protocol signature differs")
    same_environment = suite["environment_id"] == meta["environment_id"]
    if transfer_mode is None and not same_environment:
        raise LabError("Cross-environment evaluation requires an explicit transfer_mode")
    target = _environment(target_environment, meta, plan, required=transfer_mode is not None)
    if transfer_mode is not None:
        source_meta = {key: suite[key] for key in _META}
        source = _environment(suite.get("source_environment"), source_meta, plan, required=True)
        assert source is not None and target is not None
        if transfer_mode == "session" and _hardware_identity(source) != _hardware_identity(target):
            raise LabError("Session transfer requires the same GPU configuration; use explicit GPU transfer for another architecture/configuration")
        if meta["measurement_kind"] == "cuda_measured":
            keys = ("runner_version", "python_version", "pytorch_version", "cuda_runtime_version", "driver_versions", "math_policy", "execution")
            mismatch = [key for key in keys if source["identity"].get(key) != target["identity"].get(key)]
            if mismatch:
                raise LabError("Transfer is confounded by a different software/math/execution contract: " + ", ".join(mismatch))
        mode = "cross_session_frozen_parameters" if transfer_mode == "session" else "cross_gpu_zero_recalibration"
    else:
        mode = "same_environment_frozen_evaluation"
    return {"mode": mode, "source_environment_id": suite["environment_id"],
            "target_environment_id": meta["environment_id"], "same_environment_identity": same_environment,
            "refitted_on_target": False, "measurement_kind": meta["measurement_kind"],
            "interpretation": "Environment hashes check internal consistency, not hardware attestation. A repeated same-UUID process is not an independent physical GPU."}


def _groups(rows: list[dict], predicted: list[float], plan: dict) -> dict:
    from .protocol import case_groups
    indices: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        for dimension, label in case_groups(row, plan).items():
            indices[str(dimension)][str(label)].append(index)
    return {dimension: {label: evaluate_predictions([rows[i] for i in subset], [predicted[i] for i in subset])
                        for label, subset in values.items()}
            for dimension, values in indices.items()}


def _paired_bootstrap(candidate: dict, baseline: dict, rows: list[dict], plan: dict, *, seed: int = 20261101) -> dict:
    from .protocol import case_groups
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[case_groups(row, plan)["family"]].append(index)
    labels = sorted(groups)
    result = {"method": "paired shape-family cluster bootstrap", "resampling_unit": "registered_NK_shape_family",
              "n_shape_families": len(labels), "resamples": 1000, "seed": seed,
              "difference_definition": "candidate median APE minus baseline median APE, percentage points; negative favors candidate",
              "observed_difference_pp": candidate["median_ape_pct"] - baseline["median_ape_pct"],
              "limitations": "Descriptive resampling of this registered set; few families, unmodeled dependence and session drift limit inference. Raw event repeats are never resampled as independent sessions. Not a prediction interval or guaranteed population coverage."}
    if len(labels) < 2:
        result.update(status="unavailable_fewer_than_two_families", percentile_95_interval_pp=None)
        return result
    a = np.array([item["ape_pct"] for item in candidate["predictions"]])
    b = np.array([item["ape_pct"] for item in baseline["predictions"]])
    rng = np.random.default_rng(seed)
    values = []
    for sample in rng.integers(0, len(labels), size=(1000, len(labels))):
        indices = np.concatenate([groups[labels[index]] for index in sample])
        values.append(float(np.median(a[indices]) - np.median(b[indices])))
    result.update(status="descriptive_only", percentile_95_interval_pp=np.percentile(values, [2.5, 97.5]).tolist())
    return result


def evaluate_suite(suite: dict, rows: list[dict], plan: dict, *, target_environment: dict | None = None,
                   transfer_mode: str | None = None) -> dict:
    _check_seal(suite, "suite_hash")
    if suite.get("protocol_version") != VERSION or suite.get("artifact_type") != "linear_perf_suite":
        raise LabError("Unsupported suite")
    context = _evaluation_context(suite, rows, plan, target_environment, transfer_mode)
    if set(suite["models"]) != set(MODEL_ORDER) or suite["selected_model_id"] not in suite["models"]:
        raise LabError("Frozen model menu or selected model is invalid")
    metrics, groups = {}, {}
    for name, model in suite["models"].items():
        if (model.get("model_id") != name or any(model.get(key) != suite[key] for key in _META)
                or model.get("train_case_ids") != suite["train_case_ids"]
                or model.get("training_data_hash") != suite["training_data_hash"]):
            raise LabError("Model metadata differs from its frozen suite")
        prediction = predict_model(model, rows, plan["dtype"])
        result = evaluate_predictions(rows, prediction)
        for row, record in zip(rows, result["predictions"]):
            record["ood"] = _ood(row, model["training_domain"])
        metrics[name] = result
        groups[name] = _groups(rows, prediction, plan)
    selected = suite["selected_model_id"]
    comparison = _paired_bootstrap(metrics[selected], metrics["roofline_3p"], rows, plan)
    return {"schema_version": 1, "protocol_version": VERSION, "measurement_kind": suite["measurement_kind"],
            "suite_hash": suite["suite_hash"], "plan_hash": digest(plan),
            "evaluated_case_ids": [row["case_id"] for row in rows],
            "evaluation_data_hash": digest(_training_evidence(rows)),
            "selected_model_id": selected, "selection_source": "development_only_before_heldout",
            "models": metrics, "groups": groups, "transfer": context,
            "selected_vs_roofline_3p": comparison,
            "heldout_used_for_fitting_or_selection": False,
            "interpretation": "Point accuracy, tail errors, ordering, OOD flags and timing variability must be considered together. No retuning is performed here."}


# Re-export the acquisition APIs lazily, keeping a small dependency cycle out of import time.
def fit_acquisition(pool_rows: list[dict], train_rows: list[dict], development_rows: list[dict],
                    plan: dict, *, environment: dict | None = None) -> dict:
    from .acquisition import fit_acquisition as implementation
    return implementation(pool_rows, train_rows, development_rows, plan, environment=environment)


def evaluate_acquisition(acquisition: dict, heldout_rows: list[dict], plan: dict,
                         *, target_environment: dict | None = None) -> dict:
    from .acquisition import evaluate_acquisition as implementation
    return implementation(acquisition, heldout_rows, plan, target_environment=target_environment)
