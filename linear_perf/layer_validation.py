"""Optional CUDA validation on one synthetic, random-weight decoder layer.

This is a downstream experiment, not predictor training data. A hybrid estimate
adds predicted linear GEMM times to independently measured non-GEMM component
times and compares that sum with a measured complete layer. It is not a fully
predicted model, token-generation benchmark, or production inference engine.
Importing this module needs neither PyTorch nor CUDA.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from pathlib import Path

from .common import LabError, digest, utc_now, write_json
from .workloads import parse_model_config, projection_shapes

_SOURCES = [
    "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
    "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py",
    "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py",
]


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise LabError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise LabError(f"{name} must be positive and finite")
    return float(value)


def validate_layer_spec(spec, *, batch_size=1, sequence_length=128):
    """Validate exact dimensions and native context; never silently shrink them."""
    if not isinstance(spec, dict):
        raise LabError("Layer specification must be an object")
    checked = parse_model_config(spec, model_id=spec.get("id", "synthetic_decoder"))
    _positive_int(batch_size, "batch_size")
    _positive_int(sequence_length, "sequence_length")
    if sequence_length > checked["max_position_embeddings"]:
        raise LabError("Layer sequence length exceeds the source model's native context limit")
    if checked["model_type"] == "llama" and checked["attention_bias"]:
        raise LabError("Biased Llama projections require an additional declared implementation; pinned Llama configs are bias-free")
    return checked


def layer_gemm_cases(spec, *, batch_size=1, sequence_length=128):
    """Geometry-only prediction requests, one for each of seven dense GEMMs."""
    checked = validate_layer_spec(spec, batch_size=batch_size, sequence_length=sequence_length)
    return [{"id": op["operation"], "m": batch_size * sequence_length, "n": op["n"], "k": op["k"], "weight": 1.0}
            for op in projection_shapes(checked) if op["operation"] != "lm_head"]


def estimate_layer_bytes(spec, *, batch_size=1, sequence_length=128):
    """Conservative planning guard, not a claim about actual allocator peak use.

    Include represented weights plus FP32 references, retained component inputs,
    a dense-attention fallback allowance and workspace. A runtime OOM can still
    occur; it is recorded and never causes automatic dimension reduction.
    """
    checked = validate_layer_spec(spec, batch_size=batch_size, sequence_length=sequence_length)
    cases = layer_gemm_cases(checked, batch_size=batch_size, sequence_length=sequence_length)
    h, intermediate, head_dim = (checked[key] for key in ("hidden_size", "intermediate_size", "head_dim"))
    q = checked["num_attention_heads"] * head_dim
    kv = checked["num_key_value_heads"] * head_dim
    parameter_elements = sum(case["n"] * case["k"] for case in cases) + 2 * h
    if checked["attention_bias"]:
        parameter_elements += q + 2 * kv
    weight_and_reference = 6 * parameter_elements
    initialization_temporary = 4 * max(case["n"] * case["k"] for case in cases)
    # Extra factors cover graph pools and references; no measured bytes claim.
    activation_allowance = 12 * 6 * batch_size * sequence_length * (h + intermediate + q + 2 * kv)
    attention_allowance = 16 * batch_size * checked["num_attention_heads"] * sequence_length ** 2
    workspace_allowance = 512 * 1024 ** 2
    return {"parameter_elements": parameter_elements, "weight_and_reference_bytes": weight_and_reference,
            "initialization_temporary_bytes": initialization_temporary,
            "activation_allowance_bytes": activation_allowance, "attention_allowance_bytes": attention_allowance,
            "workspace_allowance_bytes": workspace_allowance,
            "estimated_peak_bytes": weight_and_reference + initialization_temporary + activation_allowance + attention_allowance + workspace_allowance,
            "interpretation": "Conservative preflight allocation estimate, not measured peak memory or a guarantee"}


def _rms_norm(torch, values, scale, epsilon):
    normalized = values.float()
    inverse = (normalized.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()
    return (normalized * inverse).to(values.dtype) * scale


def _rotate(torch, values, cosine, sine):
    half = values.shape[-1] // 2
    perpendicular = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return values * cosine + perpendicular * sine


class _SyntheticDecoder:
    """Inference-only local implementation with preallocated linear outputs.

    The mathematical structure follows the cited decoder architectures. The
    sequence is prefill with no padding or KV cache. PyTorch chooses the SDPA
    backend; a fast backend is not asserted from its API name.
    """
    def __init__(self, torch, spec, *, batch_size, sequence_length, dtype, device, seed, represented=None):
        self.torch, self.spec = torch, spec
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.dtype, self.device = dtype, device
        self.shapes = layer_gemm_cases(spec, batch_size=batch_size, sequence_length=sequence_length)
        self.outputs = {case["id"]: torch.empty((case["m"], case["n"]), dtype=dtype, device=device) for case in self.shapes}
        if represented is None:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            h, d = spec["hidden_size"], spec["head_dim"]
            self.x = torch.randn((batch_size * sequence_length, h), generator=rng, dtype=torch.float32, device=device).to(dtype)
            self.weights = {case["id"]: (torch.randn((case["n"], case["k"]), generator=rng, dtype=torch.float32, device=device)
                                        / math.sqrt(case["k"])).to(dtype) for case in self.shapes}
            self.norm_scales = [torch.ones(h, dtype=dtype, device=device) for _ in range(2)]
            self.biases = {}
            if spec["attention_bias"]:
                self.biases = {case["id"]: (0.01 * torch.randn(case["n"], generator=rng, dtype=torch.float32, device=device)).to(dtype)
                               for case in self.shapes if case["id"] in {"q", "k", "v"}}
            positions = torch.arange(sequence_length, dtype=torch.float32, device=device)
            inverse_frequency = spec["rope_theta"] ** (-torch.arange(0, d, 2, dtype=torch.float32, device=device) / d)
            angles = positions[:, None] * inverse_frequency[None, :]
            angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
            self.cosine, self.sine = angles.cos().to(dtype), angles.sin().to(dtype)
        else:
            # Exact same represented input/weight/rotary constants in FP32.
            self.x = represented.x.to(dtype)
            self.weights = {name: weight.to(dtype) for name, weight in represented.weights.items()}
            self.norm_scales = [value.to(dtype) for value in represented.norm_scales]
            self.biases = {name: value.to(dtype) for name, value in represented.biases.items()}
            self.cosine, self.sine = represented.cosine.to(dtype), represented.sine.to(dtype)
        self.transposes = {name: weight.t() for name, weight in self.weights.items()}

    def mm(self, name, x):
        self.torch.mm(x, self.transposes[name], out=self.outputs[name])
        return self.outputs[name]

    def prepare_qkv(self, q, k, v):
        b, s, d = self.batch_size, self.sequence_length, self.spec["head_dim"]
        if self.biases:
            q, k, v = q + self.biases["q"], k + self.biases["k"], v + self.biases["v"]
        q = q.view(b, s, self.spec["num_attention_heads"], d).transpose(1, 2)
        k = k.view(b, s, self.spec["num_key_value_heads"], d).transpose(1, 2)
        v = v.view(b, s, self.spec["num_key_value_heads"], d).transpose(1, 2)
        return _rotate(self.torch, q, self.cosine, self.sine), _rotate(self.torch, k, self.cosine, self.sine), v

    def attention(self, prepared):
        return self.torch.nn.functional.scaled_dot_product_attention(
            *prepared, dropout_p=0.0, is_causal=True,
            enable_gqa=self.spec["num_attention_heads"] != self.spec["num_key_value_heads"])

    def forward(self, *, record=False):
        tape = []
        shapes = {case["id"]: case for case in self.shapes}

        def execute(name, kind, operation):
            result = operation()
            if record:
                metadata = {"id": name, "kind": kind}
                if kind == "gemm":
                    metadata.update(shapes[name])
                tape.append((metadata, operation))
            return result

        norm1 = execute("input_rmsnorm", "non_gemm", lambda: _rms_norm(self.torch, self.x, self.norm_scales[0], self.spec["rms_norm_eps"]))
        q = execute("q", "gemm", lambda: self.mm("q", norm1))
        k = execute("k", "gemm", lambda: self.mm("k", norm1))
        v = execute("v", "gemm", lambda: self.mm("v", norm1))
        prepared = execute("qkv_bias_reshape_rope", "non_gemm", lambda: self.prepare_qkv(q, k, v))
        attended = execute("causal_attention", "non_gemm", lambda: self.attention(prepared))
        packed = execute("attention_layout", "non_gemm", lambda: attended.transpose(1, 2).contiguous().view(self.batch_size * self.sequence_length, -1))
        projected = execute("o", "gemm", lambda: self.mm("o", packed))
        residual1 = execute("attention_residual", "non_gemm", lambda: self.x + projected)
        norm2 = execute("post_attention_rmsnorm", "non_gemm", lambda: _rms_norm(self.torch, residual1, self.norm_scales[1], self.spec["rms_norm_eps"]))
        gate = execute("gate", "gemm", lambda: self.mm("gate", norm2))
        up = execute("up", "gemm", lambda: self.mm("up", norm2))
        activated = execute("silu_and_multiply", "non_gemm", lambda: self.torch.nn.functional.silu(gate) * up)
        down = execute("down", "gemm", lambda: self.mm("down", activated))
        output = execute("mlp_residual", "non_gemm", lambda: residual1 + down)
        return (output, tape) if record else output


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise LabError("Layer validation reached its wall-time budget; no dimensions were changed")


def _measure_graph(torch, selected, operation, *, deadline, visits, repeats,
                   warmup_min_seconds, target_replay_ms, max_operations, poison_output=False):
    """Event means of repeated graph work, with fixed-duration visit conditioning."""
    from .benchmark import _capture_operation_graph, _event_pair, _event_sample
    holder = {}

    def run():
        holder["output"] = operation()

    for _ in range(3):
        _check_deadline(deadline)
        run()
    torch.cuda.synchronize(selected)
    events = _event_pair(torch)
    pilot = [_event_sample(run, events, 4) for _ in range(3)]
    operations = max(1, min(max_operations, math.ceil(target_replay_ms / statistics.median(pilot))))
    graph = _capture_operation_graph(torch, selected, run, {"warmup": 3, "iterations": operations}, deadline)
    if poison_output:
        # A valid eager/capture-time result cannot pass for a missing replay.
        holder["output"].fill_(float("nan"))
        torch.cuda.synchronize(selected)
    blocks = []
    for visit in range(visits):
        begin, replays = time.monotonic(), 0
        while replays < 3 or time.monotonic() - begin < warmup_min_seconds:
            _check_deadline(deadline)
            graph.replay()
            torch.cuda.synchronize(selected)
            replays += 1
            if replays >= 4096:
                raise LabError("Layer graph warmup exhausted its fixed replay ceiling")
        elapsed = time.monotonic() - begin
        samples = []
        for _ in range(repeats):
            _check_deadline(deadline)
            samples.append(_event_sample(graph.replay, events, 1) / operations)
        median = statistics.median(samples)
        blocks.append({"visit": visit, "samples_ms": samples, "median_ms": median,
                       "warmup_replays": replays, "warmup_wall_seconds": elapsed,
                       "sample_cv_pct": 100 * statistics.pstdev(samples) / statistics.mean(samples)})
    medians = [block["median_ms"] for block in blocks]
    return {"median_ms": statistics.median(medians), "visits": blocks,
            "captured_operations": operations, "pilot_ms": pilot,
            "target_replay_ms": target_replay_ms,
            "estimated_timed_replay_ms": statistics.median(medians) * operations,
            "visit_max_min_ratio": max(medians) / min(medians),
            "timer": "CUDA events around one graph replay, divided by captured operations",
            "aggregation": "median of visit medians", "capture_and_warmup_excluded": True,
            "output_poisoned_after_capture": poison_output}, holder["output"]


def benchmark_layer(spec, *, batch_size=1, sequence_length=128, dtype="float16", device="cuda:0",
                    output_dir=None, seed=20261106, visits=3, repeats=9,
                    warmup_min_seconds=0.1, target_replay_ms=2.0, max_operations=32,
                    max_wall_seconds=600, max_layer_bytes=4 * 1024 ** 3,
                    memory_fraction=0.65, progress=print):
    """Measure a complete synthetic decoder and its components on CUDA only.

    Defaults are intentionally small. The result keeps raw event samples,
    numerical checks, component timing order and a repeated full-layer drift
    check. A memory preflight skip is an explicit result, never a smaller model.
    """
    checked = validate_layer_spec(spec, batch_size=batch_size, sequence_length=sequence_length)
    if dtype not in {"float16", "bfloat16"}:
        raise LabError("Layer validation supports float16 and bfloat16")
    for name, value in (("visits", visits), ("repeats", repeats), ("max_operations", max_operations), ("max_layer_bytes", max_layer_bytes)):
        _positive_int(value, name)
    for name, value in (("warmup_min_seconds", warmup_min_seconds), ("target_replay_ms", target_replay_ms), ("max_wall_seconds", max_wall_seconds)):
        _positive_number(value, name)
    if not 0 < _positive_number(memory_fraction, "memory_fraction") <= 0.9:
        raise LabError("memory_fraction must be at most 0.9")
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise LabError("Layer seed must be an integer in [0, 2**31)")
    folder = Path(output_dir).resolve() if output_dir is not None else None
    if folder is not None:
        if folder.exists():
            raise LabError("Layer output directory already exists; use a fresh path")
        folder.mkdir(parents=True)
    record = {"protocol_version": "1.1.0", "artifact_type": "synthetic_decoder_layer_validation",
              "measurement_kind": "cuda_measured", "status": "initializing", "created_at_utc": utc_now(),
              "model_spec": checked, "batch_size": batch_size, "sequence_length": sequence_length,
              "m": batch_size * sequence_length, "dtype": dtype, "cache_policy": "warm_reuse",
              "seed": seed, "components": [], "excluded_from_predictor_training": True,
              "architecture_sources": _SOURCES,
              "scope": {
                  "includes": "two RMSNorms, separate Q/K/V/O GEMMs, rotary embeddings, causal GQA SDPA, gated SiLU MLP and residual additions",
                  "excludes": "token embeddings, final model normalization, lm_head, KV caching, decoding scheduler, distributed communication and model weights",
                  "weights": "synthetic random weights; no pretrained weights are downloaded",
                  "attention_backend": "PyTorch SDPA automatic backend selection; backend identity is not inferred from the API name",
                  "non_gemm": "Everything outside the seven dense linear projections; includes attention matrix products, not only elementwise work",
                  "prediction": "Optional hybrid estimate uses model-predicted GEMMs plus measured non-GEMM components",
                  "component_context": "Components reuse standalone buffers; their sum can differ from a full-layer replay due to cache, allocation and scheduling effects",
                  "claim": "One prefill decoder layer, not full-model or serving latency",
              },
              "memory_plan": estimate_layer_bytes(checked, batch_size=batch_size, sequence_length=sequence_length)}

    def persist():
        if folder is not None:
            write_json(folder / "layer_measurements.json", record)

    persist()
    try:
        from .benchmark import _load_torch, _require_cuda, _precision_policy, _gpu_metadata, _driver_versions
        torch = _load_torch()
        selected = _require_cuda(torch, dtype, device)
        with torch.cuda.device(selected), torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), _precision_policy(torch) as policy:
            free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
            allowed = min(max_layer_bytes, int(memory_fraction * free_bytes))
            record["memory_plan"].update(free_bytes_at_start=free_bytes, total_device_bytes=total_bytes,
                                         allowed_bytes=allowed, memory_fraction=memory_fraction)
            if record["memory_plan"]["estimated_peak_bytes"] > allowed:
                record.update(status="skipped_memory_budget", reason="Original decoder dimensions exceed the declared preflight memory allowance")
                persist()
                return record
            props = torch.cuda.get_device_properties(selected)
            identity = {"device_name": str(props.name), "sm_count": int(props.multi_processor_count),
                        "compute_capability": list(torch.cuda.get_device_capability(selected)),
                        "pytorch_version": str(torch.__version__), "cuda_runtime_version": str(torch.version.cuda),
                        "gpu": _gpu_metadata(torch, selected), "driver_versions": _driver_versions(),
                        "math_policy": dict(policy), "dtype": dtype, "autocast": "disabled"}
            identity["gpu_uuid"] = identity["gpu"].get("uuid")
            record.update(environment=identity, environment_fingerprint=digest(identity), device=str(selected))
            deadline = time.monotonic() + max_wall_seconds
            represented = _SyntheticDecoder(torch, checked, batch_size=batch_size, sequence_length=sequence_length,
                                             dtype=getattr(torch, dtype), device=selected, seed=seed)
            reference_layer = _SyntheticDecoder(torch, checked, batch_size=batch_size, sequence_length=sequence_length,
                                                 dtype=torch.float32, device=selected, seed=seed, represented=represented)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                reference = reference_layer.forward().clone()
            del reference_layer
            full, actual = _measure_graph(torch, selected, represented.forward, deadline=deadline, visits=visits, repeats=repeats,
                                          warmup_min_seconds=warmup_min_seconds, target_replay_ms=target_replay_ms,
                                          max_operations=max_operations, poison_output=True)
            # Check the captured graph's actual final output, not a fresh eager result.
            actual_float = actual.float()
            finite = bool(torch.isfinite(actual_float).all().item() and torch.isfinite(reference).all().item())
            error = actual_float - reference
            ref_norm = float(torch.linalg.vector_norm(reference).item())
            max_reference = float(reference.abs().max().item())
            relative_l2 = float(torch.linalg.vector_norm(error).item()) / max(ref_norm, 1e-30)
            max_absolute = float(error.abs().max().item())
            max_scaled = max_absolute / max(max_reference, 1e-30)
            numeric_finite = all(math.isfinite(value) for value in (relative_l2, max_absolute, max_scaled))
            limits = {"relative_l2": 0.02 if dtype == "float16" else 0.15,
                      "max_error_over_reference_max": 0.04 if dtype == "float16" else 0.25}
            correctness = {"finite": finite, "relative_l2": relative_l2 if numeric_finite else None,
                           "max_abs_error": max_absolute if numeric_finite else None,
                           "max_error_over_reference_max": max_scaled if numeric_finite else None, "thresholds": limits,
                           "threshold_origin": "project-level gross-error checks for this synthetic composition, not vendor accuracy guarantees",
                           "reference": "FP32 full layer from the same already-represented inputs, weights, biases and rotary constants; IEEE matmul and explicit SDPA math backend; not a float64 oracle",
                           "passed": finite and numeric_finite and relative_l2 <= limits["relative_l2"] and max_scaled <= limits["max_error_over_reference_max"]}
            record.update(full_layer=full, numerical_check=correctness)
            persist()
            if not correctness["passed"]:
                raise LabError("Synthetic full-layer output failed its declared numerical checks")
            del reference, actual, actual_float, error
            _, tape = represented.forward(record=True)
            random.Random(seed).shuffle(tape)
            record["component_timing_order"] = [metadata["id"] for metadata, _ in tape]
            for metadata, operation in tape:
                if progress:
                    progress(f"[layer] timing {metadata['id']}")
                timing, _ = _measure_graph(torch, selected, operation, deadline=deadline, visits=visits, repeats=repeats,
                                           warmup_min_seconds=warmup_min_seconds, target_replay_ms=target_replay_ms, max_operations=max_operations)
                record["components"].append({**metadata, **timing, "excluded_from_predictor_training": True})
                persist()
            repeat, repeat_output = _measure_graph(torch, selected, represented.forward, deadline=deadline, visits=1, repeats=repeats,
                                       warmup_min_seconds=warmup_min_seconds, target_replay_ms=target_replay_ms,
                                       max_operations=max_operations, poison_output=True)
            if not bool(torch.isfinite(repeat_output).all().item()):
                raise LabError("Final full-layer graph replay did not produce a finite output")
            full_ms = full["median_ms"]
            component_ms = sum(component["median_ms"] for component in record["components"])
            record.update(full_layer_repeat=repeat,
                          full_layer_repeat_ratio=repeat["median_ms"] / full_ms,
                          measured_component_sum_ms=component_ms,
                          measured_component_sum_error_pct=100 * (component_ms - full_ms) / full_ms,
                          status="completed", finished_at_utc=utc_now())
            record["quality_flags"] = {
                "full_layer_changed_more_than_20pct": max(repeat["median_ms"], full_ms) / min(repeat["median_ms"], full_ms) > 1.2,
                "any_within_visit_cv_above_10pct": any(block["sample_cv_pct"] > 10 for timed in [full, repeat] + record["components"] for block in timed["visits"]),
                "policy": "All samples retained; thresholds are diagnostic project choices and never used to silently discard slow runs",
            }
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc), finished_at_utc=utc_now())
        persist()
        raise
    persist()
    return record


def compose_layer_estimate(record, predictions):
    """Compare a hybrid sum with actual complete-layer timing, without fitting.

    ``predictions`` maps the seven GEMM IDs to positive milliseconds. It must
    come from an already selected matching GPU/dtype/warm-reuse predictor. This
    function cannot prove that provenance from floats alone; callers attach the
    model artifact and enforce context compatibility.
    """
    if record.get("status") != "completed" or record.get("measurement_kind") != "cuda_measured":
        raise LabError("Hybrid comparison requires a completed, actual CUDA layer measurement")
    if not record.get("excluded_from_predictor_training"):
        raise LabError("Downstream layer timings must remain outside predictor fitting")
    components = record.get("components", [])
    gemms = [component for component in components if component.get("kind") == "gemm"]
    other = [component for component in components if component.get("kind") == "non_gemm"]
    expected = {case["id"] for case in layer_gemm_cases(record["model_spec"], batch_size=record["batch_size"], sequence_length=record["sequence_length"])}
    if len(gemms) != 7 or {component["id"] for component in gemms} != expected or not other:
        raise LabError("Layer record must include all seven GEMMs and non-GEMM measurements")
    if not isinstance(predictions, dict) or set(predictions) != expected:
        raise LabError("Predictions must map exactly the seven GEMM component IDs to milliseconds")
    predicted = {key: _positive_number(value, f"prediction {key}") for key, value in predictions.items()}
    full_ms = _positive_number(record["full_layer"]["median_ms"], "full-layer latency")
    measured_gemm = sum(_positive_number(component["median_ms"], "GEMM latency") for component in gemms)
    measured_other = sum(_positive_number(component["median_ms"], "non-GEMM latency") for component in other)
    predicted_gemm = sum(predicted.values())
    hybrid = predicted_gemm + measured_other
    return {
        "estimate_kind": "hybrid_predicted_linear_gemms_plus_measured_other_operations",
        "model_spec_id": record["model_spec"]["id"], "dtype": record["dtype"],
        "batch_size": record["batch_size"], "sequence_length": record["sequence_length"],
        "predicted_gemm_components_ms": predicted, "predicted_gemm_sum_ms": predicted_gemm,
        "measured_non_gemm_sum_ms": measured_other, "hybrid_estimate_ms": hybrid,
        "measured_full_layer_ms": full_ms, "absolute_percentage_error": 100 * abs(hybrid - full_ms) / full_ms,
        "signed_percentage_error": 100 * (hybrid - full_ms) / full_ms,
        "measured_gemm_sum_ms": measured_gemm, "measured_all_component_sum_ms": measured_gemm + measured_other,
        "measured_composition_gap_pct": 100 * (measured_gemm + measured_other - full_ms) / full_ms,
        "gemm_prediction_sum_error_pct": 100 * (predicted_gemm - measured_gemm) / measured_gemm,
        "non_gemm_fraction_of_measured_component_sum": measured_other / (measured_gemm + measured_other),
        "excluded_from_predictor_training": True, "fully_predicted_end_to_end_model": False,
        "interpretation": "Accuracy combines GEMM prediction error and non-additivity of isolated component timings. Non-GEMM times are measured inputs, not model predictions. No serving latency claim.",
        "context_requirement": "Same GPU/software/math-policy context and dtype as the selected warm_reuse model; caller must attach the selected model artifact",
    }
