"""Deterministic, latency-free GEMM workloads derived from public model configs.

The declared workload mixture is a design choice, not a production trace. All
copies of a shared (M, N, K) shape have one case and one split. Test M values are
unseen in training; challenge (N, K) families are wholly held out. Labels may be
collected in any order and are isolated by the modeling code, not by acquisition
time. This module deliberately imports no tensor library and downloads nothing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .common import LabError

PROTOCOL_VERSION = "1.1.0"
SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "configs" / "model_sources"
_TRAIN_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_TEST_M = (3, 12, 48, 192, 768, 3072)
_LAYOUTS = ("separate_projections", "fused_qkv_and_gate_up")
_MEASUREMENT = {
    "visits": 3, "repeats": 9, "warmup_min_seconds": 0.2,
    "target_replay_ms": 2.0, "min_operations": 1, "max_operations": 256,
    "max_wall_seconds": 7200, "rotation_enabled": True, "rotation_buffers": 4,
    "rotation_l2_multiple": 2.0, "max_case_bytes": 512 * 1024 ** 2,
    "memory_fraction": 0.65, "profile": True,
}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise LabError(f"{name} must be a positive integer")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise LabError(f"{name} must be a positive finite number")
    return value


def parse_model_config(config, *, model_id="model"):
    """Extract dense decoder dimensions without assuming K/V have H outputs."""
    if not isinstance(config, dict) or config.get("model_type") not in {"llama", "qwen2"}:
        raise LabError("Only the declared dense Llama and Qwen2 configuration families are supported")
    required = ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "vocab_size")
    spec = {key: _positive_int(config.get(key), key) for key in required}
    spec["num_key_value_heads"] = _positive_int(config.get("num_key_value_heads", spec["num_attention_heads"]), "num_key_value_heads")
    if "head_dim" in config:
        spec["head_dim"] = _positive_int(config["head_dim"], "head_dim")
    else:
        if spec["hidden_size"] % spec["num_attention_heads"]:
            raise LabError("An implicit head dimension requires hidden_size divisible by num_attention_heads")
        spec["head_dim"] = spec["hidden_size"] // spec["num_attention_heads"]
    if spec["num_attention_heads"] % spec["num_key_value_heads"]:
        raise LabError("Grouped query attention requires an integer query/KV head ratio")
    if spec["head_dim"] % 2:
        raise LabError("The declared rotary embedding convention requires an even head dimension")
    if config.get("hidden_act", "silu") != "silu":
        raise LabError("This workload builder supports the declared SiLU gated MLP only")
    if config.get("use_sliding_window", False) or config.get("rope_scaling"):
        raise LabError("Sliding-window attention or scaled RoPE requires a separate implementation contract")
    spec.update(
        id=model_id, model_type=config["model_type"],
        max_position_embeddings=_positive_int(config.get("max_position_embeddings"), "max_position_embeddings"),
        rms_norm_eps=_positive_number(config.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
        rope_theta=_positive_number(config.get("rope_theta", 10000.0), "rope_theta"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        hidden_act="silu",
        attention_bias=bool(config.get("attention_bias", config["model_type"] == "qwen2")),
    )
    return spec


def load_model_specs(source_dir=None):
    """Load pinned source bytes, verify their digest, and return provenance/specs."""
    directory = Path(source_dir) if source_dir is not None else SOURCE_DIRECTORY
    try:
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LabError(f"Cannot read pinned model source provenance: {directory}") from exc
    rows = []
    for source in provenance.get("sources", []):
        filename = source.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise LabError("Model source filenames must refer to files in the source directory")
        try:
            raw = (directory / filename).read_bytes()
        except OSError as exc:
            raise LabError(f"Missing model config snapshot: {filename}") from exc
        if hashlib.sha256(raw).hexdigest() != source.get("sha256") or len(raw) != source.get("bytes"):
            raise LabError(f"Pinned model config bytes do not match provenance: {filename}")
        try:
            config = json.loads(raw)
        except ValueError as exc:
            raise LabError(f"Invalid model config JSON: {filename}") from exc
        if source.get("revision", "") not in source.get("url", "") or len(source.get("revision", "")) != 40:
            raise LabError("Model sources must point to immutable 40-character revisions")
        rows.append({**copy.deepcopy(source), "spec": parse_model_config(config, model_id=source["id"])})
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise LabError("Pinned model source IDs must be nonempty and unique")
    return rows


def projection_shapes(spec, layout="separate_projections"):
    """Return one model pass's linear operation counts for an explicit layout.

    QKV and gate/up fusion are alternative GEMM layouts. They are never counted
    together with their unfused equivalents inside the same layout. The lm_head
    is one operation even when its weight is tied to the token embedding.
    """
    if layout not in _LAYOUTS:
        raise LabError(f"Unknown projection layout: {layout}")
    h, intermediate, layers = (spec[key] for key in ("hidden_size", "intermediate_size", "num_hidden_layers"))
    q = spec["num_attention_heads"] * spec["head_dim"]
    kv = spec["num_key_value_heads"] * spec["head_dim"]
    if layout == "separate_projections":
        operations = (("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q),
                      ("gate", intermediate, h), ("up", intermediate, h), ("down", h, intermediate))
    else:
        operations = (("qkv", q + 2 * kv, h), ("o", h, q),
                      ("gate_up", 2 * intermediate, h), ("down", h, intermediate))
    return [{"operation": name, "n": n, "k": k, "occurrences": layers} for name, n, k in operations] + [
        {"operation": "lm_head", "n": spec["vocab_size"], "k": h, "occurrences": 1}]


def baseline_seen_shapes():
    """The original published 104 cases are already exposed, not fresh tests."""
    from .protocol import build_plan
    return {(case["m"], case["n"], case["k"]) for case in build_plan()["cases"]}


def _as_shapes(items):
    result = set()
    for item in items or ():
        shape = tuple(item[key] for key in ("m", "n", "k")) if isinstance(item, dict) else tuple(item)
        if len(shape) != 3:
            raise LabError("Seen shapes must contain (M, N, K)")
        for dimension in shape:
            _positive_int(dimension, "seen shape dimension")
        result.add(shape)
    return result


def _family(n, k):
    return f"nk_{n}_{k}"


def _rank(seed, family):
    return hashlib.sha256(f"{seed}:{family[0]}:{family[1]}".encode("ascii")).hexdigest()


def _ideal_bytes(m, n, k):
    # Both declared dtypes occupy two bytes. This is not a CUDA workspace bound.
    return 2 * (m * k + n * k + m * n)


def build_workload_manifest(*, profile="standard", seed=20261106,
                            dtypes=("float16", "bfloat16"), source_dir=None, seen_shapes=None):
    """Build about 600 unique shapes per dtype without using latency labels.

    The smoke profile is for functional checks and has deliberately limited
    coverage. GPU memory/time checks may record further skips at runtime. Neither
    profile promises an exact runtime or a target prediction accuracy.
    """
    if profile not in {"standard", "smoke"}:
        raise LabError("Workload profile must be standard or smoke")
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise LabError("Workload seed must be an integer in [0, 2**31)")
    if not isinstance(dtypes, (tuple, list)) or not dtypes or not all(isinstance(value, str) for value in dtypes) or len(set(dtypes)) != len(dtypes) or not set(dtypes) <= {"float16", "bfloat16"}:
        raise LabError("Dtypes must be a unique nonempty list of float16 and/or bfloat16")
    sources = load_model_specs(source_dir)
    already_seen = baseline_seen_shapes() | _as_shapes(seen_shapes)
    families = defaultdict(list)
    model_probability = 1.0 / len(sources)
    for source in sources:
        for layout in _LAYOUTS:
            for op in projection_shapes(source["spec"], layout):
                families[(op["n"], op["k"])].append({
                    "model_id": source["id"], "revision": source["revision"],
                    "layout": layout, "operation": op["operation"],
                    "occurrences_per_model_pass": op["occurrences"],
                    "model_mixture_probability": model_probability, "layout_probability": 0.5,
                    "weighted_occurrences": model_probability * 0.5 * op["occurrences"],
                })
    measurement = copy.deepcopy(_MEASUREMENT)
    train_m, test_m = _TRAIN_M, _TEST_M
    if profile == "smoke":
        # Six smallest dense families preserve real dimensions while limiting work.
        selected = sorted(families, key=lambda nk: (nk[0] * nk[1], nk))[:6]
        families = {key: families[key] for key in selected}
        train_m, test_m = (1, 16, 256), (3, 48)
        measurement.update(visits=2, repeats=3, warmup_min_seconds=0.05, max_wall_seconds=600)
    all_m = sorted(set(train_m) | set(test_m))
    eligible = [nk for nk in families if any(_ideal_bytes(m, *nk) <= measurement["max_case_bytes"] for m in all_m)]
    challenge_count = max(1, round(len(eligible) * 0.2))
    challenge = set(sorted(eligible, key=lambda nk: _rank(seed, nk))[:challenge_count])
    cases, excluded = [], []
    family_metadata = {}
    for (n, k), contributions in sorted(families.items()):
        family_id = _family(n, k)
        occurrence_weight = sum(row["weighted_occurrences"] for row in contributions)
        family_metadata[family_id] = {
            "n": n, "k": k, "provenance": contributions,
            "weighted_occurrences_per_mixture_pass": occurrence_weight,
            "challenge_family": (n, k) in challenge,
        }
        for m in all_m:
            split = "challenge" if (n, k) in challenge else "train" if m in train_m else "test"
            shape = {"m": m, "n": n, "k": k, "split": split, "family": family_id}
            if _ideal_bytes(m, n, k) > measurement["max_case_bytes"]:
                excluded.append({**shape, "reason": "ideal_tensor_bytes_exceed_declared_per_case_budget",
                                 "ideal_tensor_bytes": _ideal_bytes(m, n, k)})
                continue
            if split != "train" and (m, n, k) in already_seen:
                excluded.append({**shape, "reason": "already_exposed_shape_not_an_independent_evaluation"})
                continue
            cases.append({
                "id": f"m{m}_n{n}_k{k}", **shape,
                "weight": occurrence_weight / len(all_m),
                "m_sampling_probability": 1.0 / len(all_m),
                "origin": "pinned_model_architecture_config",
                "regime": "decode_shaped" if m <= 256 else "prefill_shaped",
                "ideal_tensor_bytes": _ideal_bytes(m, n, k),
                "provenance_ref": family_id,
            })
    # Deliberate neighboring dimensions are a separately scored robustness set.
    # They are not presented as projections that exist in the source models.
    stress_count = 2 if profile == "smoke" else 12
    stress_m = (7,) if profile == "smoke" else (1, 16, 256, 2048)
    stress_parents = sorted(eligible, key=lambda nk: (_rank(seed + 1, nk), nk))[:stress_count]
    existing_shapes = {(case["m"], case["n"], case["k"]) for case in cases}
    for parent_n, parent_k in stress_parents:
        n, k = parent_n + 1, parent_k + 1
        if (n, k) in families:
            raise LabError("Irregular stress family unexpectedly overlaps a model-derived family")
        for m in stress_m:
            shape = {"m": m, "n": n, "k": k, "split": "stress", "family": _family(n, k)}
            if _ideal_bytes(m, n, k) > measurement["max_case_bytes"]:
                excluded.append({**shape, "reason": "ideal_tensor_bytes_exceed_declared_per_case_budget",
                                 "ideal_tensor_bytes": _ideal_bytes(m, n, k)})
                continue
            if (m, n, k) in already_seen:
                excluded.append({**shape, "reason": "already_exposed_shape_not_an_independent_evaluation"})
                continue
            if (m, n, k) in existing_shapes:
                raise LabError("Duplicate stress shape")
            cases.append({"id": f"m{m}_n{n}_k{k}", **shape, "weight": 1.0,
                          "origin": "artificial_irregular_stress", "regime": "stress",
                          "ideal_tensor_bytes": _ideal_bytes(m, n, k),
                          "parent_family": _family(parent_n, parent_k),
                          "perturbation": {"n_delta": 1, "k_delta": 1}})
            existing_shapes.add((m, n, k))
    # Budget exclusions can truncate a vocabulary projection's feasible M range.
    # Preserve that fact instead of claiming every ordinary test is interpolation.
    training_m_by_family = defaultdict(list)
    for case in cases:
        if case["split"] == "train":
            training_m_by_family[case["family"]].append(case["m"])
    for family_id, metadata in family_metadata.items():
        metadata["training_m_values_after_budget_exclusions"] = sorted(training_m_by_family[family_id])
    for case in cases:
        if case["split"] == "test":
            training_values = training_m_by_family[case["family"]]
            case["training_m_range"] = [min(training_values), max(training_values)]
            case["coverage"] = ("unseen_M_inside_training_family_range" if min(training_values) <= case["m"] <= max(training_values)
                                else "unseen_M_outside_training_family_range_after_budget_exclusions")
        else:
            case["coverage"] = {"train": "training", "challenge": "wholly_unseen_NK_family", "stress": "artificial_irregular_stress"}[case["split"]]
    counts = dict(Counter(case["split"] for case in cases))
    manifest = {
        "schema_version": 2, "protocol_version": PROTOCOL_VERSION,
        "name": f"model_config_{profile}_workload", "profile": profile, "seed": seed,
        "workflow": "collect_all_then_grouped_cv_refit_and_evaluate",
        "dtypes": list(dtypes), "cases": cases, "measurement": measurement,
        "source_metadata": sources, "families": family_metadata,
        "counts": {"unique_shapes_per_dtype": len(cases), "by_split": counts,
                   "model_config_families": len(families), "challenge_families": len(challenge),
                   "planning_exclusions": len(excluded)},
        "excluded_cases": excluded,
        "previously_exposed_shapes": [list(shape) for shape in sorted(already_seen)],
        "split_policy": {
            "train_m": list(train_m), "test_m": list(test_m),
            "challenge_assignment": "shape-only seeded hash of eligible (N,K) families; all their M values are held out",
            "challenge_families": [_family(*nk) for nk in sorted(challenge)],
            "test": "new M values in (N,K) families represented in training",
            "test_range": "Per-case coverage distinguishes interpolation from extrapolation after per-family memory exclusions",
            "stress": "artificial +1 changes to N and K; separate from model selection and ordinary test",
            "grouped_cv_key": ["n", "k"],
            "all_labels_may_be_collected_before_fitting": True,
            "test_challenge_stress_used_for_selection": False,
            "source_baseline_exclusion": "all 104 already-published baseline shapes are excluded from new test/challenge/stress",
        },
        "weight_policy": {
            "interpretation": "Architectural operation occurrences under a declared synthetic workload mixture; not observed serving traffic",
            "model_probability": "uniform over the five pinned model configurations",
            "layout_probability": {layout: 0.5 for layout in _LAYOUTS},
            "m_probability": "uniform over the profile's combined train/test M grid before budget exclusions",
            "case_weight": "sum(model_probability * layout_probability * operation_occurrences) / number_of_M_values",
            "deduplication": "Shared (M,N,K) operations aggregate their weights into a single case and split",
            "stress_weight": "uniform, reported separately; no architectural frequency interpretation",
            "budget_exclusions": "Remaining weights are normalized only when computing each reported metric; excluded shapes remain listed",
        },
        "scope": {
            "operation": "Y[M,N] = X[M,K] @ W[N,K].T; preallocated output; no bias/activation in the GEMM target",
            "fused_layout": "A separate synthetic alternative concatenates Q/K/V and gate/up output dimensions; it does not prove a deployed framework uses that fusion",
            "m_interpretation": "M is the number of rows in one GEMM, not a unique batch/sequence pairing; decode-shaped and prefill-shaped are coverage labels",
            "lm_head": "All-token vocabulary projection sweep; actual serving may project only selected token positions",
            "context_length": "GEMM row sweeps are shape experiments and may exceed a source model's native context; the decoder-layer validator enforces context limits",
            "budget": "512 MiB bounds ideal input/weight/output tensors only. Runtime checks include free memory, references and rotated buffers and record skips without shrinking model dimensions",
            "runtime": "Adaptive graph work and a wall-time budget bound the run; no fixed GPU runtime or accuracy is promised",
            "smoke": "The smoke profile checks software plumbing and is not representative performance evidence",
        },
    }
    return validate_workload_manifest(manifest)


def validate_workload_manifest(manifest):
    """Validate split isolation and shapes while allowing measurement overrides."""
    if not isinstance(manifest, dict) or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise LabError("Expected a workload manifest with protocol_version 1.1.0")
    dtypes = manifest.get("dtypes")
    if not isinstance(dtypes, list) or not dtypes or not all(isinstance(value, str) for value in dtypes) or len(set(dtypes)) != len(dtypes) or not set(dtypes) <= {"float16", "bfloat16"}:
        raise LabError("Manifest dtypes must be unique float16 and/or bfloat16 entries")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise LabError("Workload cases must be a nonempty list")
    seen_ids, seen_shapes = set(), set()
    families = defaultdict(set)
    exposed = baseline_seen_shapes() | _as_shapes(manifest.get("previously_exposed_shapes", []))
    for case in cases:
        if not isinstance(case, dict):
            raise LabError("Every workload case must be an object")
        cid = case.get("id")
        if not isinstance(cid, str) or not cid or cid in seen_ids:
            raise LabError("Workload case IDs must be nonempty and unique")
        seen_ids.add(cid)
        shape = tuple(_positive_int(case.get(key), key) for key in ("m", "n", "k"))
        if shape in seen_shapes:
            raise LabError("Duplicate (M,N,K) shapes are not independent cases")
        seen_shapes.add(shape)
        split = case.get("split")
        if split not in {"train", "test", "challenge", "stress"}:
            raise LabError("Workload splits must be train, test, challenge or stress")
        _positive_number(case.get("weight"), "case weight")
        if split != "train" and shape in exposed:
            raise LabError("An already-exposed shape cannot be a fresh held-out case")
        if case.get("family", _family(shape[1], shape[2])) != _family(shape[1], shape[2]):
            raise LabError("Case family must match its (N,K) dimensions")
        families[split].add(shape[1:])
    if len(families["train"]) < 3:
        raise LabError("Grouped cross-validation needs at least three training (N,K) families")
    if not families["test"] or not families["challenge"]:
        raise LabError("Workload must include ordinary test and held-out challenge cases")
    if not families["test"] <= families["train"]:
        raise LabError("Ordinary test must use (N,K) families present in training")
    if families["challenge"] & (families["train"] | families["test"]):
        raise LabError("Challenge (N,K) families must be wholly absent from train/test")
    if families["stress"] & (families["train"] | families["test"] | families["challenge"]):
        raise LabError("Irregular stress families must be separate from model-derived families")
    declared = manifest.get("split_policy", {})
    if declared.get("test_challenge_stress_used_for_selection", False):
        raise LabError("Held-out and stress labels cannot be used for model selection")
    return manifest
