"""Small archived numerical witnesses, independently recomputable on a CPU.

These witnesses are a bounded correctness supplement, not an archive of all
large benchmark tensors. NPZ files contain only numeric/unicode arrays and can
be loaded with allow_pickle=False. Stored float32 inputs exactly represent the
already rounded FP16/BF16 values used on the GPU.
"""

from __future__ import annotations

import hashlib
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from . import benchmark
from .common import LabError, digest, utc_now, write_json


def witness_inputs(name, seed=20261101):
    """Return deterministic FP32 source arrays; GPU casting happens later."""
    rng = np.random.default_rng(seed)
    m, k, n = 3, 17, 11
    x = rng.normal(0, 0.25, (m, k)).astype(np.float32)
    w = rng.normal(0, 0.25, (n, k)).astype(np.float32)
    if name == "zero":
        x.fill(0)
    elif name == "identity":
        w = np.eye(k, dtype=np.float32)
    elif name == "alternating":
        x = ((-1.0) ** np.arange(m * k).reshape(m, k) * 0.25).astype(np.float32)
        w = ((-1.0) ** np.arange(n * k).reshape(n, k) * 0.5).astype(np.float32)
    elif name != "random":
        raise LabError("Unknown numerical witness")
    return x, w


def audit_witness_npz(path):
    """Recompute the represented-input product using independent NumPy FP64.

    This verifies the archived small witness only. It cannot attest to GPU
    execution or prove correctness of unarchived large matrices.
    """
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise LabError("Numerical witness archive exceeds the small-file contract")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 1024**2:
            raise LabError("Numerical witness uncompressed data exceeds the small-file contract")
    with np.load(path, allow_pickle=False) as data:
        required = {"input_x", "weights_nk", "output_y", "reference_fp32", "input_dtype", "witness_name"}
        if set(data.files) != required:
            raise LabError("Numerical witness has missing or unexpected arrays")
        x, w, y, reference = [np.asarray(data[key]) for key in ("input_x", "weights_nk", "output_y", "reference_fp32")]
        dtype = str(data["input_dtype"].item())
        name = str(data["witness_name"].item())
    if dtype not in {"float16", "bfloat16"}:
        raise LabError("Unsupported witness dtype")
    if any(a.dtype != np.float32 or a.ndim != 2 for a in (x, w, y, reference)):
        raise LabError("Witness arrays must be finite rank-2 float32 represented values")
    if x.shape[1] != w.shape[1] or y.shape != (x.shape[0], w.shape[0]) or reference.shape != y.shape:
        raise LabError("Witness matrix dimensions are inconsistent")
    expected_w_shape = (17, 17) if name == "identity" else (11, 17)
    if x.shape != (3, 17) or w.shape != expected_w_shape:
        raise LabError("Witness does not have its registered small matrix dimensions")
    if x.size + w.size + y.size + reference.size > 20000:
        raise LabError("Witness exceeds the bounded small-tensor audit contract")
    if not all(np.isfinite(a).all() for a in (x, w, y, reference)):
        raise LabError("Witness contains nonfinite values")
    for array in (x, w, y):
        if dtype == "float16":
            with np.errstate(over="ignore", invalid="ignore"):
                represented = np.array_equal(array, array.astype(np.float16).astype(np.float32))
        else:
            represented = bool(np.all((array.view(np.uint32) & np.uint32(0xFFFF)) == 0))
        if not represented:
            raise LabError("Archived input/output values are not exactly representable in the declared low precision")
    exact = x.astype(np.float64) @ w.astype(np.float64).T
    reference_difference = reference.astype(np.float64) - exact
    output_difference = y.astype(np.float64) - exact
    reference_max = float(np.max(np.abs(exact)))
    reference_limit = 1e-6 + 1e-5 * reference_max
    if float(np.max(np.abs(reference_difference))) > reference_limit:
        raise LabError("Archived FP32 reference disagrees with the FP64 represented-input product")
    relative = float(np.linalg.norm(output_difference) / max(float(np.linalg.norm(exact)), 1e-30))
    maximum = float(np.max(np.abs(output_difference)))
    limits = benchmark._correctness_limits(dtype)
    allowed = limits["absolute_atol"] + limits["absolute_rtol"] * reference_max
    passed = relative <= limits["relative_l2"] and maximum <= allowed
    if name == "zero" and not np.array_equal(y, np.zeros_like(y)):
        passed = False
    if name == "zero" and not np.array_equal(x, np.zeros_like(x)):
        raise LabError("Zero witness must contain an all-zero input")
    if name == "identity" and not np.array_equal(w, np.eye(17, dtype=np.float32)):
        raise LabError("Identity witness must contain the registered identity weights")
    if name == "identity" and not np.array_equal(y, x):
        passed = False
    if name == "alternating":
        expected_x, expected_w = witness_inputs("alternating")
        if not np.array_equal(x, expected_x) or not np.array_equal(w, expected_w):
            raise LabError("Alternating witness must contain the registered sign pattern")
    if name not in {"zero", "identity", "alternating", "random"}:
        raise LabError("Unregistered witness kind")
    if not passed:
        raise LabError("Archived GPU output does not satisfy the independent small-witness check")
    return {
        "passed": True, "witness_name": name, "dtype": dtype,
        "m": x.shape[0], "n": w.shape[0], "k": x.shape[1],
        "fp64_relative_l2_error": relative, "fp64_max_abs_error": maximum,
        "fp32_reference_max_abs_error": float(np.max(np.abs(reference_difference))),
        "fp32_reference_max_abs_limit": reference_limit,
        "output_max_abs_limit": allowed,
        "comparison": "NumPy float64 product of the exact archived represented input values",
        "limitations": "Small archived witness only; no attestation of GPU execution or all main-case outputs.",
    }


def collect_numerical_witnesses(plan, environment, output_dir, *, device="cuda:0", deadline_monotonic=None):
    if environment.get("measurement_kind") not in {"simulated", "cuda_measured"}:
        raise LabError("Numerical witnesses require an explicit source environment")
    if environment.get("plan_hash") != digest(plan):
        raise LabError("Numerical witness environment differs from the registered plan")
    deadline = benchmark._session_deadline(plan, deadline_monotonic)
    benchmark._check_deadline(deadline)
    out = Path(output_dir)
    if out.exists():
        raise LabError("Numerical witness output already exists")
    out.mkdir(parents=True)
    summary = {
        "schema_version": 1, "protocol_version": "1.0.0",
        "status": "running", "plan_hash": digest(plan),
        "source_environment_id": environment["environment_id"],
        "source_measurement_kind": environment["measurement_kind"],
        "measurement_kind": "numerical_witness", "excluded_from_model": True,
        "scope": "Four small deterministic witnesses; not the main benchmark dataset.",
        "records": [], "started_at_utc": utc_now(),
        "started_monotonic_seconds": time.monotonic(),
        "session_id": environment["session_id"], "input_seed": plan["seed"],
    }
    if environment["measurement_kind"] == "simulated":
        summary.update(status="skipped", reason="simulation_no_GPU_witness", arrays_archived=False,
                       completed_at_utc=utc_now(), completed_monotonic_seconds=time.monotonic())
        write_json(out / "witnesses.json", summary)
        return summary
    write_json(out / "witnesses.json", summary)
    try:
        torch = benchmark._load_torch()
        selected = benchmark._require_cuda(torch, plan["dtype"], device)
        with torch.cuda.device(selected), benchmark._precision_policy(torch) as policy, \
                torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            current = benchmark._collect_environment(torch, plan, selected, policy)
            if current["environment_id"] != environment["environment_id"] or current["identity"] != environment["identity"]:
                raise LabError("Numerical witness environment changed")
            for name in ("zero", "identity", "alternating", "random"):
                benchmark._check_deadline(deadline)
                started, started_utc = time.monotonic(), utc_now()
                source_x, source_w = witness_inputs(name, plan["seed"])
                target_dtype = getattr(torch, plan["dtype"])
                x = torch.from_numpy(source_x).to(device=selected, dtype=target_dtype)
                w = torch.from_numpy(source_w).to(device=selected, dtype=target_dtype)
                reference = torch.mm(x.float(), w.float().t())
                output = torch.mm(x, w.t())
                check = benchmark._correctness(torch, output, reference, plan["dtype"])
                benchmark._assert_correctness(check, "witness_" + name)
                arrays = {
                    "input_x": x.float().cpu().numpy(),
                    "weights_nk": w.float().cpu().numpy(),
                    "output_y": output.float().cpu().numpy(),
                    "reference_fp32": reference.float().cpu().numpy(),
                    "input_dtype": np.array(plan["dtype"]),
                    "witness_name": np.array(name),
                }
                path = out / (name + ".npz")
                if path.exists():
                    raise LabError("Refusing to overwrite a numerical witness")
                with path.open("xb") as archive:
                    np.savez_compressed(archive, **arrays)
                    archive.flush()
                    os.fsync(archive.fileno())
                audit = audit_witness_npz(path)
                summary["records"].append({
                    "name": name, "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "gpu_check": check, "independent_cpu_recomputation": audit,
                    "captured_at_utc": utc_now(),
                    "visit_started_at_utc": started_utc,
                    "visit_finished_at_utc": utc_now(),
                    "visit_started_monotonic_seconds": started,
                    "visit_finished_monotonic_seconds": time.monotonic(),
                })
                write_json(out / "witnesses.json", summary)
                del x, w, output, reference
        summary.update(status="completed", arrays_archived=True, completed_at_utc=utc_now(),
                       completed_monotonic_seconds=time.monotonic())
        write_json(out / "witnesses.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc), failed_at_utc=utc_now(),
                       failed_monotonic_seconds=time.monotonic())
        write_json(out / "witnesses.json", summary)
        raise
