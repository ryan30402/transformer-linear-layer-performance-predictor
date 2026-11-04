"""Deterministic synthetic data for software validation; never a GPU benchmark."""

from __future__ import annotations

import math
import random
import statistics

from .common import SCHEMA_VERSION, digest, flops, ideal_bytes, validate_plan


def simulated_environment(plan: dict) -> dict:
    validate_plan(plan)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": "simulated",
        "device_name": "SIMULATED_WORLD — no GPU was used",
        "simulator_version": 1,
        "seed": plan["seed"],
        "dtype": plan["dtype"],
        "cache_policy": plan["cache_policy"],
        "plan_hash": digest(plan),
    }
    return {**identity, "environment_id": "simulated:" + digest(identity),
            "warning": "Synthetic timings verify software plumbing only. They are not NVIDIA hardware results."}


def simulate_cases(plan: dict, cases: list[dict], *, environment: dict | None = None) -> list[dict]:
    env = simulated_environment(plan)
    if environment is not None and environment["environment_id"] != env["environment_id"]:
        raise ValueError("Simulator environment mismatch")
    rows = []
    for case in cases:
        # Arbitrary illustrative parameters, not specifications of any actual GPU.
        arithmetic = flops(case) / (160.0 * 1e9)
        traffic = ideal_bytes(case, plan["dtype"]) / (650.0 * 1e6)
        shape_factor = 1.0 + 0.12 * math.exp(-case["m"] / 48.0)
        shape_factor += 0.025 * math.sin(math.log2(max(1, case["m"])))
        if case["n"] % 16 or case["k"] % 16:
            shape_factor += 0.35
        truth_ms = 0.012 + max(arithmetic, traffic) * shape_factor
        local_seed = int(digest({"seed": plan["seed"], "m": case["m"], "n": case["n"], "k": case["k"]})[:16], 16)
        rng = random.Random(local_seed)
        samples = [truth_ms * (1 + rng.uniform(-0.015, 0.015)) for _ in range(plan["timing"]["repeats"])]
        rows.append({
            "schema_version": SCHEMA_VERSION, "case_id": case["id"],
            **{key: case[key] for key in ("m", "n", "k", "split")},
            "dtype": plan["dtype"], "cache_policy": plan["cache_policy"],
            "plan_hash": digest(plan), "environment_id": env["environment_id"],
            "measurement_kind": "simulated", "status": "ok",
            "median_ms": statistics.median(samples), "samples_ms": samples,
            "flops": flops(case), "ideal_bytes": ideal_bytes(case, plan["dtype"]),
            "correctness": {"passed": True, "finite": True, "relative_l2_error": 0.0,
                            "max_abs_error": 0.0, "reference": "simulated_contract_only_no_matmul_executed",
                            "check_kind": "synthetic_contract"},
        })
    return rows
