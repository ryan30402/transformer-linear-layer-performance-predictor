"""Offline acquisition on a fixed, fully measured candidate pool.

The selector sees only shape metadata and revealed labels.  All checkpoint
models are fitted and sealed before any held-out evaluation call.  Reusing the
same measured pool removes live branch-order drift from this comparison, but
does not create independent hardware trials or eliminate noise in that pool.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict

import numpy as np

from .common import LabError, digest, utc_now
from . import models as model_api


class _PoolOracle:
    """A scoped label-revelation boundary, not a security sandbox."""

    def __init__(self, rows: list[dict]):
        self.__rows = {row["case_id"]: copy.deepcopy(row) for row in rows}
        self.revealed: list[str] = []

    def shapes(self) -> list[dict]:
        return [{"case_id": row["case_id"], "m": row["m"], "n": row["n"], "k": row["k"]}
                for row in sorted(self.__rows.values(), key=lambda row: row["case_id"])]

    def reveal(self, ids: list[str]) -> list[dict]:
        if len(set(ids)) != len(ids) or any(i not in self.__rows or i in self.revealed for i in ids):
            raise LabError("A candidate label must be registered and revealed at most once per policy run")
        self.revealed.extend(ids)
        return [copy.deepcopy(self.__rows[i]) for i in ids]


def _choose(policy: str, available_shapes: list[dict], dev: list[dict], accepted_model: dict,
            dtype: str, rng: random.Random, tie_priorities: dict[str, float], count: int) -> list[str]:
    if count > len(available_shapes):
        raise LabError("Acquisition budget exceeds available labels")
    if policy == "random":
        return rng.sample([row["case_id"] for row in available_shapes], count)
    if policy != "heuristic":
        raise LabError("Unknown offline acquisition policy")
    predicted = model_api.predict_model(accepted_model, dev, dtype)
    worst = max(zip(dev, predicted), key=lambda pair: (
        abs(pair[1] / pair[0]["median_ms"] - 1), pair[0]["case_id"]))[0]

    def key(case):
        distance = sum(abs(math.log2(case[axis] / worst[axis])) for axis in ("m", "n", "k"))
        return distance, tie_priorities[case["case_id"]], case["case_id"]

    return [row["case_id"] for row in sorted(available_shapes, key=key)[:count]]


def fit_acquisition(pool_rows: list[dict], train_rows: list[dict], development_rows: list[dict],
                    plan: dict, *, environment: dict | None = None) -> dict:
    """Run all declared seeds/budgets using only train, development and revealed pool labels."""
    meta = model_api._train_and_dev(train_rows, development_rows, plan)
    pool_meta = model_api._rows(pool_rows, plan, {"candidate"})
    if meta != pool_meta:
        raise LabError("Acquisition pool provenance differs from train/development")
    if {row["case_id"] for row in pool_rows} != set(plan["study"]["candidate_pool_ids"]):
        raise LabError("Offline acquisition requires the complete, declared candidate pool")
    if any(row["split"] != "calibration" for row in train_rows):
        raise LabError("Every offline policy must start from the same calibration-only baseline")
    env = model_api._environment(environment, meta, plan)
    train = sorted(copy.deepcopy(train_rows), key=lambda row: row["case_id"])
    dev = sorted(copy.deepcopy(development_rows), key=lambda row: row["case_id"])
    pool = sorted(copy.deepcopy(pool_rows), key=lambda row: row["case_id"])
    contract = plan["study"]["acquisition"]
    budgets = contract["budgets"]
    seeds = contract["seeds"]
    batch_size = contract["batch_size"]
    initial = model_api._fit_two_parameter(train, meta)
    initial_score = model_api._score(initial, dev, plan["dtype"])
    runs = []
    fit_cache: dict[tuple[str, ...], dict] = {(): initial}
    # Memoization only avoids recomputing the same fixed-data fit.  Each policy
    # still receives only its own sequence of labels and has a full reveal log.
    for policy in ("heuristic", "random"):
        for seed in seeds:
            oracle = _PoolOracle(pool)
            shapes = oracle.shapes()
            rng = random.Random(seed)
            tie_rng = random.Random(seed ^ 0x5A17)
            priorities = {row["case_id"]: tie_rng.random() for row in shapes}
            acquired_rows: list[dict] = []
            best = copy.deepcopy(initial)
            best_score = copy.deepcopy(initial_score)
            trials, checkpoints = [], []
            for acquired_count in range(batch_size, max(budgets) + 1, batch_size):
                unused = [row for row in shapes if row["case_id"] not in oracle.revealed]
                selected = _choose(policy, unused, dev, best, plan["dtype"], rng, priorities, batch_size)
                acquired_rows.extend(oracle.reveal(selected))
                fit_key = tuple(sorted(oracle.revealed))
                if fit_key not in fit_cache:
                    fit_rows = sorted(train + acquired_rows, key=lambda row: row["case_id"])
                    fit_cache[fit_key] = model_api._fit_two_parameter(fit_rows, meta)
                candidate = copy.deepcopy(fit_cache[fit_key])
                score = model_api._score(candidate, dev, plan["dtype"])
                accepted = score["median_ape_pct"] < best_score["median_ape_pct"]
                trials.append({
                    "round": acquired_count // batch_size, "acquired_count": acquired_count,
                    "selected_case_ids": selected, "revealed_case_ids": list(oracle.revealed),
                    "candidate_model": candidate,
                    "development_metrics": score,
                    "previous_best_development_median_ape_pct": best_score["median_ape_pct"],
                    "accepted": accepted,
                    "decision_rule": "Strictly lower development median APE; a tie retains the previous model.",
                })
                if accepted:
                    best, best_score = candidate, score
                if acquired_count in budgets:
                    checkpoints.append({
                        "budget": acquired_count, "revealed_case_ids": list(oracle.revealed),
                        "accepted_model": copy.deepcopy(best),
                        "development_metrics": copy.deepcopy(best_score),
                        "retained_train_case_ids": list(best["train_case_ids"]),
                        "training_label_count": len(best["train_case_ids"]),
                    })
            runs.append({"policy": policy, "seed": seed,
                         "label_revelation_order": list(oracle.revealed),
                         "trials": trials, "checkpoints": checkpoints})
    sequences = {policy: len({tuple(run["label_revelation_order"]) for run in runs if run["policy"] == policy})
                 for policy in ("heuristic", "random")}
    return model_api._sealed({
        "schema_version": 1, "artifact_type": "offline_acquisition", "protocol_version": model_api.VERSION,
        "created_at_utc": utc_now(), **meta, "protocol_signature": model_api._signature(plan),
        "source_environment": env, "initial_model": initial,
        "initial_development_metrics": initial_score,
        "train_case_ids": [row["case_id"] for row in train],
        "development_case_ids": [row["case_id"] for row in dev],
        "candidate_pool_case_ids": [row["case_id"] for row in pool],
        "training_data_hash": digest(model_api._training_evidence(train)),
        "development_data_hash": digest(model_api._training_evidence(dev)),
        "candidate_pool_data_hash": digest(model_api._training_evidence(pool)),
        "budgets": list(budgets), "seeds": list(seeds), "batch_size": batch_size,
        "model_family": "roofline_2p", "runs": runs,
        "distinct_revelation_sequences": sequences,
        "protocol": {
            "evaluation_kind": "offline_label_revelation_on_fixed_measured_pool",
            "labels_visible_to_selector": "development labels and previously revealed pool labels; remaining pool exposes shapes only",
            "acceptance": "development median APE only; every acquired label retained for later trial fits",
            "heuristic": "nearest unused candidates to largest development APE case in L1 log2 shape space; seeded tie order",
            "random": "uniform without replacement using the declared seed",
            "budget_unit": "newly revealed distinct candidate labels; not equal GPU time",
            "live_gpu_closed_loop": False,
            "test_labels_used": False,
            "memoized_fixed_data_fits": len(fit_cache),
            "interpretation": "Seeds vary selection on one shared measured pool. They are not independent GPU sessions; identical heuristic sequences do not add replication evidence.",
        },
    }, "acquisition_hash")


def evaluate_acquisition(acquisition: dict, heldout_rows: list[dict], plan: dict,
                         *, target_environment: dict | None = None) -> dict:
    model_api._check_seal(acquisition, "acquisition_hash")
    if acquisition.get("artifact_type") != "offline_acquisition" or acquisition.get("protocol_version") != model_api.VERSION:
        raise LabError("Unsupported acquisition artifact")
    model_api._rows(heldout_rows, plan, {"test", "challenge"})
    context = model_api._evaluation_context(acquisition, heldout_rows, plan, target_environment, None)
    contract = plan["study"]["acquisition"]
    if any(acquisition.get(key) != contract[key] for key in ("budgets", "seeds", "batch_size")):
        raise LabError("Acquisition evaluation must retain the registered seeds and budgets")
    expected_runs = {(policy, seed) for policy in ("heuristic", "random") for seed in contract["seeds"]}
    observed_runs = [(run["policy"], run["seed"]) for run in acquisition["runs"]]
    if len(observed_runs) != len(expected_runs) or set(observed_runs) != expected_runs:
        raise LabError("Missing or duplicated acquisition policy/seed run")
    allowed_pool = set(plan["study"]["candidate_pool_ids"])
    allowed_calibration = {case["id"] for case in plan["cases"] if case["split"] == "calibration"}
    training_ids = set(acquisition["train_case_ids"])
    if not training_ids <= allowed_calibration or set(acquisition["candidate_pool_case_ids"]) != allowed_pool:
        raise LabError("Acquisition baseline or pool contains unregistered/protected labels")
    if set(acquisition["initial_model"]["train_case_ids"]) != training_ids:
        raise LabError("Acquisition baseline training identity differs")
    splits = sorted({row["split"] for row in heldout_rows})
    baseline = {split: model_api.evaluate_predictions(
        [row for row in heldout_rows if row["split"] == split],
        model_api.predict_model(acquisition["initial_model"], [row for row in heldout_rows if row["split"] == split], plan["dtype"]))
        for split in splits}
    runs, grouped = [], defaultdict(list)
    for run in acquisition["runs"]:
        order = run["label_revelation_order"]
        if len(order) != max(contract["budgets"]) or len(set(order)) != len(order) or not set(order) <= allowed_pool:
            raise LabError("Invalid acquisition label-revelation order")
        if [checkpoint["budget"] for checkpoint in run["checkpoints"]] != contract["budgets"]:
            raise LabError("Acquisition checkpoints differ from registered budgets")
        for checkpoint in run["checkpoints"]:
            revealed = order[:checkpoint["budget"]]
            fitted_ids = set(checkpoint["accepted_model"]["train_case_ids"])
            if (checkpoint["revealed_case_ids"] != revealed or not training_ids <= fitted_ids
                    or not fitted_ids <= training_ids | set(revealed)
                    or set(checkpoint["retained_train_case_ids"]) != fitted_ids
                    or checkpoint["training_label_count"] != len(fitted_ids)):
                raise LabError("Checkpoint uses labels outside its declared revelation budget")
            values = {}
            for split in splits:
                rows = [row for row in heldout_rows if row["split"] == split]
                score = model_api.evaluate_predictions(rows, model_api.predict_model(checkpoint["accepted_model"], rows, plan["dtype"]))
                values[split] = score
                grouped[(run["policy"], checkpoint["budget"], split)].append({"seed": run["seed"], **{
                    metric: score[metric] for metric in ("median_ape_pct", "p90_ape_pct", "rmse_us", "pairwise_ordering_accuracy")}})
            runs.append({"policy": run["policy"], "seed": run["seed"], "budget": checkpoint["budget"],
                         "model_hash": checkpoint["accepted_model"]["model_hash"], "metrics": values,
                         "revealed_case_ids": checkpoint["revealed_case_ids"],
                         "retained_train_case_ids": checkpoint["retained_train_case_ids"]})
    curves = []
    for (policy, budget, split), scores in sorted(grouped.items()):
        medians = np.array([score["median_ape_pct"] for score in scores])
        curves.append({"policy": policy, "budget": budget, "split": split, "seed_count": len(scores),
                       "mean_median_ape_pct": float(np.mean(medians)),
                       "median_median_ape_pct": float(np.median(medians)),
                       "min_median_ape_pct": float(np.min(medians)), "max_median_ape_pct": float(np.max(medians)),
                       "sample_sd_median_ape_pct": float(np.std(medians, ddof=1)) if len(medians) > 1 else None,
                       "per_seed": scores,
                       "interpretation": "Selection-seed variation conditional on one shared data pool; not cross-session uncertainty."})
    return {"schema_version": 1, "protocol_version": model_api.VERSION,
            "measurement_kind": acquisition["measurement_kind"], "acquisition_hash": acquisition["acquisition_hash"],
            "plan_hash": digest(plan), "transfer": context, "baseline": baseline, "runs": runs, "curves": curves,
            "evaluated_case_ids": [row["case_id"] for row in heldout_rows],
            "evaluation_data_hash": digest(model_api._training_evidence(heldout_rows)),
            "distinct_revelation_sequences": acquisition["distinct_revelation_sequences"],
            "heldout_used_for_selection": False,
            "selection_after_evaluation": "prohibited; all reported checkpoints were frozen before this call"}
