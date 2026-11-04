"""CPU contract tests: synthetic timings are never NVIDIA performance evidence."""

import copy
import json
import math
import random
import unittest
from unittest.mock import patch

from linear_perf import acquisition as acquisition
from linear_perf import models as models
from linear_perf.collection import _aggregate, _block_row
from linear_perf.common import GRAPH_TIMER_SCOPE, LabError, digest
from linear_perf.protocol import build_plan
from linear_perf.simulation import simulate_cases, simulated_environment


def aggregate_fixture(plan, cases=None):
    """Retain the real aggregate validation chain around explicitly synthetic data."""
    env = simulated_environment(plan)
    output = []
    for case in cases or plan["cases"]:
        blocks = [_block_row(simulate_cases(plan, [case])[0], plan=plan, case=case,
                             env=env, block_index=i, position=1, order_seed=i)
                  for i in (1, 2, 3)]
        output.append(_aggregate(blocks, expected_blocks=3))
    return output


def scaled_fixture(rows, factor):
    output = []
    for row in rows:
        blocks = copy.deepcopy(row["measurement_blocks"])
        for block in blocks:
            block["samples_ms"] = [v * factor for v in block["samples_ms"]]
            block["median_ms"] *= factor
        output.append(_aggregate(blocks, expected_blocks=3))
    return output


def rebind_fixture(rows, env):
    """In-memory provenance fault-injection only; no purported CUDA file is saved."""
    output = []
    for row in rows:
        blocks = copy.deepcopy(row["measurement_blocks"])
        for block in blocks:
            block["environment_id"] = env["environment_id"]
            block["measurement_kind"] = env["measurement_kind"] + "_block"
            block["source_measurement_kind"] = env["measurement_kind"]
            if env["measurement_kind"] == "cuda_measured":
                block["execution_mode"] = "cuda_graph"
                block["timer_scope"] = GRAPH_TIMER_SCOPE
        output.append(_aggregate(blocks, expected_blocks=3))
    return output


def mocked_cuda_environment(plan, *, device_name="MOCK_GPU_CONTRACT_NO_CUDA_EXECUTED",
                            uuid="MOCK_UUID_A", pytorch="MOCK_PYTORCH"):
    identity = {
        "runner_version": "MOCK_RUNNER", "python_version": "MOCK_PYTHON",
        "pytorch_version": pytorch, "cuda_runtime_version": "MOCK_CUDA",
        "driver_versions": ["MOCK_DRIVER"],
        "device": {"name": device_name, "uuid": uuid, "compute_capability": [8, 0],
                   "total_memory_bytes": 1024, "multiprocessor_count": 1},
        "dtype": plan["dtype"], "cache_policy": plan["cache_policy"],
        "timing": copy.deepcopy(plan["timing"]), "collection": copy.deepcopy(plan["collection"]),
        "math_policy": {"fp32_reference_precision": "ieee"}, "execution": {"mode": "cuda_graph"},
    }
    return {"measurement_kind": "cuda_measured", "identity": identity,
            "environment_id": digest(identity), "plan_hash": digest(plan),
            "test_only": "In-memory mocked provenance; no GPU executed"}


def reseal(value, field):
    value = copy.deepcopy(value)
    value.pop(field, None)
    return models._sealed(value, field)


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan()
        cls.env = simulated_environment(cls.plan)
        cls.rows = aggregate_fixture(cls.plan)
        cls.train = [r for r in cls.rows if r["split"] == "calibration"]
        cls.dev = [r for r in cls.rows if r["split"] == "development"]
        cls.test = [r for r in cls.rows if r["split"] == "test"]
        cls.challenge = [r for r in cls.rows if r["split"] == "challenge"]
        cls.pool = [r for r in cls.rows if r["case_id"] in cls.plan["study"]["candidate_pool_ids"]]
        cls.suite = models.fit_suite(cls.train, cls.dev, cls.plan, environment=cls.env)
        cls.acq = models.fit_acquisition(cls.pool, cls.train, cls.dev, cls.plan, environment=cls.env)

    def test_five_models_share_data_and_json_round_trip(self):
        suite = json.loads(json.dumps(self.suite, allow_nan=False))
        self.assertEqual(set(suite["models"]), set(models.MODEL_ORDER))
        for model in suite["models"].values():
            self.assertEqual(model["train_case_ids"], suite["train_case_ids"])
            self.assertEqual(model["training_data_hash"], suite["training_data_hash"])
            predicted = models.predict_model(model, self.test, self.plan["dtype"])
            self.assertTrue(all(math.isfinite(v) and v > 0 for v in predicted))
        evaluated = models.evaluate_suite(suite, self.test, self.plan, target_environment=self.env)
        self.assertEqual(evaluated["suite_hash"], self.suite["suite_hash"])
        self.assertEqual(evaluated["evaluated_case_ids"], [r["case_id"] for r in self.test])

    def test_training_cannot_receive_any_protected_split(self):
        for row in (self.dev[0], self.test[0], self.challenge[0]):
            with self.subTest(split=row["split"]), self.assertRaises(LabError):
                models.fit_suite(self.train + [row], self.dev, self.plan)

    def test_diagnostics_cannot_enter_fit_despite_candidate_transport_split(self):
        diagnostic = next(r for r in self.rows if r["case_id"] in self.plan["study"]["diagnostic_case_ids"])
        self.assertEqual(diagnostic["split"], "candidate")
        with self.assertRaisesRegex(LabError, "Diagnostic"):
            models.fit_suite(self.train + [diagnostic], self.dev, self.plan)

    def test_selection_requires_all_predeclared_development_cases(self):
        with self.assertRaisesRegex(LabError, "complete predeclared development"):
            models.fit_suite(self.train, self.dev[:-1], self.plan)

    def test_hyperparameter_selection_uses_development_not_protected_labels(self):
        self.assertEqual(len(self.suite["hyperparameter_search"]["shape_log_ridge"]), 4)
        self.assertEqual(len(self.suite["hyperparameter_search"]["roofline_residual_krr"]), 12)
        self.assertEqual(set(self.suite["development_case_ids"]), {r["case_id"] for r in self.dev})
        expected = min(models.MODEL_ORDER, key=lambda name: models._rank_key(
            self.suite["development_metrics"][name], models.MODEL_ORDER.index(name)))
        self.assertEqual(self.suite["selected_model_id"], expected)
        for family, trials in self.suite["hyperparameter_search"].items():
            self.assertEqual(sum(t["selected"] for t in trials), 1)
            selected = next(t for t in trials if t["selected"])
            self.assertEqual(selected["model_hash"], self.suite["models"][family]["model_hash"])

    def test_raw_repeat_rows_and_corrupt_raw_chains_are_rejected(self):
        raw = simulate_cases(self.plan, [c for c in self.plan["cases"] if c["split"] == "calibration"])
        with self.assertRaisesRegex(LabError, "three-pass aggregate"):
            models.fit_suite(raw, self.dev, self.plan)
        broken = copy.deepcopy(self.train)
        broken[0]["measurement_blocks"][0]["samples_ms"][0] *= 2
        with self.assertRaisesRegex(LabError, "hash mismatch"):
            models.fit_suite(broken, self.dev, self.plan)

    def test_forged_shape_precision_and_mixed_environment_rejected(self):
        for key, value in (("m", 999), ("dtype", "bfloat16"), ("plan_hash", "bad"),
                           ("environment_id", "other-env")):
            rows = copy.deepcopy(self.train)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(LabError):
                models.fit_suite(rows, self.dev, self.plan)

    def test_model_and_suite_digests_detect_changes(self):
        model = copy.deepcopy(self.suite["models"]["roofline_2p"])
        model["params"]["compute_tflops"] *= 2
        with self.assertRaisesRegex(LabError, "digest mismatch"):
            models.predict_model(model, self.test, self.plan["dtype"])
        suite = copy.deepcopy(self.suite)
        suite["selected_model_id"] = "roofline_3p"
        with self.assertRaisesRegex(LabError, "digest mismatch"):
            models.evaluate_suite(suite, self.test, self.plan)

    def test_model_metadata_cannot_be_swapped_between_frozen_suites(self):
        suite = copy.deepcopy(self.suite)
        model = suite["models"]["flop_only"]
        model["environment_id"] = "foreign"
        suite["models"]["flop_only"] = reseal(model, "model_hash")
        suite = reseal(suite, "suite_hash")
        with self.assertRaisesRegex(LabError, "metadata differs"):
            models.evaluate_suite(suite, self.test, self.plan)

    def test_evaluation_never_refits_or_mutates_a_frozen_model(self):
        suite = copy.deepcopy(self.suite)
        before = copy.deepcopy(suite)
        shifted = scaled_fixture(self.test, 4.0)
        with patch.object(models, "fit_suite", side_effect=AssertionError("protected labels reached fit")), \
             patch.object(models, "_fit_two_parameter", side_effect=AssertionError("post-freeze refit")), \
             patch.object(models.predictor_api, "fit_model", side_effect=AssertionError("predictor_api refit")):
            result = models.evaluate_suite(suite, shifted, self.plan)
        self.assertEqual(suite, before)
        for name, model in suite["models"].items():
            self.assertEqual([p["predicted_ms"] for p in result["models"][name]["predictions"]],
                             models.predict_model(model, self.test, self.plan["dtype"]))

    def test_metrics_units_bias_zero_errors_and_tie_aware_ranking(self):
        rows = copy.deepcopy(self.test[:3])
        for row, target in zip(rows, (1., 2., 4.)):
            row["median_ms"] = target
            row["samples_ms"] = [target] * 3
        perfect = models.evaluate_predictions(rows, [1., 2., 4.])
        self.assertEqual(perfect["rmse_us"], 0)
        self.assertEqual(perfect["log_rmse"], 0)
        self.assertEqual(perfect["geometric_mean_multiplicative_error"], 1)
        self.assertEqual(perfect["pairwise_ordering_accuracy"], 1.)
        tied = models.evaluate_predictions(rows, [2., 2., 2.])
        self.assertEqual(tied["median_ape_pct"], 50.)
        self.assertAlmostEqual(tied["signed_bias_us"], -1000 / 3)
        self.assertAlmostEqual(tied["rmse_us"], math.sqrt(5 / 3) * 1000)
        self.assertEqual(tied["pairwise_ordering_accuracy"], .5)
        self.assertAlmostEqual(tied["log_rmse"], math.log(2) * math.sqrt(2 / 3))
        self.assertAlmostEqual(tied["geometric_mean_multiplicative_error"], 2 ** (2 / 3))
        self.assertIn("not confidence", tied["uncertainty_note"])
        with self.assertRaises(LabError):
            models.evaluate_predictions(rows + [rows[0]], [1., 2., 4., 1.])

    def test_group_metrics_and_ood_distinguish_challenge_families(self):
        result = models.evaluate_suite(self.suite, self.challenge, self.plan)
        for name, score in result["models"].items():
            self.assertTrue(all("unseen_nk_family" in p["ood"]["flags"] for p in score["predictions"]))
            self.assertEqual(sum(g["n"] for g in result["groups"][name]["family"].values()), 12)
        bootstrap = result["selected_vs_roofline_3p"]
        self.assertEqual(bootstrap["resampling_unit"], "registered_NK_shape_family")
        self.assertEqual(bootstrap["n_shape_families"], 4)
        self.assertIn("not", bootstrap["limitations"].lower())

    def test_unrelated_environment_requires_explicit_transfer_and_both_metadata(self):
        target = copy.deepcopy(self.env)
        target["environment_id"] = "simulated:other-process"
        rows = rebind_fixture(self.test, target)
        with self.assertRaisesRegex(LabError, "explicit transfer_mode"):
            models.evaluate_suite(self.suite, rows, self.plan, target_environment=target)
        with self.assertRaisesRegex(LabError, "source and target"):
            models.evaluate_suite(self.suite, rows, self.plan, transfer_mode="session")
        result = models.evaluate_suite(self.suite, rows, self.plan, target_environment=target, transfer_mode="session")
        self.assertFalse(result["transfer"]["refitted_on_target"])
        self.assertEqual(result["transfer"]["mode"], "cross_session_frozen_parameters")

    def test_cuda_transfer_metadata_guard_with_mocked_identities_only(self):
        # This test exercises contracts. It never runs or saves CUDA evidence.
        source = mocked_cuda_environment(self.plan)
        suite = {key: self.suite[key] for key in (*models._META, "protocol_signature")}
        suite.update(measurement_kind="cuda_measured", environment_id=source["environment_id"],
                     source_environment=source)
        same_gpu = mocked_cuda_environment(self.plan, uuid="MOCK_UUID_B")
        rows = rebind_fixture(self.test, same_gpu)
        context = models._evaluation_context(suite, rows, self.plan, same_gpu, "session")
        self.assertEqual(context["mode"], "cross_session_frozen_parameters")
        other_gpu = mocked_cuda_environment(self.plan, device_name="OTHER_MOCK_GPU")
        rows = rebind_fixture(self.test, other_gpu)
        with self.assertRaisesRegex(LabError, "same GPU configuration"):
            models._evaluation_context(suite, rows, self.plan, other_gpu, "session")
        models._evaluation_context(suite, rows, self.plan, other_gpu, "gpu")
        other_stack = mocked_cuda_environment(self.plan, device_name="OTHER_MOCK_GPU", pytorch="CHANGED_MOCK_STACK")
        with self.assertRaisesRegex(LabError, "different software"):
            models._evaluation_context(suite, rebind_fixture(self.test, other_stack), self.plan, other_stack, "gpu")
        corrupted = copy.deepcopy(other_gpu)
        corrupted["identity"]["math_policy"]["fp32_reference_precision"] = "tf32"
        with self.assertRaisesRegex(LabError, "identity hash"):
            models._environment(corrupted, {**suite, "environment_id": other_gpu["environment_id"]}, self.plan)

    def test_acquisition_all_seeds_budgets_and_rejected_labels_remain_visible(self):
        self.assertEqual(len(self.acq["runs"]), 20)
        base = set(self.acq["train_case_ids"])
        for run in self.acq["runs"]:
            self.assertEqual(len(run["trials"]), 6)
            self.assertEqual([c["budget"] for c in run["checkpoints"]], [2, 4, 8, 12])
            self.assertEqual(len(set(run["label_revelation_order"])), 12)
            for trial in run["trials"]:
                self.assertEqual(set(trial["candidate_model"]["train_case_ids"]),
                                 base | set(trial["revealed_case_ids"]))
                self.assertEqual(trial["accepted"], trial["development_metrics"]["median_ape_pct"] <
                                 trial["previous_best_development_median_ape_pct"])
            for checkpoint in run["checkpoints"]:
                self.assertTrue(set(checkpoint["retained_train_case_ids"]) <=
                                base | set(run["label_revelation_order"][:checkpoint["budget"]]))
        self.assertFalse(self.acq["protocol"]["live_gpu_closed_loop"])
        result = models.evaluate_acquisition(json.loads(json.dumps(self.acq)), self.test + self.challenge,
                                            self.plan, target_environment=self.env)
        self.assertEqual(len(result["runs"]), 80)
        self.assertEqual(len(result["curves"]), 16)
        self.assertTrue(all(curve["seed_count"] == 10 for curve in result["curves"]))

    def test_hidden_pool_targets_cannot_affect_the_first_query(self):
        original = acquisition._PoolOracle(self.pool)
        altered = acquisition._PoolOracle(scaled_fixture(self.pool, 1000))
        self.assertEqual(original.shapes(), altered.shapes())
        self.assertTrue(all(set(shape) == {"case_id", "m", "n", "k"} for shape in original.shapes()))
        priorities = {r["case_id"]: i for i, r in enumerate(original.shapes())}
        for policy in ("heuristic", "random"):
            first = acquisition._choose(policy, original.shapes(), self.dev, self.acq["initial_model"],
                                        self.plan["dtype"], random.Random(10), priorities, 2)
            second = acquisition._choose(policy, altered.shapes(), self.dev, self.acq["initial_model"],
                                         self.plan["dtype"], random.Random(10), priorities, 2)
            self.assertEqual(first, second)
        ids = [original.shapes()[0]["case_id"]]
        original.reveal(ids)
        with self.assertRaisesRegex(LabError, "at most once"):
            original.reveal(ids)

    def test_acquisition_evaluation_rejects_missing_runs_and_future_budget_labels(self):
        acq = copy.deepcopy(self.acq)
        acq["runs"].pop()
        with self.assertRaisesRegex(LabError, "Missing or duplicated"):
            models.evaluate_acquisition(reseal(acq, "acquisition_hash"), self.test, self.plan)
        acq = copy.deepcopy(self.acq)
        run = acq["runs"][0]
        model = run["checkpoints"][0]["accepted_model"]
        model["train_case_ids"].append(run["label_revelation_order"][-1])
        run["checkpoints"][0]["accepted_model"] = reseal(model, "model_hash")
        with self.assertRaisesRegex(LabError, "revelation budget"):
            models.evaluate_acquisition(reseal(acq, "acquisition_hash"), self.test, self.plan)

    def test_acquisition_protected_or_incomplete_pool_rejected(self):
        with self.assertRaisesRegex(LabError, "complete, declared candidate pool"):
            models.fit_acquisition(self.pool[:-1], self.train, self.dev, self.plan)
        with self.assertRaises(LabError):
            models.fit_acquisition(self.pool + [self.test[0]], self.train, self.dev, self.plan)
        with self.assertRaises(LabError):
            models.evaluate_acquisition(self.acq, self.dev, self.plan)

    def test_predictor_does_not_read_observed_latency(self):
        a = copy.deepcopy(self.test)
        b = copy.deepcopy(self.test)
        for row in b:
            row["median_ms"] = None
            row["samples_ms"] = "protected label sentinel"
        for model in self.suite["models"].values():
            self.assertEqual(models.predict_model(model, a, self.plan["dtype"]),
                             models.predict_model(model, b, self.plan["dtype"]))


if __name__ == "__main__":
    unittest.main()
