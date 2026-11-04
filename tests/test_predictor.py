"""Scientific-contract tests, runnable on a CPU with python -m unittest."""

from __future__ import annotations

import copy
import json
import math
import unittest

import numpy as np

from linear_perf.common import LabError, digest, flops, ideal_bytes
from linear_perf.predictor import evaluate_model, explain_prediction, fit_model, predict_rows


TRUE_PARAMS = {"overhead_ms": 0.004, "compute_tflops": 80.0, "bandwidth_gbps": 700.0}
META = {
    "measurement_kind": "simulated",
    "environment_id": "synthetic-unit-test-not-a-real-gpu",
    "dtype": "bfloat16",
    "cache_policy": "warm_reuse",
    "plan_hash": digest({"fixture": "held-out-shape-experiment-v1"}),
}
# Each tuple is (M, N, K).  Calibration crosses both terms of the roofline.
TRAIN = [
    (1, 1024, 1024), (1, 4096, 4096), (16, 8192, 2048), (64, 4096, 1024),
    (256, 2048, 2048), (512, 11008, 4096), (1024, 8192, 2048), (2048, 4096, 4096),
]
HELD_OUT = [
    (2, 2048, 2048), (12, 4096, 3072), (48, 6144, 2048), (192, 4096, 1536),
    (640, 3072, 3072), (1152, 8192, 4096), (1792, 11008, 2048), (2560, 6144, 4096),
]


def records(shapes, split="calibration", prefix="c", *, noise=0.0):
    rng = np.random.default_rng(24601)
    rows = []
    for index, (m, n, k) in enumerate(shapes):
        case = {"m": m, "n": n, "k": k}
        time = TRUE_PARAMS["overhead_ms"] + max(
            flops(case) / (TRUE_PARAMS["compute_tflops"] * 1e9),
            ideal_bytes(case, META["dtype"]) / (TRUE_PARAMS["bandwidth_gbps"] * 1e6),
        )
        time *= float(1 + rng.uniform(-noise, noise))
        rows.append({
            **case, **META,
            "case_id": f"{prefix}{index:02d}", "split": split,
            "status": "ok",
            "correctness": {"passed": True, "check_kind": "synthetic_contract"},
            "median_ms": time, "samples_ms": [time * 0.999, time, time * 1.001],
            "flops": flops(case), "ideal_bytes": ideal_bytes(case, META["dtype"]),
        })
    return rows


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = records(TRAIN)
        cls.held_out = records(HELD_OUT, "test", "t")
        cls.model = fit_model(cls.train)

    def test_known_roofline_recovers_parameters_and_generalizes_to_unseen_shapes(self):
        for name, expected in TRUE_PARAMS.items():
            self.assertAlmostEqual(self.model["params"][name] / expected, 1.0, places=5)
        result = evaluate_model(self.model, self.held_out)
        self.assertLess(result["max_ape_pct"], 0.001)
        self.assertEqual(result["n"], len(HELD_OUT))
        self.assertEqual(result["pairwise_ordering_accuracy"], 1.0)
        self.assertLess(result["median_ape_pct"], result["flop_only_baseline"]["median_ape_pct"])
        self.assertEqual(self.model["fit_diagnostics"]["sensitivity_rank"], 3)
        self.assertGreater(self.model["fit_diagnostics"]["kernel_term_hypothesis_counts"]["compute"], 0)
        self.assertGreater(self.model["fit_diagnostics"]["kernel_term_hypothesis_counts"]["memory"], 0)

    def test_small_repeat_noise_preserves_useful_holdout_prediction(self):
        model = fit_model(records(TRAIN, noise=0.01))
        report = evaluate_model(model, self.held_out)
        self.assertLess(report["median_ape_pct"], 2.0)
        self.assertLess(report["max_ape_pct"], 3.0)

    def test_input_data_are_not_mutated_and_model_is_json_finite(self):
        rows = copy.deepcopy(self.train)
        before = copy.deepcopy(rows)
        model = fit_model(rows)
        self.assertEqual(rows, before)
        serialized = json.dumps(model, allow_nan=False)
        self.assertIn("effective", serialized)
        self.assertEqual(model["train_case_ids"], [row["case_id"] for row in rows])
        self.assertEqual(len(model["training_data_hash"]), 64)
        model_before = copy.deepcopy(model)
        json.dumps(evaluate_model(model, self.held_out), allow_nan=False)
        self.assertEqual(model, model_before)

    def test_protected_splits_never_enter_parameter_fit(self):
        for protected in ("development", "test", "challenge"):
            with self.subTest(split=protected):
                rows = copy.deepcopy(self.train)
                rows[-1]["split"] = protected
                with self.assertRaisesRegex(LabError, "Protected split"):
                    fit_model(rows)

    def test_candidate_measurements_can_extend_training_without_using_targets_from_test(self):
        extra = records([HELD_OUT[0]], "candidate", "candidate")
        model = fit_model(self.train + extra)
        self.assertIn(extra[0]["case_id"], model["train_case_ids"])
        self.assertEqual(len(model["train_case_ids"]), len(TRAIN) + 1)
        self.assertTrue(all(case_id not in model["train_case_ids"] for case_id in [row["case_id"] for row in self.held_out]))

    def test_mixed_provenance_is_rejected_for_fit_and_prediction(self):
        changes = {
            "measurement_kind": "cuda_measured", "environment_id": "different-device-driver",
            "dtype": "float16", "cache_policy": "cold", "plan_hash": "different-registered-plan",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                rows = copy.deepcopy(self.train)
                rows[-1][field] = value
                with self.assertRaises(LabError):
                    fit_model(rows)
                held = copy.deepcopy(self.held_out)
                # All rows agree with each other but disagree with the fitted model.
                for row in held:
                    row[field] = value
                with self.assertRaises(LabError):
                    predict_rows(self.model, held)

    def test_invalid_timings_do_not_enter_fit_or_evaluation(self):
        for bad in (float("nan"), float("inf"), float("-inf"), -0.1, 0.0, True, "0.01"):
            with self.subTest(time=bad):
                train = copy.deepcopy(self.train)
                train[0]["median_ms"] = bad
                with self.assertRaises(LabError):
                    fit_model(train)
                held = copy.deepcopy(self.held_out)
                held[0]["samples_ms"][0] = bad
                with self.assertRaises(LabError):
                    evaluate_model(self.model, held)

    def test_sample_median_and_operation_accounting_are_checked(self):
        for field in ("median_ms", "flops", "ideal_bytes"):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.train)
                rows[0][field] *= 1.5
                with self.assertRaises(LabError):
                    fit_model(rows)
        rows = copy.deepcopy(self.train)
        rows[0]["samples_ms"] = rows[0]["samples_ms"][:2]
        with self.assertRaisesRegex(LabError, "three repeat"):
            fit_model(rows)

    def test_failed_unchecked_and_duplicate_cases_are_rejected(self):
        rows = copy.deepcopy(self.train)
        rows[0]["correctness"]["passed"] = False
        with self.assertRaisesRegex(LabError, "unchecked"):
            fit_model(rows)
        rows = copy.deepcopy(self.train)
        rows[0]["status"] = "failed"
        with self.assertRaises(LabError):
            fit_model(rows)
        with self.assertRaisesRegex(LabError, "duplicate"):
            fit_model(self.train + [self.train[0]])
        duplicate_shape = copy.deepcopy(self.train[0])
        duplicate_shape["case_id"] = "different-id-same-shape"
        with self.assertRaisesRegex(LabError, "Duplicate matrix shapes"):
            fit_model(self.train + [duplicate_shape])
        with self.assertRaisesRegex(LabError, "four distinct"):
            fit_model(self.train[:3])

    def test_explanation_uses_milliseconds_and_decimal_throughput_units(self):
        model = copy.deepcopy(self.model)
        model["params"] = TRUE_PARAMS.copy()
        case = {"id": "new-shape", "m": 128, "n": 1024, "k": 1024}
        explanation = explain_prediction(model, case, "bfloat16")
        expected_compute = 2 * 128 * 1024 * 1024 / (80 * 1e9)
        expected_memory = 2 * (128 * 1024 + 1024 * 1024 + 128 * 1024) / (700 * 1e6)
        self.assertEqual(explanation["case_id"], "new-shape")
        self.assertAlmostEqual(explanation["compute_ms"], expected_compute, places=14)
        self.assertAlmostEqual(explanation["memory_ms"], expected_memory, places=14)
        self.assertAlmostEqual(explanation["predicted_ms"], 0.004 + max(expected_compute, expected_memory), places=14)
        self.assertEqual(explanation["regime_hypothesis"], "overhead")
        with self.assertRaisesRegex(LabError, "dtype"):
            explain_prediction(model, case, "float16")
        with self.assertRaisesRegex(LabError, "environment_id"):
            explain_prediction(model, {**case, "environment_id": "other"}, "bfloat16")
        with self.assertRaises(LabError):
            explain_prediction(model, {**case, "m": 128.0}, "bfloat16")

    def test_exact_zero_residuals_are_valid_and_finite(self):
        # Construct targets by the actual prediction path to avoid rounding noise.
        rows = copy.deepcopy(self.held_out)
        for row in rows:
            value = explain_prediction(self.model, row, row["dtype"])["predicted_ms"]
            row["median_ms"] = value
            row["samples_ms"] = [value, value, value]
        report = evaluate_model(self.model, rows)
        self.assertEqual(report["median_ape_pct"], 0.0)
        self.assertEqual(report["p90_ape_pct"], 0.0)
        self.assertEqual(report["max_ape_pct"], 0.0)
        self.assertEqual(report["rmse_ms"], 0.0)
        self.assertTrue(all(row["error_ms"] == 0.0 for row in report["predictions"]))
        json.dumps(report, allow_nan=False)

    def test_rank_metric_excludes_near_ties_and_uses_null_when_no_order_is_known(self):
        rows = copy.deepcopy(self.held_out[:2])
        for row in rows:
            row["median_ms"] = 0.01
            row["samples_ms"] = [0.01, 0.01, 0.01]
        report = evaluate_model(self.model, rows)
        self.assertEqual(report["ranking_pairs"], 0)
        self.assertEqual(report["excluded_actual_ties"], 1)
        self.assertIsNone(report["pairwise_ordering_accuracy"])

    def test_unidentifiable_calibration_is_flagged_instead_of_claimed_as_peak(self):
        # For fixed M,N, FLOPs and ideal bytes are affinely dependent.  A good
        # latency fit cannot by itself establish both physical rate parameters.
        rows = records([(1, 1024, k) for k in (512, 1024, 2048, 4096, 8192)])
        model = fit_model(rows)
        diagnostics = model["fit_diagnostics"]
        warnings = " ".join(diagnostics["warnings"])
        self.assertTrue("identifiable" in warnings or "multi-start" in warnings)
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in model["params"].values()))
        self.assertIn("not device peak", " ".join(model["assumptions"]))

    def test_corrupt_model_parameters_are_rejected(self):
        for field in TRUE_PARAMS:
            for value in (float("nan"), float("inf"), -1.0, 0.0):
                with self.subTest(field=field, value=value):
                    model = copy.deepcopy(self.model)
                    model["params"][field] = value
                    with self.assertRaises(LabError):
                        predict_rows(model, self.held_out)

    def test_schema_and_required_prediction_columns(self):
        report = evaluate_model(self.model, self.held_out)
        required = {"case_id", "m", "n", "k", "split", "actual_ms", "predicted_ms", "ape_pct", "error_ms", "regime_hypothesis"}
        for row in report["predictions"]:
            self.assertTrue(required <= row.keys())
            self.assertIn(row["regime_hypothesis"], {"overhead", "compute", "memory"})
        self.assertEqual(self.model["measurement_kind"], "simulated")
        self.assertEqual(report["flop_only_baseline"]["n"], len(HELD_OUT))
        self.assertEqual(self.model["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
