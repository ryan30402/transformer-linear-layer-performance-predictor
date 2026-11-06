"""Behavioral tests for the offline model-development boundary; CPU only."""

import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from linear_perf import selection
from linear_perf.common import LabError, digest


def training_fixture():
    rows = []
    families = ((128, 96), (256, 128), (512, 256), (768, 512), (257, 129), (513, 257))
    for family_index, (n, k) in enumerate(families):
        for m in (1, 16, 128, 1024):
            value = 0.003 + max(2 * m * n * k / 90e9, 2 * (m*k + n*k + m*n) / 800e6)
            value *= 1 + 0.3 * (n % 16 != 0) + 0.03 * math.log2(m)
            rows.append({
                "case_id": f"f{family_index:02d}_m{m:04d}", "m": m, "n": n, "k": k,
                "median_ms": value, "split": "train", "weight": family_index + 1,
                "dtype": "float16", "cache_policy": "warm_reuse",
                "environment_id": "SIMULATED_SELECTION_TEST", "measurement_kind": "simulated",
                "status": "ok", "correctness": {"passed": True},
            })
    return rows


def heldout_fixture():
    rows = []
    for index, row in enumerate(training_fixture()[::4]):
        item = {**row, "case_id": f"test_{index:02d}", "m": 48, "split": "test"}
        item["median_ms"] = 0.004 + 2 * item["m"] * item["n"] * item["k"] / 80e9
        rows.append(item)
    row = {**rows[0], "case_id": "challenge_new_family", "n": 384, "k": 192, "split": "challenge"}
    rows.append(row)
    return rows


class OfflineSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = training_fixture()
        cls.test = heldout_fixture()
        cls.suite = selection.fit_cv_suite(cls.train, sm_count=108)

    def test_heldout_rows_are_rejected_instead_of_silently_filtered(self):
        for split in ("test", "challenge", "stress", "development", "calibration"):
            contaminated = copy.deepcopy(self.train)
            contaminated[0]["split"] = split
            with self.subTest(split=split), self.assertRaises(LabError):
                selection.fit_cv_suite(contaminated, sm_count=108)

    def test_group_folds_have_no_family_leakage_and_cover_every_train_case_once(self):
        validation_ids = []
        for fold in self.suite["cv"]["folds"]:
            left = {tuple(family) for family in fold["train_families"]}
            right = {tuple(family) for family in fold["validation_families"]}
            self.assertFalse(left & right)
            self.assertFalse(set(fold["train_case_ids"]) & set(fold["validation_case_ids"]))
            validation_ids.extend(fold["validation_case_ids"])
        self.assertCountEqual(validation_ids, [row["case_id"] for row in self.train])
        self.assertEqual(len(validation_ids), len(set(validation_ids)))
        self.assertEqual(self.suite["cv"]["method"], "leave_one_NK_family_out")

    def test_balanced_group_folds_ignore_target_times(self):
        more = training_fixture()
        for family_index in range(7, 11):
            for m in (1, 16, 128, 1024):
                more.append({**more[0], "case_id": f"extra_{family_index}_{m}",
                             "m": m, "n": 1000 + family_index, "k": 1200 + family_index})
        before = selection.grouped_folds(more, max_folds=4)
        for index, row in enumerate(more):
            row["median_ms"] *= 10**(index % 5)
        self.assertEqual(before, selection.grouped_folds(more, max_folds=4))
        self.assertEqual(len(before), 4)

    def test_standardization_fits_only_each_fold_training_shapes(self):
        lookup = {row["case_id"]: row for row in self.train}
        all_features = selection.feature_matrix(self.train, "float16", sm_count=108)
        changed = False
        for fold in self.suite["cv"]["folds"]:
            features = selection.feature_matrix([lookup[cid] for cid in fold["train_case_ids"]], "float16", sm_count=108)
            np.testing.assert_allclose(fold["feature_mean"], np.mean(features, axis=0), rtol=0, atol=1e-12)
            changed |= not np.allclose(fold["feature_mean"], np.mean(all_features, axis=0))
        self.assertTrue(changed)

    def test_all_candidates_refit_on_every_training_record(self):
        wanted = sorted(row["case_id"] for row in self.train)
        self.assertEqual(set(self.suite["models"]), set(selection.MODEL_ORDER))
        for model in self.suite["models"].values():
            self.assertEqual(model["train_case_ids"], wanted)
            self.assertEqual(model["training_data_hash"], self.suite["training_data_hash"])
        for name in ("shape_log_ridge", "roofline_residual_krr"):
            np.testing.assert_allclose(self.suite["models"][name]["feature_mean"],
                                       np.mean(selection.feature_matrix(self.train, "float16", sm_count=108), axis=0))

    def test_mean_criterion_does_not_hide_five_severe_errors_in_twelve_cases(self):
        rows = [{**self.train[0], "case_id": str(i), "median_ms": 1.0, "weight": 1.0} for i in range(12)]
        uneven = selection.regression_metrics(rows, [1.13] * 7 + [1.7] * 5)
        uniform = selection.regression_metrics(rows, [1.2] * 12)
        self.assertLess(uneven["median_ape_pct"], uniform["median_ape_pct"])
        self.assertGreater(uneven["mean_ape_pct"], uniform["mean_ape_pct"])
        self.assertGreater(uneven["weighted_p90_ape_pct"], uniform["weighted_p90_ape_pct"])
        self.assertEqual(self.suite["selection_score_key"], "weighted_mean_ape_pct")

    def test_declared_weights_have_explicit_metrics_and_ignore_zero_mass(self):
        rows = [{**self.train[0], "case_id": str(i), "median_ms": 1.0, "weight": weight}
                for i, weight in enumerate((1.0, 9.0, 0.0))]
        metrics = selection.regression_metrics(rows, [2.0, 1.0, 101.0])
        self.assertAlmostEqual(metrics["weighted_mean_ape_pct"], 10.0)
        self.assertAlmostEqual(metrics["weighted_max_ape_pct"], 100.0)
        self.assertAlmostEqual(metrics["max_ape_pct"], 10000.0)
        self.assertEqual(metrics["n_positive_weight"], 2)
        self.assertEqual(metrics["weighted_p90_ape_pct"], 0.0)

    def test_ridge_and_krr_share_shape_features(self):
        ridge = self.suite["models"]["shape_log_ridge"]
        krr = self.suite["models"]["roofline_residual_krr"]
        self.assertEqual(ridge["feature_names"], krr["feature_names"])
        self.assertEqual(ridge["feature_mean"], krr["feature_mean"])
        self.assertEqual(ridge["feature_scale"], krr["feature_scale"])

    def test_sm_wave_proxies_are_explicit_and_never_assume_a100(self):
        cases = [{"m": 1024, "n": 8192, "k": 2048}]
        unknown = selection.feature_matrix(cases, "float16")
        a100 = selection.feature_matrix(cases, "float16", sm_count=108)
        other = selection.feature_matrix(cases, "float16", sm_count=58)
        self.assertEqual(unknown[0, -1], 0.0)
        self.assertEqual(a100[0, -1], 1.0)
        for name in ("log2_wave_count_64x64_proxy", "wave_fill_64x64_proxy"):
            index = selection.FEATURE_NAMES.index(name)
            self.assertEqual(unknown[0, index], 0.0)
            self.assertNotEqual(a100[0, index], other[0, index])

    def test_profiler_fields_and_target_times_do_not_enter_features_or_predictions(self):
        noisy = copy.deepcopy(self.test)
        for row in noisy:
            row.update(kernel_names=["made_up_999x999"], actual_tile=[999, 999],
                       profiler_status="unavailable", median_ms=98765.0, cpu_submission_ms=700)
        np.testing.assert_array_equal(selection.feature_matrix(self.test, "float16", sm_count=108),
                                      selection.feature_matrix(noisy, "float16", sm_count=108))
        self.assertEqual(selection.predict_suite(self.suite, self.test), selection.predict_suite(self.suite, noisy))

    def test_evaluation_does_not_fit_select_or_mutate(self):
        before = copy.deepcopy(self.suite)
        altered = copy.deepcopy(self.test)
        for row in altered:
            row["median_ms"] *= 100.0
        with patch.object(selection, "fit_cv_suite", side_effect=AssertionError("fitting during evaluation")), \
             patch.object(selection, "_fit_physical", side_effect=AssertionError("fitting during evaluation")), \
             patch.object(selection, "_fit_learned", side_effect=AssertionError("fitting during evaluation")):
            first = selection.evaluate_cv_suite(self.suite, self.test)
            second = selection.evaluate_cv_suite(self.suite, altered)
        self.assertEqual(before, self.suite)
        self.assertEqual(first["selected_model_id"], second["selected_model_id"])
        for name in selection.MODEL_ORDER:
            self.assertEqual([p["predicted_ms"] for p in first["models"][name]["predictions"]],
                             [p["predicted_ms"] for p in second["models"][name]["predictions"]])
        self.assertFalse(first["refitted_during_evaluation"])

    def test_measurement_timestamps_do_not_need_to_follow_model_selection(self):
        rows = copy.deepcopy(self.test)
        for row in rows:
            row["measured_at_utc"] = "2000-01-01T00:00:00+00:00"
        result = selection.evaluate_cv_suite(self.suite, rows)
        self.assertEqual(len(result["evaluation_case_ids"]), len(rows))

    def test_json_round_trip_preserves_predictions_and_suite(self):
        decoded = json.loads(json.dumps(self.suite, allow_nan=False))
        self.assertEqual(selection.predict_suite(self.suite, self.test), selection.predict_suite(decoded, self.test))
        self.assertIn("tuning", self.suite["cv"]["score_scope"])
        self.assertFalse(self.suite["prediction_interval_available"])

    def test_incompatible_fit_contexts_are_rejected(self):
        for key, value in (("dtype", "bfloat16"), ("cache_policy", "rotating_buffers"),
                           ("measurement_kind", "cuda_measured"), ("environment_id", "ANOTHER_GPU")):
            rows = copy.deepcopy(self.train)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(LabError, "mix modeling contexts"):
                selection.fit_cv_suite(rows, sm_count=108)

    def test_transfer_requires_explicit_permission_and_keeps_precision_policy(self):
        other = copy.deepcopy(self.test)
        for row in other:
            row["environment_id"] = "ANOTHER_SIMULATED_ENV"
        with self.assertRaises(LabError):
            selection.evaluate_cv_suite(self.suite, other, sm_count=58)
        result = selection.evaluate_cv_suite(self.suite, other, sm_count=58, allow_environment_transfer=True)
        self.assertTrue(result["environment_transfer"])
        self.assertEqual(result["target_sm_count"], 58)
        other[0]["cache_policy"] = "different_cache_contract"
        with self.assertRaises(LabError):
            selection.evaluate_cv_suite(self.suite, other, sm_count=58, allow_environment_transfer=True)

    def test_evaluation_rejects_training_shapes_even_if_renamed(self):
        overlap = {**self.train[0], "case_id": "disguised", "split": "test"}
        with self.assertRaisesRegex(LabError, "overlaps"):
            selection.evaluate_cv_suite(self.suite, [overlap])

    def test_unknown_target_hardware_does_not_inherit_source_sm_count(self):
        other = [{**row, "environment_id": "UNKNOWN_TARGET_GPU"} for row in self.test]
        result = selection.evaluate_cv_suite(self.suite, other, allow_environment_transfer=True)
        self.assertIsNone(result["target_sm_count"])
        self.assertTrue(result["target_sm_unknown"])
        clean = sorted(other, key=lambda row: row["case_id"])
        expected = selection._predict_model(self.suite["models"]["roofline_residual_krr"],
                                            clean, "float16", None)
        self.assertEqual([row["predicted_ms"] for row in result["models"]["roofline_residual_krr"]["predictions"]],
                         expected)

    def test_checksum_rejects_changed_selected_model(self):
        changed = copy.deepcopy(self.suite)
        changed["selected_model_id"] = next(name for name in selection.MODEL_ORDER if name != changed["selected_model_id"])
        with self.assertRaisesRegex(LabError, "checksum"):
            selection.predict_suite(changed, self.test)

    def test_large_vocabulary_dimensions_are_not_limited_to_legacy_16384(self):
        cases = [{"m": 1, "n": 151936, "k": 1536}]
        features = selection.feature_matrix(cases, "bfloat16", sm_count=108)
        self.assertTrue(np.all(np.isfinite(features)))
        predictions = selection.predict_suite(self.suite, cases)
        self.assertTrue(all(value > 0 and math.isfinite(value) for values in predictions.values() for value in values))

    def test_bad_targets_weights_and_insufficient_groups_fail_before_fitting(self):
        for field, value in (("median_ms", 0), ("median_ms", math.nan), ("weight", -1), ("weight", math.inf)):
            rows = copy.deepcopy(self.train)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(LabError):
                selection.fit_cv_suite(rows, sm_count=108)
        with self.assertRaisesRegex(LabError, "three N/K families"):
            selection.fit_cv_suite(self.train[:8], sm_count=108)
        rows = copy.deepcopy(self.train)
        for row in rows:
            row["weight"] = 0
        with self.assertRaisesRegex(LabError, "positive total weight"):
            selection.fit_cv_suite(rows, sm_count=108)


if __name__ == "__main__":
    unittest.main()
