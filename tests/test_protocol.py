"""CPU-only checks of registered coverage and leakage barriers; no timing claim."""

import copy
import json
from collections import Counter
from pathlib import Path
import unittest

from linear_perf.common import LabError, digest, validate_plan
from linear_perf.protocol import (
    build_plan, case_groups, diagnostic_cases, diagnostic_shapes,
    primary_cases, protocol_signature, validate_study_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def shape(case):
    return tuple(case[k] for k in ("m", "n", "k"))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan()

    def test_checked_in_manifests_are_generated_contracts(self):
        for dtype, prefix in (("float16", "fp16"), ("bfloat16", "bf16")):
            with self.subTest(dtype=dtype):
                frozen = json.loads((ROOT / "configs" / f"{prefix}_graph.json").read_text())
                self.assertEqual(frozen, build_plan(dtype))
                self.assertIs(validate_study_plan(frozen), frozen)
                self.assertIs(validate_plan(frozen), frozen)

    def test_manifest_is_deterministic_but_changes_order_seed_contract(self):
        repeated = build_plan()
        different_seed = build_plan(seed=11)
        self.assertEqual(self.plan, repeated)
        self.assertEqual(self.plan["cases"], different_seed["cases"])
        self.assertNotEqual(digest(self.plan), digest(different_seed))
        self.assertEqual(protocol_signature(self.plan), protocol_signature(different_seed))
        self.assertNotEqual(self.plan["study"]["acquisition"]["seeds"], different_seed["study"]["acquisition"]["seeds"])

    def test_main_counts_and_matrix_shape_fields(self):
        cases = self.plan["cases"]
        roles = self.plan["study"]["case_metadata"]
        self.assertEqual(len(cases), 104)
        self.assertEqual(len(set(shape(c) for c in cases)), 104)
        self.assertEqual(Counter(m["role"] for m in roles.values()),
                         {"primary": 72, "candidate_pool": 24, "diagnostic": 8})
        self.assertEqual(Counter(c["split"] for c in primary_cases(self.plan)),
                         {"calibration": 24, "development": 12, "test": 24, "challenge": 12})
        self.assertTrue(all(set(c) == {"id", "m", "n", "k", "split"} for c in cases))

    def test_diagnostic_registry_never_enters_training_or_acquisition(self):
        reserved = diagnostic_shapes()
        self.assertEqual(len(reserved), 8)
        observed = {shape(c) for c in diagnostic_cases(self.plan)}
        self.assertEqual(reserved, observed)
        for case in self.plan["cases"]:
            role = self.plan["study"]["case_metadata"][case["id"]]["role"]
            with self.subTest(case=case["id"]):
                if role == "diagnostic":
                    self.assertIn(shape(case), reserved)
                else:
                    self.assertNotIn(shape(case), reserved)

    def test_family_holdout_not_random_rows_from_the_same_families(self):
        calibration = primary_cases(self.plan, {"calibration"})
        challenge = primary_cases(self.plan, {"challenge"})
        challenge_families = {(c["n"], c["k"]) for c in challenge}
        other_families = {(c["n"], c["k"]) for c in self.plan["cases"]
                          if c["split"] != "challenge" and c["id"] not in self.plan["study"]["diagnostic_case_ids"]}
        self.assertFalse(challenge_families & other_families)
        self.assertEqual(len(challenge_families), 4)
        self.assertEqual(len({(c["n"], c["k"]) for c in calibration}), 6)
        for subset in (calibration, challenge):
            self.assertEqual({case_groups(c, self.plan)["alignment"] for c in subset},
                             {"NK_multiple_of_16", "NK_irregular"})
            self.assertEqual({case_groups(c, self.plan)["m_bucket"] for c in subset},
                             {"small_m_le_16", "medium_m_17_to_256", "large_m_gt_256"})

    def test_test_extrapolation_is_labeled_instead_of_claimed_all_interpolation(self):
        training_max = max(c["m"] for c in primary_cases(self.plan, {"calibration"}))
        extrapolation = [c for c in primary_cases(self.plan, {"test"}) if c["m"] > training_max]
        self.assertEqual(len(extrapolation), 6)
        for c in extrapolation:
            self.assertEqual(case_groups(c, self.plan)["coverage"], "M_extrapolation_in_seen_NK_family")

    def test_candidate_pool_and_diagnostics_stay_disjoint(self):
        v = self.plan["study"]
        self.assertFalse(set(v["candidate_pool_ids"]) & set(v["diagnostic_case_ids"]))
        self.assertTrue(set(v["control_case_ids"]) <= set(v["diagnostic_case_ids"]))
        for key in ("profile_case_ids", "rotation_case_ids"):
            self.assertTrue(set(v["diagnostics"][key]) <= set(v["diagnostic_case_ids"]))
        self.assertEqual({c["split"] for c in diagnostic_cases(self.plan)}, {"candidate"})

    def test_rejects_extra_per_case_fields_that_break_matrix_schema(self):
        self.plan["cases"][0]["role"] = "primary"
        with self.assertRaises(LabError):
            validate_study_plan(self.plan)

    def test_rejects_role_relabeling_that_would_feed_diagnostics_to_acquisition(self):
        self.plan["study"]["case_metadata"]["x01"]["role"] = "candidate_pool"
        with self.assertRaises(LabError):
            validate_study_plan(self.plan)

    def test_rejects_pool_id_injection_of_diagnostic(self):
        self.plan["study"]["candidate_pool_ids"][0] = "x01"
        with self.assertRaises(LabError):
            validate_study_plan(self.plan)

    def test_rejects_moving_heldout_case_into_calibration(self):
        lookup = {c["id"]: c for c in self.plan["cases"]}
        lookup["c01"]["split"], lookup["h01"]["split"] = "challenge", "calibration"
        with self.assertRaises(LabError):
            validate_study_plan(self.plan)

    def test_rejects_unregistered_training_shape(self):
        # An unregistered shape must fail even without a duplicate in the plan.
        c = self.plan["cases"][0]
        c.update(m=16, n=8192, k=2048)
        with self.assertRaises(LabError):
            validate_study_plan(self.plan)

    def test_rejects_inserting_observed_latency_into_predeclared_metadata(self):
        self.plan["study"]["case_metadata"]["c01"]["observed_ms"] = 0.01
        with self.assertRaises(LabError):
            validate_study_plan(self.plan)

    def test_changed_primary_timing_or_budget_requires_different_protocol(self):
        changes = (("timing", "iterations", 100), ("timing", "repeats", 7),
                   ("budget", "max_new_cases", 8), ("budget", "max_wall_seconds", 3600),
                   ("collection", "order", "sorted"))
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                plan = copy.deepcopy(self.plan)
                plan[section][key] = value
                with self.assertRaises(LabError):
                    validate_study_plan(plan)

    def test_rejects_unregistered_sampling_or_diagnostic_contract(self):
        changes = (("diagnostics", "iterations", [10, 50, 100]),
                   ("diagnostics", "passes", 2.0),
                   ("diagnostics", "rotation_buffer_counts", [1, 16]),
                   ("acquisition", "seeds", [1] * 10),
                   ("acquisition", "batch_size", 4),
                   ("acquisition", "budgets", [2, 8]))
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                plan = copy.deepcopy(self.plan)
                plan["study"][section][key] = value
                with self.assertRaises(LabError):
                    validate_study_plan(plan)

    def test_rejects_false_claims_in_registered_interpretation(self):
        changes = (("diagnostics_excluded_from_model", False),
                   ("profiler_times_excluded_from_model", False),
                   ("rotation_is_cache_flush", True),
                   ("accuracy_improvement_is_established", True),
                   ("quality_thresholds_are_vendor_standards", True),
                   ("benchmark_certification", "MLPerf compliant"),
                   ("shape_metadata_contains_latency", True))
        for key, value in changes:
            with self.subTest(key=key):
                plan = copy.deepcopy(self.plan)
                plan["study"]["interpretation"][key] = value
                with self.assertRaises(LabError):
                    validate_study_plan(plan)

    def test_signature_binds_measurement_contract_not_runtime_identity(self):
        signature = protocol_signature(self.plan)
        for key, value in (("environment_id", "other_gpu"), ("session_id", "session_3"), ("gpu_uuid", "GPU-abc")):
            p = copy.deepcopy(self.plan)
            p[key] = value
            self.assertEqual(signature, protocol_signature(p))
            self.assertNotEqual(digest(self.plan), digest(p))
        self.assertNotEqual(signature, protocol_signature(build_plan("bfloat16")))
        p = copy.deepcopy(self.plan)
        p["timing"]["iterations"] = 100
        self.assertNotEqual(signature, protocol_signature(p))
        p = copy.deepcopy(self.plan)
        p["cache_policy"] = "diagnostic_working_set_rotation"
        self.assertNotEqual(signature, protocol_signature(p))

    def test_helpers_return_copies_and_validate_case_identity(self):
        initial_hash = digest(self.plan)
        primary_cases(self.plan)[0]["m"] = 99
        diagnostic_cases(self.plan)[0]["n"] = 99
        diagnostic_shapes().clear()
        self.assertEqual(len(diagnostic_shapes()), 8)
        self.assertEqual(digest(self.plan), initial_hash)
        c = self.plan["cases"][0]
        measurement_style = {"case_id": c["id"], "m": c["m"], "n": c["n"], "k": c["k"]}
        self.assertEqual(case_groups(c, self.plan), case_groups(measurement_style, self.plan))
        measurement_style["m"] += 1
        with self.assertRaises(LabError):
            case_groups(measurement_style, self.plan)
        with self.assertRaises(LabError):
            primary_cases(self.plan, {"candidate"})

    def test_seed_and_dtype_input_validation(self):
        for seed in (-1, True, 1.5, 2**31):
            with self.subTest(seed=seed), self.assertRaises(LabError):
                build_plan(seed=seed)
        with self.assertRaises(LabError):
            build_plan("float32")


if __name__ == "__main__":
    unittest.main()
