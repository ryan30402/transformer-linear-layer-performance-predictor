"""CPU-only tests for raw evidence retention, aggregation, and isolation."""

from __future__ import annotations

import copy
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from linear_perf.collection import (ReplicatedCollectionError, collect_replicated,
                                    replica_dispersion, validate_aggregate_row)
from linear_perf.common import (LabError, digest, read_json, read_jsonl,
                                validate_measurements, write_json)
from linear_perf.simulation import simulate_cases, simulated_environment


PROJECT = Path(__file__).resolve().parents[1]


def registered_plan():
    plan = read_json(PROJECT / "configs" / "fp16_graph.json")
    plan["name"] = "TEST_ONLY_simulated_replicated_contract"
    plan["collection"] = {"blocks": 3, "order": "seeded_shuffle_per_block",
                          "aggregation": "median_of_block_medians"}
    return plan


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = registered_plan()
        self.env = simulated_environment(self.plan)
        self.cases = self.plan["cases"][:4]

    def tearDown(self):
        self.temp.cleanup()

    def collect(self, **kwargs):
        return collect_replicated(self.plan, self.cases, self.env, "cuda:0", **kwargs)

    def test_three_complete_passes_preserve_every_raw_repeat(self):
        path = self.root / "raw_evidence.json"
        rows = self.collect(raw_output=path)
        evidence = read_json(path)
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["completed_blocks"], 3)
        self.assertEqual(len(evidence["rows"]), 12)
        self.assertEqual(evidence["raw_rows_hash"], digest(evidence["rows"]))
        for order in evidence["block_orders"]:
            self.assertEqual(set(order["case_ids"]), {case["id"] for case in self.cases})
            actual = [row["case_id"] for row in evidence["rows"] if row["block_index"] == order["block_index"]]
            self.assertEqual(actual, order["case_ids"])
        for row in rows:
            validate_aggregate_row(row, expected_blocks=3, expected_repeats=9)
            self.assertEqual(row["measurement_kind"], "simulated")
            self.assertEqual(row["sample_unit"], "block_median")
            self.assertEqual(row["samples_ms"], [block["median_ms"] for block in row["measurement_blocks"]])
            self.assertEqual(sum(len(block["samples_ms"]) for block in row["measurement_blocks"]), 27)
            self.assertTrue(all(block["measurement_kind"] == "simulated_block" and block["excluded_from_model"]
                                for block in row["measurement_blocks"]))
        validate_measurements(rows, self.plan)

    def test_block_order_is_reproducible_and_independent_of_requested_list_order(self):
        one, two = self.root / "one.json", self.root / "two.json"
        self.collect(raw_output=one)
        collect_replicated(self.plan, list(reversed(self.cases)), self.env, "cuda:0", raw_output=two)
        self.assertEqual(read_json(one)["block_orders"], read_json(two)["block_orders"])

    def test_target_is_median_of_all_pass_medians_and_outlier_is_retained(self):
        calls = {}

        def varied(plan, cases, *, environment):
            raw = simulate_cases(plan, cases, environment=environment)
            row = raw[0]
            index = calls.get(row["case_id"], 0)
            calls[row["case_id"]] = index + 1
            value = [1.0, 100.0, 2.0][index]
            row["samples_ms"] = [value] * 9
            row["median_ms"] = value
            row["correctness"]["relative_l2_error"] = index * 0.0001
            row["correctness"]["max_abs_error"] = index * 0.0002
            return raw

        with patch("linear_perf.simulation.simulate_cases", side_effect=varied):
            rows = self.collect()
        for row in rows:
            self.assertEqual(row["samples_ms"], [1.0, 100.0, 2.0])
            self.assertEqual(row["median_ms"], 2.0)
            self.assertEqual(row["correctness"]["relative_l2_error"], 0.0002)
            self.assertEqual(row["correctness"]["max_abs_error"], 0.0004)
            self.assertEqual(row["correctness"]["checked_blocks"], 3)
            stats = replica_dispersion(row)
            self.assertEqual(stats["max_over_min_block_median"], 100.0)
            self.assertEqual(stats["absolute_block_median_range_us"], 99_000.0)
            self.assertGreater(stats["block_median_sample_cv_pct"], 100)
            self.assertIn("not a confidence", stats["interpretation"])

    def test_raw_file_is_persisted_after_every_successful_case(self):
        path = self.root / "raw_evidence.json"
        calls = [0]

        def inspect(plan, cases, *, environment):
            self.assertEqual(len(read_json(path)["rows"]), calls[0])
            calls[0] += 1
            return simulate_cases(plan, cases, environment=environment)

        with patch("linear_perf.simulation.simulate_cases", side_effect=inspect):
            self.collect(raw_output=path)
        self.assertEqual(calls[0], 12)

    def test_failure_preserves_prior_pass_and_current_partial_rows(self):
        path = self.root / "raw_evidence.json"
        calls = [0]

        def fail(plan, cases, *, environment):
            calls[0] += 1
            if calls[0] == 6:
                raise RuntimeError("synthetic collector interruption")
            return simulate_cases(plan, cases, environment=environment)

        with patch("linear_perf.simulation.simulate_cases", side_effect=fail):
            with self.assertRaisesRegex(ReplicatedCollectionError, "all completed raw evidence retained") as caught:
                self.collect(raw_output=path)
        state = read_json(path)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed_blocks"], 1)
        self.assertEqual(len(state["rows"]), 5)
        self.assertEqual(len(caught.exception.raw_completed_rows), 5)
        self.assertEqual(caught.exception.completed_rows, [])
        self.assertEqual(caught.exception.raw_evidence_path, str(path))

    def test_partial_complete_aggregates_are_distinguished_from_incomplete_cases(self):
        calls = [0]

        def fail(plan, cases, *, environment):
            calls[0] += 1
            if calls[0] == 10:
                raise RuntimeError("after first case of third pass")
            return simulate_cases(plan, cases, environment=environment)

        with patch("linear_perf.simulation.simulate_cases", side_effect=fail):
            with self.assertRaises(ReplicatedCollectionError) as caught:
                self.collect(raw_output=self.root / "raw.json")
        self.assertEqual(len(caught.exception.raw_completed_rows), 9)
        self.assertEqual(len(caught.exception.completed_rows), 1)
        validate_aggregate_row(caught.exception.completed_rows[0], expected_blocks=3)

    def test_raw_fitter_rejection_and_aggregate_fitter_acceptance(self):
        from linear_perf.predictor import fit_model
        rows = self.collect()
        model = fit_model(rows)
        self.assertEqual(model["sample_unit"], "block_median")
        with self.assertRaisesRegex(LabError, "measurement_kind"):
            fit_model([row["measurement_blocks"][0] for row in rows])

    def test_same_median_with_different_raw_noise_changes_training_evidence_hash(self):
        from linear_perf.predictor import fit_model
        original = self.collect()
        changed = copy.deepcopy(original)
        block = changed[0]["measurement_blocks"][0]
        minimum_index = min(range(len(block["samples_ms"])), key=lambda i: block["samples_ms"][i])
        block["samples_ms"][minimum_index] *= 0.8
        self.assertEqual(statistics.median(block["samples_ms"]), block["median_ms"])
        changed[0]["measurement_blocks_hash"] = digest(changed[0]["measurement_blocks"])
        validate_aggregate_row(changed[0], expected_blocks=3)
        self.assertEqual([r["median_ms"] for r in original], [r["median_ms"] for r in changed])
        first, second = fit_model(original), fit_model(changed)
        self.assertNotEqual(first["training_data_hash"], second["training_data_hash"])
        self.assertEqual(first["params"], second["params"])

    def test_per_visit_preparation_and_memory_metadata_remain_in_every_raw_block(self):
        calls = {}

        def prepared(plan, cases, *, environment):
            rows = simulate_cases(plan, cases, environment=environment)
            row = rows[0]
            index = calls.get(row["case_id"], 0) + 1
            calls[row["case_id"]] = index
            row.update(warmup={"seconds": index * 0.25}, conditioning_budget={"used_calls": index * 500},
                       memory_guard={"available_bytes_before_case": index * 1000})
            return rows

        with patch("linear_perf.simulation.simulate_cases", side_effect=prepared):
            rows = self.collect()
        for row in rows:
            self.assertTrue(all(key not in row for key in ("warmup", "conditioning_budget", "memory_guard")))
            self.assertEqual([block["warmup"]["seconds"] for block in row["measurement_blocks"]], [0.25, 0.5, 0.75])
            self.assertEqual([block["conditioning_budget"]["used_calls"] for block in row["measurement_blocks"]], [500, 1000, 1500])
            self.assertEqual([block["memory_guard"]["available_bytes_before_case"] for block in row["measurement_blocks"]], [1000, 2000, 3000])

    def test_aggregate_evidence_tampering_is_rejected_even_if_hash_is_recomputed(self):
        original = self.collect()[0]
        mutations = [
            lambda r: r["measurement_blocks"][1].update(environment_id="other_gpu"),
            lambda r: r["measurement_blocks"][1].update(block_index=1),
            lambda r: r["measurement_blocks"][1].update(m=r["m"] + 1),
            lambda r: r["measurement_blocks"][1].update(split="test"),
            lambda r: r["measurement_blocks"][1].update(sample_unit="block_median"),
            lambda r: r["measurement_blocks"][1].update(excluded_from_model=False),
            lambda r: r["measurement_blocks"][1].update(measurement_kind="simulated"),
            lambda r: r["measurement_blocks"][1]["correctness"].update(passed=False),
            lambda r: r["measurement_blocks"][1]["samples_ms"].__setitem__(0, math.inf),
            lambda r: r["measurement_blocks"][1].update(median_ms=999.0),
        ]
        for mutate in mutations:
            row = copy.deepcopy(original)
            mutate(row)
            try:
                row["measurement_blocks_hash"] = digest(row["measurement_blocks"])
            except ValueError:
                pass  # Non-finite JSON must itself be rejected.
            with self.subTest(mutation=mutate), self.assertRaises((LabError, ValueError)):
                validate_aggregate_row(row, expected_blocks=3)

    def test_raw_hash_change_without_aggregate_change_is_rejected(self):
        row = self.collect()[0]
        row["measurement_blocks"][0]["samples_ms"][0] *= 1.01
        with self.assertRaisesRegex(LabError, "hash mismatch"):
            validate_aggregate_row(row)

    def test_aggregate_cannot_claim_a_better_correctness_summary(self):
        row = self.collect()[0]
        row["correctness"]["checked_blocks"] = 1
        with self.assertRaisesRegex(LabError, "correctness summary"):
            validate_aggregate_row(row)

    def test_existing_raw_evidence_is_never_overwritten(self):
        path = self.root / "raw.json"
        path.write_text("preserve me")
        with patch("linear_perf.simulation.simulate_cases") as collector:
            with self.assertRaisesRegex(LabError, "refusing to overwrite"):
                self.collect(raw_output=path)
            collector.assert_not_called()
        self.assertEqual(path.read_text(), "preserve me")

    def test_expired_deadline_preserves_empty_failure_record_without_measurement(self):
        path = self.root / "raw.json"
        with patch("linear_perf.collection.time.monotonic", return_value=100.0), \
             patch("linear_perf.simulation.simulate_cases") as collector:
            with self.assertRaisesRegex(ReplicatedCollectionError, "wall-time budget"):
                self.collect(raw_output=path, deadline_monotonic=99.0)
            collector.assert_not_called()
        self.assertEqual(read_json(path)["rows"], [])
        self.assertEqual(read_json(path)["status"], "failed")

    def test_case_and_environment_mismatch_fail_before_measurement(self):
        with patch("linear_perf.simulation.simulate_cases") as collector:
            for cases, env in [(self.cases + [self.cases[0]], self.env),
                               ([{**self.cases[0], "m": 999}], self.env),
                               (self.cases, {**self.env, "plan_hash": "wrong"})]:
                with self.assertRaises(LabError):
                    collect_replicated(self.plan, cases, env, "cuda:0")
            collector.assert_not_called()




if __name__ == "__main__":
    unittest.main()
