"""Architecture provenance and evaluation isolation, without CUDA or latency data."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from linear_perf.common import LabError
from linear_perf.workloads import (SOURCE_DIRECTORY, baseline_seen_shapes, build_workload_manifest,
                                   load_model_specs, parse_model_config, projection_shapes,
                                   validate_workload_manifest)


class WorkloadManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_workload_manifest()

    def test_pinned_source_bytes_and_gqa_dimensions(self):
        sources = load_model_specs()
        self.assertEqual(len(sources), 5)
        specs = {source["id"]: source["spec"] for source in sources}
        qwen = specs["qwen2_5_0_5b"]
        self.assertEqual((qwen["hidden_size"], qwen["head_dim"], qwen["num_key_value_heads"]), (896, 64, 2))
        projections = {item["operation"]: item for item in projection_shapes(qwen)}
        self.assertEqual((projections["q"]["n"], projections["k"]["n"], projections["v"]["n"]), (896, 128, 128))
        self.assertEqual(projections["q"]["occurrences"], 24)
        self.assertEqual(projections["lm_head"]["occurrences"], 1)
        self.assertTrue(qwen["tie_word_embeddings"])
        fused = {item["operation"]: item for item in projection_shapes(qwen, "fused_qkv_and_gate_up")}
        self.assertEqual(fused["qkv"]["n"], 1152)
        self.assertEqual(fused["gate_up"]["n"], 9728)
        self.assertFalse({"q", "k", "v", "gate", "up"} & fused.keys())

    def test_corrupted_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for path in SOURCE_DIRECTORY.glob("*.json"):
                (folder / path.name).write_bytes(path.read_bytes())
            with (folder / "qwen2_5_0_5b.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(LabError, "do not match provenance"):
                load_model_specs(folder)

    def test_reproducible_manifest_has_no_measured_labels(self):
        self.assertEqual(self.manifest, build_workload_manifest())
        text = json.dumps(self.manifest)
        self.assertNotIn("median_ms", text)
        self.assertNotIn("kernel_name", text)
        self.assertNotIn("profiler_duration", text)
        self.assertGreaterEqual(len(self.manifest["cases"]), 550)
        self.assertLessEqual(len(self.manifest["cases"]), 650)
        self.assertEqual(self.manifest["dtypes"], ["float16", "bfloat16"])

    def test_shared_shapes_aggregate_provenance_and_frequency_once(self):
        manifest = self.manifest
        shapes = [(case["m"], case["n"], case["k"]) for case in manifest["cases"]]
        self.assertEqual(len(shapes), len(set(shapes)))
        family = manifest["families"]["nk_2048_2048"]
        self.assertGreater(len({source["model_id"] for source in family["provenance"]}), 1)
        for case in manifest["cases"]:
            if case["split"] == "stress":
                continue
            provenance = manifest["families"][case["family"]]["provenance"]
            expected = sum(row["model_mixture_probability"] * row["layout_probability"] *
                           row["occurrences_per_model_pass"] for row in provenance)
            self.assertAlmostEqual(case["weight"], expected * case["m_sampling_probability"])

    def test_test_m_values_and_challenge_families_are_disjoint(self):
        by_split = {split: [case for case in self.manifest["cases"] if case["split"] == split]
                    for split in ("train", "test", "challenge", "stress")}
        train_families = {(case["n"], case["k"]) for case in by_split["train"]}
        test_families = {(case["n"], case["k"]) for case in by_split["test"]}
        challenge = {(case["n"], case["k"]) for case in by_split["challenge"]}
        self.assertTrue(test_families <= train_families)
        self.assertFalse(challenge & (train_families | test_families))
        self.assertFalse({case["m"] for case in by_split["train"]} & {case["m"] for case in by_split["test"]})
        self.assertEqual(max(case["m"] for case in by_split["train"]), 8192)
        self.assertEqual(len(challenge), 6)
        self.assertTrue(all(case["origin"] == "artificial_irregular_stress" for case in by_split["stress"]))

    def test_already_published_and_newly_exposed_cases_are_never_new_tests(self):
        self.assertEqual(len(baseline_seen_shapes()), 104)
        fresh = {(case["m"], case["n"], case["k"]) for case in self.manifest["cases"] if case["split"] != "train"}
        self.assertFalse(fresh & baseline_seen_shapes())
        held = next(case for case in self.manifest["cases"] if case["split"] == "test")
        replanned = build_workload_manifest(seen_shapes=[held])
        self.assertNotIn(held["id"], {case["id"] for case in replanned["cases"]})
        self.assertTrue(any(case["reason"] == "already_exposed_shape_not_an_independent_evaluation"
                            and case["m"] == held["m"] and case["n"] == held["n"] and case["k"] == held["k"]
                            for case in replanned["excluded_cases"]))

    def test_memory_exclusions_keep_original_dimensions(self):
        excluded = [case for case in self.manifest["excluded_cases"]
                    if case["reason"] == "ideal_tensor_bytes_exceed_declared_per_case_budget"]
        self.assertTrue(excluded)
        # The largest vocabulary weight is too large even at M=1. It is not shrunk.
        self.assertTrue(any((case["m"], case["n"], case["k"]) == (1, 151936, 2048) for case in excluded))
        self.assertTrue(all(case["ideal_tensor_bytes"] <= self.manifest["measurement"]["max_case_bytes"]
                            for case in self.manifest["cases"]))
        for case in self.manifest["cases"]:
            if case["split"] != "test":
                continue
            lower, upper = case["training_m_range"]
            outside = case["coverage"] == "unseen_M_outside_training_family_range_after_budget_exclusions"
            self.assertEqual(outside, not lower <= case["m"] <= upper)

    def test_smoke_has_enough_real_families_for_grouped_cv(self):
        manifest = build_workload_manifest(profile="smoke", dtypes=("float16",))
        self.assertEqual(len(manifest["cases"]), 32)
        train = [case for case in manifest["cases"] if case["split"] == "train"]
        self.assertGreaterEqual(len({(case["n"], case["k"]) for case in train}), 3)
        self.assertTrue(all(case["n"] >= 128 and case["k"] >= 896 for case in train))
        manifest["measurement"]["max_wall_seconds"] = 120
        manifest["dtypes"] = ["bfloat16"]
        self.assertIs(validate_workload_manifest(manifest), manifest)

    def test_validator_rejects_duplicates_and_family_leakage(self):
        duplicate = copy.deepcopy(self.manifest)
        duplicate["cases"].append({**duplicate["cases"][0], "id": "another_id"})
        with self.assertRaisesRegex(LabError, "Duplicate"):
            validate_workload_manifest(duplicate)
        leaked = copy.deepcopy(self.manifest)
        challenge = next(case for case in leaked["cases"] if case["split"] == "challenge")
        challenge["split"] = "train"
        with self.assertRaisesRegex(LabError, "wholly absent"):
            validate_workload_manifest(leaked)
        for bad_value in [0, -1, float("nan"), True]:
            with self.subTest(value=bad_value):
                broken = copy.deepcopy(self.manifest)
                broken["cases"][0]["weight"] = bad_value
                with self.assertRaises(LabError):
                    validate_workload_manifest(broken)

    def test_invalid_architecture_is_not_silently_coerced(self):
        config = json.loads((SOURCE_DIRECTORY / "qwen2_5_0_5b.json").read_text())
        config["num_key_value_heads"] = 3
        with self.assertRaisesRegex(LabError, "head ratio"):
            parse_model_config(config)
        config["num_key_value_heads"] = 2
        config["use_sliding_window"] = True
        with self.assertRaisesRegex(LabError, "separate implementation"):
            parse_model_config(config)

    def test_import_and_plan_do_not_require_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            module = importlib.import_module("linear_perf.workloads")
            self.assertEqual(module.build_workload_manifest(profile="smoke")["protocol_version"], "1.1.0")


if __name__ == "__main__":
    unittest.main()
