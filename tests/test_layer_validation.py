"""Downstream layer contracts; real CUDA timing is never simulated here."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from linear_perf.common import LabError
from linear_perf.layer_validation import (benchmark_layer, compose_layer_estimate, estimate_layer_bytes,
                                          layer_gemm_cases, validate_layer_spec)
from linear_perf.workloads import load_model_specs, parse_model_config


def small_spec():
    return parse_model_config({"model_type": "llama", "hidden_size": 16, "intermediate_size": 24,
                               "num_attention_heads": 4, "num_key_value_heads": 2,
                               "num_hidden_layers": 1, "vocab_size": 32, "max_position_embeddings": 64,
                               "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "attention_bias": False}, model_id="cpu_math_fixture")


class LayerContractTests(unittest.TestCase):
    def test_geometry_retains_gqa_and_separate_projection_multiplicity(self):
        source = next(source for source in load_model_specs() if source["id"] == "qwen2_5_0_5b")
        cases = layer_gemm_cases(source["spec"], batch_size=2, sequence_length=12)
        by_name = {case["id"]: case for case in cases}
        self.assertEqual(set(by_name), {"q", "k", "v", "o", "gate", "up", "down"})
        self.assertEqual(by_name["q"]["n"], 896)
        self.assertEqual(by_name["k"]["n"], 128)
        self.assertEqual(by_name["v"]["n"], 128)
        self.assertEqual(by_name["down"]["k"], 4864)
        self.assertTrue(all(case["m"] == 24 for case in cases))
        self.assertFalse(any("median_ms" in case or "split" in case for case in cases))

    def test_context_limit_and_invalid_attention_are_rejected_not_shrunk(self):
        source = next(source for source in load_model_specs() if source["id"] == "tinyllama_1_1b")
        original = copy.deepcopy(source["spec"])
        with self.assertRaisesRegex(LabError, "native context"):
            layer_gemm_cases(source["spec"], sequence_length=2049)
        self.assertEqual(source["spec"], original)
        invalid = small_spec()
        invalid["num_key_value_heads"] = 3
        with self.assertRaisesRegex(LabError, "head ratio"):
            validate_layer_spec(invalid, sequence_length=8)

    def test_memory_guard_counts_reference_and_quadratic_attention(self):
        spec = small_spec()
        one = estimate_layer_bytes(spec, sequence_length=8)
        two = estimate_layer_bytes(spec, sequence_length=16)
        self.assertEqual(two["weight_and_reference_bytes"], one["weight_and_reference_bytes"])
        self.assertEqual(two["activation_allowance_bytes"], 2 * one["activation_allowance_bytes"])
        self.assertEqual(two["attention_allowance_bytes"], 4 * one["attention_allowance_bytes"])
        self.assertGreater(one["estimated_peak_bytes"], one["weight_and_reference_bytes"])

    def record(self):
        spec = small_spec()
        return {"status": "completed", "measurement_kind": "cuda_measured", "model_spec": spec,
                "batch_size": 1, "sequence_length": 8, "dtype": "float16",
                "excluded_from_predictor_training": True, "full_layer": {"median_ms": 1.1},
                "components": [{**case, "kind": "gemm", "median_ms": 0.1} for case in layer_gemm_cases(spec, sequence_length=8)] +
                              [{"id": "fixture_other_operations", "kind": "non_gemm", "median_ms": 0.3}]}

    def test_hybrid_comparison_exposes_measured_inputs_and_additivity_gap(self):
        record = self.record()
        predictions = {case["id"]: 0.12 for case in layer_gemm_cases(record["model_spec"], sequence_length=8)}
        result = compose_layer_estimate(record, predictions)
        self.assertAlmostEqual(result["hybrid_estimate_ms"], 1.14)
        self.assertAlmostEqual(result["absolute_percentage_error"], 100 * 0.04 / 1.1)
        self.assertAlmostEqual(result["measured_composition_gap_pct"], -100 * 0.1 / 1.1)
        self.assertFalse(result["fully_predicted_end_to_end_model"])
        self.assertAlmostEqual(result["non_gemm_fraction_of_measured_component_sum"], 0.3)
        self.assertTrue(result["excluded_from_predictor_training"])
        self.assertNotIn("model_suite", result)

    def test_composition_requires_actual_complete_downstream_data_and_exact_predictions(self):
        record = self.record()
        predictions = {case["id"]: 0.1 for case in layer_gemm_cases(record["model_spec"], sequence_length=8)}
        for bad in ("simulated", "cpu_measured"):
            invalid = copy.deepcopy(record)
            invalid["measurement_kind"] = bad
            with self.assertRaisesRegex(LabError, "actual CUDA"):
                compose_layer_estimate(invalid, predictions)
        invalid = copy.deepcopy(record)
        invalid["components"].pop(0)
        with self.assertRaisesRegex(LabError, "seven GEMMs"):
            compose_layer_estimate(invalid, predictions)
        invalid = {**predictions, "q": float("nan")}
        with self.assertRaisesRegex(LabError, "positive and finite"):
            compose_layer_estimate(record, invalid)
        with self.assertRaisesRegex(LabError, "exactly"):
            compose_layer_estimate(record, {**predictions, "unexpected": 0.5})

    def test_missing_cuda_is_failure_evidence_not_cpu_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "failed_layer"
            with patch("linear_perf.benchmark._load_torch", return_value=object()), \
                    patch("linear_perf.benchmark._require_cuda", side_effect=LabError("CUDA unavailable")), \
                    self.assertRaisesRegex(LabError, "CUDA unavailable"):
                benchmark_layer(small_spec(), sequence_length=8, output_dir=output, progress=None)
            result = json.loads((output / "layer_measurements.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["components"], [])
            self.assertNotIn("full_layer", result)

    def test_invalid_batch_is_rejected_before_loading_torch(self):
        with patch("linear_perf.benchmark._load_torch") as load, self.assertRaises(LabError):
            benchmark_layer(small_spec(), batch_size=True, sequence_length=8)
        load.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Optional CPU mathematical checks require PyTorch")
class LayerCpuMathTests(unittest.TestCase):
    """CPU math checks verify architecture, not CUDA accuracy or timing."""
    def test_gqa_sdpa_matches_explicit_causal_attention(self):
        import torch
        from linear_perf.layer_validation import _SyntheticDecoder
        spec = small_spec()
        with torch.inference_mode():
            layer = _SyntheticDecoder(torch, spec, batch_size=2, sequence_length=5,
                                      dtype=torch.float32, device="cpu", seed=11)
            q = torch.randn(2, 4, 5, 4)
            k, v = torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 4)
            actual = layer.attention((q, k, v))
            repeated_k, repeated_v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
            logits = q @ repeated_k.transpose(-2, -1) / 2
            mask = torch.ones(5, 5, dtype=torch.bool).tril()
            expected = logits.masked_fill(~mask, float("-inf")).softmax(dim=-1) @ repeated_v
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_complete_layer_is_deterministic_and_components_cover_the_graph(self):
        import torch
        from linear_perf.layer_validation import _SyntheticDecoder
        with torch.inference_mode():
            layer = _SyntheticDecoder(torch, small_spec(), batch_size=1, sequence_length=7,
                                      dtype=torch.float32, device="cpu", seed=12)
            output, tape = layer.forward(record=True)
            second = layer.forward()
            torch.testing.assert_close(output, second)
            self.assertEqual(tuple(output.shape), (7, 16))
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertEqual(len(tape), 15)
            self.assertEqual(sum(metadata["kind"] == "gemm" for metadata, _ in tape), 7)
            self.assertEqual({metadata["id"] for metadata, _ in tape if metadata["kind"] == "non_gemm"},
                             {"input_rmsnorm", "qkv_bias_reshape_rope", "causal_attention", "attention_layout",
                              "attention_residual", "post_attention_rmsnorm", "silu_and_multiply", "mlp_residual"})
            for _, operation in tape:
                operation()
            torch.testing.assert_close(output, layer.forward())


if __name__ == "__main__":
    unittest.main()
