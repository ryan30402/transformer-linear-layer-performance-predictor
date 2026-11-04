"""CPU contracts for duration warmup, bounded work, and failure preservation.

Fake clocks and graphs exercise control flow only. They are not CUDA execution,
timing-stability evidence, or numerical GPU validation.
"""

from __future__ import annotations

import copy
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from linear_perf import benchmark as bench
from linear_perf.common import LabError, digest, flops
from tests.test_graph import FakeTensor, FakeTorch, graph_plan


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def duration_plan():
    plan = graph_plan()
    plan["timing"].update(graph_warmup_min_seconds=0.25, graph_warmup_max_replays=4096)
    return plan


def timed_fake(clock, replay_seconds):
    torch = FakeTorch()
    factory = torch.cuda.CUDAGraph

    def create():
        graph = factory()
        replay = graph.replay

        def run():
            replay()
            clock.value += replay_seconds

        graph.replay = run
        return graph

    torch.cuda.CUDAGraph = create
    return torch


def public_fakes(stack, torch, plan):
    env = {"environment_id": "fake-duration-environment", "plan_hash": digest(plan)}
    stack.enter_context(patch.object(bench, "_load_torch", return_value=torch))
    stack.enter_context(patch.object(bench, "_require_cuda", return_value="cuda:0"))
    stack.enter_context(patch.object(bench, "_precision_policy", side_effect=lambda *_: nullcontext({})))
    stack.enter_context(patch.object(bench, "_collect_environment", return_value=env))
    stack.enter_context(patch.object(bench, "_check_memory", return_value={}))
    stack.enter_context(patch.object(bench, "_allocate_inputs", side_effect=lambda *_: (FakeTensor(), FakeTensor(), FakeTensor())))
    stack.enter_context(patch.object(bench, "_correctness", return_value={"passed": True}))
    return env


class DurationWarmupTests(unittest.TestCase):
    def test_fixed_duration_and_minimum_replays_both_apply(self):
        clock = Clock()
        torch = timed_fake(clock, 0.01)
        budget = bench._ConditioningBudget(123)
        metadata = {}
        with patch.object(bench.time, "monotonic", side_effect=clock):
            samples = bench._measure_operation(
                torch, "cuda:0", torch.cuda.operation, duration_plan()["timing"], 100.0,
                conditioning_budget=budget, flops_per_operation=1000,
                warmup_metadata=metadata,
            )
        self.assertEqual(metadata["completed_replays"], 25)
        self.assertGreaterEqual(metadata["actual_wall_seconds"], 0.25)
        self.assertEqual(metadata["gemm_calls"], 25 * 50)
        self.assertEqual(metadata["additional_conditioning_gemm_calls"], 20 * 50)
        self.assertEqual(metadata["additional_conditioning_flops"], 20 * 50 * 1000)
        self.assertEqual(budget.additional_gemm_calls, 1000)
        self.assertFalse(metadata["sample_quality_used_for_stopping"])
        self.assertEqual(samples, [2.0] * 9)
        self.assertEqual(torch.cuda.graphs[0].replays, 25 + 9)

    def test_minimum_count_still_applies_when_one_replay_exceeds_duration(self):
        clock = Clock()
        torch = timed_fake(clock, 0.3)
        metadata = {}
        with patch.object(bench.time, "monotonic", side_effect=clock):
            bench._measure_operation(
                torch, "cuda:0", torch.cuda.operation, duration_plan()["timing"], 100.0,
                conditioning_budget=bench._ConditioningBudget(0), flops_per_operation=100,
                warmup_metadata=metadata,
            )
        self.assertEqual(metadata["completed_replays"], 5)
        self.assertAlmostEqual(metadata["actual_wall_seconds"], 1.5)
        self.assertEqual(metadata["additional_conditioning_gemm_calls"], 0)

    def test_replay_cap_fails_before_timing_and_does_not_fall_back(self):
        clock, plan = Clock(), duration_plan()
        plan["timing"]["graph_warmup_max_replays"] = 6
        torch = timed_fake(clock, 0.01)
        with patch.object(bench.time, "monotonic", side_effect=clock), \
             patch.object(bench, "_event_pair") as events:
            with self.assertRaisesRegex(LabError, "6-replay cap"):
                bench._measure_operation(
                    torch, "cuda:0", torch.cuda.operation, plan["timing"], 100.0,
                    conditioning_budget=bench._ConditioningBudget(0), flops_per_operation=100,
                )
            events.assert_not_called()
        self.assertEqual(torch.cuda.graphs[0].replays, 6)

    def test_deadline_during_warmup_fails_before_timing(self):
        clock = Clock()
        torch = timed_fake(clock, 0.02)
        with patch.object(bench.time, "monotonic", side_effect=clock), \
             patch.object(bench, "_event_pair") as events:
            with self.assertRaisesRegex(LabError, "wall-time budget"):
                bench._measure_operation(
                    torch, "cuda:0", torch.cuda.operation, duration_plan()["timing"], 0.06,
                    conditioning_budget=bench._ConditioningBudget(0), flops_per_operation=100,
                )
            events.assert_not_called()
        self.assertEqual(torch.cuda.graphs[0].replays, 3)

    def test_duration_path_cannot_omit_work_accounting(self):
        torch = FakeTorch()
        with self.assertRaisesRegex(LabError, "explicit conditioning budget"):
            bench._measure_operation(torch, "cuda:0", torch.cuda.operation, duration_plan()["timing"], float("inf"))
        self.assertEqual(torch.cuda.graphs[0].replays, 0)

    def test_extra_call_and_total_flop_guards_charge_before_submission(self):
        budget = bench._ConditioningBudget(0)
        with patch.object(bench, "MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS", 75):
            budget.charge(50, 100)
            with self.assertRaisesRegex(LabError, "GEMM-call"):
                budget.charge(50, 100)
        self.assertEqual(budget.additional_gemm_calls, 50)
        self.assertEqual(budget.additional_flops, 100)
        budget = bench._ConditioningBudget(bench.MAX_SESSION_FLOPS - 100)
        budget.charge(1, 100)
        with self.assertRaisesRegex(LabError, "FLOP limit"):
            budget.charge(1, 1)
        self.assertEqual(budget.additional_gemm_calls, 1)

    def test_budget_is_shared_across_cases_and_prior_rows_survive_failure(self):
        clock, plan = Clock(), duration_plan()
        torch = timed_fake(clock, 0.045)  # Six replays satisfy 250 ms: one extra beyond the five reserved.
        with ExitStack() as stack:
            public_fakes(stack, torch, plan)
            stack.enter_context(patch.object(bench.time, "monotonic", side_effect=clock))
            stack.enter_context(patch.object(bench, "MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS", 75))
            with self.assertRaises(bench.BenchmarkRunError) as captured:
                bench.benchmark_cases(plan, plan["cases"][:2])
        error = captured.exception
        self.assertEqual(error.failed_case_id, "dev")
        self.assertEqual(len(error.completed_rows), 1)
        row = error.completed_rows[0]
        self.assertEqual(row["case_id"], "cal")
        self.assertEqual(row["warmup"]["completed_replays"], 6)
        self.assertEqual(row["conditioning_budget"]["additional_conditioning_gemm_calls"], 50)
        self.assertEqual(torch.cuda.graphs[1].replays, 5)  # The sixth was rejected before submission.
        self.assertIn("GEMM-call", str(error))

    def test_batching_warms_each_variant_before_every_sample_and_shares_budget(self):
        clock, plan = Clock(), duration_plan()
        torch = timed_fake(clock, 0.1)
        with ExitStack() as stack:
            public_fakes(stack, torch, plan)
            stack.enter_context(patch.object(bench.time, "monotonic", side_effect=clock))
            result = bench.benchmark_batching(plan)
        for label, multiplier in (("small_gemms", 64), ("one_large_gemm", 1)):
            blocks = result[label]["warmup"]["blocks"]
            self.assertEqual(len(blocks), 9)
            self.assertEqual([b["repeat_index"] for b in blocks], list(range(9)))
            self.assertTrue(all(b["completed_replays"] == 5 for b in blocks))
            self.assertTrue(all(b["gemm_calls"] == 5 * 50 * multiplier for b in blocks))
            self.assertEqual(blocks[0]["fixed_reserved_replays"], 5)
            self.assertTrue(all(b["fixed_reserved_replays"] == 0 for b in blocks[1:]))
            self.assertEqual(result[label]["samples_ms"], [2.0] * 9)
        self.assertEqual(result["conditioning_budget"]["additional_conditioning_gemm_calls"], 8 * 5 * 50 * 65)
        self.assertEqual([g.replays for g in torch.cuda.graphs], [9 * 6, 9 * 6])
        orders = result["variant_order_per_repeat"]
        self.assertTrue(all(a == list(reversed(b)) for a, b in zip(orders, orders[1:])))
        # For each variant's sample, five warmups precede its two timing events.
        events = [i for i, item in enumerate(torch.cuda.log) if item == ("event", "record")]
        for sample_index in range(18):
            start_index = events[2 + 2 * sample_index]
            previous_end = events[1 + 2 * sample_index]
            between = torch.cuda.log[previous_end + 1:start_index]
            self.assertEqual(sum(x[0] == "replay" for x in between), 5)

    def test_batching_additional_cap_does_not_reset_per_repeat_or_variant(self):
        clock, plan = Clock(), duration_plan()
        torch = timed_fake(clock, 0.1)
        with ExitStack() as stack:
            public_fakes(stack, torch, plan)
            stack.enter_context(patch.object(bench.time, "monotonic", side_effect=clock))
            stack.enter_context(patch.object(bench, "MAX_ADDITIONAL_CONDITIONING_GEMM_CALLS", 32_499))
            with self.assertRaisesRegex(LabError, "GEMM-call"):
                bench.benchmark_batching(plan)
        self.assertLess(sum(g.replays for g in torch.cuda.graphs), 2 * 9 * 6)

    def test_duration_and_collection_are_in_environment_identity(self):
        torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="13.0"))
        plan = duration_plan()
        gpu = {"name": "fixture", "uuid": "fixture"}
        with patch.object(bench, "_driver_versions", return_value=["580"]), \
             patch.object(bench, "_gpu_metadata", return_value=gpu):
            single = bench._collect_environment(torch, plan, "cuda:0", {})
            replicated = copy.deepcopy(plan)
            replicated["collection"] = {"blocks": 3, "order": "seeded_shuffle_per_block", "aggregation": "median_of_block_medians"}
            triple = bench._collect_environment(torch, replicated, "cuda:0", {})
            changed_duration = copy.deepcopy(plan)
            changed_duration["timing"]["graph_warmup_min_seconds"] = 0.5
            changed = bench._collect_environment(torch, changed_duration, "cuda:0", {})
        self.assertNotEqual(single["environment_id"], triple["environment_id"])
        self.assertNotEqual(single["environment_id"], changed["environment_id"])
        self.assertEqual(single["identity"]["collection"]["blocks"], 1)
        self.assertEqual(triple["identity"]["collection"], replicated["collection"])
        self.assertEqual(single["identity"]["runner_version"], "cuda-events-1.0.0")


if __name__ == "__main__":
    unittest.main()
