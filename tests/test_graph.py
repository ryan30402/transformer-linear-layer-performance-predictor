"""CPU contract checks for graph scope, ordering, work limits, and lineage.

These fakes verify Python control flow; they do not validate CUDA graph support,
GPU arithmetic, GPU timings, or performance. The next Colab run does that work.
"""

from __future__ import annotations

import copy
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

from linear_perf import benchmark as bench
from linear_perf.common import EAGER_TIMER_SCOPE, GRAPH_TIMER_SCOPE, LabError, digest, validate_measurements, validate_plan
from linear_perf.simulation import simulate_cases
from tests.test_benchmark import tiny_plan


def graph_plan() -> dict:
    plan = tiny_plan()
    plan["dtype"] = "float16"
    plan["timing"] = {"mode": "cuda_graph", "warmup": 20, "iterations": 50, "repeats": 9, "graph_warmup_replays": 5}
    return plan


class FakeStream:
    def __init__(self, cuda, name):
        self.cuda, self.name = cuda, name

    def wait_stream(self, other):
        self.cuda.log.append(("wait", self.name, other.name))


class FakeGraph:
    def __init__(self, cuda):
        self.cuda, self.nodes, self.replays = cuda, 0, 0

    def replay(self):
        assert self.cuda.capturing is None
        assert self.nodes > 0
        self.replays += 1
        self.cuda.output_state = "computed"
        self.cuda.log.append(("replay", self.nodes))


class FakeEvent:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self):
        assert self.cuda.capturing is None, "Timing events must not enter the graph"
        self.cuda.log.append(("event", "record"))

    def synchronize(self):
        assert self.cuda.capturing is None
        self.cuda.log.append(("event", "synchronize"))

    def elapsed_time(self, other):
        return 100.0


class FakeCuda:
    def __init__(self):
        self.log, self.graphs = [], []
        self.capturing = None
        self.default = FakeStream(self, "default")
        self.active = self.default
        self.output_state = "initial"
        self.eager_calls = 0

    def current_stream(self, device):
        return self.active

    def Stream(self, device):
        return FakeStream(self, "side")

    def device(self, selected):
        return nullcontext()

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous

    def CUDAGraph(self):
        graph = FakeGraph(self)
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph, stream):
        assert stream.name == "side"
        assert self.capturing is None
        self.log.append(("capture", "begin"))
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None
            self.log.append(("capture", "end"))

    def synchronize(self, selected):
        assert self.capturing is None, "Synchronizing inside capture is forbidden"
        self.log.append(("device", "synchronize"))

    def Event(self, enable_timing):
        assert enable_timing
        assert self.capturing is None
        return FakeEvent(self)

    def operation(self):
        if self.capturing is not None:
            self.capturing.nodes += 1
        else:
            self.eager_calls += 1
            self.log.append(("eager", self.active.name))
            self.output_state = "computed"

    def poison(self):
        assert self.capturing is None
        self.output_state = "poisoned"
        self.log.append(("poison",))


class FakeTensor:
    def __getitem__(self, item):
        return FakeTensor()

    def float(self):
        return self

    def fill_(self, value):
        return self


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()

    def inference_mode(self):
        return nullcontext()

    def autocast(self, **kwargs):
        return nullcontext()

    def empty_like(self, other):
        return FakeTensor()

    def mm(self, x, w, *, out=None):
        if out is not None:
            self.cuda.operation()
            return out
        return FakeTensor()


class GraphProtocolTests(unittest.TestCase):
    def test_eager_and_explicit_graph_plan_are_valid(self):
        self.assertIs(validate_plan(tiny_plan())["timing"].get("mode"), None)
        self.assertEqual(validate_plan(graph_plan())["timing"]["mode"], "cuda_graph")

    def test_invalid_or_ambiguous_timing_options_fail_before_gpu_access(self):
        invalid = [
            {"mode": "magic"}, {"mode": []}, {"graph_warmup_replays": 0},
            {"graph_warmup_replays": 21}, {"graph_warmup_replays": True},
            {"unregistered_option": 1},
        ]
        for update in invalid:
            plan = graph_plan()
            plan["timing"].update(update)
            with self.subTest(update=update), patch.object(bench, "_load_torch") as loader:
                with self.assertRaises(LabError):
                    bench.benchmark_cases(plan, [plan["cases"][0]])
                loader.assert_not_called()
        plan = tiny_plan()
        plan["timing"]["graph_warmup_replays"] = 5
        with self.assertRaisesRegex(LabError, "graph-only"):
            validate_plan(plan)

    def test_graph_requires_registered_replay_warmup(self):
        plan = graph_plan()
        del plan["timing"]["graph_warmup_replays"]
        with self.assertRaisesRegex(LabError, "graph_warmup_replays"):
            validate_plan(plan)

    def test_capture_replay_denominator_and_event_boundaries(self):
        torch = FakeTorch()
        samples = bench._measure_operation(
            torch, "cuda:0", torch.cuda.operation, graph_plan()["timing"], float("inf"),
            before_graph_replay=torch.cuda.poison,
        )
        self.assertEqual(samples, [2.0] * 9)  # 100 fake ms per replay / 50 captured operations.
        self.assertEqual(torch.cuda.eager_calls, 20)
        self.assertEqual(len(torch.cuda.graphs), 1)
        self.assertEqual(torch.cuda.graphs[0].nodes, 50)
        self.assertEqual(torch.cuda.graphs[0].replays, 14)  # Five warmups, nine timed replays.
        self.assertEqual(torch.cuda.output_state, "computed")
        self.assertTrue(all(entry[1] == "side" for entry in torch.cuda.log if entry[0] == "eager"))
        self.assertIn(("wait", "side", "default"), torch.cuda.log)
        self.assertEqual(torch.cuda.log.count(("wait", "default", "side")), 2)
        end = torch.cuda.log.index(("capture", "end"))
        poison = torch.cuda.log.index(("poison",))
        replay = next(i for i, item in enumerate(torch.cuda.log) if item[0] == "replay")
        self.assertLess(end, poison)
        self.assertLess(poison, replay)
        self.assertEqual(torch.cuda.log.count(("event", "record")), 2 + 2 * 9)

    def test_capture_failure_never_falls_back_to_eager_measurements(self):
        torch = FakeTorch()
        with patch.object(torch.cuda, "graph", side_effect=RuntimeError("capture unsupported")):
            with self.assertRaisesRegex(RuntimeError, "capture unsupported"):
                bench._measure_operation(torch, "cuda:0", torch.cuda.operation, graph_plan()["timing"], float("inf"))
        self.assertEqual(torch.cuda.eager_calls, 20)
        self.assertFalse(any(item[0] == "event" for item in torch.cuda.log))

    def test_work_budget_includes_capture_and_replay_warmups(self):
        plan = graph_plan()
        self.assertEqual(bench._bounded_operation_count(plan["timing"]), 772)
        with patch.object(bench, "MAX_SESSION_GEMM_CALLS", 600):
            with self.assertRaisesRegex(LabError, "per-session safety bound"):
                bench._validate_work_budget(plan, [plan["cases"][0]])

    def test_large_batching_graph_is_rejected_before_cuda_import(self):
        plan = graph_plan()
        plan["timing"]["iterations"] = 129
        with patch.object(bench, "_load_torch") as loader:
            with self.assertRaisesRegex(LabError, "captured GEMM-node"):
                bench.benchmark_batching(plan)
            loader.assert_not_called()

    def test_batching_graph_denominator_is_whole_workloads_for_both_variants(self):
        plan, torch = graph_plan(), FakeTorch()
        env = {"environment_id": "fake", "plan_hash": digest(plan)}
        with patch.object(bench, "_load_torch", return_value=torch), \
             patch.object(bench, "_require_cuda", return_value="cuda:0"), \
             patch.object(bench, "_precision_policy", return_value=nullcontext({})), \
             patch.object(bench, "_collect_environment", return_value=env), \
             patch.object(bench, "_check_memory", return_value={}), \
             patch.object(bench, "_allocate_inputs", return_value=(FakeTensor(), FakeTensor(), FakeTensor())), \
             patch.object(bench, "_correctness", return_value={"passed": True}):
            result = bench.benchmark_batching(plan)
        self.assertEqual([graph.nodes for graph in torch.cuda.graphs], [3200, 50])
        self.assertEqual([graph.replays for graph in torch.cuda.graphs], [14, 14])
        self.assertEqual(result["small_gemms"]["median_ms"], 2.0)
        self.assertEqual(result["one_large_gemm"]["median_ms"], 2.0)
        self.assertEqual(result["captured_workloads_per_graph"], 50)
        self.assertEqual(result["timer_scope"], GRAPH_TIMER_SCOPE)
        orders = result["variant_order_per_repeat"]
        self.assertTrue(all(left == list(reversed(right)) for left, right in zip(orders, orders[1:])))

    def test_graph_identity_and_scope_are_distinct_from_eager(self):
        plan = graph_plan()
        graph = bench._execution_metadata(plan["timing"])
        eager = bench._execution_metadata(tiny_plan()["timing"])
        self.assertEqual(graph["timer_scope"], GRAPH_TIMER_SCOPE)
        self.assertEqual(graph["captured_operations_per_graph"], 50)
        self.assertEqual(eager["timer_scope"], EAGER_TIMER_SCOPE)
        self.assertNotEqual(digest(graph), digest(eager))

    def test_real_rows_cannot_silently_change_mode_or_timing_scope(self):
        plan = graph_plan()
        row = simulate_cases(plan, [plan["cases"][0]])[0]
        row.update(measurement_kind="cuda_measured", execution_mode="cuda_graph",
                   timing=copy.deepcopy(plan["timing"]), timer_scope=GRAPH_TIMER_SCOPE)
        validate_measurements([row], plan)
        for changed in ({"execution_mode": "eager"}, {"timer_scope": EAGER_TIMER_SCOPE}, {"timing": {}}):
            with self.subTest(changed=changed), self.assertRaises(LabError):
                validate_measurements([{**row, **changed}], plan)

    def test_graph_plan_cannot_create_a_mislabeled_eager_profile(self):
        plan = graph_plan()
        with patch.object(bench, "_load_torch") as loader:
            with self.assertRaisesRegex(LabError, "dedicated graph profiler"):
                bench.profile_case(plan, plan["cases"][0], "/tmp/unused-graph-profile")
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
