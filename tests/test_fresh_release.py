"""Complete-experiment packaging contracts, without CUDA or network access.

The worker is replaced with an explicit fixture. These checks exercise real
archive export/import and verify routing, deadlines and retained stage outputs.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

from tests import test_release_tools as release_contracts


class FreshExperimentReleaseTests(unittest.TestCase):
    def setUp(self):
        release_contracts.ReleaseToolsContracts.setUp(self)

    def tearDown(self):
        release_contracts.ReleaseToolsContracts.tearDown(self)

    def worker_result(self, *, returncode=0, timed_out=False, interrupted=False):
        return {"returncode": returncode, "timed_out": timed_out, "interrupted": interrupted,
                "automatic_retries": 0, "elapsed_seconds": 0.1}

    def write_stage_fixture(self, command, *, incomplete=False):
        out = Path(command[command.index("--out") + 1])
        primary = out / "primary"
        (primary / "dataset").mkdir(parents=True)
        (primary / "analysis").mkdir()
        (primary / "status.json").write_text(json.dumps({"status": "simulated_complete", "fixture_only": True}))
        (primary / "dataset/measurements.jsonl").write_text('{"fixture_only":true,"retained":"primary"}\n')
        (primary / "analysis/report.html").write_text("<p>Fixture only; no GPU observations.</p>")
        if incomplete:
            (out / "repeat/dataset").mkdir(parents=True)
            (out / "repeat/dataset/raw_visits.jsonl").write_text('{"fixture_only":true,"retained":"partial repeat"}\n')
            (out / "repeat/status.json").write_text(json.dumps({"status": "failed", "fixture_only": True}))
        summary = {"status": "incomplete" if incomplete else "simulated_complete",
                   "primary_evaluation_available": True, "fixture_only": True}
        (out / "summary.json").write_text(json.dumps(summary))
        (out / "report.html").write_text("<p>Fixture status, including retained stage results.</p>")
        return out

    def result_payload(self, *, action="experiment", status="worker_completed", primary_status="simulated_complete"):
        raw = release_contracts.ReleaseToolsContracts.result_payload(self, action=action, status=status)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if name != "export_manifest.json"}
        if action == "experiment":
            for name in tuple(files):
                if name.startswith("run/"):
                    files["run/primary/" + name.removeprefix("run/")] = files.pop(name)
            files["run/primary/status.json"] = json.dumps({"status": "simulated_complete", "fixture_only": True}).encode()
            files["run/primary/dataset/status.json"] = json.dumps({"status": primary_status, "fixture_only": True}).encode()
            files["run/summary.json"] = json.dumps({"status": "simulated_complete", "primary_evaluation_available": True,
                                                  "fixture_only": True}).encode()
        files["export_manifest.json"] = json.dumps({
            "source_sha256": self.launcher.SOURCE_SHA256,
            "files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
        }).encode()
        return release_contracts.archive_bytes(files.items())

    def test_complete_wrapper_routes_to_fresh_workload_with_time_for_all_requested_stages(self):
        for name, include_repeat, include_layer in (("full", True, False), ("primary", False, False), ("layer", True, True)):
            def success(command, **kwargs):
                self.write_stage_fixture(command)
                Path(kwargs["log_path"]).write_text("fixture worker completed\n")
                return self.worker_result()

            with self.subTest(name=name), \
                    patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                    patch.object(self.launcher, "worker", side_effect=success) as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.run_experiment(
                    experiment_id=name, simulate=True, include_repeat=include_repeat,
                    include_layer=include_layer, max_wall_seconds=300, base_directory=self.root,
                )
            worker.assert_called_once()
            command = worker.call_args.args[0]
            self.assertEqual(command[:5], [sys.executable, "-u", "-m", "linear_perf", "experiment"])
            self.assertEqual(Path(command[command.index("--config") + 1]).name, "workloads.json")
            first = command.index("--dtypes") + 1
            self.assertEqual(command[first:first + 2], ["float16", "bfloat16"])
            self.assertIn("--simulate", command)
            self.assertNotIn("--repeat-subset", command)
            self.assertEqual("--skip-repeat" in command, not include_repeat)
            self.assertEqual("--include-layer" in command, include_layer)
            self.assertEqual(worker.call_args.kwargs["timeout"], 300 * (2 if include_repeat else 1) + 900 + (900 if include_layer else 0))
            self.assertEqual(result["status"], "worker_completed")
            self.assertEqual(result["experiment_status"], "simulated_complete")
            self.assertTrue(result["primary_evaluation_available"])
            self.assertEqual(Path(result["report"]), Path(result["workspace"]) / "run/report.html")
            with zipfile.ZipFile(result["archive"]) as archive:
                status = json.loads(archive.read("launcher_status.json"))
            self.assertFalse(status["historical_results_used"])
            self.assertEqual(status["workflow"], "fresh_primary_repeat_comparison")
            self.assertEqual(status["include_repeat"], include_repeat)

    def test_repeat_failure_retains_usable_primary_and_partial_repeat_in_one_archive(self):
        for name, raises in (("failed_repeat", False), ("supervision_error", True)):
            def failed(command, **kwargs):
                self.write_stage_fixture(command, incomplete=True)
                Path(kwargs["log_path"]).write_text("primary complete; repeat fixture failed\n")
                if raises:
                    raise RuntimeError("fixture supervisor failure")
                return self.worker_result(returncode=2)

            with self.subTest(name=name), \
                    patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                    patch.object(self.launcher, "worker", side_effect=failed) as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.run_experiment(experiment_id=name, simulate=True, base_directory=self.root)
            worker.assert_called_once()
            self.assertNotEqual(result["status"], "worker_completed")
            self.assertEqual(result["experiment_status"], "incomplete")
            self.assertTrue(result["primary_evaluation_available"])
            self.assertTrue(Path(result["archive"] + ".sha256").is_file())
            with zipfile.ZipFile(result["archive"]) as archive:
                names = set(archive.namelist())
                self.assertIn("run/primary/dataset/measurements.jsonl", names)
                self.assertIn("run/primary/analysis/report.html", names)
                self.assertIn("run/repeat/dataset/raw_visits.jsonl", names)
                self.assertIn("run/repeat/status.json", names)
                self.assertIn("run/report.html", names)
                status = json.loads(archive.read("launcher_status.json"))
                self.assertEqual(status["automatic_retries"], 0)
                inventory = json.loads(archive.read("export_manifest.json"))
                self.assertEqual(set(inventory["files"]), names - {"export_manifest.json"})
                for path, expected in inventory["files"].items():
                    self.assertEqual(hashlib.sha256(archive.read(path)).hexdigest(), expected)

    def test_full_experiment_rejects_invalid_stage_options_before_creating_a_workspace(self):
        invalid = ({"include_repeat": 1}, {"include_layer": "yes"}, {"session_id": "a" * 49}, {"repeat_subset": True})
        for overrides in invalid:
            options = {"action": "experiment", "session_id": "invalid", "base_directory": self.root, **overrides}
            with self.subTest(overrides=overrides), \
                    patch.object(self.launcher, "prepare_source") as prepare, self.assertRaises(ValueError):
                self.launcher.run(**options)
            prepare.assert_not_called()

    def test_launcher_cli_defaults_to_complete_experiment(self):
        with patch.object(self.launcher, "run", return_value={"status": "worker_completed"}) as run:
            code = self.launcher.main(["--session-id", "fresh", "--simulate"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.kwargs["action"], "experiment")
        self.assertTrue(run.call_args.kwargs["include_repeat"])
        self.assertFalse(run.call_args.kwargs["include_layer"])

    def test_review_analyzes_primary_inside_complete_experiment_archive(self):
        uploads = {"experiment.zip": self.result_payload()}
        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", return_value=self.worker_result()) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.review(uploads, review_mode="analyze", review_id="primary_analysis", base_directory=self.root)
        self.assertEqual(result["status"], "worker_completed")
        worker.assert_called_once()
        command = worker.call_args.args[0]
        workspace = Path(result["workspace"])
        self.assertEqual(command[:5], [sys.executable, "-u", "-m", "linear_perf", "analyze"])
        self.assertEqual(Path(command[command.index("--dataset") + 1]), workspace / "inputs/input_01/run/primary")
        self.assertEqual(worker.call_args.kwargs["source"], workspace / "source" / release_contracts.ARCHIVE_ROOT.rstrip("/"))
        with zipfile.ZipFile(result["archive"]) as archive:
            status = json.loads(archive.read("launcher_status.json"))
        self.assertTrue(status["inputs"][0]["full_experiment_primary_selected"])
        self.assertFalse(status["gpu_executed_by_review"])

    def test_review_recovers_completed_primary_from_failed_full_experiment(self):
        for primary_status in ("completed", "simulated_complete"):
            uploads = {"incomplete.zip": self.result_payload(status="worker_failed_evidence_retained",
                                                            primary_status=primary_status)}
            with self.subTest(primary_status=primary_status), \
                    patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                    patch.object(self.launcher, "worker", return_value=self.worker_result()) as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.review(uploads, review_mode="analyze", review_id="recover_" + primary_status,
                                              base_directory=self.root)
            self.assertEqual(result["status"], "worker_completed")
            worker.assert_called_once()
            command = worker.call_args.args[0]
            self.assertEqual(Path(command[command.index("--dataset") + 1]),
                             Path(result["workspace"]) / "inputs/input_01/run/primary")
            with zipfile.ZipFile(result["archive"]) as archive:
                status = json.loads(archive.read("launcher_status.json"))
            entry = status["inputs"][0]
            self.assertEqual(entry["original_launcher_status"], "worker_failed_evidence_retained")
            self.assertTrue(entry["primary_recovered_from_incomplete_experiment"])
            self.assertFalse(status["gpu_executed_by_review"])

    def test_review_rejects_failed_full_experiment_with_incomplete_primary(self):
        for primary_status in ("failed", "collecting", None):
            uploads = {"incomplete.zip": self.result_payload(status="worker_failed_evidence_retained",
                                                            primary_status=primary_status)}
            with self.subTest(primary_status=primary_status), \
                    patch.object(self.launcher, "ensure_dependencies") as dependencies, \
                    patch.object(self.launcher, "worker") as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.review(uploads, review_mode="analyze", review_id="reject_" + str(primary_status),
                                              base_directory=self.root)
            self.assertEqual(result["status"], "review_failed_evidence_retained")
            dependencies.assert_not_called()
            worker.assert_not_called()
            with zipfile.ZipFile(result["archive"]) as archive:
                status = json.loads(archive.read("launcher_status.json"))
            self.assertIn("Only completed", status["error"])

    def test_built_notebook_defaults_run_fresh_primary_and_repeat_without_historical_payload(self):
        # Build against a tiny source fixture, but use the real builder and launcher.
        # Executing settings/bootstrap checks the cells as a Colab user encounters
        # them; only the costly experiment call is replaced with a mock.
        builder_path = release_contracts.TEMPLATE.parent / "build_release.py"
        spec = importlib.util.spec_from_file_location("fresh_notebook_builder_fixture", builder_path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        fixture_root = self.root / "project"
        fixture_root.mkdir()
        with zipfile.ZipFile(io.BytesIO(self.raw)) as archive:
            for name in archive.namelist():
                destination = fixture_root / name.removeprefix(release_contracts.ARCHIVE_ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
        (fixture_root / "release_tools").mkdir()
        for name in ("colab_launcher_template.py", "colab_recovery.py"):
            (fixture_root / "release_tools" / name).write_bytes((release_contracts.TEMPLATE.parent / name).read_bytes())
        (fixture_root / "results").mkdir()
        (fixture_root / "results/previous_measurements.jsonl").write_text('{"fixture_only":true,"historical":true}\n')
        (fixture_root / "analysis").mkdir()
        (fixture_root / "analysis/previous_predictor.json").write_text('{"fixture_only":true,"historical":true}\n')
        built = builder.build(fixture_root, self.root / "built")
        with zipfile.ZipFile(built["source_archive"]) as archive:
            self.assertFalse(any(name.startswith(release_contracts.ARCHIVE_ROOT + folder)
                                 for name in archive.namelist() for folder in ("results/", "analysis/")))
        notebook = json.loads((self.root / "built/Run_Experiment.ipynb").read_text())
        alias = json.loads((self.root / "built/Run_Benchmark.ipynb").read_text())
        self.assertEqual(notebook["cells"], alias["cells"])
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        settings_cell = next(source for source in code_cells if 'PROFILE = "standard"' in source)
        bootstrap_cell = next(source for source in code_cells if "_LAUNCHER_B64 =" in source)
        experiment_cell = next(source for source in code_cells if "result = launcher.run_experiment(" in source)
        storage_cell = next(source for source in code_cells if "drive.mount(" in source)
        self.assertLess(code_cells.index(storage_cell), code_cells.index(experiment_cell))
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(settings_cell, "settings_cell", "exec"), namespace)
            exec(compile(bootstrap_cell, "bootstrap_cell", "exec"), namespace)
        self.assertEqual(namespace["PROFILE"], "standard")
        self.assertEqual(namespace["DTYPES"], ("float16", "bfloat16"))
        self.assertFalse(namespace["SIMULATE"])
        self.assertTrue(namespace["INCLUDE_REPEAT"])
        self.assertFalse(namespace["INCLUDE_LAYER"])
        self.assertFalse(namespace["SKIP_PROFILES"])
        namespace["BACKUP_DIRECTORY"] = self.root / "saved_experiments"
        with patch.object(namespace["launcher"], "run_experiment", return_value={
            "status": "worker_completed", "experiment_status": "completed", "archive": "fixture.zip",
        }) as experiment, contextlib.redirect_stdout(io.StringIO()):
            exec(compile(experiment_cell, "experiment_cell", "exec"), namespace)
        experiment.assert_called_once()
        self.assertEqual(experiment.call_args.kwargs["profile"], "standard")
        self.assertEqual(experiment.call_args.kwargs["dtypes"], ("float16", "bfloat16"))
        self.assertFalse(experiment.call_args.kwargs["simulate"])
        self.assertTrue(experiment.call_args.kwargs["include_repeat"])
        self.assertFalse(experiment.call_args.kwargs["include_layer"])
        self.assertEqual(Path(experiment.call_args.kwargs["backup_directory"]), namespace["BACKUP_DIRECTORY"])
        recovery = json.loads((self.root / "built/Download_Saved_Results.ipynb").read_text())
        self.assertNotEqual(recovery["metadata"].get("accelerator"), "GPU")
        for item in (notebook, alias, recovery):
            for index, cell in enumerate(item["cells"]):
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile("".join(cell["source"]), f"clean_notebook_cell_{index}", "exec")

    def test_review_comparison_accepts_full_source_and_separately_collected_target(self):
        uploads = {"full.zip": self.result_payload(), "target.zip": self.result_payload(action="collect")}
        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", return_value=self.worker_result()) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.review(uploads, review_mode="gpu", review_id="mixed_sources",
                                          result_order=["full.zip", "target.zip"], base_directory=self.root)
        self.assertEqual(result["status"], "worker_completed")
        command = worker.call_args.args[0]
        workspace = Path(result["workspace"])
        self.assertEqual(Path(command[command.index("--source") + 1]), workspace / "inputs/input_01/run/primary")
        self.assertEqual(Path(command[command.index("--target") + 1]), workspace / "inputs/input_02/run")
        self.assertEqual(command[command.index("--mode") + 1], "gpu")


if __name__ == "__main__":
    unittest.main()
