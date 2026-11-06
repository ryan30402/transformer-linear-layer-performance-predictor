"""Offline release-wrapper contracts using a tiny harmless embedded archive.

These tests execute no CUDA work, network requests or package installations.
The timeout test runs one short-lived Python subprocess and checks supervision.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile


TEMPLATE = Path(__file__).resolve().parents[1] / "release_tools" / "colab_launcher_template.py"
ARCHIVE_ROOT = "transformer-linear-layer-performance-predictor-1.1.0/"


def archive_bytes(entries):
    handle = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Intentionally duplicated adversarial entries.
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries:
                archive.writestr(name, value)
    return handle.getvalue()


def fixture_payload(*, additional_source=""):
    plan = json.dumps({"protocol_version": "1.1.0", "cases": [
        {"fixture_only": True, "split": split} for split in ("train", "test", "challenge", "stress")
    ]})
    return archive_bytes([
        (ARCHIVE_ROOT + "linear_perf/cli.py", '"""Harmless syntax fixture; never executed."""\n' + additional_source),
        (ARCHIVE_ROOT + "configs/workloads.json", plan),
        (ARCHIVE_ROOT + "configs/workloads_smoke.json", plan),
        (ARCHIVE_ROOT + "requirements-analysis-lock.txt",
         "numpy==2.3.5\nscipy==1.17.0\nmatplotlib==3.10.8\n"),
    ])


class ReleaseToolsContracts(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("launcher_contract_fixture", TEMPLATE)
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)
        self.raw = fixture_payload()
        self.launcher.SOURCE_B64 = base64.b64encode(self.raw).decode("ascii")
        self.launcher.SOURCE_SHA256 = hashlib.sha256(self.raw).hexdigest()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_embedded_payload_hash_and_self_check_need_no_execution_or_network(self):
        with patch.object(self.launcher.subprocess, "run",
                          side_effect=AssertionError("self-check must not launch tools")), \
                patch.object(self.launcher.subprocess, "Popen",
                             side_effect=AssertionError("self-check must not start a worker")):
            self.assertEqual(self.launcher.payload(), self.raw)
            result = self.launcher.self_check()
        self.assertEqual(result["status"], "self_check_passed")
        self.assertFalse(result["network_used"])
        self.assertFalse(result["cuda_executed"])
        self.assertEqual(result["python_files_compiled"], 1)
        self.launcher.SOURCE_SHA256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.launcher.payload()

    def test_safe_extract_retains_exact_content_and_refuses_existing_destination(self):
        destination = self.root / "source"
        result = self.launcher.safe_extract(self.raw, destination)
        self.assertEqual(result, destination.resolve())
        source = destination / ARCHIVE_ROOT / "linear_perf" / "cli.py"
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.launcher.safe_extract(self.raw, destination)
        self.assertEqual(source.read_bytes(), original)

    def test_traversal_absolute_windows_and_dot_segments_rejected_before_writes(self):
        for index, name in enumerate((
            "../escape.txt", "safe/../../escape.txt", "/absolute.txt",
            "C:/drive.txt", r"safe\..\escape.txt", "safe/./file.txt",
        )):
            destination = self.root / f"unsafe-{index}"
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.launcher.safe_extract(archive_bytes([(name, "blocked")]), destination)
            self.assertFalse(destination.exists())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_symbolic_links_and_special_files_rejected(self):
        for mode in (stat.S_IFLNK | 0o777, stat.S_IFCHR | 0o600):
            entry = zipfile.ZipInfo("unsafe")
            entry.create_system = 3
            entry.external_attr = mode << 16
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Unsafe"):
                self.launcher.checked_members(archive_bytes([(entry, "../outside")]))

    def test_duplicate_archive_names_rejected(self):
        raw = archive_bytes([("same.txt", "first"), ("same.txt", "second")])
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.launcher.checked_members(raw)

    def test_alias_paths_cannot_bypass_duplicate_preflight(self):
        raw = archive_bytes([("folder//same.txt", "first"), ("folder/same.txt", "second")])
        with self.assertRaises(ValueError):
            self.launcher.checked_members(raw)

    def test_expanded_size_budget_rejects_small_high_compression_archive(self):
        raw = archive_bytes([("large.txt", b"A" * 4096)])
        self.assertLess(len(raw), 1024)
        destination = self.root / "oversized"
        with self.assertRaisesRegex(ValueError, "expanded-size"):
            self.launcher.safe_extract(raw, destination, limit=1024)
        self.assertFalse(destination.exists())

    def test_prepare_source_copies_original_zip_and_digest_without_reusing_workspace(self):
        workspace = self.root / "workspace"
        source = self.launcher.prepare_source(workspace)
        self.assertEqual(source, workspace / "source" / ARCHIVE_ROOT.rstrip("/"))
        self.assertEqual((workspace / "source_archive.zip").read_bytes(), self.raw)
        self.assertEqual((workspace / "source_archive.sha256").read_text().strip(),
                         hashlib.sha256(self.raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.launcher.prepare_source(workspace)

    def make_failed_workspace(self, name="failed"):
        workspace = self.root / name
        self.launcher.prepare_source(workspace)
        run = workspace / "run"
        run.mkdir()
        (run / "status.json").write_text('{"status":"failed","fixture_only":true}')
        (run / "partial_raw.json").write_text('{"retained":[1,2],"fixture_only":true}')
        (workspace / "worker.log").write_text("partial worker evidence\n")
        (workspace / "launcher_status.json").write_text('{"status":"worker_failed_evidence_retained"}')
        (workspace / "run-postseal-failure.json").write_text('{"error":"fixture audit failure"}')
        cache = run / "__pycache__"
        cache.mkdir()
        (cache / "ignored.pyc").write_bytes(b"not evidence")
        return workspace

    def test_export_preserves_failures_and_checks_every_archived_file_and_whole_zip(self):
        workspace = self.make_failed_workspace()
        destination = self.root / "RESULTS.zip"
        result = self.launcher.export_results(workspace, destination)
        self.assertEqual(Path(result), destination)
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            self.assertIn("run/partial_raw.json", names)
            self.assertIn("run-postseal-failure.json", names)
            self.assertIn("launcher_status.json", names)
            self.assertIn("worker.log", names)
            self.assertIn("source_archive.zip", names)
            self.assertFalse(any("__pycache__" in name for name in names))
            inventory = json.loads(archive.read("export_manifest.json"))
            self.assertEqual(inventory["source_sha256"], self.launcher.SOURCE_SHA256)
            self.assertEqual(set(inventory["files"]), names - {"export_manifest.json"})
            for name, expected in inventory["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
        sidecar = Path(str(destination) + ".sha256")
        self.assertTrue(sidecar.is_file(), "A whole-result-ZIP hash must be downloadable beside the ZIP")
        expected_zip_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        self.assertEqual(sidecar.read_text().split()[0], expected_zip_hash)
        original = destination.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.launcher.export_results(workspace, destination)
        self.assertEqual(destination.read_bytes(), original)

    @unittest.skipUnless(hasattr(os, "symlink"), "Operating system has no symbolic links")
    def test_export_rejects_linked_evidence_instead_of_reading_an_outside_file(self):
        workspace = self.make_failed_workspace()
        outside = self.root / "outside_fixture.txt"
        outside.write_text("should not be exported")
        (workspace / "run" / "linked.txt").symlink_to(outside)
        destination = self.root / "blocked.zip"
        with self.assertRaisesRegex(ValueError, "linked/outside"):
            self.launcher.export_results(workspace, destination)
        self.assertFalse(destination.exists())

    def test_dependency_installer_targets_analysis_requirements_and_never_torch(self):
        source = self.launcher.prepare_source(self.root / "deps")
        old = {name: "unavailable" for name in self.launcher.ANALYSIS_LOCK}
        expected = dict(self.launcher.ANALYSIS_LOCK)
        with patch.object(self.launcher, "dependency_versions", side_effect=[old, expected]), \
                patch.object(self.launcher.subprocess, "run") as install, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.ensure_dependencies(source)
        install.assert_called_once()
        command = install.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertEqual(Path(command[-1]), source / "requirements-analysis-lock.txt")
        packages = {line.split("==")[0] for line in Path(command[-1]).read_text().splitlines()}
        self.assertEqual(packages, {"numpy", "scipy", "matplotlib"})
        self.assertFalse(result["pytorch_reinstalled"])
        self.assertFalse(result["cuda_reinstalled"])
        self.assertEqual(install.call_args.kwargs["timeout"], 600)

    def test_existing_log_is_rejected_before_a_worker_can_start(self):
        log = self.root / "existing.log"
        log.write_text("retained prior log")
        with patch.object(self.launcher.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                self.launcher.worker(["unused"], source=self.root, log_path=log, timeout=1)
        popen.assert_not_called()
        self.assertEqual(log.read_text(), "retained prior log")

    def test_worker_timeout_stops_one_process_preserves_log_and_does_not_retry(self):
        log = self.root / "timeout.log"
        command = [
            sys.executable, "-u", "-c",
            "import time; print('ONE_ATTEMPT', flush=True); time.sleep(30)",
        ]
        original_popen = subprocess.Popen
        with patch.object(self.launcher.subprocess, "Popen", wraps=original_popen) as popen, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.worker(command, source=self.root, log_path=log, timeout=0.5)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["interrupted"])
        self.assertEqual(result["automatic_retries"], 0)
        self.assertLess(result["elapsed_seconds"], 10)
        popen.assert_called_once()
        self.assertIsNotNone(result["returncode"])
        self.assertEqual(log.read_text().count("ONE_ATTEMPT"), 1)

    def test_launcher_exports_the_failure_envelope_when_worker_raises(self):
        def failing(command, **kwargs):
            out = Path(command[command.index("--out") + 1])
            out.mkdir()
            (out / "partial_raw.json").write_text('{"fixture_only":true,"retained":1}')
            Path(kwargs["log_path"]).write_text("retained before failure\n")
            raise RuntimeError("fixture worker supervision failure")

        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", side_effect=failing) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.run(session_id="failure", simulate=True, base_directory=self.root)
        worker.assert_called_once()
        self.assertEqual(result["status"], "launcher_failed_evidence_retained")
        with zipfile.ZipFile(result["archive"]) as archive:
            state = json.loads(archive.read("launcher_status.json"))
            self.assertEqual(state["error_type"], "RuntimeError")
            self.assertEqual(state["automatic_retries"], 0)
            self.assertIn("run/partial_raw.json", archive.namelist())
            self.assertIn("worker.log", archive.namelist())

    def test_timeout_or_interrupt_with_zero_exit_is_still_a_failed_run(self):
        for name, interrupted, timed_out in (
            ("timeout", False, True), ("interrupt", True, False),
        ):
            worker_result = {
                "returncode": 0, "timed_out": timed_out, "interrupted": interrupted,
                "automatic_retries": 0, "elapsed_seconds": 0.1,
            }
            with self.subTest(name=name), \
                    patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                    patch.object(self.launcher, "worker", return_value=worker_result) as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.run(session_id=name, simulate=True, base_directory=self.root)
            worker.assert_called_once()
            self.assertNotEqual(result["status"], "worker_completed")
            self.assertTrue(Path(result["archive"]).is_file())
            with zipfile.ZipFile(result["archive"]) as archive:
                state = json.loads(archive.read("launcher_status.json"))
            self.assertEqual(state["worker"], worker_result)

    def test_run_forwards_both_precisions_and_selected_workload_without_cuda(self):
        success = {"returncode": 0, "timed_out": False, "interrupted": False,
                   "automatic_retries": 0, "elapsed_seconds": 0.1}
        cases = (
            ("standard", {}, "run", "workloads.json", ("float16", "bfloat16"), 7200),
            ("smoke", {"profile": "smoke", "action": "collect", "dtypes": ("bfloat16",),
                       "repeat_subset": True, "skip_profiles": True, "l2_cache_mib": 40.5,
                       "max_wall_seconds": 300}, "collect", "workloads_smoke.json", ("bfloat16",), 300),
        )
        for name, options, action, config, dtypes, budget in cases:
            with self.subTest(name=name), \
                    patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                    patch.object(self.launcher, "worker", return_value=success) as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.run(session_id=name, simulate=True, base_directory=self.root, **options)
            worker.assert_called_once()
            command = worker.call_args.args[0]
            self.assertEqual(command[:5], [sys.executable, "-u", "-m", "linear_perf", action])
            self.assertEqual(Path(command[command.index("--config") + 1]).name, config)
            start = command.index("--dtypes") + 1
            self.assertEqual(tuple(command[start:start + len(dtypes)]), dtypes)
            self.assertIn("--simulate", command)
            self.assertEqual(command[command.index("--max-wall-seconds") + 1], str(budget))
            self.assertEqual(worker.call_args.kwargs["timeout"], budget + 900)
            self.assertEqual(result["status"], "worker_completed")
            if name == "smoke":
                self.assertIn("--repeat-subset", command)
                self.assertIn("--skip-profiles", command)
                self.assertEqual(command[command.index("--l2-cache-mib") + 1], "40.5")

    def result_payload(self, *, wrong_source_hash=False, changed_source=False,
                       corrupt_member=False, extra_unlisted=False, source_raw=None,
                       wrong_launcher_hash=False, action="run", status="worker_completed"):
        recorded_source = self.raw if source_raw is None else source_raw
        recorded_hash = hashlib.sha256(recorded_source).hexdigest()
        source = recorded_source + b"changed" if changed_source else recorded_source
        files = {
            "source_archive.zip": source,
            "source_archive.sha256": (recorded_hash + "\n").encode(),
            "launcher_status.json": json.dumps({"status": status, "action": action, "fixture_only": True,
                                                "source_sha256": "0" * 64 if wrong_launcher_hash else recorded_hash}).encode(),
            "run/status.json": b'{"fixture_only":true}',
            "run/manifest.json": b'{"fixture_only":true}',
        }
        inventory = {
            "source_sha256": "0" * 64 if wrong_source_hash else recorded_hash,
            "files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
        }
        if corrupt_member:
            files["run/status.json"] = b'{"fixture_only":"tampered"}'
        if extra_unlisted:
            files["not_in_manifest.txt"] = b"unlisted"
        files["export_manifest.json"] = json.dumps(inventory).encode()
        return archive_bytes(files.items())

    def test_review_checks_inventory_and_each_inputs_own_source_before_worker(self):
        faults = (
            ("wrong-source-hash", {"wrong_source_hash": True}, "source archive checksum differs"),
            ("changed-source", {"changed_source": True}, "source archive checksum differs"),
            ("launcher-source-mismatch", {"wrong_launcher_hash": True}, "internally consistent"),
            ("wrong-checksum", {"corrupt_member": True}, "checksum mismatch"),
            ("unlisted-file", {"extra_unlisted": True}, "inventory differs"),
            ("incomplete-run", {"status": "worker_failed_evidence_retained"}, "Only completed"),
        )
        valid = self.result_payload()
        for name, kwargs, error in faults:
            uploads = {"first.zip": self.result_payload(**kwargs), "second.zip": valid}
            with self.subTest(name=name), \
                    patch.object(self.launcher, "ensure_dependencies") as dependencies, \
                    patch.object(self.launcher, "worker") as worker, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = self.launcher.review(
                    uploads, review_mode="repeat", result_order=["first.zip", "second.zip"],
                    review_id=name, base_directory=self.root,
                )
            self.assertEqual(result["status"], "review_failed_evidence_retained")
            self.assertFalse(result["gpu_executed_by_review"])
            worker.assert_not_called()
            dependencies.assert_not_called()
            with zipfile.ZipFile(result["archive"]) as archive:
                status = json.loads(archive.read("launcher_status.json"))
            self.assertIn(error, status["error"])
            self.assertEqual(result["result_archive_sha256"],
                             hashlib.sha256(Path(result["archive"]).read_bytes()).hexdigest())

    def test_review_retains_input_order_and_hashes_and_executes_only_embedded_source(self):
        uploads = {"a.zip": self.result_payload(), "b.zip": self.result_payload()}
        originals = dict(uploads)
        success = {"returncode": 0, "timed_out": False, "interrupted": False,
                   "automatic_retries": 0, "elapsed_seconds": 0.1}
        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", return_value=success) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.review(
                uploads, review_mode="gpu",
                result_order=["b.zip", "a.zip"], review_id="ordered", base_directory=self.root,
            )
        self.assertEqual(result["status"], "worker_completed")
        worker.assert_called_once()
        command = worker.call_args.args[0]
        workspace = Path(result["workspace"])
        self.assertEqual(command[:5], [
            sys.executable, "-u", "-m", "linear_perf", "compare",
        ])
        self.assertEqual(Path(command[command.index("--source") + 1]),
                         workspace / "inputs/input_01/run")
        self.assertEqual(Path(command[command.index("--target") + 1]),
                         workspace / "inputs/input_02/run")
        self.assertEqual(worker.call_args.kwargs["source"],
                         workspace / "source" / ARCHIVE_ROOT.rstrip("/"))
        self.assertEqual(command[command.index("--mode") + 1], "gpu")
        self.assertEqual(worker.call_args.kwargs["timeout"], 900)
        with zipfile.ZipFile(result["archive"]) as archive:
            status = json.loads(archive.read("launcher_status.json"))
        self.assertEqual(status["result_order"], ["b.zip", "a.zip"])
        self.assertEqual([item["uploaded_filename"] for item in status["inputs"]], ["b.zip", "a.zip"])
        self.assertEqual([item["zip_sha256"] for item in status["inputs"]],
                         [hashlib.sha256(uploads[name]).hexdigest() for name in ("b.zip", "a.zip")])
        self.assertFalse(status["gpu_executed_by_review"])
        self.assertEqual(uploads, originals)

    def test_review_analyzes_one_compatible_dataset_from_different_source_code(self):
        other_source = fixture_payload(additional_source="SOURCE_FIXTURE = 'different code; never imported'\n")
        self.assertNotEqual(hashlib.sha256(other_source).hexdigest(), self.launcher.SOURCE_SHA256)
        uploads = {"previous.zip": self.result_payload(source_raw=other_source, action="collect")}
        success = {"returncode": 0, "timed_out": False, "interrupted": False,
                   "automatic_retries": 0, "elapsed_seconds": 0.1}
        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", return_value=success) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.review(uploads, review_mode="analyze", review_id="later-code", base_directory=self.root)
        self.assertEqual(result["status"], "worker_completed")
        worker.assert_called_once()
        command = worker.call_args.args[0]
        workspace = Path(result["workspace"])
        self.assertEqual(command[:5], [sys.executable, "-u", "-m", "linear_perf", "analyze"])
        self.assertEqual(Path(command[command.index("--dataset") + 1]), workspace / "inputs/input_01/run")
        self.assertEqual(worker.call_args.kwargs["source"], workspace / "source" / ARCHIVE_ROOT.rstrip("/"))
        self.assertNotIn("SOURCE_FIXTURE", (worker.call_args.kwargs["source"] / "linear_perf/cli.py").read_text())
        with zipfile.ZipFile(result["archive"]) as archive:
            state = json.loads(archive.read("launcher_status.json"))
        self.assertEqual(state["inputs"][0]["source_sha256"], hashlib.sha256(other_source).hexdigest())
        self.assertEqual(state["source_sha256"], self.launcher.SOURCE_SHA256)
        self.assertFalse(state["gpu_executed_by_review"])

    def test_review_rejects_wrong_input_count_and_duplicate_order_before_workspace(self):
        uploads = {"a.zip": self.result_payload(), "b.zip": self.result_payload()}
        for mode, order, inputs in (("analyze", None, uploads), ("repeat", ["a.zip", "a.zip"], uploads),
                                    ("gpu", None, {"a.zip": uploads["a.zip"]})):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "exactly"):
                self.launcher.review(
                    inputs, review_mode=mode, result_order=order,
                    review_id="bad-count", base_directory=self.root,
                )
        self.assertFalse((self.root / "linear-perf_bad-count_workspace").exists())


if __name__ == "__main__":
    unittest.main()
