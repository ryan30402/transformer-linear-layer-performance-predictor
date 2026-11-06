"""Offline backup/recovery contracts for a lost Colab Python/runtime state.

Temporary local directories stand in for a mounted backup filesystem. These
checks exercise real archive copies and checksums, not Google Drive, OAuth,
CUDA measurements, or an assertion of remote-provider durability.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

from tests import test_release_tools as release_contracts


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BackupLauncherContracts(unittest.TestCase):
    def setUp(self):
        release_contracts.ReleaseToolsContracts.setUp(self)
        with zipfile.ZipFile(io.BytesIO(self.raw)) as archive:
            entries = [(name, archive.read(name)) for name in archive.namelist()]
        entries.append((release_contracts.ARCHIVE_ROOT + "linear_perf/checkpoints.py",
                        (ROOT / "linear_perf/checkpoints.py").read_bytes()))
        self.raw = release_contracts.archive_bytes(entries)
        self.launcher.SOURCE_B64 = base64.b64encode(self.raw).decode("ascii")
        self.launcher.SOURCE_SHA256 = hashlib.sha256(self.raw).hexdigest()
        self.recovery = load_module("launcher_recovery_contract", ROOT / "release_tools/colab_recovery.py")
        self.backups = self.root / "mounted_backup_fixture" / "experiments"

    def tearDown(self):
        release_contracts.ReleaseToolsContracts.tearDown(self)

    def fixture_worker(self, command, *, failed=False, **kwargs):
        out = Path(command[command.index("--out") + 1])
        (out / "primary/dataset").mkdir(parents=True)
        (out / "primary/dataset/raw_visits.jsonl").write_text('{"fixture_only":true,"retained":1}\n')
        state = {"status": "failed" if failed else "simulated_complete",
                 "measurement_kind": "simulated", "primary_evaluation_available": not failed,
                 "all_requested_stages_completed": not failed, "fixture_only": True}
        for name in ("status.json", "summary.json"):
            (out / name).write_text(json.dumps(state))
        (out / "report.html").write_text("<p>Software fixture only; no GPU observations.</p>")
        Path(kwargs["log_path"]).write_text("One fixture attempt.\n")
        return {"returncode": 2 if failed else 0, "timed_out": False, "interrupted": False,
                "automatic_retries": 0, "elapsed_seconds": 0.01}

    def test_final_copy_survives_deleted_workspace_and_routes_checkpoints_to_its_session(self):
        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", side_effect=self.fixture_worker) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.run_experiment(experiment_id="durable", simulate=True,
                                                 base_directory=self.root, backup_directory=self.backups)
        worker.assert_called_once()
        command = worker.call_args.args[0]
        expected_folder = self.backups / "linear-perf_durable"
        self.assertEqual(Path(command[command.index("--checkpoint-directory") + 1]), expected_folder)
        self.assertEqual(command[command.index("--checkpoint-source-sha256") + 1], self.launcher.SOURCE_SHA256)
        self.assertEqual(result["backup_status"], "verified")
        local, saved = Path(result["archive"]), Path(result["backup_archive"])
        self.assertEqual(saved.parent, expected_folder)
        original_bytes = local.read_bytes()
        self.assertEqual(saved.read_bytes(), original_bytes)
        self.assertEqual(result["backup_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertTrue(saved.with_name(saved.name + ".sha256").is_file())

        # A new runtime has neither the old directory nor the result variable.
        shutil.rmtree(result["workspace"])
        local.unlink()
        local.with_name(local.name + ".sha256").unlink()
        selected = self.recovery.locate_saved_result(self.backups)
        self.assertEqual(Path(selected["path"]), saved)
        self.assertEqual(selected["sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(selected["kind"], "results")
        self.assertEqual(selected["status"], "simulated_complete")
        with zipfile.ZipFile(selected["path"]) as archive:
            self.assertEqual(archive.read("run/primary/dataset/raw_visits.jsonl"),
                             b'{"fixture_only":true,"retained":1}\n')

    def test_final_copy_failure_retains_downloadable_local_archive_without_retrying_worker(self):
        publish = Mock(side_effect=OSError("fixture backup volume unavailable"))
        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", side_effect=self.fixture_worker) as worker, \
                patch.object(self.launcher, "checkpoint_module",
                             return_value=types.SimpleNamespace(publish_verified_file=publish)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.run_experiment(experiment_id="copy_failure", simulate=True,
                                                 base_directory=self.root, backup_directory=self.backups)
        worker.assert_called_once()
        publish.assert_called_once()
        self.assertEqual(result["status"], "worker_completed")
        self.assertEqual(result["backup_status"], "failed")
        self.assertIn("backup volume unavailable", result["backup_error"])
        self.assertNotIn("backup_archive", result)
        local = Path(result["archive"])
        self.assertTrue(local.is_file())
        with zipfile.ZipFile(local) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("run/primary/dataset/raw_visits.jsonl", archive.namelist())

    def test_failed_measurement_stays_failed_even_when_its_evidence_is_backed_up(self):
        def failed_worker(command, **kwargs):
            return self.fixture_worker(command, failed=True, **kwargs)

        with patch.object(self.launcher, "ensure_dependencies", return_value={"fixture_only": True}), \
                patch.object(self.launcher, "worker", side_effect=failed_worker) as worker, \
                contextlib.redirect_stdout(io.StringIO()):
            result = self.launcher.run_experiment(experiment_id="failed_measurement", simulate=True,
                                                 base_directory=self.root, backup_directory=self.backups)
        worker.assert_called_once()
        self.assertEqual(result["status"], "worker_failed_evidence_retained")
        self.assertEqual(result["backup_status"], "verified")
        selected = self.recovery.locate_saved_result(self.backups)
        self.assertEqual(selected["status"], "failed")
        self.assertFalse(selected["all_requested_stages_completed"])
        self.assertEqual(selected["launcher_status"], "worker_failed_evidence_retained")

    def test_existing_session_backup_is_preserved_before_any_work_starts(self):
        saved = self.backups / "linear-perf_existing"
        saved.mkdir(parents=True)
        evidence = saved / "retained.txt"
        evidence.write_text("Prior attempt must remain unchanged.")
        with patch.object(self.launcher, "prepare_source") as prepare, \
                patch.object(self.launcher, "ensure_dependencies") as dependencies, \
                patch.object(self.launcher, "worker") as worker, self.assertRaises(ValueError):
            self.launcher.run_experiment(experiment_id="existing", simulate=True,
                                         base_directory=self.root, backup_directory=self.backups)
        prepare.assert_not_called()
        dependencies.assert_not_called()
        worker.assert_not_called()
        self.assertEqual(evidence.read_text(), "Prior attempt must remain unchanged.")

    def test_backup_cannot_be_nested_inside_active_workspace(self):
        nested = self.root / "linear-perf_nested_workspace" / "backups"
        with patch.object(self.launcher, "prepare_source") as prepare, self.assertRaises(ValueError):
            self.launcher.run_experiment(experiment_id="nested", simulate=True,
                                         base_directory=self.root, backup_directory=nested)
        prepare.assert_not_called()
        self.assertFalse(nested.exists())

    def test_cli_reports_requested_backup_failure_as_nonzero(self):
        with patch.object(self.launcher, "run", return_value={
            "status": "worker_completed", "backup_status": "failed",
        }) as run:
            code = self.launcher.main(["--session-id", "cli_backup", "--simulate",
                                      "--backup-directory", str(self.backups)])
        self.assertNotEqual(code, 0)
        self.assertEqual(Path(run.call_args.kwargs["backup_directory"]), self.backups)


class SavedResultRecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backups = self.root / "mounted_backup_fixture" / "experiments"
        self.backups.mkdir(parents=True)
        self.recovery = load_module("saved_result_recovery_contract", ROOT / "release_tools/colab_recovery.py")
        self.storage = load_module("saved_result_storage_contract", ROOT / "linear_perf/checkpoints.py")

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, experiment_id="fresh_fixture", *, sequence=None, status="running", extra_entries=()):
        name = (f"linear-perf_{experiment_id}_RESULTS.zip" if sequence is None else
                f"linear-perf_{experiment_id}_CHECKPOINT_{sequence:04d}_primary_timing_completed.zip")
        state = {"session_id": experiment_id, "status": status, "measurement_kind": "simulated",
                 "all_requested_stages_completed": status == "simulated_complete", "fixture_only": True}
        entries = [("run/status.json", json.dumps(state)),
                   ("run/primary/dataset/raw_visits.jsonl", '{"fixture_only":true,"retained":1}\n')]
        if sequence is not None:
            entries.append(("checkpoint.json", json.dumps({
                "artifact_type": "experiment_checkpoint", "session_id": experiment_id,
                "sequence": sequence, "checkpoint_event": "primary_timing_completed", "automatic_resume": False,
            })))
        entries.extend(extra_entries)
        source = self.root / name
        source.write_bytes(release_contracts.archive_bytes(entries))
        receipt = self.storage.publish_verified_file(source, self.backups / f"linear-perf_{experiment_id}")
        source.unlink()
        return Path(receipt["path"])

    def test_timing_checkpoint_can_be_retrieved_without_promoting_it_to_complete(self):
        first = self.publish(sequence=1)
        second = self.publish(sequence=2)
        before = {p.name: p.read_bytes() for p in second.parent.iterdir() if p.is_file()}
        with patch.object(subprocess, "Popen", side_effect=AssertionError("Recovery must not start work")), \
                patch.object(subprocess, "run", side_effect=AssertionError("Recovery must not execute source")):
            selected = self.recovery.locate_saved_result(self.backups)
        self.assertEqual(Path(selected["path"]), second)
        self.assertEqual(selected["kind"], "checkpoint")
        self.assertEqual(selected["status"], "running")
        self.assertFalse(selected["all_requested_stages_completed"])
        self.assertFalse(selected["checkpoint"]["automatic_resume"])
        self.assertIn("not proof", selected["note"])
        self.assertTrue(first.is_file())
        self.assertEqual(before, {p.name: p.read_bytes() for p in second.parent.iterdir() if p.is_file()})

    def test_final_results_take_precedence_but_explicit_checkpoint_remains_available(self):
        snapshot = self.publish(sequence=3)
        final = self.publish(status="simulated_complete")
        selected = self.recovery.locate_saved_result(self.backups)
        self.assertEqual(Path(selected["path"]), final)
        self.assertEqual(selected["kind"], "results")
        selected = self.recovery.locate_saved_result(self.backups, experiment_id="fresh_fixture",
                                                     archive_name=snapshot.name)
        self.assertEqual(Path(selected["path"]), snapshot)
        self.assertEqual(selected["status"], "running")

    def test_incomplete_publication_without_checksum_is_not_selected(self):
        saved = self.publish(sequence=1)
        unfinished = saved.parent / "linear-perf_fresh_fixture_RESULTS.zip"
        unfinished.write_bytes(b"An interrupted publication is not an advertised result.")
        listed = self.recovery.list_saved_experiments(self.backups)
        self.assertEqual(listed[0]["archives"], [saved.name])
        selected = self.recovery.locate_saved_result(self.backups)
        self.assertEqual(Path(selected["path"]), saved)
        with self.assertRaises(ValueError):
            self.recovery.locate_saved_result(self.backups, archive_name=unfinished.name)

    def test_corrupt_latest_checkpoint_is_rejected_without_silently_substituting_an_older_one(self):
        first = self.publish(sequence=1)
        second = self.publish(sequence=2)
        with second.open("ab") as handle:
            handle.write(b"changed after publication")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.recovery.locate_saved_result(self.backups)
        selected = self.recovery.locate_saved_result(self.backups, archive_name=first.name)
        self.assertEqual(Path(selected["path"]), first)

    def test_multiple_saved_experiments_require_an_exact_selection(self):
        first = self.publish("first", status="simulated_complete")
        second = self.publish("second", sequence=1)
        with self.assertRaisesRegex(ValueError, "RECOVERY_EXPERIMENT_ID"):
            self.recovery.locate_saved_result(self.backups)
        selected = self.recovery.locate_saved_result(self.backups, experiment_id="first")
        self.assertEqual(Path(selected["path"]), first)
        selected = self.recovery.locate_saved_result(self.backups, experiment_id="second")
        self.assertEqual(Path(selected["path"]), second)

    def test_selection_cannot_escape_backup_folder_or_follow_an_archive_link(self):
        saved = self.publish(sequence=1)
        for options in ({"experiment_id": "../escape"}, {"archive_name": "../escape.zip"},
                        {"archive_name": str(saved.resolve())}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.recovery.locate_saved_result(self.backups, **options)
        linked = saved.parent / "linked.zip"
        linked.symlink_to(saved)
        with self.assertRaises(FileNotFoundError):
            self.recovery.locate_saved_result(self.backups, archive_name=linked.name)

    def test_checksum_receipt_must_identify_the_selected_file(self):
        saved = self.publish(sequence=1)
        sidecar = saved.with_name(saved.name + ".sha256")
        sidecar.write_text(hashlib.sha256(saved.read_bytes()).hexdigest() + "  another.zip\n")
        with self.assertRaisesRegex(ValueError, "different file"):
            self.recovery.locate_saved_result(self.backups)

    def test_local_result_is_optional_and_a_missing_local_file_falls_back_to_persisted_copy(self):
        saved = self.publish(status="simulated_complete")
        selected = self.recovery.locate_saved_result(self.backups, local_result={
            "archive": str(self.root / "deleted_runtime" / saved.name),
            "result_archive_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
        })
        self.assertEqual(Path(selected["path"]), saved)
        local = self.root / "surviving_local.zip"
        shutil.copyfile(saved, local)
        selected = self.recovery.locate_saved_result(self.root / "unmounted", local_result={
            "archive": str(local), "result_archive_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        })
        self.assertEqual(Path(selected["path"]), local)

    def test_missing_backup_does_not_fabricate_files_or_a_completed_result(self):
        missing = self.root / "not_mounted"
        with self.assertRaises(FileNotFoundError):
            self.recovery.locate_saved_result(missing)
        self.assertFalse(missing.exists())
        with self.assertRaises(ValueError):
            self.recovery.locate_saved_result(self.backups)
        self.assertEqual(list(self.backups.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
