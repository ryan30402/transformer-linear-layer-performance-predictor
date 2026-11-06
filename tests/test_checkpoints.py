"""CPU evidence-preservation checks; no remote storage or CUDA claim."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from linear_perf import checkpoints, dataset_measurement as dm
from linear_perf.common import read_json, read_jsonl, write_json
from tests.test_dataset_measurement import tiny_manifest


class VerifiedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.source = self.root / "local" / "results.zip"
        self.source.parent.mkdir()
        self.source.write_bytes(b"retained evidence\x00" * 300)
        self.backups = self.root / "persistent"

    def test_publication_preserves_bytes_and_can_be_loaded_without_package_imports(self):
        path = Path(checkpoints.__file__)
        spec = importlib.util.spec_from_file_location("standalone_checkpoint_helper", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        receipt = helper.publish_verified_file(self.source, self.backups)
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["sha256"], expected)
        self.assertEqual(Path(receipt["path"]).read_bytes(), self.source.read_bytes())
        self.assertEqual(Path(receipt["sidecar"]).read_text(), expected + "  results.zip\n")
        self.assertFalse(list(self.backups.glob("*.partial")))

    def test_existing_archive_and_sidecar_are_never_overwritten(self):
        self.backups.mkdir()
        target = self.backups / self.source.name
        target.write_bytes(b"existing independent attempt")
        with self.assertRaisesRegex(checkpoints.CheckpointError, "refusing to overwrite"):
            checkpoints.publish_verified_file(self.source, self.backups)
        self.assertEqual(target.read_bytes(), b"existing independent attempt")
        target.unlink()
        sidecar = self.backups / (self.source.name + ".sha256")
        sidecar.write_text("existing integrity record")
        with self.assertRaisesRegex(checkpoints.CheckpointError, "refusing to overwrite"):
            checkpoints.publish_verified_file(self.source, self.backups)
        self.assertEqual(sidecar.read_text(), "existing integrity record")
        self.assertFalse(target.exists())

    def test_failed_readback_does_not_publish_or_remove_earlier_backup(self):
        self.backups.mkdir()
        retained = self.backups / "earlier.zip"
        retained.write_bytes(b"earlier verified evidence")
        source_before = self.source.read_bytes()
        with patch.object(checkpoints, "_hash_file", return_value="0" * 64):
            with self.assertRaisesRegex(checkpoints.CheckpointError, "readback checksum mismatch"):
                checkpoints.publish_verified_file(self.source, self.backups)
        self.assertEqual(retained.read_bytes(), b"earlier verified evidence")
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertFalse((self.backups / self.source.name).exists())

    def test_competing_writer_between_copy_and_rename_keeps_its_file(self):
        original_hash = checkpoints._hash_file

        def concurrent_publication(path):
            (self.backups / self.source.name).write_bytes(b"another writer won the name")
            return original_hash(path)

        with patch.object(checkpoints, "_hash_file", side_effect=concurrent_publication):
            with self.assertRaises(FileExistsError):
                checkpoints.publish_verified_file(self.source, self.backups)
        self.assertEqual((self.backups / self.source.name).read_bytes(), b"another writer won the name")

    def test_linked_sources_and_unsafe_names_are_rejected(self):
        link = self.source.with_name("linked.zip")
        link.symlink_to(self.source)
        with self.assertRaisesRegex(checkpoints.CheckpointError, "regular file"):
            checkpoints.publish_verified_file(link, self.backups)
        for filename in ("../outside.zip", "/outside.zip", "folder/file.zip", "folder\\file.zip"):
            with self.subTest(filename=filename), self.assertRaises(checkpoints.CheckpointError):
                checkpoints.publish_verified_file(self.source, self.backups, filename=filename)


class CollectionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        self.local = self.root / "ephemeral_vm"
        self.run = self.local / "run"
        self.dataset = self.run / "primary" / "dataset"
        self.backups = self.root / "persistent_backups"
        self.run.mkdir(parents=True)
        write_json(self.run / "status.json", {"status": "running", "measurement_kind": "simulated"})

    def test_full_timing_checkpoint_survives_loss_before_profiler_and_never_claims_completion(self):
        writer = checkpoints.DurableCheckpoints(self.run, self.backups, session_id="preserve", progress=None)
        receipts = []

        def save(event):
            receipts.append(writer.collection(event))

        class LostRuntime(BaseException):
            pass

        # A fatal stop after timing models the concrete pre-profiling loss risk.
        # Local finally blocks may run here; the archived pre-stop bytes must
        # remain the originally recorded incomplete state regardless.
        with patch.object(dm._SimulatedBackend, "profile", side_effect=LostRuntime):
            with self.assertRaises(LostRuntime):
                dm.collect_dataset(tiny_manifest(), self.dataset, session_id="preserve_primary",
                                   simulate=True, progress=None, checkpoint=save)
        self.assertEqual(len(receipts), 3)  # Two complete visits, then all timing.
        archive = Path(receipts[-1]["path"])
        sha = receipts[-1]["sha256"]
        with zipfile.ZipFile(archive) as snapshot:
            expected_raw = snapshot.read("run/primary/dataset/raw_visits.jsonl")
            expected_measurements = snapshot.read("run/primary/dataset/measurements.jsonl")
            state = json.loads(snapshot.read("run/primary/dataset/status.json"))
            self.assertEqual(state["stage"], "timing_collection_completed")
            self.assertNotIn(state["status"], {"completed", "simulated_complete"})
            self.assertGreater(state["completed_visits"], 0)
            self.assertGreater(state["completed_rows"], 0)
            self.assertIn("timing_completed_at_utc", state)
            metadata = json.loads(snapshot.read("checkpoint.json"))
            self.assertFalse(metadata["automatic_resume"])
            self.assertEqual(metadata["collection"]["recorded_status"]["measurement_kind"], "simulated")
            inventory = json.loads(snapshot.read("export_manifest.json"))["files"]
            for name, digest in inventory.items():
                self.assertEqual(hashlib.sha256(snapshot.read(name)).hexdigest(), digest)
        shutil.rmtree(self.local)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), sha)
        with zipfile.ZipFile(archive) as snapshot:
            self.assertEqual(snapshot.read("run/primary/dataset/raw_visits.jsonl"), expected_raw)
            self.assertEqual(snapshot.read("run/primary/dataset/measurements.jsonl"), expected_measurements)

    def test_checkpoint_boundaries_preserve_values_and_allow_normal_validation(self):
        events, profile_seen = [], []
        actual_profile = dm._SimulatedBackend.profile

        def profile(*args, **kwargs):
            self.assertEqual(events[-1]["checkpoint_event"], "timing_completed")
            profile_seen.append(True)
            return actual_profile(*args, **kwargs)

        def save(event):
            events.append(event)
            records = read_jsonl(self.dataset / "raw_visits.jsonl")
            if event["checkpoint_event"] == "timing_visit_completed":
                self.assertEqual(max(row["visit_index"] for row in records), event["visit_index"])
            # The callback owns a copy, not the live collector status.
            event["recorded_status"]["status"] = "untrusted callback mutation"

        with patch.object(dm._SimulatedBackend, "profile", autospec=True, side_effect=profile):
            result = dm.collect_dataset(tiny_manifest(), self.dataset, session_id="boundary",
                                        simulate=True, progress=None, checkpoint=save)
        self.assertEqual(result["status"], "simulated_complete")
        self.assertTrue(profile_seen)
        self.assertEqual([event["checkpoint_event"] for event in events],
                         ["timing_visit_completed", "timing_visit_completed", "timing_completed", "collection_completed"])
        self.assertTrue(dm.validate_collected_dataset(self.dataset)["passed"])
        control = self.root / "without_checkpoints"
        dm.collect_dataset(tiny_manifest(), control, session_id="boundary", simulate=True, progress=None)

        def targets(folder):
            return {(r["case_id"], r["dtype"], r["cache_policy"]): r["median_ms"]
                    for r in read_jsonl(folder / "measurements.jsonl")}

        self.assertEqual(targets(self.dataset), targets(control))

    def test_failed_visit_backup_stops_following_visits_and_keeps_raw_observations(self):
        with patch.object(dm._SimulatedBackend, "profile") as profiler:
            result = dm.collect_dataset(tiny_manifest(), self.dataset, session_id="backup_failure",
                                        simulate=True, progress=None,
                                        checkpoint=lambda event: (_ for _ in ()).throw(OSError("Backup unavailable")))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Backup unavailable", result["error"])
        self.assertEqual({row["visit_index"] for row in read_jsonl(self.dataset / "raw_visits.jsonl")}, {1})
        self.assertEqual(result["checkpoint_failure"]["event"], "timing_visit_completed")
        profiler.assert_not_called()

    def test_failure_to_backup_validated_collection_propagates_without_forging_dataset_status(self):
        def unavailable_after_validation(event):
            if event["checkpoint_event"] == "collection_completed":
                raise OSError("Completed dataset could not be backed up")

        with self.assertRaisesRegex(OSError, "could not be backed up"):
            dm.collect_dataset(tiny_manifest(), self.dataset, session_id="late_backup_failure",
                               simulate=True, progress=None, checkpoint=unavailable_after_validation)
        saved = read_json(self.dataset / "status.json")
        self.assertEqual(saved["status"], "simulated_complete")
        self.assertEqual(saved["checkpoint_failure"]["event"], "collection_completed")
        self.assertTrue(dm.validate_collected_dataset(self.dataset)["passed"])

    def test_snapshot_refuses_source_mismatch_and_linked_evidence(self):
        source = self.local / "source_archive.zip"
        source.write_bytes(b"source fixture")
        writer = checkpoints.DurableCheckpoints(self.run, self.backups, session_id="mismatch",
                                                source_sha256="0" * 64, progress=None)
        with self.assertRaisesRegex(checkpoints.CheckpointError, "checksum differs"):
            writer.stage({"checkpoint_event": "experiment_initialized", "status": "running"})
        self.assertFalse(list(self.backups.glob("*.zip")))
        outside = self.root / "unrelated.txt"
        outside.write_text("Unrelated file must not enter a snapshot")
        (self.run / "linked.txt").symlink_to(outside)
        writer = checkpoints.DurableCheckpoints(self.run, self.backups, session_id="linked", progress=None)
        with self.assertRaisesRegex(checkpoints.CheckpointError, "linked"):
            writer.stage({"checkpoint_event": "experiment_initialized", "status": "running"})


if __name__ == "__main__":
    unittest.main()
