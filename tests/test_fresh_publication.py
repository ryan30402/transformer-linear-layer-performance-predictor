"""Small negative fixtures for the published-evidence integrity boundary.

These tests do not require the real experiment archive. That archive is checked
separately by scripts/verify_fresh_results.py from the full repository checkout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

from scripts.verify_fresh_results import (
    Verification, sha256_file, verify_archive, verify_asset_checksums, verify_exact_copies,
)


class FreshPublicationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self, *, bad_hash=False, omit_member=False, extra=None, duplicate=False):
        source = b"source is retained as data and is never executed"
        source_hash = hashlib.sha256(source).hexdigest()
        content = {
            "source_archive.zip": source,
            "source_archive.sha256": (source_hash + "\n").encode(),
            "launcher_status.json": b'{"status":"fixture"}',
            "run/metrics.csv": b"metric,value\nerror,4.5\n",
        }
        if extra is not None:
            name = extra.filename if isinstance(extra, zipfile.ZipInfo) else extra
            content[name] = b"rejected unsafe entry"
        members = {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}
        if bad_hash:
            members["run/metrics.csv"] = "0" * 64
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in content.items():
                if omit_member and name == "run/metrics.csv":
                    continue
                entry = extra if isinstance(extra, zipfile.ZipInfo) and extra.filename == name else name
                archive.writestr(entry, value)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("run/metrics.csv", content["run/metrics.csv"])
            archive.writestr("export_manifest.json", json.dumps({"files": members, "source_sha256": source_hash}))
        return path, {"expected_sha256": sha256_file(path), "expected_source_sha256": source_hash,
                      "expected_manifest_files": len(members)}

    def verify_fixture(self, path, settings):
        return verify_archive(path, self.root / "extracted", Verification(), **settings)

    def test_valid_archive_materializes_only_run_records(self):
        path, settings = self.fixture()
        manifest, launcher = self.verify_fixture(path, settings)
        self.assertEqual(4, len(manifest["files"]))
        self.assertEqual("fixture", launcher["status"])
        self.assertEqual(b"metric,value\nerror,4.5\n", (self.root / "extracted/run/metrics.csv").read_bytes())
        self.assertFalse((self.root / "extracted/source_archive.zip").exists())

    def test_archive_tampering_fails_before_extraction(self):
        path, settings = self.fixture()
        with path.open("ab") as stream:
            stream.write(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "Result archive SHA-256 mismatch"):
            self.verify_fixture(path, settings)
        self.assertFalse((self.root / "extracted").exists())

    def test_manifest_hash_mismatch_is_rejected(self):
        path, settings = self.fixture(bad_hash=True)
        with self.assertRaisesRegex(ValueError, "Export member hash mismatch"):
            self.verify_fixture(path, settings)

    def test_manifest_missing_member_is_rejected(self):
        path, settings = self.fixture(omit_member=True)
        with self.assertRaisesRegex(ValueError, "Export manifest coverage mismatch"):
            self.verify_fixture(path, settings)

    def test_source_identity_cannot_be_replaced(self):
        path, settings = self.fixture()
        settings["expected_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source provenance mismatch"):
            self.verify_fixture(path, settings)

    def test_traversal_cannot_write_outside_temporary_root(self):
        path, settings = self.fixture(extra="../escaped.txt")
        with self.assertRaisesRegex(ValueError, "Unsafe publication/archive path"):
            self.verify_fixture(path, settings)
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_symlink_archive_member_is_rejected(self):
        link = zipfile.ZipInfo("run/link")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        path, settings = self.fixture(extra=link)
        with self.assertRaisesRegex(ValueError, "non-regular file"):
            self.verify_fixture(path, settings)

    def test_duplicate_member_is_rejected(self):
        path, settings = self.fixture(duplicate=True)
        with self.assertRaisesRegex(ValueError, "Duplicate archive member"):
            self.verify_fixture(path, settings)

    def test_readable_metric_copy_must_match_preserved_archive(self):
        path, _ = self.fixture()
        copied = self.root / "metrics.csv"
        copied.write_bytes(b"metric,value\nerror,4.5\n")
        mapping = {"metrics.csv": "run/metrics.csv"}
        verify_exact_copies(path, self.root, mapping, Verification())
        copied.write_bytes(b"metric,value\nerror,0.1\n")
        with self.assertRaisesRegex(ValueError, "Readable CSV copy differs"):
            verify_exact_copies(path, self.root, mapping, Verification())

    def test_asset_checksum_detects_changed_summary(self):
        folder = self.root / "publication"
        folder.mkdir()
        (folder / "summary.json").write_text('{"quality_passed": false}\n')
        digest = sha256_file(folder / "summary.json")
        (folder / "SHA256SUMS.txt").write_text(f"{digest}  summary.json\n")
        self.assertEqual(1, verify_asset_checksums(folder, Verification()))
        (folder / "summary.json").write_text('{"quality_passed": true}\n')
        with self.assertRaisesRegex(ValueError, "Publication asset checksum mismatch"):
            verify_asset_checksums(folder, Verification())


if __name__ == "__main__":
    unittest.main()
