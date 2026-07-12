"""Test R8Z1 checksum completeness verification."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8z1_semantic_prefix_closure import (
    audit_checksum_completeness,
)


class ChecksumCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _write_file(self, relpath: str, content: str = "test") -> str:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()

    def _write_sums(self, entries: list[tuple[str, str]]) -> None:
        """Write SHA256SUMS file. entries = [(sha, relpath), ...]"""
        lines = [f"{sha}  {relpath}\n" for sha, relpath in entries]
        (self.root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")

    def _write_sums_sha(self):
        sums_path = self.root / "SHA256SUMS"
        sha = hashlib.sha256(sums_path.read_bytes()).hexdigest()
        (self.root / "SHA256SUMS.sha256").write_text(
            f"{sha}  SHA256SUMS\n", encoding="utf-8"
        )

    def test_complete_checksum_passes(self):
        sha = self._write_file("data.json")
        self._write_sums([(sha, "data.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["hash_mismatches"], 0)
        self.assertEqual(result["extra_files"], 0)
        self.assertEqual(result["missing_files"], 0)

    def test_hash_mismatch_detected(self):
        self._write_file("data.json", "correct")
        self._write_sums([("0" * 64, "data.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertGreater(result["hash_mismatches"], 0)

    def test_extra_file_detected(self):
        sha = self._write_file("data.json")
        self._write_file("extra.json")  # not in sums
        self._write_sums([(sha, "data.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["extra_files"], 1)

    def test_missing_file_detected(self):
        sha = self._write_file("data.json")
        self._write_sums([(sha, "data.json"), ("0" * 64, "missing.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_files"], 1)

    def test_duplicate_entry_detected(self):
        sha = self._write_file("data.json")
        self._write_sums([(sha, "data.json"), (sha, "data.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["duplicate_entries"], 1)

    def test_absolute_path_detected(self):
        sha = self._write_file("data.json")
        self._write_sums([(sha, "/absolute/path/data.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertGreater(result["absolute_paths"], 0)

    def test_parent_ref_path_detected(self):
        sha = self._write_file("data.json")
        self._write_sums([(sha, "../data.json")])
        self._write_sums_sha()
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertGreater(result["parent_refs"], 0)

    def test_bad_sums_sha256_detected(self):
        sha = self._write_file("data.json")
        self._write_sums([(sha, "data.json")])
        # Write wrong SHA256SUMS.sha256
        (self.root / "SHA256SUMS.sha256").write_text("0" * 64 + "  SHA256SUMS\n")
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["complete"])
        self.assertFalse(result["sums_sha256_ok"])

    def test_missing_sums_file(self):
        self._write_file("data.json")
        result = audit_checksum_completeness(self.root)
        self.assertFalse(result["sums_exists"])
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
