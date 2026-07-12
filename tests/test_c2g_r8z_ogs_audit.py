import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8z_ogs_full1500 import (
    _verify_derived_receipt,
)
from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    verify_checksums,
    write_checksums,
    write_json,
    write_report_sidecar,
)


class AuditClosureTests(unittest.TestCase):
    def test_checksum_and_report_sidecar_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            write_json(report, {"status": "PASS"})
            sidecar = write_report_sidecar(report)
            write_checksums(root)
            self.assertTrue(sidecar.is_file())
            self.assertEqual(verify_checksums(root), (True, "PASS"))
            report.write_text("{}\n", encoding="utf-8")
            passed, reason = verify_checksums(root)
            self.assertFalse(passed)
            self.assertIn("mismatch", reason)

    def test_missing_episode_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = {
                "parent_key": "p",
                "suite": "libero_object",
                "task_index": 0,
                "state_id": 0,
                "cohort": "DETECTOR_TRAIN",
                "split": "train",
            }
            with self.assertRaises(FileNotFoundError):
                _verify_derived_receipt(Path(tmp), expected, "a" * 40)


if __name__ == "__main__":
    unittest.main()

