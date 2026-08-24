import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from n5.phase3_student.audit_d0_custodian import extract_keys, run


class TestD0Custodian(unittest.TestCase):
    def test_duplicate_and_string_identity_accounting_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.csv"
            with clean.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["suite", "task_idx", "state_id"])
                writer.writeheader()
                writer.writerow({"suite": "libero_10", "task_idx": "0", "state_id": "0"})
                writer.writerow({"suite": "libero_10", "task_idx": "0", "state_id": "1"})
            protected = root / "protected.json"
            protected.write_text(json.dumps({"identities": [
                "libero_10/task_00/state_01",
                "libero_10/task_00/state_02",
            ], "identity_metadata": [{"suite": "libero_10", "task_idx": 0, "state_id": 2}]}), encoding="utf-8")
            clean_audit = extract_keys(clean)
            protected_audit = extract_keys(protected)
            self.assertEqual(clean_audit["unique_records"], 2)
            self.assertEqual(clean_audit["duplicate_records"], 0)
            self.assertEqual(protected_audit["unique_records"], 2)
            self.assertEqual(protected_audit["duplicate_records"], 1)
            self.assertEqual(len(clean_audit["keys"] & protected_audit["keys"]), 1)

    def test_receipt_does_not_emit_identity_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.csv"
            clean.write_text("suite,task_idx,state_id\nlibero_10,0,0\n", encoding="utf-8")
            protected = root / "protected.json"
            protected.write_text(json.dumps({"identities": ["libero_10/task_00/state_01"]}), encoding="utf-8")
            result = run(Namespace(clean_manifest=str(clean), protected_manifest=[str(protected)], source_root=None, out_parent=str(root), output_name="out"))
            receipt = (root / "out" / "D0_FEASIBILITY_RECEIPT.json").read_text(encoding="utf-8")
            self.assertEqual(result["protected_identity_values_emitted"], 0)
            self.assertNotIn("task_00/state_01", receipt)


if __name__ == "__main__":
    unittest.main()
