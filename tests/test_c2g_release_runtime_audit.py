import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.stageb.audit_c2g_matched_load_run_release import audit_release


class ReleaseRuntimeAuditTests(unittest.TestCase):
    def base_report(self, unexpected):
        return {
            "status": "HOLD_C2G_MATCHED_LOAD_RUN_AUDIT",
            "missing_jobs": [],
            "unexpected_jobs": unexpected,
            "violation_count": 0,
            "violations": [],
            "parents": [],
            "jobs": [],
        }

    def write_excluded(self, root: Path, parents):
        path = root / "jobs.jsonl.excluded.jsonl"
        path.write_text(
            "".join(
                json.dumps({"parent_key": parent, "reason": "DETECTOR_NO_EMIT"}) + "\n"
                for parent in parents
            ),
            encoding="utf-8",
        )
        return path

    def test_ledger_bound_clean_artifact_is_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            excluded = self.write_excluded(root, ["p0"])
            with patch(
                "scripts.stageb.audit_c2g_matched_load_run_release.base_audit",
                return_value=self.base_report([("p0", "CLEAN")]),
            ):
                report = audit_release(
                    jobs=root / "jobs.jsonl",
                    output_root=root / "online",
                    excluded_ledger=excluded,
                )
            self.assertEqual(report["status"], "PASS_C2G_MATCHED_LOAD_RUN_AUDIT")
            self.assertEqual(report["allowed_excluded_clean_jobs"], [("p0", "CLEAN")])
            self.assertEqual(report["unexpected_jobs"], [])

    def test_nonledger_unexpected_job_remains_hold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            excluded = self.write_excluded(root, ["p0"])
            with patch(
                "scripts.stageb.audit_c2g_matched_load_run_release.base_audit",
                return_value=self.base_report([("p0", "CLEAN"), ("p1", "CLEAN")]),
            ):
                report = audit_release(
                    jobs=root / "jobs.jsonl",
                    output_root=root / "online",
                    excluded_ledger=excluded,
                )
            self.assertEqual(report["status"], "HOLD_C2G_MATCHED_LOAD_RUN_AUDIT")
            self.assertEqual(report["unexpected_jobs"], [("p1", "CLEAN")])

    def test_missing_excluded_clean_artifact_is_hold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            excluded = self.write_excluded(root, ["p0"])
            with patch(
                "scripts.stageb.audit_c2g_matched_load_run_release.base_audit",
                return_value=self.base_report([]),
            ):
                report = audit_release(
                    jobs=root / "jobs.jsonl",
                    output_root=root / "online",
                    excluded_ledger=excluded,
                )
            self.assertEqual(report["status"], "HOLD_C2G_MATCHED_LOAD_RUN_AUDIT")
            reasons = {row["reason"] for row in report["violations"]}
            self.assertIn("EXCLUDED_DENOMINATOR_CLEAN_ARTIFACT_MISSING", reasons)


if __name__ == "__main__":
    unittest.main()
