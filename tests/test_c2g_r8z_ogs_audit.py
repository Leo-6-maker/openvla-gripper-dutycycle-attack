import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8z_ogs_full1500 import (
    _verify_derived_receipt,
    validate_r8z_teacher_row,
)
from src.gripper_attack.c2g_clean_window_schema import CLEAN_TEACHER_SCHEMA_VERSION
from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    verify_checksums,
    write_checksums,
    write_json,
    write_report_sidecar,
)


class AuditClosureTests(unittest.TestCase):
    def test_r8z_attack_start_alias_is_validated_outside_frozen_schema(self):
        row = {
            "teacher_schema_version": CLEAN_TEACHER_SCHEMA_VERSION,
            "episode_key": "libero_object/task_0/ep_0",
            "step": 0,
            "suite": "libero_object",
            "task_index": 0,
            "mechanism_type": "pick_place_transfer",
            "mechanism_eligible": True,
            "teacher_phase": "TRANSPORT",
            "teacher_reason_code": "TARGET_CRITICAL_WINDOW",
            "teacher_confidence": 1.0,
            "grounding_confidence": 1.0,
            "teacher_known": True,
            "label_known_mask": True,
            "resolved_target_objects": ["milk"],
            "resolved_target_manipulable_entities": [],
            "contacted_entities": ["milk"],
            "uses_privileged_sim_state": True,
            "uses_attack_outcome": False,
            "uses_future_student_input": False,
            "y_target_relevant": True,
            "y_contact_or_grasp_stable": True,
            "y_gripper_dependency": True,
            "y_clean_close_intent": True,
            "y_lift_transport_or_constraint": True,
            "y_release_safe": False,
            "y_gripper_critical_window": True,
            "y_burst_feasible": False,
            "y_attack_start_b": False,
        }
        row["y_attack_start_B"] = row["y_attack_start_b"]
        row["y_manipulation_progress_active"] = row[
            "y_lift_transport_or_constraint"
        ]
        validate_r8z_teacher_row(row)
        row["y_attack_start_B"] = True
        with self.assertRaisesRegex(ValueError, "alias"):
            validate_r8z_teacher_row(row)

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
