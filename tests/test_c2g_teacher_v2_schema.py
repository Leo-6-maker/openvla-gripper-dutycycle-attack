import copy
import unittest

from src.gripper_attack.c2g_teacher_v2_schema import (
    TEACHER_SCHEMA_VERSION,
    assert_student_feature_names,
    validate_candidate_manifest_row,
    validate_teacher_v2_row,
)


def base_row():
    return {
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "teacher_confidence": 0.9,
        "teacher_reason_code": "PRIMARY_TARGET_CARRY",
        "teacher_known": True,
        "label_known_mask": 1,
        "grounding_source": "structured_bddl_predicates+contact",
        "grounding_confidence": 0.9,
        "contacted_objects": ["milk_1"],
        "resolved_target_objects": ["milk_1"],
        "resolved_receptacles": ["basket_1"],
        "resolved_sites": [],
        "target_match": True,
        "object_relative_lift": 0.08,
        "release_distance": None,
        "release_safe_evidence": False,
        "candidate_stratum": "STABLE_CARRY",
        "candidate_reason": "bilateral target contact and relative lift",
        "y_cmdopen_vulnerable": 1,
        "y_contact_loss": 1,
        "y_object_drop": 0,
        "y_progress_regression": 0,
        "y_success_flip": 0,
        "y_release_safe": 0,
        "y_contact_stable": 1,
        "y_grounding_confident": 1,
    }


class TeacherV2SchemaTests(unittest.TestCase):
    def test_known_positive_and_known_negative(self):
        validate_teacher_v2_row(base_row())
        row = base_row()
        row.update({
            "teacher_reason_code": "APPROACH_OR_SETUP",
            "target_match": False,
            "y_cmdopen_vulnerable": 0,
            "y_contact_loss": 0,
        })
        validate_teacher_v2_row(row)

    def test_unknown_is_null_not_negative(self):
        row = base_row()
        row.update({"teacher_known": False, "label_known_mask": 0, "teacher_reason_code": "NOT_REPLAYED"})
        for field in [key for key in row if key.startswith("y_")]:
            row[field] = None
        validate_teacher_v2_row(row)
        row["y_cmdopen_vulnerable"] = 0
        with self.assertRaisesRegex(ValueError, "implicit negatives"):
            validate_teacher_v2_row(row)

    def test_unresolved_target_and_restore_mismatch_cannot_be_known_negative(self):
        for reason in ("TARGET_ID_UNRESOLVED", "RESTORE_MISMATCH"):
            row = base_row()
            row.update({
                "teacher_reason_code": reason,
                "resolved_target_objects": [],
                "target_match": False,
                "y_cmdopen_vulnerable": 0,
                "y_contact_loss": 0,
            })
            with self.assertRaisesRegex(ValueError, "known negative"):
                validate_teacher_v2_row(row)

    def test_release_safe_veto(self):
        row = base_row()
        row["y_release_safe"] = 1
        with self.assertRaisesRegex(ValueError, "vetoes"):
            validate_teacher_v2_row(row)

    def test_confidence_bounds_and_reason_consistency(self):
        row = base_row()
        row["teacher_confidence"] = float("nan")
        with self.assertRaises(ValueError):
            validate_teacher_v2_row(row)
        row = base_row()
        row["teacher_reason_code"] = "NO_CONFIDENT_CONTACT_OBJECT"
        with self.assertRaises(ValueError):
            validate_teacher_v2_row(row)

    def test_vulnerability_requires_harm(self):
        row = base_row()
        for field in ("y_contact_loss", "y_object_drop", "y_progress_regression", "y_success_flip"):
            row[field] = 0
        with self.assertRaisesRegex(ValueError, "causal harm"):
            validate_teacher_v2_row(row)

    def test_candidate_strata_and_student_feature_boundary(self):
        candidate = {
            "candidate_stratum": "RANDOM_NONCANDIDATE_AUDIT",
            "candidate_phase": "noncandidate",
            "candidate_reason": "deterministic recall audit sample",
            "sampling_probability": 0.01,
            "deterministic_seed": 7,
            "selection_used_privileged_state": False,
            "random_noncandidate_recall_audit": True,
        }
        validate_candidate_manifest_row(candidate)
        bad = copy.deepcopy(candidate)
        bad["random_noncandidate_recall_audit"] = False
        with self.assertRaises(ValueError):
            validate_candidate_manifest_row(bad)
        assert_student_feature_names(["feature_0", "siglip_patch_0"])
        with self.assertRaisesRegex(ValueError, "teacher-only"):
            assert_student_feature_names(["feature_0", "contacted_objects"])


if __name__ == "__main__":
    unittest.main()
