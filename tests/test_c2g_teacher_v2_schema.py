import copy
import unittest

from src.gripper_attack.c2g_teacher_v2_schema import (
    ATTACK_PROTOCOL_NAME,
    ATTACK_PROTOCOL_VERSION,
    COMPARISON_TIERS,
    TEACHER_SCHEMA_VERSION,
    assert_student_feature_names,
    validate_candidate_manifest_row,
    validate_teacher_v2_row,
)


def base_grounding_row():
    return {
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "teacher_confidence": 0.9,
        "teacher_reason_code": "PRIMARY_TARGET_CARRY",
        "teacher_known": True,
        "label_known_mask": 0,
        "causal_label_source": "GROUNDING_ONLY",
        "counterfactual_manifest_sha256": "",
        "counterfactual_replay_valid": False,
        "comparison_tier": "",
        "attack_protocol_name": "",
        "attack_protocol_version": "",
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
        "y_cmdopen_vulnerable": None,
        "y_contact_loss": None,
        "y_object_drop": None,
        "y_progress_regression": None,
        "y_success_flip": None,
        "y_release_safe": None,
        "y_contact_stable": 1,
        "y_grounding_confident": 1,
    }


def causal_positive_row():
    row = base_grounding_row()
    row.update({
        "teacher_reason_code": "CONTACT_LOSS_AFTER_CMDOPEN",
        "label_known_mask": 1,
        "causal_label_source": "COUNTERFACTUAL_TIER_A",
        "counterfactual_manifest_sha256": "a" * 64,
        "counterfactual_replay_valid": True,
        "comparison_tier": COMPARISON_TIERS[0],
        "attack_protocol_name": ATTACK_PROTOCOL_NAME,
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        "y_cmdopen_vulnerable": 1,
        "y_contact_loss": 1,
        "y_object_drop": 0,
        "y_progress_regression": 0,
        "y_success_flip": 0,
        "y_release_safe": 0,
    })
    return row


class TeacherV2SchemaTests(unittest.TestCase):
    def test_grounding_only_and_known_positive(self):
        validate_teacher_v2_row(base_grounding_row())
        validate_teacher_v2_row(causal_positive_row())

    def test_grounding_only_cannot_be_known_causal(self):
        row = base_grounding_row()
        row["label_known_mask"] = 1
        row["y_cmdopen_vulnerable"] = 0
        with self.assertRaisesRegex(ValueError, "GROUNDING_ONLY"):
            validate_teacher_v2_row(row)

    def test_unknown_causal_replay_is_null_not_negative(self):
        row = causal_positive_row()
        row.update({
            "teacher_known": False,
            "label_known_mask": 0,
            "teacher_reason_code": "RESTORE_MISMATCH",
            "counterfactual_replay_valid": False,
        })
        for field in [key for key in row if key.startswith("y_") and key not in {"y_contact_stable", "y_grounding_confident"}]:
            row[field] = None
        validate_teacher_v2_row(row)
        row["y_cmdopen_vulnerable"] = 0
        with self.assertRaisesRegex(ValueError, "implicit negatives"):
            validate_teacher_v2_row(row)

    def test_known_negative_requires_complete_no_harm_and_reason(self):
        row = causal_positive_row()
        row.update({
            "teacher_reason_code": "NO_MATERIAL_HARM_AFTER_CMDOPEN",
            "y_cmdopen_vulnerable": 0,
            "y_contact_loss": 0,
        })
        validate_teacher_v2_row(row)
        row["y_object_drop"] = None
        with self.assertRaisesRegex(ValueError, "complete explicit causal outcomes"):
            validate_teacher_v2_row(row)
        row = causal_positive_row()
        row.update({"y_cmdopen_vulnerable": 0, "y_contact_loss": 0})
        with self.assertRaisesRegex(ValueError, "known causal negative requires reason"):
            validate_teacher_v2_row(row)

    def test_release_safe_veto(self):
        row = causal_positive_row()
        row["y_release_safe"] = 1
        with self.assertRaisesRegex(ValueError, "vetoes"):
            validate_teacher_v2_row(row)

    def test_confidence_bounds_and_reason_consistency(self):
        row = causal_positive_row()
        row["teacher_confidence"] = float("nan")
        with self.assertRaises(ValueError):
            validate_teacher_v2_row(row)
        row = base_grounding_row()
        row["teacher_reason_code"] = "NO_CONFIDENT_CONTACT_OBJECT"
        with self.assertRaises(ValueError):
            validate_teacher_v2_row(row)

    def test_vulnerability_requires_harm(self):
        row = causal_positive_row()
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
            assert_student_feature_names(["counterfactual_replay_valid"])


if __name__ == "__main__":
    unittest.main()
