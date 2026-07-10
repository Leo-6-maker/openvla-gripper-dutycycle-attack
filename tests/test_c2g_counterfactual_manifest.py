import copy
import unittest

from src.gripper_attack.c2g_counterfactual_manifest import (
    COUNTERFACTUAL_MANIFEST_VERSION,
    validate_counterfactual_manifest,
)
from src.gripper_attack.c2g_teacher_v2_schema import TEACHER_SCHEMA_VERSION


def complete_manifest(tier="TIER_A_MATCHED_ACTION_SHORT_HORIZON"):
    return {
        "manifest_version": COUNTERFACTUAL_MANIFEST_VERSION,
        "comparison_tier": tier,
        "run_id": "c2g-cf-static-fixture",
        "episode_key": "libero_object/task_00/state_000/clean/attempt_01",
        "suite": "libero_object",
        "task_index": 0,
        "state_id": 0,
        "step": 42,
        "candidate_stratum": "PERSISTENT_CONTACT",
        "candidate_reason": "bilateral target contact",
        "snapshot_hash": "a" * 64,
        "snapshot_fields_present": ["qpos", "qvel", "mocap_pos", "mocap_quat", "act", "time"],
        "restore_state_hash": "b" * 64,
        "restore_parity_pass": True,
        "restore_parity_metrics": {"qpos_linf": 0.0, "qvel_linf": 0.0},
        "clean_action_source": "recorded_clean_policy_action",
        "matched_action_alignment_pass": True,
        "short_horizon": 10,
        "closed_loop_continuation_enabled": tier == "TIER_B_CLOSED_LOOP_CONTINUATION",
        "attack_horizon": 10,
        "delivered_attack_steps": 10,
        "force_open_raw_command": 1.0,
        "force_open_env_command": -1.0,
        "clean_continuation_hash": "c" * 64,
        "attack_continuation_hash": "d" * 64,
        "label_known_mask": 1,
        "unknown_reason": "",
        "effect_thresholds": {"drop_z": 0.04, "progress_regression": 0.05},
        "progress_metric_version": "c2g.progress.v1",
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "code_commit": "e" * 40,
        "git_clean": True,
        "simulator_version": "mujoco-static-recorded",
        "libero_version": "recorded-version",
        "policy_model_manifest_sha256": "f" * 64,
        "processor_manifest_sha256": "0" * 64,
        "random_seed": 7,
        "created_at": "2026-07-10T00:00:00Z",
    }


def make_unknown(row, reason):
    row["label_known_mask"] = 0
    row["unknown_reason"] = reason
    row["clean_continuation_hash"] = ""
    row["attack_continuation_hash"] = ""
    return row


class CounterfactualManifestTests(unittest.TestCase):
    def test_complete_tier_a_and_tier_b(self):
        validate_counterfactual_manifest(complete_manifest())
        validate_counterfactual_manifest(complete_manifest("TIER_B_CLOSED_LOOP_CONTINUATION"))

    def test_missing_snapshot_field_is_unknown_masked(self):
        row = make_unknown(complete_manifest(), "SNAPSHOT_INCOMPLETE")
        row["snapshot_fields_present"].remove("qvel")
        validate_counterfactual_manifest(row)
        row["label_known_mask"] = 1
        row["unknown_reason"] = ""
        row["clean_continuation_hash"] = "c" * 64
        row["attack_continuation_hash"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "invalid replay"):
            validate_counterfactual_manifest(row)

    def test_restore_parity_and_action_alignment_fail_closed(self):
        row = make_unknown(complete_manifest(), "RESTORE_MISMATCH")
        row["restore_parity_pass"] = False
        validate_counterfactual_manifest(row)
        row = make_unknown(complete_manifest(), "ACTION_ALIGNMENT_FAILED")
        row["matched_action_alignment_pass"] = False
        validate_counterfactual_manifest(row)

    def test_incomplete_t10_delivery_is_unknown(self):
        row = make_unknown(complete_manifest(), "INCOMPLETE_ATTACK_DELIVERY")
        row["delivered_attack_steps"] = 9
        validate_counterfactual_manifest(row)
        known = copy.deepcopy(row)
        known.update({
            "label_known_mask": 1,
            "unknown_reason": "",
            "clean_continuation_hash": "c" * 64,
            "attack_continuation_hash": "d" * 64,
        })
        with self.assertRaisesRegex(ValueError, "invalid replay"):
            validate_counterfactual_manifest(known)

    def test_unknown_masking_and_provenance_validation(self):
        row = make_unknown(complete_manifest(), "NOT_REPLAYED")
        validate_counterfactual_manifest(row)
        row["code_commit"] = "short"
        with self.assertRaisesRegex(ValueError, "full git commit"):
            validate_counterfactual_manifest(row)


if __name__ == "__main__":
    unittest.main()
