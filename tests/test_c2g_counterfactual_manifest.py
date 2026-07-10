import copy
import unittest

from src.gripper_attack.c2g_counterfactual_manifest import (
    COUNTERFACTUAL_MANIFEST_VERSION,
    REQUIRED_PARITY_METRICS,
    REQUIRED_SNAPSHOT_FIELDS,
    validate_counterfactual_manifest,
)
from src.gripper_attack.c2g_teacher_v2_schema import (
    ATTACK_PROTOCOL_NAME,
    ATTACK_PROTOCOL_VERSION,
    COMPARISON_TIERS,
    TEACHER_SCHEMA_VERSION,
)


def complete_manifest(tier=COMPARISON_TIERS[0]):
    hashes = {name: "a" * 64 for name in REQUIRED_SNAPSHOT_FIELDS}
    metrics = {name: 0.0 for name in REQUIRED_PARITY_METRICS}
    thresholds = {name: 1e-6 for name in REQUIRED_PARITY_METRICS}
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
        "snapshot_hash": "b" * 64,
        "snapshot_fields_present": sorted(REQUIRED_SNAPSHOT_FIELDS),
        "snapshot_component_hashes": hashes,
        "restore_state_hash": "c" * 64,
        "restore_component_hashes": dict(hashes),
        "restore_parity_pass": True,
        "restore_parity_metrics": metrics,
        "restore_parity_thresholds": thresholds,
        "clean_action_source": "recorded_clean_policy_action",
        "matched_action_alignment_pass": True,
        "short_horizon": 20,
        "closed_loop_continuation_enabled": tier == COMPARISON_TIERS[1],
        "attack_protocol_name": ATTACK_PROTOCOL_NAME,
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        "attack_horizon": 10,
        "delivered_attack_steps": 10,
        "force_open_raw_command": 1.0,
        "force_open_env_command": -1.0,
        "clean_continuation_hash": "d" * 64,
        "attack_continuation_hash": "e" * 64,
        "label_known_mask": 1,
        "unknown_reason": "",
        "effect_thresholds": {
            "contact_loss_horizon": 10,
            "object_drop_z_margin": 0.04,
            "progress_regression_margin": 0.05,
            "success_flip_horizon": 100,
            "release_safe_distance": 0.05,
        },
        "progress_metric_version": "c2g.progress.v1",
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "code_commit": "f" * 40,
        "git_clean": True,
        "simulator_version": "mujoco-static-recorded",
        "libero_version": "recorded-version",
        "policy_model_manifest_sha256": "1" * 64,
        "processor_manifest_sha256": "2" * 64,
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
        validate_counterfactual_manifest(complete_manifest(COMPARISON_TIERS[1]))

    def test_missing_snapshot_field_is_unknown_masked(self):
        row = make_unknown(complete_manifest(), "SNAPSHOT_INCOMPLETE")
        row["snapshot_fields_present"].remove("qvel")
        validate_counterfactual_manifest(row)
        row["label_known_mask"] = 1
        row["unknown_reason"] = ""
        row["clean_continuation_hash"] = "d" * 64
        row["attack_continuation_hash"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "invalid replay"):
            validate_counterfactual_manifest(row)

    def test_restore_parity_and_action_alignment_fail_closed(self):
        row = make_unknown(complete_manifest(), "RESTORE_MISMATCH")
        row["restore_parity_metrics"]["qpos_linf"] = 1.0
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
            "clean_continuation_hash": "d" * 64,
            "attack_continuation_hash": "e" * 64,
        })
        with self.assertRaisesRegex(ValueError, "invalid replay"):
            validate_counterfactual_manifest(known)

    def test_protocol_horizon_sign_and_thresholds_are_frozen(self):
        row = complete_manifest()
        row["attack_horizon"] = 9
        with self.assertRaisesRegex(ValueError, "exactly T10"):
            validate_counterfactual_manifest(row)
        row = complete_manifest()
        row["force_open_env_command"] = 1.0
        with self.assertRaisesRegex(ValueError, "sign/value"):
            validate_counterfactual_manifest(row)
        row = complete_manifest()
        del row["effect_thresholds"]["release_safe_distance"]
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            validate_counterfactual_manifest(row)

    def test_unknown_masking_and_provenance_validation(self):
        row = make_unknown(complete_manifest(), "NOT_REPLAYED")
        validate_counterfactual_manifest(row)
        row["code_commit"] = "short"
        with self.assertRaisesRegex(ValueError, "full git commit"):
            validate_counterfactual_manifest(row)


if __name__ == "__main__":
    unittest.main()
