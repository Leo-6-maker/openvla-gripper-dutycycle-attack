import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.stageb.audit_c2f_track_a_run import PROTOCOL_NAME, PROTOCOL_VERSION, audit_run
from src.gripper_attack.c2g_causal_vulnerability_detector import (
    C2gCausalVulnerabilityDetector,
    _persistent_score,
    c2g_loss,
    first_trigger_episode_losses,
    positive_interval_triggerability,
)
from src.gripper_attack.c2g_counterfactual_manifest import (
    COUNTERFACTUAL_MANIFEST_VERSION,
    REQUIRED_PARITY_METRICS,
    REQUIRED_SNAPSHOT_FIELDS,
    validate_counterfactual_manifest,
)
from src.gripper_attack.c2g_teacher_v2_contact_identity import analyze_contact_pairs
from src.gripper_attack.c2g_teacher_v2_schema import (
    ATTACK_PROTOCOL_NAME,
    ATTACK_PROTOCOL_VERSION,
    COMPARISON_TIERS,
    TEACHER_SCHEMA_VERSION,
    validate_teacher_v2_row,
)
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets
from tools.multisuite_detector.c2g_dataset_scaffold import assert_split_viability, split_label_coverage


class C2gP0PatchTests(unittest.TestCase):
    def test_persistence_never_joins_distant_points(self):
        p = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.8])
        eligible = torch.tensor([1, 0, 0, 0, 1], dtype=torch.bool)
        self.assertEqual(float(_persistent_score(p, eligible)), 0.0)

    def test_untriggerable_positive_is_explicitly_counted(self):
        labels = torch.tensor([[1, 0, 0], [1, 1, 0]])
        known = torch.ones_like(labels, dtype=torch.bool)
        diag = positive_interval_triggerability(labels, known)
        self.assertEqual(float(diag["triggerable_positive_episode_count"]), 1.0)
        self.assertEqual(float(diag["untriggerable_positive_episode_count"]), 1.0)
        losses = first_trigger_episode_losses(torch.zeros(2, 3), labels, known, return_diagnostics=True)
        self.assertEqual(float(losses["untriggerable_positive_episode_count"]), 1.0)

    def test_patch_mask_and_sequence_loss_contract(self):
        model = C2gCausalVulnerabilityDetector(8, 6, hidden=12, dropout=0.0, patch_dim=10).eval()
        temporal = torch.zeros(2, 4, 25)
        visual = torch.zeros(2, 8)
        language = torch.zeros(2, 6)
        patches = torch.zeros(2, 5, 10)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        out = model(temporal, visual, language, return_sequence=True, patch_tokens=patches, patch_token_mask=mask)
        self.assertEqual(out["vulnerability"].shape, (2, 4))
        with self.assertRaisesRegex(ValueError, "at least one valid patch"):
            model(temporal, visual, language, patch_tokens=patches, patch_token_mask=torch.zeros_like(mask))
        last = model(temporal, visual, language)
        targets = {name: torch.zeros_like(value) for name, value in last.items()}
        masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in last.items()}
        with self.assertRaisesRegex(ValueError, "return_sequence=True"):
            c2g_loss(last, targets, masks, include_episode_losses=True)

    def test_role_aware_target_and_contact_resolution(self):
        contains = resolve_task_targets({
            "objects": ["milk_1"],
            "receptacles": ["basket_1"],
            "goal_predicates": [("contains", "basket_1", "milk_1")],
        })
        self.assertEqual(contains.resolved_target_objects, ("milk_1",))
        opened = resolve_task_targets({"fixtures": ["drawer_1"], "goal_predicates": [("open", "drawer_1")]})
        self.assertEqual(opened.resolved_manipulable_entities, ("drawer_1",))
        contact = analyze_contact_pairs([
            ("robot0_l_finger_collision", "drawer_1_handle_collision"),
            ("robot0_r_finger_collision", "drawer_1_handle_visual"),
        ], object_names=[], receptacle_names=["drawer_1"], manipulable_receptacle_names=["drawer_1"])
        self.assertEqual(contact.contacted_manipulable_entities, ("drawer_1",))
        self.assertTrue(contact.bilateral_grasp_candidate)

    def _grounding_row(self):
        return {
            "teacher_schema_version": TEACHER_SCHEMA_VERSION,
            "teacher_confidence": 0.8,
            "teacher_reason_code": "PRIMARY_TARGET_CARRY",
            "teacher_known": True,
            "label_known_mask": 0,
            "causal_label_source": "GROUNDING_ONLY",
            "counterfactual_manifest_sha256": "",
            "counterfactual_replay_valid": False,
            "comparison_tier": "",
            "attack_protocol_name": "",
            "attack_protocol_version": "",
            "grounding_source": "structured+contact",
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
            "candidate_reason": "bilateral contact",
            "y_cmdopen_vulnerable": None,
            "y_contact_loss": None,
            "y_object_drop": None,
            "y_progress_regression": None,
            "y_success_flip": None,
            "y_release_safe": None,
            "y_contact_stable": 1,
            "y_grounding_confident": 1,
        }

    def test_grounding_only_cannot_create_causal_label(self):
        row = self._grounding_row()
        validate_teacher_v2_row(row)
        row["label_known_mask"] = 1
        row["y_cmdopen_vulnerable"] = 0
        with self.assertRaisesRegex(ValueError, "GROUNDING_ONLY"):
            validate_teacher_v2_row(row)

    def _manifest(self):
        hashes = {name: "a" * 64 for name in REQUIRED_SNAPSHOT_FIELDS}
        metrics = {name: 0.0 for name in REQUIRED_PARITY_METRICS}
        thresholds = {name: 1e-6 for name in REQUIRED_PARITY_METRICS}
        return {
            "manifest_version": COUNTERFACTUAL_MANIFEST_VERSION,
            "comparison_tier": COMPARISON_TIERS[0],
            "run_id": "run",
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
            "closed_loop_continuation_enabled": False,
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
            "simulator_version": "mujoco",
            "libero_version": "libero",
            "policy_model_manifest_sha256": "1" * 64,
            "processor_manifest_sha256": "2" * 64,
            "random_seed": 7,
            "created_at": "2026-07-10T00:00:00Z",
        }

    def test_manifest_freezes_snapshot_and_t10(self):
        row = self._manifest()
        validate_counterfactual_manifest(row)
        row["attack_horizon"] = 9
        with self.assertRaisesRegex(ValueError, "exactly T10"):
            validate_counterfactual_manifest(row)

    def _write_episode(self, out: Path, parent: str, condition: str):
        root = out / parent / condition
        root.mkdir(parents=True, exist_ok=True)
        (root / "episode_metadata.json").write_text(json.dumps({
            "parent_key": parent,
            "condition": condition,
            "runtime_valid": True,
            "success": True,
            "git_commit": "a" * 40,
            "protocol_name": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
        }))
        (root / "step_records.jsonl").write_text(json.dumps({"step": 0}) + "\n")

    def test_run_audit_is_closed_world(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "run"
            out = root / "out"
            run.mkdir()
            parent = "libero_object/task_00/state_000/clean/attempt_01"
            self._write_episode(out, parent, "CLEAN")
            jobs = run / "jobs.txt"
            jobs.write_text(f"{parent}|CLEAN\n{parent}|CLEAN\n")
            parents = run / "parents.jsonl"
            parents.write_text("")
            audit = audit_run(run, out, parents, jobs, expected_commit="a" * 40)
            self.assertFalse(audit["complete"])
            self.assertTrue(audit["duplicate_expected_jobs"])
            jobs.write_text(f"{parent}|CLEAN\n")
            self._write_episode(out, parent, "TRUE_CMDOPEN_T10_C2F")
            audit = audit_run(run, out, parents, jobs, expected_commit="a" * 40)
            self.assertFalse(audit["complete"])
            self.assertTrue(audit["unexpected_episode_keys"])

    def test_split_reports_triggerable_positives(self):
        rows = [
            {"episode_key": "a", "split": "train", "suite": "s", "task_index": 0, "step": 0, "label_known_mask": 1, "y_cmdopen_vulnerable": 1},
            {"episode_key": "a", "split": "train", "suite": "s", "task_index": 0, "step": 1, "label_known_mask": 1, "y_cmdopen_vulnerable": 0},
            {"episode_key": "b", "split": "train", "suite": "s", "task_index": 0, "step": 0, "label_known_mask": 1, "y_cmdopen_vulnerable": 1},
            {"episode_key": "b", "split": "train", "suite": "s", "task_index": 0, "step": 1, "label_known_mask": 1, "y_cmdopen_vulnerable": 1},
            {"episode_key": "n", "split": "train", "suite": "s", "task_index": 0, "step": 0, "label_known_mask": 1, "y_cmdopen_vulnerable": 0},
        ]
        coverage = split_label_coverage(rows)
        self.assertEqual(coverage["train"]["triggerable_attackable_episodes"], 1)
        self.assertEqual(coverage["train"]["untriggerable_positive_episodes"], 1)
        assert_split_viability(coverage, required_splits=("train",), min_triggerable_attackable_episodes=1)


if __name__ == "__main__":
    unittest.main()
