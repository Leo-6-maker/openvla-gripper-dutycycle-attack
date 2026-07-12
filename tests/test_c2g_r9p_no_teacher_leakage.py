"""Test R9P teacher leakage prevention — no forbidden fields in student tensors."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    FORBIDDEN_STUDENT_FIELDS,
)
from tools.multisuite_detector.materialize_c2g_r9p_ogs1500 import (
    FORBIDDEN_NPZ_KEYS,
    _validate_npz_keys,
    materialize_episode,
    write_episode_npz,
)


def _make_episode_with_extra_keys(root: Path, extra_keys: dict) -> Path:
    suite = "libero_spatial"
    parent_key = "libero_spatial/task_0/state_0/detector_train/episode_000"
    ep_dir = root / "episodes" / suite / parent_key
    ep_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": suite, "task_index": 0, "state_id": 0,
        "parent_key": parent_key, "cohort": "DETECTOR_TRAIN",
        "split": "train", "task_language": "test",
    }
    (ep_dir / "derived_episode_metadata.json").write_text(json.dumps(meta))
    steps = []
    labels = []
    for i in range(10):
        step = {"step": i, "features_25d": [0.0]*25, "clean_policy_intent_9d": [0.0]*9}
        step.update(extra_keys)
        steps.append(step)
        labels.append({"step": i, "label_known_mask": True,
                       "y_attack_start_b": False, "y_burst_feasible": False,
                       "y_gripper_critical_window": False, "y_release_safe": False,
                       "y_contact_or_grasp_stable": False, "grounding_confidence": 0.5})
    (ep_dir / "step_records_prefix.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in steps))
    (ep_dir / "teacher_v2_labels.jsonl").write_text(
        "".join(json.dumps(l) + "\n" for l in labels))
    return ep_dir


class NoTeacherLeakageTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_forbidden_npz_keys_blocked(self):
        violations = _validate_npz_keys({"features_25d", "object_pose", "valid_mask"})
        self.assertIn("object_pose", violations)

    def test_clean_npz_has_no_forbidden_keys(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        npz_path = self.root / "clean.npz"
        write_episode_npz(data, npz_path)
        loaded = np.load(npz_path, allow_pickle=False)
        violations = _validate_npz_keys(set(loaded.keys()))
        self.assertEqual(violations, [])

    def test_teacher_phase_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"teacher_phase": "APPROACH"})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_teacher_reason_code_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"teacher_reason_code": "TARGET_CRITICAL_WINDOW"})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_resolved_target_objects_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"resolved_target_objects": ["obj1"]})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_attack_outcome_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"attack_outcome": "success"})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_post_intervention_state_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"post_intervention_state": {}})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_contact_pairs_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"contact_pairs": []})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_object_pose_raises_in_step(self):
        ep_dir = _make_episode_with_extra_keys(self.root, {"object_pose": [0, 0, 0]})
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_forbidden_field_in_npz_keys_detected(self):
        # Manually construct a case where NPZ would have forbidden key
        bad_keys = {"features_25d", "teacher_phase", "attack_outcome"}
        violations = _validate_npz_keys(bad_keys)
        self.assertEqual(len(violations), 2)
        self.assertIn("attack_outcome", violations)
        self.assertIn("teacher_phase", violations)


if __name__ == "__main__":
    unittest.main()
