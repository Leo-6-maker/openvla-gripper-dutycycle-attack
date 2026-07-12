"""Test R9P materialization: NPZ format, identity closure, finite checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.multisuite_detector.audit_c2g_r9p_materialization import (
    audit_episode_npz,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
    R9P_MODEL_TARGET_MAP,
)
from tools.multisuite_detector.materialize_c2g_r9p_ogs1500 import (
    STUDENT_ALLOWLIST,
    _labels_to_targets,
    materialize_episode,
    write_episode_npz,
)


def _make_episode_dir(root: Path, suite: str, parent_key: str,
                      n_steps: int = 50, known_fraction: float = 0.8,
                      has_positive: bool = True) -> Path:
    ep_dir = root / "episodes" / suite / parent_key
    ep_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "suite": suite,
        "task_index": 0,
        "state_id": 0,
        "parent_key": parent_key,
        "cohort": "DETECTOR_TRAIN",
        "split": "train",
        "task_language": "test task language",
    }
    (ep_dir / "derived_episode_metadata.json").write_text(json.dumps(meta))

    steps = []
    labels = []
    for i in range(n_steps):
        f25 = np.random.randn(25).astype(np.float32).tolist()
        f9 = np.random.randn(9).astype(np.float32).tolist()
        steps.append({
            "step": i,
            "features_25d": f25,
            "clean_policy_intent_9d": f9,
        })
        known = (i % 5) != 0 if known_fraction < 1.0 else True
        label = {
            "step": i,
            "label_known_mask": known,
            "y_attack_start_b": known and has_positive and i == 10,
            "y_burst_feasible": known and has_positive and i == 10,
            "y_gripper_critical_window": known and has_positive and 10 <= i < 20,
            "y_release_safe": known and i > n_steps - 3,
            "y_contact_or_grasp_stable": known and 15 <= i < 25,
            "grounding_confidence": 0.5,
        }
        labels.append(label)

    (ep_dir / "step_records_prefix.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in steps))
    (ep_dir / "teacher_v2_labels.jsonl").write_text(
        "".join(json.dumps(l) + "\n" for l in labels))
    return ep_dir


class MaterializeEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_identity_closure(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=50)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        T = data["n_steps"]
        self.assertEqual(data["features_25d"].shape, (T, 25))
        self.assertEqual(data["features_9d"].shape, (T, 9))
        for h in R9P_HEAD_NAMES:
            self.assertEqual(data["targets"][h].shape, (T,))
            self.assertEqual(data["masks"][h].shape, (T,))
        self.assertEqual(data["valid_mask"].shape, (T,))
        self.assertEqual(data["known_mask"].shape, (T,))

    def test_step_contiguity(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=30)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        np.testing.assert_array_equal(data["step"], np.arange(30))

    def test_features_finite(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=20)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        self.assertTrue(np.isfinite(data["features_25d"]).all())
        self.assertTrue(np.isfinite(data["features_9d"]).all())

    def test_unknown_masking(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=20, known_fraction=0.5)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        unknown = ~data["known_mask"]
        for h in R9P_HEAD_NAMES:
            if h == "grounding_confidence":
                continue
            self.assertFalse(data["targets"][h][unknown].any(), f"{h} non-zero on unknown")
            self.assertFalse(data["masks"][h][unknown].any(), f"{h} mask True on unknown")

    def test_grounding_always_known(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=20, known_fraction=0.5)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        self.assertTrue(data["masks"]["grounding_confidence"].all())

    def test_npz_roundtrip(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=30)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        npz_dir = self.root / "npz_output"
        npz_path = npz_dir / "test.npz"
        sha = write_episode_npz(data, npz_path)
        self.assertEqual(len(sha), 64)

        loaded = np.load(npz_path, allow_pickle=False)
        np.testing.assert_array_equal(loaded["features_25d"], data["features_25d"])
        np.testing.assert_array_equal(loaded["features_9d"], data["features_9d"])
        for h in R9P_HEAD_NAMES:
            np.testing.assert_array_equal(loaded[f"y_{h}"], data["targets"][h])
            np.testing.assert_array_equal(loaded[f"m_{h}"], data["masks"][h])

    def test_audit_passes_valid_npz(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=30)
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        data = materialize_episode(ep_dir, meta)
        npz_path = self.root / "test.npz"
        write_episode_npz(data, npz_path)
        result = audit_episode_npz(npz_path)
        self.assertTrue(result["valid"], msg=f"Issues: {result['issues']}")

    def test_missing_9d_raises(self):
        """Missing clean_policy_intent_9d must fail-closed."""
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=10)
        # Remove 9D from step records
        steps_path = ep_dir / "step_records_prefix.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        for s in steps:
            del s["clean_policy_intent_9d"]
        steps_path.write_text("".join(json.dumps(s) + "\n" for s in steps))
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_missing_grounding_confidence_raises(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=10)
        labels_path = ep_dir / "teacher_v2_labels.jsonl"
        labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
        for l in labels:
            del l["grounding_confidence"]
        labels_path.write_text("".join(json.dumps(l) + "\n" for l in labels))
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)

    def test_allowlist_projection(self):
        """Only STUDENT_ALLOWLIST fields are projected into NPZ."""
        self.assertIn("features_25d", STUDENT_ALLOWLIST)
        self.assertIn("clean_policy_intent_9d", STUDENT_ALLOWLIST)
        self.assertNotIn("teacher_phase", STUDENT_ALLOWLIST)
        self.assertNotIn("resolved_target_objects", STUDENT_ALLOWLIST)

    def test_step_discontinuity_raises(self):
        ep_dir = _make_episode_dir(self.root, "libero_spatial",
                                   "libero_spatial/task_0/state_0/detector_train/episode_000",
                                   n_steps=10)
        # Corrupt step index
        steps_path = ep_dir / "step_records_prefix.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        steps[5]["step"] = 99
        steps_path.write_text("".join(json.dumps(s) + "\n" for s in steps))
        meta = json.loads((ep_dir / "derived_episode_metadata.json").read_text())
        with self.assertRaises(ValueError):
            materialize_episode(ep_dir, meta)


if __name__ == "__main__":
    unittest.main()
