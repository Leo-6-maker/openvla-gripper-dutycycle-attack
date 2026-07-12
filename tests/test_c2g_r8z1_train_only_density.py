"""Test R8Z1 train-only label density analysis."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8z1_semantic_prefix_closure import (
    compute_train_density,
)


def _make_label_rows(n: int, known: bool = True, start_pos: bool = False) -> list[dict]:
    rows = []
    for i in range(n):
        row = {
            "step": i,
            "label_known_mask": known and (i % 3 != 0),
            "teacher_reason_code": "TARGET_CRITICAL_WINDOW" if start_pos and i == 10 else "APPROACH_NO_CONTACT",
            "teacher_phase": "APPROACH" if i < 5 else "ENGAGED",
            "y_attack_start_b": start_pos and i == 10,
            "y_burst_feasible": start_pos and i == 10,
            "y_release_safe": i > n - 3,
        }
        rows.append(row)
    return rows


class TrainDensityTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _make_episode(self, task_idx: int, ep_idx: int, known: bool = True,
                      start_pos: bool = False, n_steps: int = 50):
        rows = _make_label_rows(n_steps, known=known, start_pos=start_pos)
        # R8Z layout: episodes/{suite}/{suite}/task_X/state_Y/cohort/ep_Z/
        ep_dir = self.root / "episodes" / "libero_spatial" / "libero_spatial" / f"task_{task_idx}" / "state_0" / "detector_train" / f"episode_{ep_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        label_path = ep_dir / "teacher_v2_labels.jsonl"
        label_path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        meta_path = ep_dir / "derived_episode_metadata.json"
        meta = {"cohort": "DETECTOR_TRAIN", "task_index": task_idx}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def test_counts_known_steps(self):
        self._make_episode(0, 0, known=True)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertEqual(result["episode_count"], 1)
        self.assertGreater(result["known_step_count"], 0)
        self.assertGreater(result["unknown_step_count"], 0)

    def test_counts_start_positive(self):
        self._make_episode(0, 0, start_pos=True)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertGreater(result["start_positive_count"], 0)

    def test_counts_burst_feasible(self):
        self._make_episode(0, 0, start_pos=True)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertGreater(result["burst_feasible_count"], 0)

    def test_counts_release_safe(self):
        self._make_episode(0, 0, n_steps=50)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertGreater(result["release_safe_count"], 0)

    def test_per_task_breakdown(self):
        for t in range(3):
            self._make_episode(t, t)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertEqual(len(result["per_task"]), 3)

    def test_skips_nontrain(self):
        self._make_episode(0, 0)
        # Add a non-train episode
        ep_dir = self.root / "episodes" / "libero_spatial" / "libero_spatial" / "task_0" / "state_0" / "detector_val" / "episode_000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "derived_episode_metadata.json").write_text(json.dumps({"cohort": "DETECTOR_VAL", "task_index": 0}))
        (ep_dir / "teacher_v2_labels.jsonl").write_text(json.dumps({"step": 0, "label_known_mask": True}) + "\n")

        result = compute_train_density(self.root, "libero_spatial")
        self.assertEqual(result["episode_count"], 1)  # only train

    def test_hard_negative_detection(self):
        # All known, no start-positive
        rows = []
        for i in range(20):
            rows.append({
                "step": i, "label_known_mask": True,
                "teacher_reason_code": "APPROACH_NO_CONTACT",
                "teacher_phase": "APPROACH",
                "y_attack_start_b": False, "y_burst_feasible": False,
                "y_release_safe": False,
            })
        ep_dir = self.root / "episodes" / "libero_spatial" / "libero_spatial" / "task_0" / "state_0" / "detector_train" / "episode_000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "teacher_v2_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (ep_dir / "derived_episode_metadata.json").write_text(json.dumps({"cohort": "DETECTOR_TRAIN", "task_index": 0}))
        result = compute_train_density(self.root, "libero_spatial")
        self.assertEqual(result["hard_negative_count"], 1)

    def test_reason_codes_collected(self):
        self._make_episode(0, 0)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertGreater(len(result["reason_codes"]), 0)

    def test_phases_collected(self):
        self._make_episode(0, 0)
        result = compute_train_density(self.root, "libero_spatial")
        self.assertGreater(len(result["phases"]), 0)


if __name__ == "__main__":
    unittest.main()
