"""Test R9P sealed cohort enforcement — VAL/TEST/ATTACK_EVAL must never be read."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    discover_episodes,
)


def _make_suite_with_cohorts(root: Path, suite: str) -> Path:
    for cohort, count, ep_per_task in [("DETECTOR_TRAIN", 300, 30), ("DETECTOR_VAL", 50, 50),
                                        ("DETECTOR_TEST_WITHIN_TASK", 50, 50),
                                        ("ATTACK_EVAL_PREREGISTERED", 100, 50)]:
        slug = cohort.lower()
        for i in range(count):
            task_idx = i // ep_per_task
            state_id = i % ep_per_task
            local_idx = i % 10
            parent_key = f"{suite}/task_{task_idx}/state_{state_id}/{slug}/episode_{local_idx:03d}"
            ep_dir = root / "episodes" / suite / parent_key
            ep_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "suite": suite, "task_index": task_idx, "state_id": state_id,
                "parent_key": parent_key, "cohort": cohort, "split": "train",
                "task_language": f"task {task_idx}",
            }
            (ep_dir / "derived_episode_metadata.json").write_text(json.dumps(meta))
    return root


class SealedCohortTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_discover_only_returns_train(self):
        _make_suite_with_cohorts(self.root, "libero_spatial")
        rows = discover_episodes(self.root, "libero_spatial")
        cohorts = {r["cohort"] for r in rows}
        self.assertEqual(cohorts, {"DETECTOR_TRAIN"})
        self.assertEqual(len(rows), 300)

    def test_val_cohort_not_in_discovery(self):
        _make_suite_with_cohorts(self.root, "libero_spatial")
        rows = discover_episodes(self.root, "libero_spatial")
        for r in rows:
            self.assertNotEqual(r["cohort"], "DETECTOR_VAL")

    def test_test_cohort_not_in_discovery(self):
        _make_suite_with_cohorts(self.root, "libero_spatial")
        rows = discover_episodes(self.root, "libero_spatial")
        for r in rows:
            self.assertNotEqual(r["cohort"], "DETECTOR_TEST_WITHIN_TASK")

    def test_attack_eval_not_in_discovery(self):
        _make_suite_with_cohorts(self.root, "libero_spatial")
        rows = discover_episodes(self.root, "libero_spatial")
        for r in rows:
            self.assertNotEqual(r["cohort"], "ATTACK_EVAL_PREREGISTERED")

    def test_discover_counts_across_suites(self):
        for suite in ["libero_spatial", "libero_object", "libero_goal"]:
            _make_suite_with_cohorts(self.root / suite, suite)
        total = 0
        for suite in ["libero_spatial", "libero_object", "libero_goal"]:
            rows = discover_episodes(self.root / suite, suite)
            self.assertEqual(len(rows), 300, f"{suite}: expected 300 train")
            total += len(rows)
        self.assertEqual(total, 900)

    def test_train_count_validation(self):
        _make_suite_with_cohorts(self.root, "libero_spatial")
        rows = discover_episodes(self.root, "libero_spatial")
        # Should be exactly 300
        self.assertEqual(len(rows), 300)
        # Modifying to 299 should be caught
        from tools.multisuite_detector.build_c2g_r9p_preview_plan import _validate_episode_counts
        with self.assertRaises(ValueError):
            _validate_episode_counts(rows[:299], "libero_spatial")


if __name__ == "__main__":
    unittest.main()
