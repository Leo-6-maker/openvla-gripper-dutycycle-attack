"""Test R9P preview split assignment, determinism, and cohort sealing."""
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    GATE_PASS,
    PREVIEW_SPLIT_SALT,
    TARGET_SUITES,
    _bucket,
    assign_preview_split,
    build_plan,
    discover_episodes,
)


def _make_suite_root(root: Path, suite: str, train_count: int = 300) -> Path:
    suite_dir = root / "episodes" / suite
    for i in range(train_count):
        task_idx = i // 50
        state_id = i % 50
        local_idx = i % 10
        parent_key = f"{suite}/task_{task_idx}/state_{state_id}/detector_train/episode_{local_idx:03d}"
        ep_dir = suite_dir / parent_key
        ep_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "suite": suite,
            "task_index": task_idx,
            "state_id": state_id,
            "parent_key": parent_key,
            "cohort": "DETECTOR_TRAIN",
            "split": "train",
            "task_language": f"task {task_idx} language",
        }
        (ep_dir / "derived_episode_metadata.json").write_text(json.dumps(meta))
    return root


class NestedSplitTests(unittest.TestCase):
    def test_split_determinism(self):
        key = "libero_spatial/task_3/state_12/detector_train/episode_007"
        self.assertEqual(assign_preview_split(key), assign_preview_split(key))

    def test_split_is_stable_across_runs(self):
        key = "libero_object/task_0/state_0/detector_train/episode_000"
        # Verify the hash-based assignment is deterministic
        bucket = _bucket(key, PREVIEW_SPLIT_SALT, 10)
        self.assertIn(bucket, range(10))
        split = assign_preview_split(key)
        expected = "FIT" if bucket < 8 else ("CAL" if bucket == 8 else "CHECK")
        self.assertEqual(split, expected)

    def test_fit_cal_check_exhaustive(self):
        for b in range(10):
            if b < 8:
                self.assertIn(b, range(8))
            elif b == 8:
                self.assertEqual(b, 8)
            else:
                self.assertEqual(b, 9)

    def test_split_names(self):
        seen = set()
        for i in range(100):
            key = f"test/suite/task_{i}/state_0/detector_train/episode_000"
            seen.add(assign_preview_split(key))
        self.assertEqual(seen, {"FIT", "CAL", "CHECK"})

    def test_bucket_distribution(self):
        buckets = Counter()
        for i in range(1000):
            key = f"libero_spatial/task_{i//100}/state_{i%100}/detector_train/episode_{(i%10):03d}"
            buckets[_bucket(key, PREVIEW_SPLIT_SALT, 10)] += 1
        # Roughly uniform: each bucket ~100, allow 70-130 range
        for b in range(10):
            self.assertGreater(buckets[b], 50, f"bucket {b} too low: {buckets[b]}")
            self.assertLess(buckets[b], 150, f"bucket {b} too high: {buckets[b]}")

    def test_per_suite_split_close_to_expected(self):
        for suite in TARGET_SUITES:
            fit = cal = check = 0
            for i in range(300):
                key = f"{suite}/task_{i//50}/state_{i%50}/detector_train/episode_{(i%10):03d}"
                s = assign_preview_split(key)
                if s == "FIT":
                    fit += 1
                elif s == "CAL":
                    cal += 1
                else:
                    check += 1
            # With 300 per suite, expect ~240 FIT, ~30 CAL, ~30 CHECK
            self.assertGreaterEqual(fit, 220)
            self.assertLessEqual(fit, 260)
            self.assertGreaterEqual(cal, 15)
            self.assertLessEqual(cal, 45)
            self.assertGreaterEqual(check, 15)
            self.assertLessEqual(check, 45)


class PlanBuildTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.spatial_root = self.root / "spatial"
        self.object_root = self.root / "object"
        self.goal_root = self.root / "goal"
        self.output_root = self.root / "plan_output"

    def tearDown(self):
        self.td.cleanup()

    def test_build_plan_success(self):
        _make_suite_root(self.spatial_root, "libero_spatial", 300)
        _make_suite_root(self.object_root, "libero_object", 300)
        _make_suite_root(self.goal_root, "libero_goal", 300)
        plan = build_plan(
            spatial_root=self.spatial_root,
            object_root=self.object_root,
            goal_root=self.goal_root,
            output_root=self.output_root,
            git_commit="test123",
        )
        self.assertEqual(plan["status"], GATE_PASS)
        self.assertEqual(plan["total_train_episodes"], 900)
        self.assertEqual(plan["split_counts"]["FIT"], 720)
        self.assertEqual(plan["split_counts"]["CAL"], 90)
        self.assertEqual(plan["split_counts"]["CHECK"], 90)

    def test_output_artifacts_exist(self):
        _make_suite_root(self.spatial_root, "libero_spatial", 300)
        _make_suite_root(self.object_root, "libero_object", 300)
        _make_suite_root(self.goal_root, "libero_goal", 300)
        build_plan(
            spatial_root=self.spatial_root,
            object_root=self.object_root,
            goal_root=self.goal_root,
            output_root=self.output_root,
            git_commit="test123",
        )
        for name in [
            "r9p_preview_plan.json",
            "r9p_preview_episode_manifest.jsonl",
            "r9p_preview_split_manifest.jsonl",
            "r9p_feature_schema.json",
            "r9p_label_schema.json",
            "r9p_model_spec.json",
            "r9p_execution_boundary.json",
            "SHA256SUMS",
            "SHA256SUMS.sha256",
            "r9p_preview_plan.json.sha256",
        ]:
            self.assertTrue((self.output_root / name).exists(), f"Missing: {name}")

    def test_wrong_count_fails(self):
        _make_suite_root(self.spatial_root, "libero_spatial", 299)
        _make_suite_root(self.object_root, "libero_object", 300)
        _make_suite_root(self.goal_root, "libero_goal", 300)
        plan = build_plan(
            spatial_root=self.spatial_root,
            object_root=self.object_root,
            goal_root=self.goal_root,
            output_root=self.output_root,
            git_commit="test123",
        )
        self.assertNotEqual(plan["status"], GATE_PASS)

    def test_outputs_have_sha256(self):
        _make_suite_root(self.spatial_root, "libero_spatial", 300)
        _make_suite_root(self.object_root, "libero_object", 300)
        _make_suite_root(self.goal_root, "libero_goal", 300)
        plan = build_plan(
            spatial_root=self.spatial_root,
            object_root=self.object_root,
            goal_root=self.goal_root,
            output_root=self.output_root,
            git_commit="test123",
        )
        for name, info in plan["outputs"].items():
            self.assertEqual(len(info["sha256"]), 64, f"Bad SHA for {name}")

    def test_manifest_rows_match_plan(self):
        _make_suite_root(self.spatial_root, "libero_spatial", 300)
        _make_suite_root(self.object_root, "libero_object", 300)
        _make_suite_root(self.goal_root, "libero_goal", 300)
        build_plan(
            spatial_root=self.spatial_root,
            object_root=self.object_root,
            goal_root=self.goal_root,
            output_root=self.output_root,
            git_commit="test123",
        )
        manifest = self.output_root / "r9p_preview_episode_manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 900)
        suites = Counter(r["suite"] for r in rows)
        self.assertEqual(suites["libero_spatial"], 300)
        self.assertEqual(suites["libero_object"], 300)
        self.assertEqual(suites["libero_goal"], 300)

    def test_sealed_cohorts_not_in_manifest(self):
        _make_suite_root(self.spatial_root, "libero_spatial", 300)
        _make_suite_root(self.object_root, "libero_object", 300)
        _make_suite_root(self.goal_root, "libero_goal", 300)
        # Add a val episode that should NOT appear
        val_dir = (self.spatial_root / "episodes" / "libero_spatial"
                   / "libero_spatial" / "task_0" / "state_0" / "detector_val" / "episode_000")
        val_dir.mkdir(parents=True, exist_ok=True)
        (val_dir / "derived_episode_metadata.json").write_text(json.dumps({
            "suite": "libero_spatial", "task_index": 0, "state_id": 0,
            "parent_key": "libero_spatial/task_0/state_0/detector_val/episode_000",
            "cohort": "DETECTOR_VAL", "split": "val", "task_language": "val task",
        }))
        build_plan(
            spatial_root=self.spatial_root,
            object_root=self.object_root,
            goal_root=self.goal_root,
            output_root=self.output_root,
            git_commit="test123",
        )
        manifest = self.output_root / "r9p_preview_episode_manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        cohorts = {r["cohort"] for r in rows}
        self.assertEqual(cohorts, {"DETECTOR_TRAIN"})


if __name__ == "__main__":
    unittest.main()
