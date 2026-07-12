import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb import collect_c2g_r8w_teacher_v2_clean as collector


class FullCleanCollectorTests(unittest.TestCase):
    def test_r8t_seed_compatibility(self):
        parent = "libero_goal/task_1/state_2/detector_train/episode_003"
        expected = collector.stable_episode_seed(20260711, parent)
        self.assertEqual(expected, collector.stable_episode_seed(20260711, parent))
        self.assertNotEqual(expected, collector.stable_episode_seed(20260711, parent + "x"))

    def test_success_done_and_max_step_termination(self):
        self.assertEqual(collector.termination_after_step(True, True), "ENV_CHECK_SUCCESS")
        self.assertEqual(collector.termination_after_step(False, True), "DONE_WITHOUT_SUCCESS")
        self.assertIsNone(collector.termination_after_step(False, False))
        self.assertTrue(collector.canonical_clean_success(True, False))
        self.assertTrue(collector.canonical_clean_success(False, True))
        self.assertFalse(collector.canonical_clean_success(False, False))

    def test_post_step_schema(self):
        row = {
            "reward_after_step": 0.0,
            "done_after_step": False,
            "env_check_success_after_step": False,
            "info_success_after_step": None,
            "info_task_success_after_step": None,
            "info_is_success_after_step": None,
        }
        self.assertTrue(collector.validate_post_step_outcome(row))
        del row["reward_after_step"]
        self.assertFalse(collector.validate_post_step_outcome(row))

    def test_mixed_manifest_max_steps_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "mixed max_steps"):
            collector.frozen_manifest_max_steps([{"max_steps": 300}, {"max_steps": 500}])
        self.assertEqual(collector.frozen_manifest_max_steps([{"max_steps": 300}]), 300)

    def test_valid_receipt_skips_and_hash_mismatch_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            episode = Path(td)
            (episode / "rgb").mkdir()
            (episode / "rgb" / "frame_000000.png").write_bytes(b"rgb")
            metadata = episode / "episode_metadata.json"
            metadata.write_text(json.dumps({"runtime_valid": True, "clean_success_observed": False}), encoding="utf-8")
            steps = episode / "step_records.jsonl"
            steps.write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
            rgb_manifest = episode / "rgb_manifest.jsonl"
            _, rgb_sha = collector.build_rgb_manifest(episode / "rgb", rgb_manifest)
            receipt = {
                "schema": collector.EPISODE_RECEIPT_SCHEMA,
                "parent_key": "parent",
                "worker_id": "g4_object",
                "shard_id": "libero_object__shard_0",
                "git_head": "a" * 40,
                "manifest_sha256": "b" * 64,
                "metadata_sha256": collector.sha256_file(metadata),
                "step_records_sha256": collector.sha256_file(steps),
                "rgb_manifest_sha256": rgb_sha,
                "runtime_valid": True,
            }
            collector.write_json(episode / "episode_receipt.json", receipt)
            valid, reason = collector.validate_episode_receipt(
                episode,
                expected_parent_key="parent",
                expected_worker_id="g4_object",
                expected_shard_id="libero_object__shard_0",
                expected_git_head="a" * 40,
                expected_manifest_sha="b" * 64,
            )
            self.assertTrue(valid, reason)
            steps.write_text("tampered\n", encoding="utf-8")
            valid, reason = collector.validate_episode_receipt(
                episode,
                expected_parent_key="parent",
                expected_worker_id="g4_object",
                expected_shard_id="libero_object__shard_0",
                expected_git_head="a" * 40,
                expected_manifest_sha="b" * 64,
            )
            self.assertFalse(valid)
            self.assertIn("step records SHA mismatch", reason)


if __name__ == "__main__":
    unittest.main()
