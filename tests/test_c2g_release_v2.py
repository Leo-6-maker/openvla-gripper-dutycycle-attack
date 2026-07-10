import json
import tempfile
import unittest
from pathlib import Path

from scripts.stageb.build_c2g_suite_model_map import selected_model_manifest
from scripts.stageb.run_c2g_clean_timing_jobs_strict import normalized_parent
from scripts.stageb.run_c2g_matched_load_jobs_release import validate_frozen_clean_parent


class SuiteModelMapTests(unittest.TestCase):
    def test_selected_model_manifest_hashes_required_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.model").write_bytes(b"tokenizer")
            (root / "processor_config.json").write_text("{}", encoding="utf-8")
            result = selected_model_manifest(root)
            self.assertEqual(result["selected_file_count"], 3)
            self.assertEqual(len(result["selected_manifest_sha256"]), 64)

    def test_missing_tokenizer_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "tokenizer"):
                selected_model_manifest(root)


class CleanTimingParentTests(unittest.TestCase):
    def test_parent_normalization_is_identity_preserving(self):
        row = normalized_parent(
            {
                "suite": "libero_goal",
                "task_index": 2,
                "state_id": 5,
                "eval_seed": 100,
                "max_steps": 250,
            }
        )
        self.assertEqual(row["parent_key"], "libero_goal/task_2/state_5")
        self.assertEqual(row["eval_seed"], 100)
        self.assertEqual(row["max_steps"], 250)

    def test_unknown_suite_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown suite"):
            normalized_parent({"suite": "magic", "task_index": 0, "state_id": 0})


class FrozenCleanParentTests(unittest.TestCase):
    def clean_job(self):
        return {
            "parent_key": "libero_object/task_0/state_0/eval_0",
            "suite": "libero_object",
            "task_index": 0,
            "state_id": 0,
            "detector_checkpoint_sha256": "a" * 64,
        }

    def write_clean(self, root: Path, *, attacked=False, commit="b" * 40):
        directory = root / "libero_object/task_0/state_0/eval_0/CLEAN"
        directory.mkdir(parents=True)
        metadata = {
            "runtime_valid": True,
            "parent_key": "libero_object/task_0/state_0/eval_0",
            "condition": "CLEAN",
            "suite": "libero_object",
            "task_index": 0,
            "state_id": 0,
            "protocol_name": "C2G_CLEAN_WINDOW_VIS_PGD",
            "protocol_version": "2026-07-10.v1",
            "git_commit": commit,
            "detector_checkpoint_sha256": "a" * 64,
            "attack_delivery_count": int(attacked),
            "success": True,
        }
        (directory / "episode_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (directory / "step_records.jsonl").write_text(
            json.dumps(
                {
                    "step": 0,
                    "trigger_started": True,
                    "attack_delivered": attacked,
                }
            ) + "\n",
            encoding="utf-8",
        )
        return directory

    def test_valid_clean_parent_passes_without_objective_seed_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.write_clean(root)
            summary = validate_frozen_clean_parent(root, self.clean_job(), "b" * 40)
            self.assertEqual(summary["detector_start_step"], 0)
            self.assertFalse(summary.get("clean_parent_rewritten", False))

    def test_attacked_clean_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.write_clean(root, attacked=True)
            with self.assertRaisesRegex(ValueError, "attacked"):
                validate_frozen_clean_parent(root, self.clean_job(), "b" * 40)

    def test_commit_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.write_clean(root, commit="c" * 40)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_frozen_clean_parent(root, self.clean_job(), "b" * 40)


if __name__ == "__main__":
    unittest.main()
