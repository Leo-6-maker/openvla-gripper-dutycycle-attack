import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.stageb.extract_c2g_detector_timing import main as extract_timing_main
from scripts.stageb.prepare_c2g_eval_parents import array_sha256, combined_file_sha256
from src.gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from tools.multisuite_detector.materialize_c2g_clean_window_dataset import DATASET_SCHEMA_VERSION
from tools.multisuite_detector.materialize_c2g_multisuite_dataset import merge_datasets
from tools.multisuite_detector.run_c2g_clean_window_folds import fold_split


class BddlMetadataTests(unittest.TestCase):
    def test_typed_declarations_and_goal_predicates(self):
        text = """
        (define (problem demo)
          (:objects
            milk ketchup - object
            basket - receptacle
            drawer - fixture
          )
          (:goal (and (in milk basket) (open drawer)))
        )
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "demo.bddl"
            path.write_text(text, encoding="utf-8")
            metadata = parse_bddl_task_metadata(path)
        # Preserve every :objects declaration as an object identity. Destination and
        # fixture capability are additional roles, not mutually exclusive classes.
        self.assertEqual(
            metadata["object_declarations"],
            ["basket", "drawer", "ketchup", "milk"],
        )
        self.assertEqual(metadata["receptacle_declarations"], ["basket"])
        self.assertEqual(metadata["fixture_declarations"], ["drawer"])
        self.assertIn(["in", "milk", "basket"], metadata["goal_predicates"])
        self.assertIn(["open", "drawer"], metadata["goal_predicates"])

    def test_missing_goal_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "demo.bddl"
            path.write_text("(define (problem demo) (:objects milk - object))", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no goal predicates"):
                parse_bddl_task_metadata(path)


class FoldSplitTests(unittest.TestCase):
    def test_loto_holds_out_entire_task_and_is_deterministic(self):
        suite = np.asarray(["libero_object"] * 4 + ["libero_goal"] * 2)
        task = np.asarray([0, 0, 1, 1, 0, 0])
        episode = np.asarray([f"episode_{index}" for index in range(6)])
        first = fold_split(
            suite, task, episode,
            mode="loto", held_out="libero_object:1", seed=42, val_fraction=0.25,
        )
        second = fold_split(
            suite, task, episode,
            mode="loto", held_out="libero_object:1", seed=42, val_fraction=0.25,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[2:4] == "test"))
        self.assertFalse(np.any(first[[0, 1, 4, 5]] == "test"))

    def test_loso_holds_out_entire_suite(self):
        suite = np.asarray(["libero_object", "libero_goal", "libero_goal"])
        task = np.asarray([0, 0, 1])
        episode = np.asarray(["a", "b", "c"])
        split = fold_split(
            suite, task, episode,
            mode="loso", held_out="libero_goal", seed=9, val_fraction=0.0,
        )
        self.assertEqual(split.tolist(), ["train", "test", "test"])


class MultisuiteMergeTests(unittest.TestCase):
    @staticmethod
    def write_dataset(path: Path, suite: str, count: int, visual_dim: int = 4):
        payload = {
            "schema_version": np.asarray(DATASET_SCHEMA_VERSION),
            "feature_names_policy": np.asarray(["a", "b"]),
            "X_proprio": np.zeros((count, 3, 25), dtype=np.float32),
            "X_policy": np.zeros((count, 3, 2), dtype=np.float32),
            "X_visual": np.zeros((count, visual_dim), dtype=np.float32),
            "X_language": np.zeros((count, 5), dtype=np.float32),
            "suite": np.asarray([suite] * count),
            "task_index": np.zeros(count, dtype=np.int64),
            "episode_key": np.asarray([f"{suite}_{i}" for i in range(count)]),
            "step": np.arange(count, dtype=np.int64),
            "split": np.asarray(["train"] * count),
        }
        np.savez_compressed(path, **payload)

    def test_merge_concatenates_sample_fields_and_preserves_constants(self):
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first.npz"
            second = Path(td) / "second.npz"
            output = Path(td) / "combined.npz"
            self.write_dataset(first, "libero_object", 2)
            self.write_dataset(second, "libero_goal", 3)
            report = merge_datasets([first, second], output)
            self.assertEqual(report["combined_samples"], 5)
            merged = np.load(output, allow_pickle=False)
            self.assertEqual(merged["X_proprio"].shape, (5, 3, 25))
            self.assertEqual(set(merged["suite"].astype(str)), {"libero_object", "libero_goal"})

    def test_merge_rejects_embedding_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first.npz"
            second = Path(td) / "second.npz"
            self.write_dataset(first, "libero_object", 1, visual_dim=4)
            self.write_dataset(second, "libero_goal", 1, visual_dim=6)
            with self.assertRaisesRegex(ValueError, "shape differs"):
                merge_datasets([first, second], Path(td) / "combined.npz")


class ParentBindingTests(unittest.TestCase):
    def test_array_hash_binds_dtype_shape_and_content(self):
        value = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        self.assertEqual(array_sha256(value), array_sha256(value.copy()))
        self.assertNotEqual(array_sha256(value), array_sha256(value.astype(np.float64)))
        self.assertNotEqual(array_sha256(value), array_sha256(value.reshape(4)))

    def test_combined_file_hash_is_ordered_and_content_bound(self):
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "a.json"
            second = Path(td) / "b.jsonl"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}\n", encoding="utf-8")
            digest = combined_file_sha256((first, second))
            self.assertEqual(len(digest), 64)
            self.assertNotEqual(digest, combined_file_sha256((second, first)))


class DetectorTimingExtractionTests(unittest.TestCase):
    def test_clean_trigger_start_is_extracted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "runs"
            episode = root / "libero_object" / "task_0" / "state_0" / "CLEAN"
            episode.mkdir(parents=True)
            (episode / "episode_metadata.json").write_text(
                json.dumps(
                    {
                        "runtime_valid": True,
                        "parent_key": "libero_object/task_0/state_0",
                        "suite": "libero_object",
                        "task_index": 0,
                        "state_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            (episode / "step_records.jsonl").write_text(
                "".join(
                    json.dumps({"step": step, "trigger_started": step == 8}) + "\n"
                    for step in range(12)
                ),
                encoding="utf-8",
            )
            output = Path(td) / "timing.jsonl"
            rc = extract_timing_main(
                ["--clean-output-root", str(root), "--output", str(output)]
            )
            self.assertEqual(rc, 0)
            row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(row["detector_start_step"], 8)

    def test_missing_trigger_holds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "runs"
            episode = root / "libero_object" / "task_0" / "state_0" / "CLEAN"
            episode.mkdir(parents=True)
            (episode / "episode_metadata.json").write_text(
                json.dumps(
                    {
                        "runtime_valid": True,
                        "parent_key": "libero_object/task_0/state_0",
                        "suite": "libero_object",
                        "task_index": 0,
                        "state_id": 0,
                    }
                ),
                encoding="utf-8",
            )
            (episode / "step_records.jsonl").write_text(
                json.dumps({"step": 0, "trigger_started": False}) + "\n",
                encoding="utf-8",
            )
            output = Path(td) / "timing.jsonl"
            self.assertEqual(
                extract_timing_main(
                    ["--clean-output-root", str(root), "--output", str(output)]
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
