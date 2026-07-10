import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.stageb.build_c2g_matched_load_jobs import deterministic_random_start
from src.gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from src.gripper_attack.c2g_clean_window_runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    C2gCleanWindowRuntime,
    derive_gripper_token_semantics,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
)
from tools.multisuite_detector.materialize_c2g_clean_window_dataset import (
    DATASET_SCHEMA_VERSION,
    HEADS,
    materialize,
    ordered_unique_rows,
)
from tools.multisuite_detector.train_c2g_clean_window_detector import main as train_main


def contact_pairs(entity="milk"):
    return [
        ["robot0_left_finger_collision", f"{entity}_collision"],
        ["robot0_right_finger_collision", f"{entity}_collision"],
    ]


def rich_rows(count=6):
    policy = [0.05, 0.95, -2.0, 0.4, 0.8, 0.0, 1.0, 0.4, 0.0]
    rows = []
    for step in range(count):
        rows.append(
            {
                "step": step,
                "features_25d": [float(step) / 10.0] * 25,
                "clean_policy_intent_9d": policy,
                "contact_pairs": contact_pairs(),
                "gripper_command": 0.0,
                "object_relative_lift": 0.03,
                "near_target": False,
                "task_language": "put the milk in the basket",
            }
        )
    return rows


def metadata(key="ep0"):
    return {
        "episode_key": key,
        "parent_key": key,
        "suite": "libero_object",
        "task_index": 0,
        "task_language": "put the milk in the basket",
        "object_declarations": ["milk", "ketchup"],
        "receptacle_declarations": ["basket"],
        "structured_goal_metadata": {
            "target_objects": ["milk"],
            "target_receptacles": ["basket"],
        },
        "mechanism_type": "pick_place_transfer",
        "gripper_command_semantics": "negative_is_close",
    }


def write_episode(root: Path, key="ep0"):
    directory = root / "episodes" / "libero_object" / key
    directory.mkdir(parents=True)
    (directory / "episode_metadata.json").write_text(json.dumps(metadata(key)), encoding="utf-8")
    (directory / "step_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rich_rows()),
        encoding="utf-8",
    )


class MaterializerTests(unittest.TestCase):
    def test_ordering_rejects_duplicate_steps(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ordered_unique_rows([{"step": 1}, {"step": 1}])

    def test_clean_materializer_writes_train_ready_npz(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "input"
            out = Path(td) / "out"
            write_episode(root)
            args = argparse.Namespace(
                input_root=str(root), output_dir=str(out), window=3, burst_length=3,
                contact_persistence_steps=2, relative_lift_threshold=0.015,
                target_progress_threshold=0.01, grounding_confidence_threshold=0.5,
                backend="stats", model_name="unused", openvla_model_path="",
                device="cpu", embedding_dim=16, use_visual=False,
                use_policy_intent=True, drop_all_unknown=False, positive_weight=2.0,
                split_mode="within_task", held_out_task="", held_out_suite="",
                val_fraction=0.15, test_fraction=0.15, seed=42, max_episodes=0,
                fail_fast=True, require_zero_errors=True, git_commit="0" * 40,
            )
            report = materialize(args)
            self.assertEqual(report["status"], "PASS_MATERIALIZED")
            self.assertGreater(report["n_windows"], 0)
            dataset = np.load(report["dataset_path"], allow_pickle=False)
            self.assertEqual(str(dataset["schema_version"]), DATASET_SCHEMA_VERSION)
            self.assertEqual(dataset["X_proprio"].shape[1:], (3, 25))
            self.assertEqual(dataset["X_policy"].shape[-1], len(CLEAN_POLICY_FEATURE_NAMES))
            self.assertIn("y_critical_window", dataset.files)
            self.assertTrue(dataset["m_critical_window"].any())


class MockProcessor:
    def __init__(self):
        self.tokenizer = self

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


class MockVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.bin_centers = np.asarray([-1.0, 1.0], dtype=np.float32)
        self.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=100), pad_to_multiple_of=0)
        self.language_model = SimpleNamespace(get_input_embeddings=lambda: nn.Embedding(10, 4))

    def get_action_stats(self, key):
        return {
            "q01": np.zeros(7, dtype=np.float32),
            "q99": np.ones(7, dtype=np.float32),
            "mask": np.ones(7, dtype=bool),
        }


class RuntimeTests(unittest.TestCase):
    def test_token_semantics_and_runtime_are_checkpoint_bound(self):
        with tempfile.TemporaryDirectory() as td:
            model = MockVLA()
            semantics = derive_gripper_token_semantics(model, "libero_object")
            self.assertEqual(len(semantics["open_token_ids"]), 1)
            self.assertEqual(len(semantics["close_token_ids"]), 1)
            config = C2gDetectorConfig(
                visual_dim=1, language_dim=4, policy_intent_dim=9,
                hidden=8, dropout=0.0, use_policy_intent=True,
                use_visual=False, use_language_conditioning=False,
            )
            detector = C2gGripperCriticalWindowDetector(config)
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state_dict": detector.state_dict(),
                "model_config": config.__dict__,
                "window": 2,
                "thresholds": {
                    "tau_critical": 0.5, "tau_release": 0.5, "tau_ground": 0.5,
                    "persistence_window": 3, "persistence_required": 2,
                },
            }
            path = Path(td) / "checkpoint.pt"
            torch.save(checkpoint, path)
            runtime = C2gCleanWindowRuntime(
                path, openvla_model=model, openvla_processor=MockProcessor(),
                unnorm_key="libero_object", device="cpu", burst_length=3,
            )
            logits = torch.full((100,), -5.0)
            logits[semantics["close_token_ids"][0]] = 5.0
            first = runtime.predict(
                features_25d=[0.0] * 25,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                task_language="test task",
                clean_gripper_logits=logits,
            )
            second = runtime.predict(
                features_25d=[0.1] * 25,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                task_language="test task",
                clean_gripper_logits=logits,
            )
            self.assertFalse(first["ready"])
            self.assertTrue(second["ready"])
            self.assertTrue(second["policy"]["clean_top1_is_close"])


class TrainingTests(unittest.TestCase):
    def make_dataset(self, path: Path):
        n, time_steps = 12, 3
        rng = np.random.default_rng(4)
        split = np.asarray(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
        positive = np.zeros((n, time_steps), dtype=np.float32)
        positive[::2, 1:] = 1.0
        known = np.ones_like(positive, dtype=bool)
        payload = {
            "schema_version": np.asarray(DATASET_SCHEMA_VERSION),
            "X_proprio": rng.normal(size=(n, time_steps, 25)).astype(np.float32),
            "X_policy": rng.normal(size=(n, time_steps, 9)).astype(np.float32),
            "X_visual": np.zeros((n, 1), dtype=np.float32),
            "X_language": rng.normal(size=(n, 4)).astype(np.float32),
            "suite": np.asarray(["libero_object"] * n),
            "task_index": np.zeros(n, dtype=np.int64),
            "episode_key": np.asarray([f"episode_{i}" for i in range(n)]),
            "step": np.arange(n, dtype=np.int64),
            "split": split,
            "episode_fully_known_negative": np.asarray([not row.any() for row in positive], dtype=bool),
            "sample_weight": np.ones((n, time_steps), dtype=np.float32),
            "feature_names_policy": np.asarray(CLEAN_POLICY_FEATURE_NAMES),
        }
        for head in HEADS:
            target = positive.copy()
            if head == "release_safe":
                target = np.zeros_like(positive)
            elif head == "grounding_confidence":
                target = np.ones_like(positive)
            payload[f"y_{head}"] = target
            payload[f"m_{head}"] = known
        np.savez_compressed(path, **payload)

    def test_one_epoch_training_exports_runtime_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "dataset.npz"
            output = Path(td) / "train"
            self.make_dataset(dataset)
            rc = train_main([
                "--dataset", str(dataset), "--output-dir", str(output),
                "--device", "cpu", "--epochs", "1", "--batch-size", "3",
                "--hidden", "8", "--no-use-visual", "--no-use-language-conditioning",
                "--git-commit", "0" * 40,
            ])
            self.assertEqual(rc, 0)
            checkpoint = torch.load(output / "c2g_clean_window_detector.pt", map_location="cpu")
            self.assertEqual(checkpoint["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            self.assertIn("thresholds", checkpoint)


class MatchedLoadHelperTests(unittest.TestCase):
    def test_random_start_is_deterministic_and_not_detector_start(self):
        first = deterministic_random_start(
            "parent", minimum=0, maximum_inclusive=20,
            detector_start=7, master_seed=42,
        )
        second = deterministic_random_start(
            "parent", minimum=0, maximum_inclusive=20,
            detector_start=7, master_seed=42,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, 7)


if __name__ == "__main__":
    unittest.main()
