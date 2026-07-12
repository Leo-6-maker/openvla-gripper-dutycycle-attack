"""Test R9P full-episode training: batching, loss, forward pass, Model A/B."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.stageb.train_c2g_r9p_preview_detector import (
    R9PEpisodeDataset,
    _hash_language_embedding,
    collate_episodes,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    clean_window_loss,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
)


def _make_npz(path: Path, n_steps: int = 30, suite: str = "libero_spatial",
              has_positive: bool = True) -> dict:
    features_25d = np.random.randn(n_steps, 25).astype(np.float32)
    features_9d = np.random.randn(n_steps, 9).astype(np.float32)
    valid_mask = np.ones(n_steps, dtype=bool)
    known_mask = np.ones(n_steps, dtype=bool)
    step = np.arange(n_steps, dtype=np.int64)
    arrays = {
        "features_25d": features_25d,
        "features_9d": features_9d,
        "valid_mask": valid_mask,
        "known_mask": known_mask,
        "step": step,
    }
    for h in R9P_HEAD_NAMES:
        if h == "critical_window" and has_positive:
            arrays[f"y_{h}"] = np.array(
                [1.0 if 10 <= i < 20 else 0.0 for i in range(n_steps)], dtype=np.float32)
        elif h == "window_start" and has_positive:
            arrays[f"y_{h}"] = np.array(
                [1.0 if i == 10 else 0.0 for i in range(n_steps)], dtype=np.float32)
        elif h == "burst_feasible" and has_positive:
            arrays[f"y_{h}"] = np.array(
                [1.0 if i == 10 else 0.0 for i in range(n_steps)], dtype=np.float32)
        else:
            arrays[f"y_{h}"] = np.zeros(n_steps, dtype=np.float32)
        if h == "grounding_confidence":
            arrays[f"m_{h}"] = np.ones(n_steps, dtype=bool)
        else:
            arrays[f"m_{h}"] = known_mask.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"n_steps": n_steps}


class HashLanguageTests(unittest.TestCase):
    def test_deterministic(self):
        e1 = _hash_language_embedding("pick up the object")
        e2 = _hash_language_embedding("pick up the object")
        np.testing.assert_array_equal(e1, e2)

    def test_different_texts_different(self):
        e1 = _hash_language_embedding("pick up the object")
        e2 = _hash_language_embedding("put down the object")
        self.assertFalse(np.allclose(e1, e2))

    def test_shape(self):
        e = _hash_language_embedding("test")
        self.assertEqual(e.shape, (128,))

    def test_normalized(self):
        e = _hash_language_embedding("test")
        self.assertAlmostEqual(float(np.linalg.norm(e)), 1.0, places=5)


class EpisodeBatchingTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _make_dataset(self, n_episodes: int = 5) -> list[dict]:
        rows = []
        for i in range(n_episodes):
            n_steps = 30 if i % 2 == 0 else 50
            parent_key = f"libero_spatial/task_0/state_{i}/detector_train/episode_{i:03d}"
            npz_path = self.root / "episodes" / f"{parent_key}.npz"
            _make_npz(npz_path, n_steps=n_steps)
            rows.append({
                "npz_path": f"episodes/{parent_key}.npz",
                "preview_split": "FIT",
                "task_language": f"task {i}",
                "parent_key": parent_key,
                "suite": "libero_spatial",
            })
        return rows

    def test_collate_pads_to_max_len(self):
        rows = self._make_dataset(4)
        ds = R9PEpisodeDataset(rows, self.root)
        batch = collate_episodes([ds[i] for i in range(len(ds))])
        self.assertEqual(batch["proprio_25d"].shape, (4, 50, 25))
        self.assertEqual(batch["policy_intent"].shape, (4, 50, 9))

    def test_padding_mask_correct(self):
        rows = self._make_dataset(3)
        ds = R9PEpisodeDataset(rows, self.root)
        batch = collate_episodes([ds[i] for i in range(len(ds))])
        # First episode has 30 steps
        self.assertTrue(batch["padding_mask"][0, :30].all())
        self.assertFalse(batch["padding_mask"][0, 30:].any())

    def test_dataset_filters_by_split(self):
        rows = self._make_dataset(6)
        for i, r in enumerate(rows):
            r["preview_split"] = "FIT" if i < 3 else "CAL"
        ds = R9PEpisodeDataset(rows, self.root, split_filter="FIT")
        self.assertEqual(len(ds), 3)

    def test_language_embedding_in_batch(self):
        rows = self._make_dataset(2)
        ds = R9PEpisodeDataset(rows, self.root)
        batch = collate_episodes([ds[i] for i in range(len(ds))])
        self.assertEqual(batch["language"].shape, (2, 128))


class ModelForwardTests(unittest.TestCase):
    def test_model_a_forward_shape(self):
        config = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, policy_intent_dim=9, hidden=32,
            use_policy_intent=False, use_visual=False, use_language_conditioning=True,
            head_names=R9P_HEAD_NAMES,
        )
        model = C2gGripperCriticalWindowDetector(config)
        proprio = torch.randn(3, 50, 25)
        language = torch.randn(3, 128)
        outputs = model(proprio, language, return_sequence=True)
        for h in R9P_HEAD_NAMES:
            self.assertEqual(outputs[h].shape, (3, 50))

    def test_model_b_forward_shape(self):
        config = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, policy_intent_dim=9, hidden=32,
            use_policy_intent=True, use_visual=False, use_language_conditioning=True,
            head_names=R9P_HEAD_NAMES,
        )
        model = C2gGripperCriticalWindowDetector(config)
        proprio = torch.randn(3, 50, 25)
        policy = torch.randn(3, 50, 9)
        language = torch.randn(3, 128)
        outputs = model(proprio, language, policy_intent=policy, return_sequence=True)
        for h in R9P_HEAD_NAMES:
            self.assertEqual(outputs[h].shape, (3, 50))

    def test_model_a_rejects_policy(self):
        config = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, hidden=32,
            use_policy_intent=False, use_visual=False, use_language_conditioning=True,
            head_names=R9P_HEAD_NAMES,
        )
        model = C2gGripperCriticalWindowDetector(config)
        proprio = torch.randn(3, 50, 25)
        policy = torch.randn(3, 50, 9)
        language = torch.randn(3, 128)
        with self.assertRaises(ValueError):
            model(proprio, language, policy_intent=policy, return_sequence=True)

    def test_loss_backward(self):
        config = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, policy_intent_dim=9, hidden=32,
            use_policy_intent=True, use_visual=False, use_language_conditioning=True,
            head_names=R9P_HEAD_NAMES,
        )
        model = C2gGripperCriticalWindowDetector(config)
        proprio = torch.randn(2, 30, 25)
        policy = torch.randn(2, 30, 9)
        language = torch.randn(2, 128)
        outputs = model(proprio, language, policy_intent=policy, return_sequence=True)
        targets = {h: torch.zeros(2, 30) for h in R9P_HEAD_NAMES}
        masks = {h: torch.ones(2, 30, dtype=torch.bool) for h in R9P_HEAD_NAMES}
        loss_dict = clean_window_loss(
            outputs, targets, masks,
            include_episode_losses=True,
        )
        loss = loss_dict["total"]
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(),
                                f"Non-finite grad in {name}")

    def test_head_names_configurable(self):
        custom_heads = ("window_start", "critical_window", "release_safe")
        config = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, hidden=32,
            use_policy_intent=False, use_visual=False, use_language_conditioning=True,
            head_names=custom_heads,
        )
        model = C2gGripperCriticalWindowDetector(config)
        self.assertEqual(set(model.heads.keys()), set(custom_heads))
        proprio = torch.randn(2, 20, 25)
        language = torch.randn(2, 128)
        outputs = model(proprio, language, return_sequence=True)
        self.assertEqual(set(outputs.keys()), set(custom_heads))


if __name__ == "__main__":
    unittest.main()
