import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.gripper_attack.c2g_clean_policy_signals import CLEAN_POLICY_FEATURE_NAMES
from src.gripper_attack.c2g_clean_window_runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    SUSCEPTIBILITY_SCHEMA_VERSION,
    C2gCleanWindowRuntime,
)
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
)
from tools.multisuite_detector.calibrate_c2g_clean_susceptibility import calibrate


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


def calibration_dataset():
    n, time_steps = 8, 3
    policy = np.zeros((n, time_steps, len(CLEAN_POLICY_FEATURE_NAMES)), dtype=np.float32)
    margin_index = CLEAN_POLICY_FEATURE_NAMES.index("clean_open_minus_close_log_mass")
    entropy_index = CLEAN_POLICY_FEATURE_NAMES.index("clean_action_token_entropy_normalized")
    close_index = CLEAN_POLICY_FEATURE_NAMES.index("clean_top1_is_close")
    policy[:, -1, margin_index] = np.asarray([-5, -4, -3, -2, -1, 0, 1, 2], dtype=np.float32)
    policy[:, -1, entropy_index] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    policy[:, -1, close_index] = 1.0
    target = np.zeros((n, time_steps), dtype=np.float32)
    target[:6, -1] = 1.0
    mask = np.ones((n, time_steps), dtype=bool)
    return {
        "feature_names_policy": np.asarray(CLEAN_POLICY_FEATURE_NAMES),
        "split": np.asarray(["val"] * n),
        "X_policy": policy,
        "y_critical_window": target,
        "m_critical_window": mask,
    }


class SusceptibilityCalibrationTests(unittest.TestCase):
    def test_calibration_uses_clean_validation_positive_rows(self):
        result = calibrate(calibration_dataset(), positive_retention=0.5)
        self.assertEqual(result["schema_version"], SUSCEPTIBILITY_SCHEMA_VERSION)
        self.assertFalse(result["uses_attack_outcomes"])
        self.assertEqual(result["calibration_positive_close_count"], 6)
        self.assertGreater(result["minimum_open_minus_close_log_mass"], -6)
        self.assertGreaterEqual(result["minimum_entropy"], 0.0)

    def test_policy_feature_order_is_frozen(self):
        data = calibration_dataset()
        data["feature_names_policy"] = np.asarray(list(reversed(CLEAN_POLICY_FEATURE_NAMES)))
        with self.assertRaisesRegex(ValueError, "feature order"):
            calibrate(data)

    def test_runtime_prefers_checkpoint_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            config = C2gDetectorConfig(
                visual_dim=1,
                language_dim=4,
                policy_intent_dim=9,
                hidden=8,
                dropout=0.0,
                use_policy_intent=True,
                use_visual=False,
                use_language_conditioning=False,
            )
            model = C2gGripperCriticalWindowDetector(config)
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state_dict": model.state_dict(),
                "model_config": config.__dict__,
                "window": 2,
                "thresholds": {
                    "tau_critical": 0.5,
                    "tau_release": 0.5,
                    "tau_ground": 0.5,
                    "persistence_window": 3,
                    "persistence_required": 2,
                },
                "susceptibility": {
                    "schema_version": SUSCEPTIBILITY_SCHEMA_VERSION,
                    "require_clean_close": True,
                    "minimum_open_minus_close_log_mass": -1.25,
                    "minimum_entropy": 0.33,
                    "uses_attack_outcomes": False,
                },
            }
            checkpoint_path = Path(td) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            runtime = C2gCleanWindowRuntime(
                checkpoint_path,
                openvla_model=MockVLA(),
                openvla_processor=MockProcessor(),
                unnorm_key="libero_object",
                device="cpu",
                minimum_open_minus_close_log_mass=-99.0,
                minimum_entropy=0.0,
            )
            self.assertEqual(runtime.susceptibility["source"], "checkpoint_clean_validation")
            self.assertAlmostEqual(runtime.minimum_open_minus_close_log_mass, -1.25)
            self.assertAlmostEqual(runtime.minimum_entropy, 0.33)

    def test_attack_outcome_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            config = C2gDetectorConfig(
                visual_dim=1, language_dim=4, policy_intent_dim=9,
                hidden=8, use_policy_intent=True, use_visual=False,
                use_language_conditioning=False,
            )
            model = C2gGripperCriticalWindowDetector(config)
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_state_dict": model.state_dict(),
                "model_config": config.__dict__,
                "window": 2,
                "thresholds": {},
                "susceptibility": {
                    "schema_version": SUSCEPTIBILITY_SCHEMA_VERSION,
                    "require_clean_close": True,
                    "minimum_open_minus_close_log_mass": -1.0,
                    "minimum_entropy": 0.1,
                    "uses_attack_outcomes": True,
                },
            }
            path = Path(td) / "bad.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "exclude attacked outcomes"):
                C2gCleanWindowRuntime(
                    path,
                    openvla_model=MockVLA(),
                    openvla_processor=MockProcessor(),
                    unnorm_key="libero_object",
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
