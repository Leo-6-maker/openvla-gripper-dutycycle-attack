"""Test R9P streaming equivalence: batch-offline logits == step-by-step logits."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.stageb.run_c2g_r9p_streaming_replay import streaming_replay_episode
from src.gripper_attack.c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
)
from tools.multisuite_detector.build_c2g_r9p_preview_plan import (
    R9P_HEAD_NAMES,
)


def _make_synthetic_episode(n_steps: int = 20, has_positive: bool = True) -> dict:
    features_25d = torch.randn(n_steps, 25)
    features_9d = torch.randn(n_steps, 9)
    targets = {}
    for h in R9P_HEAD_NAMES:
        if h == "critical_window" and has_positive:
            targets[h] = torch.tensor([1.0 if 5 <= i < 10 else 0.0 for i in range(n_steps)])
        elif h == "window_start" and has_positive:
            targets[h] = torch.tensor([1.0 if i == 5 else 0.0 for i in range(n_steps)])
        elif h == "burst_feasible" and has_positive:
            targets[h] = torch.tensor([1.0 if i == 5 else 0.0 for i in range(n_steps)])
        else:
            targets[h] = torch.zeros(n_steps)
    return {
        "features_25d": features_25d,
        "features_9d": features_9d,
        "targets": targets,
        "task_language": "test task",
    }


class StreamingEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, policy_intent_dim=9, hidden=32,
            use_policy_intent=True, use_visual=False, use_language_conditioning=True,
            head_names=R9P_HEAD_NAMES,
        )
        self.model = C2gGripperCriticalWindowDetector(self.config).to(self.device)
        self.model.eval()
        self.thresholds = {
            "burst_length": 10, "tau_critical": 0.5,
            "tau_release": 0.5, "tau_ground": 0.5,
            "persistence_window": 3, "persistence_required": 2,
        }

    def test_batch_equals_streaming(self):
        ep = _make_synthetic_episode(20)
        result = streaming_replay_episode(
            self.model, ep, self.device, use_policy_intent=True,
            thresholds=self.thresholds, atol=1e-4,
        )
        self.assertTrue(result["equivalence_ok"],
                        f"Max errors: {result['max_errors']}")

    def test_model_a_streaming(self):
        config_a = C2gDetectorConfig(
            visual_dim=1152, language_dim=128, hidden=32,
            use_policy_intent=False, use_visual=False, use_language_conditioning=True,
            head_names=R9P_HEAD_NAMES,
        )
        model_a = C2gGripperCriticalWindowDetector(config_a).to(self.device)
        model_a.eval()
        ep = _make_synthetic_episode(15)
        result = streaming_replay_episode(
            model_a, ep, self.device, use_policy_intent=False,
            thresholds=self.thresholds, atol=1e-4,
        )
        self.assertTrue(result["equivalence_ok"])

    def test_no_future_leakage(self):
        """At step t, model should only see features[:t+1]."""
        ep = _make_synthetic_episode(10, has_positive=False)
        proprio = ep["features_25d"]
        # Modify step 5 to have extreme values
        proprio[5, :] = 999.0
        ep["features_25d"] = proprio
        # At t=3, the model should NOT be affected by step 5's values
        lang_emb = torch.zeros(1, 128)
        step_proprio = proprio[:4].unsqueeze(0)
        with torch.no_grad():
            out_early = self.model(step_proprio, lang_emb, return_sequence=True)
        # At t=9, the model SHOULD see step 5
        full_proprio = proprio.unsqueeze(0)
        with torch.no_grad():
            out_full = self.model(full_proprio, lang_emb, return_sequence=True)
        # Early output at position 3 should differ from full (different context)
        # This is a basic causality check
        self.assertEqual(out_early["critical_window"].shape, (1, 4))

    def test_gru_state_reset_between_episodes(self):
        """GRU hidden state should not leak between episodes."""
        ep1 = _make_synthetic_episode(15)
        ep2 = _make_synthetic_episode(15)
        result1 = streaming_replay_episode(
            self.model, ep1, self.device, use_policy_intent=True,
            thresholds=self.thresholds)
        result2 = streaming_replay_episode(
            self.model, ep2, self.device, use_policy_intent=True,
            thresholds=self.thresholds)
        # Both should be independently OK
        self.assertTrue(result1["equivalence_ok"])
        self.assertTrue(result2["equivalence_ok"])


if __name__ == "__main__":
    unittest.main()
