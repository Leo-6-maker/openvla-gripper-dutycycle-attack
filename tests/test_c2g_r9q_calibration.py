"""CPU-only tests for R9Q calibration masking and one-shot semantics."""

from types import SimpleNamespace
import unittest

import torch

from scripts.stageb.calibrate_c2g_r9p_preview_thresholds import evaluate_episode_t10
from tools.multisuite_detector.build_c2g_r9p_preview_plan import R9P_HEAD_NAMES


class _DummyDetector:
    def __init__(self, mode: str):
        self.config = SimpleNamespace(use_policy_intent=False)
        self.mode = mode

    def __call__(self, proprio, language, policy_intent=None, return_sequence=True):
        t = proprio.shape[1]
        logits = {head: torch.full((1, t), -10.0) for head in R9P_HEAD_NAMES}
        if self.mode == "first_step":
            logits["critical_window"][0, 0] = 10.0
            logits["grounding_confidence"][0, 0] = 10.0
            logits["release_safe"][0, 0] = -10.0
        elif self.mode == "unknown_step":
            logits["critical_window"][0, 1] = 10.0
            logits["grounding_confidence"][0, 1] = 10.0
            logits["release_safe"][0, 1] = -10.0
        return logits


def _episode(length=3):
    return {
        "features_25d": torch.zeros(length, 25),
        "features_9d": torch.zeros(length, 9),
        "task_language": "",
        "valid_mask": torch.ones(length, dtype=torch.bool),
        "targets": {head: torch.zeros(length) for head in R9P_HEAD_NAMES},
        "masks": {head: torch.ones(length, dtype=torch.bool) for head in R9P_HEAD_NAMES},
    }


class CalibrationMaskTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = {
            "burst_length": 10,
            "tau_critical": 0.5,
            "tau_release": 0.5,
            "tau_ground": 0.5,
            "persistence_window": 1,
            "persistence_required": 1,
        }

    def test_invalid_step_cannot_trigger(self):
        episode = _episode()
        episode["valid_mask"][0] = False
        result = evaluate_episode_t10(
            _DummyDetector("first_step"), episode, torch.device("cpu"), False, self.scheduler
        )
        self.assertFalse(result["triggered"])

    def test_partial_known_episode_is_not_negative(self):
        episode = _episode()
        episode["masks"]["critical_window"][1] = False
        result = evaluate_episode_t10(
            _DummyDetector("first_step"), episode, torch.device("cpu"), False, self.scheduler
        )
        self.assertTrue(result["triggered"])
        self.assertFalse(result["fully_known"])
        self.assertFalse(result["trigger_negative"])
        self.assertFalse(result["negative_any_trigger"])

    def test_fully_known_negative_trigger_is_counted(self):
        episode = _episode()
        result = evaluate_episode_t10(
            _DummyDetector("first_step"), episode, torch.device("cpu"), False, self.scheduler
        )
        self.assertTrue(result["fully_known"])
        self.assertTrue(result["trigger_negative"])
        self.assertTrue(result["negative_any_trigger"])


if __name__ == "__main__":
    unittest.main()
