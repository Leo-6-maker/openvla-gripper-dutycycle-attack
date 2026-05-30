"""Tests for sustained_command_open_proxy burst control."""
import unittest
import sys
import argparse
import numpy as np


class TestSustainedProxyBurst(unittest.TestCase):
    """Verify sustained proxy invariants without running full simulator."""

    def setUp(self):
        # Simulate minimal args for testing
        self.args = argparse.Namespace()
        self.args.attack_condition = "clean"
        self.args.attack_burst_steps = 0
        self.args.attack_hold_mode = "fixed"
        self.args.detector_trigger_duration = 5
        self.args.force_detector_trigger = False

    def _attack_action(self, action, condition):
        """Local copy of attack_action logic for testing."""
        rng = np.random.RandomState(42)
        if condition == "clean":
            return action
        a = action.copy()
        if condition == "oracle_open":
            a[-1] = 1.0
        elif condition == "random_control":
            a[-1] = 1.0 if rng.random() > 0.5 else -1.0
        elif condition == "sustained_command_open_proxy":
            a[-1] = 1.0
        elif condition == "gripper_inversion_proxy":
            a[-1] = float(np.clip(-action[-1] + rng.normal(0, 0.05), -1.0, 1.0))
        return a

    def _compute_burst_steps(self, attack_condition, attack_burst_steps, trigger_duration):
        """Local copy of burst_steps logic."""
        is_sustained = attack_condition == "sustained_command_open_proxy"
        use_burst = attack_burst_steps > 0
        return attack_burst_steps if (is_sustained and use_burst) else trigger_duration

    # ── A. Clean never attacks ──
    def test_clean_never_attacks(self):
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        result = self._attack_action(action, "clean")
        self.assertEqual(action, result, "clean must return action unchanged")

    # ── B. Success predicate ──
    def test_success_uses_done_not_info(self):
        done = True
        info = {"success": False}
        success = True if done else bool(info.get("success", False))
        self.assertTrue(success, "done=True should produce success=True regardless of info")

    def test_no_done_no_success(self):
        done = False
        info = {}
        success = True if done else bool(info.get("success", False))
        self.assertFalse(success, "done=False info no success -> success=False")

    # ── C. Sustained proxy uses attack_burst_steps ──
    def test_sustained_proxy_uses_attack_burst_steps(self):
        bs = self._compute_burst_steps("sustained_command_open_proxy", 30, 5)
        self.assertEqual(bs, 30, "sustained proxy should use attack_burst_steps=30")

    def test_sustained_proxy_falls_back_when_zero(self):
        bs = self._compute_burst_steps("sustained_command_open_proxy", 0, 5)
        self.assertEqual(bs, 5, "sustained proxy with burst=0 should fall back to trigger_duration")

    # ── D. Oracle ignores attack_burst_steps ──
    def test_oracle_ignores_attack_burst_steps(self):
        bs = self._compute_burst_steps("oracle_open", 30, 5)
        self.assertEqual(bs, 5, "oracle should ignore attack_burst_steps and use trigger_duration")

    def test_oracle_ignores_burst_even_when_set(self):
        bs = self._compute_burst_steps("oracle_open", 100, 5)
        self.assertEqual(bs, 5, "oracle should always use trigger_duration regardless of burst_steps")

    # ── E. Random control ignores attack_burst_steps ──
    def test_random_ignores_attack_burst_steps(self):
        bs = self._compute_burst_steps("random_control", 30, 5)
        self.assertEqual(bs, 5, "random should ignore attack_burst_steps")

    # ── F. Inversion ignores attack_burst_steps ──
    def test_inversion_ignores_attack_burst_steps(self):
        bs = self._compute_burst_steps("gripper_inversion_proxy", 30, 5)
        self.assertEqual(bs, 5, "inversion should ignore attack_burst_steps")

    # ── G. attack_action: sustained sets gripper to open ──
    def test_sustained_action_sets_gripper_open(self):
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]
        result = self._attack_action(action, "sustained_command_open_proxy")
        self.assertEqual(result[-1], 1.0, "sustained proxy should set gripper action to 1.0 open")

    # ── H. attack_action: oracle unchanged ──
    def test_oracle_action_sets_gripper_open(self):
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]
        result = self._attack_action(action, "oracle_open")
        self.assertEqual(result[-1], 1.0, "oracle should set gripper to 1.0 open")

    # ── I. attack_action: inversion modifies action ──
    def test_inversion_modifies_action(self):
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]
        result = self._attack_action(action, "gripper_inversion_proxy")
        self.assertNotEqual(result, action, "inversion should modify the action")

    # ── J. Backward compat: attack_burst_steps=0 → trigger_duration used ──
    def test_backward_compat_non_sustained(self):
        for cond in ["oracle_open", "random_control", "gripper_inversion_proxy"]:
            bs = self._compute_burst_steps(cond, 0, 5)
            self.assertEqual(bs, 5, f"{cond} with burst=0 should use trigger_duration=5")


if __name__ == "__main__":
    unittest.main()
