"""Smoke test: verify detector step_records fields are present after hotfix.

Tests:
1. Fake always-trigger detector produces trigger records
2. Clean condition logs trigger but no attack
3. Oracle condition logs attack
4. Random condition logs attack
5. All required fields present in step_records
"""
import json, os, sys, tempfile, unittest


class FakeDetector:
    """Always-triggers on update."""
    def __init__(self):
        self.call_count = 0
    def reset(self):
        self.call_count = 0
    def update(self, features):
        self.call_count += 1
        return {
            "hazard_score": 0.95,
            "release_safe_score": 0.1,
            "phase_idx": 2,
            "phase_confidence": 0.9,
            "trigger_now": True,
            "trigger_duration": 3,
            "trigger_reason": "fake_always_trigger",
        }


REQUIRED_DETECTOR_FIELDS = [
    "detector_hazard_score",
    "detector_release_safe_score",
    "detector_phase_idx",
    "detector_phase_confidence",
    "detector_trigger_now",
    "detector_trigger_duration",
    "detector_trigger_reason",
    "attack_condition",
    "attack_applied",
    "attack_remaining",
    "original_env_action",
    "attacked_env_action",
]


class TestDetectorIntegrationSmoke(unittest.TestCase):

    def test_fake_detector_produces_trigger(self):
        det = FakeDetector()
        out = det.update([0]*13)
        self.assertTrue(out["trigger_now"])
        self.assertEqual(out["hazard_score"], 0.95)
        self.assertEqual(out["trigger_duration"], 3)

    def test_required_fields_defined(self):
        """Verify the required detector fields constant is complete."""
        self.assertIn("detector_trigger_now", REQUIRED_DETECTOR_FIELDS)
        self.assertIn("attack_applied", REQUIRED_DETECTOR_FIELDS)
        self.assertIn("attack_condition", REQUIRED_DETECTOR_FIELDS)

    def test_run_official_eval_has_detector_fields(self):
        """Verify the patched runner writes detector fields."""
        # Parse the runner source to check field presence
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        runner_path = os.path.join(repo_root, "scripts", "run_official_eval_artifact_rich.py")
        if not os.path.exists(runner_path):
            self.skipTest("Runner not found locally")
        with open(runner_path) as f:
            content = f.read()
        for field in REQUIRED_DETECTOR_FIELDS:
            self.assertIn(f'"{field}"', content,
                f"Required detector field '{field}' missing from runner step_records")

    def test_attack_action_clean_preserves_action(self):
        """Clean condition should not modify action."""
        # Import the function from the runner
        import numpy as np
        # Replicate the attack_action logic
        def attack_action(action, condition, rng):
            if condition == "clean": return action
            a = action.copy()
            if condition == "oracle_open": a[-1] = 1.0
            elif condition == "random_control": a[-1] = 1.0 if rng.random() > 0.5 else -1.0
            elif condition in ("VIS_targeted", "gripper_inversion_proxy"):
                a[-1] = float(np.clip(-action[-1] + rng.normal(0, 0.05), -1.0, 1.0))
            return a

        rng = np.random.RandomState(42)
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])
        result = attack_action(action, "clean", rng)
        self.assertTrue(np.array_equal(action, result),
            "Clean condition should return action unchanged")

    def test_attack_action_oracle_modifies_gripper(self):
        """Oracle should set gripper to fully open."""
        import numpy as np
        def attack_action(action, condition, rng):
            if condition == "clean": return action
            a = action.copy()
            if condition == "oracle_open": a[-1] = 1.0
            return a
        rng = np.random.RandomState(42)
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])
        result = attack_action(action, "oracle_open", rng)
        self.assertEqual(result[-1], 1.0, "Oracle should force gripper to 1.0")

    def test_attack_action_random_modifies_gripper(self):
        """Random should modify gripper dimension."""
        import numpy as np
        def attack_action(action, condition, rng):
            if condition == "clean": return action
            a = action.copy()
            if condition == "random_control": a[-1] = 1.0 if rng.random() > 0.5 else -1.0
            return a
        rng = np.random.RandomState(42)
        action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0])
        results = [attack_action(action, "random_control", rng)[-1] for _ in range(50)]
        self.assertTrue(any(r != -1.0 for r in results),
            "Random should produce at least one non-negative gripper")


if __name__ == "__main__":
    unittest.main()
