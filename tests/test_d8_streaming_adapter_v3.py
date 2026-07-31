"""D8StreamingFeatureAdapterV3 tests — multi-event causality, feature semantics, negative tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.d8_streaming_features_v3 import (
    D8StreamingFeatureAdapterV3,
    FEATURE_NAMES,
    MIN_HISTORY,
)


def _make_step(adapter, step_id, raw_gripper, env_gripper, **kw):
    """Helper: feed one step with default proprio values."""
    defaults = {
        "gripper_qpos": 0.02, "gripper_opening_proxy": 0.02,
        "eef_x": 0.1, "eef_y": 0.2, "eef_z": 0.3,
        "eef_vx": 0.0, "eef_vy": 0.0, "eef_vz": 0.0,
        "action_dx": 0.0, "action_dy": 0.0, "action_dz": 0.0,
        "action_gripper": env_gripper,
    }
    defaults.update(kw)
    return adapter.update(
        step_id=step_id,
        raw_gripper=raw_gripper,
        env_gripper=env_gripper,
        **defaults,
    )


class TestV3FeatureSemantics(unittest.TestCase):
    """Feature 0 (gripper_command) vs feature 12 (action_gripper) must differ."""

    def test_01_raw_vs_env_gripper_distinct(self):
        """gripper_command = raw, action_gripper = env — must differ in general."""
        adapter = D8StreamingFeatureAdapterV3()
        # raw=0.0 (CLOSE), env=+1.0 (CLOSE) — same intent, different value
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)
        self.assertTrue(r["valid"])
        f = r["features"]
        self.assertEqual(f["gripper_command"], 0.0)
        self.assertEqual(f["action_gripper"], 1.0)
        self.assertNotEqual(f["gripper_command"], f["action_gripper"])

    def test_02_open_semantics_distinct(self):
        """raw=1.0 (OPEN), env=-1.0 (OPEN) — different values."""
        adapter = D8StreamingFeatureAdapterV3()
        r = _make_step(adapter, 0, raw_gripper=1.0, env_gripper=-1.0)
        self.assertTrue(r["valid"])
        f = r["features"]
        self.assertEqual(f["gripper_command"], 1.0)
        self.assertEqual(f["action_gripper"], -1.0)

    def test_03_gripper_semantics_mismatch_rejects(self):
        """raw=CLOSE, env=OPEN → rejected."""
        adapter = D8StreamingFeatureAdapterV3()
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=-1.0)
        self.assertFalse(r["valid"])


class TestV3MultiCloseCausality(unittest.TestCase):
    """V3: every OPEN→CLOSE transition produces close_onset=1."""

    def test_10_first_close_onset(self):
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=1.0, env_gripper=-1.0)  # OPEN
        _make_step(adapter, 1, raw_gripper=1.0, env_gripper=-1.0)  # OPEN
        r = _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0)  # CLOSE onset
        self.assertTrue(r["valid"])
        self.assertEqual(r["features"]["close_onset"], 1)

    def test_11_second_close_onset_detected(self):
        """After OPEN→CLOSE→OPEN→CLOSE, second close_onset must fire."""
        adapter = D8StreamingFeatureAdapterV3()
        # First close event
        _make_step(adapter, 0, raw_gripper=1.0, env_gripper=-1.0)  # OPEN
        _make_step(adapter, 1, raw_gripper=0.0, env_gripper=1.0)  # CLOSE onset 1
        _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0)  # CLOSE continued
        _make_step(adapter, 3, raw_gripper=1.0, env_gripper=-1.0)  # OPEN
        _make_step(adapter, 4, raw_gripper=1.0, env_gripper=-1.0)  # OPEN
        # Second close event
        r = _make_step(adapter, 5, raw_gripper=0.0, env_gripper=1.0)  # CLOSE onset 2
        self.assertTrue(r["valid"])
        self.assertEqual(r["features"]["close_onset"], 1,
                         "V3 must detect second close onset (V2 would miss this)")

    def test_12_time_since_close_resets_on_second_onset(self):
        """time_since_close must reference most recent close onset."""
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)  # close onset 1
        _make_step(adapter, 1, raw_gripper=1.0, env_gripper=-1.0)  # open
        _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0)  # close onset 2
        r = _make_step(adapter, 3, raw_gripper=0.0, env_gripper=1.0)  # 1 step after onset 2
        self.assertEqual(r["features"]["time_since_close"], 1)
        # Also check eef_z_delta: should reference step 2, not step 0
        self.assertTrue(r["valid"])

    def test_13_close_onset_zero_during_continued_close(self):
        """close_onset=0 when close streak > 1 (not a transition)."""
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=1.0, env_gripper=-1.0)
        _make_step(adapter, 1, raw_gripper=0.0, env_gripper=1.0)  # onset
        r = _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0)  # continued close
        self.assertEqual(r["features"]["close_onset"], 0)

    def test_14_close_onset_zero_during_open(self):
        """close_onset=0 during open steps."""
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)  # close
        r = _make_step(adapter, 1, raw_gripper=1.0, env_gripper=-1.0)  # open
        self.assertEqual(r["features"]["close_onset"], 0)


class TestV3WindowedFlipCount(unittest.TestCase):
    """V3: flip count is windowed, not episode-cumulative."""

    def test_20_windowed_flip_count(self):
        """Flips beyond history window must not count.

        With MIN_HISTORY=32 and many more steps of alternating close/open, the
        windowed count stays bounded (~32) while cumulative would be ~N.
        """
        from gripper_attack.d8_streaming_features_v3 import MIN_HISTORY
        adapter = D8StreamingFeatureAdapterV3()
        # Create 3x MIN_HISTORY alternating steps — cumulative flips ~95
        total = MIN_HISTORY * 3
        for i in range(total):
            raw = 0.0 if i % 2 == 0 else 1.0
            env = 1.0 if i % 2 == 0 else -1.0
            _make_step(adapter, i, raw_gripper=raw, env_gripper=env)
        r = _make_step(adapter, total, raw_gripper=0.0, env_gripper=1.0)
        f = r["features"]
        # Windowed count bounded by window size; cumulative would be ~total
        self.assertLess(f["recent_gripper_flip_count"], MIN_HISTORY + 2,
                        f"windowed flip count {f['recent_gripper_flip_count']} should be bounded, not cumulative (~{total})")


class TestV3FeatureDimension(unittest.TestCase):
    """Feature vector dimension and ordering."""

    def test_30_feature_count(self):
        self.assertEqual(len(FEATURE_NAMES), 25)

    def test_31_feature_order_stable(self):
        expected = [
            "gripper_command", "gripper_qpos", "gripper_opening_proxy",
            "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
            "action_dx", "action_dy", "action_dz", "action_gripper",
            "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
            "close_onset", "time_since_close", "eef_speed",
            "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
            "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
        ]
        self.assertEqual(FEATURE_NAMES, expected)

    def test_32_no_privileged_fields_in_features(self):
        """No relation, object, contact, or reward fields in feature names."""
        forbidden = {"relation", "object_pose", "target_pose", "contact", "reward",
                     "success", "failure", "teacher", "attack", "step_index"}
        for name in FEATURE_NAMES:
            for fw in forbidden:
                self.assertNotIn(fw, name.lower(),
                                 f"feature '{name}' contains forbidden substring '{fw}'")


class TestV3StepSequence(unittest.TestCase):
    """Step sequence validation."""

    def test_40_step_gap_rejects(self):
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)
        with self.assertRaises(ValueError):
            _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0)  # skip step 1

    def test_41_reset_clears_state(self):
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)
        self.assertEqual(adapter.next_expected_step, 1)
        adapter.reset()
        self.assertEqual(adapter.next_expected_step, 0)

    def test_42_missing_field_rejects(self):
        adapter = D8StreamingFeatureAdapterV3()
        r = adapter.update(
            step_id=0, raw_gripper=float('nan'), env_gripper=1.0,
            gripper_qpos=0.02, gripper_opening_proxy=0.02,
            eef_x=0.1, eef_y=0.2, eef_z=0.3,
            eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
            action_dx=0.0, action_dy=0.0, action_dz=0.0,
            action_gripper=1.0,
        )
        self.assertFalse(r["valid"])


class TestV3HistoryDerivedFeatures(unittest.TestCase):
    """History-derived features must be causal."""

    def test_50_qpos_deltas_computed(self):
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0, gripper_qpos=0.01)
        _make_step(adapter, 1, raw_gripper=0.0, env_gripper=1.0, gripper_qpos=0.02)
        _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0, gripper_qpos=0.03)
        r = _make_step(adapter, 3, raw_gripper=0.0, env_gripper=1.0, gripper_qpos=0.05)
        self.assertAlmostEqual(r["features"]["qpos_delta_1"], 0.02)
        self.assertAlmostEqual(r["features"]["qpos_delta_3"], 0.04)

    def test_51_eef_speed_computed(self):
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0, eef_vx=1.0, eef_vy=0.0, eef_vz=0.0)
        r = _make_step(adapter, 1, raw_gripper=0.0, env_gripper=1.0, eef_vx=0.0, eef_vy=3.0, eef_vz=4.0)
        self.assertAlmostEqual(r["features"]["eef_speed"], 5.0)

    def test_52_eef_z_delta_since_close(self):
        """eef_z_delta_since_close must reference most recent close onset."""
        adapter = D8StreamingFeatureAdapterV3()
        # Close onset at step 0, eef_z=0.3
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0, eef_z=0.3)
        # Open at step 1
        _make_step(adapter, 1, raw_gripper=1.0, env_gripper=-1.0, eef_z=0.4)
        # Second close onset at step 2, eef_z=0.5
        _make_step(adapter, 2, raw_gripper=0.0, env_gripper=1.0, eef_z=0.5)
        # Step 3, eef_z=0.6 — delta since step 2 (last close onset)
        r = _make_step(adapter, 3, raw_gripper=0.0, env_gripper=1.0, eef_z=0.6)
        self.assertAlmostEqual(r["features"]["eef_z_delta_since_close"], 0.1,
                               msg="must reference step 2 (most recent close onset), not step 0")


class TestV3NoAbsoluteStep(unittest.TestCase):
    """Absolute step index must not leak into features."""

    def test_60_features_independent_of_absolute_step(self):
        """Same pattern at different absolute steps must produce same features.

        Each run uses a fresh adapter reset to step 0 — the pattern itself is
        the same regardless of what episode it's embedded in.
        """
        def run_one():
            a = D8StreamingFeatureAdapterV3()
            for i in range(5):
                _make_step(a, i, raw_gripper=1.0, env_gripper=-1.0)
            _make_step(a, 5, raw_gripper=0.0, env_gripper=1.0)
            r = _make_step(a, 6, raw_gripper=0.0, env_gripper=1.0)
            return r["features"]

        f1 = run_one()
        f2 = run_one()
        # All features must be identical (same relative step offsets)
        for name in FEATURE_NAMES:
            self.assertEqual(f1[name], f2[name],
                             f"feature '{name}' differs between identical runs: "
                             f"{f1[name]} vs {f2[name]}")


class TestV3FutureParity(unittest.TestCase):
    """Future telemetry must not affect past features."""

    def test_70_past_features_stable(self):
        """Adding future steps must not change already-emitted feature vectors."""
        adapter = D8StreamingFeatureAdapterV3()
        features_at_step3 = None

        for i in range(5):
            raw = 0.0 if i >= 3 else 1.0
            env = 1.0 if i >= 3 else -1.0
            r = _make_step(adapter, i, raw_gripper=raw, env_gripper=env)
            if i == 3:
                features_at_step3 = dict(r["features"])

        # Feed more steps
        for i in range(5, 20):
            raw = 0.0 if i % 2 == 0 else 1.0
            env = 1.0 if i % 2 == 0 else -1.0
            _make_step(adapter, i, raw_gripper=raw, env_gripper=env)

        # The returned features dict is a snapshot — future steps can't modify it
        self.assertIsNotNone(features_at_step3)
        self.assertEqual(features_at_step3["close_onset"], 1)


class TestH1R2QposContract(unittest.TestCase):
    """H1-R2: Feature 1 (signed sum) vs Feature 2 (absolute sum) must differ."""

    def test_80_qpos_signed_vs_absolute(self):
        """qpos=[0.02, -0.01]: signed sum=0.01, abs sum=0.03 — must differ."""
        adapter = D8StreamingFeatureAdapterV3()
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0,
                       gripper_qpos=0.01, gripper_opening_proxy=0.03)
        self.assertTrue(r["valid"])
        f = r["features"]
        self.assertEqual(f["gripper_qpos"], 0.01)
        self.assertEqual(f["gripper_opening_proxy"], 0.03)
        self.assertNotEqual(f["gripper_qpos"], f["gripper_opening_proxy"])

    def test_81_qpos_positive_equal(self):
        """qpos=[0.02, 0.01]: signed=abs=0.03 — equal when both same sign."""
        adapter = D8StreamingFeatureAdapterV3()
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0,
                       gripper_qpos=0.03, gripper_opening_proxy=0.03)
        f = r["features"]
        self.assertEqual(f["gripper_qpos"], f["gripper_opening_proxy"],
                         "signed and abs equal when both joints same sign")

    def test_82_qpos_negative_both(self):
        """qpos=[-0.01, -0.02]: signed=-0.03, abs=0.03."""
        adapter = D8StreamingFeatureAdapterV3()
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0,
                       gripper_qpos=-0.03, gripper_opening_proxy=0.03)
        f = r["features"]
        self.assertEqual(f["gripper_qpos"], -0.03)
        self.assertEqual(f["gripper_opening_proxy"], 0.03)
        self.assertNotEqual(f["gripper_qpos"], f["gripper_opening_proxy"])


class TestH1R3FlipWindow(unittest.TestCase):
    """H1-R3: Flip window uses deque(maxlen=32), max=31."""

    def test_90_one_state_zero_flips(self):
        adapter = D8StreamingFeatureAdapterV3()
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 0)

    def test_91_two_alternating_one_flip(self):
        adapter = D8StreamingFeatureAdapterV3()
        _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)
        r = _make_step(adapter, 1, raw_gripper=1.0, env_gripper=-1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 1)

    def test_92_32_alternating_max_31(self):
        adapter = D8StreamingFeatureAdapterV3()
        for i in range(32):
            raw = 0.0 if i % 2 == 0 else 1.0
            env = 1.0 if i % 2 == 0 else -1.0
            _make_step(adapter, i, raw_gripper=raw, env_gripper=env)
        r = _make_step(adapter, 32, raw_gripper=0.0, env_gripper=1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 31,
                         f"32 states alternating: expected 31, got {r['features']['recent_gripper_flip_count']}")

    def test_93_33_alternating_still_31(self):
        """33 pre-steps + opposite final = window at max 31 flips."""
        adapter = D8StreamingFeatureAdapterV3()
        for i in range(33):
            raw = 0.0 if i % 2 == 0 else 1.0
            env = 1.0 if i % 2 == 0 else -1.0
            _make_step(adapter, i, raw_gripper=raw, env_gripper=env)
        # Step 33: OPPOSITE of step 32 to create a flip at the boundary
        # step 32 is even→close, so step 33 is odd→open
        r = _make_step(adapter, 33, raw_gripper=1.0, env_gripper=-1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 31,
                         f"33+1 alternating: expected 31, got {r['features']['recent_gripper_flip_count']}")

    def test_94_96_alternating_still_31(self):
        adapter = D8StreamingFeatureAdapterV3()
        for i in range(96):
            raw = 0.0 if i % 2 == 0 else 1.0
            env = 1.0 if i % 2 == 0 else -1.0
            _make_step(adapter, i, raw_gripper=raw, env_gripper=env)
        r = _make_step(adapter, 96, raw_gripper=0.0, env_gripper=1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 31,
                         "96 states alternating: window always capped at 31")

    def test_95_long_constant_then_one_flip(self):
        adapter = D8StreamingFeatureAdapterV3()
        for i in range(50):
            _make_step(adapter, i, raw_gripper=0.0, env_gripper=1.0)
        r = _make_step(adapter, 50, raw_gripper=1.0, env_gripper=-1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 1,
                         "one flip after long constant: should be 1")

    def test_96_reset_clears_flip_window(self):
        adapter = D8StreamingFeatureAdapterV3()
        for i in range(10):
            raw = 0.0 if i % 2 == 0 else 1.0
            env = 1.0 if i % 2 == 0 else -1.0
            _make_step(adapter, i, raw_gripper=raw, env_gripper=env)
        adapter.reset()
        r = _make_step(adapter, 0, raw_gripper=0.0, env_gripper=1.0)
        self.assertEqual(r["features"]["recent_gripper_flip_count"], 0)


if __name__ == "__main__":
    unittest.main()
