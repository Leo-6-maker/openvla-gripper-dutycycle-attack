"""Test R8Z1 exact prefix derivation logic."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.multisuite_detector.audit_c2g_r8z1_semantic_prefix_closure import (
    compute_expected_prefix, compare_rows,
)


def _make_steps(n: int, success_at: int | None = None) -> list[dict]:
    steps = []
    for i in range(n):
        row = {
            "step": i,
            "features_25d": [float(i + j) for j in range(25)],
            "clean_policy_intent_9d": [0.1 * i] * 9,
            "clean_action_raw_7d": [0.01 * i] * 7,
            "applied_action_7d": [0.01 * i] * 7,
            "env_check_success_after_step": (i == success_at),
            "reward_after_step": 0.0,
            "done_after_step": False,
        }
        steps.append(row)
    return steps


class ComputeExpectedPrefixTests(unittest.TestCase):
    def test_spatial_retains_0_to_219(self):
        steps = _make_steps(300)
        prefix, success, reason = compute_expected_prefix(steps, 220)
        self.assertEqual(len(prefix), 220)
        self.assertFalse(success)
        self.assertIn("MAX_POLICY_STEPS", reason)

    def test_object_retains_0_to_279(self):
        steps = _make_steps(300)
        prefix, success, reason = compute_expected_prefix(steps, 280)
        self.assertEqual(len(prefix), 280)

    def test_goal_retains_0_to_299(self):
        steps = _make_steps(300)
        prefix, success, reason = compute_expected_prefix(steps, 300)
        self.assertEqual(len(prefix), 300)

    def test_early_success_truncates(self):
        steps = _make_steps(300, success_at=45)
        prefix, success, reason = compute_expected_prefix(steps, 220)
        self.assertEqual(len(prefix), 46)  # 0..45 inclusive
        self.assertTrue(success)
        self.assertEqual(reason, "ENV_CHECK_SUCCESS")

    def test_success_beyond_horizon_not_in_prefix(self):
        steps = _make_steps(300, success_at=250)
        prefix, success, reason = compute_expected_prefix(steps, 220)
        self.assertEqual(len(prefix), 220)  # truncated at horizon
        self.assertFalse(success)  # success not within prefix

    def test_late_success_not_in_spatial_prefix(self):
        steps = _make_steps(300, success_at=225)
        prefix, success, _ = compute_expected_prefix(steps, 220)
        self.assertFalse(success)
        self.assertEqual(len(prefix), 220)

    def test_source_too_short_fails(self):
        steps = _make_steps(100)
        with self.assertRaises(ValueError):
            compute_expected_prefix(steps, 220)

    def test_source_short_but_early_success_passes(self):
        steps = _make_steps(200, success_at=50)
        prefix, success, _ = compute_expected_prefix(steps, 220)
        self.assertEqual(len(prefix), 51)
        self.assertTrue(success)


class CompareRowsTests(unittest.TestCase):
    def test_identical_rows(self):
        a = {"step": 0, "value": [1.0, 2.0], "name": "test"}
        b = {"step": 0, "value": [1.0, 2.0], "name": "test"}
        self.assertEqual(compare_rows(a, b), [])

    def test_scalar_mismatch(self):
        a = {"step": 0}
        b = {"step": 1}
        diffs = compare_rows(a, b)
        self.assertTrue(any("step" in d for d in diffs))

    def test_list_len_mismatch(self):
        a = {"vec": [1, 2, 3]}
        b = {"vec": [1, 2]}
        diffs = compare_rows(a, b)
        self.assertTrue(any("len" in d for d in diffs))

    def test_list_element_mismatch(self):
        a = {"vec": [1.0, 2.0, 3.0]}
        b = {"vec": [1.0, 2.0, 4.0]}
        diffs = compare_rows(a, b)
        self.assertTrue(any("vec[2]" in d for d in diffs))

    def test_extra_field_in_derived(self):
        a = {"step": 0}
        b = {"step": 0, "extra": True}
        diffs = compare_rows(a, b)
        self.assertTrue(any("extra" in d for d in diffs))

    def test_25d_mismatch_detected(self):
        a = {"features_25d": [0.0] * 25}
        b = {"features_25d": [0.0] * 24 + [1.0]}
        diffs = compare_rows(a, b)
        self.assertTrue(any("features_25d" in d for d in diffs))

    def test_action_mismatch_detected(self):
        a = {"applied_action_7d": [0.0] * 7}
        b = {"applied_action_7d": [1.0] * 7}
        diffs = compare_rows(a, b)
        self.assertTrue(any("applied_action_7d" in d for d in diffs))


if __name__ == "__main__":
    unittest.main()
