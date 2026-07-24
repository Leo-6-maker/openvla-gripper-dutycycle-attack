"""Test frozen official suite horizons."""
import unittest

from tools.multisuite_detector.c2g_official_suite_horizons import (
    CANONICAL_SUITES,
    OFFICIAL_DUMMY_WAIT_STEPS,
    OFFICIAL_MAX_POLICY_STEPS,
    official_max_policy_steps,
    validate_all_canonical_horizons,
    validate_official_suite_horizon,
)


class HorizonConstantsTests(unittest.TestCase):
    def test_spatial_220(self):
        self.assertEqual(OFFICIAL_MAX_POLICY_STEPS["libero_spatial"], 220)

    def test_object_280(self):
        self.assertEqual(OFFICIAL_MAX_POLICY_STEPS["libero_object"], 280)

    def test_goal_300(self):
        self.assertEqual(OFFICIAL_MAX_POLICY_STEPS["libero_goal"], 300)

    def test_l10_520(self):
        self.assertEqual(OFFICIAL_MAX_POLICY_STEPS["libero_10"], 520)

    def test_dummy_wait_10(self):
        self.assertEqual(OFFICIAL_DUMMY_WAIT_STEPS, 10)

    def test_canonical_suites_complete(self):
        self.assertEqual(
            CANONICAL_SUITES,
            {"libero_spatial", "libero_object", "libero_goal", "libero_10"},
        )


class ValidateHorizonTests(unittest.TestCase):
    def test_l10_520_accepted(self):
        validate_official_suite_horizon("libero_10", 520)

    def test_goal_300_accepted(self):
        validate_official_suite_horizon("libero_goal", 300)

    def test_object_280_accepted(self):
        validate_official_suite_horizon("libero_object", 280)

    def test_spatial_220_accepted(self):
        validate_official_suite_horizon("libero_spatial", 220)

    def test_l10_300_rejected(self):
        with self.assertRaises(ValueError):
            validate_official_suite_horizon("libero_10", 300)

    def test_l10_400_rejected(self):
        with self.assertRaises(ValueError):
            validate_official_suite_horizon("libero_10", 400)

    def test_l10_521_rejected(self):
        with self.assertRaises(ValueError):
            validate_official_suite_horizon("libero_10", 521)

    def test_unknown_suite_rejected(self):
        with self.assertRaises(ValueError):
            validate_official_suite_horizon("libero_unknown", 300)


class ValidateAllCanonicalTests(unittest.TestCase):
    def test_all_correct_accepted(self):
        validate_all_canonical_horizons({
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
        })

    def test_missing_suite_rejected(self):
        with self.assertRaises(ValueError):
            validate_all_canonical_horizons({
                "libero_spatial": 220,
                "libero_object": 280,
                "libero_goal": 300,
            })

    def test_wrong_horizon_rejected(self):
        with self.assertRaises(ValueError):
            validate_all_canonical_horizons({
                "libero_spatial": 220,
                "libero_object": 280,
                "libero_goal": 300,
                "libero_10": 300,
            })


class OfficialMaxPolicyStepsTests(unittest.TestCase):
    def test_returns_520_for_l10(self):
        self.assertEqual(official_max_policy_steps("libero_10"), 520)

    def test_returns_300_for_goal(self):
        self.assertEqual(official_max_policy_steps("libero_goal"), 300)

    def test_unknown_raises_key_error(self):
        with self.assertRaises(KeyError):
            official_max_policy_steps("unknown")


if __name__ == "__main__":
    unittest.main()
