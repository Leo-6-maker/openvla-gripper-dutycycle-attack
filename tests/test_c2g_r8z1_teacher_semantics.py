"""Test R8Z1 Teacher temporal semantics analysis."""
import unittest

from tools.multisuite_detector.audit_c2g_r8z1_semantic_prefix_closure import (
    analyze_teacher_semantics,
)


class TeacherSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.sem = analyze_teacher_semantics()

    def test_student_input_is_strictly_causal(self):
        self.assertEqual(self.sem["student_input_temporality"], "STRICTLY_CAUSAL")

    def test_teacher_is_offline_oracle(self):
        self.assertEqual(self.sem["teacher_supervision_mode"],
                         "OFFLINE_PRIVILEGED_PREFIX_ORACLE")

    def test_burst_uses_future_context(self):
        self.assertTrue(self.sem["burst_label_uses_future_context"])

    def test_burst_lookahead_is_9(self):
        self.assertEqual(self.sem["burst_max_lookahead_steps"], 9)

    def test_no_post_horizon_context(self):
        self.assertFalse(self.sem["teacher_uses_post_official_horizon_context"])

    def test_no_future_student_input(self):
        self.assertFalse(self.sem["teacher_uses_future_student_input"])

    def test_no_attack_outcome(self):
        self.assertFalse(self.sem["teacher_uses_attack_outcome"])

    def test_contact_persistence_uses_no_future_context(self):
        self.assertFalse(self.sem["contact_label_uses_future_context"])

    def test_contact_persistence_steps_2(self):
        self.assertEqual(self.sem["contact_persistence_steps"], 2)

    def test_uses_future_step_field_deprecated(self):
        self.assertEqual(
            self.sem["uses_future_step_for_teacher_field"]["classification"],
            "DEPRECATED_AMBIGUOUS_FIELD",
        )

    def test_within_prefix_future_true(self):
        self.assertTrue(self.sem["teacher_uses_within_prefix_future_context"])

    def test_max_lookahead_is_9(self):
        self.assertEqual(self.sem["teacher_max_required_lookahead_steps"], 9)

    def test_burst_feasible_needs_9_future_steps(self):
        """y_burst_feasible at step t requires observing steps t..t+9 (10 steps total)"""
        self.assertEqual(self.sem["burst_max_lookahead_steps"], 9)

    def test_contact_no_cross_episode_lookahead(self):
        self.assertIn("same contact block", self.sem["contact_persistence_mechanism"])

    def test_uses_future_step_field_value_is_false(self):
        self.assertFalse(self.sem["uses_future_step_for_teacher_field"]["value"])


if __name__ == "__main__":
    unittest.main()
