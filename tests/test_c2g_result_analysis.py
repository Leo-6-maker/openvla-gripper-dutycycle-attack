import unittest

from scripts.stageb.analyze_c2g_matched_load_results import (
    analyze,
    exact_two_sided_binomial,
    paired_comparison,
)


class ResultAnalysisTests(unittest.TestCase):
    def parent(self, key, **success):
        return {"parent_key": key, "success": success}

    def test_exact_binomial_is_symmetric_and_bounded(self):
        self.assertEqual(exact_two_sided_binomial(0, 0), 1.0)
        self.assertEqual(exact_two_sided_binomial(1, 3), exact_two_sided_binomial(3, 1))
        self.assertGreaterEqual(exact_two_sided_binomial(1, 3), 0.0)
        self.assertLessEqual(exact_two_sided_binomial(1, 3), 1.0)

    def test_paired_comparison_uses_only_boolean_pairs(self):
        parents = [
            self.parent("p0", A=True, B=False),
            self.parent("p1", A=False, B=True),
            self.parent("p2", A=True, B=True),
            self.parent("p3", A=None, B=False),
        ]
        result = paired_comparison(parents, "A", "B")
        self.assertEqual(result["paired_n"], 3)
        self.assertEqual(result["first_success_second_failure"], 1)
        self.assertEqual(result["first_failure_second_success"], 1)
        self.assertEqual(result["both_success"], 1)

    def test_pass_audit_produces_timing_objective_and_suite_results(self):
        conditions = {
            "CLEAN": True,
            "DET_GRIPPER_VIS_PGD": False,
            "DET_RANDOM_VIS_ATTACK": True,
            "RANDTIME_GRIPPER_VIS_PGD": True,
            "RANDTIME_RANDOM_VIS_ATTACK": True,
        }
        audit = {
            "status": "PASS_C2G_MATCHED_LOAD_RUN_AUDIT",
            "parents": [
                self.parent("libero_object/task_0/state_0/eval_0", **conditions),
                self.parent("libero_goal/task_0/state_0/eval_0", **conditions),
            ],
            "jobs": [
                {"parent_key": "libero_object/task_0/state_0/eval_0"},
                {"parent_key": "libero_goal/task_0/state_0/eval_0"},
            ],
        }
        denominator = {
            "input_parent_count": 3,
            "included_parent_count": 2,
            "excluded_parent_count": 1,
            "detector_emit_burst_feasible_coverage": 2 / 3,
            "excluded_reason_counts": {"DETECTOR_NO_EMIT": 1},
        }
        result = analyze(audit, denominator)
        self.assertEqual(result["status"], "PASS_C2G_MATCHED_LOAD_RESULT_ANALYSIS")
        self.assertEqual(
            result["paired_comparisons"]["timing_value_gripper_objective"]["paired_n"],
            2,
        )
        self.assertEqual(set(result["per_suite"]), {"libero_object", "libero_goal"})
        self.assertTrue(result["factorial_effects"]["available"])
        self.assertEqual(
            result["detector_coverage_denominator"]["excluded_parent_count"],
            1,
        )

    def test_analysis_rejects_nonpass_audit(self):
        with self.assertRaisesRegex(ValueError, "PASS"):
            analyze({"status": "HOLD", "parents": []})


if __name__ == "__main__":
    unittest.main()
