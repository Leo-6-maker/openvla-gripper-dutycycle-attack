import unittest

from tools.multisuite_detector.audit_c2g_r8z_ogs_full1500 import (
    assert_public_report_sealed,
)
from tools.multisuite_detector.build_c2g_r8z_ogs_official_views import _train_health
from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    TARGET_SUITES,
    TRAIN_COHORT,
    select_canary_rows,
)


def summary(value):
    return {
        "row_count": 10,
        "known_step_count": value,
        "critical_active_step_count": value,
        "start_positive_episode": bool(value),
        "burst_feasible_episode": bool(value),
        "fully_known_hard_negative_episode": not bool(value),
        "release_safe_step_count": value,
        "target_grounding_known_step_count": value,
        "reason_code_counts": {"X": value},
    }


class SealedCohortTests(unittest.TestCase):
    def test_canary_is_train_only_and_source_key_ranked(self):
        rows = []
        for suite in TARGET_SUITES:
            for task in range(3):
                for state in range(3):
                    rows.append(
                        {
                            "suite": suite,
                            "task_index": task,
                            "state_id": state,
                            "parent_key": f"{suite}/task_{task}/state_{state}",
                            "cohort": TRAIN_COHORT,
                            "split": "train",
                        }
                    )
            rows.append(
                {
                    "suite": suite,
                    "task_index": 9,
                    "state_id": 9,
                    "parent_key": f"{suite}/sealed",
                    "cohort": "DETECTOR_TEST_WITHIN_TASK",
                    "split": "test",
                }
            )
        selected = select_canary_rows(rows)
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(row["cohort"] == TRAIN_COHORT for row in selected))
        for suite in TARGET_SUITES:
            suite_rows = [row for row in selected if row["suite"] == suite]
            self.assertEqual(len(suite_rows), 4)
            self.assertEqual(len({row["task_index"] for row in suite_rows}), 2)

    def test_train_health_ignores_nontrain_label_metrics(self):
        rows = [
            {
                "cohort": TRAIN_COHORT,
                "task_index": 0,
                "summary": summary(1),
            },
            {
                "cohort": "DETECTOR_TEST_WITHIN_TASK",
                "task_index": 0,
                "summary": summary(999),
            },
        ]
        health = _train_health(rows)
        self.assertEqual(health["episode_count"], 1)
        self.assertEqual(health["known_step_count"], 1)
        self.assertEqual(health["critical_active_step_count"], 1)

    def test_public_report_rejects_nontrain_metrics(self):
        assert_public_report_sealed(
            {
                "status": "PASS",
                "train_only_label_health": {"known_step_fraction": 0.5},
                "nontrain_metrics_exposed": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "exposes"):
            assert_public_report_sealed(
                {
                    "status": "PASS",
                    "test_positive_rate": 0.5,
                    "nontrain_metrics_exposed": False,
                }
            )


if __name__ == "__main__":
    unittest.main()

