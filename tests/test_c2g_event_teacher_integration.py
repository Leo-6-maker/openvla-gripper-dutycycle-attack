import unittest

from src.gripper_attack.c2g_clean_window_schema import assert_clean_student_feature_names
from src.gripper_attack.c2g_teacher_v2_contact_identity import analyze_contact_pairs
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)


class EventTeacherIntegrationTests(unittest.TestCase):
    def test_multi_target_current_contact_can_be_known_positive(self):
        metadata = {
            "episode_key": "libero_10/task_0/state_0/seed_42/clean",
            "suite": "libero_10",
            "task_index": 0,
            "mechanism_type": "multi_object_transfer",
            "object_declarations": [
                "alphabet_soup_1",
                "tomato_sauce_1",
                "basket_1",
            ],
            "receptacle_declarations": [],
            "site_declarations": ["basket_1_contain_region"],
            "fixture_declarations": ["living_room_table"],
            "region_owner_by_site": {
                "basket_1_contain_region": "basket_1",
            },
            "goal_predicates": [
                ["in", "alphabet_soup_1", "basket_1_contain_region"],
                ["in", "tomato_sauce_1", "basket_1_contain_region"],
            ],
            "ordered_subgoals": [
                ["in", "alphabet_soup_1", "basket_1_contain_region"],
                ["in", "tomato_sauce_1", "basket_1_contain_region"],
            ],
        }
        rows = [
            {
                "step": step,
                "mujoco_contact_pairs": [
                    ["robot0_finger_joint1_tip", "tomato_sauce_1_collision"],
                    ["robot0_finger_joint2_tip", "tomato_sauce_1_collision"],
                ],
                "clean_close_intent": True,
                "manipulation_progress_active": True,
                "object_relative_lift": 0.03,
                "near_target": False,
                "supported_at_target": False,
                "release_safe": False,
                "active_target_known": True,
                "active_target_entity": "tomato_sauce_1",
                "active_subgoal_index": 1,
            }
            for step in range(12)
        ]
        labels = build_clean_teacher_episode(
            rows,
            metadata,
            thresholds=CleanTeacherThresholds(burst_length=10),
        )
        self.assertTrue(all(row["label_known_mask"] for row in labels))
        self.assertTrue(all(row["y_gripper_critical_window"] for row in labels))
        self.assertEqual(sum(bool(row["y_attack_start_b"]) for row in labels), 1)
        self.assertGreaterEqual(sum(bool(row["y_burst_feasible"]) for row in labels), 1)

    def test_disjoint_multi_target_windows_keep_one_episode_global_start(self):
        metadata = {
            "episode_key": "libero_10/task_0/state_9/train/episode_000",
            "suite": "libero_10",
            "task_index": 0,
            "mechanism_type": "multi_object_transfer",
            "object_declarations": [
                "alphabet_soup_1",
                "tomato_sauce_1",
                "basket_1",
            ],
            "receptacle_declarations": [],
            "site_declarations": ["basket_1_contain_region"],
            "fixture_declarations": ["living_room_table"],
            "region_owner_by_site": {
                "basket_1_contain_region": "basket_1",
            },
            "goal_predicates": [
                ["in", "alphabet_soup_1", "basket_1_contain_region"],
                ["in", "tomato_sauce_1", "basket_1_contain_region"],
            ],
            "ordered_subgoals": [
                ["in", "alphabet_soup_1", "basket_1_contain_region"],
                ["in", "tomato_sauce_1", "basket_1_contain_region"],
            ],
        }

        def positive_row(step, entity, subgoal_index):
            return {
                "step": step,
                "mujoco_contact_pairs": [
                    ["robot0_finger_joint1_tip", f"{entity}_collision"],
                    ["robot0_finger_joint2_tip", f"{entity}_collision"],
                ],
                "clean_close_intent": True,
                "manipulation_progress_active": True,
                "object_relative_lift": 0.03,
                "near_target": False,
                "supported_at_target": False,
                "release_safe": False,
                "active_target_known": True,
                "active_target_entity": entity,
                "active_subgoal_index": subgoal_index,
            }

        rows = [
            positive_row(step, "alphabet_soup_1", 0)
            for step in range(10)
        ]
        rows.append(
            {
                "step": 10,
                "mujoco_contact_pairs": [],
                "clean_close_intent": False,
                "manipulation_progress_active": False,
                "near_target": False,
                "supported_at_target": False,
                "release_safe": False,
                "active_target_known": False,
                "active_target_entity": None,
                "active_subgoal_index": None,
            }
        )
        rows.extend(
            positive_row(step, "tomato_sauce_1", 1)
            for step in range(11, 21)
        )

        labels = build_clean_teacher_episode(
            rows,
            metadata,
            thresholds=CleanTeacherThresholds(burst_length=10),
        )
        starts = [row for row in labels if row["y_attack_start_b"]]
        feasible_steps = [row["step"] for row in labels if row["y_burst_feasible"]]

        self.assertTrue(all(row["label_known_mask"] for row in labels))
        self.assertEqual(feasible_steps, [0, 11])
        self.assertEqual([row["step"] for row in starts], [0])
        self.assertEqual(starts[0]["teacher_reason_code"], "TARGET_CRITICAL_WINDOW_START")
        self.assertFalse(labels[11]["y_attack_start_b"])

    def test_object_and_receptacle_duplicate_role_is_not_contact_ambiguity(self):
        identity = analyze_contact_pairs(
            [
                ["robot0_finger_joint1_tip", "basket_1_collision"],
                ["robot0_finger_joint2_tip", "basket_1_collision"],
            ],
            object_names=["basket_1"],
            receptacle_names=["basket_1"],
        )
        self.assertEqual(identity.contacted_objects, ("basket_1",))
        self.assertTrue(identity.bilateral_grasp_candidate)
        self.assertEqual(identity.ambiguity_reason, "")

    def test_active_event_privileged_fields_are_forbidden_student_inputs(self):
        for name in (
            "active_target_entity",
            "active_subgoal_index",
            "active_destination_entity",
            "goal_event_bindings",
            "contacted_goal_targets",
        ):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                assert_clean_student_feature_names([name])


if __name__ == "__main__":
    unittest.main()
