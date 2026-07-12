import unittest

from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    derive_official_prefix,
    rebuild_teacher_labels,
)


def metadata():
    return {
        "episode_key": "libero_spatial/task_0/ep_0",
        "parent_key": "libero_spatial/task_0/ep_0",
        "suite": "libero_spatial",
        "task_index": 0,
        "state_id": 0,
        "cohort": "DETECTOR_TRAIN",
        "split": "train",
        "mechanism_type": "pick_place_transfer",
        "object_declarations": ["milk", "ketchup"],
        "receptacle_declarations": ["basket"],
        "structured_goal_metadata": {
            "target_objects": ["milk"],
            "target_receptacles": ["basket"],
        },
        "gripper_command_semantics": "positive_is_close",
        "official_horizon": 220,
    }


def contact(entity):
    return [
        ["robot0_left_finger_collision", f"{entity}_collision"],
        ["robot0_right_finger_collision", f"{entity}_collision"],
    ]


def teacher_rows(count=230, critical_start=215):
    rows = []
    for step in range(count):
        positive = step >= critical_start
        rows.append(
            {
                "step": step,
                "contact_pairs": contact("milk" if positive else "ketchup"),
                "gripper_command": 1.0,
                "object_relative_lift": 0.03,
                "near_target": False,
                "active_subgoal_index": 0,
                "env_check_success_after_step": False,
                "done_after_step": False,
                "reward_after_step": 0.0,
            }
        )
    return rows


class TeacherRebuildTests(unittest.TestCase):
    def test_future_critical_rows_cannot_make_tail_burst_feasible(self):
        source = teacher_rows()
        prefix = derive_official_prefix(source, official_horizon=220)
        labels = rebuild_teacher_labels(prefix.rows, metadata())
        tail = [row for row in labels if row["step"] >= 215]
        self.assertEqual(len(tail), 5)
        self.assertTrue(all(row["y_gripper_critical_window"] for row in tail))
        self.assertTrue(all(not row["y_burst_feasible"] for row in tail))
        self.assertTrue(all(not row["y_attack_start_B"] for row in tail))

    def test_b10_marks_only_complete_prefix_bursts(self):
        source = teacher_rows(count=220, critical_start=210)
        labels = rebuild_teacher_labels(source, metadata())
        feasible = [row["step"] for row in labels if row["y_burst_feasible"]]
        self.assertEqual(feasible, [210])
        self.assertEqual(
            [row["step"] for row in labels if row["y_attack_start_B"]], [210]
        )

    def test_unknown_is_not_converted_to_negative(self):
        meta = metadata()
        meta.pop("gripper_command_semantics")
        labels = rebuild_teacher_labels(teacher_rows(count=3, critical_start=0), meta)
        self.assertTrue(all(row["label_known_mask"] is False for row in labels))
        for row in labels:
            self.assertIsNone(row["y_gripper_critical_window"])
            self.assertIsNone(row["y_manipulation_progress_active"])
            self.assertIsNone(row["y_attack_start_B"])

    def test_active_goal_event_index_is_carried_from_same_step(self):
        labels = rebuild_teacher_labels(teacher_rows(count=2, critical_start=0), metadata())
        self.assertEqual([row["active_goal_event_index"] for row in labels], [0, 0])

    def test_outcome_and_future_metadata_are_not_visible_to_teacher(self):
        meta = metadata()
        meta.update(
            {
                "clean_success_observed": True,
                "late_success_in_extended_source": True,
                "uses_attack_outcome": False,
            }
        )
        labels = rebuild_teacher_labels(teacher_rows(count=2, critical_start=0), meta)
        self.assertEqual(len(labels), 2)
        self.assertTrue(all(row["uses_attack_outcome"] is False for row in labels))


if __name__ == "__main__":
    unittest.main()
