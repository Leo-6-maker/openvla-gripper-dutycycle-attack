import unittest

from n5.phase2_labels.run_v23_dev_pilot import RunnerHold, run_episode
from n5.phase2_labels.c3_t0_semantic_contract import ContractError, FALSE, TRUE, UNKNOWN
from n5.phase3_student.c3_g_predicate_evaluator import load_contract


def _case(step, position, expected_identity=None):
    return {
        "episode_id": "libero_10/task_00/state_00",
        "step": step,
        "predicate": "In",
        "expected_identity": expected_identity or {
            "episode_id": "libero_10/task_00/state_00",
            "step": step,
            "object_id": "obj_1",
            "target_id": "region_1",
        },
        "object": {
            "id": "obj_1", "role": "MANIPULATED_OBJECT",
            "half_extents": [0.1, 0.1, 0.1],
            "pose": {"pos": list(position), "quat": [1.0, 0.0, 0.0, 0.0]},
        },
        "target": {
            "id": "region_1", "role": "REGION_TARGET",
            "half_extents": [1.0, 1.0, 1.0],
            "pose": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        },
        "object_position": list(position),
        "previous_object_position": [0.0, 0.0, 0.0],
    }


def _row(step, qpos, contacts):
    return {
        "step": step,
        "robot0_gripper_qpos": list(qpos),
        "robot0_eef_pos": [float(step), 0.0, 0.0],
        "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
        "mujoco_contact_pairs": contacts,
    }


class TestV23Runner(unittest.TestCase):
    def test_runner_uses_physical_prefix_and_contract_heads(self):
        rows = [
            _row(0, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
            _row(1, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
            _row(2, [0.4, 0.4], []),
        ]
        geometry = {i: _case(i, [0.0, 0.0, 0.0]) for i in range(3)}
        result = run_episode(rows, geometry, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(result["step_count"], 3)
        self.assertEqual(result["forbidden_reads"], 0)
        self.assertEqual(result["unknown_to_false"], 0)
        self.assertEqual(result["steps"][1]["heads"]["physical_criticality"]["value"], TRUE)

    def test_unknown_geometry_is_not_negative(self):
        rows = [_row(0, [0.05, 0.05], [["obj_1", "gripper0_finger1"]])]
        result = run_episode(rows, {}, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(result["steps"][0]["heads"]["physical_criticality"]["value"], UNKNOWN)
        self.assertEqual(result["unknown_to_false"], 0)

    def test_forbidden_physical_injection_rejected(self):
        rows = [_row(0, [0.05, 0.05], [["obj_1", "gripper0_finger1"]])]
        rows[0]["task_success"] = False
        with self.assertRaises(ContractError):
            run_episode(rows, {}, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)

    def test_noncontiguous_steps_rejected(self):
        rows = [_row(1, [0.05, 0.05], [])]
        with self.assertRaises(RunnerHold):
            run_episode(rows, {}, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)

    def test_protocol_horizon_is_not_observed_episode_length(self):
        short = [_row(0, [0.05, 0.05], []), _row(1, [0.05, 0.05], [])]
        longer = short + [_row(2, [0.05, 0.05], [])]
        a = run_episode(short, {}, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
        b = run_episode(longer, {}, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(a["steps"][0]["protocol_steps_remaining"], 519)
        self.assertEqual(b["steps"][0]["protocol_steps_remaining"], 519)
        self.assertEqual(a["steps"][0]["observed_future_steps_available"], 1)
        self.assertEqual(b["steps"][0]["observed_future_steps_available"], 2)

    def test_multi_relation_geometry_is_preserved(self):
        rows = [_row(0, [0.05, 0.05], [["obj_1", "gripper0_finger1"]])]
        first = _case(0, [0.0, 0.0, 0.0])
        second = _case(0, [0.0, 0.0, 0.0])
        second["relation_index"] = 1
        second["expected_identity"] = {**second["expected_identity"], "object_id": "obj_2", "target_id": "region_2"}
        second["object"]["id"] = "obj_2"
        second["target"]["id"] = "region_2"
        result = run_episode(rows, {0: {"relations": [first, second]}},
                             "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(len(result["steps"][0]["geometry"]["relations"]), 2)


if __name__ == "__main__":
    unittest.main()
