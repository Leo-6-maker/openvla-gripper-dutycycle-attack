import unittest

from n5.phase2_labels.run_v23_dev_pilot import RunnerHold, load_semantic_contract, run_episode
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
    def test_semantic_contract_v11_is_frozen_and_thresholded(self):
        contract = load_semantic_contract()
        self.assertEqual(contract["schema"], "C3_T0_TEACHER_SEMANTIC_CONTRACT_V1_1")
        self.assertGreater(contract["quality_thresholds"]["comotion"]["cosine_threshold"], 0.0)

    def test_runner_uses_physical_prefix_and_contract_heads(self):
        rows = [
            _row(0, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
            _row(1, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
            _row(2, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
            _row(3, [0.4, 0.4], []),
        ]
        geometry = {i: _case(i, [float(i), 0.0, 0.0]) for i in range(4)}
        result = run_episode(rows, geometry, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(result["step_count"], 4)
        self.assertEqual(result["forbidden_reads"], 0)
        self.assertEqual(result["unknown_to_false"], 0)
        self.assertEqual(result["steps"][1]["heads"]["physical_criticality"]["value"], UNKNOWN)
        self.assertEqual(result["steps"][2]["heads"]["physical_criticality"]["value"], TRUE)

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
        self.assertEqual(a["steps"][0]["heads"]["k10_feasible"]["value"], TRUE)
        self.assertEqual(
            [item["heads"] for item in a["steps"]],
            [item["heads"] for item in b["steps"][:2]],
        )

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

    def test_multi_relation_conjunction_false_and_unknown_are_not_true(self):
        rows = [_row(0, [0.05, 0.05], [])]
        true_case = _case(0, [0.0, 0.0, 0.0])
        false_case = _case(0, [2.0, 0.0, 0.0])
        unknown_case = _case(0, [0.0, 0.0, 0.0])
        unknown_case["object"]["pose"]["pos"] = [float("nan"), 0.0, 0.0]
        for relations, expected in (((true_case, false_case), FALSE),
                                    ((true_case, unknown_case), UNKNOWN),
                                    ((false_case, unknown_case), FALSE)):
            result = run_episode(rows, {0: {"relations": list(relations)}},
                                 "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2)
            self.assertEqual(result["steps"][0]["physical_components"]["placement"], expected)

    def test_released_state_requires_open_qpos_and_no_object_contact(self):
        geometry = {0: _case(0, [0.0, 0.0, 0.0])}
        still_contact = run_episode(
            [_row(0, [0.4, 0.4], [["obj_1", "gripper0_finger1"]])],
            geometry, "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2,
        )
        self.assertEqual(still_contact["steps"][0]["physical_components"]["released_state"], FALSE)
        released = run_episode(
            [_row(0, [0.4, 0.4], [])], geometry,
            "libero_10/task_00/state_00", load_contract(), {"obj_1"}, 0.2,
        )
        self.assertEqual(released["steps"][0]["physical_components"]["released_state"], TRUE)

    def test_k10_uses_protocol_horizon_not_observed_suffix(self):
        rows_short = [_row(0, [0.05, 0.05], [])]
        rows_long = rows_short + [_row(1, [0.05, 0.05], []), _row(2, [0.05, 0.05], [])]
        geometry_short = {0: _case(0, [2.0, 0.0, 0.0])}
        geometry_long = {i: _case(i, [2.0, 0.0, 0.0]) for i in range(3)}
        short = run_episode(rows_short, geometry_short, "libero_10/task_00/state_00",
                            load_contract(), {"obj_1"}, 0.2)
        long = run_episode(rows_long, geometry_long, "libero_10/task_00/state_00",
                           load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(short["steps"][0]["heads"]["k10_feasible"]["value"], TRUE)
        self.assertEqual(long["steps"][0]["heads"]["k10_feasible"]["value"], TRUE)
        self.assertFalse(short["steps"][0]["audit_observation_mask"])
        self.assertFalse(long["steps"][0]["audit_observation_mask"])

    def test_slip_is_distinct_from_contact_loss(self):
        rows = [
            _row(0, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
            _row(1, [0.05, 0.05], [["obj_1", "gripper0_finger1"]]),
        ]
        geometry = {0: _case(0, [0.0, 0.0, 0.0]), 1: _case(1, [0.0, 0.0, 0.0])}
        result = run_episode(rows, geometry, "libero_10/task_00/state_00",
                             load_contract(), {"obj_1"}, 0.2)
        self.assertEqual(result["steps"][1]["physical_components"]["slip"], TRUE)
        self.assertEqual(result["steps"][1]["physical_components"]["contact_loss"], FALSE)


if __name__ == "__main__":
    unittest.main()
