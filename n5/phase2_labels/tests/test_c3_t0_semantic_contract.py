import math
import unittest

from n5.phase2_labels.c3_t0_semantic_contract import (
    ContractError, FALSE, TRUE, UNKNOWN, apply_persistence,
    apply_right_censor, evaluate_heads, gripper_closing_state,
    k10_feasible, physical_criticality, quaternion_equivalent, safe_release,
)


def _physical(**updates):
    row = {
        "physical_known": True, "stable_grasp": True,
        "transport_or_manipulation": False, "placement": False,
        "release": False, "stability": True, "remaining_steps": 12,
        "horizon_known": True, "right_censored": False,
        "safe_release": False, "slip": False, "regrasp": False,
        "contact_loss": False, "gripper_qpos": 0.1,
        "qpos_close_threshold": 0.2,
    }
    row.update(updates)
    return row


class TestC3T0SemanticContract(unittest.TestCase):
    def test_all_five_heads_have_tri_state_mask_and_reason(self):
        result = evaluate_heads(_physical())
        self.assertEqual(set(result), {
            "physical_criticality", "k10_feasible", "safe_release",
            "instability", "gripper_closing_state",
        })
        self.assertTrue(all(set(item) == {"value", "mask", "reason"}
                            for item in result.values()))
        self.assertEqual(result["physical_criticality"]["value"], TRUE)

    def test_safe_release_requires_all_three_components_and_unknown_is_not_false(self):
        self.assertEqual(safe_release(_physical(placement=True, release=True,
                                                 stability=True))["value"], TRUE)
        self.assertEqual(safe_release(_physical(placement=True, release=True,
                                                 stability=False))["value"], FALSE)
        unknown = safe_release(_physical(placement=True, release=None, stability=True))
        self.assertEqual(unknown, {"value": UNKNOWN, "mask": False,
                                   "reason": "SAFE_RELEASE_COMPONENT_UNKNOWN"})

    def test_physical_label_rejects_outcome_and_task_success_leakage(self):
        self.assertEqual(physical_criticality(_physical())["value"], TRUE)
        for forbidden in ("task_success", "terminal", "outcome", "future"):
            with self.assertRaises(ContractError):
                physical_criticality({**_physical(), forbidden: False})

    def test_gripper_closing_uses_only_finite_physical_qpos(self):
        self.assertEqual(gripper_closing_state(_physical(gripper_qpos=0.2))["value"], TRUE)
        self.assertEqual(gripper_closing_state(_physical(gripper_qpos=0.3))["value"], FALSE)
        self.assertEqual(gripper_closing_state(_physical(gripper_qpos=math.nan))["value"], UNKNOWN)
        self.assertEqual(gripper_closing_state(_physical(gripper_qpos=math.inf))["value"], UNKNOWN)

    def test_persistence_and_right_censor_are_conservative(self):
        persisted = apply_persistence([{"value": TRUE}, {"value": TRUE}, {"value": FALSE}])
        self.assertEqual([item["value"] for item in persisted], [UNKNOWN, TRUE, FALSE])
        self.assertEqual(apply_right_censor(persisted[1], 9)["value"], UNKNOWN)
        self.assertEqual(apply_right_censor(persisted[1], 10)["value"], TRUE)

    def test_k10_right_censor_and_known_horizon(self):
        self.assertEqual(k10_feasible(_physical(remaining_steps=10,
                                                 safe_release=False))["value"], TRUE)
        self.assertEqual(k10_feasible(_physical(remaining_steps=9,
                                                 safe_release=False))["value"], FALSE)
        self.assertEqual(k10_feasible(_physical(right_censored=True))["value"], UNKNOWN)

    def test_quaternion_sign_equivalence_and_nonfinite_rejection(self):
        q = (0.5, 0.5, 0.5, 0.5)
        self.assertTrue(quaternion_equivalent(q, tuple(-x for x in q)))
        self.assertFalse(quaternion_equivalent(q, (math.nan, 0.5, 0.5, 0.5)))


if __name__ == "__main__":
    unittest.main()
