import unittest

from src.gripper_attack.c2g_teacher_v2_contact_identity import (
    analyze_contact_pairs,
    canonicalize_mujoco_name,
)


class ContactIdentityTests(unittest.TestCase):
    def test_suffix_canonicalization_preserves_instance(self):
        self.assertEqual(canonicalize_mujoco_name("cream_cheese_1_link0_visual"), "cream_cheese_1")
        self.assertEqual(canonicalize_mujoco_name("cream_cheese_1_collision"), "cream_cheese_1")
        self.assertEqual(canonicalize_mujoco_name("alphabet_soup_1"), "alphabet_soup_1")

    def test_bilateral_contact_maps_to_one_object(self):
        result = analyze_contact_pairs([
            ("robot0_left_finger_collision", "cream_cheese_1_link0_visual"),
            ("robot0_right_finger_collision", "cream_cheese_1_collision"),
        ], object_names=["cream_cheese_1"])
        self.assertEqual(result.contacted_objects, ("cream_cheese_1",))
        self.assertTrue(result.left_finger_contact)
        self.assertTrue(result.right_finger_contact)
        self.assertTrue(result.bilateral_grasp_candidate)

    def test_multiple_objects_are_explicitly_ambiguous(self):
        result = analyze_contact_pairs([
            ("robot0_left_finger_collision", "milk_1_collision"),
            ("robot0_right_finger_collision", "butter_1_visual"),
        ], object_names=["milk_1", "butter_1"])
        self.assertEqual(set(result.contacted_objects), {"milk_1", "butter_1"})
        self.assertEqual(result.ambiguity_reason, "MULTIPLE_SIMULTANEOUS_CONTACTED_OBJECTS")

    def test_static_receptacle_and_finger_finger_contacts_are_not_objects(self):
        result = analyze_contact_pairs([
            ("robot0_left_finger_collision", "table"),
            ("robot0_right_finger_collision", "basket_1_geom"),
            ("robot0_left_finger_collision", "robot0_right_finger_collision"),
            ("robot0_left_finger_collision", "floor"),
        ], object_names=["milk_1"], receptacle_names=["basket_1"])
        self.assertFalse(result.contacted_objects)
        self.assertFalse(result.bilateral_grasp_candidate)
        self.assertEqual(result.ambiguity_reason, "NO_OBJECT_CONTACT")

    def test_composite_component_maps_to_declared_instance(self):
        result = analyze_contact_pairs([
            ("robot0_left_finger_collision", "alphabet_soup_1_cap_collision"),
        ], object_names=["alphabet_soup_1"])
        self.assertEqual(result.contacted_objects, ("alphabet_soup_1",))
        self.assertEqual(result.ambiguity_reason, "UNILATERAL_OBJECT_CONTACT")

    def test_duplicate_declarations_are_deduplicated_not_ambiguous(self):
        result = analyze_contact_pairs([
            ("robot0_left_finger_collision", "basket_1_visual"),
            ("robot0_right_finger_collision", "basket_1_collision"),
        ], object_names=["basket_1", "basket_1"], receptacle_names=["basket_1"])
        self.assertEqual(result.contacted_objects, ("basket_1",))
        self.assertTrue(result.bilateral_grasp_candidate)
        self.assertEqual(result.ambiguity_reason, "")


if __name__ == "__main__":
    unittest.main()
