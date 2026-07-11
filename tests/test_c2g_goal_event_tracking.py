import tempfile
import unittest
from pathlib import Path

from src.gripper_attack.c2g_bddl_metadata import parse_bddl_task_metadata
from src.gripper_attack.c2g_clean_event_tracking import (
    goal_event_bindings,
    joint_hint_from_interaction_site,
    select_active_goal_event,
)
from src.gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets


def parse(text: str):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "task.bddl"
        path.write_text(text, encoding="utf-8")
        return parse_bddl_task_metadata(path)


class OfficialLiberoGoalParsingTests(unittest.TestCase):
    def test_object_goal_region_owner_and_destination(self):
        metadata = parse(
            """
            (define (problem demo)
              (:regions
                (contain_region (:target basket_1))
              )
              (:fixtures floor - floor)
              (:objects
                alphabet_soup_1 - alphabet_soup
                basket_1 - basket
              )
              (:goal (And (In alphabet_soup_1 basket_1_contain_region)))
            )
            """
        )
        self.assertEqual(
            metadata["object_declarations"], ["alphabet_soup_1", "basket_1"]
        )
        self.assertEqual(metadata["site_declarations"], ["basket_1_contain_region"])
        self.assertEqual(
            metadata["region_owner_by_site"]["basket_1_contain_region"], "basket_1"
        )
        resolution = resolve_task_targets(metadata)
        self.assertEqual(resolution.resolved_target_objects, ("alphabet_soup_1",))
        self.assertEqual(resolution.resolved_sites, ("basket_1_contain_region",))
        self.assertEqual(
            resolution.resolved_destination_entities, ("basket_1_contain_region",)
        )
        self.assertEqual(infer_clean_mechanism_type(metadata, resolution=resolution), "pick_place_transfer")

    def test_spatial_context_language_does_not_invalidate_exact_goal(self):
        metadata = parse(
            """
            (define (problem demo)
              (:language pick the bowl between the plate and ramekin and place it on the plate)
              (:fixtures main_table - table)
              (:objects
                bowl_1 bowl_2 - bowl
                plate_1 - plate
                ramekin_1 - ramekin
              )
              (:goal (And (On bowl_1 plate_1)))
            )
            """
        )
        metadata["task_language"] = (
            "pick the bowl between the plate and ramekin and place it on the plate"
        )
        resolution = resolve_task_targets(metadata)
        self.assertEqual(resolution.resolved_target_objects, ("bowl_1",))
        self.assertEqual(resolution.resolved_destination_entities, ("plate_1",))
        self.assertIn("LANGUAGE_STRUCTURED_CONFLICT", resolution.ambiguities)
        self.assertEqual(infer_clean_mechanism_type(metadata, resolution=resolution), "pick_place_transfer")

    def test_open_drawer_region_resolves_to_fixture_owner(self):
        metadata = parse(
            """
            (define (problem demo)
              (:regions
                (middle_region (:target wooden_cabinet_1))
              )
              (:fixtures
                main_table - table
                wooden_cabinet_1 - wooden_cabinet
              )
              (:objects bowl_1 - bowl)
              (:goal (And (Open wooden_cabinet_1_middle_region)))
            )
            """
        )
        resolution = resolve_task_targets(metadata)
        self.assertEqual(
            resolution.resolved_manipulable_entities, ("wooden_cabinet_1",)
        )
        bindings = goal_event_bindings(resolution)
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0].target_entity, "wooden_cabinet_1")
        self.assertEqual(
            bindings[0].interaction_site, "wooden_cabinet_1_middle_region"
        )
        self.assertEqual(
            joint_hint_from_interaction_site(
                bindings[0].target_entity, bindings[0].interaction_site
            ),
            "middle",
        )
        self.assertEqual(infer_clean_mechanism_type(metadata, resolution=resolution), "articulated_object")


class MultiTargetEventTrackingTests(unittest.TestCase):
    def setUp(self):
        self.metadata = parse(
            """
            (define (problem demo)
              (:regions (contain_region (:target basket_1)))
              (:fixtures living_room_table - table)
              (:objects
                alphabet_soup_1 - alphabet_soup
                tomato_sauce_1 - tomato_sauce
                basket_1 - basket
              )
              (:goal
                (And
                  (In alphabet_soup_1 basket_1_contain_region)
                  (In tomato_sauce_1 basket_1_contain_region)
                )
              )
            )
            """
        )
        self.resolution = resolve_task_targets(self.metadata)
        self.bindings = goal_event_bindings(self.resolution)

    def test_two_targets_have_distinct_bindings(self):
        self.assertEqual(
            self.resolution.resolved_target_objects,
            ("alphabet_soup_1", "tomato_sauce_1"),
        )
        self.assertEqual(len(self.bindings), 2)
        self.assertEqual(
            infer_clean_mechanism_type(self.metadata, resolution=self.resolution),
            "multi_object_transfer",
        )

    def test_current_bilateral_contact_selects_tomato_subgoal(self):
        event = select_active_goal_event(
            [
                ["robot0_finger_joint1_tip", "tomato_sauce_1_collision"],
                ["robot0_finger_joint2_tip", "tomato_sauce_1_collision"],
            ],
            self.bindings,
        )
        self.assertTrue(event["active_target_known"])
        self.assertTrue(event["active_target_bilateral_contact"])
        self.assertEqual(event["active_target_entity"], "tomato_sauce_1")
        self.assertEqual(event["active_destination_entity"], "basket_1_contain_region")

    def test_simultaneous_two_target_contact_remains_unknown(self):
        event = select_active_goal_event(
            [
                ["robot0_finger_joint1_tip", "alphabet_soup_1_collision"],
                ["robot0_finger_joint2_tip", "alphabet_soup_1_collision"],
                ["robot0_finger_joint1_tip", "tomato_sauce_1_collision"],
                ["robot0_finger_joint2_tip", "tomato_sauce_1_collision"],
            ],
            self.bindings,
        )
        self.assertFalse(event["active_target_known"])
        self.assertEqual(event["active_target_reason"], "MULTIPLE_CONTACTED_GOAL_TARGETS")


if __name__ == "__main__":
    unittest.main()
