import unittest

from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets


class TargetResolutionTests(unittest.TestCase):
    def test_single_object_to_receptacle(self):
        result = resolve_task_targets({
            "objects": ["cream_cheese_1"],
            "receptacles": ["basket_1"],
            "goal_predicates": [{"predicate": "in", "args": ["cream_cheese_1", "basket_1"]}],
        })
        self.assertEqual(result.resolved_target_objects, ("cream_cheese_1",))
        self.assertEqual(result.resolved_receptacles, ("basket_1",))
        self.assertEqual(result.reason_code, "RESOLVED_STRUCTURED")

    def test_object_to_site_and_duplicate_instance_identity(self):
        result = resolve_task_targets({
            "objects": ["alphabet_soup_1", "alphabet_soup_2"],
            "sites": ["left_plate_site"],
            "goal_predicates": [("at", "alphabet_soup_2", "left_plate_site")],
        })
        self.assertEqual(result.resolved_target_objects, ("alphabet_soup_2",))
        self.assertEqual(result.resolved_sites, ("left_plate_site",))

    def test_conjunction_and_ordered_subgoals_preserve_multiple_objects(self):
        result = resolve_task_targets({
            "objects": ["milk_1", "butter_1"],
            "receptacles": ["basket_1"],
            "goal_predicates": [
                ("in", "milk_1", "basket_1"),
                ("in", "butter_1", "basket_1"),
            ],
            "ordered_subgoals": [
                ("in", "milk_1", "basket_1"),
                ("in", "butter_1", "basket_1"),
            ],
        })
        self.assertEqual(result.resolved_target_objects, ("milk_1", "butter_1"))
        self.assertEqual(len(result.ordered_subgoals), 2)

    def test_ambiguous_explicit_targets_are_not_arbitrarily_selected(self):
        result = resolve_task_targets({"valid_target_objects": ["milk_1", "butter_1"]})
        self.assertEqual(result.resolved_target_objects, ("milk_1", "butter_1"))
        self.assertEqual(result.reason_code, "AMBIGUOUS_MULTIPLE_TARGETS")
        self.assertLess(result.resolution_confidence, 1.0)

    def test_missing_target_metadata(self):
        result = resolve_task_targets({"objects": ["milk_1"]})
        self.assertEqual(result.reason_code, "TARGET_METADATA_MISSING")
        self.assertFalse(result.resolved_target_objects)

    def test_structured_metadata_wins_but_records_language_conflict(self):
        result = resolve_task_targets({
            "objects": ["milk_1", "butter_1"],
            "receptacles": ["basket_1"],
            "task_language": "put the butter in the basket",
            "goal_predicates": [("in", "milk_1", "basket_1")],
        })
        self.assertEqual(result.resolved_target_objects, ("milk_1",))
        self.assertIn("LANGUAGE_STRUCTURED_CONFLICT", result.ambiguities)

    def test_language_fallback_has_reduced_confidence(self):
        result = resolve_task_targets({
            "objects": ["milk_1"],
            "receptacles": ["basket_1"],
            "task_language": "put milk 1 in basket 1",
        })
        self.assertEqual(result.resolution_source, "language_fallback")
        self.assertLess(result.resolution_confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
