import unittest

from scripts.stageb.build_c2g_clean_manifests import deterministic_order


class CleanManifestBuilderTests(unittest.TestCase):
    def test_deterministic_order_is_reproducible_and_complete(self):
        states = list(range(50))
        first = deterministic_order("libero_object", 3, states, 42)
        second = deterministic_order("libero_object", 3, states, 42)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), states)
        self.assertEqual(len(set(first)), len(states))

    def test_selection_changes_with_seed_or_task_identity(self):
        states = list(range(50))
        baseline = deterministic_order("libero_object", 3, states, 42)
        different_seed = deterministic_order("libero_object", 3, states, 43)
        different_task = deterministic_order("libero_object", 4, states, 42)
        self.assertNotEqual(baseline, different_seed)
        self.assertNotEqual(baseline, different_task)

    def test_train_and_eval_slices_are_disjoint(self):
        order = deterministic_order("libero_goal", 1, range(100), 7)
        train = set(order[:40])
        evaluation = set(order[40:50])
        self.assertEqual(len(train), 40)
        self.assertEqual(len(evaluation), 10)
        self.assertFalse(train & evaluation)

    def test_input_order_does_not_change_selection(self):
        ascending = deterministic_order("libero_10", 2, range(30), 9)
        descending = deterministic_order("libero_10", 2, reversed(range(30)), 9)
        self.assertEqual(ascending, descending)


if __name__ == "__main__":
    unittest.main()
