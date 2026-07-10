import unittest

from scripts.stageb.build_c2g_clean_manifests_release import (
    deterministic_order,
    parent_key,
    task_count,
)


class DummySuiteWithProperty:
    n_tasks = 10


class DummySuiteWithMethod:
    def get_num_tasks(self):
        return 7


class DummySuiteWithTasks:
    tasks = [1, 2, 3]


class ReleaseManifestBuilderTests(unittest.TestCase):
    def test_parent_key_has_exactly_five_components(self):
        value = parent_key("libero_object", 2, 17, "eval", 3)
        self.assertEqual(
            value,
            "libero_object/task_2/state_17/eval/episode_003",
        )
        self.assertEqual(len(value.split("/")), 5)

    def test_train_and_eval_parent_names_are_distinct(self):
        train = parent_key("libero_goal", 1, 8, "train", 0)
        evaluation = parent_key("libero_goal", 1, 8, "eval", 0)
        self.assertNotEqual(train, evaluation)

    def test_task_count_supports_common_libero_interfaces(self):
        self.assertEqual(task_count(DummySuiteWithProperty()), 10)
        self.assertEqual(task_count(DummySuiteWithMethod()), 7)
        self.assertEqual(task_count(DummySuiteWithTasks()), 3)

    def test_deterministic_selection_is_stable(self):
        first = deterministic_order("libero_spatial", 5, range(100), 42)
        second = deterministic_order("libero_spatial", 5, range(100), 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))


if __name__ == "__main__":
    unittest.main()
