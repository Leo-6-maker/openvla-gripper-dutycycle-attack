import unittest

import numpy as np

from tools.multisuite_detector.validate_c2g_clean_window_dataset import audit_dataset


def dataset(*, leak=False, missing_test_positive=False):
    # Two episodes per split, with one persistent positive and one negative episode.
    split = []
    episode = []
    step = []
    suite = []
    task = []
    target = []
    known = []
    for split_name in ("train", "val", "test"):
        for kind in ("positive", "negative"):
            key = f"{split_name}_{kind}"
            if leak and split_name == "val" and kind == "negative":
                key = "train_negative"
            values = [0, 1, 1] if kind == "positive" else [0, 0, 0]
            if missing_test_positive and split_name == "test" and kind == "positive":
                values = [0, 0, 0]
            for index, value in enumerate(values):
                split.append(split_name)
                episode.append(key)
                step.append(index)
                suite.append("libero_object")
                task.append(0)
                target.append(value)
                known.append(True)
    n = len(split)
    time_steps = 3
    y = np.zeros((n, time_steps), dtype=np.float32)
    m = np.ones((n, time_steps), dtype=bool)
    y[:, -1] = np.asarray(target, dtype=np.float32)
    return {
        "split": np.asarray(split),
        "episode_key": np.asarray(episode),
        "step": np.asarray(step, dtype=np.int64),
        "suite": np.asarray(suite),
        "task_index": np.asarray(task, dtype=np.int64),
        "y_critical_window": y,
        "m_critical_window": m,
    }


class DatasetTrainabilityTests(unittest.TestCase):
    def test_balanced_dataset_passes(self):
        report = audit_dataset(dataset())
        self.assertEqual(report["status"], "PASS_C2G_DATASET_TRAINABILITY")
        for split_name in ("train", "val", "test"):
            self.assertEqual(
                report["split_reports"][split_name]["triggerable_positive_episode_count"],
                1,
            )

    def test_episode_split_leakage_holds(self):
        report = audit_dataset(dataset(leak=True))
        self.assertEqual(report["status"], "HOLD_C2G_DATASET_TRAINABILITY")
        self.assertEqual(report["episode_split_leakage_count"], 1)

    def test_missing_test_positive_can_be_required_or_diagnostic(self):
        strict = audit_dataset(dataset(missing_test_positive=True), require_test_support=True)
        self.assertEqual(strict["status"], "HOLD_C2G_DATASET_TRAINABILITY")
        diagnostic = audit_dataset(dataset(missing_test_positive=True), require_test_support=False)
        self.assertEqual(diagnostic["status"], "PASS_C2G_DATASET_TRAINABILITY")

    def test_isolated_positive_is_not_triggerable(self):
        value = dataset()
        test_episode = value["episode_key"] == "test_positive"
        current = value["y_critical_window"][:, -1]
        current[test_episode] = np.asarray([0, 1, 0], dtype=np.float32)
        report = audit_dataset(value)
        self.assertEqual(report["status"], "HOLD_C2G_DATASET_TRAINABILITY")
        self.assertEqual(
            report["split_reports"]["test"]["triggerable_positive_episode_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
