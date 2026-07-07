from tools.multisuite_detector.recompute_c5_safety_from_frozen_detector_v1 import summarize_episode_rows


def test_summarize_episode_rows_by_suite_and_split():
    episodes = [
        {"episode_key": "g0", "suite": "libero_goal", "split": "test", "false_trigger": True, "max_score": 0.8},
        {"episode_key": "g1", "suite": "libero_goal", "split": "test", "false_trigger": False, "max_score": 0.2},
        {"episode_key": "o0", "suite": "libero_object", "split": "val", "false_trigger": False, "max_score": 0.1},
        {"episode_key": "s0", "suite": "libero_spatial", "split": "train", "false_trigger": True, "max_score": 0.9},
        {"episode_key": "l0", "suite": "libero_10", "split": "test", "false_trigger": True, "max_score": 0.7},
    ]
    suite_rows, split_rows = summarize_episode_rows(episodes)
    by_suite = {row["suite"]: row for row in suite_rows}
    assert by_suite["libero_goal"]["safety_false_trigger_rate"] == 0.5
    assert by_suite["libero_object"]["safety_false_trigger_rate"] == 0.0
    assert by_suite["libero_spatial"]["role"] == "primary_positive"
    assert by_suite["libero_10"]["role"] == "diagnostic_only"
    by_pair = {(row["suite"], row["split"]): row for row in split_rows}
    assert by_pair[("libero_goal", "test")]["episode_count"] == 2
