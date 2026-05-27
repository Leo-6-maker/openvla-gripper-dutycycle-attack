import pytest

from src.utils.proprio_causal_student import (
    CATEGORICAL_FEATURES,
    EVAL_ONLY_COLUMNS,
    NUMERIC_FEATURES,
    assert_feature_whitelist,
)


def test_feature_whitelist_excludes_eval_identity_visual_and_outcomes():
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert_feature_whitelist(features)
    for forbidden in [
        "teacher_window_start",
        "teacher_window_end",
        "teacher_anchor_step",
        "task_id",
        "state_id",
        "run_id",
        "episode_key",
        "object_pose",
        "target_pose",
        "object_to_target_distance",
        "attack_outcome",
        "manual_outcome",
        "visual_feature_path",
        "image_path",
    ]:
        assert forbidden not in features
    for col in EVAL_ONLY_COLUMNS:
        assert col not in features


def test_forbidden_feature_rejected():
    with pytest.raises(ValueError):
        assert_feature_whitelist(NUMERIC_FEATURES + ["teacher_window_start"])

