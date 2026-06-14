from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (
    MAIN_CONDITIONS,
    compare_surrogate_official,
    load_config,
    write_csv,
)


CONFIG = Path("configs/m3_step78_true_pgd_31744.yaml")
LOGRATIO_V2_CONFIG = Path("configs/m3_step78_true_pgd_31744_logratio_v2.yaml")
LOGRATIO_ARM_V3_CONFIG = Path("configs/m3_step78_true_pgd_31744_logratio_arm_v3.yaml")


def test_step78_config_is_fixed_to_preregistered_conditions():
    cfg = load_config(CONFIG)
    assert cfg["input"]["task"] == "tomato_sauce"
    assert cfg["input"]["state_id"] == 0
    assert cfg["input"]["absolute_step"] == 78
    assert cfg["attack_optimizer"]["strict_route"] is True
    assert cfg["attack_optimizer"]["allow_fallback"] is False
    assert cfg["attack_optimizer"]["method"] == "token_prefix_pgd"
    assert cfg["attack_optimizer"]["objective"] == "autoregressive_prefix_gripper_target_token_cw_v1"
    assert cfg["attack_optimizer"]["target_token_id"] == 31744
    assert cfg["attack_optimizer"]["target_execution_class"] == "CLIP_MEDIATED_OPEN"
    assert cfg["attack_optimizer"]["surrogate_score_path"] == "cached_autoregressive_generate_v1"
    assert cfg["conditions"] == MAIN_CONDITIONS


def test_preflight_comparison_distinguishes_match_and_mismatch():
    surrogate = {
        "target_token_score": 4.0,
        "target_minus_best_competitor_margin": -0.25,
    }
    official = {
        "target_stats": {
            "target_token_score": 4.0,
            "target_minus_best_competitor_margin": -0.25,
        }
    }
    assert (
        compare_surrogate_official(surrogate, official, tolerance=1e-6)
        == "SURROGATE_OFFICIAL_SCORE_PATH_MATCH"
    )

    official["target_stats"]["target_minus_best_competitor_margin"] = 2.0
    assert (
        compare_surrogate_official(surrogate, official, tolerance=1e-6)
        == "SURROGATE_OFFICIAL_SCORE_PATH_MISMATCH"
    )


def test_write_csv_preserves_declared_schema(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(path, [{"b": 2, "a": 1, "extra": 3}], ["a", "b"])
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["a", "b"]
        rows = list(reader)
    assert rows == [{"a": "1", "b": "2"}]


def test_config_does_not_enable_libero_rollout_result_modes():
    cfg = load_config(CONFIG)
    assert "full_window" not in cfg.get("stage", "").lower()
    assert "rollout" not in cfg.get("stage", "").lower()
    assert "legacy" not in cfg["conditions"]
    assert cfg["controls"]["shuffled_grad_control"] == "single_pgd20_trajectory"
    assert "shuffled_grad_count" not in cfg["controls"]


def test_logratio_v2_config_keeps_target_and_requires_cached_path():
    cfg = load_config(LOGRATIO_V2_CONFIG)
    assert cfg["input"]["task"] == "tomato_sauce"
    assert cfg["input"]["absolute_step"] == 78
    assert cfg["attack_optimizer"]["target_token_id"] == 31744
    assert cfg["attack_optimizer"]["target_execution_class"] == "CLIP_MEDIATED_OPEN"
    assert cfg["attack_optimizer"]["objective"] == "autoregressive_prefix_gripper_target_token_logratio_v2"
    assert cfg["attack_optimizer"]["surrogate_score_path"] == "cached_autoregressive_generate_v1"
    assert cfg["controls"]["rand20_selection_metric"] == "surrogate_target_objective_margin"
    assert cfg["conditions"] == MAIN_CONDITIONS


def test_logratio_arm_v3_config_keeps_fixed_frame_and_adds_arm_penalty():
    cfg = load_config(LOGRATIO_ARM_V3_CONFIG)
    assert cfg["input"]["task"] == "tomato_sauce"
    assert cfg["input"]["state_id"] == 0
    assert cfg["input"]["absolute_step"] == 78
    assert cfg["attack_optimizer"]["target_token_id"] == 31744
    assert cfg["attack_optimizer"]["target_execution_class"] == "CLIP_MEDIATED_OPEN"
    assert cfg["attack_optimizer"]["epsilon"] == 0.023529411764705882
    assert cfg["attack_optimizer"]["objective"] == "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
    assert cfg["attack_optimizer"]["surrogate_score_path"] == "cached_autoregressive_generate_v1"
    assert cfg["attack_optimizer"]["arm_preserve_weight"] == 0.5
    assert cfg["attack_optimizer"]["arm_gate_min_match_count"] == 5
    assert cfg["conditions"] == MAIN_CONDITIONS
