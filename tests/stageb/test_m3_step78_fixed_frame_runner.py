from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (
    MAIN_CONDITIONS,
    PANEL_ALL_CAPTURE_FRAMES,
    PANEL_MAIN_FRAMES,
    PANEL_POSITIVE_CONTROL_FRAME,
    V4_CONDITIONS,
    clean_frame_eligibility,
    compare_surrogate_official,
    frame_full_selective_status,
    load_config,
    panel_aggregate_status,
    select_hard_feasible_candidate,
    step78_parity_status,
    validate_panel_seed,
    write_csv,
)


CONFIG = Path("configs/m3_step78_true_pgd_31744.yaml")
LOGRATIO_V2_CONFIG = Path("configs/m3_step78_true_pgd_31744_logratio_v2.yaml")
LOGRATIO_ARM_V3_CONFIG = Path("configs/m3_step78_true_pgd_31744_logratio_arm_v3.yaml")
LOGRATIO_ARM_V4_CONFIG = Path("configs/m3_step78_true_pgd_31744_logratio_arm_v4.yaml")


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


def test_logratio_arm_v4_config_freezes_hard_feasible_selection_protocol():
    cfg = load_config(LOGRATIO_ARM_V4_CONFIG)
    assert cfg["input"]["task"] == "tomato_sauce"
    assert cfg["input"]["state_id"] == 0
    assert cfg["input"]["absolute_step"] == 78
    assert cfg["attack_optimizer"]["target_token_id"] == 31744
    assert cfg["attack_optimizer"]["target_execution_class"] == "CLIP_MEDIATED_OPEN"
    assert cfg["attack_optimizer"]["epsilon"] == 0.023529411764705882
    assert cfg["attack_optimizer"]["num_steps"] == 20
    assert cfg["attack_optimizer"]["step_size"] == 0.0017647058823529412
    assert cfg["attack_optimizer"]["objective"] == "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
    assert cfg["attack_optimizer"]["arm_preserve_weight"] == 0.5
    assert cfg["gates"]["arm_prefix_min_match_count"] == 5
    assert cfg["controls"]["rand21_count"] == 21
    assert cfg["controls"]["shuffled_grad_control"] == "trajectory21_selective"
    assert cfg["conditions"] == V4_CONDITIONS


def test_hard_feasible_selection_filters_arm_and_target_before_margin():
    rows = [
        {
            "candidate_id": 0,
            "arm_prefix_match_count": 6,
            "official_gripper_token": 31872,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 100.0,
            "processor_linf": 0.01,
        },
        {
            "candidate_id": 1,
            "arm_prefix_match_count": 2,
            "official_gripper_token": 31744,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 90.0,
            "processor_linf": 0.01,
        },
        {
            "candidate_id": 2,
            "arm_prefix_match_count": 5,
            "official_gripper_token": 31744,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 3.0,
            "processor_linf": 0.02,
        },
    ]
    selected = select_hard_feasible_candidate(rows, arm_gate_min_match_count=5, target_token_id=31744)
    assert selected is rows[2]


def test_hard_feasible_selection_tie_breaks_linf_then_earlier_candidate():
    rows = [
        {
            "candidate_id": 3,
            "arm_prefix_match_count": 5,
            "official_gripper_token": 31744,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 7.0,
            "processor_linf": 0.02,
        },
        {
            "candidate_id": 2,
            "arm_prefix_match_count": 5,
            "official_gripper_token": 31744,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 7.0,
            "processor_linf": 0.01,
        },
        {
            "candidate_id": 1,
            "arm_prefix_match_count": 5,
            "official_gripper_token": 31744,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 7.0,
            "processor_linf": 0.01,
        },
    ]
    selected = select_hard_feasible_candidate(rows, arm_gate_min_match_count=5, target_token_id=31744)
    assert selected is rows[2]


def test_hard_feasible_selection_does_not_fallback_to_arm_breaking_candidate():
    rows = [
        {
            "candidate_id": 0,
            "arm_prefix_match_count": 4,
            "official_gripper_token": 31744,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 30.0,
            "processor_linf": 0.01,
        },
        {
            "candidate_id": 1,
            "arm_prefix_match_count": 6,
            "official_gripper_token": 31872,
            "score_invariant_status": "PASS",
            "official_target31744_margin": 1.0,
            "processor_linf": 0.01,
        },
    ]
    assert select_hard_feasible_candidate(rows, arm_gate_min_match_count=5, target_token_id=31744) is None


def test_panel_frame_set_freezes_main_denominator_and_positive_control():
    assert PANEL_MAIN_FRAMES == [70, 72, 74, 76, 80, 82, 84, 86]
    assert PANEL_POSITIVE_CONTROL_FRAME == 78
    assert PANEL_ALL_CAPTURE_FRAMES == [70, 72, 74, 76, 78, 80, 82, 84, 86]


def test_clean_already_target_is_ineligible():
    official = {
        "tokens": [1, 2, 3, 4, 5, 6, 31744],
        "gripper_token": 31744,
        "score_invariant": {"tie_aware_pass": True},
    }
    status = clean_frame_eligibility(official)
    assert status["status"] == "CLEAN_ALREADY_TARGET"
    assert status["clean_gripper_token"] == 31744


def test_clean_non_close_non_target_is_ineligible():
    official = {
        "tokens": [1, 2, 3, 4, 5, 6, 12345],
        "gripper_token": 12345,
        "score_invariant": {"tie_aware_pass": True},
    }
    status = clean_frame_eligibility(official)
    assert status["status"] == "CLEAN_NOT_CLOSE"


def _selected(token=31744, arm=6, margin=1.0):
    return {
        "condition_result": "SELECTED",
        "official_gripper_token": token,
        "arm_prefix_match_count": arm,
        "official_target31744_margin": margin,
    }


def test_true_feasible_control_infeasible_is_auto_win_without_paired_margin():
    row = frame_full_selective_status(
        true_row=_selected(margin=10.0),
        rand_row=None,
        shuffled_row=_selected(margin=1.0),
    )
    assert row["status"] == "FRAME_FULL_SELECTIVE_PASS"
    assert row["rand_control_status"] == "CONTROL_INFEASIBLE_AUTO_WIN"
    assert row["rand_paired_margin"] == ""
    assert row["shuffled_paired_margin"] == 9.0


def test_true_infeasible_fails_even_when_controls_infeasible():
    row = frame_full_selective_status(
        true_row=_selected(token=31872, arm=6, margin=10.0),
        rand_row=None,
        shuffled_row=None,
    )
    assert row["status"] == "FRAME_FAIL_TRUE_INFEASIBLE"
    assert row["full_pass"] is False


def test_panel_aggregate_requires_same_frame_full_passes():
    rows = []
    for frame in PANEL_MAIN_FRAMES[:5]:
        rows.append(
            {
                "frame": frame,
                "main_denominator": True,
                "frame_status": "FRAME_FULL_SELECTIVE_PASS",
                "rand_paired_margin": 1.0,
                "shuffled_paired_margin": 1.0,
            }
        )
    for frame in PANEL_MAIN_FRAMES[5:]:
        rows.append(
            {
                "frame": frame,
                "main_denominator": True,
                "frame_status": "FRAME_FAIL_CONTROL_NOT_BEATEN",
                "rand_paired_margin": 1.0,
                "shuffled_paired_margin": 1.0,
            }
        )
    agg = panel_aggregate_status(rows)
    assert agg["panel_status"] == "PANEL_SINGLE_SEED_FAIL"
    assert "fewer_than_6_full_selective_pass_frames" in agg["failure_reasons"]


def test_panel_aggregate_requires_four_finite_paired_frames_per_control():
    rows = []
    for frame in PANEL_MAIN_FRAMES:
        rows.append(
            {
                "frame": frame,
                "main_denominator": True,
                "frame_status": "FRAME_FULL_SELECTIVE_PASS",
                "rand_paired_margin": 1.0 if len(rows) < 3 else "",
                "shuffled_paired_margin": 1.0,
            }
        )
    agg = panel_aggregate_status(rows)
    assert agg["panel_status"] == "PANEL_SINGLE_SEED_FAIL"
    assert "rand_paired_frames_below_4" in agg["failure_reasons"]


def test_panel_aggregate_does_not_allow_replacing_multiple_ineligible_frames():
    rows = []
    for idx, frame in enumerate(PANEL_MAIN_FRAMES):
        rows.append(
            {
                "frame": frame,
                "main_denominator": True,
                "frame_status": "CLEAN_NOT_CLOSE" if idx < 2 else "FRAME_FULL_SELECTIVE_PASS",
                "rand_paired_margin": 1.0,
                "shuffled_paired_margin": 1.0,
            }
        )
    agg = panel_aggregate_status(rows)
    assert agg["panel_status"] == "PANEL_SINGLE_SEED_FAIL"
    assert "too_many_clean_ineligible_frames" in agg["failure_reasons"]


def test_panel_seed_is_frozen_to_85_only():
    validate_panel_seed(85)
    with pytest.raises(SystemExit):
        validate_panel_seed(84)
    with pytest.raises(SystemExit):
        validate_panel_seed(86)


def test_step78_parity_mismatch_stops_positive_control():
    frozen = {
        "raw_image_sha256": "a",
        "processed_tensor_sha256": "b",
        "prompt_token_ids_sha256": "c",
        "clean_exact_7_tokens": "[1,2,3,4,5,6,31872]",
        "clean_arm_prefix": "[1,2,3,4,5,6]",
        "clean_gripper_token": "31872",
    }
    new = dict(frozen)
    assert step78_parity_status(new, frozen) == "POSITIVE_CONTROL_INPUT_MATCH"
    new["processed_tensor_sha256"] = "changed"
    assert step78_parity_status(new, frozen) == "POSITIVE_CONTROL_INPUT_MISMATCH"
