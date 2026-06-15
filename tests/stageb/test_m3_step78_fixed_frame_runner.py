from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stageb.run_m3_step78_true_pgd_fixed_frame import (
    MAIN_CONDITIONS,
    PANEL_ALL_CAPTURE_FRAMES,
    PANEL_MAIN_FRAMES,
    PANEL_POSITIVE_CONTROL_FRAME,
    V4_CONDITIONS,
    claim_one_shot_sentinel,
    clean_frame_eligibility,
    compare_surrogate_official,
    frame_full_selective_status,
    frame_status_from_v4_artifacts,
    load_config,
    panel_aggregate_status,
    parse_panel_steps,
    run_panel_seed85,
    select_hard_feasible_candidate,
    step78_parity_status,
    validate_panel_seed,
    validate_manifest_provenance,
    write_csv,
    write_json,
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


def test_panel_seed85_entry_rejects_other_seed_before_work(tmp_path):
    args = SimpleNamespace(
        attack_seed=84,
        panel_steps="",
        output_dir=str(tmp_path),
        frozen_step78_manifest="",
    )
    with pytest.raises(SystemExit):
        run_panel_seed85(args, load_config(LOGRATIO_ARM_V4_CONFIG))
    assert not any(tmp_path.iterdir())


def test_panel_steps_must_match_exact_frozen_set():
    assert parse_panel_steps("") == PANEL_ALL_CAPTURE_FRAMES
    assert parse_panel_steps(",".join(str(x) for x in PANEL_ALL_CAPTURE_FRAMES)) == PANEL_ALL_CAPTURE_FRAMES
    with pytest.raises(SystemExit):
        parse_panel_steps("70,72,74,76,78,80,82,84")
    with pytest.raises(SystemExit):
        parse_panel_steps("70,72,74,76,78,80,82,84,84")


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


def test_aggregate_rejects_duplicate_or_wrong_main_frame_set():
    rows = []
    for idx in range(8):
        rows.append(
            {
                "frame": 70 if idx < 2 else PANEL_MAIN_FRAMES[idx],
                "main_denominator": True,
                "frame_status": "FRAME_FULL_SELECTIVE_PASS",
                "rand_paired_margin": 1.0,
                "shuffled_paired_margin": 1.0,
            }
        )
    agg = panel_aggregate_status(rows)
    assert agg["panel_status"] == "PANEL_SINGLE_SEED_FAIL"
    assert "wrong_main_frame_set" in agg["failure_reasons"]


def test_manifest_provenance_fails_closed_on_dirty_or_missing_gpu(tmp_path):
    row = {
        "dirty_status": "DIRTY: M file.py",
        "gpu_query": "0, GPU-uuid, Test, 0 MiB, 1 MiB, 0 %, 30",
        "model_fingerprint": '{"ok": true}',
    }
    write_csv(tmp_path / "m3_step78_manifest.csv", [row], list(row.keys()))
    write_csv(tmp_path / "m3_artifact_hash_manifest.csv", [{"file": "x", "size_bytes": 1, "sha256": "abc"}], ["file", "size_bytes", "sha256"])
    with pytest.raises(RuntimeError, match="dirty_status"):
        validate_manifest_provenance(tmp_path)

    row["dirty_status"] = "CLEAN"
    row["gpu_query"] = "NVIDIA_SMI_UNAVAILABLE"
    write_csv(tmp_path / "m3_step78_manifest.csv", [row], list(row.keys()))
    with pytest.raises(RuntimeError, match="gpu_query"):
        validate_manifest_provenance(tmp_path)

    row["gpu_query"] = "0, GPU-uuid, Test, 0 MiB, 1 MiB, 0 %, 30"
    row["model_fingerprint"] = "PENDING_MODEL_LOAD"
    write_csv(tmp_path / "m3_step78_manifest.csv", [row], list(row.keys()))
    with pytest.raises(RuntimeError, match="model_fingerprint"):
        validate_manifest_provenance(tmp_path)


def test_one_shot_sentinel_prevents_rerun(tmp_path):
    claim_one_shot_sentinel(tmp_path, stage="m3_panel_seed85", seed=85)
    with pytest.raises(RuntimeError, match="one-shot sentinel"):
        claim_one_shot_sentinel(tmp_path, stage="m3_panel_seed85", seed=85)


def test_frame_status_from_real_artifacts_uses_clean_gate_and_candidate_counts(tmp_path):
    clean_dir = tmp_path / "capture" / "step70"
    run_dir = tmp_path / "frames" / "step70"
    clean_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    write_json(
        clean_dir / "clean_generation_step70.json",
        {
            "official": {
                "tokens": [1, 2, 3, 4, 5, 6, 31872],
                "gripper_token": 31872,
                "score_invariant": {"tie_aware_pass": True},
            }
        },
    )
    selected_rows = [
        _selected(margin=10.0) | {"condition": "TRUE_PGD_TRAJECTORY21_SELECTIVE"},
        _selected(margin=1.0) | {"condition": "RAND21_SELECTIVE"},
        _selected(margin=2.0) | {"condition": "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"},
    ]
    write_csv(run_dir / "m3_v4_selected_results.csv", selected_rows, list(selected_rows[0].keys()))
    candidate_rows = []
    for condition in ["TRUE_PGD_TRAJECTORY21_SELECTIVE", "RAND21_SELECTIVE", "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"]:
        for idx in range(21):
            candidate_rows.append({"condition": condition, "candidate_id": idx})
    write_csv(run_dir / "m3_v4_candidate_audit.csv", candidate_rows, ["condition", "candidate_id"])
    route_rows = []
    for condition in ["TRUE_PGD_TRAJECTORY21_SELECTIVE", "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"]:
        route_rows.append(
            {
                "condition": condition,
                "strict_route": "true",
                "allow_fallback": "false",
                "fallback_used": "false",
                "resolved_adapter_class": "TokenPrefixPGDAttacker",
                "num_backwards": "20",
                "trajectory_candidate_count": "21",
            }
        )
    write_csv(run_dir / "m3_v4_route_audit.csv", route_rows, list(route_rows[0].keys()))
    status = frame_status_from_v4_artifacts(run_dir, step=70, main_denominator=True, clean_dir=clean_dir)
    assert status["frame_status"] == "FRAME_FULL_SELECTIVE_PASS"
    assert status["true_candidate_count"] == 21


def test_frame_status_skips_attack_artifacts_when_clean_ineligible(tmp_path):
    clean_dir = tmp_path / "capture" / "step70"
    run_dir = tmp_path / "frames" / "step70"
    clean_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    write_json(
        clean_dir / "clean_generation_step70.json",
        {
            "official": {
                "tokens": [1, 2, 3, 4, 5, 6, 31744],
                "gripper_token": 31744,
                "score_invariant": {"tie_aware_pass": True},
            }
        },
    )
    status = frame_status_from_v4_artifacts(run_dir, step=70, main_denominator=True, clean_dir=clean_dir)
    assert status["frame_status"] == "CLEAN_ALREADY_TARGET"
    assert status["infra_status"] == "SKIPPED_CLEAN_INELIGIBLE"
