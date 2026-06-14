from __future__ import annotations

import json

from scripts.stageb.audit_m3_v2_pgd_trajectory_candidates import (
    EXPECTED_DELTA0_SHA,
    EXPECTED_FINAL_ARM_MATCH,
    EXPECTED_FINAL_DELTA_SHA,
    EXPECTED_FINAL_MARGIN,
    EXPECTED_FINAL_PROCESSOR_SHA,
    EXPECTED_FINAL_TOKENS,
    classify_feasible_intermediate,
    extract_offline_telemetry,
    validate_reconstruction,
)


def test_extract_offline_telemetry_marks_missing_best_margin():
    debug = {
        "true_pgd": {
            "clean_generated_arm_prefix_token_ids": [1, 2, 3, 4, 5, 6],
            "target_token_logratio_margin_trajectory": [-0.25, 3.5],
            "generated_arm_prefix_trajectory": [[1, 2, 3, 4, 5, 6], [1, 2, 9, 4, 5, 6]],
            "gradient_norm_trajectory": [
                {"l1": 1.0, "l2": 2.0, "linf": 3.0},
                {"l1": 4.0, "l2": 5.0, "linf": 6.0},
            ],
        }
    }
    rows = extract_offline_telemetry(debug)
    assert rows[0]["iteration"] == 0
    assert rows[0]["surrogate_best_competitor_margin"] == "NOT_RECORDED"
    assert rows[0]["arm_match_count"] == 6
    assert rows[1]["arm_match_count"] == 5


def test_feasible_intermediate_requires_token_margin_and_arm_gate():
    rows = [
        {"iteration": 0, "official_gripper_token": 31744, "official_target31744_margin": 9.0, "arm_prefix_match_count": 6},
        {"iteration": 1, "official_gripper_token": 31744, "official_target31744_margin": 6.0, "arm_prefix_match_count": 6},
        {"iteration": 2, "official_gripper_token": 31744, "official_target31744_margin": 7.0, "arm_prefix_match_count": 4},
    ]
    assert classify_feasible_intermediate(rows) == "NO_FEASIBLE_INTERMEDIATE"
    rows.append({"iteration": 3, "official_gripper_token": 31744, "official_target31744_margin": 7.0, "arm_prefix_match_count": 5})
    assert classify_feasible_intermediate(rows) == "FEASIBLE_INTERMEDIATE_EXISTS"


def test_validate_reconstruction_accepts_expected_terminal_artifact():
    rows = [
        {
            "iteration": 0,
            "delta_sha256": EXPECTED_DELTA0_SHA,
        },
        {
            "iteration": 20,
            "delta_sha256": EXPECTED_FINAL_DELTA_SHA,
            "processor_input_sha256": EXPECTED_FINAL_PROCESSOR_SHA,
            "official_tokens": json.dumps(EXPECTED_FINAL_TOKENS),
            "official_target31744_margin": EXPECTED_FINAL_MARGIN,
            "arm_prefix_match_count": EXPECTED_FINAL_ARM_MATCH,
        },
    ]
    status, issues = validate_reconstruction(rows)
    assert status == "RECONSTRUCTION_VALID"
    assert issues == []


def test_validate_reconstruction_rejects_final_hash_mismatch():
    rows = [
        {"iteration": 0, "delta_sha256": EXPECTED_DELTA0_SHA},
        {
            "iteration": 20,
            "delta_sha256": "bad",
            "processor_input_sha256": EXPECTED_FINAL_PROCESSOR_SHA,
            "official_tokens": json.dumps(EXPECTED_FINAL_TOKENS),
            "official_target31744_margin": EXPECTED_FINAL_MARGIN,
            "arm_prefix_match_count": EXPECTED_FINAL_ARM_MATCH,
        },
    ]
    status, issues = validate_reconstruction(rows)
    assert status == "RECONSTRUCTION_INVALID"
    assert "final delta hash mismatch" in issues
