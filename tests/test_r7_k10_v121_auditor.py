"""Focused tests for the independent R7 K10 V1.2.1 artifact auditor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "detector_v4"))
from audit_k10_v121_artifact import K, audit_label_episode, recompute_start


def row(step: int, *, window_id: str = "candidate:0", student_valid: bool = True,
        known_mask: bool = True, release_valid: bool = True,
        regrasp_valid: bool = True, bitmask: int = 1,
        burst: bool = False) -> dict:
    return {
        "step": step,
        "episode_key": "libero_object/task_00/state_00",
        "candidate_close": True,
        "known_mask": known_mask,
        "student_valid": student_valid,
        "window_id": window_id,
        "critical_t": True,
        "burst_feasible_t": burst,
        "is_feasible_start": burst,
        "component_bitmask": bitmask,
        "release_risk_valid": release_valid,
        "regrasp_risk_valid": regrasp_valid,
        "teacher_reason_code": "critical",
    }


def valid_episode(n: int = 12) -> list[dict]:
    rows = [row(i) for i in range(n)]
    for i in range(n - K + 1):
        rows[i]["burst_feasible_t"] = True
        rows[i]["is_feasible_start"] = True
    return rows


def test_valid_episode_recomputes_all_starts():
    rows = valid_episode(12)
    result = audit_label_episode(rows, "libero_object/task_00/state_00")
    assert result["feasible_start_count"] == 3
    assert result["has_feasible_k10"] is True


def test_segment_crossing_is_detected_even_when_generator_flag_is_true():
    rows = valid_episode(12)
    rows[5]["window_id"] = "candidate:1"
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_student_invalid_inside_window_is_detected():
    rows = valid_episode(12)
    rows[5]["student_valid"] = False
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_unknown_inside_window_is_detected():
    rows = valid_episode(12)
    rows[5]["known_mask"] = False
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_release_validity_false_inside_window_is_detected():
    rows = valid_episode(12)
    rows[5]["release_risk_valid"] = False
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_regrasp_validity_false_inside_window_is_detected():
    rows = valid_episode(12)
    rows[5]["regrasp_risk_valid"] = False
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_zero_component_bitmask_inside_window_is_detected():
    rows = valid_episode(12)
    rows[5]["component_bitmask"] = 0
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_tampered_false_negative_is_detected():
    rows = valid_episode(12)
    rows[0]["burst_feasible_t"] = False
    rows[0]["is_feasible_start"] = False
    with pytest.raises(ValueError, match="burst recomputation mismatch"):
        audit_label_episode(rows, "libero_object/task_00/state_00")


def test_recompute_rejects_out_of_bounds():
    rows = valid_episode(12)
    assert recompute_start(rows, 3) is False
