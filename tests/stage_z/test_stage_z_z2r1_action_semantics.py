from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.stage_v_m3_5_phase_classifier import classify_trajectory  # noqa: E402
from stage_z_preparation.action_semantics import (  # noqa: E402
    MODEL_M0,
    MODEL_M1,
    MODEL_M2,
    classify_trajectory_with_action_semantics,
    validate_action_pair,
)


def _row(step: int, raw: list[float], final: list[float], *, contact: bool = False, z: float = 0.0, distance: float = 0.3) -> dict:
    return {
        "step": step,
        "clean_record_valid": True,
        "clean_terminal": False,
        "remaining_horizon": 40,
        "object_identity": "cube_1",
        "object_position": [0.0, 0.0, z],
        "eef_position": [0.0, 0.0, z + distance],
        "object_eef_distance_m": distance,
        "object_gripper_contact": contact,
        "object_support_contact": not contact,
        "contact_telemetry_valid": True,
        "raw_gripper": raw[-1],
        "env_gripper": final[-1],
        "raw_action_7d": raw,
        "env_action_7d": final,
        "model_boundary": True,
    }


def test_m2_uses_official_clip_and_fails_closed():
    raw = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837]
    final = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837]
    assert validate_action_pair(MODEL_M2, raw, final)["accepted"] is True
    assert validate_action_pair(MODEL_M2, raw, [0.9, *final[1:]])["accepted"] is False
    assert validate_action_pair(MODEL_M2, [float("nan"), *raw[1:]], final)["accepted"] is False
    assert validate_action_pair(MODEL_M2, raw[:-1], final[:-1])["accepted"] is False


def test_m0_m1_wrapper_is_byte_behavior_equivalent_to_historical_classifier():
    rows = [
        _row(0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        _row(1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], contact=True, distance=0.08),
    ]
    expected = classify_trajectory(rows)
    for family in (MODEL_M0, MODEL_M1):
        actual, diagnostics = classify_trajectory_with_action_semantics(rows, family, sys.modules["gripper_attack.stage_v_m3_5_phase_classifier"])
        assert actual == expected
        assert diagnostics["invalid"] == 0


def test_m2_valid_boundary_reuses_frozen_geometry_and_invalid_row_abstains():
    rows = [
        _row(0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837]),
        _row(1, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837], contact=True, distance=0.08),
        _row(2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837], contact=True, z=0.02, distance=0.03),
    ]
    labels, diagnostics = classify_trajectory_with_action_semantics(
        rows, MODEL_M2, sys.modules["gripper_attack.stage_v_m3_5_phase_classifier"]
    )
    assert [item["clean_only_phase_label"] for item in labels] == ["PRE_CONTACT", "CONTACT_MANIPULATION", "ENGAGED_LIFT"]
    assert diagnostics["invalid"] == 0

    rows[1]["env_action_7d"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9986837]
    labels, diagnostics = classify_trajectory_with_action_semantics(
        rows, MODEL_M2, sys.modules["gripper_attack.stage_v_m3_5_phase_classifier"]
    )
    assert labels[1]["clean_only_phase_label"] == "UNKNOWN"
    assert labels[1]["abstain_reason"] == "CLEAN_RECORD_INVALID_OR_TERMINAL"
    assert diagnostics["invalid"] == 1


def test_historical_phase_classifier_bytes_are_unchanged():
    path = ROOT / "src/gripper_attack/stage_v_m3_5_phase_classifier.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "e3c48886777a45cfe7d2b9942bb eeb6ebc992f29568846180d31d064b839fbe6".replace(" ", "")
