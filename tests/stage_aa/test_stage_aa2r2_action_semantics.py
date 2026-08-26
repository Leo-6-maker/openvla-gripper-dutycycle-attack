from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper  # noqa: E402
from stage_aa.action_semantics_v2 import MODEL_M0, MODEL_M1, MODEL_M2, validate_action_pair  # noqa: E402
from stage_z_preparation.action_semantics import validate_action_pair as validate_action_pair_v1  # noqa: E402


def _action(gripper: float) -> list[float]:
    return [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, gripper]


def test_official_three_state_boundary_and_nextafter_values():
    below = float(np.nextafter(np.float64(0.5), -np.inf))
    above = float(np.nextafter(np.float64(0.5), np.inf))
    probes = [(0.4999999, 1.0, "CLOSE"), (0.5, 0.0, "NEUTRAL_BOUNDARY"), (0.5000001, -1.0, "OPEN"), (below, 1.0, "CLOSE"), (above, -1.0, "OPEN")]
    for raw, expected, state in probes:
        for family in (MODEL_M0, MODEL_M1):
            result = validate_action_pair(family, _action(raw), _action(expected))
            assert result["accepted"] is True
            assert result["semantic_state"] == state
            assert result["expected_final_gripper"] == expected
        assert raw_gripper_to_env_gripper(raw) == expected


def test_exact_threshold_wrong_final_values_fail():
    for family in (MODEL_M0, MODEL_M1):
        for final in (1.0, -1.0):
            result = validate_action_pair(family, _action(0.5), _action(final))
            assert result["accepted"] is False
            assert result["reason"] == "OPENVLA_GRIPPER_MAPPING_MISMATCH"
            assert result["expected_final_gripper"] == 0.0


def test_malformed_and_nonfinite_actions_fail_closed():
    for family in (MODEL_M0, MODEL_M1, MODEL_M2):
        assert validate_action_pair(family, _action(float("nan")), _action(0.0))["accepted"] is False
        assert validate_action_pair(family, _action(0.0)[:-1], _action(0.0))["accepted"] is False
        assert validate_action_pair(family, _action(0.0), _action(float("inf")))["accepted"] is False


def test_m2_clip_behavior_is_preserved():
    raw = [2.0, -2.0, 0.2, 0.0, 0.0, 0.0, -0.9986837]
    final = [1.0, -1.0, 0.2, 0.0, 0.0, 0.0, -0.9986837]
    result = validate_action_pair(MODEL_M2, raw, final)
    assert result["accepted"] is True
    assert validate_action_pair(MODEL_M2, raw, [0.9, *final[1:]])["accepted"] is False


def test_historical_pass_implies_v2_pass_with_same_open_close_meaning():
    for family in (MODEL_M0, MODEL_M1):
        for raw in (-1.0, 0.0, 0.499, 0.501, 1.0, 2.0):
            expected = raw_gripper_to_env_gripper(raw)
            old = validate_action_pair_v1(family, _action(raw), _action(expected))
            new = validate_action_pair(family, _action(raw), _action(expected))
            if old["accepted"]:
                assert new["accepted"] is True
                assert new["semantic_state"] == ("OPEN" if raw > 0.5 else "CLOSE")
                assert math.isclose(new["expected_final_gripper"], expected, rel_tol=0.0, abs_tol=1e-12)
