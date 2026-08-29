from __future__ import annotations

import importlib.util
from math import isclose
from pathlib import Path

import numpy as np

from stage_ac.eligibility_v2 import classify_calibration_control, evaluate_candidate, scan_candidates


def _row(step: int, *, contact: bool = True, support: bool = False, terminal_after: bool = False) -> dict:
    # Object and EEF move together: absolute separation is 0.08 m, while
    # relative carry displacement is zero.  This is the construct AA2 could
    # not distinguish from a failed 0.04 m absolute-distance gate.
    object_position = [0.08 + step * 0.01, 0.2, 0.3]
    eef_position = [step * 0.01, 0.2, 0.3]
    return {
        "step": step,
        "terminal_before": False,
        "terminal_after": terminal_after,
        "contact_telemetry_valid": True,
        "object_identity": "object",
        "object_position": object_position,
        "eef_position": eef_position,
        "object_eef_distance_m": 0.08,
        "object_gripper_contact": contact,
        "object_support_contact": support,
    }


def _actions(count: int) -> list[dict]:
    return [{"boundary": index == 0} for index in range(count)]


def test_local_continuation_does_not_require_full_episode_horizon():
    rows = [_row(index) for index in range(20)]
    result = evaluate_candidate(rows, _actions(20), 0, 0.2)
    assert result["eligible"] is True
    assert result["metrics"]["absolute_object_eef_distance_max_m"] == 0.08
    assert isclose(result["metrics"]["relative_drift_max_m"], 0.0, abs_tol=1e-12)


def test_terminal_after_final_local_row_is_allowed_but_early_terminal_is_not():
    rows = [_row(index) for index in range(20)]
    rows[-1]["terminal_after"] = True
    assert evaluate_candidate(rows, _actions(20), 0, 0.2)["eligible"] is True
    rows[5]["terminal_after"] = True
    assert "TERMINAL_BEFORE_CONTINUATION_END" in evaluate_candidate(rows, _actions(20), 0, 0.2)["reason_codes"]


def test_support_contact_and_contact_flicker_are_separate_controls():
    rows = [_row(index) for index in range(20)]
    rows[10]["object_support_contact"] = True
    result = evaluate_candidate(rows, _actions(20), 0, 0.2)
    assert "SUPPORT_CONTACT_PRESENT" in result["reason_codes"]

    rows = [_row(index) for index in range(20)]
    rows[10]["object_gripper_contact"] = False
    strict = evaluate_candidate(rows, _actions(20), 0, 0.2, max_contact_false_rows=0)
    permissive = evaluate_candidate(rows, _actions(20), 0, 0.2, max_contact_false_rows=1)
    assert strict["eligible"] is False
    assert permissive["eligible"] is True


def test_precontact_control_is_not_a_eligible_critical_anchor():
    rows = [_row(index, contact=False) for index in range(20)]
    assert classify_calibration_control(rows, 0, 0.2) == "PRE_CONTACT"
    result = evaluate_candidate(rows, _actions(20), 0, 0.2, max_contact_false_rows=1)
    assert result["eligible"] is False
    assert "STABLE_GRASP_WINDOW_INVALID" in result["reason_codes"]


def test_scan_is_deterministic_and_hash_ranked():
    rows = [_row(index) for index in range(21)]
    actions = _actions(21)
    actions[1]["boundary"] = True
    first, reasons_first = scan_candidates(rows, actions, "M0_OPENVLA", "p", 0.2, "salt")
    second, reasons_second = scan_candidates(rows, actions, "M0_OPENVLA", "p", 0.2, "salt")
    assert first == second
    assert reasons_first == reasons_second
    assert {row["step"] for row in first} == {0, 1}


def test_ac2_pi05_adapter_applies_official_clip_to_float_overshoot():
    path = Path(__file__).parents[2] / "scripts/stage_ac/run_stage_ac2_clean_screen.py"
    spec = importlib.util.spec_from_file_location("ac2_clean_screen_clip_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    raw = np.asarray([[1.0000005, -1.0000005, 0.0, 0.0, 0.0, 0.0, -0.9986837]], dtype=np.float32)

    def infer(_obs, _language):
        return raw.copy(), {"raw_action_chunk": raw.tolist()}

    final, meta = runner.official_final_action_adapter(infer, "M2_PI05_LIBERO")({}, "pick")
    assert np.allclose(final, [[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -0.9986837]], atol=0.0)
    assert meta["ac2_official_final_clip_applied"] is True


def test_ac2r2_source_binds_same_immutable_launch_manifest(tmp_path):
    path = Path(__file__).parents[2] / "scripts/stage_ac/run_stage_ac2_clean_screen.py"
    spec = importlib.util.spec_from_file_location("ac2_manifest_binding_test", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    manifest = tmp_path / "manifest.json"
    data = b'{"cell_count":720}\n'
    manifest.write_bytes(data)
    binding = {"path": "manifest.json", "bytes": len(data), "sha256": runner.sha256_bytes(data)}
    source = {
        "status": "STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_FROZEN",
        "input_authorities": {"launch_manifest": binding},
    }
    runner.validate_manifest_source_binding(tmp_path, {"source_bindings": {"runtime_source_authority": {"sha256": "old"}}}, source, tmp_path / "new-source.json", manifest)
