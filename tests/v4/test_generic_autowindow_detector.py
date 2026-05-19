import json

from scripts.detect_contact_window_from_clean import DetectorConfig, candidate_rows, detect_window


def _row(step, gripper=1.0, z=0.0, eef_z=0.0, dist=1.0):
    return {
        "step": step,
        "clean_gripper_env": gripper,
        "object_z_after": z,
        "eef_z_after": eef_z,
        "receptacle_dist": dist,
    }


def test_generic_detector_selects_preplace_window():
    rows = []
    for step in range(40):
        rows.append(
            _row(
                step,
                gripper=1.0 if step < 30 else -1.0,
                z=0.0 if step < 6 else 0.04,
                eef_z=0.0 if step < 6 else 0.05 - max(0, step - 25) * 0.002,
                dist=max(0.02, 0.6 - step * 0.02),
            )
        )
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["window_detected"] is True
    assert result["detector_mode"] in {"preplace_cue", "release_intent"}
    assert result["confidence"] in {"high", "medium"}


def test_release_intent_after_lift_selects_window_before_release():
    rows = []
    for step in range(45):
        rows.append(
            _row(
                step,
                gripper=1.0 if step < 34 else -1.0,
                z=0.0 if step < 8 else 0.05,
                eef_z=0.0 if step < 8 else 0.06,
                dist=0.5,
            )
        )
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["detector_mode"] == "release_intent"
    assert result["auto_window_start"] == 24
    assert result["auto_window_end"] == 33


def test_eef_descent_after_lift_selects_window_before_descent():
    rows = []
    for step in range(45):
        eef = 0.0 if step < 8 else 0.08
        if step >= 34:
            eef = 0.065
        rows.append(_row(step, gripper=1.0, z=0.0 if step < 8 else 0.05, eef_z=eef, dist=None))
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["detector_mode"] == "preplace_cue"
    assert result["auto_window_start"] == 24
    assert result["auto_window_end"] == 33


def test_early_near_target_alone_does_not_select_early_high_confidence_window():
    rows = []
    for step in range(45):
        rows.append(
            _row(
                step,
                gripper=1.0,
                z=0.0 if step < 8 else 0.05,
                eef_z=0.0 if step < 8 else 0.06,
                dist=max(0.02, 0.4 - min(step, 16) * 0.02),
            )
        )
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert not (result["detector_mode"] == "preplace_cue" and result["auto_window_start"] < 20)
    assert result["detector_mode"] in {"late_carry_fallback", "near_target_late"}
    assert result["confidence"] in {"low", "medium"}


def test_near_target_with_late_eef_descent_selects_late_descent():
    rows = []
    for step in range(45):
        eef = 0.0 if step < 8 else 0.08
        if step >= 34:
            eef = 0.065
        rows.append(
            _row(
                step,
                gripper=1.0,
                z=0.0 if step < 8 else 0.05,
                eef_z=eef,
                dist=max(0.02, 0.5 - step * 0.015),
            )
        )
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["detector_mode"] == "preplace_cue"
    assert result["selected_preplace_reason"] == "eef_descent"
    assert result["auto_window_start"] == 24


def test_close_without_lift_abstains_or_low_confidence():
    rows = [_row(step, gripper=1.0, z=0.0, eef_z=0.0, dist=0.5) for step in range(30)]
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["window_detected"] is False or result["confidence"] == "low"
    assert result["failure_reason"] in {"no_lift_detected", "no_reliable_preplace_cue"}


def test_lift_without_stable_grasp_or_carry_does_not_force_formal_window():
    rows = [
        _row(step, gripper=-1.0, z=0.05 if step > 5 else 0.0, eef_z=0.06 if step > 5 else 0.0, dist=0.4)
        for step in range(30)
    ]
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["window_detected"] is False or result["confidence"] == "low"
    assert result["failure_reason"] in {"no_stable_grasp", "no_stable_carry"}


def test_missing_pose_with_proxy_uses_low_or_medium_fallback():
    rows = []
    for step in range(35):
        rows.append(
            {
                "step": step,
                "clean_gripper_env": 1.0,
                "proxy_lift_carry_eef_z": 0.0 if step < 7 else 0.05,
            }
        )
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    assert result["window_detected"] is True
    assert result["detector_mode"] == "late_carry_fallback"
    assert result["confidence"] in {"low", "medium"}


def test_lift_only_uses_normalized_late_carry_fallback():
    rows = [_row(step, gripper=1.0, z=0.0 if step < 7 else 0.05, eef_z=0.0 if step < 7 else 0.06, dist=None) for step in range(50)]
    result = detect_window(rows, clean_success=True, cfg=DetectorConfig(late_offset_ratio=0.5))
    assert result["detector_mode"] == "late_carry_fallback"
    assert result["confidence"] in {"high", "medium", "low"}
    assert result["auto_window_start"] == 29


def test_candidate_rows_skips_non_clean_manifest(tmp_path):
    run_dir = tmp_path / "example_control_clean_suffix"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"trigger_name": "control", "task_id": "example"}), encoding="utf-8")
    (run_dir / "episode_records.jsonl").write_text(json.dumps({"success": True}) + "\n", encoding="utf-8")
    (run_dir / "step_records.jsonl").write_text(json.dumps(_row(0)) + "\n", encoding="utf-8")

    rows = candidate_rows(tmp_path, DetectorConfig(), "test_hash")

    assert rows == []


def test_metadata_does_not_influence_detector_output():
    rows = [_row(step, gripper=1.0 if step < 20 else -1.0, z=0.0 if step < 5 else 0.05, eef_z=0.06, dist=0.5) for step in range(30)]
    first = detect_window(rows, clean_success=True, cfg=DetectorConfig())
    second = detect_window([dict(row, task_id="other", state=999) for row in rows], clean_success=True, cfg=DetectorConfig())
    assert first == second
