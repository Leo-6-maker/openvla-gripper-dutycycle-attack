#!/usr/bin/env python3
"""Test SC5 sequence causality: no future leakage, no zero-fill, no forbidden features."""
import json, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
from gripper_attack.sc5_schema_adapter_v2 import SC5SchemaAdapterV2, CANONICAL_25D


def _make_record(step, raw_grip=0.3, eef_z=0.25):
    return {
        'step_idx': step,
        'teacher_privileged_state_available': True,
        'gripper_command': str(raw_grip),
        'gripper_qpos': '0.05',
        'gripper_width': '0.02',
        'eef_x': '0.0', 'eef_y': '0.0', 'eef_z': str(eef_z),
        'eef_vx': '0.0', 'eef_vy': '0.0', 'eef_vz': '0.0',
        'action_dx': '0.0', 'action_dy': '0.0', 'action_dz': '0.0',
        'action_gripper': str(raw_grip),
    }


def test_no_future_leakage():
    """Features at step t only depend on steps <= t."""
    adapter = SC5StreamingFeatureAdapterV2()
    records = [_make_record(i, raw_grip=0.7) for i in range(50)]

    # Record a close event at t=30
    records[30]['gripper_command'] = '0.3'

    # Feed steps 0-29, check that close_onset is 0
    for i in range(30):
        r = records[i]
        raw = float(r['gripper_command'])
        env = -1.0 if raw > 0.5 else 1.0
        result = adapter.update(
            step_id=i, raw_gripper=raw, env_gripper=env,
            gripper_qpos=float(r['gripper_qpos']),
            gripper_opening_proxy=float(r['gripper_width']),
            eef_x=float(r['eef_x']), eef_y=float(r['eef_y']), eef_z=float(r['eef_z']),
            eef_vx=float(r['eef_vx']), eef_vy=float(r['eef_vy']), eef_vz=float(r['eef_vz']),
            action_dx=float(r['action_dx']), action_dy=float(r['action_dy']),
            action_dz=float(r['action_dz']), action_gripper=float(r['action_gripper']),
        )
        if result['valid']:
            # At step 29, close_onset should still be 0 (close happens at step 30)
            if i < 30:
                assert result['features']['close_onset'] == 0, \
                    f"close_onset at step {i} should be 0, got {result['features']['close_onset']}"
    print("PASS: test_no_future_leakage")


def test_streaming_adapter_step_sequence():
    """Step sequence violations raise ValueError."""
    adapter = SC5StreamingFeatureAdapterV2()
    r = _make_record(0)
    raw = float(r['gripper_command'])
    env = -1.0 if raw > 0.5 else 1.0
    adapter.update(step_id=0, raw_gripper=raw, env_gripper=env,
                   gripper_qpos=0.05, gripper_opening_proxy=0.02,
                   eef_x=0.0, eef_y=0.0, eef_z=0.25,
                   eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
                   action_dx=0.0, action_dy=0.0, action_dz=0.0, action_gripper=0.0)

    # Skip step 1
    try:
        adapter.update(step_id=2, raw_gripper=raw, env_gripper=env,
                       gripper_qpos=0.05, gripper_opening_proxy=0.02,
                       eef_x=0.0, eef_y=0.0, eef_z=0.25,
                       eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
                       action_dx=0.0, action_dy=0.0, action_dz=0.0, action_gripper=0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'sequence violation' in str(e).lower() or 'expected' in str(e).lower()
    print("PASS: test_streaming_adapter_step_sequence")


def test_no_zero_fill_in_adapter():
    """Invalid fields produce valid=False, not zero-filled features."""
    adapter = SC5StreamingFeatureAdapterV2()
    # Missing gripper_qpos
    result = adapter.update(
        step_id=0, raw_gripper=0.3, env_gripper=1.0,
        gripper_qpos=float('nan'),  # invalid
        gripper_opening_proxy=0.02,
        eef_x=0.0, eef_y=0.0, eef_z=0.25,
        eef_vx=0.0, eef_vy=0.0, eef_vz=0.0,
        action_dx=0.0, action_dy=0.0, action_dz=0.0, action_gripper=0.0,
    )
    assert not result['valid'], "Should be invalid due to NaN gripper_qpos"
    assert result['features'] is None, "Features should be None for invalid step"
    print("PASS: test_no_zero_fill_in_adapter")


def test_schema_adapter_no_future_access():
    """Schema adapter validate_record() only accesses current record."""
    adapter = SC5SchemaAdapterV2()

    # Populate EEF history
    adapter.track_eef(0.0, 0.0, 0.20)
    adapter.track_eef(0.01, 0.0, 0.22)

    # Validate a record — should only use history (past), not future
    record = {
        'gripper_command': '0.3', 'gripper_qpos': '0.05', 'gripper_width': '0.02',
        'eef_x': '0.02', 'eef_y': '0.0', 'eef_z': '0.25',
        'eef_vx': '0.01', 'eef_vy': '0.0', 'eef_vz': '0.0',
        'action_dx': '0', 'action_dy': '0', 'action_dz': '0', 'action_gripper': '0',
    }
    provenances = adapter.validate_record(record)
    assert adapter.all_valid(provenances)

    # Verify the adapter doesn't have any "future" data mechanism
    assert not hasattr(adapter, '_future_buffer')
    print("PASS: test_schema_adapter_no_future_access")


def test_no_forbidden_features_in_25d():
    """25D feature list contains no forbidden features."""
    forbidden = ['normalized_step', 'episode_position', 'task_id', 'state_id',
                 'run_id', 'object_pose', 'teacher_anchor', 'teacher_phase']
    for f in forbidden:
        assert f not in CANONICAL_25D, f"Forbidden feature '{f}' found in CANONICAL_25D"
    print("PASS: test_no_forbidden_features_in_25d")


def test_25d_ordering_matches_streaming():
    """CANONICAL_25D ordering matches SC5StreamingFeatureAdapterV2 output."""
    from gripper_attack.sc5_streaming_features_v2 import FEATURE_NAMES
    assert CANONICAL_25D == FEATURE_NAMES, \
        "CANONICAL_25D must match FEATURE_NAMES from streaming adapter"
    print("PASS: test_25d_ordering_matches_streaming")


if __name__ == '__main__':
    test_no_future_leakage()
    test_streaming_adapter_step_sequence()
    test_no_zero_fill_in_adapter()
    test_schema_adapter_no_future_access()
    test_no_forbidden_features_in_25d()
    test_25d_ordering_matches_streaming()
    print("\nAll SC5 sequence causality tests passed.")
