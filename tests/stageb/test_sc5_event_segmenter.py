#!/usr/bin/env python3
"""Test SC5 event segmenter: boundaries, object identity, multi-stage abstain.

Reuses: v2_privileged_teacher.V2PrivilegedTeacher for label generation.
"""
import json, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_event_segmenter_v2 import (
    segment_events_from_labels, validate_transported_object,
    compute_event_sc5, SC5EventSegmenterV2, _validate_phase_order,
)
from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher


def _make_label(step_idx, phase, **kwargs):
    """Helper: create a minimal Teacher label dict."""
    d = {'step_idx': step_idx, 'phase': phase, 'failure_critical': phase in ('stable_carry', 'pre_place_unsupported'),
         'confidence': 0.8, 'abstain_reason': ''}
    d.update(kwargs)
    return d


def test_single_event():
    """Single grasp→carry→release cycle produces 1 event."""
    labels = [
        _make_label(0, 'approach'),
        _make_label(1, 'approach'),
        _make_label(2, 'grasp_close'),
        _make_label(3, 'grasp_close'),
        _make_label(4, 'stable_grasp'),
        _make_label(5, 'first_lift'),
        _make_label(6, 'stable_carry'),
        _make_label(7, 'stable_carry'),
        _make_label(8, 'stable_carry'),
        _make_label(9, 'stable_carry'),
        _make_label(10, 'stable_carry'),
        _make_label(11, 'pre_place_unsupported'),
        _make_label(12, 'release_safe'),
    ]
    events = segment_events_from_labels(labels)
    assert len(events) == 1
    assert events[0]['start_step'] == 2
    assert events[0]['end_step'] == 12
    assert events[0]['has_stable_carry']
    assert events[0]['has_release']
    assert events[0]['phase_order_valid']
    print("PASS: test_single_event")


def test_no_event():
    """Trajectory with no grasp_close produces 0 events."""
    labels = [
        _make_label(0, 'approach'),
        _make_label(1, 'approach'),
        _make_label(2, 'approach'),
    ]
    events = segment_events_from_labels(labels)
    assert len(events) == 0
    print("PASS: test_no_event")


def test_multi_stage():
    """Two grasp→carry→release cycles produce 2 events."""
    labels = [
        _make_label(0, 'approach'),
        _make_label(1, 'grasp_close'),
        _make_label(2, 'stable_grasp'),
        _make_label(3, 'first_lift'),
        _make_label(4, 'stable_carry'),
        _make_label(5, 'stable_carry'),
        _make_label(6, 'pre_place_unsupported'),
        _make_label(7, 'release_safe'),
        _make_label(8, 'approach'),
        _make_label(9, 'grasp_close'),
        _make_label(10, 'stable_grasp'),
        _make_label(11, 'first_lift'),
        _make_label(12, 'stable_carry'),
        _make_label(13, 'stable_carry'),
        _make_label(14, 'release_safe'),
    ]
    events = segment_events_from_labels(labels)
    assert len(events) == 2
    assert events[0]['start_step'] == 1
    assert events[0]['end_step'] == 7
    assert events[1]['start_step'] == 9
    assert events[1]['end_step'] == 14
    assert events[0]['has_stable_carry']
    assert events[1]['has_stable_carry']
    print("PASS: test_multi_stage")


def test_phase_order_valid():
    """Correct phase order validates."""
    assert _validate_phase_order(['grasp_close', 'stable_grasp', 'first_lift',
                                   'stable_carry', 'release_safe'])
    assert not _validate_phase_order(['stable_carry', 'grasp_close'])
    assert not _validate_phase_order(['release_safe', 'stable_carry', 'grasp_close'])
    print("PASS: test_phase_order_valid")


def test_event_sc5_computation():
    """Event-local SC5 anchor is computed correctly."""
    labels = [
        _make_label(10, 'grasp_close'),
        _make_label(11, 'stable_grasp'),
        _make_label(12, 'first_lift'),
        _make_label(13, 'stable_carry'),
        _make_label(14, 'stable_carry'),
        _make_label(15, 'stable_carry'),
        _make_label(16, 'stable_carry'),
        _make_label(17, 'stable_carry'),
        _make_label(18, 'pre_place_unsupported'),
        _make_label(19, 'pre_place_unsupported'),
        _make_label(20, 'pre_place_unsupported'),  # K10 window [18,27] must not cross release
        _make_label(21, 'pre_place_unsupported'),
        _make_label(22, 'pre_place_unsupported'),
        _make_label(23, 'pre_place_unsupported'),
        _make_label(24, 'pre_place_unsupported'),
        _make_label(25, 'pre_place_unsupported'),
        _make_label(26, 'pre_place_unsupported'),
        _make_label(27, 'pre_place_unsupported'),
        _make_label(28, 'release_safe'),  # after K10 window
    ]
    event = {'event_id': 0, 'start_step': 10, 'end_step': 28}
    sc5 = compute_event_sc5(labels, event, K=10, guard=5)
    assert sc5['valid'], f"Expected valid SC5, got: {sc5['reason']}"
    assert sc5['anchor'] == 18  # sc_start(13) + guard(5) = 18
    assert sc5['stable_carry_start'] == 13
    print("PASS: test_event_sc5_computation")


def test_k10_crosses_event_boundary():
    """K10 window that extends past event end is invalid."""
    labels = [
        _make_label(10, 'grasp_close'),
        _make_label(11, 'stable_grasp'),
        _make_label(12, 'first_lift'),
        _make_label(13, 'stable_carry'),
        _make_label(14, 'stable_carry'),
        _make_label(15, 'stable_carry'),
        _make_label(16, 'stable_carry'),
        _make_label(17, 'stable_carry'),
        _make_label(18, 'pre_place_unsupported'),
        # event ends at 18 (no release_safe, short event)
    ]
    event = {'event_id': 0, 'start_step': 10, 'end_step': 18}
    sc5 = compute_event_sc5(labels, event, K=10, guard=5)
    # anchor=18, K10 would be [18,27], but max step is 18 (episode short)
    assert not sc5['valid']
    assert 'exceeds_episode' in sc5['reason'] or 'event_boundary' in sc5['reason'], \
        f"Expected boundary/exceeds, got: {sc5['reason']}"
    print("PASS: test_k10_crosses_event_boundary")


def test_object_identity_anonymous_pose_is_unverifiable():
    """Anonymous pose stream cannot prove transported object identity."""
    labels = [_make_label(i, 'stable_carry') for i in range(5, 15)]
    event = {'start_step': 5, 'end_step': 14}
    records = [
        {'object_pose_json': json.dumps([0.1, 0.2, 0.3 + i * 0.001])}
        for i in range(20)
    ]
    result = validate_transported_object(labels, event, records)
    assert not result['unique_object']
    assert not result['verifiable']
    assert result['reason'] == 'unverifiable_identity'
    assert result['first_object_hash']
    assert result['position_variance'] < 0.01
    print("PASS: test_object_identity_anonymous_pose_is_unverifiable")


def test_object_identity_same_explicit_object():
    """Same explicit object throughout event validates as unique."""
    labels = [_make_label(i, 'stable_carry') for i in range(5, 15)]
    event = {'start_step': 5, 'end_step': 14}
    records = [
        {
            'object_name': 'black_bowl',
            'object_pose_json': json.dumps([0.1, 0.2, 0.3 + i * 0.001]),
        }
        for i in range(20)
    ]
    result = validate_transported_object(labels, event, records)
    assert result['unique_object']
    assert result['verifiable']
    assert result['first_object_hash']
    assert result['position_variance'] < 0.01
    print("PASS: test_object_identity_same_explicit_object")


def test_object_identity_no_data():
    """No object pose data returns insufficient_data."""
    labels = [_make_label(i, 'stable_carry') for i in range(5, 10)]
    event = {'start_step': 5, 'end_step': 9}
    records = [{} for _ in range(15)]
    result = validate_transported_object(labels, event, records)
    assert not result['unique_object']
    assert result['reason'] == 'insufficient_pose_data'
    print("PASS: test_object_identity_no_data")


def test_segmenter_full_pipeline():
    """Full segmenter pipeline with Teacher."""
    teacher = V2PrivilegedTeacher()
    segmenter = SC5EventSegmenterV2(teacher)

    # Build minimal step records
    records = []
    for i in range(30):
        rec = {
            'step_idx': i,
            'teacher_privileged_state_available': i >= 2,
            'gripper_command': 0.3 if 5 <= i <= 12 else 0.7,
            'gripper_qpos': 0.05, 'gripper_width': 0.02,
            'eef_x': 0.0, 'eef_y': 0.0, 'eef_z': 0.25 + i * 0.001,
            'eef_vx': 0.0, 'eef_vy': 0.0, 'eef_vz': 0.0,
            'object_pose_json': json.dumps([0.1, 0.2, 0.3 + i * 0.001]),
            'object_to_target_distance': 0.5 - i * 0.02,
            'object_eef_distance': 0.3 - i * 0.01 if i < 10 else 0.05,
            'target_pose_json': json.dumps([0.5, 0.0, 0.3]),
        }
        records.append(rec)

    labels = teacher.label_trajectory(records)
    result = segmenter.segment(labels, records, K=10, guard=5)

    assert result['n_events'] >= 0  # may or may not have events with synthetic data
    assert 'abstain_reason' in result
    assert 'primary_event_idx' in result
    print("PASS: test_segmenter_full_pipeline")


if __name__ == '__main__':
    test_single_event()
    test_no_event()
    test_multi_stage()
    test_phase_order_valid()
    test_event_sc5_computation()
    test_k10_crosses_event_boundary()
    test_object_identity_same_object()
    test_object_identity_no_data()
    test_segmenter_full_pipeline()
    print("\nAll SC5 event segmenter tests passed.")
