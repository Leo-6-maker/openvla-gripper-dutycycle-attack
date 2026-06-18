#!/usr/bin/env python3
"""SC5 Event Segmenter v2 — segments multi-stage episodes into grasp→carry→release events.

Reuses: v2_privileged_teacher.V2PrivilegedTeacher for phase labeling,
        v2_privileged_teacher.find_sc5_anchor_v2 for event-local SC5 anchors.

New: event boundary detection, transported-object identity validation,
     event-local phase sequence validation, multi-stage abstain logic.

Design: thin wrapper over existing Teacher — no new phase logic.
"""
from __future__ import annotations

import json
from typing import Optional, List, Dict, Tuple

# Grasp→carry→release event phase sequence (frozen)
EVENT_START_PHASES = {'grasp_close', 'stable_grasp'}
EVENT_CARRY_PHASES = {'first_lift', 'stable_carry', 'pre_place_unsupported'}
EVENT_END_PHASES = {'release_safe', 'recovery_or_regrasp'}
EVENT_CORE_PHASES = EVENT_START_PHASES | EVENT_CARRY_PHASES | EVENT_END_PHASES

# Minimum consecutive stable_carry steps for a valid event
MIN_STABLE_CARRY_STEPS = 3


def segment_events_from_labels(labels: List[dict]) -> List[dict]:
    """Segment a trajectory's Teacher labels into grasp→carry→release events.

    Returns list of event dicts, each containing:
      - event_id: int (0-based within episode)
      - start_step: int (first step of event)
      - end_step: int (last step of event, inclusive)
      - n_steps: int
      - has_stable_carry: bool
      - stable_carry_start: int (-1 if none)
      - has_release: bool
      - phase_sequence: list of phases in order
      - phase_order_valid: bool
    """
    events = []
    current_event = None

    for l in labels:
        phase = l['phase']
        step = l['step_idx']

        if phase in EVENT_START_PHASES and current_event is None:
            # Start new event
            current_event = {
                'event_id': len(events),
                'start_step': step,
                'end_step': step,
                'phases_seen': [],
                'stable_carry_start': -1,
                'has_stable_carry': False,
                'has_release': False,
                'phase_seq': [],
            }

        if current_event is not None:
            current_event['end_step'] = step
            if phase not in current_event['phases_seen']:
                current_event['phases_seen'].append(phase)
            current_event['phase_seq'].append(phase)

            if phase == 'stable_carry' and not current_event['has_stable_carry']:
                current_event['has_stable_carry'] = True
                current_event['stable_carry_start'] = step

            if phase == 'release_safe':
                current_event['has_release'] = True

            # End event on release or recovery
            if phase in EVENT_END_PHASES:
                current_event['n_steps'] = current_event['end_step'] - current_event['start_step'] + 1
                current_event['phase_order_valid'] = _validate_phase_order(
                    current_event['phase_seq'])
                current_event['has_all_required_phases'] = _validate_required_phases(
                    current_event['phases_seen'])
                events.append(current_event)
                current_event = None

    # Close dangling event
    if current_event is not None:
        current_event['n_steps'] = current_event['end_step'] - current_event['start_step'] + 1
        current_event['phase_order_valid'] = _validate_phase_order(
            current_event['phase_seq'])
        current_event['has_all_required_phases'] = _validate_required_phases(
            current_event['phases_seen'])
        events.append(current_event)

    return events


REQUIRED_EVENT_PHASES = [
    'grasp_close', 'stable_grasp', 'first_lift',
    'stable_carry', 'release_safe',
]


def _validate_required_phases(phases_seen: List[str]) -> bool:
    """All 5 required phases must be present in the event."""
    return all(p in phases_seen for p in REQUIRED_EVENT_PHASES)


def _validate_phase_order(phases: List[str]) -> bool:
    """Validate phase ordering: grasp_close→stable_grasp→first_lift→stable_carry→release_safe."""
    order = {'grasp_close': 0, 'stable_grasp': 1, 'first_lift': 2,
             'stable_carry': 3, 'pre_place_unsupported': 4, 'release_safe': 5}
    last_idx = -1
    for p in phases:
        if p in order:
            idx = order[p]
            if idx < last_idx:
                return False
            last_idx = idx
    return True


def _count_consecutive_stable_carry(labels: List[dict], event: dict) -> int:
    """Count max consecutive stable_carry steps within the event span."""
    max_run = 0; current_run = 0
    for l in labels:
        if l['step_idx'] < event['start_step'] or l['step_idx'] > event['end_step']:
            continue
        if l['phase'] == 'stable_carry':
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def validate_transported_object(labels: List[dict], event: dict,
                                step_records: List[dict]) -> dict:
    """Validate transported object identity within event.

    For multi-object tasks: requires explicit object name/ID or multi-object
    pose map. Anonymous single object_pose_json stream is INSUFFICIENT for
    multi-object confirmation — must abstain.

    Returns dict with:
      - unique_object: bool (False if identity unverifiable)
      - object_positions: list of (step, x, y, z)
      - verifiable: bool (whether identity can be confidently determined)
    """
    positions = []
    object_names = set()
    for l in labels:
        step = l['step_idx']
        if step < event['start_step'] or step > event['end_step']:
            continue
        rec = step_records[step] if step < len(step_records) else {}
        obj_str = rec.get('object_pose_json', '')
        obj_name = rec.get('object_name', rec.get('transported_object_name', ''))
        if obj_name:
            object_names.add(str(obj_name))
        if not obj_str:
            continue
        try:
            obj_pose = json.loads(obj_str)
            x, y, z = float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])
            positions.append((step, x, y, z))
        except (json.JSONDecodeError, (ValueError, IndexError, TypeError)):
            continue

    # Object identity verifiability:
    # - Explicit object name/ID present → verifiable
    # - Single anonymous pose stream with continuous trajectory → weakly verifiable
    # - Multi-object task with only anonymous pose → NOT verifiable
    has_explicit_id = len(object_names) > 0
    has_pose_data = len(positions) >= 2

    if not has_pose_data:
        return {'unique_object': False, 'verifiable': False,
                'object_positions': positions, 'object_names': list(object_names),
                'first_object_hash': '', 'position_variance': 0.0,
                'reason': 'insufficient_pose_data'}

    # For tasks with explicit object IDs: verify single object throughout
    if has_explicit_id:
        unique = len(object_names) == 1
        reason = 'explicit_id_verified' if unique else 'multiple_object_names_detected'
    else:
        # Anonymous pose stream: position continuity suggests single object
        # but cannot be proven for multi-object tasks
        unique = True  # weakly assumed for single-object primary tasks
        reason = 'anonymous_pose_stream_weak_identity'

    x0, y0, z0 = positions[0][1], positions[0][2], positions[0][3]
    first_hash = f"{x0:.4f}_{y0:.4f}_{z0:.4f}"
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    zs = [p[3] for p in positions]
    max_dev = max(
        max(abs(x - sum(xs) / len(xs)) for x in xs),
        max(abs(y - sum(ys) / len(ys)) for y in ys),
        max(abs(z - sum(zs) / len(zs)) for z in zs),
    ) if xs else 0.0

    return {
        'unique_object': unique,
        'verifiable': has_explicit_id,
        'object_positions': positions,
        'object_names': list(object_names),
        'first_object_hash': first_hash,
        'position_variance': round(max_dev, 6),
        'reason': reason,
    }


def compute_event_sc5(event_labels: List[dict], event: dict,
                       K: int = 10, guard: int = 5) -> dict:
    """Compute SC5 anchor for a single event, using only event-local labels.

    Reuses: find_sc5_anchor_v2() on the event's label slice.

    Returns SC5 result dict, plus event boundary containment check.
    """
    from gripper_attack.v2_privileged_teacher import find_sc5_anchor_v2

    # Filter labels to event span
    local_labels = [l for l in event_labels
                    if event['start_step'] <= l['step_idx'] <= event['end_step']]

    sc5 = find_sc5_anchor_v2(local_labels, K=K, guard=guard)

    # Additional check: K10 window must not cross event end
    if sc5['valid']:
        window_end = sc5['anchor'] + K - 1
        if window_end > event['end_step']:
            sc5['valid'] = False
            sc5['reason'] = 'k10_crosses_event_boundary'

    sc5['event_id'] = event['event_id']
    sc5['event_start'] = event['start_step']
    sc5['event_end'] = event['end_step']
    return sc5


class SC5EventSegmenterV2:
    """Thin wrapper: uses V2PrivilegedTeacher labels to segment events.

    Must be used per-episode. Call reset() between episodes.
    """

    def __init__(self, teacher=None):
        self.teacher = teacher
        self._events: List[dict] = []
        self._n_events: int = 0

    def reset(self):
        self._events = []
        self._n_events = 0

    def segment(self, labels: List[dict], step_records: List[dict] = None,
                K: int = 10, guard: int = 5) -> dict:
        """Segment one episode into events and compute event-local SC5.

        Returns dict with:
          - n_events: int
          - is_multi_stage: bool
          - events: list of per-event dicts
          - primary_event_idx: int (index of first valid SC5 event, or -1)
          - abstain_reason: str (empty if at least one valid SC5 event)
        """
        self.reset()
        events = segment_events_from_labels(labels)

        result = {
            'n_events': len(events),
            'is_multi_stage': len(events) > 1,
            'events': [],
            'primary_event_idx': -1,
            'abstain_reason': '',
        }

        for evt in events:
            evt_result = {**evt, 'sc5': None, 'object_ok': False,
                          'event_valid': False, 'reject_reason': ''}

            # Gate 1: required phases
            if not evt.get('has_all_required_phases', False):
                evt_result['reject_reason'] = 'missing_required_phases'
                result['events'].append(evt_result)
                continue

            # Gate 2: stable_carry minimum length
            sc_len = _count_consecutive_stable_carry(labels, evt)
            if sc_len < MIN_STABLE_CARRY_STEPS:
                evt_result['reject_reason'] = f'stable_carry_too_short({sc_len}<{MIN_STABLE_CARRY_STEPS})'
                result['events'].append(evt_result)
                continue

            # Gate 3: object identity
            if step_records:
                obj_result = validate_transported_object(labels, evt, step_records)
                evt_result['object_ok'] = obj_result['unique_object']
                evt_result['object_hash'] = obj_result['first_object_hash']
                evt_result['object_verifiable'] = obj_result['verifiable']

                # For anonymous pose streams in multi-event episodes: require verifiable identity
                if result['n_events'] > 1 and not obj_result['verifiable']:
                    evt_result['reject_reason'] = 'OBJECT_IDENTITY_UNVERIFIABLE'
                    result['events'].append(evt_result)
                    continue
            else:
                evt_result['object_ok'] = False
                evt_result['reject_reason'] = 'no_step_records_for_identity'
                result['events'].append(evt_result)
                continue

            # Gate 4: event-local SC5
            if evt['has_stable_carry'] and evt_result['object_ok']:
                sc5 = compute_event_sc5(labels, evt, K=K, guard=guard)
                evt_result['sc5'] = sc5
                evt_result['event_valid'] = sc5['valid']
                if sc5['valid']:
                    if result['primary_event_idx'] < 0:
                        result['primary_event_idx'] = evt['event_id']
                else:
                    evt_result['reject_reason'] = f'sc5_invalid:{sc5.get("reason","")}'
            else:
                evt_result['sc5'] = {
                    'anchor': -1, 'window': None, 'valid': False,
                    'reason': 'no_stable_carry' if not evt['has_stable_carry']
                    else 'object_identity_failed',
                    'stable_carry_start': evt.get('stable_carry_start', -1),
                    'event_id': evt['event_id'],
                }
                evt_result['reject_reason'] = evt_result['sc5']['reason']

            result['events'].append(evt_result)

        if result['primary_event_idx'] < 0:
            result['abstain_reason'] = 'no_valid_sc5_event'

        self._events = result['events']
        self._n_events = len(events)
        return result

    @property
    def n_events(self) -> int:
        return self._n_events

    @property
    def is_multi_stage(self) -> bool:
        return self._n_events > 1
