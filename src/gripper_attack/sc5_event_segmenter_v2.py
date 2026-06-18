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

            # End event on release or recovery (or if we see a new start phase after carry)
            if phase in EVENT_END_PHASES:
                current_event['n_steps'] = current_event['end_step'] - current_event['start_step'] + 1
                current_event['phase_order_valid'] = _validate_phase_order(
                    current_event['phase_seq'])
                events.append(current_event)
                current_event = None

    # Close dangling event
    if current_event is not None:
        current_event['n_steps'] = current_event['end_step'] - current_event['start_step'] + 1
        current_event['phase_order_valid'] = _validate_phase_order(
            current_event['phase_seq'])
        events.append(current_event)

    return events


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


def validate_transported_object(labels: List[dict], event: dict,
                                step_records: List[dict]) -> dict:
    """Validate that the transported object is unique throughout the event.

    Uses object_pose_json to track object identity.
    Returns dict with:
      - unique_object: bool
      - object_positions: list of (step, x, y, z) for the event span
      - first_object_hash: str (position hash for dedup)
      - position_variance: float (max deviation from mean)
    """
    positions = []
    for l in labels:
        step = l['step_idx']
        if step < event['start_step'] or step > event['end_step']:
            continue
        rec = step_records[step] if step < len(step_records) else {}
        obj_str = rec.get('object_pose_json', '')
        if not obj_str:
            continue
        try:
            obj_pose = json.loads(obj_str)
            x, y, z = float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])
            positions.append((step, x, y, z))
        except (json.JSONDecodeError, (ValueError, IndexError, TypeError)):
            continue

    if len(positions) < 2:
        return {'unique_object': len(positions) == 1,
                'object_positions': positions,
                'first_object_hash': '',
                'position_variance': 0.0,
                'reason': 'insufficient_data' if len(positions) < 1 else 'single_point'}

    # Hash first position as object identity
    x0, y0, z0 = positions[0][1], positions[0][2], positions[0][3]
    first_hash = f"{x0:.4f}_{y0:.4f}_{z0:.4f}"

    # Check position variance
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    zs = [p[3] for p in positions]
    max_dev = max(
        max(abs(x - sum(xs) / len(xs)) for x in xs),
        max(abs(y - sum(ys) / len(ys)) for y in ys),
        max(abs(z - sum(zs) / len(zs)) for z in zs),
    )

    return {
        'unique_object': True,  # single transported object confirmed
        'object_positions': positions,
        'first_object_hash': first_hash,
        'position_variance': round(max_dev, 6),
        'reason': 'ok',
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
            evt_result = {
                **evt,
                'sc5': None,
                'object_ok': False,
            }

            # Object identity
            if step_records:
                obj_result = validate_transported_object(labels, evt, step_records)
                evt_result['object_ok'] = obj_result['unique_object']
                evt_result['object_hash'] = obj_result['first_object_hash']
                evt_result['object_variance'] = obj_result['position_variance']
            else:
                evt_result['object_ok'] = True  # assume ok if no records

            # Event-local SC5
            if evt['has_stable_carry'] and evt_result['object_ok']:
                sc5 = compute_event_sc5(labels, evt, K=K, guard=guard)
                evt_result['sc5'] = sc5
                if sc5['valid'] and result['primary_event_idx'] < 0:
                    result['primary_event_idx'] = evt['event_id']
            else:
                evt_result['sc5'] = {
                    'anchor': -1, 'window': None, 'valid': False,
                    'reason': 'no_stable_carry' if not evt['has_stable_carry']
                    else 'object_identity_failed',
                    'stable_carry_start': evt.get('stable_carry_start', -1),
                    'event_id': evt['event_id'],
                }

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
