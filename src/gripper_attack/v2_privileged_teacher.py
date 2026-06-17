#!/usr/bin/env python3
"""V2 Privileged Teacher: clean-only phase labeling from MuJoCo privileged state.

Operates on Object100-style step_records.jsonl.
Rules:
  - ONLY clean-simulation privileged state (object/target/EEF/gripper).
  - NEVER sees attack outcome, VIS, RAND, or task failure.
  - Fail-closed abstain on missing data.
  - Phase ordering: approach -> grasp_close -> stable_grasp -> first_lift ->
    stable_carry -> pre_place_unsupported -> release_safe.
"""
from __future__ import annotations

import json, math, os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np


PHASES = [
    'approach', 'grasp_close', 'stable_grasp', 'first_lift',
    'stable_carry', 'pre_place_unsupported', 'release_safe',
    'recovery_or_regrasp', 'abstain_unsupported',
]

FAILURE_CRITICAL_PHASES = {'stable_carry', 'pre_place_unsupported'}


@dataclass
class TeacherConfig:
    gripper_close_threshold: float = 0.02
    grasp_sustain_steps: int = 3
    eef_obj_dist_max: float = 0.12
    eef_obj_dist_stable_var: float = 0.005
    lift_z_threshold: float = 0.015
    lift_sustain_steps: int = 2
    carry_obj_z_var_max: float = 0.01
    carry_window: int = 8
    preplace_target_dist_min: float = 0.05
    preplace_target_dist_max: float = 0.25
    release_target_dist_max: float = 0.08
    release_obj_z_var_max: float = 0.005
    regrasp_eef_obj_dist_max: float = 0.10
    stability_window: int = 5
    version: str = 'v2_teacher_draft_0'
    calibrated_from: str = 'uncalibrated_draft'


def _extract_state(rec: dict) -> dict:
    obj_pose = json.loads(rec.get('object_pose_json', '[0,0,0,0,0,0,0]'))
    target_pose = json.loads(rec.get('target_pose_json', '[0,0,0]'))
    return {
        'obj_x': float(obj_pose[0]), 'obj_y': float(obj_pose[1]),
        'obj_z': float(obj_pose[2]),
        'target_x': float(target_pose[0]), 'target_y': float(target_pose[1]),
        'target_z': float(target_pose[2]),
        'obj_target_dist': float(rec.get('object_to_target_distance', 0)),
        'obj_eef_dist': float(rec.get('object_eef_distance', 0)),
        'gripper_qpos': float(rec.get('gripper_qpos', 0)),
        'gripper_width': float(rec.get('gripper_width', 0)),
        'gripper_command': float(rec.get('gripper_command', 0.5)),
        'eef_x': float(rec.get('eef_x', 0)), 'eef_y': float(rec.get('eef_y', 0)),
        'eef_z': float(rec.get('eef_z', 0)),
        'step_idx': int(rec.get('step_idx', 0)),
        'policy_step_idx': int(rec.get('policy_step_idx', -1)),
        'phase': rec.get('phase', 'policy'),
        'has_priv': bool(rec.get('teacher_privileged_state_available', False)),
    }


def _window_mean(history: List[dict], key: str, window: int) -> Optional[float]:
    vals = [s[key] for s in history[-window:]
            if s.get(key) is not None and not math.isnan(s[key])]
    return float(np.mean(vals)) if vals else None


def _window_var(history: List[dict], key: str, window: int) -> Optional[float]:
    vals = [s[key] for s in history[-window:]
            if s.get(key) is not None and not math.isnan(s[key])]
    return float(np.var(vals)) if len(vals) >= 3 else None


def _consecutive_count(history: List[dict], key: str, condition_fn) -> int:
    n = 0
    for s in reversed(history):
        val = s.get(key, 0)
        if val is not None and not (isinstance(val, float) and math.isnan(val)) and condition_fn(val):
            n += 1
        else:
            break
    return n


class V2PrivilegedTeacher:

    def __init__(self, config: TeacherConfig = None):
        self.cfg = config or TeacherConfig()

    def label_trajectory(self, step_records: List[dict]) -> List[dict]:
        states = [_extract_state(r) for r in step_records]
        priv_indices = [i for i, s in enumerate(states) if s['has_priv']]
        if not priv_indices:
            return [{'step_idx': s['step_idx'], 'phase': 'abstain_unsupported',
                     'failure_critical': False, 'confidence': 0.0,
                     'abstain_reason': 'no_privileged_data'} for s in states]

        labels = []
        prev_phase = 'approach'
        obj_z0 = None
        was_lifted = False

        for i, s in enumerate(states):
            hist = states[:i + 1]

            if not s['has_priv'] or s['phase'] == 'wait':
                labels.append({'step_idx': s['step_idx'], 'phase': 'abstain_unsupported',
                               'failure_critical': False, 'confidence': 0.0,
                               'abstain_reason': 'wait_or_no_privilege'})
                continue

            if obj_z0 is None:
                obj_z0 = s['obj_z']

            cfg = self.cfg
            # Use gripper_command > 0.5 for CLOSE detection (matches D5 semantics,
            # robust across qpos/qpos_sum/gripper_width encoding differences)
            gripper_closed = s['gripper_command'] > 0.5
            close_consecutive = _consecutive_count(hist, 'gripper_command',
                lambda c: c > 0.5)
            grasp_stable = close_consecutive >= cfg.grasp_sustain_steps

            obj_eef_close = s['obj_eef_dist'] < cfg.eef_obj_dist_max
            obj_eef_var = _window_var(hist, 'obj_eef_dist', cfg.stability_window)

            obj_lifted = (s['obj_z'] - obj_z0) > cfg.lift_z_threshold
            lift_consecutive = _consecutive_count(hist, 'obj_z',
                lambda z: (z - obj_z0) > cfg.lift_z_threshold)
            lift_stable = lift_consecutive >= cfg.lift_sustain_steps

            obj_z_var = _window_var(hist, 'obj_z', cfg.carry_window)

            near_target = s['obj_target_dist'] < cfg.release_target_dist_max
            in_preplace_band = (cfg.preplace_target_dist_min <= s['obj_target_dist']
                                <= cfg.preplace_target_dist_max)

            obj_falling = s.get('eef_vz', 0) < -0.01 and obj_lifted

            phase = prev_phase

            if near_target and not obj_falling:
                phase = 'release_safe'
            elif was_lifted and not obj_lifted and s['obj_eef_dist'] < cfg.regrasp_eef_obj_dist_max:
                phase = 'recovery_or_regrasp'
            elif (prev_phase in ('stable_carry', 'first_lift') and
                  in_preplace_band and not near_target):
                phase = 'pre_place_unsupported'
            elif (lift_stable and obj_eef_close and
                  obj_eef_var is not None and obj_eef_var < cfg.eef_obj_dist_stable_var and
                  obj_z_var is not None and obj_z_var < cfg.carry_obj_z_var_max):
                phase = 'stable_carry'
            elif lift_stable and not was_lifted:
                phase = 'first_lift'
            elif grasp_stable and obj_eef_close:
                phase = 'stable_grasp'
            elif gripper_closed:
                phase = 'grasp_close'
            elif not gripper_closed and not obj_lifted:
                phase = 'approach'

            if lift_stable:
                was_lifted = True
            prev_phase = phase

            confidence = 0.8 if phase in ('stable_carry', 'release_safe', 'stable_grasp') else \
                         0.6 if phase in ('first_lift', 'pre_place_unsupported') else \
                         0.5 if phase == 'approach' else 0.4

            labels.append({
                'step_idx': s['step_idx'], 'policy_step_idx': s['policy_step_idx'],
                'phase': phase,
                'failure_critical': phase in FAILURE_CRITICAL_PHASES,
                'confidence': round(confidence, 3), 'abstain_reason': '',
                'gripper_command': s['gripper_command'],
                'gripper_width': s.get('gripper_width', 0),
                'gripper_closed': gripper_closed,
                'obj_z': round(s['obj_z'], 6), 'obj_z0': round(obj_z0, 6) if obj_z0 else None,
                'obj_lifted': obj_lifted, 'obj_eef_dist': round(s['obj_eef_dist'], 6),
                'obj_target_dist': round(s['obj_target_dist'], 6),
                'close_consecutive': close_consecutive,
                'lift_consecutive': lift_consecutive,
            })

        return labels

    def find_teacher_anchor(self, labels: List[dict]) -> dict:
        fc_labels = [l for l in labels if l['failure_critical']]
        if not fc_labels:
            return {'anchor': -1, 'phase': 'none', 'reason': 'no_failure_critical_phase'}

        pp_labels = [l for l in fc_labels if l['phase'] == 'pre_place_unsupported']
        if pp_labels:
            best = max(pp_labels, key=lambda l: l['confidence'])
            return {'anchor': best['step_idx'], 'phase': best['phase'],
                    'confidence': best['confidence'],
                    'reason': 'pre_place_unsupported_preferred'}

        sc_labels = [l for l in fc_labels if l['phase'] == 'stable_carry']
        if sc_labels:
            mid = sc_labels[len(sc_labels) // 2]
            return {'anchor': mid['step_idx'], 'phase': mid['phase'],
                    'confidence': mid['confidence'],
                    'reason': 'stable_carry_midpoint'}

        return {'anchor': -1, 'phase': 'none', 'reason': 'unexpected'}


def calibrate_thresholds(trajectory_paths: List[str]) -> TeacherConfig:
    cfg = TeacherConfig()
    all_widths_closed = []
    all_obj_z_deltas = []
    all_obj_eef_dists = []
    all_target_dists = []

    for path in trajectory_paths:
        with open(path) as f:
            records = [json.loads(line) for line in f]
        states = [_extract_state(r) for r in records if r.get('teacher_privileged_state_available')]
        if not states:
            continue
        obj_z0 = states[0]['obj_z']

        close_steps = [s for s in states if s['gripper_command'] > 0.5]
        for s in close_steps:
            if s['gripper_width'] > 0:
                all_widths_closed.append(s['gripper_width'])

        for s in states:
            all_obj_z_deltas.append(s['obj_z'] - obj_z0)

        grasp_steps = [s for s in states if s['gripper_width'] < 0.03]
        for s in grasp_steps:
            all_obj_eef_dists.append(s['obj_eef_dist'])

        for s in states:
            d = s.get('obj_target_dist', 0)
            if d > 0:
                all_target_dists.append(d)

    if all_widths_closed:
        cfg.gripper_close_threshold = round(np.percentile(all_widths_closed, 95), 4)
    if all_obj_z_deltas:
        pos_deltas = [z for z in all_obj_z_deltas if z > 0]
        cfg.lift_z_threshold = round(max(0.005, np.percentile(pos_deltas, 25) if pos_deltas else 0.015), 4)
    if all_obj_eef_dists:
        cfg.eef_obj_dist_max = round(np.percentile(all_obj_eef_dists, 90), 4)
    if all_target_dists:
        near_dists = [d for d in all_target_dists if d < 0.3]
        cfg.release_target_dist_max = round(np.percentile(near_dists, 20) if near_dists else 0.08, 4)
        mid_dists = [d for d in all_target_dists if 0.05 < d < 0.4]
        cfg.preplace_target_dist_max = round(np.percentile(mid_dists, 60) if mid_dists else 0.25, 4)

    cfg.calibrated_from = '%d_butter_clean_trajectories' % len(trajectory_paths)
    cfg.version = 'v2_teacher_calibrated_0'
    return cfg
