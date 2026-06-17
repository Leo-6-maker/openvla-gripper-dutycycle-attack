#!/usr/bin/env python3
"""V2 Privileged Teacher: clean-only phase labeling from MuJoCo privileged state.

Gripper semantics (matching OpenVLA->LIBERO postprocess and frozen D5):
  raw_gripper <= 0.5  -> CLOSE intent
  raw_gripper >  0.5  -> OPEN intent
  env_gripper <  0    -> OPEN command
  env_gripper >  0    -> CLOSE command

Rules:
  - ONLY clean-simulation privileged state (object/target/EEF/gripper).
  - NEVER sees attack outcome, VIS, RAND, or task failure.
  - Fail-closed: abstain on any missing privileged field (no default-zero fill).
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

# Required privileged fields for valid labeling
REQUIRED_PRIVILEGED_FIELDS = [
    'object_pose_json', 'target_pose_json',
    'object_to_target_distance', 'object_eef_distance',
    'gripper_command',  # raw_gripper from OpenVLA
    'eef_x', 'eef_y', 'eef_z',
]


@dataclass
class TeacherConfig:
    grasp_close_sustain: int = 3       # consecutive CLOSE commands for stable grasp
    grasp_open_proxy_max: float = 0.12 # gripper_opening_proxy below this = closed plateau
    eef_obj_dist_max: float = 0.12     # max obj-EEF distance during grasp
    eef_obj_dist_stable_var: float = 0.005  # max variance for stable following
    lift_z_threshold: float = 0.015    # obj_z - obj_z0 > this = lifted
    lift_sustain_steps: int = 2        # consecutive lifted steps
    carry_obj_z_var_max: float = 0.01  # max z variance during carry
    carry_window: int = 8              # window for carry stability
    preplace_target_dist_min: float = 0.05
    preplace_target_dist_max: float = 0.25
    release_target_dist_max: float = 0.08
    regrasp_eef_obj_dist_max: float = 0.10
    stability_window: int = 5
    version: str = 'v2_teacher_draft_0'
    calibrated_from: str = 'uncalibrated_draft'


def _check_required_fields(rec: dict) -> List[str]:
    """Return list of missing required privileged fields."""
    missing = []
    for fld in REQUIRED_PRIVILEGED_FIELDS:
        val = rec.get(fld)
        if val is None or val == '' or val == 'nan':
            missing.append(fld)
    return missing


def _extract_state(rec: dict, fail_closed: bool = True) -> Optional[dict]:
    """Extract numerical state from step record. Returns None if fail-closed on missing fields."""
    if fail_closed:
        missing = _check_required_fields(rec)
        if missing:
            return None

    obj_pose_str = rec.get('object_pose_json', '')
    if not obj_pose_str:
        return None
    obj_pose = json.loads(obj_pose_str)

    target_pose_str = rec.get('target_pose_json', '')
    target_pose = json.loads(target_pose_str) if target_pose_str else [float('nan')]*3

    obj_z0_early = rec.get('obj_z0', None)
    obj_z0_early = float(obj_z0_early) if obj_z0_early is not None else None

    # Raw gripper from OpenVLA: <= 0.5 = CLOSE, > 0.5 = OPEN
    raw_gripper = float(rec.get('gripper_command', float('nan')))

    return {
        'obj_x': float(obj_pose[0]), 'obj_y': float(obj_pose[1]),
        'obj_z': float(obj_pose[2]),
        'target_x': float(target_pose[0]), 'target_y': float(target_pose[1]),
        'target_z': float(target_pose[2]),
        'obj_target_dist': float(rec.get('object_to_target_distance', float('nan'))),
        'obj_eef_dist': float(rec.get('object_eef_distance', float('nan'))),
        'gripper_qpos': float(rec.get('gripper_qpos', float('nan'))),
        'gripper_opening_proxy': float(rec.get('gripper_width', rec.get('gripper_opening_proxy', float('nan')))),
        'raw_gripper': raw_gripper,
        'gripper_close': raw_gripper <= 0.5,   # CORRECTED: <=0.5 = CLOSE intent
        'eef_x': float(rec.get('eef_x', float('nan'))),
        'eef_y': float(rec.get('eef_y', float('nan'))),
        'eef_z': float(rec.get('eef_z', float('nan'))),
        'eef_vx': float(rec.get('eef_vx', 0.0)),
        'eef_vy': float(rec.get('eef_vy', 0.0)),
        'eef_vz': float(rec.get('eef_vz', 0.0)),  # NOW EXTRACTED
        'step_idx': int(rec.get('step_idx', 0)),
        'policy_step_idx': int(rec.get('policy_step_idx', -1)),
        'has_priv': bool(rec.get('teacher_privileged_state_available', False)),
        'obj_z0_early': obj_z0_early,
    }


def _is_valid(v) -> bool:
    if v is None: return False
    if isinstance(v, float) and math.isnan(v): return False
    return True


def _window_mean(history: List[dict], key: str, window: int) -> Optional[float]:
    vals = [s[key] for s in history[-window:] if _is_valid(s.get(key))]
    return float(np.mean(vals)) if vals else None


def _window_var(history: List[dict], key: str, window: int) -> Optional[float]:
    vals = [s[key] for s in history[-window:] if _is_valid(s.get(key))]
    return float(np.var(vals)) if len(vals) >= 3 else None


def _consecutive_count(history: List[dict], key: str, condition_fn) -> int:
    n = 0
    for s in reversed(history):
        val = s.get(key)
        if _is_valid(val) and condition_fn(val):
            n += 1
        else:
            break
    return n


class V2PrivilegedTeacher:

    def __init__(self, config: TeacherConfig = None):
        self.cfg = config or TeacherConfig()

    def label_trajectory(self, step_records: List[dict]) -> List[dict]:
        states = []
        for r in step_records:
            if not r.get('teacher_privileged_state_available') or r.get('phase') == 'wait':
                states.append(None)
            else:
                s = _extract_state(r, fail_closed=True)
                states.append(s)

        labels = []
        prev_phase = 'approach'
        obj_z0 = None
        was_lifted = False
        cfg = self.cfg

        for i, s in enumerate(states):
            if s is None:
                labels.append({
                    'step_idx': int(step_records[i].get('step_idx', i)),
                    'phase': 'abstain_unsupported', 'failure_critical': False,
                    'confidence': 0.0,
                    'abstain_reason': 'no_privilege_or_missing_fields',
                })
                continue

            hist = [x for x in states[:i + 1] if x is not None]

            if obj_z0 is None:
                obj_z0 = s['obj_z']

            # ── Gripper: CLOSE = raw <= 0.5 (CORRECTED semantics) ──
            gripper_close = s['gripper_close']
            close_consecutive = _consecutive_count(hist, 'gripper_close', lambda c: c)
            grasp_stable = close_consecutive >= cfg.grasp_close_sustain

            # ── Opening proxy plateau: proxy below threshold during grasp ──
            opening_ok = _is_valid(s['gripper_opening_proxy']) and s['gripper_opening_proxy'] < cfg.grasp_open_proxy_max

            # ── Object-EEF proximity ──
            obj_eef_close = _is_valid(s['obj_eef_dist']) and s['obj_eef_dist'] < cfg.eef_obj_dist_max
            obj_eef_var = _window_var(hist, 'obj_eef_dist', cfg.stability_window)

            # ── Lift detection ──
            obj_lifted = _is_valid(s['obj_z']) and (s['obj_z'] - obj_z0) > cfg.lift_z_threshold
            lift_consecutive = _consecutive_count(hist, 'obj_z',
                lambda z: _is_valid(z) and (z - obj_z0) > cfg.lift_z_threshold)
            lift_stable = lift_consecutive >= cfg.lift_sustain_steps

            obj_z_var = _window_var(hist, 'obj_z', cfg.carry_window)

            # ── Target proximity ──
            near_target = _is_valid(s['obj_target_dist']) and s['obj_target_dist'] < cfg.release_target_dist_max
            in_preplace_band = (_is_valid(s['obj_target_dist']) and
                cfg.preplace_target_dist_min <= s['obj_target_dist'] <= cfg.preplace_target_dist_max)

            # ── Object falling (descent velocity) ──
            obj_falling = s.get('eef_vz', 0.0) < -0.01 and obj_lifted

            # ── Phase detection ──
            phase = prev_phase

            if near_target and not obj_falling:
                phase = 'release_safe'
            elif was_lifted and not obj_lifted and obj_eef_close:
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
                # stable_grasp: sustained CLOSE command + EEF near object
                # (opening_proxy is task/mechanism-dependent, not a hard gate)
                phase = 'stable_grasp'
            elif gripper_close:
                phase = 'grasp_close'
            elif not gripper_close and not obj_lifted:
                phase = 'approach'

            if lift_stable:
                was_lifted = True
            prev_phase = phase

            # Confidence
            confidence = 0.8 if phase in ('stable_carry', 'release_safe', 'stable_grasp') else \
                         0.6 if phase in ('first_lift', 'pre_place_unsupported') else \
                         0.5 if phase == 'approach' else 0.4

            labels.append({
                'step_idx': s['step_idx'], 'policy_step_idx': s['policy_step_idx'],
                'phase': phase,
                'failure_critical': phase in FAILURE_CRITICAL_PHASES,
                'confidence': round(confidence, 3), 'abstain_reason': '',
                'raw_gripper': s['raw_gripper'],
                'gripper_close': gripper_close,
                'gripper_opening_proxy': s['gripper_opening_proxy'],
                'opening_proxy_ok': opening_ok,
                'obj_z': round(s['obj_z'], 6), 'obj_z0': round(obj_z0, 6),
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
    """Estimate thresholds from clean trajectories. gripper_close uses CLOSE command semantics."""
    cfg = TeacherConfig()
    all_open_proxy_closed = []
    all_obj_z_deltas = []
    all_obj_eef_dists_close = []
    all_target_dists = []

    for path in trajectory_paths:
        with open(path) as f:
            records = [json.loads(line) for line in f]
        states = [_extract_state(r, fail_closed=False) for r in records
                  if r.get('teacher_privileged_state_available')]
        states = [s for s in states if s is not None]
        if not states:
            continue
        obj_z0 = states[0]['obj_z']

        # Opening proxy when CLOSE command active (raw <= 0.5)
        close_cmd_steps = [s for s in states if s['gripper_close']]
        for s in close_cmd_steps:
            if _is_valid(s['gripper_opening_proxy']):
                all_open_proxy_closed.append(s['gripper_opening_proxy'])

        for s in states:
            if _is_valid(s['obj_z']):
                all_obj_z_deltas.append(s['obj_z'] - obj_z0)

        # EEF-obj distance when CLOSE command + opening proxy low
        grasp_steps = [s for s in states
                       if s['gripper_close'] and _is_valid(s['gripper_opening_proxy'])
                       and s['gripper_opening_proxy'] < 0.10]
        for s in grasp_steps:
            if _is_valid(s['obj_eef_dist']):
                all_obj_eef_dists_close.append(s['obj_eef_dist'])

        for s in states:
            d = s.get('obj_target_dist', 0)
            if _is_valid(d) and d > 0.001:
                all_target_dists.append(d)

    if all_open_proxy_closed:
        cfg.grasp_open_proxy_max = round(np.percentile(all_open_proxy_closed, 90), 4)
    if all_obj_z_deltas:
        pos_deltas = [z for z in all_obj_z_deltas if z > 0.005]
        cfg.lift_z_threshold = round(max(0.01, np.percentile(pos_deltas, 20) if pos_deltas else 0.015), 4)
    if all_obj_eef_dists_close:
        cfg.eef_obj_dist_max = round(np.percentile(all_obj_eef_dists_close, 90), 4)
    if all_target_dists:
        near_dists = [d for d in all_target_dists if d < 0.3]
        cfg.release_target_dist_max = round(np.percentile(near_dists, 15) if near_dists else 0.08, 4)
        mid_dists = [d for d in all_target_dists if 0.03 < d < 0.5]
        cfg.preplace_target_dist_max = round(np.percentile(mid_dists, 65) if mid_dists else 0.25, 4)

    cfg.calibrated_from = '%d_butter_clean_trajectories' % len(trajectory_paths)
    cfg.version = 'v2_teacher_fixed_semantics'
    return cfg
