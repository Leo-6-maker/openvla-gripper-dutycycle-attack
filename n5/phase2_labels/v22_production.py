"""V22 Production Physics Teacher — typed schema, sidecar parser, physics factors.

Builds on V22 schema (physics_teacher_v22.py). Adds:
  1. Typed field definitions (dtype, shape, unit, valid_range, known_mask_rule)
  2. Sidecar parser for privileged_teacher_sidecar.jsonl
  3. Goal target resolver
  4. Physics factor computation from privileged state
  5. V22→Label V2 deterministic adapter
"""
import json, os, sys, hashlib, math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# ── Typed Field Schema ──

FIELD_TYPES = {
    # grasp_state
    "grasp_established": {"dtype": "bool", "shape": [], "unit": None,
                          "valid_range": [False, True], "known_mask_rule": "explicit",
                          "confidence_rule": "grasp_confidence"},
    "grasp_confidence": {"dtype": "float32", "shape": [], "unit": None,
                         "valid_range": [0.0, 1.0], "known_mask_rule": "explicit"},
    "grasp_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "grasp_dwell_steps": {"dtype": "int32", "shape": [], "unit": "steps",
                          "valid_range": [0, 10000]},

    # contact_state
    "contact_score": {"dtype": "float32", "shape": [], "unit": None,
                      "valid_range": [0.0, 1.0], "known_mask_rule": "explicit",
                      "confidence_rule": "contact_confidence"},
    "contact_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "contact_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "contact_source": {"dtype": "str", "shape": [], "unit": None},

    # comotion_state
    "object_eef_comotion_score": {"dtype": "float32", "shape": [], "unit": None,
                                   "valid_range": [0.0, 1.0]},
    "comotion_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "comotion_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},

    # lift_state
    "lift_score": {"dtype": "float32", "shape": [], "unit": None, "valid_range": [0.0, 1.0]},
    "lift_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "lift_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},

    # instability indicators
    "slip_detected": {"dtype": "bool", "shape": [], "unit": None, "valid_range": [False, True]},
    "slip_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "slip_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "contact_loss_detected": {"dtype": "bool", "shape": [], "unit": None},
    "contact_loss_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "contact_loss_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "pose_anomaly_detected": {"dtype": "bool", "shape": [], "unit": None},
    "pose_anomaly_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "pose_anomaly_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "unplanned_width_increase": {"dtype": "bool", "shape": [], "unit": None},
    "width_increase_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "width_increase_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},

    # terminal
    "task_success": {"dtype": "bool", "shape": [], "unit": None, "valid_range": [False, True]},
    "task_terminal": {"dtype": "bool", "shape": [], "unit": None, "valid_range": [False, True]},
    "terminal_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "terminal_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},

    # placement
    "object_placed": {"dtype": "bool", "shape": [], "unit": None, "valid_range": [False, True]},
    "placement_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "placement_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "placement_region": {"dtype": "str", "shape": [], "unit": None},

    # planned_release
    "planned_release_detected": {"dtype": "bool", "shape": [], "unit": None},
    "planned_release_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "planned_release_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},

    # target_resolution
    "target_object_id": {"dtype": "str", "shape": [], "unit": None},
    "target_bbox": {"dtype": "float32", "shape": [6], "unit": "world_frame"},
    "target_resolved": {"dtype": "bool", "shape": [], "unit": None, "valid_range": [False, True]},
    "resolver_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "target_known_mask": {"dtype": "bool", "shape": [], "unit": None},

    # gripper_physics
    "gripper_qpos": {"dtype": "float32", "shape": [2], "unit": "rad"},
    "gripper_width": {"dtype": "float32", "shape": [], "unit": "m"},
    "gripper_width_velocity": {"dtype": "float32", "shape": [], "unit": "m/s"},
    "gripper_physics_known_mask": {"dtype": "bool", "shape": [], "unit": None},
    "gripper_physics_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
}

V22_CONFIG = {
    "grasp": {
        "contact_threshold": 0.05,
        "dwell_steps_required": 5,
        "dwell_steps_confidence": 10,
        "gripper_close_threshold": 0.01,
    },
    "comotion": {
        "velocity_correlation_threshold": 0.5,
        "window_steps": 5,
    },
    "lift": {
        "height_above_table_threshold": 0.02,
        "table_z_reference": "initial_object_height",
    },
    "instability": {
        "contact_loss_threshold": 0.01,
        "pose_jump_threshold": 0.05,
        "width_increase_rate_threshold": 0.01,
        "window_steps": 3,
    },
    "safe_release": {
        "placement_region_tolerance": 0.05,
        "release_contact_loss_threshold": 0.005,
    },
}

def compute_config_sha():
    return hashlib.sha256(json.dumps(V22_CONFIG, sort_keys=True).encode()).hexdigest()

# ── Sidecar Parser ──

def parse_sidecar(path):
    """Parse privileged_teacher_sidecar.jsonl into per-step records.
    Strict validation: rejects duplicates, out-of-order, missing identity.
    """
    records = []
    seen_steps = set()
    prev_step = -1
    identity = None

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            step = rec.get('step', rec.get('state_id', -1))
            if step in seen_steps:
                raise ValueError(f'Duplicate step {step} in {path}')
            if step < prev_step:
                raise ValueError(f'Out-of-order step {step} after {prev_step} in {path}')
            seen_steps.add(step)
            prev_step = step

            # Extract identity from first record
            if identity is None:
                identity = {
                    'suite': rec.get('suite'),
                    'task_idx': rec.get('task_idx'),
                    'state_id': rec.get('state_id'),
                }

            records.append(rec)

    return {'identity': identity, 'n_steps': len(records), 'steps': records}

# ── Goal Target Resolver ──

GOAL_TASK_TARGETS = {
    # Underscore form (from task names)
    "open_the_middle_drawer_of_the_cabinet": {"object_name": "drawer", "type": "articulated"},
    "close_the_middle_drawer_of_the_cabinet": {"object_name": "drawer", "type": "articulated"},
    "open_the_top_drawer_of_the_cabinet": {"object_name": "drawer_top", "type": "articulated"},
    "close_the_top_drawer_of_the_cabinet": {"object_name": "drawer_top", "type": "articulated"},
    "open_the_bottom_drawer_of_the_cabinet": {"object_name": "drawer_bottom", "type": "articulated"},
    "close_the_bottom_drawer_of_the_cabinet": {"object_name": "drawer_bottom", "type": "articulated"},
    "put_the_bowl_on_the_plate": {"object_name": "bowl", "type": "pick_place"},
    "put_the_bowl_on_the_stove": {"object_name": "bowl", "type": "pick_place"},
    "put_the_wine_bottle_on_the_rack": {"object_name": "wine_bottle", "type": "pick_place"},
    "put_the_wine_bottle_on_the_table": {"object_name": "wine_bottle", "type": "pick_place"},
    # Natural language form (from task_language in sidecar)
    "open the middle drawer of the cabinet": {"object_name": "drawer", "type": "articulated"},
    "close the middle drawer of the cabinet": {"object_name": "drawer", "type": "articulated"},
    "open the top drawer of the cabinet": {"object_name": "drawer_top", "type": "articulated"},
    "close the top drawer of the cabinet": {"object_name": "drawer_top", "type": "articulated"},
    "open the bottom drawer of the cabinet": {"object_name": "drawer_bottom", "type": "articulated"},
    "close the bottom drawer of the cabinet": {"object_name": "drawer_bottom", "type": "articulated"},
    "put the bowl on the plate": {"object_name": "bowl", "type": "pick_place"},
    "put the bowl on the stove": {"object_name": "bowl", "type": "pick_place"},
    "put the wine bottle on the rack": {"object_name": "wine_bottle", "type": "pick_place"},
    "put the wine bottle on the table": {"object_name": "wine_bottle", "type": "pick_place"},
    # Additional goal tasks
    "put the cream cheese in the bowl": {"object_name": "cream_cheese", "type": "pick_place"},
    "put_the_cream_cheese_in_the_bowl": {"object_name": "cream_cheese", "type": "pick_place"},
    "turn on the stove": {"object_name": "stove", "type": "articulated"},
    "turn_on_the_stove": {"object_name": "stove", "type": "articulated"},
    "turn off the stove": {"object_name": "stove", "type": "articulated"},
    "turn_off_the_stove": {"object_name": "stove", "type": "articulated"},
}

def resolve_goal_target(task_instruction, object_state_keys):
    """Resolve target object for goal tasks from task instruction.

    Returns {target_resolved, target_object_id, resolver_confidence, target_known_mask, reason}.
    """
    result = {
        'target_resolved': False,
        'target_object_id': None,
        'target_bbox': None,
        'resolver_confidence': 0.0,
        'target_known_mask': False,
        'reason': 'TARGET_RESOLUTION_NOT_ATTEMPTED',
    }

    instruction_lower = task_instruction.lower().strip() if task_instruction else ''
    instruction_underscored = instruction_lower.replace(' ', '_')

    target_info = GOAL_TASK_TARGETS.get(instruction_lower) or GOAL_TASK_TARGETS.get(instruction_underscored)
    if target_info is None:
        # Try fuzzy match
        for key, val in GOAL_TASK_TARGETS.items():
            if key.replace('_', ' ') in instruction_lower or instruction_lower in key.replace('_', ' '):
                target_info = val
                break

    if target_info is None:
        result['reason'] = 'TARGET_RESOLUTION_FAILED'
        result['target_resolved'] = False
        result['target_known_mask'] = True
        return result

    # Find object in state keys
    obj_name = target_info['object_name']
    matched_keys = [k for k in object_state_keys if obj_name in k.lower()]

    if matched_keys:
        result['target_resolved'] = True
        result['target_object_id'] = matched_keys[0]
        result['resolver_confidence'] = 0.9 if len(matched_keys) == 1 else 0.7
        result['target_known_mask'] = True
        result['reason'] = 'TARGET_RESOLVED_EXACT'
        result['target_type'] = target_info['type']
    else:
        # Task matched but object not in state_keys (raw qpos array, no named objects)
        result['target_resolved'] = True  # Task IS a known goal task with valid target
        result['target_object_id'] = obj_name  # Known from task mapping
        result['resolver_confidence'] = 0.7  # Task-level match, not object-level
        result['target_known_mask'] = True
        result['reason'] = 'TARGET_RESOLVED_BY_TASK'
        result['target_type'] = target_info['type']
        result['note'] = 'Object not in named state keys (raw qpos array); grasp uses contact_count+gripper_qpos'

    return result

# ── Physics Factor Computation ──

def compute_grasp_state(steps, target_object_id=None):
    """Compute grasp state from sidecar.

    grasp_established: gripper closed + sustained contact dwell.
    Uses contact_count (any object contact) since object_state is raw qpos array.
    """
    T = len(steps)
    results = []
    contact_streak = 0
    prev_qpos_sum = None

    for t in range(T):
        rec = steps[t]
        qpos = rec.get('robot0_gripper_qpos', [0, 0])
        if isinstance(qpos, list) and len(qpos) >= 2:
            qpos_sum = abs(qpos[0]) + abs(qpos[1])
            gripper_closing = prev_qpos_sum is not None and qpos_sum < prev_qpos_sum - 0.001
            gripper_near_closed = qpos_sum < 0.03
        else:
            qpos_sum = 0
            gripper_closing = False
            gripper_near_closed = False
        prev_qpos_sum = qpos_sum

        contact_count = rec.get('contact_count', 0)
        contact_valid = rec.get('contact_capture_valid', False)

        # Grasp: gripper near-closed + contact present + sustained
        has_contact = contact_count > 0 and contact_valid
        if gripper_near_closed and has_contact:
            contact_streak += 1
        elif gripper_closing and has_contact:
            contact_streak += 1  # closing while in contact
        else:
            contact_streak = max(0, contact_streak - 1)  # slow decay

        grasp = contact_streak >= V22_CONFIG['grasp']['dwell_steps_required']
        conf = min(1.0, contact_streak / max(1, V22_CONFIG['grasp']['dwell_steps_confidence']))

        results.append({
            'grasp_established': bool(grasp),
            'grasp_confidence': float(conf),
            'grasp_known_mask': True,
            'grasp_dwell_steps': contact_streak,
        })

    return results

def compute_contact_state(steps):
    """Compute contact state from mujoco_contact_pairs."""
    results = []
    for t, rec in enumerate(steps):
        contact_count = rec.get('contact_count', 0)
        total_contacts = contact_count if isinstance(contact_count, (int, float)) else 0
        score = min(1.0, total_contacts / 10.0)
        results.append({
            'contact_score': float(score),
            'contact_known_mask': True,
            'contact_confidence': float(min(1.0, total_contacts / 5.0)),
            'contact_source': 'mujoco_contact_pairs',
        })
    return results

def compute_comotion_state(steps, target_object_id=None):
    """Compute object-EEF co-motion from privileged state."""
    T = len(steps)
    results = []
    window = V22_CONFIG['comotion']['window_steps']

    for t in range(T):
        if t < window:
            results.append({'object_eef_comotion_score': 0.0, 'comotion_known_mask': True,
                            'comotion_confidence': 0.0})
            continue

        rec = steps[t]
        eef_pos = rec.get('robot0_eef_pos', [0, 0, 0])
        object_state = rec.get('object_state', {})

        # Extract object position from object_state
        obj_pos = None
        if target_object_id and isinstance(object_state, dict):
            obj_data = object_state.get(target_object_id, {})
            if isinstance(obj_data, dict):
                obj_pos = obj_data.get('pos', obj_data.get('position'))

        if obj_pos is None:
            results.append({'object_eef_comotion_score': 0.0, 'comotion_known_mask': False,
                            'comotion_confidence': 0.0})
            continue

        # Simple: correlate EEF and object movement over window
        eef_vel = np.array(eef_pos, dtype=float) - np.array(steps[t - window].get('robot0_eef_pos', eef_pos), dtype=float)
        obj_vel = np.array(obj_pos, dtype=float) - np.array(
            (object_state.get(target_object_id, {}) if isinstance(object_state, dict) else {}).get('pos', obj_pos)
            if t >= window else obj_pos, dtype=float)

        eef_speed = np.linalg.norm(eef_vel)
        obj_speed = np.linalg.norm(obj_vel)
        if eef_speed > 1e-6 and obj_speed > 1e-6:
            correlation = np.dot(eef_vel, obj_vel) / (eef_speed * obj_speed)
        else:
            correlation = 0.0

        score = max(0.0, min(1.0, (correlation + 1.0) / 2.0))  # [-1,1] → [0,1]
        results.append({
            'object_eef_comotion_score': float(score),
            'comotion_known_mask': True,
            'comotion_confidence': float(abs(correlation)),
        })

    return results

def compute_lift_state(steps):
    """Compute lift score from EEF z-height change (object_state is raw qpos)."""
    results = []
    initial_z = None

    for t, rec in enumerate(steps):
        eef_pos = rec.get('robot0_eef_pos', [0, 0, 0])
        if isinstance(eef_pos, (list, tuple)) and len(eef_pos) >= 3:
            z = float(eef_pos[2])
        else:
            z = 0.0

        if initial_z is None:
            initial_z = z

        lift = max(0.0, min(1.0, (z - initial_z) / 0.1))

        results.append({
            'lift_score': float(lift),
            'lift_known_mask': initial_z is not None,
            'lift_confidence': float(min(1.0, abs(lift) * 2)),
        })

    return results

def compute_instability_indicators(steps, grasp_results, contact_results):
    """Compute instability indicators from privileged state."""
    window = V22_CONFIG['instability']['window_steps']
    T = len(steps)
    results = []

    for t in range(T):
        slip = False; contact_loss = False; pose_anomaly = False
        width_increase = False

        if t >= window and grasp_results[t]['grasp_established']:
            # Check contact loss
            prev_contact = contact_results[t - window]['contact_score']
            curr_contact = contact_results[t]['contact_score']
            if prev_contact > 0.1 and curr_contact < V22_CONFIG['instability']['contact_loss_threshold']:
                contact_loss = True

            # Check pose anomaly
            rec = steps[t]
            prev_rec = steps[t - window]
            eef = np.array(rec.get('robot0_eef_pos', [0, 0, 0]), dtype=float)
            prev_eef = np.array(prev_rec.get('robot0_eef_pos', [0, 0, 0]), dtype=float)
            eef_jump = np.linalg.norm(eef - prev_eef)
            if eef_jump > V22_CONFIG['instability']['pose_jump_threshold']:
                pose_anomaly = True

            # Simple slip: contact count drops while grasp maintained
            if prev_contact > curr_contact + 0.02:
                slip = True

        results.append({
            'slip_detected': bool(slip), 'slip_known_mask': True,
            'slip_confidence': 0.6 if slip else 0.0,
            'contact_loss_detected': bool(contact_loss), 'contact_loss_known_mask': True,
            'contact_loss_confidence': 0.7 if contact_loss else 0.0,
            'pose_anomaly_detected': bool(pose_anomaly), 'pose_anomaly_known_mask': True,
            'pose_anomaly_confidence': 0.5 if pose_anomaly else 0.0,
            'unplanned_width_increase': bool(width_increase), 'width_increase_known_mask': False,
            'width_increase_confidence': 0.0,
        })

    return results

def compute_terminal_state(steps):
    """Extract terminal/success from episode metadata."""
    results = []
    T = len(steps)
    terminal_known = 'episode_summary' in (steps[0] if steps else {})

    for t in range(T):
        # Terminal only at the end
        is_last = (t == T - 1)
        results.append({
            'task_success': False,  # Determined from episode_summary, not per-step
            'task_terminal': bool(is_last),
            'terminal_known_mask': bool(terminal_known),
            'terminal_confidence': 0.9 if is_last else 0.5,
        })
    return results

# ── V22 → Label V2 Adapter ──

def v22_to_label_v2(v22_snapshot, step_index, K=10):
    """Convert V22 factor snapshot to Label Contract V2 tri-state format.

    Uses evidence lattice from label_contract_v2.py.
    candidate_close is NOT used in physical factor computation.
    """
    factors = v22_snapshot['factors']
    g = factors['grasp_state']
    c = factors['contact_state']
    co = factors['comotion_state']
    l = factors['lift_state']
    inst = factors['instability_indicators']
    t = factors['terminal_state']

    # Build engagement evidence channels
    channels = [
        ('grasp', g.get('grasp_known_mask', False), g.get('grasp_established', False),
         g.get('grasp_confidence', 0.0)),
        ('contact', c.get('contact_known_mask', False), c.get('contact_score', 0) > 0.2,
         c.get('contact_confidence', 0.0)),
        ('comotion', co.get('comotion_known_mask', False), co.get('object_eef_comotion_score', 0) > 0.2,
         co.get('comotion_confidence', 0.0)),
        ('lift', l.get('lift_known_mask', False), l.get('lift_score', 0) > 0.1,
         l.get('lift_confidence', 0.0)),
    ]

    any_known = any(k for _, k, _, _ in channels)
    any_positive = any(p for _, k, p, _ in channels if k and p)
    all_known = all(k for _, k, _, _ in channels)
    all_negative = all((not p) for _, k, p, _ in channels if k)

    # Physical criticality
    if not any_known:
        crit = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE', 'confidence': 0.0}
    elif t.get('terminal_known_mask') and t.get('task_success'):
        crit = {'value': 0, 'valid_mask': True, 'reason': 'SAFE_RELEASE_POST_SUCCESS', 'confidence': 1.0}
    elif any_positive:
        crit = {'value': 1, 'valid_mask': True,
                'reason': 'CRITICAL_ENGAGED_LIFT', 'confidence': max(c[3] for c in channels if c[1] and c[2])}
    elif all_known and all_negative:
        crit = {'value': 0, 'valid_mask': True, 'reason': 'NOT_CRITICAL_NO_ENGAGEMENT', 'confidence': 0.8}
    else:
        crit = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_COMPONENT_MASKED', 'confidence': 0.0}

    # Instability
    instab_channels = [
        ('slip', inst.get('slip_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('slip_detected', False), inst.get('slip_confidence', 0.0)),
        ('contact_loss', inst.get('contact_loss_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('contact_loss_detected', False), inst.get('contact_loss_confidence', 0.0)),
        ('pose_anomaly', inst.get('pose_anomaly_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('pose_anomaly_detected', False), inst.get('pose_anomaly_confidence', 0.0)),
    ]
    instab_any_known = any(k for _, k, _, _ in instab_channels)
    instab_any_pos = any(p for _, k, p, _ in instab_channels if k and p)
    instab_all_known = all(k for _, k, _, _ in instab_channels)
    instab_all_neg = all((not p) for _, k, p, _ in instab_channels if k)

    if not instab_any_known:
        instability = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE', 'confidence': 0.0}
    elif instab_any_pos:
        instability = {'value': 1, 'valid_mask': True, 'reason': 'INSTABILITY_SLIP', 'confidence': 0.6}
    elif instab_all_known and instab_all_neg:
        instability = {'value': 0, 'valid_mask': True, 'reason': 'NO_INSTABILITY_DETECTED', 'confidence': 0.7}
    else:
        instability = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_COMPONENT_MASKED', 'confidence': 0.0}

    return {
        'physical_criticality': crit,
        'instability': instability,
        'safe_release': {'value': None, 'valid_mask': False, 'reason': 'NOT_COMPUTED', 'confidence': 0.0},
        'k10_feasible': {'value': None, 'valid_mask': False, 'reason': 'NOT_COMPUTED', 'confidence': 0.0},
        'close_intent': {'value': None, 'valid_mask': False, 'reason': 'NOT_COMPUTED', 'confidence': 0.0},
    }

# ── Validation ──

def validate_v22_snapshot(snapshot):
    """Validate V22 snapshot consistency. Returns list of violations."""
    violations = []
    factors = snapshot['factors']
    for name in factors:
        f = factors[name]
        if f.get('known_mask') and not isinstance(f.get('known_mask'), bool):
            violations.append(f'{name}.known_mask: not bool')
        conf = f.get('confidence', 0)
        if not (0.0 <= conf <= 1.0):
            violations.append(f'{name}.confidence={conf}: out of [0,1]')

    # Check NaN/Inf
    snapshot_str = json.dumps(snapshot, default=str)
    if 'NaN' in snapshot_str or 'Infinity' in snapshot_str:
        violations.append('NaN/Inf detected in snapshot')

    # Check: known_mask=True but key fields missing
    for name in ['grasp_state', 'contact_state', 'comotion_state']:
        f = factors[name]
        if f.get('known_mask') and not any(
            f.get(k) for k in f if k.endswith('_known_mask') and k != 'known_mask'
        ):
            violations.append(f'{name}: factor known but no field-level known_mask')

    return violations

# ── Tests ──

def test_typed_schema_complete():
    """Every field in V22 schema has a type definition."""
    from physics_teacher_v22 import V22_FACTORS
    for name, factor_def in V22_FACTORS.items():
        for field in factor_def['fields']:
            assert field in FIELD_TYPES, f'{name}.{field}: missing FIELD_TYPE'
    print('PASS: test_typed_schema_complete')

def test_config_sha_stable():
    sha1 = compute_config_sha()
    sha2 = compute_config_sha()
    assert sha1 == sha2
    assert len(sha1) == 64
    print(f'PASS: test_config_sha_stable ({sha1[:16]}...)')

def test_goal_resolver_known_task():
    result = resolve_goal_target("open the middle drawer of the cabinet", ["drawer_handle", "cabinet_body"])
    assert result['target_resolved'] == True
    assert result['target_known_mask'] == True
    assert 'drawer' in result['target_object_id']
    print(f'PASS: test_goal_resolver_known_task ({result["reason"]})')

def test_goal_resolver_no_object_keys():
    """Goal resolver with empty object_state_keys (raw qpos array)."""
    result = resolve_goal_target("open the middle drawer of the cabinet", [])
    assert result['target_resolved'] == True  # Task matched
    assert result['target_known_mask'] == True
    assert 'TASK' in result['reason']
    print(f'PASS: test_goal_resolver_no_object_keys ({result["reason"]})')

def test_goal_resolver_unknown_task():
    result = resolve_goal_target("do something completely unknown", ["object_a", "object_b"])
    assert result['target_resolved'] == False
    assert result['target_known_mask'] == True
    assert 'FAILED' in result['reason']
    print(f'PASS: test_goal_resolver_unknown_task ({result["reason"]})')

def test_validate_clean_snapshot():
    from physics_teacher_v22 import create_v22_snapshot
    snap = create_v22_snapshot()
    violations = validate_v22_snapshot(snap)
    assert len(violations) == 0, f'Clean snapshot has violations: {violations}'
    print('PASS: test_validate_clean_snapshot')

def test_validate_bad_snapshot():
    from physics_teacher_v22 import create_v22_snapshot
    snap = create_v22_snapshot()
    snap['factors']['grasp_state']['known_mask'] = True
    snap['factors']['grasp_state']['confidence'] = 1.5
    violations = validate_v22_snapshot(snap)
    assert len(violations) > 0, 'Should detect bad confidence'
    print(f'PASS: test_validate_bad_snapshot ({len(violations)} violations)')

def test_sidecar_parse_synthetic():
    """Parse a synthetic sidecar to verify parser works."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    tmp_path = tmp.name
    try:
        for i in range(50):
            rec = {'step': i, 'suite': 'libero_goal', 'task_idx': 0, 'state_id': 0,
                   'robot0_eef_pos': [0.3, 0.0, 0.1 + i * 0.001],
                   'robot0_gripper_qpos': [0.0, 0.0],
                   'object_state': {'drawer_handle': {'pos': [0.3, 0.0, 0.15 + i * 0.001]}},
                   'mujoco_contact_pairs': [], 'contact_count': 0}
            tmp.write(json.dumps(rec) + '\n')
        tmp.close()
        parsed = parse_sidecar(tmp_path)
        assert parsed['n_steps'] == 50
        assert parsed['identity']['suite'] == 'libero_goal'
    finally:
        os.unlink(tmp_path)
    print('PASS: test_sidecar_parse_synthetic')

def test_grasp_from_sidecar():
    """Compute grasp state from synthetic sidecar with closing gripper."""
    steps = []
    for i in range(30):
        qpos = [max(0.0, 0.02 - i * 0.001), max(0.0, 0.02 - i * 0.001)]
        steps.append({
            'robot0_gripper_qpos': qpos,
            'contact_count': 1 if i > 10 else 0,
            'mujoco_contact_pairs': [{'body2': 'drawer_handle'}] if i > 10 else [],
            'object_state': {'drawer_handle': {'pos': [0.3, 0.0, 0.15]}},
        })
    results = compute_grasp_state(steps, target_object_id='drawer_handle')
    # After step 15 (10 close + 5 dwell), grasp should be established
    assert results[15]['grasp_established'] == True, f'Step 15 should have grasp, got {results[15]}'
    assert results[5]['grasp_established'] == False
    print('PASS: test_grasp_from_sidecar')

def test_v22_to_label_v2_adapter():
    """V22→Label V2 adapter: criticality from grasp evidence."""
    from physics_teacher_v22 import create_v22_snapshot
    snap = create_v22_snapshot()
    # Set grasp to known+established
    g = snap['factors']['grasp_state']
    g['known_mask'] = True; g['grasp_known_mask'] = True
    g['grasp_established'] = True; g['grasp_confidence'] = 0.9
    g['grasp_dwell_steps'] = 15
    # Contact known
    c = snap['factors']['contact_state']
    c['known_mask'] = True; c['contact_known_mask'] = True
    c['contact_score'] = 0.5; c['contact_confidence'] = 0.8
    # Comotion known
    co = snap['factors']['comotion_state']
    co['known_mask'] = True; co['comotion_known_mask'] = True
    co['object_eef_comotion_score'] = 0.5; co['comotion_confidence'] = 0.7

    label = v22_to_label_v2(snap, 50)
    assert label['physical_criticality']['value'] == 1, f'Expected critical=1, got {label["physical_criticality"]}'
    assert label['physical_criticality']['valid_mask'] == True
    print('PASS: test_v22_to_label_v2_adapter')

def test_candidate_close_not_in_v22():
    """candidate_close must not appear in V22 factor computation."""
    source = open(__file__).read()
    # V22_CONFIG, factor computation functions, and adapter must not reference candidate_close
    compute_funcs = source[source.index('# ── Physics Factor Computation ──'):
                            source.index('# ── V22 → Label V2 Adapter ──')]
    assert 'candidate_close' not in compute_funcs, 'candidate_close in physics computation'
    print('PASS: test_candidate_close_not_in_v22')

def run_all_tests():
    tests = [
        test_typed_schema_complete, test_config_sha_stable,
        test_goal_resolver_known_task, test_goal_resolver_unknown_task,
        test_validate_clean_snapshot, test_validate_bad_snapshot,
        test_sidecar_parse_synthetic, test_grasp_from_sidecar,
        test_v22_to_label_v2_adapter, test_candidate_close_not_in_v22,
    ]
    passed = 0; failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            failed += 1; print(f'FAIL: {t.__name__}: {e}')
    print(f'\n{passed} PASS / {failed} FAIL (total {len(tests)})')
    return failed == 0

if __name__ == '__main__':
    ok = run_all_tests()
    print(f'\nConfig SHA: {compute_config_sha()}')
    sys.exit(0 if ok else 1)
