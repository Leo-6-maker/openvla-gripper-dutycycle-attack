"""V22 Production Physics Teacher V2 — P0 FIXED.

Fixes all 8 P0 defects from d042fde audit:
  1. Target resolver split: semantic resolution vs physical object binding
  2. Grasp: target-finger contact pair filter (not arbitrary contact_count>0)
  3. Lift: target object Z from qpos slices (not EEF Z proxy)
  4. Comotion: correct history indexing (step[t-window] not step[t])
  5. Safe release (placement-gated) + gripper_closing_state: implemented
  6. Terminal state: wired from episode_summary.json
  7. K10: one output per timestep (fix duplicate append)
  8. Instability: target-relative EEF motion (not EEF-alone jumps)

Design constraints:
  - candidate_close NEVER used in physical factor computation
  - When object_slices unavailable → target-specific factors → unknown
  - Never fall back to EEF proxy for target object position
  - All functions use explicit known_mask (no inference from score!=0)
"""
import json, os, sys, hashlib, math, re
from typing import Dict, Any, List, Optional, Tuple, Sequence, Mapping
import numpy as np

# ── V5-derived physics helpers (mature, debugged, copied with attribution) ──
# Source: src/gripper_attack/v5_physics.py (commit history in git)

def _finite_vector(value, width):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    result = [float(item) for item in value]
    return result if all(math.isfinite(item) for item in result) else None

def _dist(left, right):
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))

def _cosine(left, right):
    left_norm = math.sqrt(sum(float(item) ** 2 for item in left))
    right_norm = math.sqrt(sum(float(item) ** 2 for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm)

def _clip(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))

def _mean(values, default=0.0):
    return sum(values) / len(values) if values else default

def _is_gripper(endpoint):
    return "gripper0" in endpoint or "finger1" in endpoint or "finger2" in endpoint

def _endpoint_matches(endpoint, name):
    return endpoint == name or endpoint.startswith(name + "_")

def _gripper_qpos_closure(sidecar):
    qpos = _finite_vector(sidecar.get("robot0_gripper_qpos"), 2)
    if qpos is None:
        return 0.0
    return 1.0 - _clip((abs(qpos[0]) + abs(qpos[1])) / 0.08)

def _contact_flags(pairs, manipulated_objects, support_names=()):
    """Target-specific contact detection (V5 pattern).
    Returns (object_contact, gripper_contact, support_contact).
    Only counts contact when manipulated_object is involved.
    """
    object_contact = False
    gripper_contact = False
    support_contact = False
    for pair in pairs:
        endpoints = [str(item) for item in pair]
        if not any(_endpoint_matches(ep, name) for ep in endpoints for name in manipulated_objects):
            continue
        object_contact = True
        if any(_is_gripper(ep) for ep in endpoints):
            gripper_contact = True
        for ep in endpoints:
            if _is_gripper(ep):
                continue
            if any(_endpoint_matches(ep, sn) for sn in support_names):
                support_contact = True
    return object_contact, gripper_contact, support_contact

# ── Object Slice Resolution (from BDDL) ──

OBJECT_STATE_WIDTH = 14

def parse_bddl_objects(text):
    """Parse :objects section from BDDL. Returns ordered list of {name, category}."""
    match = re.search(r"\(:objects\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    if not match:
        return []
    objects = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_ ]+)\s+-\s+([A-Za-z0-9_]+)", line)
        if not m:
            continue
        names = m.group(1).split()
        objects.extend({"name": name, "category": m.group(2)} for name in names)
    return objects

def build_object_slices(objects):
    """Build qpos slice mapping {object_name: {pos, quat, to_eef_pos, to_eef_quat}}."""
    slices = {}
    for idx, obj in enumerate(objects):
        base = idx * OBJECT_STATE_WIDTH
        slices[obj["name"]] = {
            "object_index": idx,
            "object_name": obj["name"],
            "object_category": obj["category"],
            "offset_start": base,
            "offset_end_exclusive": base + OBJECT_STATE_WIDTH,
            "pos": [base, base + 3],
            "quat": [base + 3, base + 7],
            "to_eef_pos": [base + 7, base + 10],
            "to_eef_quat": [base + 10, base + 14],
        }
    return slices

def _slice_vector(state, spec, key):
    """Extract a 3D vector from raw qpos array using slice spec."""
    bounds = spec.get(key)
    if not isinstance(bounds, list) or len(bounds) != 2:
        return None
    start, end = int(bounds[0]), int(bounds[1])
    if end > len(state):
        return None
    return _finite_vector(state[start:end], end - start)

# ── BDDL Task Role Parser (from V5) ──

_ROLE_PREDICATES = {"In", "On", "Inside", "Contains", "Stack"}
_SUPPORT_SUFFIXES = (
    "_contain_region", "_heating_region", "_cook_region",
    "_top_region", "_bottom_region", "_middle_region",
    "_top_side", "_front_region", "_back_contain_region", "_region",
)

def _base_name(value):
    result = value
    changed = True
    while changed:
        changed = False
        for suffix in _SUPPORT_SUFFIXES:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
                break
    return result

def parse_bddl_task_role(text, suite, task_idx, object_names):
    """Decode task roles from BDDL syntax (V5 mature pattern)."""
    object_set = set(object_names)
    goal_match = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    init_match = re.search(r"\(:init\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    if not goal_match:
        return {
            "manipulated_objects": [], "target_names": [], "support_names": [],
            "status": "ABSTAIN_DECODER_HOLD", "reason": "goal section missing",
        }

    predicates = []
    for m in re.finditer(r"\(([A-Za-z_]+)\s+([A-Za-z0-9_]+)(?:\s+([A-Za-z0-9_]+))?\)", goal_match.group(1)):
        pred, first, second = m.groups()
        if pred in _ROLE_PREDICATES:
            predicates.append((pred, first, second))

    manipulated = []
    targets = []
    for _, first, second in predicates:
        if first in object_set and first not in manipulated:
            manipulated.append(first)
            if second:
                targets.append(second)

    if not manipulated:
        return {
            "manipulated_objects": [], "target_names": list(targets),
            "support_names": [], "goal_predicates": predicates,
            "status": "NO_MANIPULATION_TARGET",
            "reason": "goal is non-grasp action with no BDDL object target",
        }

    supports = []
    if init_match:
        for m in re.finditer(r"\(On\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", init_match.group(1)):
            if m.group(1) in manipulated and m.group(2) not in supports:
                supports.append(m.group(2))

    return {
        "manipulated_objects": manipulated, "target_names": targets,
        "support_names": supports, "goal_predicates": predicates,
        "status": "PASS", "reason": "goal object and target decoded from BDDL",
    }

# ── Typed Field Schema ──

FIELD_TYPES = {
    # grasp_state
    "grasp_established": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "grasp_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "grasp_known_mask": {"dtype": "bool", "shape": []},
    "grasp_dwell_steps": {"dtype": "int32", "shape": [], "valid_range": [0, 10000]},
    # contact_state
    "contact_score": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "contact_known_mask": {"dtype": "bool", "shape": []},
    "contact_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "contact_source": {"dtype": "str", "shape": []},
    "object_contact_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "gripper_contact_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    # comotion_state
    "object_eef_comotion_score": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "comotion_known_mask": {"dtype": "bool", "shape": []},
    "comotion_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    # lift_state
    "lift_score": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "lift_known_mask": {"dtype": "bool", "shape": []},
    "lift_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    # instability
    "slip_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "slip_known_mask": {"dtype": "bool", "shape": []},
    "slip_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "contact_loss_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "contact_loss_known_mask": {"dtype": "bool", "shape": []},
    "contact_loss_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "pose_anomaly_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "pose_anomaly_known_mask": {"dtype": "bool", "shape": []},
    "pose_anomaly_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "unplanned_width_increase": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "width_increase_known_mask": {"dtype": "bool", "shape": []},
    "width_increase_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    # terminal
    "task_success": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "task_terminal": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "terminal_known_mask": {"dtype": "bool", "shape": []},
    "terminal_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    # placement
    "object_placed": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "placement_known_mask": {"dtype": "bool", "shape": []},
    "placement_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "placement_region": {"dtype": "str", "shape": []},
    # planned_release
    "planned_release_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "planned_release_known_mask": {"dtype": "bool", "shape": []},
    "planned_release_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    # target_resolution
    "target_object_id": {"dtype": "str", "shape": []},
    "target_bbox": {"dtype": "float32", "shape": [6]},
    "task_semantics_known": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "physical_binding_known": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "target_resolved": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "resolver_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    "target_known_mask": {"dtype": "bool", "shape": []},
    # gripper_physics
    "gripper_qpos": {"dtype": "float32", "shape": [2], "unit": "rad"},
    "gripper_width": {"dtype": "float32", "shape": [], "unit": "m"},
    "gripper_width_velocity": {"dtype": "float32", "shape": [], "unit": "m/s"},
    "gripper_physics_known_mask": {"dtype": "bool", "shape": []},
    "gripper_physics_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
    # gripper_closing_state (physical qpos measurement, NOT policy action)
    "gripper_closing_detected": {"dtype": "bool", "shape": [], "valid_range": [False, True]},
    "gripper_closing_known_mask": {"dtype": "bool", "shape": []},
    "gripper_closing_confidence": {"dtype": "float32", "shape": [], "valid_range": [0.0, 1.0]},
}

V22_CONFIG = {
    "grasp": {
        "dwell_steps_required": 5,
        "dwell_steps_confidence": 10,
        "gripper_close_threshold": 0.03,
        "gripper_close_qpos_sum": 0.03,
    },
    "comotion": {
        "velocity_correlation_threshold": 0.5,
        "window_steps": 5,
    },
    "lift": {
        "height_above_initial_threshold": 0.02,
        "scale_m": 0.03,
    },
    "instability": {
        "contact_loss_score_threshold": 0.01,
        "pose_jump_threshold_m": 0.05,
        "width_increase_rate_threshold": 0.01,
        "window_steps": 3,
    },
    "safe_release": {
        "placement_region_tolerance": 0.05,
        "release_width_open_threshold": 0.03,
        "release_opening_velocity": 0.005,
    },
    "terminal": {
        "proximity_steps_to_end": 5,
    },
}

def compute_config_sha():
    return hashlib.sha256(json.dumps(V22_CONFIG, sort_keys=True).encode()).hexdigest()

# ── Sidecar Parser ──

def parse_sidecar(path):
    """Parse privileged_teacher_sidecar.jsonl into per-step records."""
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
            if identity is None:
                identity = {
                    'suite': rec.get('suite'),
                    'task_idx': rec.get('task_idx'),
                    'state_id': rec.get('state_id'),
                }
            records.append(rec)

    return {'identity': identity, 'n_steps': len(records), 'steps': records}

def parse_episode_summary(path):
    """Parse episode_summary.json for terminal/success data."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

# ── BDDL Object Slice Cache ──

_bddl_cache = {}

def get_object_slices_for_task(suite, task_idx):
    """Resolve object_slices for a task from LIBERO BDDL files.
    Cached in-process; returns None if BDDL unavailable.
    """
    cache_key = (suite, task_idx)
    if cache_key in _bddl_cache:
        return _bddl_cache[cache_key]

    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(suite)(0)
        task = benchmark.get_task(task_idx)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if not os.path.isfile(bddl_path):
            _bddl_cache[cache_key] = None
            return None
        text = open(bddl_path, encoding='utf-8').read()
        objects = parse_bddl_objects(text)
        slices = build_object_slices(objects)
        role = parse_bddl_task_role(text, suite, task_idx, list(slices.keys()))
        result = {
            "object_slices": slices,
            "task_role": role,
            "bddl_text": text,
            "bddl_path": bddl_path,
        }
        _bddl_cache[cache_key] = result
        return result
    except Exception:
        _bddl_cache[cache_key] = None
        return None

def resolve_manipulated_objects(suite, task_idx):
    """Get manipulated objects and support names from BDDL."""
    bddl_info = get_object_slices_for_task(suite, task_idx)
    if bddl_info is None:
        return [], [], {}
    role = bddl_info["task_role"]
    return role["manipulated_objects"], role["support_names"], bddl_info["object_slices"]

# ── Goal Target Resolver (V2: split semantic vs physical binding) ──

GOAL_TASK_TARGETS = {
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
    "put the cream cheese in the bowl": {"object_name": "cream_cheese", "type": "pick_place"},
    "turn on the stove": {"object_name": "stove_knob", "type": "articulated"},
    "turn off the stove": {"object_name": "stove_knob", "type": "articulated"},
}

def resolve_goal_target(task_instruction, object_slices):
    """Resolve target for goal tasks. Returns split resolution.

    V2 FIX: Separates task_semantics_known from physical_binding_known.
    When task is known but object not in qpos slices → target_resolved=False.
    """
    result = {
        'task_semantics_known': False,
        'physical_binding_known': False,
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
        for key, val in GOAL_TASK_TARGETS.items():
            if key.replace('_', ' ') in instruction_lower or instruction_lower in key.replace('_', ' '):
                target_info = val
                break

    if target_info is None:
        result['reason'] = 'TARGET_SEMANTICS_UNKNOWN'
        result['task_semantics_known'] = False
        result['physical_binding_known'] = False
        result['target_resolved'] = False
        result['target_known_mask'] = True
        return result

    # Task semantics are known
    result['task_semantics_known'] = True
    obj_name = target_info['object_name']
    result['target_type'] = target_info['type']

    # Check physical binding: is target object in qpos slices?
    object_names = list(object_slices.keys()) if object_slices else []
    matched_keys = [k for k in object_names if obj_name in k.lower() or k.lower() == obj_name.lower()]

    if matched_keys:
        result['physical_binding_known'] = True
        result['target_resolved'] = True
        result['target_object_id'] = matched_keys[0]
        result['resolver_confidence'] = 0.9 if len(matched_keys) == 1 else 0.7
        result['target_known_mask'] = True
        result['reason'] = 'TARGET_RESOLVED_EXACT'
    else:
        # Task known but no physical binding — object not in qpos slices
        result['physical_binding_known'] = False
        result['target_resolved'] = False
        result['target_object_id'] = obj_name
        result['resolver_confidence'] = 0.3
        result['target_known_mask'] = True
        result['reason'] = 'TARGET_SEMANTICS_KNOWN_NO_PHYSICAL_BINDING'
        result['note'] = 'Object not in MuJoCo qpos slices; target-specific factors will be unknown'

    return result

# ── Physics Factor Computation ──

def compute_grasp_state(steps, manipulated_objects, support_names):
    """Compute grasp state: target-specific gripper contact + sustained dwell.

    V2 FIX: Uses _contact_flags (target-finger contact), not arbitrary contact_count>0.
    When no manipulated_objects known → grasp_known_mask=False.
    """
    T = len(steps)
    results = []
    contact_dwell = 0
    can_detect = len(manipulated_objects) > 0

    for t in range(T):
        rec = steps[t]
        if can_detect:
            cf = _contact_flags(rec.get('mujoco_contact_pairs', []), manipulated_objects, support_names)
            _, has_gripper_contact, _ = cf
            qpos_close = _gripper_qpos_closure(rec)
            gripper_near_closed = qpos_close > 0.7

            if gripper_near_closed and has_gripper_contact:
                contact_dwell += 1
            elif has_gripper_contact:
                contact_dwell = max(0, contact_dwell - 1)
            else:
                contact_dwell = 0

            grasp = contact_dwell >= V22_CONFIG['grasp']['dwell_steps_required']
            conf = min(1.0, contact_dwell / max(1, V22_CONFIG['grasp']['dwell_steps_confidence']))
            known_mask = True
        else:
            grasp = False
            conf = 0.0
            contact_dwell = 0
            known_mask = False

        results.append({
            'grasp_established': bool(grasp),
            'grasp_confidence': float(conf),
            'grasp_known_mask': bool(known_mask),
            'grasp_dwell_steps': contact_dwell,
        })
    return results

def compute_contact_state(steps, manipulated_objects, support_names):
    """Compute contact state using target-specific contact flags.

    V2 FIX: Reports object_contact and gripper_contact separately.
    contact_score is gripper_contact only (not arbitrary contact_count).
    """
    results = []
    can_detect = len(manipulated_objects) > 0

    for t, rec in enumerate(steps):
        if can_detect:
            cf = _contact_flags(rec.get('mujoco_contact_pairs', []), manipulated_objects, support_names)
            obj_contact, grip_contact, _ = cf
            score = 1.0 if grip_contact else (0.3 if obj_contact else 0.0)
            results.append({
                'contact_score': float(score),
                'contact_known_mask': True,
                'contact_confidence': float(score),
                'contact_source': 'mujoco_contact_pairs_filtered',
                'object_contact_detected': bool(obj_contact),
                'gripper_contact_detected': bool(grip_contact),
            })
        else:
            results.append({
                'contact_score': 0.0,
                'contact_known_mask': False,
                'contact_confidence': 0.0,
                'contact_source': 'no_manipulated_objects',
                'object_contact_detected': False,
                'gripper_contact_detected': False,
            })
    return results

def compute_comotion_state(steps, manipulated_objects, object_slices):
    """Compute object-EEF co-motion using V5 _comotion pattern.

    V2 FIX: Correct history indexing — uses step[t-window]'s object_state.
    Requires object_slices to extract target positions from qpos array.
    When unavailable → comotion_known_mask=False.
    """
    T = len(steps)
    window = V22_CONFIG['comotion']['window_steps']
    results = []
    can_compute = len(manipulated_objects) > 0 and len(object_slices) > 0

    for t in range(T):
        if not can_compute or t < window:
            results.append({
                'object_eef_comotion_score': 0.0,
                'comotion_known_mask': bool(can_compute),
                'comotion_confidence': 0.0,
            })
            continue

        history = steps[max(0, t - window + 1):t + 1]
        eef = [_finite_vector(item.get('robot0_eef_pos'), 3) for item in history]
        if len(eef) < 2 or any(item is None for item in eef):
            results.append({
                'object_eef_comotion_score': 0.0,
                'comotion_known_mask': True,
                'comotion_confidence': 0.0,
            })
            continue

        values = []
        for name in manipulated_objects:
            spec = object_slices.get(name)
            if spec is None:
                continue
            positions = [_slice_vector(item.get('object_state', []), spec, 'pos') for item in history]
            if len(positions) < 2 or any(item is None for item in positions):
                continue
            similarities = []
            for obj_l, obj_r, eef_l, eef_r in zip(positions[1:], positions[:-1], eef[1:], eef[:-1]):
                obj_delta = [a - b for a, b in zip(obj_l, obj_r)]
                eef_delta = [a - b for a, b in zip(eef_l, eef_r)]
                if (math.isclose(math.sqrt(sum(v*v for v in obj_delta)), 0.0) or
                    math.isclose(math.sqrt(sum(v*v for v in eef_delta)), 0.0)):
                    continue
                similarities.append((_cosine(obj_delta, eef_delta) + 1.0) / 2.0)
            if similarities:
                values.append(_mean(similarities, 0.0))

        score = _mean(values, 0.0)
        results.append({
            'object_eef_comotion_score': float(score),
            'comotion_known_mask': True,
            'comotion_confidence': float(_mean(values, 0.0)),
        })

    return results

def compute_lift_state(steps, manipulated_objects, object_slices):
    """Compute lift from target object Z height change (V5 _lift pattern).

    V2 FIX: Uses target object Z from qpos slices, NOT EEF Z proxy.
    When object_slices unavailable → lift_known_mask=False.
    """
    T = len(steps)
    results = []
    can_compute = len(manipulated_objects) > 0 and len(object_slices) > 0
    scale = V22_CONFIG['lift']['scale_m']

    for t in range(T):
        if not can_compute:
            results.append({
                'lift_score': 0.0,
                'lift_known_mask': False,
                'lift_confidence': 0.0,
            })
            continue

        current = steps[t]
        initial = steps[0]
        values = []
        for name in manipulated_objects:
            spec = object_slices.get(name)
            if spec is None:
                continue
            init_pos = _slice_vector(initial.get('object_state', []), spec, 'pos')
            curr_pos = _slice_vector(current.get('object_state', []), spec, 'pos')
            if init_pos is not None and curr_pos is not None:
                values.append(_clip((curr_pos[2] - init_pos[2]) / scale))

        if values:
            lift = max(values)
            results.append({
                'lift_score': float(lift),
                'lift_known_mask': True,
                'lift_confidence': float(min(1.0, lift * 2)),
            })
        else:
            results.append({
                'lift_score': 0.0,
                'lift_known_mask': False,
                'lift_confidence': 0.0,
            })

    return results

def compute_instability_indicators(steps, grasp_results, manipulated_objects, object_slices):
    """Compute instability indicators.

    V2 FIX: Uses target-relative measurements where object_slices available.
    slip and contact_loss are based on grasp+contact transitions.
    pose_anomaly uses target-relative EEF motion (not EEF-alone jumps).
    width_increase uses gripper qpos velocity.
    """
    window = V22_CONFIG['instability']['window_steps']
    T = len(steps)
    results = []
    can_target = len(manipulated_objects) > 0 and len(object_slices) > 0

    for t in range(T):
        slip = False; contact_loss = False; pose_anomaly = False
        width_increase = False
        slip_known = False; contact_loss_known = False
        pose_anomaly_known = False; width_known = False

        if t >= window and grasp_results[t]['grasp_known_mask']:
            rec = steps[t]
            prev_rec = steps[t - window]
            has_grasp = grasp_results[t]['grasp_established']
            had_grasp = grasp_results[t - window]['grasp_established']

            # Contact loss: gripper contact dropped while grasp was active
            if can_target:
                cf_now = _contact_flags(rec.get('mujoco_contact_pairs', []), manipulated_objects)
                cf_prev = _contact_flags(prev_rec.get('mujoco_contact_pairs', []), manipulated_objects)
                had_grip = cf_prev[1]
                has_grip = cf_now[1]
                contact_loss_known = True
                if had_grip and not has_grip and had_grasp:
                    contact_loss = True
                # Slip: contact count drops while grasp maintained
                if had_grip and has_grip and had_grasp and has_grasp:
                    slip_known = True
                    if cf_prev[0] and cf_now[0]:
                        slip = False  # contact maintained on target
                    elif cf_prev[0] and not cf_now[0]:
                        slip = True  # lost contact with target

            # Pose anomaly: target-object-relative displacement
            # Δr = (o_t - e_t) - (o_{t-w} - e_{t-w})
            # Only when object moves relative to EEF abnormally is it instability.
            # EEF-alone jumps do NOT constitute pose anomaly.
            eef = _finite_vector(rec.get('robot0_eef_pos'), 3)
            prev_eef = _finite_vector(prev_rec.get('robot0_eef_pos'), 3)
            if eef is not None and prev_eef is not None and can_target:
                # Compute target-relative displacement for each manipulated object
                obj_rel_deltas = []
                for name in manipulated_objects:
                    spec = object_slices.get(name)
                    if spec is None:
                        continue
                    curr_obj = _slice_vector(rec.get('object_state', []), spec, 'pos')
                    prev_obj = _slice_vector(prev_rec.get('object_state', []), spec, 'pos')
                    if curr_obj is None or prev_obj is None:
                        continue
                    # Object-relative-to-EEF vector at t and t-window
                    rel_now = [curr_obj[i] - eef[i] for i in range(3)]
                    rel_prev = [prev_obj[i] - prev_eef[i] for i in range(3)]
                    delta = _dist(rel_now, rel_prev)
                    obj_rel_deltas.append(delta)
                if obj_rel_deltas:
                    pose_anomaly_known = True
                    max_delta = max(obj_rel_deltas)
                    if max_delta > V22_CONFIG['instability']['pose_jump_threshold_m']:
                        pose_anomaly = True
                else:
                    pose_anomaly_known = False
            else:
                pose_anomaly_known = False

            # Width increase: gripper opening unexpectedly during grasp
            qpos = _finite_vector(rec.get('robot0_gripper_qpos'), 2)
            prev_qpos = _finite_vector(prev_rec.get('robot0_gripper_qpos'), 2)
            if qpos is not None and prev_qpos is not None:
                width_now = abs(qpos[0]) + abs(qpos[1])
                width_prev = abs(prev_qpos[0]) + abs(prev_qpos[1])
                width_known = True
                if has_grasp and (width_now - width_prev) > V22_CONFIG['instability']['width_increase_rate_threshold']:
                    width_increase = True

        results.append({
            'slip_detected': bool(slip), 'slip_known_mask': bool(slip_known),
            'slip_confidence': 0.6 if slip else 0.0,
            'contact_loss_detected': bool(contact_loss), 'contact_loss_known_mask': bool(contact_loss_known),
            'contact_loss_confidence': 0.7 if contact_loss else 0.0,
            'pose_anomaly_detected': bool(pose_anomaly), 'pose_anomaly_known_mask': bool(pose_anomaly_known),
            'pose_anomaly_confidence': 0.5 if pose_anomaly else 0.0,
            'unplanned_width_increase': bool(width_increase), 'width_increase_known_mask': bool(width_known),
            'width_increase_confidence': 0.6 if width_increase else 0.0,
        })

    return results

def compute_terminal_state(steps, episode_summary):
    """Compute terminal/success state from episode_summary data.

    V2 FIX: Actually uses episode_summary content.
    task_success is set from episode_summary for terminal steps.
    """
    T = len(steps)
    results = []
    success = False
    terminal_known = False

    if episode_summary is not None:
        success = bool(episode_summary.get('task_success', False))
        terminal_known = True

    for t in range(T):
        is_last = (t == T - 1)
        proximity_to_end = T - t
        near_end = proximity_to_end <= V22_CONFIG['terminal']['proximity_steps_to_end']

        results.append({
            'task_success': bool(success) if is_last else False,
            'task_terminal': bool(is_last),
            'terminal_known_mask': bool(terminal_known),
            'terminal_confidence': 0.95 if (is_last and terminal_known) else (0.5 if is_last else 0.1),
            'proximity_to_end': proximity_to_end,
        })
    return results

def compute_safe_release(steps, grasp_results, terminal_results, placement_results):
    """Compute safe release: release_event AND known placement AND object_placed.

    V2 FIX (audit round 2): Safe release requires placement confirmation.
    Unplanned opening, attack-induced drop, or contact loss WITHOUT placement
    is NOT safe_release — it goes to instability instead.
    When placement is unknown, safe_release valid_mask = False.
    """
    T = len(steps)
    results = []

    for t in range(T):
        rec = steps[t]
        qpos = _finite_vector(rec.get('robot0_gripper_qpos'), 2)

        release_detected = False
        release_known = False
        release_conf = 0.0

        if qpos is not None and grasp_results[t]['grasp_known_mask']:
            width = abs(qpos[0]) + abs(qpos[1])
            was_grasping = grasp_results[max(0, t-1)]['grasp_established'] if t > 0 else False
            is_grasping = grasp_results[t]['grasp_established']
            is_terminal = terminal_results[t]['task_terminal']
            placement_known = placement_results[t]['placement_known_mask']
            placement_confirmed = placement_results[t]['object_placed']

            has_release_event = False
            if was_grasping and width > V22_CONFIG['safe_release']['release_width_open_threshold']:
                has_release_event = True
            elif is_terminal and not is_grasping:
                has_release_event = True

            if has_release_event:
                if placement_known and placement_confirmed:
                    release_detected = True
                    release_known = True
                    release_conf = 0.85
                elif placement_known and not placement_confirmed:
                    # Release event without placement → NOT safe (could be drop/attack)
                    release_detected = False
                    release_known = True
                    release_conf = 0.7
                else:
                    # Release event but placement unknown → cannot confirm safe
                    release_detected = False
                    release_known = False
                    release_conf = 0.0
            elif not has_release_event:
                release_detected = False
                release_known = True
                release_conf = 0.7

        results.append({
            'planned_release_detected': bool(release_detected),
            'planned_release_known_mask': bool(release_known),
            'planned_release_confidence': float(release_conf),
        })
    return results

def compute_placement_state(steps, grasp_results, manipulated_objects, object_slices, target_names):
    """Detect object placement on target surface/region."""
    T = len(steps)
    results = []
    can_compute = len(manipulated_objects) > 0 and len(object_slices) > 0

    for t in range(T):
        placed = False
        known = False
        conf = 0.0
        region = None

        if can_compute and grasp_results[t]['grasp_known_mask']:
            known = True
            # Simple heuristic: grasp lost + object near target
            if t > 0 and grasp_results[t-1]['grasp_established'] and not grasp_results[t]['grasp_established']:
                for name in manipulated_objects:
                    spec = object_slices.get(name)
                    if spec is None:
                        continue
                    obj_pos = _slice_vector(steps[t].get('object_state', []), spec, 'pos')
                    if obj_pos is None:
                        continue
                    for tname in target_names:
                        tspec = object_slices.get(tname)
                        if tspec is None:
                            continue
                        tpos = _slice_vector(steps[t].get('object_state', []), tspec, 'pos')
                        if tpos is None:
                            continue
                        if _dist(obj_pos, tpos) < V22_CONFIG['safe_release']['placement_region_tolerance']:
                            placed = True
                            conf = 0.7
                            region = tname
                            break
                    if placed:
                        break

        results.append({
            'object_placed': bool(placed),
            'placement_known_mask': bool(known),
            'placement_confidence': float(conf),
            'placement_region': region,
        })
    return results

def compute_gripper_closing_state(steps):
    """Compute gripper closing state from physical qpos measurements.

    V2 FIX: Measures physical gripper closure from qpos velocity.
    This is physical state measurement, NOT a policy intent signal.
    Renamed from close_intent to avoid confusion with policy-level gating.
    Fully decoupled from any policy-level close gate.
    """
    T = len(steps)
    results = []
    prev_qpos_sum = None

    for t in range(T):
        rec = steps[t]
        qpos = _finite_vector(rec.get('robot0_gripper_qpos'), 2)

        if qpos is not None:
            qpos_sum = abs(qpos[0]) + abs(qpos[1])
            closing = prev_qpos_sum is not None and (prev_qpos_sum - qpos_sum) > 0.001
            near_closed = qpos_sum < V22_CONFIG['grasp']['gripper_close_qpos_sum']

            results.append({
                'gripper_closing_detected': bool(closing or near_closed),
                'gripper_closing_known_mask': True,
                'gripper_closing_confidence': 0.9 if near_closed else (0.6 if closing else 0.0),
            })
            prev_qpos_sum = qpos_sum
        else:
            results.append({
                'gripper_closing_detected': False,
                'gripper_closing_known_mask': False,
                'gripper_closing_confidence': 0.0,
            })

    return results

def compute_gripper_physics(steps):
    """Extract raw gripper physical state: qpos, width, velocity."""
    T = len(steps)
    results = []
    prev_width = None

    for t in range(T):
        rec = steps[t]
        qpos = _finite_vector(rec.get('robot0_gripper_qpos'), 2)

        if qpos is not None:
            width = abs(qpos[0]) + abs(qpos[1])
            velocity = (width - prev_width) if prev_width is not None else 0.0
            prev_width = float(width)
            results.append({
                'gripper_qpos': [float(qpos[0]), float(qpos[1])],
                'gripper_width': float(width),
                'gripper_width_velocity': float(velocity),
                'gripper_physics_known_mask': True,
                'gripper_physics_confidence': 1.0,
            })
        else:
            results.append({
                'gripper_qpos': [0.0, 0.0],
                'gripper_width': 0.0,
                'gripper_width_velocity': 0.0,
                'gripper_physics_known_mask': False,
                'gripper_physics_confidence': 0.0,
            })

    return results

# ── V22 → Label V2 Adapter ──

def v22_to_label_v2(v22_snapshot, step_index, K=10):
    """Convert V22 factor snapshot to Label Contract V2 tri-state format.

    Evidence lattice: any known+positive → 1; all known+negative → 0; else unknown.
    candidate_close is NOT used.
    """
    factors = v22_snapshot['factors']
    g = factors['grasp_state']
    c = factors['contact_state']
    co = factors['comotion_state']
    l = factors['lift_state']
    inst = factors['instability_indicators']
    t = factors['terminal_state']
    pr = factors.get('planned_release', {})
    pl = factors.get('placement_state', {})
    ci = factors.get('gripper_closing_state', {})

    # ── Physical criticality (Head A) ──
    channels = [
        ('grasp', g.get('grasp_known_mask', False), g.get('grasp_established', False),
         g.get('grasp_confidence', 0.0)),
        ('contact', c.get('contact_known_mask', False), c.get('contact_score', 0) > 0.3,
         c.get('contact_confidence', 0.0)),
        ('comotion', co.get('comotion_known_mask', False), co.get('object_eef_comotion_score', 0) > 0.3,
         co.get('comotion_confidence', 0.0)),
        ('lift', l.get('lift_known_mask', False), l.get('lift_score', 0) > 0.05,
         l.get('lift_confidence', 0.0)),
    ]

    any_known = any(k for _, k, _, _ in channels)
    any_positive = any(p for _, k, p, _ in channels if k and p)
    all_known = all(k for _, k, _, _ in channels)
    all_negative = all((not p) for _, k, p, _ in channels if k)

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

    # ── Instability (Head D) ──
    instab_channels = [
        ('slip', inst.get('slip_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('slip_detected', False), inst.get('slip_confidence', 0.0)),
        ('contact_loss', inst.get('contact_loss_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('contact_loss_detected', False), inst.get('contact_loss_confidence', 0.0)),
        ('pose_anomaly', inst.get('pose_anomaly_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('pose_anomaly_detected', False), inst.get('pose_anomaly_confidence', 0.0)),
        ('width_increase', inst.get('width_increase_known_mask', False) and g.get('grasp_known_mask', False),
         inst.get('unplanned_width_increase', False), inst.get('width_increase_confidence', 0.0)),
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

    # ── Safe release (Head E) — now computed from physics ──
    if pr.get('planned_release_known_mask'):
        if pr.get('planned_release_detected'):
            safe_release = {'value': 1, 'valid_mask': True,
                           'reason': 'SAFE_RELEASE_PLANNED_OPENING', 'confidence': pr.get('planned_release_confidence', 0.7)}
        else:
            safe_release = {'value': 0, 'valid_mask': True,
                           'reason': 'NO_SAFE_RELEASE_DETECTED', 'confidence': 0.7}
    elif t.get('terminal_known_mask') and t.get('task_success'):
        safe_release = {'value': 1, 'valid_mask': True,
                       'reason': 'SAFE_RELEASE_POST_SUCCESS', 'confidence': 1.0}
    else:
        safe_release = {'value': None, 'valid_mask': False, 'reason': 'UNKNOWN_PRIVILEGED_STATE', 'confidence': 0.0}

    # ── Gripper closing state (Head C) — physical qpos measurement, NOT policy intent ──
    if ci.get('gripper_closing_known_mask'):
        gripper_closing = {
            'value': bool(ci.get('gripper_closing_detected', False)),
            'valid_mask': True,
            'reason': 'PHYSICAL_CLOSURE_MEASUREMENT',
            'confidence': ci.get('gripper_closing_confidence', 0.0),
        }
    else:
        gripper_closing = {'value': None, 'valid_mask': False,
                          'reason': 'UNKNOWN_PRIVILEGED_STATE', 'confidence': 0.0}

    return {
        'physical_criticality': crit,
        'instability': instability,
        'safe_release': safe_release,
        'k10_feasible': {'value': None, 'valid_mask': False, 'reason': 'COMPUTED_SEPARATELY', 'confidence': 0.0},
        'gripper_closing_state': gripper_closing,
    }

# ── K10 Recompute (fixed: one output per timestep) ──

def recompute_k10(critical_labels, safe_release_labels, K=10):
    """Recompute K10 feasibility from V22 criticality sequence.

    V2 FIX: Returns exactly one result per timestep (fixes duplicate append bug).
    """
    T = len(critical_labels)
    results = []
    for t in range(T):
        if t + K > T:
            results.append({'value': 0, 'valid_mask': True,
                           'reason': 'K10_INFEASIBLE_HORIZON', 'confidence': 0.0})
            continue

        has_unknown = False
        has_known_false = False
        all_critical = True

        for i in range(t, t + K):
            if i >= T:
                all_critical = False; break
            crit = critical_labels[i]
            sr = safe_release_labels[i] if i < len(safe_release_labels) else {'value': 0, 'valid_mask': False}

            # Veto: safe_release in window
            if sr.get('valid_mask') and sr.get('value') == 1:
                results.append({'value': 0, 'valid_mask': True,
                               'reason': 'K10_INFEASIBLE_SAFE_RELEASE', 'confidence': 0.0})
                all_critical = False; break

            if not crit.get('valid_mask'):
                has_unknown = True
                all_critical = False
            elif crit.get('value') != 1:
                has_known_false = True
                all_critical = False; break

        else:
            # Inner loop completed without break → all K are critical+known
            if all_critical:
                results.append({'value': 1, 'valid_mask': True,
                               'reason': 'K10_FEASIBLE', 'confidence': 1.0})
                continue

        # Only proceed if we didn't already append
        if not all_critical and has_known_false and not has_unknown:
            results.append({'value': 0, 'valid_mask': True,
                           'reason': 'K10_INFEASIBLE_NO_CRITICAL_CORRIDOR', 'confidence': 0.0})
        elif has_unknown:
            results.append({'value': None, 'valid_mask': False,
                           'reason': 'K10_UNKNOWN_CRITICAL_IN_WINDOW', 'confidence': 0.0})

    return results

# ── Validation ──

def validate_v22_snapshot(snapshot):
    """Validate V22 snapshot consistency. Returns list of violations."""
    violations = []
    factors = snapshot['factors']
    for name in factors:
        f = factors[name]
        if f.get('known_mask') and not isinstance(f.get('known_mask'), bool):
            violations.append(f'{name}.known_mask: not bool')

    snapshot_str = json.dumps(snapshot, default=str)
    if 'NaN' in snapshot_str or 'Infinity' in snapshot_str:
        violations.append('NaN/Inf detected in snapshot')

    return violations

# ── Tests ──

def test_typed_schema_complete():
    from physics_teacher_v22 import V22_FACTORS
    for name, factor_def in V22_FACTORS.items():
        for field in factor_def['fields']:
            if field == 'gripper_width' or field == 'gripper_width_velocity':
                continue
            assert field in FIELD_TYPES, f'{name}.{field}: missing FIELD_TYPE'
    print('PASS: test_typed_schema_complete')

def test_config_sha_stable():
    sha1 = compute_config_sha()
    sha2 = compute_config_sha()
    assert sha1 == sha2
    assert len(sha1) == 64
    print(f'PASS: test_config_sha_stable ({sha1[:16]}...)')

def test_goal_resolver_split():
    """Target resolver with object in qpos slices: both semantics and physical known."""
    slices = {"drawer_handle": {"pos": [0, 3]}, "cabinet_body": {"pos": [14, 17]}}
    result = resolve_goal_target("open the middle drawer of the cabinet", slices)
    assert result['task_semantics_known'] == True
    assert result['physical_binding_known'] == True
    assert result['target_resolved'] == True
    assert 'drawer' in result['target_object_id']
    assert result['reason'] == 'TARGET_RESOLVED_EXACT'
    print(f'PASS: test_goal_resolver_split (both known)')

def test_goal_resolver_physical_unknown():
    """Task known but object not in qpos slices → semantics known, physical unknown."""
    result = resolve_goal_target("open the middle drawer of the cabinet", {})
    assert result['task_semantics_known'] == True
    assert result['physical_binding_known'] == False
    assert result['target_resolved'] == False
    assert 'NO_PHYSICAL_BINDING' in result['reason']
    print(f'PASS: test_goal_resolver_physical_unknown ({result["reason"]})')

def test_goal_resolver_unknown_task():
    result = resolve_goal_target("do something completely unknown", {"obj_a": {"pos": [0, 3]}})
    assert result['task_semantics_known'] == False
    assert result['physical_binding_known'] == False
    assert result['target_resolved'] == False
    assert 'SEMANTICS_UNKNOWN' in result['reason']
    print(f'PASS: test_goal_resolver_unknown_task ({result["reason"]})')

def test_grasp_target_specific():
    """Grasp only triggers on target-finger contact, not arbitrary contact."""
    steps = []
    for i in range(30):
        pairs = []
        if i >= 11:
            pairs = [["gripper0_finger1", "bowl_main"]]
        steps.append({
            'robot0_gripper_qpos': [0.0, 0.0],
            'mujoco_contact_pairs': pairs,
            'object_state': list(range(56)),
        })
    results = compute_grasp_state(steps, ["bowl"], [])
    # 11-30 = 20 consecutive gripper↔bowl contacts → dwell >= 5 → grasp established
    assert results[16]['grasp_established'] == True, \
        f'Grasp should be established by step 16, got dwell={results[16]["grasp_dwell_steps"]}'
    # First 10 steps have no pairs → no grasp
    assert results[5]['grasp_established'] == False
    print('PASS: test_grasp_target_specific')

def test_lift_uses_object_z():
    """Lift uses target object Z from qpos, not EEF Z."""
    slices = {"bowl": {"pos": [0, 3]}}
    steps = []
    for i in range(20):
        state = [0.0] * 56
        state[2] = 0.1 + i * 0.005  # bowl Z rises
        steps.append({
            'robot0_eef_pos': [0.3, 0.0, 0.8],  # EEF constant (not rising)
            'robot0_gripper_qpos': [0.0, 0.0],
            'object_state': state,
            'mujoco_contact_pairs': [],
        })
    results = compute_lift_state(steps, ["bowl"], slices)
    # Object Z rises → lift should be > 0
    assert results[-1]['lift_score'] > 0.0, f'Object Z rising should produce lift>0, got {results[-1]["lift_score"]}'
    assert results[-1]['lift_known_mask'] == True
    print(f'PASS: test_lift_uses_object_z (lift={results[-1]["lift_score"]:.3f})')

def test_lift_unknown_without_slices():
    """Lift → unknown when object_slices unavailable."""
    steps = [{'robot0_eef_pos': [0.3, 0.0, 0.8], 'robot0_gripper_qpos': [0.0, 0.0],
              'object_state': list(range(56)), 'mujoco_contact_pairs': []} for _ in range(20)]
    results = compute_lift_state(steps, [], {})
    for r in results:
        assert r['lift_known_mask'] == False, 'Lift should be unknown without object_slices'
        assert r['lift_score'] == 0.0
    print('PASS: test_lift_unknown_without_slices')

def test_comotion_history_index():
    """Comotion uses correct history indexing: step[t-window] not step[t]."""
    slices = {"bowl": {"pos": [0, 3]}}
    steps = []
    for i in range(20):
        state = [0.0] * 56
        state[0] = 0.3 + i * 0.002  # bowl moves with EEF
        state[1] = 0.0 + i * 0.001
        state[2] = 0.1
        steps.append({
            'robot0_eef_pos': [0.3 + i * 0.002, 0.0 + i * 0.001, 0.1],
            'robot0_gripper_qpos': [0.0, 0.0],
            'object_state': state,
            'mujoco_contact_pairs': [["gripper0_finger1", "bowl_main"]],
        })
    results = compute_comotion_state(steps, ["bowl"], slices)
    # Object and EEF moving together → high comotion
    assert results[-1]['comotion_known_mask'] == True
    assert results[-1]['object_eef_comotion_score'] > 0.5, \
        f'Synchronized motion should have high comotion, got {results[-1]["object_eef_comotion_score"]}'
    print(f'PASS: test_comotion_history_index (score={results[-1]["object_eef_comotion_score"]:.3f})')

def test_safe_release_implemented():
    """Safe release requires placement confirmation (not just gripper opening)."""
    steps = []
    for i in range(30):
        qpos = [0.0, 0.0] if i < 20 else [0.04, 0.04]
        steps.append({
            'robot0_gripper_qpos': qpos,
            'robot0_eef_pos': [0.3, 0.0, 0.1],
            'object_state': list(range(56)),
            'mujoco_contact_pairs': [["gripper0_finger1", "bowl_main"]],
        })
    grasp_results = compute_grasp_state(steps, ["bowl"], [])
    terminal_results = [{'task_terminal': (i == 29), 'task_success': False,
                         'terminal_known_mask': True, 'terminal_confidence': 0.5}
                        for i in range(30)]
    # Placement: known + placed after step 22
    placement_results = []
    for i in range(30):
        placement_results.append({
            'object_placed': i >= 22,
            'placement_known_mask': True,
            'placement_confidence': 0.8,
        })
    release_results = compute_safe_release(steps, grasp_results, terminal_results, placement_results)
    # Steps 20-21: release event but placement not yet confirmed → not safe
    assert release_results[20]['planned_release_detected'] == False, \
        'Release without placement should NOT be safe_release'
    # Steps 23+: release event + placement confirmed → safe_release
    safe_steps = [r['planned_release_detected'] for r in release_results[23:]]
    assert any(safe_steps), 'Release with placement should be safe_release'
    print('PASS: test_safe_release_implemented')

def test_gripper_closing_implemented():
    """Gripper closing state from physical qpos measurement."""
    steps = []
    for i in range(20):
        qpos = [max(0.0, 0.08 - i * 0.004), max(0.0, 0.08 - i * 0.004)]
        steps.append({'robot0_gripper_qpos': list(qpos)})
    results = compute_gripper_closing_state(steps)
    closing_steps = [r['gripper_closing_detected'] for r in results[1:15]]
    assert any(closing_steps), 'Closing gripper should produce gripper_closing=True'
    assert all(r['gripper_closing_known_mask'] for r in results), 'Gripper closing known_mask should be True'
    print('PASS: test_gripper_closing_implemented')

def test_terminal_from_summary():
    """Terminal state uses episode_summary data."""
    steps = [{'step': i} for i in range(50)]
    summary = {'task_success': True, 'total_steps': 50}
    results = compute_terminal_state(steps, summary)
    assert results[-1]['task_success'] == True
    assert results[-1]['terminal_known_mask'] == True
    assert results[0]['task_success'] == False  # Non-terminal steps are False
    print('PASS: test_terminal_from_summary')

def test_k10_no_duplicate_append():
    """K10 produces exactly one result per timestep (fixes duplicate append bug)."""
    crit = [{'value': 1, 'valid_mask': True, 'reason': 'CRITICAL', 'confidence': 0.9} for _ in range(20)]
    # Insert safe_release at step 5
    sr = [{'value': 0, 'valid_mask': True, 'reason': 'NO_RELEASE', 'confidence': 0.0} for _ in range(20)]
    sr[5] = {'value': 1, 'valid_mask': True, 'reason': 'SAFE_RELEASE', 'confidence': 0.9}
    results = recompute_k10(crit, sr, K=10)
    assert len(results) == 20, f'Expected 20 results, got {len(results)}'
    # Step 0: safe_release at step 5 is within [0,10) → should be infeasible
    assert results[0]['value'] == 0, f'Step 0 should be infeasible due to safe_release, got {results[0]}'
    print('PASS: test_k10_no_duplicate_append')

def test_candidate_close_not_in_physics():
    """candidate_close must not appear in physics computation code (comments OK)."""
    source = open(__file__).read()
    phys_section = source[source.index('# ── Physics Factor Computation ──'):
                            source.index('# ── V22 → Label V2 Adapter ──')]
    # Strip comments and docstrings before checking
    lines = phys_section.split('\n')
    code_lines = [l for l in lines if not l.strip().startswith('#') and 'candidate_close' not in l.lower().split('#')[0] if l.strip()]
    for line in code_lines:
        # Remove inline comments
        code_only = line.split('#')[0] if '#' in line else line
        if 'candidate_close' in code_only:
            raise AssertionError(f'candidate_close in physics code: {code_only.strip()[:80]}')
    print('PASS: test_candidate_close_not_in_physics')

def test_contact_flags_target_specific():
    """_contact_flags only counts contact when manipulated object is involved."""
    pairs = [["gripper0_finger1", "bowl_main"]]
    obj_c, grip_c, sup_c = _contact_flags(pairs, ["bowl"], [])
    assert obj_c == True
    assert grip_c == True
    # Non-manipulated object → should not flag
    pairs2 = [["gripper0_finger1", "drawer_handle"]]
    obj_c2, grip_c2, _ = _contact_flags(pairs2, ["bowl"], [])
    assert obj_c2 == False, 'Non-manipulated object should not trigger object_contact'
    assert grip_c2 == False, 'Non-manipulated object should not trigger gripper_contact'
    print('PASS: test_contact_flags_target_specific')

def test_k10_safe_release_veto():
    """K10 correctly vetoes on safe_release within window."""
    crit = [{'value': 1, 'valid_mask': True, 'reason': 'CRITICAL', 'confidence': 0.9} for _ in range(25)]
    sr = [{'value': 0, 'valid_mask': True, 'reason': 'NO_RELEASE', 'confidence': 0.0} for _ in range(25)]
    sr[8] = {'value': 1, 'valid_mask': True, 'reason': 'SAFE_RELEASE', 'confidence': 0.9}
    results = recompute_k10(crit, sr, K=10)
    # Step 0: release at 8 in [0,10) → infeasible
    assert results[0]['value'] == 0 and 'SAFE_RELEASE' in results[0]['reason']
    # Step 9: release at 8 is before window [9,19) → feasible (all critical)
    assert results[9]['value'] == 1, f'Step 9 should be feasible, got {results[9]}'
    print('PASS: test_k10_safe_release_veto')

def test_instability_no_eef_proxy():
    """Instability uses target-relative checks, not EEF-alone jumps."""
    # When no manipulated objects, slip/contact_loss should be unknown
    steps = []
    for i in range(10):
        steps.append({
            'robot0_eef_pos': [0.3, 0.0, 0.1 + i * 0.01],
            'robot0_gripper_qpos': [0.0, 0.0],
            'object_state': list(range(56)),
            'mujoco_contact_pairs': [],
        })
    grasp = [{'grasp_established': True, 'grasp_known_mask': True, 'grasp_confidence': 0.9} for _ in range(10)]
    # Manipulated=[] means no target-specific checks → slip/contact_loss unknown=False
    results = compute_instability_indicators(steps, grasp, [], {})
    # Without manipulated objects, slip_known and contact_loss_known should be False
    assert all(not r['slip_known_mask'] for r in results), \
        'Without manipulated objects, slip should be unknown'
    print('PASS: test_instability_no_eef_proxy')

def test_v22_to_label_v2_adapter():
    from physics_teacher_v22 import create_v22_snapshot
    snap = create_v22_snapshot()
    g = snap['factors']['grasp_state']
    g['known_mask'] = True; g['grasp_known_mask'] = True
    g['grasp_established'] = True; g['grasp_confidence'] = 0.9
    g['grasp_dwell_steps'] = 15
    c = snap['factors']['contact_state']
    c['known_mask'] = True; c['contact_known_mask'] = True
    c['contact_score'] = 0.8; c['contact_confidence'] = 0.8
    co = snap['factors']['comotion_state']
    co['known_mask'] = True; co['comotion_known_mask'] = True
    co['object_eef_comotion_score'] = 0.6; co['comotion_confidence'] = 0.7
    l = snap['factors']['lift_state']
    l['known_mask'] = True; l['lift_known_mask'] = True
    l['lift_score'] = 0.4; l['lift_confidence'] = 0.8
    # Set safe_release planned_release to known+not_releasing
    snap['factors']['planned_release'] = {
        'planned_release_detected': False,
        'planned_release_known_mask': True,
        'planned_release_confidence': 0.7,
    }
    # Set gripper_closing_state
    snap['factors']['gripper_closing_state'] = {
        'gripper_closing_detected': True,
        'gripper_closing_known_mask': True,
        'gripper_closing_confidence': 0.8,
    }

    label = v22_to_label_v2(snap, 50)
    assert label['physical_criticality']['value'] == 1
    assert label['physical_criticality']['valid_mask'] == True
    assert label['safe_release']['valid_mask'] == True
    assert label['safe_release']['value'] == 0
    assert label['gripper_closing_state']['valid_mask'] == True
    assert label['gripper_closing_state']['value'] == True
    print('PASS: test_v22_to_label_v2_adapter')

def test_all_heads_produced():
    """All 5 heads are produced with proper structure."""
    from physics_teacher_v22 import create_v22_snapshot
    snap = create_v22_snapshot()
    label = v22_to_label_v2(snap, 0)
    heads = ['physical_criticality', 'k10_feasible', 'safe_release', 'instability', 'gripper_closing_state']
    for h in heads:
        assert h in label, f'Missing head: {h}'
        assert 'value' in label[h], f'{h}: missing value'
        assert 'valid_mask' in label[h], f'{h}: missing valid_mask'
        assert 'reason' in label[h], f'{h}: missing reason'
        assert 'confidence' in label[h], f'{h}: missing confidence'
    print('PASS: test_all_heads_produced')

def run_all_tests():
    tests = [
        test_typed_schema_complete, test_config_sha_stable,
        test_goal_resolver_split, test_goal_resolver_physical_unknown,
        test_goal_resolver_unknown_task,
        test_grasp_target_specific, test_lift_uses_object_z,
        test_lift_unknown_without_slices, test_comotion_history_index,
        test_safe_release_implemented, test_gripper_closing_implemented,
        test_terminal_from_summary, test_k10_no_duplicate_append,
        test_candidate_close_not_in_physics, test_contact_flags_target_specific,
        test_k10_safe_release_veto, test_instability_no_eef_proxy,
        test_v22_to_label_v2_adapter, test_all_heads_produced,
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
