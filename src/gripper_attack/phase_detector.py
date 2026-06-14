"""Layer1: Causal manipulation-phase estimator.

Two teacher variants:
  Teacher-P: privileged simulation teacher — uses object/target pose to
             identify task-critical grasp closures. Abstains if privileged
             fields are unavailable or insufficient.
  Teacher-R: clean-action rule teacher — low-cost rule baseline using only
             deployment-safe action/gripper fields. Fast but may confuse
             spurious early closes with critical ones.

Student variant must use only proprio/action history.
"""

from __future__ import annotations

import numpy as np

# ── Phase taxonomy ──
PHASE_LABELS = [
    "approach",
    "grasp_close",
    "lift",
    "carry",
    "pre_close_transition",
    "pre_release",
    "release_safe",
    "other",
]

# ── Privileged fields (Teacher-P only) ──
PRIVILEGED_FIELDS = [
    "obj_x", "obj_y", "obj_z",
    "target_obj_x", "target_obj_y", "target_obj_z",
    "eef_to_obj_distance", "obj_to_target_distance",
]

PRIVILEGED_REQUIRED_FIELDS = ["eef_to_obj_distance", "obj_to_target_distance"]

# Grasp privilege: needs eef + object pose only (NOT target)
GRASP_PRIVILEGE_FIELDS = ["eef_x", "eef_y", "eef_z", "obj_x", "obj_y", "obj_z",
                           "eef_to_obj_distance"]
# Placement privilege: needs grasp + target pose
PLACEMENT_PRIVILEGE_FIELDS = ["target_obj_x", "target_obj_y", "target_obj_z",
                               "obj_to_target_distance"]

# Fields that must be locally valid at each candidate close for Teacher-P
PRIVILEGED_LOCAL_CANDIDATE_FIELDS = [
    "eef_to_obj_distance", "obj_x", "obj_y", "obj_z"
]

# ── Deployment-safe fields ──
DEPLOYMENT_SAFE_FIELDS = [
    "step",
    "clean_gripper_env", "clean_gripper_raw",
    "gripper_qpos_before", "gripper_qpos_after",
    "qpos_abs_before", "qpos_abs_after",
    "eef_x", "eef_y", "eef_z",
    "clean_close", "close_onset", "close_streak",
    "decoded_open_bool",
]

# ── Teacher-P thresholds ──
EEF_TO_OBJ_NEAR_THRESHOLD = 0.08       # meters: EEF within 8cm of object
OBJECT_LIFT_MIN_DELTA = 0.005           # meters: 5mm movement = lift evidence
OBJECT_LIFT_LOOKAHEAD = 15              # steps: look ahead for lift after close
SUSTAINED_MOTION_FRAMES = 2             # consecutive frames for robust evidence

# ── Motion evidence types ──
MOTION_SUSTAINED_VERTICAL_LIFT = "sustained_vertical_lift"
MOTION_SUSTAINED_HORIZONTAL_TRANSPORT = "sustained_horizontal_transport"
MOTION_NO_SUSTAINED_MOTION = "no_sustained_motion"


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _field_is_valid(r: dict, field: str) -> bool:
    """Check a single field is present, non-empty, non-NaN."""
    val = r.get(field)
    if val is None or val == "":
        return False
    try:
        if np.isnan(float(val)):
            return False
    except (ValueError, TypeError):
        return False
    return True


def _check_privileged_fields_available(records: list[dict]) -> bool:
    """Return True if privileged fields are present and valid in at least
    one record (global availability check)."""
    if not records:
        return False
    sample = records[:min(3, len(records))]
    for field in PRIVILEGED_REQUIRED_FIELDS:
        found = False
        for r in sample:
            if _field_is_valid(r, field):
                found = True
                break
        if not found:
            return False
    return True


def _check_grasp_privilege_valid(records: list[dict], t: int,
                                  lookahead: int = None) -> bool:
    """Check grasp privilege: eef pose + object pose + eef_to_obj_distance
    are locally valid at step t and in the lookahead window.
    Does NOT require target pose."""
    if lookahead is None:
        lookahead = OBJECT_LIFT_LOOKAHEAD
    T = len(records)
    for i in range(t, min(t + lookahead, T)):
        r = records[i]
        # Check eef pose validity
        eef_valid = all(_field_is_valid(r, f) for f in ["eef_x", "eef_y", "eef_z"])
        # Check object pose validity
        obj_valid = all(_field_is_valid(r, f) for f in ["obj_x", "obj_y", "obj_z"])
        # Check distance validity
        dist_valid = _field_is_valid(r, "eef_to_obj_distance")
        if not (eef_valid and obj_valid and dist_valid):
            return False
    return True


def _check_placement_privilege_valid(records: list[dict], t: int,
                                      lookahead: int = None) -> bool:
    """Check placement privilege: grasp privilege + target pose.
    Returns True only when target coordinates are locally valid."""
    if not _check_grasp_privilege_valid(records, t, lookahead):
        return False
    if lookahead is None:
        lookahead = OBJECT_LIFT_LOOKAHEAD
    T = len(records)
    for i in range(t, min(t + lookahead, T)):
        r = records[i]
        tgt_valid = all(_field_is_valid(r, f) for f in
                        ["target_obj_x", "target_obj_y", "target_obj_z"])
        if not tgt_valid:
            return False
    return True


def _check_local_privileged_fields(records: list[dict], t: int,
                                     lookahead: int = None) -> bool:
    """Verify privileged fields exist and are valid at candidate close `t`
    and in the lookahead window. Returns False if any required field is
    missing, empty, or NaN — prevents _safe_float silently defaulting to 0."""
    if lookahead is None:
        lookahead = OBJECT_LIFT_LOOKAHEAD
    T = len(records)
    check_range = range(t, min(t + lookahead, T))
    for i in check_range:
        r = records[i]
        for field in PRIVILEGED_LOCAL_CANDIDATE_FIELDS:
            val = r.get(field)
            if val is None or val == "":
                return False
            try:
                if np.isnan(float(val)):
                    return False
            except (ValueError, TypeError):
                return False
    return True


def _classify_motion_evidence(records: list[dict], anchor_t: int,
                                lookahead: int = OBJECT_LIFT_LOOKAHEAD,
                                min_delta: float = OBJECT_LIFT_MIN_DELTA,
                                sustained_frames: int = SUSTAINED_MOTION_FRAMES,
                                eef_near_threshold: float = EEF_TO_OBJ_NEAR_THRESHOLD) -> dict:
    """Classify post-close object motion using CUMULATIVE displacement from anchor.

    Uses cumulative_z = obj_z[t] - obj_z[anchor] rather than per-frame deltas.
    This prevents control-frequency sensitivity: slow but real lifts that don't
    exceed min_delta per individual frame are still captured.

    Returns dict with:
      motion_evidence_type: one of MOTION_* constants
      cumulative_vertical_dz: total positive z displacement from anchor
      cumulative_horizontal_dxy: total horizontal displacement from anchor
      sustained_above_threshold_frames: consecutive frames where cumulative_z >= min_delta
      eef_attachment_consistent: whether EEF stayed near object during motion
    """
    T = len(records)
    result = {
        "motion_evidence_type": MOTION_NO_SUSTAINED_MOTION,
        "cumulative_vertical_dz": 0.0,
        "cumulative_horizontal_dxy": 0.0,
        "sustained_above_threshold_frames": 0,
        "eef_attachment_consistent": True,
    }
    if anchor_t >= T - sustained_frames:
        return result

    anchor_obj_y = _safe_float(records[anchor_t].get("obj_y", 0))
    anchor_obj_z = _safe_float(records[anchor_t].get("obj_z", 0))

    max_cumulative_dz = 0.0
    max_cumulative_dxy = 0.0
    vert_above_cons = 0
    horiz_above_cons = 0
    max_vert_cons = 0
    max_horiz_cons = 0

    for future_t in range(anchor_t + 1, min(anchor_t + lookahead, T)):
        obj_y_after = _safe_float(records[future_t].get("obj_y", 0))
        obj_z_after = _safe_float(records[future_t].get("obj_z", 0))
        eef_to_obj = _safe_float(records[future_t].get("eef_to_obj_distance", 999))

        # Cumulative displacement from anchor
        cumulative_dz = obj_z_after - anchor_obj_z
        cumulative_dy = abs(obj_y_after - anchor_obj_y)
        eef_near = eef_to_obj < eef_near_threshold

        max_cumulative_dz = max(max_cumulative_dz, cumulative_dz)
        max_cumulative_dxy = max(max_cumulative_dxy, cumulative_dy)

        # Vertical lift: cumulative z >= threshold AND EEF near
        if cumulative_dz >= min_delta and eef_near:
            vert_above_cons += 1
        else:
            vert_above_cons = 0

        # Horizontal transport: cumulative xy >= threshold AND EEF near AND no vertical
        if cumulative_dy >= min_delta and eef_near and cumulative_dz < min_delta:
            horiz_above_cons += 1
        else:
            horiz_above_cons = 0

        max_vert_cons = max(max_vert_cons, vert_above_cons)
        max_horiz_cons = max(max_horiz_cons, horiz_above_cons)

        if not eef_near:
            result["eef_attachment_consistent"] = False

    result["cumulative_vertical_dz"] = max_cumulative_dz
    result["cumulative_horizontal_dxy"] = max_cumulative_dxy

    if max_vert_cons >= sustained_frames:
        result["motion_evidence_type"] = MOTION_SUSTAINED_VERTICAL_LIFT
        result["sustained_above_threshold_frames"] = max_vert_cons
    elif max_horiz_cons >= sustained_frames:
        result["motion_evidence_type"] = MOTION_SUSTAINED_HORIZONTAL_TRANSPORT
        result["sustained_above_threshold_frames"] = max_horiz_cons

    return result


def _check_sustained_object_lift(records: list[dict], anchor_t: int,
                                  lookahead: int = OBJECT_LIFT_LOOKAHEAD,
                                  min_delta: float = OBJECT_LIFT_MIN_DELTA,
                                  sustained_frames: int = SUSTAINED_MOTION_FRAMES) -> bool:
    """Robust lift: object VERTICAL displacement sustained for >= `sustained_frames`
    consecutive frames, with EEF remaining near object during movement.

    This is equivalent to _classify_motion_evidence returning
    MOTION_SUSTAINED_VERTICAL_LIFT. Horizontal-only transport does NOT qualify.
    """
    evidence = _classify_motion_evidence(
        records, anchor_t, lookahead=lookahead, min_delta=min_delta,
        sustained_frames=sustained_frames)
    return evidence["motion_evidence_type"] == MOTION_SUSTAINED_VERTICAL_LIFT


def compute_eef_velocity(records: list[dict], window: int = 3) -> list[float]:
    """Compute EEF velocity magnitude from consecutive positions."""
    velocities = []
    for i in range(len(records)):
        if i < window:
            velocities.append(0.0)
            continue
        dx = _safe_float(records[i].get("eef_x", 0)) - _safe_float(records[i - window].get("eef_x", 0))
        dy = _safe_float(records[i].get("eef_y", 0)) - _safe_float(records[i - window].get("eef_y", 0))
        dz = _safe_float(records[i].get("eef_z", 0)) - _safe_float(records[i - window].get("eef_z", 0))
        velocities.append(float(np.sqrt(dx**2 + dy**2 + dz**2)))
    return velocities


def teacher_rule_phase_labels(records: list[dict]) -> list[str]:
    """Teacher-R: rule-based phase labels using deployment-safe fields + time.

    Uses only clean action/gripper/proprio history and absolute step heuristics.
    This is a fast rule baseline, NOT a privileged phase teacher.
    """
    T = len(records)
    labels = ["other"] * T

    # Pre-compute EEF velocity
    eef_vel = compute_eef_velocity(records)

    for t in range(T):
        r = records[t]
        clean_close = int(_safe_float(r.get("clean_close", 0)))
        close_onset = int(_safe_float(r.get("close_onset", 0)))
        decoded_open = int(_safe_float(r.get("decoded_open_bool", 0)))
        gripper_qpos = _safe_float(r.get("gripper_qpos_before", 0))

        # release_safe: gripper OPEN after a period of CLOSE (post-grasp release)
        if decoded_open and gripper_qpos > 0.01 and t > 20:
            labels[t] = "release_safe"
            continue

        # grasp_close: first CLOSE onset
        if close_onset and clean_close:
            labels[t] = "grasp_close"
            continue

        # pre_close_transition: CLOSE command issued but gripper not yet responded
        if clean_close and gripper_qpos < 0.005 and not decoded_open:
            labels[t] = "pre_close_transition"
            continue

        # carry: EEF moving with gripper still CLOSE
        if clean_close and eef_vel[t] > 0.001 and t > 10:
            labels[t] = "carry"
            continue

        # lift: early phase with high EEF velocity and CLOSE
        if clean_close and eef_vel[t] > 0.002 and t < 30:
            labels[t] = "lift"
            continue

        # approach: early phase, EEF moving toward object
        if eef_vel[t] > 0.001 and t < 20:
            labels[t] = "approach"
            continue

        # pre_release: last few CLOSE steps before OPEN
        if clean_close and t > T - 20:
            found_open = False
            for fwd in range(t + 1, min(t + 15, T)):
                if int(_safe_float(records[fwd].get("decoded_open_bool", 0))):
                    found_open = True
                    break
            if found_open:
                labels[t] = "pre_release"

    return labels


def teacher_rule_critical_close_anchor(records: list[dict]) -> int:
    """Teacher-R: rule-based first critical CLOSE anchor (clean-only, no privilege).

    Uses only deployment-safe features: close_onset, clean_close, gripper_qpos.
    Fast but may accept spurious early closes that are not task-critical
    (e.g. butter_s2 step 4 before EEF reaches object).

    Returns step index or -1 if no close found.
    """
    for t, r in enumerate(records):
        if int(_safe_float(r.get("close_onset", 0))) and int(_safe_float(r.get("clean_close", 0))):
            qpos = _safe_float(r.get("gripper_qpos_before", 0))
            if qpos < 0.01:
                return t
    return -1


def teacher_privileged_critical_close_anchor(records: list[dict]) -> int:
    """Teacher-P: privileged critical-close anchor using grasp privilege.

    Requires only grasp_privilege (eef pose + object pose + distance).
    Does NOT require placement privilege (target pose).

    A CLOSE is considered \"critical\" only when:
      1. Grasp privilege fields are locally valid at the candidate close
      2. EEF is near the object (eef_to_obj_distance < threshold)
      3. The CLOSE is followed by sustained VERTICAL lift (>=2 consecutive
         frames of positive z with EEF remaining near object)
      4. Gripper is not already open

    Abstains (returns -1) if:
      - Grasp privilege check fails (object/eef pose not valid)
      - No close satisfies all criticality conditions

    The caller can check _check_placement_privilege_valid() separately
    for placement-specific capabilities.
    """
    T = len(records)

    for t in range(T):
        r = records[t]

        # Must be a CLOSE onset
        if not (int(_safe_float(r.get("close_onset", 0))) and
                int(_safe_float(r.get("clean_close", 0)))):
            continue

        # Gripper must not already be open
        if int(_safe_float(r.get("decoded_open_bool", 0))):
            continue

        # ── Grasp privilege check per candidate ──
        if not _check_grasp_privilege_valid(records, t):
            continue

        # EEF must be near object
        eef_to_obj = _safe_float(r.get("eef_to_obj_distance", 999))
        if eef_to_obj > EEF_TO_OBJ_NEAR_THRESHOLD:
            continue

        # Must have sustained VERTICAL lift with EEF proximity
        if not _check_sustained_object_lift(records, t):
            continue

        return t

    return -1


def check_teacher_p_privilege_capability(records: list[dict]) -> dict:
    """Audit Teacher-P privilege capability for this trace.

    Returns dict with:
      grasp_privilege_valid: bool
      placement_privilege_valid: bool
      privilege_missing_fields: list of field names
    """
    result = {
        "grasp_privilege_valid": False,
        "placement_privilege_valid": False,
        "privilege_missing_fields": [],
    }
    if not records:
        return result

    # Check grasp privilege globally
    has_grasp = False
    for field in GRASP_PRIVILEGE_FIELDS:
        if any(_field_is_valid(r, field) for r in records[:3]):
            has_grasp = True
        else:
            result["privilege_missing_fields"].append(field)
    result["grasp_privilege_valid"] = has_grasp and not result["privilege_missing_fields"]

    # Check placement privilege
    has_placement = True
    for field in PLACEMENT_PRIVILEGE_FIELDS:
        if not any(_field_is_valid(r, field) for r in records[:3]):
            has_placement = False
            result["privilege_missing_fields"].append(field)
    result["placement_privilege_valid"] = has_placement and has_grasp

    return result


def teacher_window_proposal(anchor: int, window_len: int = 10,
                            pre_offset: int = 2) -> tuple[int, int]:
    """Return (window_start, window_end) centered around anchor step.

    pre_offset: steps BEFORE anchor to include in window start.
    window_len: total window length.
    """
    if anchor < 0:
        return -1, -1
    ws = max(0, anchor - pre_offset)
    we = ws + window_len
    return ws, we
