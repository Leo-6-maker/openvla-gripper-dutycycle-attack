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


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _check_privileged_fields_available(records: list[dict]) -> bool:
    """Return True if privileged object/target pose fields are present and valid."""
    if not records:
        return False
    # Sample first 3 records to confirm field availability
    sample = records[:min(3, len(records))]
    for field in PRIVILEGED_REQUIRED_FIELDS:
        for r in sample:
            val = r.get(field)
            if val is None or val == "":
                return False
            try:
                if np.isnan(float(val)):
                    return False
            except (ValueError, TypeError):
                return False
    return True


def _check_subsequent_object_lift(records: list[dict], anchor_t: int,
                                   lookahead: int = OBJECT_LIFT_LOOKAHEAD,
                                   min_delta: float = OBJECT_LIFT_MIN_DELTA) -> bool:
    """Check if object moves (is lifted) within `lookahead` steps after `anchor_t`."""
    T = len(records)
    if anchor_t >= T - 1:
        return False

    obj_y_before = _safe_float(records[anchor_t].get("obj_y", 0))
    obj_z_before = _safe_float(records[anchor_t].get("obj_z", 0))

    for future_t in range(anchor_t + 1, min(anchor_t + lookahead, T)):
        obj_y_after = _safe_float(records[future_t].get("obj_y", 0))
        obj_z_after = _safe_float(records[future_t].get("obj_z", 0))

        dy = abs(obj_y_after - obj_y_before)
        dz = obj_z_after - obj_z_before  # positive = lifted

        if dy > min_delta or dz > min_delta:
            return True

    return False


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


def teacher_phase_labels(records: list[dict]) -> list[str]:
    """Privileged teacher: classify each step using object + target pose.

    Returns list of phase labels, one per record.
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

        # ── Rule-based teacher ──
        # release_safe: gripper OPEN after a period of CLOSE (post-grasp release)
        if decoded_open and gripper_qpos > 0.01 and t > 20:
            labels[t] = "release_safe"
            continue

        # grasp_close: first CLOSE onset after approach phase
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
    """Teacher-P: privileged critical-close anchor using object/target pose.

    A CLOSE is considered \"critical\" only when:
      1. EEF is near the object (eef_to_obj_distance < threshold)
      2. The CLOSE is followed by object movement (lift/transport evidence)
      3. Gripper is not already open

    Abstains (returns -1) if:
      - Privileged fields are missing or contain NaN
      - No close satisfies all criticality conditions

    This prevents accepting spurious early closes that occur before the EEF
    reaches the object.
    """
    if not _check_privileged_fields_available(records):
        return -1

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

        # EEF must be near object
        eef_to_obj = _safe_float(r.get("eef_to_obj_distance", 999))
        if eef_to_obj > EEF_TO_OBJ_NEAR_THRESHOLD:
            continue

        # Must have subsequent object lift/transport evidence
        if not _check_subsequent_object_lift(records, t):
            continue

        return t

    return -1


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
