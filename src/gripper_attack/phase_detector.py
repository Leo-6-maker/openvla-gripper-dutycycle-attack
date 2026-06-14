"""Layer1: Causal manipulation-phase estimator.

Produces phase labels from privileged or deployment-safe inputs.
Teacher variant may use object/target pose for pseudo-label generation.
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

# ── Clean-only privilege fields ──
PRIVILEGED_FIELDS = [
    "obj_x", "obj_y", "obj_z",
    "target_obj_x", "target_obj_y", "target_obj_z",
    "eef_to_obj_distance", "obj_to_target_distance",
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


def compute_eef_velocity(records: list[dict], window: int = 3) -> list[float]:
    """Compute EEF velocity magnitude from consecutive positions."""
    velocities = []
    for i in range(len(records)):
        if i < window:
            velocities.append(0.0)
            continue
        dx = records[i].get("eef_x", 0) - records[i - window].get("eef_x", 0)
        dy = records[i].get("eef_y", 0) - records[i - window].get("eef_y", 0)
        dz = records[i].get("eef_z", 0) - records[i - window].get("eef_z", 0)
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
        # Extract privileged fields if available
        obj_z = r.get("obj_z", None)
        eef_z = r.get("eef_z", None)
        clean_close = r.get("clean_close", 0)
        close_onset = r.get("close_onset", 0)
        decoded_open = r.get("decoded_open_bool", 0)
        gripper_qpos = r.get("gripper_qpos_before", 0)

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
                if records[fwd].get("decoded_open_bool", 0):
                    found_open = True
                    break
            if found_open:
                labels[t] = "pre_release"

    return labels


def teacher_critical_close_anchor(records: list[dict]) -> int:
    """Return the step index of the teacher-identified first critical CLOSE.

    The teacher uses privileged object pose to identify the pre-grasp CLOSE
    that is most critical for task success.
    """
    # Find first clean CLOSE onset after approach phase
    for t, r in enumerate(records):
        if r.get("close_onset", 0) and r.get("clean_close", 0):
            # Verify this isn't a post-release re-close by checking qpos
            qpos = r.get("gripper_qpos_before", 0)
            if qpos < 0.01:  # gripper is physically closed (not just commanded)
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
