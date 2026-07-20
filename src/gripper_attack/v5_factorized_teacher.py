"""Factorized causal physics Teacher (Gate F1).

Three independent state heads with prefix invariance:
  grasp_established      — object stably held
  manipulation_active    — grasped object being transported/placed
  release_or_instability — object being released, dropped, or regrasped

Mechanism routing from BDDL task role only — no task ID, state ID, or
attack outcome leakage.

Design constraint (frozen): no threshold, formula, or route rule may be
changed based on FIT-DEV, CAL, CHECK, CS200, or any attack result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .action_contract import CanonicalActionState
from .v5_physics import (
    PhysicsTaskRole,
    _clip,
    _comotion,
    _contact_flags,
    _cosine,
    _dist,
    _finite_vector,
    _lift,
    _mean,
    _median,
    _object_slice,
    _relative_stability,
    _slice_vector,
    _target_progress,
    parse_bddl_task_role,
)


class MechanismRoute(Enum):
    SINGLE_OBJECT_PICK_PLACE = "single_object_pick_place"
    MULTI_OBJECT_TRANSFER = "multi_object_transfer"
    ARTICULATED_OR_PLANAR = "articulated_or_planar"
    UNKNOWN_OR_AMBIGUOUS = "unknown_or_ambiguous"

    @property
    def supported(self) -> bool:
        return self in (MechanismRoute.SINGLE_OBJECT_PICK_PLACE, MechanismRoute.MULTI_OBJECT_TRANSFER)


class EventRole(Enum):
    TARGET = "TARGET"
    DISTRACTOR = "DISTRACTOR"
    NONE = "NONE"


FACTORIZED_TEACHER_FIELDS = frozenset({
    "step", "canonical_parent_key", "state_id",
    "mechanism_type", "event_id", "event_role", "target_relevant",
    "close_event_onset",
    "grasp_established", "grasp_established_known_mask", "grasp_established_confidence",
    "manipulation_active", "manipulation_active_known_mask", "manipulation_active_confidence",
    "release_or_instability", "release_or_instability_known_mask", "release_or_instability_confidence",
    "strict_k10_feasible", "strict_k10_known_mask",
    "gripper_contact_score", "relative_pose_stability", "object_eef_comotion_score",
    "lift_score", "support_removed", "target_progress", "target_progress_known",
    "student_valid", "action_intent", "action_known", "candidate_close",
    "physics_protocol_schema", "source_artifact_recursive_sha256",
})

FACTORIZED_TEACHER_SCHEMA = "DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1"
FACTORIZED_LABEL_FILENAME = "factorized_teacher_v1.jsonl"
FACTORIZED_MANIFEST_SCHEMA = "DETECTOR_V5_FACTORIZED_TEACHER_V1_MANIFEST"


def _determine_mechanism(role: PhysicsTaskRole, object_names: Sequence[str]) -> MechanismRoute:
    """Route from BDDL task role only."""
    if not role.applicable:
        return MechanismRoute.UNKNOWN_OR_AMBIGUOUS

    n_manipulated = len(role.manipulated_objects)
    n_targets = len(role.target_names)

    # Check for articulated/planar predicates
    for pred, _, _ in role.goal_predicates:
        if pred in ("Open", "Close"):
            return MechanismRoute.ARTICULATED_OR_PLANAR

    if n_manipulated == 0:
        return MechanismRoute.UNKNOWN_OR_AMBIGUOUS

    if n_manipulated >= 2 and n_targets >= 2:
        return MechanismRoute.MULTI_OBJECT_TRANSFER

    if n_manipulated == 1:
        return MechanismRoute.SINGLE_OBJECT_PICK_PLACE

    return MechanismRoute.UNKNOWN_OR_AMBIGUOUS


def _opening_trend(
    action_history: Sequence[Mapping[str, Any]],
    window: int = 3,
) -> float:
    """Recent OPEN action trend: 1.0 = consistently opening, 0.0 = not."""
    if len(action_history) < 2:
        return 0.0
    recent = action_history[-window:]
    opens = sum(
        1 for r in recent
        if r.get("action_known") is True and r.get("action_intent") == "OPEN"
    )
    return _clip(opens / max(1, len(recent)))


def _contact_loss_evidence(
    contact_history: Sequence[bool],
    window: int = 5,
) -> float:
    """Evidence of recent contact loss: 1.0 = lost, 0.0 = maintained."""
    if len(contact_history) < 2:
        return 0.0
    recent = contact_history[-window:]
    had_contact = any(recent[:-1])
    lost = had_contact and not recent[-1]
    return 1.0 if lost else 0.0


def _object_eef_separation(
    history: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
) -> float:
    """Increasing object-EEF distance: 1.0 = separating, 0.0 = stable."""
    if len(history) < 2:
        return 0.0
    values: list[float] = []
    for name in role.manipulated_objects:
        spec = _object_slice(object_slices, name)
        if spec is None:
            continue
        positions = [_slice_vector(item.get("object_state", []), spec, "to_eef_pos") for item in history[-2:]]
        positions = [p for p in positions if p is not None]
        if len(positions) >= 2:
            dist_now = _dist(positions[-1], [0.0, 0.0, 0.0])
            dist_prev = _dist(positions[-2], [0.0, 0.0, 0.0])
            if dist_prev > 1e-6:
                values.append(_clip((dist_now - dist_prev) / 0.02))
    return max(values) if values else 0.0


def _gripper_qpos_closure(
    sidecar: Mapping[str, Any],
) -> float:
    """Gripper qpos indicates closure: 1.0 = closed, 0.0 = wide open."""
    qpos = _finite_vector(sidecar.get("robot0_gripper_qpos"), 2)
    if qpos is None:
        return 0.0
    # Both fingers near zero = closed
    closure = 1.0 - _clip((abs(qpos[0]) + abs(qpos[1])) / 0.08)
    return closure


def _horizontal_transport(
    history: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
) -> float:
    """Object has moved horizontally from initial position: 1.0 = transported."""
    if len(history) < 2:
        return 0.0
    values: list[float] = []
    for name in role.manipulated_objects:
        spec = _object_slice(object_slices, name)
        if spec is None:
            continue
        initial = _slice_vector(history[0].get("object_state", []), spec, "pos")
        current = _slice_vector(history[-1].get("object_state", []), spec, "pos")
        if initial is not None and current is not None:
            h_dist = math.sqrt((current[0] - initial[0]) ** 2 + (current[1] - initial[1]) ** 2)
            values.append(_clip(h_dist / 0.05))
    return max(values) if values else 0.0


def derive_factorized_rows(
    step_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
    object_names: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive factorized three-head Teacher labels for one episode."""

    if len(step_rows) != len(sidecar_rows) or not step_rows:
        raise ValueError("step and sidecar lengths must match and be nonzero")

    constants = protocol["fixed_constants"]
    history_size = int(protocol["history"]["score_window_steps"])
    mechanism = _determine_mechanism(role, object_names)

    # ── Compute action states ──────────────────────────────────────────
    action_states: list[CanonicalActionState] = []
    for step in step_rows:
        state = CanonicalActionState.from_step(step, field="clean_action_raw_7d")
        step["_action_state"] = state
        action_states.append(state)

    # ── Compute contact and physics evidence ────────────────────────────
    contact_flags: list[tuple[bool, bool, bool]] = []
    gripper_contacts: list[bool] = []
    qpos_closures: list[float] = []
    for sidecar in sidecar_rows:
        cf = _contact_flags(sidecar.get("mujoco_contact_pairs", []), role)
        contact_flags.append(cf)
        gripper_contacts.append(cf[1])
        qpos_closures.append(_gripper_qpos_closure(sidecar))

    # ── Per-step physics ────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for index in range(len(step_rows)):
        history = sidecar_rows[max(0, index - history_size + 1):index + 1]
        contact = contact_flags[index]
        has_grip = contact[1]

        stable = _relative_stability(
            history, role, object_slices,
            float(constants["relative_position_scale_m"]),
            float(constants["relative_quaternion_scale"]),
            reducer="median",
        )
        comotion = _comotion(history, role, object_slices, inactive_value=0.0)
        lift = _lift(
            sidecar_rows[:index + 1], sidecar_rows[index], role, object_slices,
            float(constants["lift_scale_m"]),
        )
        tp, tp_known = _target_progress(
            sidecar_rows[0], sidecar_rows[index], role, object_slices,
            float(constants["target_progress_scale_m"]),
            unknown_default=0.0,
        )

        as_ = action_states[index]
        action_known = as_.action_known
        candidate_close = as_.candidate_close

        # ── Evidence components ──────────────────────────────────────
        opening_now = 1.0 if (action_known and as_.action_intent == "OPEN") else 0.0
        qpos_closed = qpos_closures[index]
        contact_loss = _contact_loss_evidence(gripper_contacts[:index + 1], window=5)
        separation = _object_eef_separation(
            sidecar_rows[max(0, index - 3):index + 1], role, object_slices,
        )

        # Contact toggle rate
        contact_hist = gripper_contacts[max(0, index - history_size + 1):index + 1]
        toggles = sum(
            left != right for left, right in zip(contact_hist[1:], contact_hist[:-1])
        )
        toggle_rate = toggles / max(1, len(contact_hist) - 1)

        # Contact dwell (consecutive steps with gripper contact)
        contact_dwell = 0
        for j in range(index, -1, -1):
            if gripper_contacts[j]:
                contact_dwell += 1
            else:
                break

        # Support removal (cumulative: had support before, none now)
        has_support_now = contact[2]
        had_support_before = any(contact_flags[j][2] for j in range(max(0, index - 10), index + 1))
        support_removed = 1.0 if had_support_before and not has_support_now else 0.0

        # Horizontal transport
        h_transport = _horizontal_transport(
            sidecar_rows[:index + 1], role, object_slices,
        )

        # ── grasp_established ─────────────────────────────────────────
        role_ok = role.applicable
        physics_valid = (
            stable > 0.3
            and has_grip
            and contact_dwell >= 3
            and contact_loss < 0.5
        )
        grasp_known = bool(role_ok and action_known and physics_valid)
        grasp_value = False
        grasp_conf = 0.0
        if grasp_known:
            grasp_score = _clip(
                0.30 * stable
                + 0.25 * float(has_grip)
                + 0.20 * qpos_closed
                + 0.15 * (1.0 - toggle_rate)
                + 0.10 * (1.0 - contact_loss)
            )
            grasp_value = grasp_score >= 0.5
            grasp_conf = grasp_score

        # ── manipulation_active ───────────────────────────────────────
        manip_known = bool(grasp_known and grasp_value)
        manip_value = False
        manip_conf = 0.0
        if manip_known:
            manip_score = _clip(
                0.25 * lift
                + 0.25 * h_transport
                + 0.20 * comotion
                + 0.15 * support_removed
                + 0.15 * float(tp_known and tp > 0.1)
            )
            manip_value = manip_score >= 0.4
            manip_conf = manip_score

        # ── release_or_instability ────────────────────────────────────
        release_known = bool(role_ok and action_known)
        release_value = False
        release_conf = 0.0
        if release_known:
            release_score = _clip(
                0.25 * contact_loss
                + 0.20 * separation
                + 0.20 * opening_now
                + 0.20 * toggle_rate
                + 0.15 * (1.0 - stable)
            )
            release_value = release_score >= 0.5
            release_conf = release_score

        # ── strict_k10_feasible (evaluation only, future-allowed) ────
        future = action_states[index:min(len(action_states), index + 10)]
        future_close = any(
            s.action_known and s.action_intent == "CLOSE"
            for s in future[1:]
        )
        k10_known = bool(role_ok and candidate_close)
        k10_feasible = bool(k10_known and future_close)

        # ── close_event_onset ─────────────────────────────────────────
        close_onset = False
        if index > 0:
            prev_grasp = rows[index - 1].get("grasp_established", False) if rows else False
            close_onset = candidate_close and grasp_value and not prev_grasp

        # ── Event ID assignment (simplified: per contiguous grasp segment) ──
        event_id = -1
        event_role = EventRole.NONE.value
        target_relevant = False
        if grasp_value:
            if index == 0 or not (rows[index - 1].get("grasp_established", False) if rows else False):
                event_id = index  # new event starts
            else:
                event_id = rows[index - 1].get("event_id", index)
            event_role = EventRole.TARGET.value
            target_relevant = True
        elif index > 0 and rows[index - 1].get("grasp_established", False):
            event_id = rows[index - 1].get("event_id", -1)

        rows.append({
            "step": index,
            "mechanism_type": mechanism.value,
            "event_id": event_id,
            "event_role": event_role,
            "target_relevant": target_relevant,
            "close_event_onset": close_onset,
            "grasp_established": grasp_value,
            "grasp_established_known_mask": grasp_known,
            "grasp_established_confidence": round(grasp_conf, 4),
            "manipulation_active": manip_value,
            "manipulation_active_known_mask": manip_known,
            "manipulation_active_confidence": round(manip_conf, 4),
            "release_or_instability": release_value,
            "release_or_instability_known_mask": release_known,
            "release_or_instability_confidence": round(release_conf, 4),
            "strict_k10_feasible": k10_feasible,
            "strict_k10_known_mask": k10_known,
            "gripper_contact_score": 1.0 if has_grip else 0.0,
            "relative_pose_stability": stable,
            "object_eef_comotion_score": comotion,
            "lift_score": lift,
            "support_removed": support_removed,
            "target_progress": tp,
            "target_progress_known": tp_known,
            "student_valid": bool(step_rows[index].get("valid", True)),
            "action_intent": as_.action_intent,
            "action_known": as_.action_known,
            "candidate_close": candidate_close,
        })

    # ── Post-pass: fill event_id for non-grasp steps ──────────────────
    current_event = -1
    for row in rows:
        if row["event_id"] >= 0:
            current_event = row["event_id"]
        else:
            row["event_id"] = current_event
        if row["event_role"] == EventRole.NONE.value and current_event >= 0:
            row["event_role"] = EventRole.TARGET.value

    # ── Event summary ─────────────────────────────────────────────────
    events: list[dict[str, Any]] = []
    seen_events: set[int] = set()
    for row in rows:
        eid = row["event_id"]
        if eid < 0 or eid in seen_events:
            continue
        seen_events.add(eid)
        members = [r for r in rows if r["event_id"] == eid]
        events.append({
            "event_id": eid,
            "mechanism_type": mechanism.value,
            "event_role": EventRole.TARGET.value,
            "start_step": members[0]["step"],
            "end_step": members[-1]["step"],
            "step_count": len(members),
            "has_grasp": any(r["grasp_established"] for r in members),
            "has_manipulation": any(r["manipulation_active"] for r in members),
            "has_release": any(r["release_or_instability"] for r in members),
            "grasp_steps": sum(1 for r in members if r["grasp_established"]),
            "manipulation_steps": sum(1 for r in members if r["manipulation_active"]),
            "release_steps": sum(1 for r in members if r["release_or_instability"]),
        })

    return rows, events


# ── Prefix invariance validator ────────────────────────────────────────────

def verify_prefix_invariance(
    step_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
    object_names: Sequence[str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify primary heads are prefix-invariant: computing at step t
    should give same result whether we see only prefix 0:t or full trajectory."""

    T = len(step_rows)
    violations = 0
    head_names = ("grasp_established", "manipulation_active", "release_or_instability")

    # Full trajectory
    full_rows, _ = derive_factorized_rows(
        step_rows, sidecar_rows, role, object_slices, object_names, protocol,
    )

    for t in range(1, T):
        prefix_steps = step_rows[:t + 1]
        prefix_sidecars = sidecar_rows[:t + 1]
        prefix_rows, _ = derive_factorized_rows(
            prefix_steps, prefix_sidecars, role, object_slices, object_names, protocol,
        )
        for head in head_names:
            if prefix_rows[t][head] != full_rows[t][head]:
                violations += 1

    return {
        "total_steps": T,
        "violations": violations,
        "prefix_invariant": violations == 0,
        "heads_checked": list(head_names),
    }


__all__ = [
    "FACTORIZED_TEACHER_FIELDS",
    "FACTORIZED_TEACHER_SCHEMA",
    "FACTORIZED_LABEL_FILENAME",
    "FACTORIZED_MANIFEST_SCHEMA",
    "MechanismRoute",
    "EventRole",
    "_determine_mechanism",
    "derive_factorized_rows",
    "verify_prefix_invariance",
]
