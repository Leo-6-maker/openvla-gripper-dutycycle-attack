"""Factorized causal physics Teacher (Gate F1.1).

Three independent state heads with prefix invariance:
  grasp_established      — object stably held (known negatives exist)
  manipulation_active    — grasped object being transported/placed
  release_or_instability — object being released, dropped, or regrasped

Event state machine per episode:
  IDLE → GRASPED → MANIPULATING → RELEASED → IDLE

Known mask is independent of positive/negative — base_known can be True
while a head value is False (known negative).  Unknown only arises from
unsupported mechanisms, action_unknown, student_invalid, or missing fields.

Design constraint (frozen): no threshold, formula, or route rule may be
changed based on FIT-DEV, CAL, CHECK, CS200, or any attack result.
"""

from __future__ import annotations

import math, re
from enum import Enum
from typing import Any, Mapping, Sequence

from .action_contract import CanonicalActionState
from .v5_physics import (
    PhysicsTaskRole,
    _clip,
    _comotion,
    _contact_flags,
    _dist,
    _finite_vector,
    _lift,
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


class EventPhase(Enum):
    IDLE = "IDLE"
    GRASPED = "GRASPED"
    MANIPULATING = "MANIPULATING"
    RELEASED = "RELEASED"


FACTORIZED_TEACHER_FIELDS = frozenset({
    "step", "canonical_parent_key", "state_id",
    "mechanism_type", "event_id", "event_phase", "event_role", "target_relevant",
    "active_object_name",
    "close_event_onset",
    "grasp_established", "grasp_established_known_mask", "grasp_established_confidence",
    "manipulation_active", "manipulation_active_known_mask", "manipulation_active_confidence",
    "release_or_instability", "release_or_instability_known_mask", "release_or_instability_confidence",
    "strict_k10_feasible", "strict_k10_known_mask", "strict_k10_binding_schema",
    "gripper_contact_score", "relative_pose_stability", "object_eef_comotion_score",
    "lift_score", "support_removed", "target_progress", "target_progress_known",
    "student_valid", "action_intent", "action_known", "candidate_close",
    "physics_protocol_schema", "source_artifact_recursive_sha256",
})

FACTORIZED_TEACHER_SCHEMA = "DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1"
FACTORIZED_LABEL_FILENAME = "factorized_teacher_v1.jsonl"
FACTORIZED_MANIFEST_SCHEMA = "DETECTOR_V5_FACTORIZED_TEACHER_V1_MANIFEST"


def _determine_mechanism(role: PhysicsTaskRole, bddl_text: str | None = None) -> MechanismRoute:
    """Route from BDDL task role only — no task ID, state ID, or attack outcome."""
    # Check BDDL goal text for articulated predicates (Open/Close not in _ROLE_PREDICATES)
    if bddl_text:
        import re as _re
        goal_match = _re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n", bddl_text, flags=_re.DOTALL)
        if goal_match:
            for match in _re.finditer(r"\(([A-Za-z_]+)\s+", goal_match.group(1)):
                if match.group(1) in ("Open", "Close"):
                    return MechanismRoute.ARTICULATED_OR_PLANAR

    if not role.applicable:
        return MechanismRoute.UNKNOWN_OR_AMBIGUOUS

    n_manipulated = len(role.manipulated_objects)
    n_targets = len(role.target_names)

    if n_manipulated == 0:
        return MechanismRoute.UNKNOWN_OR_AMBIGUOUS
    if n_manipulated >= 2 and n_targets >= 2:
        return MechanismRoute.MULTI_OBJECT_TRANSFER
    if n_manipulated == 1:
        return MechanismRoute.SINGLE_OBJECT_PICK_PLACE

    return MechanismRoute.UNKNOWN_OR_AMBIGUOUS


# ── Evidence helpers ────────────────────────────────────────────────────────

def _opening_trend(
    action_states: Sequence[CanonicalActionState],
    index: int,
    window: int = 3,
) -> float:
    """Recent OPEN action trend: 1.0 = consistently opening, 0.0 = not.

    This is the FROZEN definition per protocol: 3-step OPEN proportion.
    """
    start = max(0, index - window + 1)
    recent = action_states[start:index + 1]
    opens = sum(1 for s in recent if s.action_known and s.action_intent == "OPEN")
    return _clip(opens / max(1, len(recent)))


def _contact_loss_evidence(
    contact_history: Sequence[bool],
) -> float:
    """Evidence of recent contact loss: 1.0 = lost, 0.0 = maintained."""
    if len(contact_history) < 2:
        return 0.0
    recent = contact_history[-5:] if len(contact_history) >= 5 else contact_history
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
        positions = [_slice_vector(item.get("object_state", []), spec, "to_eef_pos")
                     for item in history[-2:]]
        positions = [p for p in positions if p is not None]
        if len(positions) >= 2:
            dist_now = _dist(positions[-1], [0.0, 0.0, 0.0])
            dist_prev = _dist(positions[-2], [0.0, 0.0, 0.0])
            if dist_prev > 1e-6:
                values.append(_clip((dist_now - dist_prev) / 0.02))
    return max(values) if values else 0.0


def _gripper_qpos_closure(sidecar: Mapping[str, Any]) -> float:
    """Gripper qpos indicates closure: 1.0 = closed, 0.0 = wide open."""
    qpos = _finite_vector(sidecar.get("robot0_gripper_qpos"), 2)
    if qpos is None:
        return 0.0
    return 1.0 - _clip((abs(qpos[0]) + abs(qpos[1])) / 0.08)


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


# ── Strict K10 binding ─────────────────────────────────────────────────────

def _resolve_k10_feasible(
    index: int,
    action_states: Sequence[CanonicalActionState],
    k10_labels: Sequence[Mapping[str, Any]] | None,
    k10_label_schema: str | None,
) -> tuple[bool, bool, str | None]:
    """Resolve strict_k10_feasible from an external sealed K10 label stream.

    If k10_labels is provided, use the external binding.
    Otherwise, compute a simplified internal proxy (evaluation-only).
    """
    if k10_labels is not None and k10_label_schema is not None:
        if len(k10_labels) != len(action_states):
            raise ValueError("K10 label stream length mismatch")
        k10_row = k10_labels[index]
        k10_known = bool(k10_row.get("k10_known_mask", False))
        k10_feasible = bool(k10_row.get("k10_feasible", False)) if k10_known else False
        return k10_feasible, k10_known, k10_label_schema

    # Internal fallback — NOT authoritative; external binding preferred
    as_ = action_states[index]
    k10_known = bool(as_.action_known and as_.candidate_close)
    future = action_states[index:min(len(action_states), index + 10)]
    future_close = any(
        s.action_known and s.action_intent == "CLOSE" for s in future[1:]
    )
    return bool(k10_known and future_close), k10_known, "INTERNAL_SIMPLIFIED_V1"


# ═══════════════════════════════════════════════════════════════════════════════
# Main Teacher derivation
# ═══════════════════════════════════════════════════════════════════════════════

def derive_factorized_rows(
    step_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    k10_labels: Sequence[Mapping[str, Any]] | None = None,
    k10_label_schema: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive factorized three-head Teacher labels for one episode.

    Does NOT mutate step_rows or sidecar_rows.
    """

    T = len(step_rows)
    if T != len(sidecar_rows) or T == 0:
        raise ValueError("step and sidecar lengths must match and be nonzero")

    constants = protocol["fixed_constants"]
    history_size = int(protocol["history"]["score_window_steps"])
    thresholds = protocol["head_thresholds"]
    mechanism = _determine_mechanism(role)

    # ── Extract action states (no mutation of input) ────────────────────
    action_states: list[CanonicalActionState] = []
    for step in step_rows:
        action_states.append(CanonicalActionState.from_step(step, field="clean_action_raw_7d"))

    # ── Pre-compute contact and physics evidence ────────────────────────
    contact_flags: list[tuple[bool, bool, bool]] = []
    gripper_contacts: list[bool] = []
    qpos_closures: list[float] = []
    for sc in sidecar_rows:
        cf = _contact_flags(sc.get("mujoco_contact_pairs", []), role)
        contact_flags.append(cf)
        gripper_contacts.append(cf[1])
        qpos_closures.append(_gripper_qpos_closure(sc))

    # ── Per-step computation ────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    # Event state machine
    event_counter = -1
    event_phase = EventPhase.IDLE
    active_object: str | None = None

    for index in range(T):
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
        sv = bool(step_rows[index].get("valid", True))

        # ── Evidence components ──────────────────────────────────────
        opening_trend = _opening_trend(action_states, index, window=3)
        qpos_closed = qpos_closures[index]
        contact_loss = _contact_loss_evidence(gripper_contacts[:index + 1])
        separation = _object_eef_separation(
            sidecar_rows[max(0, index - 3):index + 1], role, object_slices,
        )

        contact_hist = gripper_contacts[max(0, index - history_size + 1):index + 1]
        toggles = sum(l != r for l, r in zip(contact_hist[1:], contact_hist[:-1]))
        toggle_rate = toggles / max(1, len(contact_hist) - 1)

        contact_dwell = 0
        for j in range(index, -1, -1):
            if gripper_contacts[j]:
                contact_dwell += 1
            else:
                break

        has_support_now = contact[2]
        had_support_before = any(contact_flags[j][2] for j in range(max(0, index - 10), index + 1))
        support_removed = 1.0 if had_support_before and not has_support_now else 0.0

        h_transport = _horizontal_transport(sidecar_rows[:index + 1], role, object_slices)

        # ── base_known: label semantics are decidable ─────────────────
        # Separate from positive/negative: base_known=True allows known negatives.
        physics_inputs_valid = (
            math.isfinite(stable)
            and math.isfinite(comotion)
            and math.isfinite(lift)
            and _finite_vector(sidecar_rows[index].get("object_state"), 14) is not None
        )
        route_ok = mechanism.supported and role.applicable
        base_known = bool(
            route_ok
            and sv
            and as_.action_known
            and physics_inputs_valid
        )

        # ── grasp_established ─────────────────────────────────────────
        grasp_known = base_known
        grasp_value = False
        grasp_conf = 0.0
        if grasp_known:
            has_contact = (
                has_grip
                and contact_dwell >= int(thresholds.get("grasp_min_contact_dwell", 3))
            )
            if has_contact:
                grasp_score = _clip(
                    0.30 * stable
                    + 0.20 * qpos_closed
                    + 0.20 * (1.0 - toggle_rate)
                    + 0.15 * (1.0 - contact_loss)
                    + 0.15 * float(has_grip)
                )
                grasp_value = grasp_score >= float(thresholds["grasp_min_score"])
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
            manip_value = manip_score >= float(thresholds["manipulation_min_score"])
            manip_conf = manip_score

        # ── release_or_instability ────────────────────────────────────
        release_known = base_known
        release_value = False
        release_conf = 0.0
        if release_known:
            release_score = _clip(
                0.35 * contact_loss
                + 0.20 * separation
                + 0.15 * opening_trend
                + 0.15 * toggle_rate
                + 0.15 * (1.0 - stable)
            )
            release_value = release_score >= float(thresholds["release_min_score"])
            release_conf = release_score

        # ── strict_k10_feasible ───────────────────────────────────────
        k10_feasible, k10_known, k10_schema = _resolve_k10_feasible(
            index, action_states, k10_labels, k10_label_schema,
        )

        # ── close_event_onset ─────────────────────────────────────────
        prev_grasp = rows[-1]["grasp_established"] if rows else False
        close_onset = bool(as_.candidate_close and grasp_value and not prev_grasp)

        # ── Event state machine ──────────────────────────────────────
        # IDLE → GRASPED on fresh grasp
        # GRASPED → MANIPULATING on manipulation
        # MANIPULATING/GRASPED → RELEASED on release
        # RELEASED → IDLE after release clears

        if event_phase == EventPhase.IDLE:
            if grasp_value:
                event_counter += 1
                event_phase = EventPhase.GRASPED
                active_object = role.manipulated_objects[0] if role.manipulated_objects else None
        elif event_phase == EventPhase.RELEASED:
            if grasp_value:
                event_counter += 1
                event_phase = EventPhase.GRASPED
                active_object = role.manipulated_objects[0] if role.manipulated_objects else None
            elif not release_value:
                event_phase = EventPhase.IDLE
        elif event_phase == EventPhase.GRASPED:
            if release_value:
                event_phase = EventPhase.RELEASED
            elif manip_value:
                event_phase = EventPhase.MANIPULATING
        elif event_phase == EventPhase.MANIPULATING:
            if release_value:
                event_phase = EventPhase.RELEASED

        # Determine event_role and event_id
        if event_phase in (EventPhase.GRASPED, EventPhase.MANIPULATING):
            event_role = EventRole.TARGET.value
            target_relevant = True
            current_event_id = event_counter
        elif event_phase == EventPhase.RELEASED:
            event_role = EventRole.NONE.value
            target_relevant = False
            current_event_id = -1
        else:  # IDLE
            event_role = EventRole.NONE.value
            target_relevant = False
            current_event_id = -1

        rows.append({
            "step": index,
            "mechanism_type": mechanism.value,
            "event_id": current_event_id,
            "event_phase": event_phase.value,
            "event_role": event_role,
            "target_relevant": target_relevant,
            "active_object_name": active_object,
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
            "strict_k10_binding_schema": k10_schema,
            "gripper_contact_score": 1.0 if has_grip else 0.0,
            "relative_pose_stability": stable,
            "object_eef_comotion_score": comotion,
            "lift_score": lift,
            "support_removed": support_removed,
            "target_progress": tp,
            "target_progress_known": tp_known,
            "student_valid": sv,
            "action_intent": as_.action_intent,
            "action_known": as_.action_known,
            "candidate_close": as_.candidate_close,
        })

    # ── Event summary ─────────────────────────────────────────────────────
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
            "event_role": max((r["event_role"] for r in members), key=lambda v: 0 if v == "NONE" else 1),
            "start_step": members[0]["step"],
            "end_step": members[-1]["step"],
            "step_count": len(members),
            "has_grasp": any(r["grasp_established"] for r in members),
            "has_manipulation": any(r["manipulation_active"] for r in members),
            "has_release": any(r["release_or_instability"] for r in members),
            "grasp_steps": sum(1 for r in members if r["grasp_established"]),
            "manipulation_steps": sum(1 for r in members if r["manipulation_active"]),
            "release_steps": sum(1 for r in members if r["release_or_instability"]),
            "active_object_name": members[0].get("active_object_name"),
        })

    return rows, events


# ── Prefix invariance validator ─────────────────────────────────────────────

def verify_prefix_invariance(
    step_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    k10_labels: Sequence[Mapping[str, Any]] | None = None,
    k10_label_schema: str | None = None,
) -> dict[str, Any]:
    """Verify primary heads are prefix-invariant."""

    T = len(step_rows)
    violations = 0
    head_names = ("grasp_established", "manipulation_active", "release_or_instability")

    full_rows, _ = derive_factorized_rows(
        step_rows, sidecar_rows, role, object_slices, protocol,
        k10_labels=k10_labels, k10_label_schema=k10_label_schema,
    )

    for t in range(1, T):
        prefix_rows, _ = derive_factorized_rows(
            step_rows[:t + 1], sidecar_rows[:t + 1], role, object_slices, protocol,
            k10_labels=k10_labels[:t + 1] if k10_labels is not None else None,
            k10_label_schema=k10_label_schema,
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


def verify_deterministic_derive(
    step_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    role: PhysicsTaskRole,
    object_slices: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify two independent runs produce identical output."""
    rows1, ev1 = derive_factorized_rows(step_rows, sidecar_rows, role, object_slices, protocol)
    rows2, ev2 = derive_factorized_rows(step_rows, sidecar_rows, role, object_slices, protocol)

    import json
    identical = (json.dumps(rows1, sort_keys=True) == json.dumps(rows2, sort_keys=True)
                 and json.dumps(ev1, sort_keys=True) == json.dumps(ev2, sort_keys=True))
    return {"deterministic": identical, "step_count": len(step_rows)}


__all__ = [
    "FACTORIZED_TEACHER_FIELDS",
    "FACTORIZED_TEACHER_SCHEMA",
    "FACTORIZED_LABEL_FILENAME",
    "FACTORIZED_MANIFEST_SCHEMA",
    "MechanismRoute",
    "EventRole",
    "EventPhase",
    "_determine_mechanism",
    "derive_factorized_rows",
    "verify_prefix_invariance",
    "verify_deterministic_derive",
]
