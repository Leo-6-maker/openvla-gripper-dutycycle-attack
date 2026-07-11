"""Fail-closed schema for clean-only C2g gripper-critical window labels.

This schema intentionally excludes all attacked-rollout, counterfactual-outcome, and
post-intervention fields from the primary detector path. Privileged simulator state
is permitted only in the offline teacher that emits these rows.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CLEAN_TEACHER_SCHEMA_VERSION = "c2g.clean_teacher.2026-07-10.v1"

MECHANISM_TYPES = frozenset(
    {
        "pick_place_transfer",
        "multi_object_transfer",
        "articulated_object",
        "constrained_manipulation",
        "planar_or_rearrangement",
        "unsupported_or_unknown",
    }
)

TEACHER_PHASES = frozenset(
    {
        "APPROACH",
        "CLOSE_ONSET",
        "TARGET_CONTACT",
        "STABLE_GRASP",
        "LIFT_ONSET",
        "TRANSPORT",
        "CONSTRAINED_MANIPULATION",
        "PRE_RELEASE",
        "RELEASE_SAFE",
        "DISTRACTOR_CONTACT",
        "TARGET_UNRESOLVED",
        "CONTACT_UNRESOLVED",
        "UNSUPPORTED_MECHANISM",
        "OTHER",
    }
)

TEACHER_REASON_CODES = frozenset(
    {
        "TARGET_UNRESOLVED",
        "CONTACT_UNRESOLVED",
        "CLOSE_SEMANTICS_UNRESOLVED",
        "PROGRESS_SEMANTICS_UNRESOLVED",
        "RELEASE_SEMANTICS_UNRESOLVED",
        "UNSUPPORTED_MECHANISM",
        "APPROACH_NO_CONTACT",
        "DISTRACTOR_CONTACT",
        "TARGET_CONTACT_NO_STABLE_GRASP",
        "TARGET_STABLE_GRASP",
        "TARGET_LIFT_ACTIVE",
        "TARGET_TRANSPORT_ACTIVE",
        "TARGET_CONSTRAINED_MANIPULATION",
        "TARGET_PRE_RELEASE",
        "TARGET_RELEASE_SAFE",
        "TARGET_CRITICAL_WINDOW",
        "TARGET_CRITICAL_WINDOW_START",
    }
)

LABEL_FIELDS = (
    "y_target_relevant",
    "y_contact_or_grasp_stable",
    "y_gripper_dependency",
    "y_clean_close_intent",
    "y_lift_transport_or_constraint",
    "y_release_safe",
    "y_gripper_critical_window",
    "y_burst_feasible",
    "y_attack_start_b",
)

REQUIRED_ROW_FIELDS = (
    "teacher_schema_version",
    "episode_key",
    "step",
    "suite",
    "task_index",
    "mechanism_type",
    "mechanism_eligible",
    "teacher_phase",
    "teacher_reason_code",
    "teacher_confidence",
    "grounding_confidence",
    "teacher_known",
    "label_known_mask",
    "resolved_target_objects",
    "resolved_target_manipulable_entities",
    "contacted_entities",
    "uses_privileged_sim_state",
    "uses_attack_outcome",
    "uses_future_student_input",
    *LABEL_FIELDS,
)

TEACHER_ONLY_FIELDS = frozenset(
    {
        "teacher_phase",
        "teacher_reason_code",
        "teacher_confidence",
        "label_known_mask",
        "resolved_target_objects",
        "resolved_target_manipulable_entities",
        "contacted_entities",
        "object_pose",
        "target_pose",
        "object_target_distance",
        "object_eef_distance",
        "contact_pairs",
        "mujoco_contact_pairs",
        "active_target_entity",
        "active_target_known",
        "active_target_bilateral_contact",
        "active_target_contact",
        "active_target_reason",
        "active_subgoal_index",
        "active_operator",
        "active_destination_entity",
        "active_interaction_site",
        "contacted_goal_targets",
        "bilateral_goal_targets",
        "per_target_contact_reason",
        "goal_event_bindings",
        "fixture_joint_motion",
        "object_relative_lift",
        "target_distance_decrease",
        "release_safe",
        "attack_outcome",
        "post_intervention_state",
        *LABEL_FIELDS,
    }
)
TEACHER_ONLY_PREFIXES = (
    "teacher_",
    "y_",
    "resolved_target_",
    "contacted_",
    "bilateral_goal_",
    "per_target_",
    "active_target_",
    "active_subgoal_",
    "active_destination_",
    "active_interaction_",
    "goal_event_",
    "privileged_",
    "object_pose",
    "target_pose",
    "attack_outcome",
    "post_intervention_",
)
ATTACK_OR_OUTCOME_TOKENS = (
    "attack",
    "counterfactual",
    "post_intervention",
    "vis_success",
    "random_success",
    "task_failure_after",
    "qpos_delta_after",
    "open_count_after",
)


def _require_fields(row: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError("missing required clean-teacher fields: " + ", ".join(missing))


def _binary_or_null(value: Any) -> bool:
    return value is None or type(value) is bool or value in (0, 1)


def _name_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list or tuple")
    names = tuple(str(item).strip() for item in value)
    if any(not name for name in names):
        raise ValueError(f"{field} contains an empty identity")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate identities")
    return names


def validate_clean_teacher_row(row: Mapping[str, Any]) -> None:
    """Validate one clean-only teacher row and reject hidden outcome leakage."""

    _require_fields(row, REQUIRED_ROW_FIELDS)
    if row["teacher_schema_version"] != CLEAN_TEACHER_SCHEMA_VERSION:
        raise ValueError("teacher_schema_version mismatch")
    if not str(row["episode_key"]).strip():
        raise ValueError("episode_key is required")
    if type(row["step"]) is not int or int(row["step"]) < 0:
        raise ValueError("step must be a non-negative integer")
    if type(row["task_index"]) is not int or int(row["task_index"]) < 0:
        raise ValueError("task_index must be a non-negative integer")
    if str(row["mechanism_type"]) not in MECHANISM_TYPES:
        raise ValueError("unknown mechanism_type")
    if type(row["mechanism_eligible"]) is not bool:
        raise ValueError("mechanism_eligible must be boolean")
    if str(row["teacher_phase"]) not in TEACHER_PHASES:
        raise ValueError("unknown teacher_phase")
    reason = str(row["teacher_reason_code"])
    if reason not in TEACHER_REASON_CODES:
        raise ValueError("unknown teacher_reason_code")

    for field in ("teacher_confidence", "grounding_confidence"):
        value = float(row[field])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be finite and in [0,1]")

    if type(row["teacher_known"]) is not bool or type(row["label_known_mask"]) is not bool:
        raise ValueError("teacher_known and label_known_mask must be explicit booleans")
    for field in ("uses_privileged_sim_state", "uses_attack_outcome", "uses_future_student_input"):
        if type(row[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    if row["uses_attack_outcome"]:
        raise ValueError("clean teacher cannot use attacked-rollout outcomes")
    if row["uses_future_student_input"]:
        raise ValueError("future clean steps may define teacher boundaries but cannot be student inputs")
    if not row["uses_privileged_sim_state"]:
        raise ValueError("clean teacher rows must disclose privileged simulator-state use")

    targets = set(_name_tuple(row["resolved_target_objects"], "resolved_target_objects"))
    target_manipulable = set(
        _name_tuple(row["resolved_target_manipulable_entities"], "resolved_target_manipulable_entities")
    )
    contacts = set(_name_tuple(row["contacted_entities"], "contacted_entities"))

    for field in LABEL_FIELDS:
        if not _binary_or_null(row[field]):
            raise ValueError(f"{field} must be binary or null")

    known = bool(row["label_known_mask"])
    if known:
        if not row["teacher_known"]:
            raise ValueError("known clean-teacher row requires teacher_known=true")
        if any(row[field] is None for field in LABEL_FIELDS):
            raise ValueError("known clean-teacher row requires complete explicit labels")
    else:
        if row["teacher_known"]:
            raise ValueError("unknown row cannot claim teacher_known=true")
        if any(row[field] is not None for field in LABEL_FIELDS):
            raise ValueError("label_known_mask=false requires null labels, never implicit negatives")

    if not row["mechanism_eligible"] and known:
        raise ValueError("unsupported mechanisms must remain unknown/abstain in the primary label space")

    if known:
        target_relevant = bool(row["y_target_relevant"])
        contact_stable = bool(row["y_contact_or_grasp_stable"])
        gripper_dependency = bool(row["y_gripper_dependency"])
        close_intent = bool(row["y_clean_close_intent"])
        progress = bool(row["y_lift_transport_or_constraint"])
        release_safe = bool(row["y_release_safe"])
        critical = bool(row["y_gripper_critical_window"])
        burst_feasible = bool(row["y_burst_feasible"])
        attack_start = bool(row["y_attack_start_b"])
        expected_critical = (
            target_relevant
            and gripper_dependency
            and close_intent
            and progress
            and not release_safe
        )
        if critical != expected_critical:
            raise ValueError("critical-window label must equal the frozen clean-only conjunction")
        if contact_stable and not contacts:
            raise ValueError("stable contact requires at least one contacted entity")
        if target_relevant and not ((targets | target_manipulable) & contacts):
            raise ValueError("target-relevant contact requires contacted target identity")
        if release_safe and critical:
            raise ValueError("release-safe evidence vetoes a critical-window positive")
        if burst_feasible and not critical:
            raise ValueError("burst-feasible requires critical-window positive")
        if attack_start and not burst_feasible:
            raise ValueError("attack-start requires burst-feasible positive")

    if reason == "TARGET_UNRESOLVED" and (targets or target_manipulable or known):
        raise ValueError("TARGET_UNRESOLVED must have no target identities and remain unknown")
    if reason == "CONTACT_UNRESOLVED" and known:
        raise ValueError("CONTACT_UNRESOLVED must remain unknown")
    if reason == "UNSUPPORTED_MECHANISM" and (row["mechanism_eligible"] or known):
        raise ValueError("UNSUPPORTED_MECHANISM must abstain")
    if reason == "DISTRACTOR_CONTACT" and known:
        if row["y_target_relevant"] not in (0, False) or not contacts:
            raise ValueError("DISTRACTOR_CONTACT requires a known non-target contact")
    if reason == "TARGET_RELEASE_SAFE" and known and row["y_release_safe"] not in (1, True):
        raise ValueError("TARGET_RELEASE_SAFE requires release-safe=true")
    if reason == "TARGET_CRITICAL_WINDOW_START" and known and row["y_attack_start_b"] not in (1, True):
        raise ValueError("TARGET_CRITICAL_WINDOW_START requires y_attack_start_b=true")
    if reason == "TARGET_CRITICAL_WINDOW" and known and row["y_gripper_critical_window"] not in (1, True):
        raise ValueError("TARGET_CRITICAL_WINDOW requires critical-window=true")

    leakage = sorted(
        key
        for key in row
        if key not in {"uses_attack_outcome", *LABEL_FIELDS}
        and any(token in key.lower() for token in ATTACK_OR_OUTCOME_TOKENS)
    )
    if leakage:
        raise ValueError("attacked-rollout/outcome fields are forbidden in clean teacher rows: " + ", ".join(leakage))


def assert_clean_student_feature_names(feature_names: Sequence[str]) -> None:
    """Fail closed if a proposed student input leaks teacher or attack information."""

    forbidden = sorted(
        {
            name
            for name in feature_names
            if name in TEACHER_ONLY_FIELDS
            or any(name.startswith(prefix) for prefix in TEACHER_ONLY_PREFIXES)
            or any(token in name.lower() for token in ATTACK_OR_OUTCOME_TOKENS)
            or name in {"task_index", "task_id", "suite_id", "task_hash", "normalized_step"}
        }
    )
    if forbidden:
        raise ValueError("forbidden primary student features: " + ", ".join(forbidden))
