"""Frozen row and candidate schemas for C2g Teacher-v2."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


TEACHER_SCHEMA_VERSION = "c2g.teacher_v2.2026-07-10.v1"

TEACHER_REASON_CODES = frozenset({
    "PRIMARY_TARGET_CARRY",
    "AUXILIARY_GRASP",
    "DISTRACTOR_CARRY",
    "NO_CONFIDENT_CONTACT_OBJECT",
    "TARGET_ID_UNRESOLVED",
    "PRE_LIFT_GRASP",
    "RELEASE_NEAR_TARGET",
    "RELEASE_AWAY_FROM_TARGET",
    "APPROACH_OR_SETUP",
    "CONTACT_LOSS_AFTER_CMDOPEN",
    "OBJECT_DROP_AFTER_CMDOPEN",
    "PROGRESS_REGRESSION_AFTER_CMDOPEN",
    "SUCCESS_FLIP_AFTER_CMDOPEN",
    "RELEASE_SAFE_COUNTERFACTUAL",
    "RESTORE_MISMATCH",
    "ACTION_ALIGNMENT_FAILED",
    "AMBIGUOUS_EFFECT",
    "NOT_REPLAYED",
    "TARGET_GROUNDING_FAILED",
    "SNAPSHOT_INCOMPLETE",
})

CANDIDATE_STRATA = (
    "CLOSE_ONSET",
    "STABLE_GRASP",
    "PERSISTENT_CONTACT",
    "RELATIVE_OBJECT_MOTION",
    "STABLE_CARRY",
    "PRE_RELEASE",
    "RANDOM_NONCANDIDATE_AUDIT",
)

LABEL_FIELDS = (
    "y_cmdopen_vulnerable",
    "y_contact_loss",
    "y_object_drop",
    "y_progress_regression",
    "y_success_flip",
    "y_release_safe",
    "y_contact_stable",
    "y_grounding_confident",
)

REQUIRED_ROW_FIELDS = (
    "teacher_schema_version",
    "teacher_confidence",
    "teacher_reason_code",
    "teacher_known",
    "label_known_mask",
    "grounding_source",
    "grounding_confidence",
    "contacted_objects",
    "resolved_target_objects",
    "resolved_receptacles",
    "resolved_sites",
    "target_match",
    "object_relative_lift",
    "release_distance",
    "release_safe_evidence",
    "candidate_stratum",
    "candidate_reason",
    *LABEL_FIELDS,
)

TEACHER_ONLY_FIELDS = frozenset({
    "contacted_objects",
    "resolved_target_objects",
    "resolved_receptacles",
    "resolved_sites",
    "target_match",
    "object_relative_lift",
    "release_distance",
    "release_safe_evidence",
    "teacher_reason_code",
    "teacher_confidence",
    "label_known_mask",
    *LABEL_FIELDS,
    "object_pose",
    "target_pose",
    "object_target_distance",
    "attack_outcome",
    "post_intervention_state",
})

REQUIRED_CANDIDATE_FIELDS = (
    "candidate_stratum",
    "candidate_phase",
    "candidate_reason",
    "sampling_probability",
    "deterministic_seed",
    "selection_used_privileged_state",
    "random_noncandidate_recall_audit",
)


def _binary_or_null(value: Any) -> bool:
    return value is None or type(value) is bool or value in (0, 1)


def _require_fields(row: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError("missing required fields: " + ", ".join(missing))


def _as_names(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list or tuple")
    names = tuple(str(item) for item in value if str(item))
    if len(names) != len(value):
        raise ValueError(f"{field} contains an empty identity")
    return names


def validate_teacher_v2_row(row: Mapping[str, Any]) -> None:
    """Fail closed on unknown masking, causal semantics, and reason consistency."""
    _require_fields(row, REQUIRED_ROW_FIELDS)
    if row["teacher_schema_version"] != TEACHER_SCHEMA_VERSION:
        raise ValueError("teacher_schema_version mismatch")
    confidence = float(row["teacher_confidence"])
    grounding_confidence = float(row["grounding_confidence"])
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("teacher_confidence must be finite and in [0,1]")
    if not math.isfinite(grounding_confidence) or not 0.0 <= grounding_confidence <= 1.0:
        raise ValueError("grounding_confidence must be finite and in [0,1]")
    reason = str(row["teacher_reason_code"])
    if reason not in TEACHER_REASON_CODES:
        raise ValueError(f"unknown teacher_reason_code: {reason}")
    if row["candidate_stratum"] not in CANDIDATE_STRATA:
        raise ValueError("unknown candidate_stratum")
    if not str(row["candidate_reason"]).strip() or not str(row["grounding_source"]).strip():
        raise ValueError("candidate_reason and grounding_source are required")
    if row["target_match"] not in (None, False, True):
        raise ValueError("target_match must be boolean or null")
    for field in ("object_relative_lift", "release_distance"):
        if row[field] is not None and not math.isfinite(float(row[field])):
            raise ValueError(f"{field} must be finite or null")
    if type(row["teacher_known"]) is not bool or row["label_known_mask"] not in (0, 1, False, True):
        raise ValueError("teacher_known and label_known_mask must be explicit booleans")
    for field in LABEL_FIELDS:
        if not _binary_or_null(row[field]):
            raise ValueError(f"{field} must be binary or null")
    contacted = set(_as_names(row["contacted_objects"], "contacted_objects"))
    targets = set(_as_names(row["resolved_target_objects"], "resolved_target_objects"))
    _as_names(row["resolved_receptacles"], "resolved_receptacles")
    _as_names(row["resolved_sites"], "resolved_sites")

    known = bool(row["label_known_mask"])
    if not known:
        if row["teacher_known"]:
            raise ValueError("unknown label cannot set teacher_known=true")
        if any(row[field] is not None for field in LABEL_FIELDS):
            raise ValueError("label_known_mask=0 requires null labels, never implicit negatives")
    else:
        if not row["teacher_known"] or row["y_cmdopen_vulnerable"] is None:
            raise ValueError("known row requires teacher_known and vulnerability label")

    harm_fields = ("y_contact_loss", "y_object_drop", "y_progress_regression", "y_success_flip")
    if row["y_cmdopen_vulnerable"] in (1, True):
        if not any(row[field] in (1, True) for field in harm_fields) and not row.get("composite_rule_documented"):
            raise ValueError("vulnerability positive requires a causal harm signal")
    if row["y_release_safe"] in (1, True) and row["y_cmdopen_vulnerable"] in (1, True):
        raise ValueError("release-safe evidence vetoes vulnerability under schema v1")
    if reason in {"TARGET_ID_UNRESOLVED", "TARGET_GROUNDING_FAILED", "RESTORE_MISMATCH", "SNAPSHOT_INCOMPLETE", "ACTION_ALIGNMENT_FAILED", "NOT_REPLAYED"}:
        if known:
            raise ValueError("unresolved/replay failure cannot become a known negative or positive")
    if reason == "PRIMARY_TARGET_CARRY":
        if row["target_match"] is not True or not (contacted & targets):
            raise ValueError("PRIMARY_TARGET_CARRY requires contacted target match")
    if reason in {"AUXILIARY_GRASP", "DISTRACTOR_CARRY"} and row["target_match"] is True:
        raise ValueError(f"{reason} cannot have target_match=true")
    if reason == "NO_CONFIDENT_CONTACT_OBJECT" and contacted:
        raise ValueError("NO_CONFIDENT_CONTACT_OBJECT cannot contain contacted objects")
    if reason == "TARGET_ID_UNRESOLVED" and targets:
        raise ValueError("TARGET_ID_UNRESOLVED cannot contain resolved targets")
    if reason == "RELEASE_NEAR_TARGET" and not row["release_safe_evidence"]:
        raise ValueError("RELEASE_NEAR_TARGET requires release-safe evidence")
    if reason == "RELEASE_AWAY_FROM_TARGET" and row["release_safe_evidence"]:
        raise ValueError("RELEASE_AWAY_FROM_TARGET conflicts with release-safe evidence")
    required_effect = {
        "CONTACT_LOSS_AFTER_CMDOPEN": "y_contact_loss",
        "OBJECT_DROP_AFTER_CMDOPEN": "y_object_drop",
        "PROGRESS_REGRESSION_AFTER_CMDOPEN": "y_progress_regression",
        "SUCCESS_FLIP_AFTER_CMDOPEN": "y_success_flip",
        "RELEASE_SAFE_COUNTERFACTUAL": "y_release_safe",
    }.get(reason)
    if required_effect and row[required_effect] not in (1, True):
        raise ValueError(f"{reason} requires {required_effect}=1")


def validate_candidate_manifest_row(row: Mapping[str, Any]) -> None:
    _require_fields(row, REQUIRED_CANDIDATE_FIELDS)
    stratum = str(row["candidate_stratum"])
    if stratum not in CANDIDATE_STRATA:
        raise ValueError("unknown candidate_stratum")
    probability = float(row["sampling_probability"])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("sampling_probability must be finite and in [0,1]")
    if type(row["deterministic_seed"]) is not int or row["deterministic_seed"] < 0:
        raise ValueError("deterministic_seed must be a non-negative integer")
    for field in ("selection_used_privileged_state", "random_noncandidate_recall_audit"):
        if type(row[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    is_random = stratum == "RANDOM_NONCANDIDATE_AUDIT"
    if bool(row["random_noncandidate_recall_audit"]) != is_random:
        raise ValueError("random noncandidate audit flag must match candidate stratum")
    if not str(row["candidate_reason"]).strip():
        raise ValueError("candidate_reason is required")
    if not str(row["candidate_phase"]).strip():
        raise ValueError("candidate_phase is required")


def assert_student_feature_names(feature_names: Sequence[str]) -> None:
    forbidden = sorted(set(feature_names) & TEACHER_ONLY_FIELDS)
    if forbidden:
        raise ValueError("teacher-only fields cannot be student inputs: " + ", ".join(forbidden))
