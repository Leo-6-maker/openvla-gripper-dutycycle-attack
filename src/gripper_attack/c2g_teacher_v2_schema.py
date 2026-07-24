"""Frozen row and candidate schemas for C2g Teacher-v2."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


TEACHER_SCHEMA_VERSION = "c2g.teacher_v2.2026-07-10.v2"
ATTACK_PROTOCOL_NAME = "C2G_CMDOPEN_CAUSAL_REPLAY"
ATTACK_PROTOCOL_VERSION = "2026-07-10.v1"
COMPARISON_TIERS = ("TIER_A_MATCHED_ACTION_SHORT_HORIZON", "TIER_B_CLOSED_LOOP_CONTINUATION")
CAUSAL_LABEL_SOURCES = frozenset({
    "GROUNDING_ONLY",
    "COUNTERFACTUAL_TIER_A",
    "COUNTERFACTUAL_TIER_B",
    "COUNTERFACTUAL_TIER_A_AND_B",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TEACHER_REASON_CODES = frozenset({
    "PRIMARY_TARGET_CARRY", "AUXILIARY_GRASP", "DISTRACTOR_CARRY",
    "NO_CONFIDENT_CONTACT_OBJECT", "TARGET_ID_UNRESOLVED", "PRE_LIFT_GRASP",
    "RELEASE_NEAR_TARGET", "RELEASE_AWAY_FROM_TARGET", "APPROACH_OR_SETUP",
    "CONTACT_LOSS_AFTER_CMDOPEN", "OBJECT_DROP_AFTER_CMDOPEN",
    "PROGRESS_REGRESSION_AFTER_CMDOPEN", "SUCCESS_FLIP_AFTER_CMDOPEN",
    "RELEASE_SAFE_COUNTERFACTUAL", "NO_MATERIAL_HARM_AFTER_CMDOPEN",
    "RESTORE_MISMATCH", "ACTION_ALIGNMENT_FAILED", "AMBIGUOUS_EFFECT",
    "NOT_REPLAYED", "TARGET_GROUNDING_FAILED", "SNAPSHOT_INCOMPLETE",
    "INCOMPLETE_ATTACK_DELIVERY",
})

CANDIDATE_STRATA = (
    "CLOSE_ONSET", "STABLE_GRASP", "PERSISTENT_CONTACT",
    "RELATIVE_OBJECT_MOTION", "STABLE_CARRY", "PRE_RELEASE",
    "RANDOM_NONCANDIDATE_AUDIT",
)

CAUSAL_LABEL_FIELDS = (
    "y_cmdopen_vulnerable", "y_contact_loss", "y_object_drop",
    "y_progress_regression", "y_success_flip", "y_release_safe",
)
AUXILIARY_LABEL_FIELDS = ("y_contact_stable", "y_grounding_confident")
LABEL_FIELDS = (*CAUSAL_LABEL_FIELDS, *AUXILIARY_LABEL_FIELDS)

REQUIRED_ROW_FIELDS = (
    "teacher_schema_version", "teacher_confidence", "teacher_reason_code",
    "teacher_known", "label_known_mask", "causal_label_source",
    "counterfactual_manifest_sha256", "counterfactual_replay_valid",
    "comparison_tier", "attack_protocol_name", "attack_protocol_version",
    "grounding_source", "grounding_confidence", "contacted_objects",
    "resolved_target_objects", "resolved_receptacles", "resolved_sites",
    "target_match", "object_relative_lift", "release_distance",
    "release_safe_evidence", "candidate_stratum", "candidate_reason",
    *LABEL_FIELDS,
)

TEACHER_ONLY_FIELDS = frozenset({
    "contacted_objects", "resolved_target_objects", "resolved_receptacles",
    "resolved_sites", "target_match", "object_relative_lift", "release_distance",
    "release_safe_evidence", "teacher_reason_code", "teacher_confidence",
    "label_known_mask", "causal_label_source", "counterfactual_manifest_sha256",
    "counterfactual_replay_valid", "comparison_tier", *LABEL_FIELDS,
    "object_pose", "target_pose", "object_target_distance", "attack_outcome",
    "post_intervention_state",
})
TEACHER_ONLY_PREFIXES = (
    "teacher_", "y_", "target_pose", "object_pose", "contacted_", "resolved_",
    "counterfactual_", "post_intervention_", "attack_outcome", "release_safe_evidence",
)

REQUIRED_CANDIDATE_FIELDS = (
    "candidate_stratum", "candidate_phase", "candidate_reason",
    "sampling_probability", "deterministic_seed",
    "selection_used_privileged_state", "random_noncandidate_recall_audit",
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


def _validate_counterfactual_binding(row: Mapping[str, Any], known: bool) -> None:
    source = str(row["causal_label_source"])
    manifest_sha = str(row["counterfactual_manifest_sha256"] or "")
    tier = str(row["comparison_tier"] or "")
    replay_valid = row["counterfactual_replay_valid"]
    if type(replay_valid) is not bool:
        raise ValueError("counterfactual_replay_valid must be boolean")
    if source == "GROUNDING_ONLY":
        if manifest_sha or tier or row["attack_protocol_name"] or row["attack_protocol_version"]:
            raise ValueError("GROUNDING_ONLY cannot claim counterfactual provenance")
        if replay_valid or known:
            raise ValueError("GROUNDING_ONLY cannot produce known causal vulnerability labels")
        return
    if not SHA256_RE.fullmatch(manifest_sha):
        raise ValueError("counterfactual labels require a full manifest SHA256")
    expected_tiers = {
        "COUNTERFACTUAL_TIER_A": {COMPARISON_TIERS[0]},
        "COUNTERFACTUAL_TIER_B": {COMPARISON_TIERS[1]},
        "COUNTERFACTUAL_TIER_A_AND_B": set(COMPARISON_TIERS),
    }[source]
    supplied_tiers = {item.strip() for item in tier.split("+") if item.strip()}
    if supplied_tiers != expected_tiers:
        raise ValueError("causal_label_source and comparison_tier mismatch")
    if row["attack_protocol_name"] != ATTACK_PROTOCOL_NAME or row["attack_protocol_version"] != ATTACK_PROTOCOL_VERSION:
        raise ValueError("counterfactual label attack protocol mismatch")
    if known and not replay_valid:
        raise ValueError("known causal label requires valid counterfactual replay")


def validate_teacher_v2_row(row: Mapping[str, Any]) -> None:
    """Fail closed on causal provenance, unknown masking, and reason consistency."""
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
    source = str(row["causal_label_source"])
    if source not in CAUSAL_LABEL_SOURCES:
        raise ValueError("unknown causal_label_source")
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
    _validate_counterfactual_binding(row, known)
    if not known:
        if any(row[field] is not None for field in CAUSAL_LABEL_FIELDS):
            raise ValueError("label_known_mask=0 requires null causal labels, never implicit negatives")
    else:
        if not row["teacher_known"]:
            raise ValueError("known causal row requires teacher_known=true")
        if any(row[field] is None for field in CAUSAL_LABEL_FIELDS):
            raise ValueError("known causal row requires complete explicit causal outcomes")

    harm_fields = ("y_contact_loss", "y_object_drop", "y_progress_regression", "y_success_flip")
    if row["y_cmdopen_vulnerable"] in (1, True):
        if not any(row[field] in (1, True) for field in harm_fields) and not row.get("composite_rule_documented"):
            raise ValueError("vulnerability positive requires a causal harm signal")
    if row["y_release_safe"] in (1, True) and row["y_cmdopen_vulnerable"] in (1, True):
        raise ValueError("release-safe evidence vetoes vulnerability under schema v2")
    if known and row["y_cmdopen_vulnerable"] in (0, False):
        if any(row[field] not in (0, False) for field in harm_fields):
            raise ValueError("known causal negative requires every harm outcome to be explicitly false")
        expected_reason = "RELEASE_SAFE_COUNTERFACTUAL" if row["y_release_safe"] in (1, True) else "NO_MATERIAL_HARM_AFTER_CMDOPEN"
        if reason != expected_reason:
            raise ValueError(f"known causal negative requires reason {expected_reason}")

    replay_failure_reasons = {
        "TARGET_ID_UNRESOLVED", "TARGET_GROUNDING_FAILED", "RESTORE_MISMATCH",
        "SNAPSHOT_INCOMPLETE", "ACTION_ALIGNMENT_FAILED", "NOT_REPLAYED",
        "INCOMPLETE_ATTACK_DELIVERY", "AMBIGUOUS_EFFECT",
    }
    if reason in replay_failure_reasons and known:
        raise ValueError("unresolved/replay failure cannot become a known causal label")
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
    forbidden = sorted({
        name for name in feature_names
        if name in TEACHER_ONLY_FIELDS or any(name.startswith(prefix) for prefix in TEACHER_ONLY_PREFIXES)
    })
    if forbidden:
        raise ValueError("teacher-only fields cannot be student inputs: " + ", ".join(forbidden))
