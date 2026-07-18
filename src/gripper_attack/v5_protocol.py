"""Small, explicit contracts for Official V5 development.

V5 is a FIT-only development line.  This module contains no attack path and
does not promote a clean-only utility proxy to causal attack evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .b3_formal import B3_FEATURES_25D, B3_POLICY_INTENT_FEATURES_9D


V5_PROTOCOL_SCHEMA = "DETECTOR_V5_DEVELOPMENT_PROTOCOL_V1"
V5_FEATURES_25D = tuple(B3_FEATURES_25D)
V5_FEATURES_9D = tuple(B3_POLICY_INTENT_FEATURES_9D)
V5_VARIANTS = (
    "V5_A_PROPRIO",
    "V5_B_PROPRIO_POLICY_INTENT",
    "V5_C_PROPRIO_CAUSAL_VISUAL",
    "V5_D_PROPRIO_POLICY_INTENT_CAUSAL_VISUAL",
)
V5_PHYSICS_CANDIDATE_ALIASES = {
    "V5_A_PHYSICS": "V5_A_PROPRIO",
    "V5_B_PHYSICS": "V5_B_PROPRIO_POLICY_INTENT",
}
V5_PHASES = (
    "PRE_SUPPORT",
    "VALID_RETENTION",
    "RELEASE_IMMINENT_TAIL",
    "POST_RELEASE",
    "UNSTABLE_TRANSITION",
    "UNKNOWN",
)
V5_TEACHER_FIELDS = frozenset(
    {
        "canonical_parent_key",
        "step",
        "event_id",
        "phase_id",
        "window_id",
        "phase_name",
        "window_start",
        "window_end",
        "candidate_close",
        "quality_valid",
        "veto_invalid",
        "release_imminent",
        "regrasp_or_unstable",
        "known_mask",
        "utility_tier",
        "ranking_group",
    }
)
V5_STUDENT_FORBIDDEN_FIELDS = frozenset(
    {
        "quality_valid",
        "veto_invalid",
        "release_imminent",
        "regrasp_or_unstable",
        "known_mask",
        "utility_tier",
        "ranking_group",
        "contact",
        "object_state",
        "object_pose",
        "task_success",
        "attack_outcome",
    }
)


def json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_order_sha(names: Sequence[str]) -> str:
    return json_sha(list(names))


def is_variant(value: str) -> bool:
    return value in V5_VARIANTS


def canonical_variant(value: str) -> str:
    """Map development candidate aliases to their model contract variant."""

    return V5_PHYSICS_CANDIDATE_ALIASES.get(value, value)


def variant_uses_intent(variant: str) -> bool:
    variant = canonical_variant(variant)
    return variant in {"V5_B_PROPRIO_POLICY_INTENT", "V5_D_PROPRIO_POLICY_INTENT_CAUSAL_VISUAL"}


def variant_uses_visual(variant: str) -> bool:
    variant = canonical_variant(variant)
    return variant in {"V5_C_PROPRIO_CAUSAL_VISUAL", "V5_D_PROPRIO_POLICY_INTENT_CAUSAL_VISUAL"}


def validate_student_features(
    row: Mapping[str, Any], *, expected_proprio_width: int = 25, visual_key: str = "visual_embedding"
) -> None:
    """Reject Teacher/privileged fields at the Student boundary."""

    forbidden = sorted(set(row) & V5_STUDENT_FORBIDDEN_FIELDS)
    if forbidden:
        raise ValueError(f"V5 Student row contains forbidden Teacher fields: {forbidden}")
    proprio = row.get("features_25d")
    if not isinstance(proprio, list) or len(proprio) != expected_proprio_width:
        raise ValueError("V5 Student row must contain exactly 25 proprio features")
    if visual_key in row and row[visual_key] is not None and not isinstance(row[visual_key], list):
        raise ValueError("visual_embedding must be a list when present")


def validate_teacher_row(row: Mapping[str, Any]) -> None:
    missing = sorted(V5_TEACHER_FIELDS - set(row))
    extra = sorted(set(row) - V5_TEACHER_FIELDS)
    if missing or extra:
        raise ValueError(f"invalid V5 Teacher row: missing={missing}, extra={extra}")
    if row["phase_name"] not in V5_PHASES:
        raise ValueError(f"unknown V5 phase: {row['phase_name']}")
    tier = row["utility_tier"]
    if not isinstance(row["known_mask"], bool):
        raise ValueError("known_mask must be bool")
    if row["known_mask"] and (not isinstance(tier, int) or tier not in range(4)):
        raise ValueError("known utility_tier must be an integer in [0, 3]")
    if not row["known_mask"] and tier is not None:
        raise ValueError("unknown V5 rows must not carry a utility tier")
    if row["known_mask"]:
        if bool(row["quality_valid"]) == bool(row["veto_invalid"]):
            raise ValueError("known V5 supervision must satisfy XOR")


@dataclass(frozen=True)
class V5Window:
    episode_id: str
    window_id: str
    start: int
    end: int
    phase_name: str
    utility_tier: int | None
    known: bool
    candidate_close: bool
    step_indices: tuple[int, ...] = ()
    decision_anchor_step: int | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.window_id or self.start < 0 or self.end < self.start:
            raise ValueError("invalid V5 window identity or bounds")
        if self.phase_name not in V5_PHASES:
            raise ValueError("invalid V5 window phase")
        if self.known and (self.utility_tier is None or self.utility_tier not in range(4)):
            raise ValueError("known V5 windows require utility tier 0..3")
        if not self.known and self.utility_tier is not None:
            raise ValueError("unknown V5 windows cannot carry a utility tier")
        indices = tuple(self.step_indices) if self.step_indices else tuple(range(self.start, self.end + 1))
        if indices != tuple(range(self.start, self.end + 1)):
            raise ValueError("V5 window steps must be one contiguous exact segment")
        anchor = self.decision_anchor_step
        if anchor is None:
            anchor = self.start + 9 if len(indices) >= 10 else self.end
        if anchor < self.start or anchor > self.end:
            raise ValueError("V5 decision anchor must lie inside the window")
        object.__setattr__(self, "step_indices", indices)
        object.__setattr__(self, "decision_anchor_step", anchor)

    @property
    def minimum_dwell_met(self) -> bool:
        return len(self.step_indices) >= 10

    @property
    def rankable(self) -> bool:
        return self.known and self.candidate_close and self.phase_name != "UNKNOWN" and not self.window_id.startswith("none:")


@dataclass(frozen=True)
class V5ModelContract:
    variant: str
    visual_dim: int = 0
    hidden_dim: int = 128
    intent_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if not is_variant(self.variant):
            raise ValueError(f"unsupported V5 variant: {self.variant}")
        if self.hidden_dim != 128 or self.intent_hidden_dim != 64:
            raise ValueError("V5 development fixes hidden dimensions at 128 and 64")
        if variant_uses_visual(self.variant) and self.visual_dim <= 0:
            raise ValueError("visual V5 variants require a positive visual_dim")
        if not variant_uses_visual(self.variant) and self.visual_dim != 0:
            raise ValueError("non-visual V5 variants cannot declare visual_dim")
        if variant_uses_intent(self.variant) is False and self.intent_hidden_dim != 64:
            raise ValueError("intent hidden dimension contract mismatch")

    @property
    def proprio_order_sha256(self) -> str:
        return feature_order_sha(V5_FEATURES_25D)

    @property
    def intent_order_sha256(self) -> str:
        return feature_order_sha(V5_FEATURES_9D)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "DETECTOR_V5_MODEL_CONTRACT_V1",
            "variant": self.variant,
            "visual_dim": self.visual_dim,
            "hidden_dim": self.hidden_dim,
            "intent_hidden_dim": self.intent_hidden_dim,
            "proprio_features": list(V5_FEATURES_25D),
            "policy_intent_features": list(V5_FEATURES_9D),
            "proprio_order_sha256": self.proprio_order_sha256,
            "intent_order_sha256": self.intent_order_sha256,
            "student_future_leakage": False,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }


def validate_phase_windows(windows: Sequence[V5Window]) -> None:
    """Check deterministic window bounds without assuming one event = one window."""

    seen: set[tuple[str, str]] = set()
    for window in windows:
        key = (window.episode_id, window.window_id)
        if key in seen:
            raise ValueError(f"duplicate V5 window: {key}")
        seen.add(key)
        if not window.rankable:
            raise ValueError("V5 rankable window collection contains a non-rankable window")


__all__ = [
    "V5_PROTOCOL_SCHEMA",
    "V5_FEATURES_25D",
    "V5_FEATURES_9D",
    "V5_VARIANTS",
    "V5_PHYSICS_CANDIDATE_ALIASES",
    "V5_PHASES",
    "V5_TEACHER_FIELDS",
    "V5_STUDENT_FORBIDDEN_FIELDS",
    "V5Window",
    "V5ModelContract",
    "feature_order_sha",
    "is_variant",
    "canonical_variant",
    "variant_uses_intent",
    "variant_uses_visual",
    "validate_student_features",
    "validate_teacher_row",
    "validate_phase_windows",
    "json_sha",
]
