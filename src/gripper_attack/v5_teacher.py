"""Deterministic clean-only V5 utility proxy derivation.

This intentionally produces a proxy ranking target, not a counterfactual
attack label.  It accepts the sealed phase-segmented V2.1.3 Teacher stream and
does not read attack outcomes or protected splits.
"""

from __future__ import annotations

from typing import Any

from .v5_protocol import V5_PHASES, validate_teacher_row


_PHASE_MAP = {
    "approach": "PRE_SUPPORT",
    "grasp_close": "PRE_SUPPORT",
    "stable_grasp": "PRE_SUPPORT",
    "pre_support": "PRE_SUPPORT",
    "pre_support_unsupported": "PRE_SUPPORT",
    "no_close": "UNKNOWN",
    "close_invalid": "PRE_SUPPORT",
    "first_lift": "VALID_RETENTION",
    "stable_carry": "VALID_RETENTION",
    "valid_retention": "VALID_RETENTION",
    "pre_place_unsupported": "RELEASE_IMMINENT_TAIL",
    "release_safe": "RELEASE_IMMINENT_TAIL",
    "release_imminent_tail": "RELEASE_IMMINENT_TAIL",
    "post_release": "POST_RELEASE",
    "recovery_or_regrasp": "UNSTABLE_TRANSITION",
    "unstable_transition": "UNSTABLE_TRANSITION",
    "abstain_unsupported": "UNKNOWN",
    "unknown": "UNKNOWN",
}

HIGH_VALUE_RETENTION_WINDOW_MIN_STEPS = 10


def map_phase(value: Any) -> str:
    normalized = str(value).strip()
    if normalized in V5_PHASES:
        return normalized
    return _PHASE_MAP.get(normalized.lower(), "UNKNOWN")


def derive_utility_tier(row: dict[str, Any], phase_name: str) -> int | None:
    """Map clean Teacher evidence to an ordinal proxy, fail-closed on unknown."""

    known = bool(row.get("known_mask", False))
    candidate = bool(row.get("candidate_close", False))
    quality = bool(row.get("quality_valid", False))
    veto = bool(row.get("veto_invalid", False))
    if not known or quality == veto:
        return None
    window_start = int(row.get("window_start", 0))
    window_end = int(row.get("window_end", window_start))
    window_steps = max(0, window_end - window_start + 1)
    if phase_name == "VALID_RETENTION" and candidate and quality and not veto and window_steps >= HIGH_VALUE_RETENTION_WINDOW_MIN_STEPS:
        # V2.1.3 does not carry a separate remaining-retention duration.  A
        # sealed T10-or-longer valid phase is the highest available clean
        # proxy; this is explicitly not a counterfactual attack label.
        return 3
    if phase_name == "VALID_RETENTION" and candidate and quality and not veto:
        continuation = row.get("retention_continuation_t10", row.get("later_event_known", False))
        return 3 if bool(continuation) and not bool(row.get("release_imminent", False)) else 2
    if phase_name in {"RELEASE_IMMINENT_TAIL", "POST_RELEASE", "UNSTABLE_TRANSITION"}:
        return 0 if not candidate else 1
    if phase_name == "PRE_SUPPORT":
        return 1 if candidate else 0
    return 0


def convert_teacher_row(row: dict[str, Any], identity: str, index: int) -> dict[str, Any]:
    if int(row.get("step", index)) != index:
        raise ValueError(f"{identity}: non-contiguous Teacher step {index}")
    phase_name = map_phase(row.get("phase_name", row.get("phase", "UNKNOWN")))
    event_id = int(row.get("event_id", -1))
    phase_segment = int(row.get("phase_segment_index", -1))
    raw_window = row.get("window_id", -1)
    window_id = f"{event_id}:{phase_segment}" if event_id >= 0 else f"none:{phase_name}"
    if raw_window not in (-1, None, "-1") and event_id >= 0:
        window_id = f"{event_id}:{phase_segment}"
    known = bool(row.get("known_mask", False)) and bool(row.get("quality_valid", False)) != bool(row.get("veto_invalid", False))
    converted = {
        "canonical_parent_key": identity,
        "step": index,
        "event_id": event_id,
        "phase_id": int(row.get("phase_id", -1)),
        "window_id": window_id,
        "phase_name": phase_name,
        "window_start": int(row.get("window_start", index)),
        "window_end": int(row.get("window_end", index)),
        "candidate_close": bool(row.get("candidate_close", False)),
        "quality_valid": bool(row.get("quality_valid", False)),
        "veto_invalid": bool(row.get("veto_invalid", False)),
        "release_imminent": bool(row.get("release_imminent", phase_name == "RELEASE_IMMINENT_TAIL")),
        "regrasp_or_unstable": bool(row.get("regrasp_or_unstable", phase_name == "UNSTABLE_TRANSITION")),
        "known_mask": known,
        "utility_tier": derive_utility_tier(row, phase_name) if known else None,
        "ranking_group": identity,
    }
    validate_teacher_row(converted)
    return converted


__all__ = ["map_phase", "derive_utility_tier", "convert_teacher_row"]
