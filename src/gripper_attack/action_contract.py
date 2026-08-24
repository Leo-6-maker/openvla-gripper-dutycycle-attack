"""Canonical action-space contract (Gate D2.0).

OpenVLA raw action: [0, 1], 0=CLOSE, 1=OPEN.
LIBERO env action:  [-1, 1], -1=OPEN, +1=CLOSE.
Postprocess: env = -sign(2*raw - 1).

Single source of truth for all Teacher, Student, S1, and runtime code.
No module may implement its own close/open threshold independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class GripperIntent(Enum):
    CLOSE = "CLOSE"
    OPEN = "OPEN"
    BOUNDARY = "BOUNDARY"


# Threshold for raw action space. Must be < (not <=) because raw=0.5
# maps to env=0 which is neither close nor open. See close_threshold_raw.
# This is NOT a tunable parameter — it follows from the OpenVLA action
# space definition where 0=close, 1=open.
RAW_CLOSE_REGION_MAX = 0.5
RAW_CLOSE_EPSILON = 1e-6


def classify_openvla_raw_gripper(raw: float, eps: float = RAW_CLOSE_EPSILON) -> GripperIntent:
    """Classify OpenVLA raw action gripper value.

    raw in [0, 1]: 0 = CLOSE, 1 = OPEN.
    raw == 0.5 (within eps): BOUNDARY (maps to env=0, neither close nor open).
    raw < 0.5: CLOSE.
    raw > 0.5: OPEN.
    """
    if abs(float(raw) - RAW_CLOSE_REGION_MAX) <= float(eps):
        return GripperIntent.BOUNDARY
    if float(raw) < RAW_CLOSE_REGION_MAX:
        return GripperIntent.CLOSE
    return GripperIntent.OPEN


def raw_gripper_is_close(raw: float) -> bool:
    """True when raw action gripper indicates close intent (raw < 0.5)."""
    return classify_openvla_raw_gripper(raw) == GripperIntent.CLOSE


def postprocess_gripper_openvla_to_libero(raw: float) -> float:
    """Official OpenVLA → LIBERO gripper postprocess.

    env = -sign(2*raw - 1)
    raw=0.0 → env=+1.0 (CLOSE)
    raw=0.5 → env= 0.0 (BOUNDARY)
    raw=1.0 → env=-1.0 (OPEN)
    """
    return float(-np.sign(2.0 * float(raw) - 1.0))


def classify_libero_env_gripper(env: float) -> GripperIntent:
    """Classify LIBERO env action gripper value.

    env in [-1, 1]: +1 = CLOSE, -1 = OPEN.
    env == 0: BOUNDARY.
    env > 0: CLOSE.
    env < 0: OPEN.
    """
    if abs(float(env)) <= RAW_CLOSE_EPSILON:
        return GripperIntent.BOUNDARY
    if float(env) > 0.0:
        return GripperIntent.CLOSE
    return GripperIntent.OPEN


def env_gripper_is_close(env: float) -> bool:
    """True when env action gripper indicates close (env > 0)."""
    return classify_libero_env_gripper(env) == GripperIntent.CLOSE


def action_semantics_parity(raw: float, env: float) -> bool:
    """Raw and env agree on close/open (False only at boundary)."""
    raw_i = classify_openvla_raw_gripper(raw)
    env_i = classify_libero_env_gripper(env)
    if raw_i == GripperIntent.BOUNDARY or env_i == GripperIntent.BOUNDARY:
        return False
    # After official postprocess, raw close ↔ env close always
    return (raw_i == env_i)


def validate_raw_action(record: dict[str, Any], field: str = "clean_action_raw_7d") -> tuple[float, GripperIntent]:
    """Extract and classify raw gripper from a step record. Fail-closed."""
    raw_action = record.get(field)
    if not isinstance(raw_action, (list, tuple, np.ndarray)) or len(raw_action) < 7:
        raise KeyError("MISSING_RAW_ACTION_FIELD:{}".format(field))
    raw = float(raw_action[6])
    if not (-0.1 <= raw <= 1.1) or not np.isfinite(raw):
        raise ValueError("INVALID_RAW_GRIPPER_VALUE:{}".format(raw))
    return raw, classify_openvla_raw_gripper(raw)


@dataclass(frozen=True)
class CanonicalActionState:
    """Structured action state for a single step. Never defaults to CLOSE/OPEN."""
    raw_gripper: float | None       # None if field missing or non-finite
    action_intent: str               # CLOSE / OPEN / BOUNDARY / UNKNOWN
    action_known: bool               # False for BOUNDARY, missing, or NaN
    candidate_close: bool            # True ONLY for known CLOSE

    @staticmethod
    def from_step(step: dict[str, Any], field: str = "clean_action_raw_7d"):
        try:
            raw, intent = validate_raw_action(step, field=field)
        except (KeyError, ValueError):
            return CanonicalActionState(
                raw_gripper=None,
                action_intent="UNKNOWN",
                action_known=False,
                candidate_close=False,
            )
        return CanonicalActionState(
            raw_gripper=raw,
            action_intent=intent.value,
            action_known=intent is not GripperIntent.BOUNDARY,
            candidate_close=intent is GripperIntent.CLOSE,
        )


ACTION_CONTRACT_SCHEMA = "CANONICAL_ACTION_CONTRACT_V1"
V21C_PROTOCOL_SCHEMA = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL"
V21C_LABEL_FILENAME = "physics_teacher_v21c.jsonl"
V21C_MANIFEST_SCHEMA = "DETECTOR_V5_PHYSICS_TEACHER_V21C_MANIFEST"

ACTION_CONTRACT = {
    "schema": ACTION_CONTRACT_SCHEMA,
    "raw_action_space": "OPENVLA_RAW",
    "raw_close_region": [0.0, RAW_CLOSE_REGION_MAX],
    "raw_close_operator": "<",
    "raw_close_threshold": RAW_CLOSE_REGION_MAX,
    "low_region": "CLOSE",
    "high_region": "OPEN",
    "boundary_value": 0.5,
    "boundary_policy": "ABSTAIN",
    "env_action_space": "LIBERO",
    "env_close_region": [0.0, 1.0],
    "env_close_operator": ">",
    "postprocess": "env = -sign(2*raw - 1)",
    "source_field": "clean_action_raw_7d[6]",
    "fallback_field": "action_raw[6]",
}


__all__ = [
    "ACTION_CONTRACT",
    "ACTION_CONTRACT_SCHEMA",
    "CanonicalActionState",
    "GripperIntent",
    "V21C_LABEL_FILENAME",
    "V21C_MANIFEST_SCHEMA",
    "V21C_PROTOCOL_SCHEMA",
    "action_semantics_parity",
    "classify_libero_env_gripper",
    "classify_openvla_raw_gripper",
    "env_gripper_is_close",
    "postprocess_gripper_openvla_to_libero",
    "raw_gripper_is_close",
    "validate_raw_action",
]
