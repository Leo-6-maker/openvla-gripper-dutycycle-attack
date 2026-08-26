"""AA2R2 versioned action semantics bound to the official OpenVLA-LIBERO transform.

This namespace is intentionally separate from the historical Stage-Z validator.
The only changed behavior is removal of its artificial raw-threshold dead band.
"""

from __future__ import annotations

import math
from typing import Any

try:
    from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper
except ModuleNotFoundError:  # pragma: no cover - supports direct repository imports
    from src.gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper


MODEL_M0 = "M0_OPENVLA"
MODEL_M1 = "M1_OPENVLA_OFT"
MODEL_M2 = "M2_PI05_LIBERO"
OPENVLA_FAMILIES = frozenset({MODEL_M0, MODEL_M1})
MODELS = frozenset({MODEL_M0, MODEL_M1, MODEL_M2})
ACTION_DIM = 7
RAW_OPEN_THRESHOLD = 0.5
FINAL_TOLERANCE = 1e-6
VALIDATOR_VERSION = "STAGE_AA_AA2R2_ACTION_SEMANTICS_V2"
OPENVLA_RULE = "OPENVLA_OFFICIAL_THREE_STATE_EXECUTABLE_TRANSFORM_V2"
PI05_RULE = "PI05_CLIP_RAW_TO_LIBERO_V1"


def _values(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _finite_vector(value: Any) -> tuple[float, ...] | None:
    result = _values(value)
    if result is None or len(result) != ACTION_DIM or not all(math.isfinite(item) for item in result):
        return None
    return result


def _list_or_none(value: tuple[float, ...] | None) -> list[float] | None:
    return None if value is None else list(value)


def _result(
    *,
    model_family: str,
    accepted: bool,
    rule: str,
    reason: str,
    semantic_state: str,
    raw_action: tuple[float, ...] | None = None,
    final_action: tuple[float, ...] | None = None,
    raw_gripper: float | None = None,
    final_gripper: float | None = None,
    expected_final_gripper: float | None = None,
    expected_final_action: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    return {
        "validator_version": VALIDATOR_VERSION,
        "model_family": model_family,
        "accepted": bool(accepted),
        "rule": rule,
        "reason": reason,
        "semantic_state": semantic_state,
        "raw_action": _list_or_none(raw_action),
        "final_action": _list_or_none(final_action),
        "raw_gripper": raw_gripper,
        "final_gripper": final_gripper,
        "expected_final_gripper": expected_final_gripper,
        "expected_final_action": _list_or_none(expected_final_action),
        "final_value_tolerance": FINAL_TOLERANCE,
    }


def _openvla_result(model_family: str, raw_action: Any, final_action: Any, raw_gripper: Any, final_gripper: Any) -> dict[str, Any]:
    raw_values = _finite_vector(raw_action)
    if raw_values is None:
        return _result(
            model_family=model_family,
            accepted=False,
            rule=OPENVLA_RULE,
            reason="MALFORMED_OR_NONFINITE_RAW_ACTION",
            semantic_state="INVALID",
        )
    final_values = _finite_vector(final_action)
    if final_values is None:
        return _result(
            model_family=model_family,
            accepted=False,
            rule=OPENVLA_RULE,
            reason="MALFORMED_OR_NONFINITE_FINAL_ACTION",
            semantic_state="INVALID",
            raw_action=raw_values,
        )
    try:
        raw_value = float(raw_values[-1] if raw_gripper is None else raw_gripper)
        final_value = float(final_values[-1] if final_gripper is None else final_gripper)
    except (TypeError, ValueError):
        return _result(
            model_family=model_family,
            accepted=False,
            rule=OPENVLA_RULE,
            reason="NON_NUMERIC_GRIPPER",
            semantic_state="INVALID",
            raw_action=raw_values,
            final_action=final_values,
        )
    if not math.isfinite(raw_value) or not math.isfinite(final_value):
        return _result(
            model_family=model_family,
            accepted=False,
            rule=OPENVLA_RULE,
            reason="NONFINITE_GRIPPER",
            semantic_state="INVALID",
            raw_action=raw_values,
            final_action=final_values,
            raw_gripper=raw_value,
            final_gripper=final_value,
        )

    expected = float(raw_gripper_to_env_gripper(raw_value, binarize=True))
    semantic_state = "OPEN" if raw_value > RAW_OPEN_THRESHOLD else "CLOSE" if raw_value < RAW_OPEN_THRESHOLD else "NEUTRAL_BOUNDARY"
    if not math.isclose(final_value, expected, rel_tol=0.0, abs_tol=FINAL_TOLERANCE):
        return _result(
            model_family=model_family,
            accepted=False,
            rule=OPENVLA_RULE,
            reason="OPENVLA_GRIPPER_MAPPING_MISMATCH",
            semantic_state=semantic_state,
            raw_action=raw_values,
            final_action=final_values,
            raw_gripper=raw_value,
            final_gripper=final_value,
            expected_final_gripper=expected,
        )
    return _result(
        model_family=model_family,
        accepted=True,
        rule=OPENVLA_RULE,
        reason="NEUTRAL_BOUNDARY" if semantic_state == "NEUTRAL_BOUNDARY" else "OK",
        semantic_state=semantic_state,
        raw_action=raw_values,
        final_action=final_values,
        raw_gripper=raw_value,
        final_gripper=final_value,
        expected_final_gripper=expected,
    )


def _pi05_result(model_family: str, raw_action: Any, final_action: Any) -> dict[str, Any]:
    raw_values = _finite_vector(raw_action)
    if raw_values is None:
        return _result(
            model_family=model_family,
            accepted=False,
            rule=PI05_RULE,
            reason="MALFORMED_OR_NONFINITE_RAW_ACTION",
            semantic_state="INVALID",
        )
    final_values = _finite_vector(final_action)
    if final_values is None:
        return _result(
            model_family=model_family,
            accepted=False,
            rule=PI05_RULE,
            reason="MALFORMED_OR_NONFINITE_FINAL_ACTION",
            semantic_state="INVALID",
            raw_action=raw_values,
        )
    if any(value < -1.0 or value > 1.0 for value in final_values):
        return _result(
            model_family=model_family,
            accepted=False,
            rule=PI05_RULE,
            reason="FINAL_ACTION_OUT_OF_LIBERO_RANGE",
            semantic_state="CONTINUOUS_CLIPPED",
            raw_action=raw_values,
            final_action=final_values,
        )
    expected = tuple(min(1.0, max(-1.0, value)) for value in raw_values)
    if any(not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=FINAL_TOLERANCE) for actual, wanted in zip(final_values, expected)):
        return _result(
            model_family=model_family,
            accepted=False,
            rule=PI05_RULE,
            reason="PI05_CLIP_MAPPING_MISMATCH",
            semantic_state="CONTINUOUS_CLIPPED",
            raw_action=raw_values,
            final_action=final_values,
            expected_final_action=expected,
        )
    return _result(
        model_family=model_family,
        accepted=True,
        rule=PI05_RULE,
        reason="OK",
        semantic_state="CONTINUOUS_CLIPPED",
        raw_action=raw_values,
        final_action=final_values,
        expected_final_action=expected,
    )


def validate_action_pair(
    model_family: str,
    raw_action: Any,
    final_action: Any,
    *,
    raw_gripper: Any = None,
    final_gripper: Any = None,
) -> dict[str, Any]:
    """Validate one 7-D model action pair under the AA2R2 versioned contract."""

    if model_family in OPENVLA_FAMILIES:
        return _openvla_result(model_family, raw_action, final_action, raw_gripper, final_gripper)
    if model_family == MODEL_M2:
        return _pi05_result(model_family, raw_action, final_action)
    return _result(
        model_family=model_family,
        accepted=False,
        rule="UNKNOWN",
        reason="UNKNOWN_MODEL_FAMILY",
        semantic_state="INVALID",
    )


__all__ = [
    "ACTION_DIM",
    "FINAL_TOLERANCE",
    "MODEL_M0",
    "MODEL_M1",
    "MODEL_M2",
    "MODELS",
    "OPENVLA_RULE",
    "PI05_RULE",
    "RAW_OPEN_THRESHOLD",
    "VALIDATOR_VERSION",
    "validate_action_pair",
]
