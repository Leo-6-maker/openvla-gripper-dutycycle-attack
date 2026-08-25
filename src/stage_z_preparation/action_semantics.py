"""Stage-Z-only action-boundary semantics for clean phase classification.

The historical phase classifier has an OpenVLA gripper check.  This module
keeps that check byte-for-byte untouched and supplies the model-family-aware
boundary check needed by the Stage-Z clean-reference runner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


MODEL_M0 = "M0_OPENVLA"
MODEL_M1 = "M1_OPENVLA_OFT"
MODEL_M2 = "M2_PI05_LIBERO"
OPENVLA_FAMILIES = frozenset({MODEL_M0, MODEL_M1})
ACTION_DIM = 7
TOLERANCE = 1e-6
RAW_OPEN_THRESHOLD = 0.5


def _values(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result


def _finite_vector(value: Any) -> tuple[float, ...] | None:
    result = _values(value)
    if result is None or len(result) != ACTION_DIM or not all(math.isfinite(item) for item in result):
        return None
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE)


def _result(
    *,
    model_family: str,
    accepted: bool,
    rule: str,
    reason: str,
    raw_action: tuple[float, ...] | None = None,
    final_action: tuple[float, ...] | None = None,
    expected_final_action: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "accepted": bool(accepted),
        "rule": rule,
        "reason": reason,
        "tolerance": TOLERANCE,
        "raw_action": None if raw_action is None else list(raw_action),
        "final_action": None if final_action is None else list(final_action),
        "expected_final_action": None if expected_final_action is None else list(expected_final_action),
    }


def _openvla_gripper_result(model_family: str, raw: Any, final: Any) -> dict[str, Any]:
    try:
        raw_value = float(raw)
        final_value = float(final)
    except (TypeError, ValueError):
        return _result(
            model_family=model_family,
            accepted=False,
            rule="OPENVLA_THRESHOLD_INVERT_V1",
            reason="NON_NUMERIC_GRIPPER",
        )
    if not math.isfinite(raw_value) or not math.isfinite(final_value):
        return _result(
            model_family=model_family,
            accepted=False,
            rule="OPENVLA_THRESHOLD_INVERT_V1",
            reason="NONFINITE_GRIPPER",
        )
    if _close(raw_value, RAW_OPEN_THRESHOLD):
        return _result(
            model_family=model_family,
            accepted=False,
            rule="OPENVLA_THRESHOLD_INVERT_V1",
            reason="RAW_GRIPPER_AT_THRESHOLD",
        )
    expected = -1.0 if raw_value > RAW_OPEN_THRESHOLD else 1.0
    if not _close(final_value, expected):
        return _result(
            model_family=model_family,
            accepted=False,
            rule="OPENVLA_THRESHOLD_INVERT_V1",
            reason="OPENVLA_GRIPPER_MAPPING_MISMATCH",
        )
    return _result(
        model_family=model_family,
        accepted=True,
        rule="OPENVLA_THRESHOLD_INVERT_V1",
        reason="OK",
    )


def validate_action_pair(
    model_family: str,
    raw_action: Any,
    final_action: Any,
    *,
    raw_gripper: Any = None,
    final_gripper: Any = None,
) -> dict[str, Any]:
    """Validate one model output against its frozen final-action contract.

    M2 accepts finite raw values outside ``[-1, 1]`` only when the final
    action is exactly the official element-wise clip.  M0/M1 retain the old
    scalar threshold/inversion rule; the full vectors are diagnostic only.
    """

    if model_family in OPENVLA_FAMILIES:
        raw_values = _values(raw_action)
        final_values = _values(final_action)
        raw_value = raw_gripper if raw_gripper is not None else (None if raw_values is None or len(raw_values) != ACTION_DIM else raw_values[-1])
        final_value = final_gripper if final_gripper is not None else (None if final_values is None or len(final_values) != ACTION_DIM else final_values[-1])
        return _openvla_gripper_result(model_family, raw_value, final_value)

    if model_family != MODEL_M2:
        return _result(model_family=model_family, accepted=False, rule="UNKNOWN", reason="UNKNOWN_MODEL_FAMILY")

    raw_values = _finite_vector(raw_action)
    final_values = _finite_vector(final_action)
    if raw_values is None:
        return _result(
            model_family=model_family,
            accepted=False,
            rule="PI05_CLIP_RAW_TO_LIBERO_V1",
            reason="MALFORMED_OR_NONFINITE_RAW_ACTION",
        )
    if final_values is None:
        return _result(
            model_family=model_family,
            accepted=False,
            rule="PI05_CLIP_RAW_TO_LIBERO_V1",
            reason="MALFORMED_NONFINITE_OR_OUT_OF_RANGE_FINAL_ACTION",
            raw_action=raw_values,
        )
    expected = tuple(min(1.0, max(-1.0, value)) for value in raw_values)
    if any(not _close(actual, wanted) for actual, wanted in zip(final_values, expected)):
        return _result(
            model_family=model_family,
            accepted=False,
            rule="PI05_CLIP_RAW_TO_LIBERO_V1",
            reason="CLIP_INCONSISTENT_FINAL_ACTION",
            raw_action=raw_values,
            final_action=final_values,
            expected_final_action=expected,
        )
    return _result(
        model_family=model_family,
        accepted=True,
        rule="PI05_CLIP_RAW_TO_LIBERO_V1",
        reason="OK_RAW_IN_RANGE" if raw_values == expected else "OK_RAW_OUT_OF_RANGE_CLIPPED",
        raw_action=raw_values,
        final_action=final_values,
        expected_final_action=expected,
    )


def classify_trajectory_with_action_semantics(
    rows: Sequence[Mapping[str, Any]],
    model_family: str,
    classifier_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run frozen geometry classification with a family-aware boundary gate."""

    checks: list[dict[str, Any]] = []
    for row in rows:
        checks.append(
            validate_action_pair(
                model_family,
                row.get("raw_action_7d"),
                row.get("env_action_7d"),
                raw_gripper=row.get("raw_gripper"),
                final_gripper=row.get("env_gripper"),
            )
        )

    diagnostics = {
        "schema": "STAGE_Z_ACTION_SEMANTICS_DIAGNOSTICS_V1",
        "model_family": model_family,
        "checks": len(checks),
        "accepted": sum(bool(item["accepted"]) for item in checks),
        "invalid": sum(not bool(item["accepted"]) for item in checks),
        "rule": checks[0]["rule"] if checks else None,
    }
    if model_family in OPENVLA_FAMILIES:
        # Preserve the historical classifier behavior exactly for M0/M1.
        return classifier_module.classify_trajectory(rows), diagnostics

    adapted: list[dict[str, Any]] = []
    for row, check in zip(rows, checks):
        current = dict(row)
        if check["accepted"]:
            # The historical classifier only sees a valid placeholder pair;
            # M2 validity remains in the independent diagnostics above.
            current["raw_gripper"] = 0.0
            current["env_gripper"] = 1.0
        else:
            current["clean_record_valid"] = False
        adapted.append(current)
    diagnostics["historical_classifier_gripper_placeholder"] = "raw=0.0,env=1.0_for_valid_M2_rows"
    return classifier_module.classify_trajectory(adapted), diagnostics


__all__ = [
    "MODEL_M0",
    "MODEL_M1",
    "MODEL_M2",
    "classify_trajectory_with_action_semantics",
    "validate_action_pair",
]
