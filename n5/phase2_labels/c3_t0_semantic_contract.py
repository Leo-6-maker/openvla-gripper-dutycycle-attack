"""C3-T0 synthetic five-head semantic contract.

This module deliberately accepts only physical/causal evidence records.  It
does not load episodes, models, Teacher artifacts, or outcome metadata.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"
HEADS = (
    "physical_criticality",
    "k10_feasible",
    "safe_release",
    "instability",
    "gripper_closing_state",
)
FORBIDDEN_TOKENS = {
    "task_success", "task_terminal", "terminal", "reward", "outcome",
    "attack", "future", "teacher", "episode_success",
}


class ContractError(ValueError):
    pass


def _reject_forbidden(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_TOKENS:
                raise ContractError(f"forbidden field: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def label(value: str, reason: str) -> dict[str, Any]:
    if value not in (TRUE, FALSE, UNKNOWN):
        raise ContractError(f"invalid truth value: {value}")
    return {"value": value, "mask": value != UNKNOWN, "reason": reason}


def _tri(value: Any) -> str | None:
    if value is None:
        return None
    if value is True or value == TRUE:
        return TRUE
    if value is False or value == FALSE:
        return FALSE
    raise ContractError(f"expected tri-state boolean, got {value!r}")


def physical_criticality(record: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden(record)
    known = record.get("physical_known")
    stable = _tri(record.get("stable_grasp"))
    transport = _tri(record.get("transport_or_manipulation"))
    if known is not True or stable is None or transport is None:
        return label(UNKNOWN, "PHYSICAL_EVIDENCE_UNKNOWN")
    return label(
        TRUE if TRUE in (stable, transport) else FALSE,
        "PHYSICAL_EVIDENCE_PRESENT",
    )


def safe_release(record: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden(record)
    values = [_tri(record.get(name)) for name in ("placement", "release", "stability")]
    if any(value is None for value in values):
        return label(UNKNOWN, "SAFE_RELEASE_COMPONENT_UNKNOWN")
    return label(
        TRUE if all(value == TRUE for value in values) else FALSE,
        "PLACEMENT_RELEASE_STABILITY_CONJUNCTION",
    )


def k10_feasible(record: Mapping[str, Any], horizon: int = 10) -> dict[str, Any]:
    _reject_forbidden(record)
    if record.get("right_censored") is True or record.get("horizon_known") is not True:
        return label(UNKNOWN, "RIGHT_CENSORED_OR_HORIZON_UNKNOWN")
    safe = _tri(record.get("safe_release"))
    remaining = record.get("remaining_steps")
    if safe is None or not isinstance(remaining, int) or isinstance(remaining, bool):
        return label(UNKNOWN, "K10_EVIDENCE_UNKNOWN")
    return label(
        TRUE if remaining >= horizon and safe == FALSE else FALSE,
        "K10_HORIZON_AND_RELEASE_CHECK",
    )


def instability(record: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden(record)
    values = [_tri(record.get(name)) for name in ("slip", "regrasp", "contact_loss")]
    if any(value is None for value in values):
        return label(UNKNOWN, "INSTABILITY_EVIDENCE_UNKNOWN")
    return label(
        TRUE if TRUE in values else FALSE,
        "PHYSICAL_INSTABILITY_EVIDENCE",
    )


def gripper_closing_state(record: Mapping[str, Any]) -> dict[str, Any]:
    _reject_forbidden(record)
    qpos = record.get("gripper_qpos")
    threshold = record.get("qpos_close_threshold")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value)) for value in (qpos, threshold)):
        return label(UNKNOWN, "PHYSICAL_QPOS_UNKNOWN")
    return label(
        TRUE if float(qpos) <= float(threshold) else FALSE,
        "PHYSICAL_QPOS_THRESHOLD",
    )


def apply_persistence(labels: Sequence[Mapping[str, Any]], min_steps: int = 2) -> list[dict[str, Any]]:
    if min_steps < 1:
        raise ContractError("min_steps must be positive")
    out: list[dict[str, Any]] = []
    streak = 0
    for item in labels:
        value = item.get("value")
        if value == TRUE:
            streak += 1
            out.append(label(TRUE, "PERSISTENCE_CONFIRMED") if streak >= min_steps
                       else label(UNKNOWN, "PERSISTENCE_NOT_MET"))
        else:
            streak = 0
            out.append(label(value if value in (FALSE, UNKNOWN) else UNKNOWN,
                             item.get("reason", "PERSISTENCE_RESET")))
    return out


def apply_right_censor(item: Mapping[str, Any], remaining_steps: int | None,
                       required_steps: int = 10) -> dict[str, Any]:
    if remaining_steps is None or remaining_steps < required_steps:
        return label(UNKNOWN, "RIGHT_CENSORED")
    return label(item.get("value", UNKNOWN), item.get("reason", "UNCENSORED"))


def quaternion_equivalent(q1: Iterable[float], q2: Iterable[float], tol: float = 1e-9) -> bool:
    a, b = tuple(float(x) for x in q1), tuple(float(x) for x in q2)
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(x) for x in a + b):
        return False
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return False
    dot = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return abs(abs(dot) - 1.0) <= tol


def evaluate_heads(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _reject_forbidden(record)
    return {
        "physical_criticality": physical_criticality(record),
        "k10_feasible": k10_feasible(record),
        "safe_release": safe_release(record),
        "instability": instability(record),
        "gripper_closing_state": gripper_closing_state(record),
    }
