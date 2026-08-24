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
PROTOCOL_HORIZONS = {
    "libero_10": 520,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_spatial": 220,
}
FORBIDDEN_TOKENS = {
    "task_success", "task_terminal", "terminal", "terminal_state",
    "reward", "outcome", "attack", "future", "teacher", "episode_success",
    "episode_summary", "policy_action", "action", "command", "close_intent",
}

HEAD_INPUT_ALLOWLIST = {
    "physical_criticality": frozenset({"physical_known", "stable_grasp", "transport_or_manipulation"}),
    "safe_release": frozenset({"placement", "released_state", "placement_stability"}),
    "k10_feasible": frozenset({"protocol_steps_remaining", "safe_release_computed"}),
    "instability": frozenset({"slip", "regrasp", "contact_loss"}),
    "gripper_closing_state": frozenset({"gripper_qpos", "qpos_close_threshold"}),
}


class ContractError(ValueError):
    pass


def protocol_horizon_for_suite(suite: Any) -> int | None:
    return PROTOCOL_HORIZONS.get(suite) if isinstance(suite, str) else None


def protocol_steps_remaining(suite: Any, step: Any) -> int | None:
    horizon = protocol_horizon_for_suite(suite)
    if horizon is None or not isinstance(step, int) or isinstance(step, bool):
        return None
    if step < 0 or step >= horizon:
        return None
    return horizon - step - 1


def _reject_forbidden(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_TOKENS:
                raise ContractError(f"forbidden field: {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _project(value: Mapping[str, Any], head: str) -> dict[str, Any]:
    _reject_forbidden(value)
    allowed = HEAD_INPUT_ALLOWLIST[head]
    return {key: value[key] for key in allowed if key in value}


def label(value: str, reason: str) -> dict[str, Any]:
    if value not in (TRUE, FALSE, UNKNOWN):
        raise ContractError(f"invalid truth value: {value}")
    return {"value": value, "mask": value != UNKNOWN, "reason": reason}


def _tri(value: Any) -> str | None:
    if value is None or value == UNKNOWN:
        return None
    if value is True or value == TRUE:
        return TRUE
    if value is False or value == FALSE:
        return FALSE
    raise ContractError(f"expected tri-state boolean, got {value!r}")


def aggregate_tri_conjunction(values: Iterable[Any]) -> str:
    """Three-valued AND: FALSE dominates UNKNOWN, which dominates TRUE."""
    normalized = [_tri(value) for value in values]
    if any(value == FALSE for value in normalized):
        return FALSE
    if not normalized or any(value is None for value in normalized):
        return UNKNOWN
    return TRUE


def aggregate_tri_disjunction(values: Iterable[Any]) -> str:
    """Three-valued OR: TRUE dominates UNKNOWN, which dominates FALSE."""
    normalized = [_tri(value) for value in values]
    if any(value == TRUE for value in normalized):
        return TRUE
    if not normalized or any(value is None for value in normalized):
        return UNKNOWN
    return FALSE


def physical_criticality(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _project(record, "physical_criticality")
    known = record.get("physical_known")
    stable = _tri(record.get("stable_grasp"))
    transport = _tri(record.get("transport_or_manipulation"))
    if known is not True or stable is None or transport is None:
        return label(UNKNOWN, "PHYSICAL_EVIDENCE_UNKNOWN")
    return label(
        aggregate_tri_disjunction((stable, transport)),
        "PHYSICAL_EVIDENCE_PRESENT",
    )


def safe_release(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _project(record, "safe_release")
    values = [_tri(record.get(name)) for name in ("placement", "released_state", "placement_stability")]
    aggregate = aggregate_tri_conjunction(values)
    if aggregate == UNKNOWN:
        return label(UNKNOWN, "SAFE_RELEASE_COMPONENT_UNKNOWN")
    return label(
        aggregate,
        "PLACEMENT_RELEASED_STATE_PLACEMENT_STABILITY_CONJUNCTION",
    )


def k10_feasible(record: Mapping[str, Any], horizon: int = 10) -> dict[str, Any]:
    record = _project(record, "k10_feasible")
    safe_record = record.get("safe_release_computed")
    safe = _tri(safe_record.get("value")) if isinstance(safe_record, Mapping) else _tri(safe_record)
    remaining = record.get("protocol_steps_remaining")
    if remaining is None:
        return label(UNKNOWN, "RIGHT_CENSORED_PROTOCOL_STEPS_REMAINING")
    if safe is None or not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        return label(UNKNOWN, "K10_EVIDENCE_UNKNOWN")
    return label(
        TRUE if remaining >= horizon and safe == FALSE else FALSE,
        "K10_HORIZON_AND_RELEASE_CHECK",
    )


def instability(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _project(record, "instability")
    values = [_tri(record.get(name)) for name in ("slip", "regrasp", "contact_loss")]
    aggregate = aggregate_tri_disjunction(values)
    if aggregate == UNKNOWN:
        return label(UNKNOWN, "INSTABILITY_EVIDENCE_UNKNOWN")
    return label(
        aggregate,
        "PHYSICAL_INSTABILITY_EVIDENCE",
    )


def gripper_closing_state(record: Mapping[str, Any]) -> dict[str, Any]:
    record = _project(record, "gripper_closing_state")
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


def apply_right_censor(item: Mapping[str, Any], observed_future_steps_available: Any,
                       required_steps: int = 10) -> dict[str, Any]:
    if (not isinstance(observed_future_steps_available, int)
            or isinstance(observed_future_steps_available, bool)
            or observed_future_steps_available < required_steps):
        return label(UNKNOWN, "RIGHT_CENSORED_OBSERVED_FUTURE_STEPS")
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
    safe = safe_release(record)
    result = {
        "physical_criticality": physical_criticality(record),
        "k10_feasible": k10_feasible({"protocol_steps_remaining": record.get("protocol_steps_remaining"), "safe_release_computed": safe}),
        "safe_release": safe,
        "instability": instability(record),
        "gripper_closing_state": gripper_closing_state(record),
    }
    if result["safe_release"]["value"] == TRUE and result["k10_feasible"]["value"] == TRUE:
        raise ContractError("cross-head invariant violated: safe_release TRUE with k10 TRUE")
    if result["k10_feasible"]["value"] == TRUE and result["safe_release"]["value"] != FALSE:
        raise ContractError("cross-head invariant violated: K10 requires safe_release FALSE")
    return result
