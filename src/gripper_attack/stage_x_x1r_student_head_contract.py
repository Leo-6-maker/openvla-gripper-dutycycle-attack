"""Frozen Stage-X Student output-key contract.

The historical semantic label ``k10_feasibility`` is metadata only.  The
tracked frozen Python model emits the actual key ``k10_feasible``.
"""

from __future__ import annotations

from typing import Iterable, Any


RUNTIME_HEAD_NAMES = (
    "physical_criticality",
    "k10_feasible",
    "safe_release",
    "instability",
    "gripper_closing_state",
)

HISTORICAL_SEMANTIC_ALIASES = {
    "k10_feasibility": "k10_feasible",
}

EMIT_HEAD_NAMES = (
    "physical_criticality",
    "gripper_closing_state",
)


def validate_runtime_head_names(names: Iterable[str]) -> tuple[str, ...]:
    actual = tuple(str(name) for name in names)
    if actual != RUNTIME_HEAD_NAMES:
        raise RuntimeError(f"STUDENT_RUNTIME_HEAD_CONTRACT_MISMATCH:{actual!r}")
    return actual


def runtime_head_names(model: Any) -> tuple[str, ...]:
    return validate_runtime_head_names(getattr(model, "HEAD_NAMES", ()))
