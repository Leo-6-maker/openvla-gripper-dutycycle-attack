"""Architecture-independent Stage-Z execution contract.

The contract is intentionally pure Python.  A future runtime may provide
model-specific callbacks, but it must expose the final 7-D LIBERO action here
before the gripper-only intervention.  No callback is invoked by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


ACTION_DIM = 7
ARM_INDICES = tuple(range(6))
GRIPPER_INDEX = 6
LIBERO_OPEN = -1.0
LIBERO_CLOSE = 1.0
DOSES = (3, 5, 10)
Z0R2_PASS = "STAGE_Z_Z0R2_THREE_MODEL_AUTHORITY_CLOSURE_PASS"
FROZEN_PARENT_COUNT = 36


class StageZHold(ValueError):
    """Fail-closed error for an absent, stale, or invalid Stage-Z authority."""


def _as_finite_action(action: Iterable[float], *, label: str) -> tuple[float, ...]:
    if isinstance(action, (str, bytes, bytearray)):
        raise StageZHold(f"{label}_MUST_BE_NUMERIC_SEQUENCE")
    try:
        values = tuple(float(value) for value in action)
    except (TypeError, ValueError) as exc:
        raise StageZHold(f"{label}_MUST_BE_NUMERIC_SEQUENCE") from exc
    if len(values) != ACTION_DIM:
        raise StageZHold(f"{label}_DIMENSION_{len(values)}_EXPECTED_{ACTION_DIM}")
    if not all(math.isfinite(value) for value in values):
        raise StageZHold(f"{label}_NONFINITE")
    if not all(-1.0 <= value <= 1.0 for value in values):
        raise StageZHold(f"{label}_OUTSIDE_LIBERO_RANGE")
    return values


def validate_final_action(action: Iterable[float], *, label: str = "FINAL_ACTION") -> tuple[float, ...]:
    """Validate the authoritative final LIBERO action without changing it."""

    return _as_finite_action(action, label=label)


def assert_arm_preserved(reference: Sequence[float], candidate: Sequence[float]) -> None:
    reference_values = _as_finite_action(reference, label="REFERENCE_ACTION")
    candidate_values = _as_finite_action(candidate, label="CANDIDATE_ACTION")
    if candidate_values[:6] != reference_values[:6]:
        raise StageZHold("ARM_COORDINATES_CHANGED")


def intervene_gripper_open(action: Iterable[float], *, duration: int) -> tuple[float, ...]:
    """Replace only final gripper coordinate with the sealed LIBERO OPEN value."""

    if int(duration) not in DOSES:
        raise StageZHold(f"UNFROZEN_OPEN_DURATION:{duration}")
    source = validate_final_action(action)
    intervened = source[:GRIPPER_INDEX] + (LIBERO_OPEN,)
    assert_arm_preserved(source, intervened)
    if intervened[GRIPPER_INDEX] != LIBERO_OPEN:
        raise StageZHold("OPEN_VALUE_NOT_NATIVE_LIBERO_OPEN")
    return intervened


@dataclass(frozen=True)
class FinalLiberoAction:
    """Model-produced final action at a model decision boundary."""

    values: tuple[float, ...]
    model_id: str
    authority_id: str
    boundary_kind: str
    residual_action_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", validate_final_action(self.values))
        if not self.model_id:
            raise StageZHold("MODEL_ID_REQUIRED")
        if not self.authority_id:
            raise StageZHold("ACTION_AUTHORITY_REQUIRED")
        if int(self.residual_action_count) < 0:
            raise StageZHold("NEGATIVE_RESIDUAL_ACTION_COUNT")


@dataclass(frozen=True)
class ProtectedCounters:
    """Counters that must remain zero during static preparation and at entry."""

    model_inference: int = 0
    simulator: int = 0
    env_step: int = 0
    physical_intervention: int = 0
    v_phys: int = 0
    pgd: int = 0
    protected_reads: int = 0
    eval160_reads: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in self.__dict__.items()}

    def assert_zero(self) -> None:
        nonzero = {name: value for name, value in self.as_dict().items() if value != 0}
        if nonzero:
            raise StageZHold(f"PROTECTED_COUNTERS_NONZERO:{nonzero}")


@dataclass(frozen=True)
class ExecutionAuthorization:
    """All authorities required before a real Stage-Z callback may run.

    The default is intentionally disabled.  ``None`` is never interpreted as
    an implicit pass for any authority field.
    """

    execution_enabled: bool = False
    z0r2_status: str | None = None
    root_seal_sha256: str | None = None
    expected_root_seal_sha256: str | None = None
    model_authority_sha256: str | None = None
    expected_model_authority_sha256: str | None = None
    common_libero_sha256: str | None = None
    expected_common_libero_sha256: str | None = None
    panel_sha256: str | None = None
    expected_panel_sha256: str | None = None
    frozen_parent_keys: frozenset[str] = field(default_factory=frozenset)
    phase: str | None = None
    authorized_phases: frozenset[str] = field(default_factory=frozenset)
    counters: ProtectedCounters = field(default_factory=ProtectedCounters)

    def validate(self, *, parent_key: str | None = None, phase: str | None = None) -> None:
        if not self.execution_enabled:
            raise StageZHold("EXECUTION_DISABLED")
        if self.z0r2_status != Z0R2_PASS:
            raise StageZHold("Z0R2_AUTHORITY_NOT_PASS")
        required_pairs = (
            ("root_seal_sha256", self.root_seal_sha256, self.expected_root_seal_sha256),
            ("model_authority_sha256", self.model_authority_sha256, self.expected_model_authority_sha256),
            ("common_libero_sha256", self.common_libero_sha256, self.expected_common_libero_sha256),
            ("panel_sha256", self.panel_sha256, self.expected_panel_sha256),
        )
        for name, actual, expected in required_pairs:
            if not actual or not expected or actual != expected:
                raise StageZHold(f"{name.upper()}_MISMATCH_OR_MISSING")
        if parent_key is None or parent_key not in self.frozen_parent_keys:
            raise StageZHold("PARENT_NOT_IN_FROZEN_PANEL")
        requested_phase = phase or self.phase
        if not requested_phase or requested_phase not in self.authorized_phases:
            raise StageZHold("PHASE_NOT_EXPLICITLY_AUTHORIZED")
        self.counters.assert_zero()


def require_execution_authorized(
    authorization: ExecutionAuthorization,
    *,
    parent_key: str,
    phase: str,
) -> None:
    """Guard a future real execution entry point before any callback is called."""

    if not isinstance(authorization, ExecutionAuthorization):
        raise StageZHold("EXECUTION_AUTHORIZATION_OBJECT_REQUIRED")
    authorization.validate(parent_key=parent_key, phase=phase)


def assert_no_forbidden_counters(counters: Mapping[str, Any]) -> None:
    """Validate external counter mappings at a receipt boundary."""

    forbidden = {
        "model_inference",
        "simulator",
        "env_step",
        "physical_intervention",
        "v_phys",
        "pgd",
        "protected_reads",
        "eval160_reads",
    }
    nonzero = {key: counters.get(key) for key in forbidden if int(counters.get(key, 0)) != 0}
    if nonzero:
        raise StageZHold(f"FORBIDDEN_COUNTERS_NONZERO:{nonzero}")


__all__ = [
    "ACTION_DIM",
    "ARM_INDICES",
    "DOSES",
    "ExecutionAuthorization",
    "FinalLiberoAction",
    "FROZEN_PARENT_COUNT",
    "GRIPPER_INDEX",
    "LIBERO_CLOSE",
    "LIBERO_OPEN",
    "ProtectedCounters",
    "StageZHold",
    "Z0R2_PASS",
    "assert_arm_preserved",
    "assert_no_forbidden_counters",
    "intervene_gripper_open",
    "require_execution_authorized",
    "validate_final_action",
]
