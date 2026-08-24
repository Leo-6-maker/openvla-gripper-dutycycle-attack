"""Pure-Python branch replay contract for Stage X1R2 Q3R3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


ARMS = ("CLEAN", "TRUE_PGD_T5", "RAND_UNIFORM_T5", "SHUFFLED_GRAD_T5", "TRUE_PGD_RANDOM_TIME_T5")
STATE_FIELDS = ("model_identity", "suite_task_state_identity", "seed_and_dummy_wait", "wrapper_step_index", "qpos", "qvel", "act", "ctrl", "time", "mocap_state", "task_object_state", "controller_state")
EXACT_STATE_FIELDS = STATE_FIELDS[:4]
FLOAT_STATE_FIELDS = STATE_FIELDS[4:]
STATE_ATOL = 1e-12
STATE_RTOL = 0.0


class BranchContractError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceClean:
    initial_state: Any
    dummy_wait_steps: int
    horizon: int
    t_emit: int
    t5: int
    h_phys: int
    env_actions: tuple[tuple[float, ...], ...]
    observations: tuple[bytes, ...]
    student_calls: int = 1

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ReferenceClean":
        if record.get("status") != "PASS_REFERENCE_CLEAN" or not record.get("clean_success"):
            raise BranchContractError("REFERENCE_CLEAN_NOT_VALID")
        emit = record.get("first_emit_step")
        if emit is None:
            raise BranchContractError("REFERENCE_CLEAN_NO_EMIT")
        actions = tuple(tuple(float(value) for value in action) for action in record.get("env_actions", ()))
        observations = tuple(_observation_bytes(value) for value in record.get("observation_bytes", ()))
        result = cls(
            initial_state=record.get("initial_state"),
            dummy_wait_steps=int(record.get("dummy_wait_steps", 0)),
            horizon=int(record["policy_horizon"]),
            t_emit=int(emit),
            t5=int(record.get("t5", 5)),
            h_phys=int(record.get("h_phys", 10)),
            env_actions=actions,
            observations=observations,
            student_calls=int(record.get("student_calls", 1)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.initial_state is None:
            raise BranchContractError("REFERENCE_INITIAL_STATE_REQUIRED")
        if self.student_calls != 1:
            raise BranchContractError("REFERENCE_STUDENT_TIMING_MUST_BE_ONE_SHOT")
        if self.t_emit < 0 or self.t_emit + self.t5 + self.h_phys > self.horizon:
            raise BranchContractError("REFERENCE_EMIT_WINDOW_ILLEGAL")
        if len(self.env_actions) < self.t_emit:
            raise BranchContractError("REFERENCE_ACTION_PREFIX_SHORT")
        if len(self.observations) <= self.t_emit:
            raise BranchContractError("REFERENCE_BRANCH_OBSERVATION_MISSING")


@dataclass(frozen=True)
class BranchReplay:
    reference: ReferenceClean
    arm: str
    branch_step: int | None = None
    random_time: bool = False

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise BranchContractError(f"UNKNOWN_BRANCH_ARM:{self.arm}")
        step = self.reference.t_emit if self.branch_step is None else int(self.branch_step)
        if step < 0 or step + self.reference.t5 + self.reference.h_phys > self.reference.horizon:
            raise BranchContractError("BRANCH_WINDOW_ILLEGAL")
        if step > len(self.reference.env_actions) or step >= len(self.reference.observations):
            raise BranchContractError("BRANCH_PREFIX_OR_OBSERVATION_MISSING")
        object.__setattr__(self, "branch_step", step)

    @property
    def prebranch_actions(self) -> tuple[tuple[float, ...], ...]:
        return self.reference.env_actions[: int(self.branch_step)]

    @property
    def common_first_observation(self) -> bytes:
        return bytes(self.reference.observations[int(self.branch_step)])

    def replay_prefix(self, env_step: Callable[[int, tuple[float, ...]], None]) -> int:
        """Replay only sealed actions before the branch; no model callback exists here."""
        for step, action in enumerate(self.prebranch_actions):
            env_step(step, action)
        return len(self.prebranch_actions)

    def validate_first_decision(self, step: int, observation: bytes) -> None:
        if int(step) != int(self.branch_step):
            raise BranchContractError("FIRST_DECISION_STEP_MISMATCH")
        if bytes(observation) != self.common_first_observation:
            raise BranchContractError("COMMON_FIRST_OBSERVATION_MISMATCH")

    def authorize_attacked_step(self, step: int, structural_gates_passed: bool) -> None:
        if int(step) < int(self.branch_step):
            raise BranchContractError("ATTACKED_STEP_BEFORE_BRANCH")
        if int(step) != int(self.branch_step):
            raise BranchContractError("FIRST_ATTACKED_STEP_MISMATCH")
        if not structural_gates_passed:
            raise BranchContractError("ATTACKED_STEP_BEFORE_STRUCTURAL_GATES")


@dataclass(frozen=True)
class ProtectedCounters:
    model_inference_calls: int = 0
    env_step_calls: int = 0
    pgd_calls: int = 0
    physical_interventions: int = 0
    vphys_reads: int = 0
    protected_reads: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in self.__dict__.items()}

    def assert_zero(self) -> None:
        if any(self.as_dict().values()):
            raise BranchContractError("PROTECTED_COUNTER_NONZERO")


def compare_branch_state(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in STATE_FIELDS if field not in reference or field not in candidate]
    mismatches: list[str] = []
    if not missing:
        mismatches.extend(field for field in EXACT_STATE_FIELDS if reference[field] != candidate[field])
        mismatches.extend(field for field in FLOAT_STATE_FIELDS if not _close_tree(reference[field], candidate[field]))
    return {"equal": not missing and not mismatches, "missing": missing, "mismatches": mismatches, "atol": STATE_ATOL, "rtol": STATE_RTOL}


def _observation_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    raise BranchContractError("REFERENCE_OBSERVATION_BYTES_REQUIRED")


def _close_tree(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=STATE_RTOL, abs_tol=STATE_ATOL)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_close_tree(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and isinstance(right, Sequence) and not isinstance(left, (str, bytes, bytearray)) and not isinstance(right, (str, bytes, bytearray)):
        return len(left) == len(right) and all(_close_tree(a, b) for a, b in zip(left, right))
    return left == right
