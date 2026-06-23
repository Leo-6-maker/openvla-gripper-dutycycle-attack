#!/usr/bin/env python3
"""CPU-side contracts for Layer3 exact prefix-once branching.

This module intentionally contains no OpenVLA, LIBERO, CUDA, or attacker
imports.  It defines the audit fields and fail-closed checks that a future
GPU runner must satisfy before VIS/RAND pilot results can be interpreted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

ARM_TOLERANCE = 1e-7
ACTION_DIM = 7
ARM_DIM = 6
LAYER3_BRANCH_CONDITIONS = ("CLEAN_REPLAY", "VIS", "RAND", "SHUFFLED")
DEFAULT_REQUIRED_PILOT_CONDITIONS = ("CLEAN_REPLAY", "VIS", "RAND")
ALLOWED_BRANCH_SOURCES = ("EXACT_PREFIX_RESTORE", "EXACT_ACTION_PREFIX_REPLAY")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Layer3BranchingContractError(ValueError):
    """Raised when a Layer3 exact-branching invariant is violated."""


def sha256_jsonable(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Layer3BranchingContractError(f"{field} must be a 64-character lowercase hex SHA256")


def _float_list(values: Sequence[Any], *, name: str, exact_len: int = ACTION_DIM) -> list[float]:
    out = [float(v) for v in values]
    if len(out) != exact_len:
        raise Layer3BranchingContractError(f"{name} must have exactly {exact_len} values; got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise Layer3BranchingContractError(f"{name} contains non-finite values")
    return out


@dataclass(frozen=True)
class PrefixBranchSnapshot:
    """Immutable metadata for a single clean prefix saved at Student emit."""

    suite: str
    task_idx: int
    state_id: int
    eval_seed: int
    emit_step: int
    observation_sha256: str
    sim_state_sha256: str
    policy_rng_sha256: str
    detector_state_sha256: str
    feature_history_sha256: str
    source_episode_relpath: str
    snapshot_boundary: str = "PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T"

    def __post_init__(self) -> None:
        if int(self.emit_step) < 0:
            raise Layer3BranchingContractError("emit_step must be non-negative for exact branching")
        if self.snapshot_boundary != "PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T":
            raise Layer3BranchingContractError(
                "snapshot_boundary must be PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T"
            )
        for field in (
            "observation_sha256",
            "sim_state_sha256",
            "policy_rng_sha256",
            "detector_state_sha256",
            "feature_history_sha256",
        ):
            require_sha256(getattr(self, field), field=field)

    @property
    def snapshot_sha256(self) -> str:
        return sha256_jsonable(asdict(self))


def _require_token_tuple(values: Sequence[Any], *, field: str, exact_len: int = ACTION_DIM) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if len(out) != exact_len:
        raise Layer3BranchingContractError(f"{field} must have exactly {exact_len} token ids; got {len(out)}")
    if any(v < 0 for v in out):
        raise Layer3BranchingContractError(f"{field} contains a negative token id")
    return out


@dataclass(frozen=True)
class BranchRunRecord:
    """Per-condition provenance record for one branch from a saved prefix."""

    condition: str
    prefix_snapshot_sha256: str
    branch_source: str
    restored_sim_state_sha256: str
    restored_observation_sha256: str
    restored_policy_rng_sha256: str
    restored_detector_state_sha256: str
    restored_feature_history_sha256: str
    trigger_step: int
    first_env_step: int
    snapshot_boundary: str = "PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T"
    branch_input_source: str = "RECAPTURED_ENV_OBSERVATION"
    branch_policy_input_sha256: str = ""
    diagnostic_recaptured_observation_sha256: str = ""
    prefix_trace_sha256: str = ""
    init_state_sha256: str = ""
    dummy_wait_contract_sha256: str = ""
    prefix_step_count: int = 0
    last_prefix_step: int = -1
    pre_branch_sim_state_sha256: str = ""
    pre_branch_student_state_sha256: str = ""
    pre_branch_feature_history_sha256: str = ""

    def __post_init__(self) -> None:
        if self.condition not in LAYER3_BRANCH_CONDITIONS:
            raise Layer3BranchingContractError(f"unexpected condition: {self.condition}")
        if self.branch_source not in ALLOWED_BRANCH_SOURCES:
            raise Layer3BranchingContractError(
                f"{self.condition} branch_source must be one of {ALLOWED_BRANCH_SOURCES}, got {self.branch_source}"
            )
        if self.snapshot_boundary != "PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T":
            raise Layer3BranchingContractError(
                f"{self.condition} snapshot_boundary must be PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T"
            )
        if int(self.trigger_step) < 0 or int(self.first_env_step) < 0:
            raise Layer3BranchingContractError("trigger_step and first_env_step must be non-negative")
        if self.branch_source == "EXACT_ACTION_PREFIX_REPLAY":
            for field in (
                "prefix_trace_sha256",
                "init_state_sha256",
                "dummy_wait_contract_sha256",
                "pre_branch_sim_state_sha256",
                "pre_branch_student_state_sha256",
                "pre_branch_feature_history_sha256",
            ):
                require_sha256(getattr(self, field), field=field)
            if int(self.prefix_step_count) <= 0:
                raise Layer3BranchingContractError("EXACT_ACTION_PREFIX_REPLAY prefix_step_count must be positive")
            if int(self.last_prefix_step) != int(self.first_env_step) - 1:
                raise Layer3BranchingContractError(
                    "EXACT_ACTION_PREFIX_REPLAY last_prefix_step must be first_env_step - 1"
                )


@dataclass(frozen=True)
class PrefixReplayStep:
    """Per-step provenance for exact action-prefix replay.

    This is a contract object only. Runtime replay artifacts may carry the
    actual arrays alongside these hashes, but the scientific record must retain
    full 64-character hashes and exact seven-token provenance for every prefix
    step.
    """

    step: int
    raw_action_sha256: str
    env_action_sha256: str
    tokens: tuple[int, ...]
    tokens_sha256: str
    observation_sha256: str
    policy_input_sha256: str
    qpos_sha256: str
    qvel_sha256: str
    flat_sim_state_sha256: str
    student_state_sha256: str
    feature_history_sha256: str
    reward: float | None
    done: bool

    def __post_init__(self) -> None:
        if int(self.step) < 0:
            raise Layer3BranchingContractError("PrefixReplayStep.step must be non-negative")
        object.__setattr__(self, "tokens", _require_token_tuple(self.tokens, field="tokens"))
        for field in (
            "raw_action_sha256",
            "env_action_sha256",
            "tokens_sha256",
            "observation_sha256",
            "policy_input_sha256",
            "qpos_sha256",
            "qvel_sha256",
            "flat_sim_state_sha256",
            "student_state_sha256",
            "feature_history_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        expected_tokens_sha = sha256_jsonable(list(self.tokens))
        if self.tokens_sha256 != expected_tokens_sha:
            raise Layer3BranchingContractError("tokens_sha256 must match the exact seven-token sequence")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise Layer3BranchingContractError("reward must be finite when present")


@dataclass(frozen=True)
class ExactActionPrefixReplayPayload:
    """Immutable provenance contract for C3 exact action-prefix replay."""

    protocol_version: str
    parent_key: str
    init_state_sha256: str
    dummy_wait_contract_sha256: str
    prefix_steps: tuple[PrefixReplayStep, ...]
    prefix_step_count: int
    last_prefix_step: int
    branch_step: int
    prefix_trace_sha256: str
    expected_branch_observation_sha256: str
    expected_branch_policy_input_sha256: str
    expected_branch_student_state_sha256: str
    expected_branch_feature_history_sha256: str
    expected_pre_branch_qpos_sha256: str
    expected_pre_branch_qvel_sha256: str
    expected_pre_branch_flat_sim_state_sha256: str

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise Layer3BranchingContractError("protocol_version is required")
        if not self.parent_key or self.parent_key.count("|") != 4:
            raise Layer3BranchingContractError("parent_key must be canonical suite|task|state|seed|condition")
        object.__setattr__(self, "prefix_steps", tuple(self.prefix_steps))
        if int(self.prefix_step_count) != len(self.prefix_steps):
            raise Layer3BranchingContractError("prefix_step_count must equal len(prefix_steps)")
        if int(self.prefix_step_count) <= 0:
            raise Layer3BranchingContractError("prefix_step_count must be positive")
        if int(self.last_prefix_step) != int(self.branch_step) - 1:
            raise Layer3BranchingContractError("last_prefix_step must be branch_step - 1")
        expected_steps = tuple(range(self.prefix_step_count))
        actual_steps = tuple(int(step.step) for step in self.prefix_steps)
        if actual_steps != expected_steps:
            raise Layer3BranchingContractError(
                f"prefix_steps must be contiguous from 0; got {actual_steps[:5]}...{actual_steps[-5:]}"
            )
        if int(self.last_prefix_step) != actual_steps[-1]:
            raise Layer3BranchingContractError("last_prefix_step must equal the final PrefixReplayStep.step")
        for field in (
            "init_state_sha256",
            "dummy_wait_contract_sha256",
            "prefix_trace_sha256",
            "expected_branch_observation_sha256",
            "expected_branch_policy_input_sha256",
            "expected_branch_student_state_sha256",
            "expected_branch_feature_history_sha256",
            "expected_pre_branch_qpos_sha256",
            "expected_pre_branch_qvel_sha256",
            "expected_pre_branch_flat_sim_state_sha256",
        ):
            require_sha256(getattr(self, field), field=field)

    @property
    def payload_sha256(self) -> str:
        return sha256_jsonable(asdict(self))


def make_gripper_only_executed_action(
    clean_env_action: Sequence[Any],
    attacked_env_action: Sequence[Any],
) -> list[float]:
    """Return clean arm with attacked gripper, enforcing 7D action inputs."""

    clean = _float_list(clean_env_action, name="clean_env_action")
    attacked = _float_list(attacked_env_action, name="attacked_env_action")
    executed = list(clean)
    executed[-1] = attacked[-1]
    return executed


def arm_preservation_telemetry(
    *,
    step: int,
    condition: str,
    clean_action: Sequence[Any],
    attacked_decoded_action: Sequence[Any],
    executed_action: Sequence[Any],
    tolerance: float = ARM_TOLERANCE,
) -> dict[str, Any]:
    """Build the required per-step arm-preservation telemetry row.

    The formal Layer3 runner must write these fields for every executed step.
    The check is on the actually executed arm, not just source code intent.
    """

    clean = _float_list(clean_action, name="clean_action")
    attacked = _float_list(attacked_decoded_action, name="attacked_decoded_action")
    executed = _float_list(executed_action, name="executed_action")
    if condition not in LAYER3_BRANCH_CONDITIONS and condition not in {"CLEAN", "TRUE", "TRUE_PGD"}:
        raise Layer3BranchingContractError(f"unexpected condition: {condition}")

    row: dict[str, Any] = {"step": int(step), "condition": condition}
    diffs = []
    for idx in range(ARM_DIM):
        clean_v = float(clean[idx])
        attacked_v = float(attacked[idx])
        executed_v = float(executed[idx])
        diff = abs(executed_v - clean_v)
        diffs.append(diff)
        row[f"clean_arm_{idx}"] = clean_v
        row[f"attacked_decoded_arm_{idx}"] = attacked_v
        row[f"executed_arm_{idx}"] = executed_v
        row[f"arm_abs_diff_{idx}"] = diff
    row["arm_max_abs_diff"] = max(diffs) if diffs else 0.0
    row["clean_gripper"] = float(clean[-1])
    row["attacked_gripper"] = float(attacked[-1])
    row["executed_gripper"] = float(executed[-1])
    if condition in {"CLEAN", "CLEAN_REPLAY"}:
        expected_gripper = row["clean_gripper"]
        expected_source = "clean"
    else:
        expected_gripper = row["attacked_gripper"]
        expected_source = "attacked"
    row["executed_gripper_expected_source"] = expected_source
    row["gripper_abs_diff"] = abs(row["executed_gripper"] - expected_gripper)
    row["arm_preservation_pass"] = bool(row["arm_max_abs_diff"] <= tolerance)
    if not row["arm_preservation_pass"]:
        raise Layer3BranchingContractError(
            f"executed arm differs from clean arm by {row['arm_max_abs_diff']:.9g} > {tolerance}"
        )
    if row["gripper_abs_diff"] > tolerance:
        raise Layer3BranchingContractError(
            f"{condition} executed gripper differs from {expected_source} gripper by "
            f"{row['gripper_abs_diff']:.9g} > {tolerance}"
        )
    return row


def validate_branch_records(
    snapshot: PrefixBranchSnapshot,
    records: Iterable[BranchRunRecord | Mapping[str, Any]],
    *,
    required_conditions: Sequence[str] = DEFAULT_REQUIRED_PILOT_CONDITIONS,
) -> dict[str, Any]:
    """Validate that all conditions branch from exactly the same prefix."""

    expected_sha = snapshot.snapshot_sha256
    seen: dict[str, BranchRunRecord] = {}
    for raw in records:
        rec = raw if isinstance(raw, BranchRunRecord) else BranchRunRecord(**dict(raw))
        if rec.condition in seen:
            raise Layer3BranchingContractError(f"duplicate branch condition: {rec.condition}")
        if rec.prefix_snapshot_sha256 != expected_sha:
            raise Layer3BranchingContractError(f"{rec.condition} does not use the frozen prefix snapshot")
        if rec.restored_sim_state_sha256 != snapshot.sim_state_sha256:
            raise Layer3BranchingContractError(f"{rec.condition} restored sim_state hash mismatch")
        if rec.restored_observation_sha256 != snapshot.observation_sha256:
            raise Layer3BranchingContractError(f"{rec.condition} restored observation hash mismatch")
        if rec.restored_policy_rng_sha256 != snapshot.policy_rng_sha256:
            raise Layer3BranchingContractError(f"{rec.condition} restored policy RNG hash mismatch")
        if rec.restored_detector_state_sha256 != snapshot.detector_state_sha256:
            raise Layer3BranchingContractError(f"{rec.condition} restored detector state hash mismatch")
        if rec.restored_feature_history_sha256 != snapshot.feature_history_sha256:
            raise Layer3BranchingContractError(f"{rec.condition} restored feature history hash mismatch")
        if int(rec.trigger_step) != int(snapshot.emit_step):
            raise Layer3BranchingContractError(f"{rec.condition} trigger_step does not match emit_step")
        if int(rec.first_env_step) != int(snapshot.emit_step):
            raise Layer3BranchingContractError(f"{rec.condition} first_env_step does not match emit_step")
        if rec.branch_input_source == "CAPTURED_PREFIX_OBSERVATION":
            if rec.branch_policy_input_sha256 != snapshot.observation_sha256:
                raise Layer3BranchingContractError(f"{rec.condition} branch policy input hash mismatch")
            if rec.restored_observation_sha256 != snapshot.observation_sha256:
                raise Layer3BranchingContractError(
                    f"{rec.condition} restored_observation_sha256 must record captured prefix input"
                )
        seen[rec.condition] = rec

    invalid_required = [name for name in required_conditions if name not in LAYER3_BRANCH_CONDITIONS]
    if invalid_required:
        raise Layer3BranchingContractError(f"invalid required conditions: {','.join(invalid_required)}")
    missing = [name for name in required_conditions if name not in seen]
    if missing:
        raise Layer3BranchingContractError(f"missing branch conditions: {','.join(missing)}")
    return {
        "prefix_snapshot_sha256": expected_sha,
        "condition_count": len(seen),
        "required_conditions": list(required_conditions),
        "conditions": sorted(seen),
        "snapshot_boundary": snapshot.snapshot_boundary,
        "exact_prefix_branching_pass": True,
    }

