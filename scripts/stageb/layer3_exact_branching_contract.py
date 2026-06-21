#!/usr/bin/env python3
"""CPU-side contracts for Layer3 exact prefix-once branching.

This module intentionally contains no OpenVLA, LIBERO, CUDA, or attacker
imports.  It defines the audit fields and fail-closed checks that a future
GPU runner must satisfy before VIS/RAND pilot results can be interpreted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

ARM_TOLERANCE = 1e-7
ACTION_DIM = 7
ARM_DIM = 6
LAYER3_BRANCH_CONDITIONS = ("CLEAN_REPLAY", "VIS", "RAND", "SHUFFLED")


class Layer3BranchingContractError(ValueError):
    """Raised when a Layer3 exact-branching invariant is violated."""


def sha256_jsonable(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _float_list(values: Sequence[Any], *, name: str, min_len: int = ACTION_DIM) -> list[float]:
    out = [float(v) for v in values]
    if len(out) < min_len:
        raise Layer3BranchingContractError(f"{name} must have at least {min_len} values; got {len(out)}")
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

    def __post_init__(self) -> None:
        if int(self.emit_step) < 0:
            raise Layer3BranchingContractError("emit_step must be non-negative for exact branching")
        for field in (
            "observation_sha256",
            "sim_state_sha256",
            "policy_rng_sha256",
            "detector_state_sha256",
            "feature_history_sha256",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) < 16:
                raise Layer3BranchingContractError(f"{field} is missing or too short")

    @property
    def snapshot_sha256(self) -> str:
        return sha256_jsonable(asdict(self))


@dataclass(frozen=True)
class BranchRunRecord:
    """Per-condition provenance record for one branch from a saved prefix."""

    condition: str
    prefix_snapshot_sha256: str
    branch_source: str
    restored_sim_state_sha256: str
    restored_observation_sha256: str
    restored_policy_rng_sha256: str
    trigger_step: int
    first_env_step: int

    def __post_init__(self) -> None:
        if self.condition not in LAYER3_BRANCH_CONDITIONS:
            raise Layer3BranchingContractError(f"unexpected condition: {self.condition}")
        if self.branch_source != "EXACT_PREFIX_RESTORE":
            raise Layer3BranchingContractError(
                f"{self.condition} branch_source must be EXACT_PREFIX_RESTORE, got {self.branch_source}"
            )
        if int(self.trigger_step) < 0 or int(self.first_env_step) < 0:
            raise Layer3BranchingContractError("trigger_step and first_env_step must be non-negative")


def make_gripper_only_executed_action(
    clean_env_action: Sequence[Any],
    attacked_env_action: Sequence[Any],
) -> list[float]:
    """Return clean arm with attacked gripper, enforcing 7D action inputs."""

    clean = _float_list(clean_env_action, name="clean_env_action")
    attacked = _float_list(attacked_env_action, name="attacked_env_action")
    executed = list(clean[:ACTION_DIM])
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
    row["arm_preservation_pass"] = bool(row["arm_max_abs_diff"] <= tolerance)
    if not row["arm_preservation_pass"]:
        raise Layer3BranchingContractError(
            f"executed arm differs from clean arm by {row['arm_max_abs_diff']:.9g} > {tolerance}"
        )
    return row


def validate_branch_records(
    snapshot: PrefixBranchSnapshot,
    records: Iterable[BranchRunRecord | Mapping[str, Any]],
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
        if int(rec.trigger_step) != int(snapshot.emit_step):
            raise Layer3BranchingContractError(f"{rec.condition} trigger_step does not match emit_step")
        if int(rec.first_env_step) != int(snapshot.emit_step):
            raise Layer3BranchingContractError(f"{rec.condition} first_env_step does not match emit_step")
        seen[rec.condition] = rec

    missing = [name for name in LAYER3_BRANCH_CONDITIONS if name not in seen]
    if missing:
        raise Layer3BranchingContractError(f"missing branch conditions: {','.join(missing)}")
    return {
        "prefix_snapshot_sha256": expected_sha,
        "condition_count": len(seen),
        "conditions": list(LAYER3_BRANCH_CONDITIONS),
        "exact_prefix_branching_pass": True,
    }

