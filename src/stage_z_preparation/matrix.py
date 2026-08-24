"""Static five-arm Stage-Z branch matrix preparation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .anchors import AnchorSelection
from .contract import FinalLiberoAction, StageZHold, intervene_gripper_open, validate_final_action


class StageZArm(str, Enum):
    CLEAN_BRANCH_CRITICAL = "CLEAN_BRANCH_CRITICAL"
    COMMAND_OPEN_T3_CRITICAL = "COMMAND_OPEN_T3_CRITICAL"
    COMMAND_OPEN_T5_CRITICAL = "COMMAND_OPEN_T5_CRITICAL"
    COMMAND_OPEN_T10_CRITICAL = "COMMAND_OPEN_T10_CRITICAL"
    COMMAND_OPEN_T5_NONCRITICAL_CONTROL = "COMMAND_OPEN_T5_NONCRITICAL_CONTROL"


@dataclass(frozen=True)
class BranchSnapshot:
    parent_key: str
    state_hash: str
    observation_hash: str
    clean_prefix_hash: str
    decision_boundary: str

    def validate(self) -> None:
        if not all((self.parent_key, self.state_hash, self.observation_hash, self.clean_prefix_hash, self.decision_boundary)):
            raise StageZHold("BRANCH_SNAPSHOT_BINDING_MISSING")


@dataclass(frozen=True)
class ArmPlan:
    arm: StageZArm
    duration: int
    anchor_class: str
    intervention: bool


@dataclass(frozen=True)
class MatrixPreparation:
    status: str
    critical_anchor: AnchorSelection
    noncritical_anchor: AnchorSelection
    arms: tuple[ArmPlan, ...] = ()


def prepare_five_arm_matrix(
    *,
    critical_anchor: AnchorSelection,
    noncritical_anchor: AnchorSelection,
) -> MatrixPreparation:
    """Freeze the five arms or explicitly abstain if a control is unavailable."""

    if critical_anchor.selected is None:
        return MatrixPreparation("ABSTAIN_NO_CRITICAL_ANCHOR", critical_anchor, noncritical_anchor)
    if noncritical_anchor.selected is None:
        return MatrixPreparation("ABSTAIN_NO_NONCRITICAL_CONTROL_ANCHOR", critical_anchor, noncritical_anchor)
    arms = (
        ArmPlan(StageZArm.CLEAN_BRANCH_CRITICAL, 0, "CRITICAL", False),
        ArmPlan(StageZArm.COMMAND_OPEN_T3_CRITICAL, 3, "CRITICAL", True),
        ArmPlan(StageZArm.COMMAND_OPEN_T5_CRITICAL, 5, "CRITICAL", True),
        ArmPlan(StageZArm.COMMAND_OPEN_T10_CRITICAL, 10, "CRITICAL", True),
        ArmPlan(StageZArm.COMMAND_OPEN_T5_NONCRITICAL_CONTROL, 5, "NONCRITICAL", True),
    )
    return MatrixPreparation("READY_FIVE_ARMS", critical_anchor, noncritical_anchor, arms)


def action_for_arm(model_action: FinalLiberoAction | Sequence[float], arm: StageZArm) -> tuple[float, ...]:
    """Prepare an executed action without calling an environment."""

    source = model_action.values if isinstance(model_action, FinalLiberoAction) else validate_final_action(model_action)
    if arm is StageZArm.CLEAN_BRANCH_CRITICAL:
        return source
    duration = {
        StageZArm.COMMAND_OPEN_T3_CRITICAL: 3,
        StageZArm.COMMAND_OPEN_T5_CRITICAL: 5,
        StageZArm.COMMAND_OPEN_T10_CRITICAL: 10,
        StageZArm.COMMAND_OPEN_T5_NONCRITICAL_CONTROL: 5,
    }.get(arm)
    if duration is None:
        raise StageZHold(f"UNKNOWN_STAGE_Z_ARM:{arm}")
    return intervene_gripper_open(source, duration=duration)


__all__ = ["ArmPlan", "BranchSnapshot", "MatrixPreparation", "StageZArm", "action_for_arm", "prepare_five_arm_matrix"]
