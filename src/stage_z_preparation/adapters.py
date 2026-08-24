"""Model-family boundary adapters using synthetic actions only.

These classes do not load a checkpoint or call a policy.  They validate the
state a future official adapter must expose at its final LIBERO action
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import FinalLiberoAction, StageZHold, validate_final_action


MODEL_M0 = "Z-M0_OPENVLA"
MODEL_M1 = "Z-M1_OPENVLA_OFT"
MODEL_M2 = "Z-M2_PI05_LIBERO"
BOUNDARY_M0 = "FRESH_PER_STEP"
BOUNDARY_M1 = "FRESH_OFT_ACTION_QUEUE"
BOUNDARY_M2 = "FRESH_PI05_REPLAN"


@dataclass(frozen=True)
class OFTQueueState:
    """Synthetic state for official OFT chunk/open-loop boundary checks."""

    num_actions_chunk: int = 8
    num_open_loop_steps: int = 8
    residual_actions: int = 0
    chunk_id: int = 0

    def require_fresh_boundary(self) -> None:
        if self.num_actions_chunk != 8 or self.num_open_loop_steps != 8:
            raise StageZHold("OFT_CHUNK_CONTRACT_CHANGED")
        if self.residual_actions != 0:
            raise StageZHold("OFT_RESIDUAL_QUEUE_AT_BRANCH")

    def consume_one(self) -> "OFTQueueState":
        if self.residual_actions <= 0:
            raise StageZHold("OFT_QUEUE_EMPTY")
        return OFTQueueState(
            num_actions_chunk=self.num_actions_chunk,
            num_open_loop_steps=self.num_open_loop_steps,
            residual_actions=self.residual_actions - 1,
            chunk_id=self.chunk_id,
        )

    def start_synthetic_chunk(self) -> "OFTQueueState":
        return OFTQueueState(
            num_actions_chunk=self.num_actions_chunk,
            num_open_loop_steps=self.num_open_loop_steps,
            residual_actions=self.num_actions_chunk,
            chunk_id=self.chunk_id + 1,
        )


@dataclass(frozen=True)
class Pi05ReplanState:
    """Synthetic state for the official pi05 fresh-replan boundary."""

    replan_steps: int = 5
    action_horizon: int = 10
    steps_since_replan: int = 0
    residual_actions: int = 0
    replan_id: int = 0

    def require_fresh_boundary(self) -> None:
        if self.replan_steps != 5 or self.action_horizon != 10:
            raise StageZHold("PI05_REPLAN_CONTRACT_CHANGED")
        if self.steps_since_replan != 0 or self.residual_actions != 0:
            raise StageZHold("PI05_NOT_AT_FRESH_REPLAN_BOUNDARY")

    def advance_one(self) -> "Pi05ReplanState":
        return Pi05ReplanState(
            replan_steps=self.replan_steps,
            action_horizon=self.action_horizon,
            steps_since_replan=self.steps_since_replan + 1,
            residual_actions=max(0, self.residual_actions - 1),
            replan_id=self.replan_id,
        )


@dataclass(frozen=True)
class M0Adapter:
    suite: str
    authority_id: str

    def expose_final_action(self, action: tuple[float, ...]) -> FinalLiberoAction:
        values = validate_final_action(action)
        return FinalLiberoAction(values, MODEL_M0, self.authority_id, BOUNDARY_M0, 0)


@dataclass(frozen=True)
class OFTAdapter:
    suite: str
    authority_id: str

    def expose_final_action(self, action: tuple[float, ...], state: OFTQueueState) -> FinalLiberoAction:
        state.require_fresh_boundary()
        values = validate_final_action(action)
        return FinalLiberoAction(values, MODEL_M1, self.authority_id, BOUNDARY_M1, state.residual_actions)


@dataclass(frozen=True)
class Pi05Adapter:
    authority_id: str

    def expose_final_action(self, action: tuple[float, ...], state: Pi05ReplanState) -> FinalLiberoAction:
        state.require_fresh_boundary()
        values = validate_final_action(action)
        return FinalLiberoAction(values, MODEL_M2, self.authority_id, BOUNDARY_M2, state.residual_actions)


__all__ = [
    "BOUNDARY_M0",
    "BOUNDARY_M1",
    "BOUNDARY_M2",
    "MODEL_M0",
    "MODEL_M1",
    "MODEL_M2",
    "M0Adapter",
    "OFTAdapter",
    "OFTQueueState",
    "Pi05Adapter",
    "Pi05ReplanState",
]
