"""Frozen clean-runtime semantics for the prospective Stage X D1 gate.

This module is intentionally pure: importing it does not discover files,
load weights, import a simulator, or authorize execution.
"""

from __future__ import annotations

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASKS = tuple(range(10))
HORIZONS = {
    "libero_10": 520,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_spatial": 220,
}
NUM_STEPS_WAIT = 10
T5_STEPS = 5
H_PHYS = 10
ATTACK_WINDOW = tuple(range(T5_STEPS))
FOLLOWUP_WINDOW = tuple(range(T5_STEPS, T5_STEPS + H_PHYS))
SUCCESS_SEMANTIC_ORIGIN = "OPENVLA_LIBERO_CANONICAL_ENV_STEP_DONE"
HORIZON_SEMANTIC_ORIGIN = "OPENVLA_LIBERO_CANONICAL_POLICY_DECISION_HORIZON"


def configured_episode_length(suite: str, task_index: int) -> int:
    """Return the frozen policy-decision horizon for one four-suite task."""
    if suite not in HORIZONS:
        raise ValueError(f"unsupported suite: {suite}")
    if int(task_index) not in TASKS:
        raise ValueError(f"unsupported task index: {task_index}")
    return HORIZONS[suite]


def legal_horizon(t_emit: int, episode_length: int) -> bool:
    """Require all five primary and ten follow-up policy decisions."""
    return int(t_emit) + T5_STEPS + H_PHYS <= int(episode_length)


def policy_step_indices(suite: str, task_index: int) -> tuple[int, ...]:
    """Return the zero-based policy-decision indices available to the task."""
    return tuple(range(configured_episode_length(suite, task_index)))


def success_from_canonical_done(done: object) -> bool:
    """The canonical evaluator treats the post-step environment ``done`` as success."""
    return bool(done)

