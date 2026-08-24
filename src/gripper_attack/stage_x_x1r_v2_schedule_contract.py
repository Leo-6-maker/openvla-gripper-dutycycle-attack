"""Pure prospective Stage-X timing contract.

This module intentionally has no model, simulator, GPU, or filesystem work.
"""
from __future__ import annotations


ATTACK_WINDOW_LENGTH = 5
PHYSICAL_FOLLOWUP_LENGTH = 10
NO_EMIT = "NO_EMIT"
ONE_SHOT = True
PREV_DELTA_BOUNDARIES = {
    "entry": "reset_to_zero_at_attack_window_entry",
    "carry": "same_condition_same_episode_same_5_step_window_only",
    "reset_boundaries": ("parent", "condition", "episode", "attack_window"),
}


def _step(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}_MUST_BE_NONNEGATIVE_INT")
    return value


def attack_steps(t_emit: int) -> tuple[int, ...]:
    start = _step(t_emit, "t_emit")
    return tuple(range(start, start + ATTACK_WINDOW_LENGTH))


def followup_steps(t_emit: int) -> tuple[int, ...]:
    start = _step(t_emit, "t_emit") + ATTACK_WINDOW_LENGTH
    return tuple(range(start, start + PHYSICAL_FOLLOWUP_LENGTH))


def legal_horizon(t_emit: int, episode_length: int) -> bool:
    start = _step(t_emit, "t_emit")
    horizon = _step(episode_length, "episode_length")
    return start + ATTACK_WINDOW_LENGTH + PHYSICAL_FOLLOWUP_LENGTH <= horizon


def first_emit_or_no_emit(t_emit: int | None) -> int | str:
    if t_emit is None:
        return NO_EMIT
    return _step(t_emit, "t_emit")
