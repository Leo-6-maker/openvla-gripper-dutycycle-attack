# -*- coding: utf-8 -*-
"""Backward-compatible wrapper for canonical OpenVLA-LIBERO gripper semantics.

New code should import :mod:`gripper_attack.openvla_libero_exec_spec` directly.
This module remains for older scripts that already import ``gripper_semantics``.
"""

from __future__ import annotations

from .openvla_libero_exec_spec import (
    OPEN_THRESHOLD_RAW,
    RAW_GRIPPER_CLOSE_VALUE,
    RAW_GRIPPER_OPEN_VALUE,
    ENV_GRIPPER_OPEN_VALUE,
    ENV_GRIPPER_CLOSE_VALUE,
    OPENVLA_LIBERO_EXEC_SPEC_VERSION,
    raw_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_is_boundary,
    raw_gripper_to_env_gripper,
    decoded_action_to_env_gripper,
    env_gripper_is_open,
    env_gripper_is_close,
    classify_raw_gripper,
)

OPEN_THRESHOLD = OPEN_THRESHOLD_RAW
CANONICAL_OPEN_SEMANTICS_VERSION = OPENVLA_LIBERO_EXEC_SPEC_VERSION

# Legacy qpos aliases. Physical OPEN in current LIBERO convention corresponds
# to increasing obs["robot0_gripper_qpos"] abs_sum / finger width.
QPOS_OPEN_MIN = 0.03
QPOS_CLOSED_MAX = 0.005
QPOS_OPEN_MAX = QPOS_CLOSED_MAX
QPOS_CLOSED_MIN = QPOS_OPEN_MIN


def classify_gripper_action(raw_gripper: float, *, threshold: float = OPEN_THRESHOLD) -> str:
    value = classify_raw_gripper(raw_gripper, threshold=threshold)
    if value == "close":
        return "close_or_hold"
    return value
