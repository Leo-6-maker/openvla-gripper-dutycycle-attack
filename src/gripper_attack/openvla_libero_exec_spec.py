# -*- coding: utf-8 -*-
"""Executable OpenVLA-LIBERO semantics used by this repository.

This module is the single source of truth for OpenVLA action execution in
LIBERO-style environments.  It mirrors the official OpenVLA LIBERO eval chain:

    predict_action/get_action raw 7-DoF action
    -> normalize_gripper_action(action, binarize=True)
    -> invert_gripper_action(action)
    -> env.step(action)

Official convention:
    raw gripper 0 = close, raw gripper 1 = open
    LIBERO env gripper -1 = open, +1 = close

Therefore:
    raw >= 0.5 -> env -1 -> physical OPEN
    raw <  0.5 -> env +1 -> physical CLOSE
"""

from __future__ import annotations

from typing import Mapping, Sequence

OPEN_THRESHOLD_RAW: float = 0.5
RAW_GRIPPER_CLOSE_VALUE: float = 0.0
RAW_GRIPPER_OPEN_VALUE: float = 1.0
ENV_GRIPPER_OPEN_VALUE: float = -1.0
ENV_GRIPPER_CLOSE_VALUE: float = 1.0
OPENVLA_LIBERO_EXEC_SPEC_VERSION = "openvla_libero_exec_spec_v1_20260607"
OFFICIAL_PROMPT_STYLE = "official_in_out"
OFFICIAL_UNNORM_KEY_LIBERO_OBJECT = "libero_object"
OFFICIAL_QPOS_SOURCE = 'obs["robot0_gripper_qpos"]'
OFFICIAL_IMAGE_PREPROCESSING = "agentview_image_rot180_then_octo_resize"


def official_prompt(instruction: str) -> str:
    """Return the official OpenVLA In:/Out: prompt."""
    return f"In: What action should the robot take to {str(instruction).lower()}?\nOut:"


def normalize_gripper_raw(raw: float, *, binarize: bool = True) -> float:
    """Official gripper normalization from raw [0, 1] to [-1, +1]."""
    val = 2.0 * float(raw) - 1.0
    if binarize:
        if val > 0:
            val = 1.0
        elif val < 0:
            val = -1.0
        else:
            val = 1.0  # tie-break: 0 → +1
    return float(val)


def raw_gripper_to_env_gripper(raw: float, *, binarize: bool = True) -> float:
    """Apply official normalize_gripper_action + invert_gripper_action."""
    return float(-normalize_gripper_raw(raw, binarize=binarize))


def decoded_action_to_env_gripper(raw: float, *, binarize: bool = True) -> float:
    """Alias for callers that use decoded action terminology."""
    return raw_gripper_to_env_gripper(raw, binarize=binarize)


def raw_gripper_is_open(raw: float, *, threshold: float = OPEN_THRESHOLD_RAW) -> bool:
    """OPEN in raw/decoded OpenVLA action space."""
    return float(raw) >= float(threshold)


def raw_gripper_is_close(raw: float, *, threshold: float = OPEN_THRESHOLD_RAW) -> bool:
    """CLOSE in raw/decoded OpenVLA action space."""
    return float(raw) < float(threshold)


def env_gripper_is_open(env: float) -> bool:
    """OPEN in LIBERO env action space."""
    return float(env) < -0.5


def env_gripper_is_close(env: float) -> bool:
    """CLOSE in LIBERO env action space."""
    return float(env) > 0.5


def classify_raw_gripper(raw: float, *, threshold: float = OPEN_THRESHOLD_RAW) -> str:
    return "open" if raw_gripper_is_open(raw, threshold=threshold) else "close"


def classify_env_gripper(env: float) -> str:
    if env_gripper_is_open(env):
        return "open"
    if env_gripper_is_close(env):
        return "close"
    return "neutral_or_invalid"


def open_token_ids_from_decoded_action(token_action_map: Mapping[int, float], *, threshold: float = OPEN_THRESHOLD_RAW) -> list[int]:
    """Return token ids whose decoded raw gripper action is physical OPEN."""
    return sorted(int(tid) for tid, raw in token_action_map.items() if raw_gripper_is_open(raw, threshold=threshold))


def close_token_ids_from_decoded_action(token_action_map: Mapping[int, float], *, threshold: float = OPEN_THRESHOLD_RAW) -> list[int]:
    """Return token ids whose decoded raw gripper action is physical CLOSE."""
    return sorted(int(tid) for tid, raw in token_action_map.items() if raw_gripper_is_close(raw, threshold=threshold))


def validate_open_close_token_sets(
    open_token_ids: Sequence[int],
    close_token_ids: Sequence[int],
    token_action_map: Mapping[int, float],
    *,
    threshold: float = OPEN_THRESHOLD_RAW,
) -> None:
    """Hard-fail if token sets violate the executable gripper spec."""
    open_set = {int(t) for t in open_token_ids}
    close_set = {int(t) for t in close_token_ids}
    if not open_set:
        raise AssertionError("OPEN token set is empty")
    if not close_set:
        raise AssertionError("CLOSE token set is empty")
    overlap = open_set & close_set
    if overlap:
        raise AssertionError(f"OPEN/CLOSE token sets overlap: {sorted(overlap)[:10]}")
    for tid in open_set:
        raw = float(token_action_map[int(tid)])
        env = raw_gripper_to_env_gripper(raw)
        if not (raw >= float(threshold) and env_gripper_is_open(env)):
            raise AssertionError(f"OPEN token {tid} decodes to raw={raw:.6f}, env={env:.1f}")
    for tid in close_set:
        raw = float(token_action_map[int(tid)])
        env = raw_gripper_to_env_gripper(raw)
        if not (raw < float(threshold) and env_gripper_is_close(env)):
            raise AssertionError(f"CLOSE token {tid} decodes to raw={raw:.6f}, env={env:.1f}")


def get_libero_image_official(obs: Mapping, *, resize_size=None):
    """Apply the official LIBERO image orientation step.

    This helper intentionally performs only the rotation unless a resize callable
    is supplied by the caller.  Official OpenVLA eval then applies Octo-style
    JPEG encode/decode + lanczos resize; heavyweight TensorFlow imports should
    remain in runner code, not in this lightweight semantics module.
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    if resize_size is None:
        return img
    raise NotImplementedError(
        "Official resize requires the OpenVLA/LIBERO eval TensorFlow path; "
        "call the official libero_utils.resize_image in execution runners."
    )


def _self_check() -> None:
    cases = [
        (0.996, -1.0, True),
        (1.0, -1.0, True),
        (0.5, -1.0, True),
        (0.0, 1.0, False),
        (0.499, 1.0, False),
    ]
    for raw, expected_env, expected_open in cases:
        env = raw_gripper_to_env_gripper(raw)
        assert env == expected_env, (raw, env, expected_env)
        assert raw_gripper_is_open(raw) is expected_open, (raw, expected_open)
        assert env_gripper_is_open(env) is expected_open, (env, expected_open)
    assert env_gripper_is_close(1.0)
    assert not env_gripper_is_close(-1.0)


_self_check()
