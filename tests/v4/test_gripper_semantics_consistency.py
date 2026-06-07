# -*- coding: utf-8 -*-
"""Compatibility tests for gripper_semantics wrapper."""

import numpy as np

from gripper_attack.gripper_semantics import (
    OPEN_THRESHOLD,
    CANONICAL_OPEN_SEMANTICS_VERSION,
    classify_gripper_action,
    decoded_action_to_env_gripper,
    env_gripper_is_close,
    env_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_is_open,
)


def test_raw_open_values():
    assert raw_gripper_is_open(0.5)
    assert raw_gripper_is_open(0.8)
    assert raw_gripper_is_open(0.996)
    assert not raw_gripper_is_open(0.499)
    assert not raw_gripper_is_open(0.0)


def test_raw_close_values():
    assert raw_gripper_is_close(0.0)
    assert raw_gripper_is_close(0.499)
    assert not raw_gripper_is_close(0.5)
    assert not raw_gripper_is_close(0.996)


def test_raw_to_env_mapping():
    assert decoded_action_to_env_gripper(0.996) == -1.0
    assert decoded_action_to_env_gripper(0.5) == -1.0
    assert decoded_action_to_env_gripper(0.0) == 1.0


def test_env_open_close_values():
    assert env_gripper_is_open(-1.0)
    assert env_gripper_is_open(-0.6)
    assert not env_gripper_is_open(1.0)
    assert env_gripper_is_close(1.0)
    assert env_gripper_is_close(0.6)
    assert not env_gripper_is_close(-1.0)


def test_classify_wrapper():
    assert classify_gripper_action(0.996) == "open"
    assert classify_gripper_action(0.5) == "open"
    assert classify_gripper_action(0.0) == "close_or_hold"


def test_raw_env_equivalence():
    for raw_val in np.linspace(0.0, 1.0, 51):
        env_val = decoded_action_to_env_gripper(raw_val)
        assert raw_gripper_is_open(raw_val) == env_gripper_is_open(env_val)


def test_semantics_version():
    assert isinstance(CANONICAL_OPEN_SEMANTICS_VERSION, str)
    assert "20260607" in CANONICAL_OPEN_SEMANTICS_VERSION
    assert OPEN_THRESHOLD == 0.5
