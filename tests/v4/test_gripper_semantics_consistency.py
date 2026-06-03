# -*- coding: utf-8 -*-
"""Test gripper semantics consistency across the codebase."""

import pytest
import numpy as np
from gripper_attack.gripper_semantics import (
    raw_gripper_is_open,
    raw_gripper_is_close,
    decoded_action_to_env_gripper,
    env_gripper_is_open,
    env_gripper_is_close,
    classify_gripper_action,
    OPEN_THRESHOLD,
    CANONICAL_OPEN_SEMANTICS_VERSION,
)


class TestRawGripperIsOpen:
    def test_open_values(self):
        """Values below threshold are OPEN."""
        assert raw_gripper_is_open(-0.996)
        assert raw_gripper_is_open(-0.5)
        assert raw_gripper_is_open(0.0)
        assert raw_gripper_is_open(0.3)
        assert raw_gripper_is_open(0.499)

    def test_close_values(self):
        """Values above threshold are CLOSE."""
        assert not raw_gripper_is_open(0.501)
        assert not raw_gripper_is_open(0.8)
        assert not raw_gripper_is_open(0.996)

    def test_threshold_boundary(self):
        """Exactly at threshold."""
        assert not raw_gripper_is_open(OPEN_THRESHOLD)

    def test_raw_gripper_is_close(self):
        assert raw_gripper_is_close(0.996)
        assert not raw_gripper_is_close(0.0)


class TestDecodedActionToEnvGripper:
    def test_open_maps_to_positive_env(self):
        """Decoded OPEN (~0.0) → env = +1."""
        assert decoded_action_to_env_gripper(0.0) == 1.0
        assert decoded_action_to_env_gripper(-0.5) == 1.0
        assert decoded_action_to_env_gripper(-0.996) == 1.0

    def test_close_maps_to_negative_env(self):
        """Decoded CLOSE (~0.996) → env = -1."""
        assert decoded_action_to_env_gripper(0.996) == -1.0
        assert decoded_action_to_env_gripper(0.8) == -1.0

    def test_exactly_zero_5(self):
        """decoded=0.5: 2*0.5-1=0, sign(0)=0, treat as +1, invert → -1 (CLOSE)."""
        result = decoded_action_to_env_gripper(0.5)
        assert result == -1.0, f"decoded=0.5 → env={result}, expected -1 (CLOSE, boundary→closed)"


class TestEnvGripperIsOpen:
    def test_positive_env_is_open(self):
        assert env_gripper_is_open(1.0)
        assert env_gripper_is_open(0.5)

    def test_negative_env_is_close(self):
        assert not env_gripper_is_open(-1.0)
        assert env_gripper_is_close(-1.0)


class TestClassifyGripperAction:
    def test_open(self):
        assert classify_gripper_action(0.0) == "open"
        assert classify_gripper_action(-0.996) == "open"

    def test_close(self):
        assert classify_gripper_action(0.996) == "close_or_hold"

    def test_boundary(self):
        assert classify_gripper_action(0.5) == "boundary"


class TestEquivalence:
    """raw_gripper_is_open ⇔ env_gripper_is_open for non-boundary values."""

    @pytest.mark.parametrize("raw_val", [-0.996, -0.5, 0.0, 0.3, 0.6, 0.8, 0.996])
    def test_raw_env_equivalence(self, raw_val):
        env_val = decoded_action_to_env_gripper(raw_val)
        assert raw_gripper_is_open(raw_val) == env_gripper_is_open(env_val), \
            f"raw={raw_val}: raw_is_open={raw_gripper_is_open(raw_val)} env_is_open={env_gripper_is_open(env_val)}"


class TestSemanticsVersion:
    def test_version_is_string(self):
        assert isinstance(CANONICAL_OPEN_SEMANTICS_VERSION, str)
        assert len(CANONICAL_OPEN_SEMANTICS_VERSION) > 0
        assert "20260603" in CANONICAL_OPEN_SEMANTICS_VERSION


class TestNoRawComparisonsInHelpers:
    """The helpers themselves use the threshold, but calling code should not use
    raw < > 0.5 directly. This test verifies the helpers are self-consistent."""

    def test_classify_matches_raw_is_open(self):
        for raw_val in np.linspace(-0.996, 0.996, 50):
            cls = classify_gripper_action(raw_val)
            is_open = raw_gripper_is_open(raw_val)
            if cls == "boundary":
                continue
            assert is_open == (cls == "open"), \
                f"raw={raw_val:.4f}: classify={cls} raw_is_open={is_open}"
