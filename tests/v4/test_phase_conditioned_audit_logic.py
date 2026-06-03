"""Test phase-conditioned VIS audit logic: claim gates, bridge taxonomy."""

import csv, io, os, sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from gripper_attack.gripper_semantics import raw_gripper_is_open


def _compute_bridge_taxonomy(open_cnt, total, qpos_delta, done, clean_ratio, random_done=True, random_open=0):
    """Mirror audit_phase_conditioned_vis claim gate logic."""
    action_pos = open_cnt >= max(1, total - 2) if total > 0 else False
    physical_pos = qpos_delta >= 0.03 if total > 0 else False
    task_fail = not done if total > 0 else False
    confounded = clean_ratio > 0.5
    denom_clean = random_done and random_open == 0
    claim = action_pos and physical_pos and task_fail and denom_clean and not confounded
    return {
        "action_bridge_positive": action_pos,
        "physical_bridge_positive": physical_pos,
        "task_failure_positive": task_fail,
        "natural_release_confounded": confounded,
        "denominator_clean": denom_clean,
        "claim_usable": claim,
    }


class TestEarlyWindowPositive:
    def test_early_window_claim_usable(self):
        """Early window: 18/18 OPEN, qpos=0.038, done=False, clean_ratio=0, random clean."""
        g = _compute_bridge_taxonomy(18, 18, 0.038, False, 0.0)
        assert g["action_bridge_positive"] is True
        assert g["physical_bridge_positive"] is True
        assert g["task_failure_positive"] is True
        assert g["natural_release_confounded"] is False
        assert g["denominator_clean"] is True
        assert g["claim_usable"] is True


class TestLateWindowNegative:
    def test_late_window_action_positive_physical_negative(self):
        """Late window: 18/18 OPEN, qpos=0.0001, done=True, clean_ratio=0.5."""
        g = _compute_bridge_taxonomy(18, 18, 0.0001, True, 0.5)
        assert g["action_bridge_positive"] is True
        assert g["physical_bridge_positive"] is False
        assert g["task_failure_positive"] is False
        assert g["claim_usable"] is False

    def test_late_window_label_action_positive_physical_negative(self):
        """Late window explicitly labeled."""
        g = _compute_bridge_taxonomy(18, 18, 0.00001, True, 0.0)
        assert not g["physical_bridge_positive"]
        assert not g["claim_usable"]


class TestNaturalReleaseConfounded:
    def test_high_clean_open_confounds(self):
        """Window with clean natural OPEN >50% is confounded."""
        g = _compute_bridge_taxonomy(18, 18, 0.04, False, 0.75)
        assert g["natural_release_confounded"] is True
        assert g["claim_usable"] is False


class TestDenominatorPolluted:
    def test_random_fail_pollutes(self):
        """Random task failure pollutes denominator."""
        g = _compute_bridge_taxonomy(18, 18, 0.038, False, 0.0, random_done=False)
        assert g["denominator_clean"] is False
        assert g["claim_usable"] is False

    def test_random_open_pollutes(self):
        """Random OPEN pollutes denominator."""
        g = _compute_bridge_taxonomy(18, 18, 0.038, False, 0.0, random_open=5)
        assert g["denominator_clean"] is False
        assert g["claim_usable"] is False


class TestGeneratedOpenInsufficient:
    def test_open_alone_insufficient(self):
        """18/18 OPEN but qpos≈0 and done=True: not a usable claim."""
        g = _compute_bridge_taxonomy(18, 18, 0.0, True, 0.0)
        assert g["action_bridge_positive"] is True
        assert g["claim_usable"] is False
