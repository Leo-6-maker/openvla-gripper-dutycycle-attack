"""Test phase-conditioned VIS audit logic — imports from real audit module."""

import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'diagnostics'))

# Import real functions from the audit module
from audit_phase_conditioned_vis import classify_bridge_taxonomy, compute_trace_metrics
from gripper_attack.gripper_semantics import raw_gripper_is_open, QPOS_OPEN_MAX, QPOS_CLOSED_MIN


def _fake_metrics(open_cnt=0, total=18, qpos_delta=0.0, done=True, attack_invalid=False):
    return {"generated_OPEN_count":open_cnt,"generated_OPEN_total":total,
            "qpos_delta_post":qpos_delta,"done":done,"armL2_max":0.0,
            "attack_invalid":attack_invalid,"valid":not attack_invalid}


class TestEarlyWindowPositive:
    def test_claim_usable(self):
        """Early window: 18/18 OPEN, qpos=0.038, done=False, random clean."""
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(0,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["claim_usable"] is True
        assert tax["taxonomy_label"] == "claim_usable"


class TestLateWindowNegative:
    def test_action_positive_physical_negative(self):
        """Late window: 18/18 OPEN, qpos=0.0001, done=True."""
        vis = _fake_metrics(18,18,0.0001,True)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(0,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["action_bridge_positive"] is True
        assert tax["physical_bridge_positive"] is False
        assert tax["taxonomy_label"] == "action_bridge_positive_physical_bridge_negative"

    def test_no_action_bridge(self):
        """VIS generates only 2/18 OPEN."""
        vis = _fake_metrics(2,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, None, None)
        assert tax["action_bridge_positive"] is False
        assert tax["taxonomy_label"] == "no_action_bridge"


class TestNaturalReleaseConfounded:
    def test_clean_open_high_confounds(self):
        """Clean natural OPEN at 75% confounds the window."""
        vis = _fake_metrics(18,18,0.04,False)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(14,18,0.0,True)  # 14/18 = 78% natural OPEN
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["natural_release_confounded"] is True
        assert tax["taxonomy_label"] == "natural_release_confounded"


class TestDenominatorPolluted:
    def test_random_fail(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(0,18,0.0006,False)  # random task failed
        tax = classify_bridge_taxonomy(vis, rand, None)
        assert tax["denominator_clean"] is False

    def test_random_open(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(5,18,0.0006,True)  # random has OPEN
        tax = classify_bridge_taxonomy(vis, rand, None)
        assert tax["denominator_clean"] is False


class TestGeneratedOpenInsufficient:
    def test_open_alone_insufficient(self):
        vis = _fake_metrics(18,18,0.0,True)
        rand = _fake_metrics(0,18,0.0006,True)
        tax = classify_bridge_taxonomy(vis, rand, None)
        assert tax["action_bridge_positive"] is True
        assert tax["claim_usable"] is False


class TestInvalidExcluded:
    def test_oom_trace_excluded(self):
        """OOM/invalid trace should be excluded."""
        vis = _fake_metrics(0,0,0.0,True,attack_invalid=True)
        assert vis["valid"] is False
        tax = classify_bridge_taxonomy(vis, None, None)
        assert tax["claim_usable"] is False


class TestQposDirection:
    """Verify qpos direction: OPEN=qpos_low, CLOSE=qpos_high."""
    def test_open_qpos_low(self):
        assert raw_gripper_is_open(0.0) is True

    def test_close_qpos_high_raw(self):
        assert raw_gripper_is_open(0.996) is False

    def test_physical_open_threshold(self):
        assert QPOS_OPEN_MAX < QPOS_CLOSED_MIN
