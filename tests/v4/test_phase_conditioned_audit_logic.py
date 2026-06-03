"""Test phase-conditioned VIS audit logic — imports from real audit module."""

import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'diagnostics'))

from audit_phase_conditioned_vis import (
    classify_bridge_taxonomy, compute_trace_metrics,
    get_raw_gripper, QPOS_OPENING_DELTA_THRESH,
)
from gripper_attack.gripper_semantics import raw_gripper_is_open, QPOS_OPEN_MAX, QPOS_CLOSED_MIN


def _fake_metrics(open_cnt=0, total=18, qpos_opening=0.0, done=True, attack_invalid=False):
    return {"generated_OPEN_count":open_cnt,"generated_OPEN_total":total,
            "qpos_opening_delta":qpos_opening,"qpos_abs_delta":abs(qpos_opening),
            "qpos_post_start":0.039,"qpos_post_min":0.039-qpos_opening,
            "done":done,"armL2_max":0.0,
            "attack_invalid":attack_invalid,"valid":not attack_invalid}


class TestEarlyWindowPositive:
    def test_claim_usable(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(0,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["claim_usable"] is True
        assert "claim_usable" in tax["taxonomy_label"]


class TestLateWindowNegative:
    def test_action_positive_physical_negative(self):
        vis = _fake_metrics(18,18,0.0001,True)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(0,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["action_bridge_positive"] is True
        assert tax["physical_bridge_positive"] is False
        assert "action_positive_physical_negative" in tax["taxonomy_label"]

    def test_no_action_bridge(self):
        vis = _fake_metrics(2,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, None, None)
        assert tax["action_bridge_positive"] is False
        assert "no_action_bridge" in tax["taxonomy_label"]


class TestNaturalReleaseConfounded:
    def test_confounded_with_clean_denom(self):
        vis = _fake_metrics(18,18,0.04,False)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(14,18,0.0,True)  # 78% natural OPEN
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["natural_release_confounded"] is True
        assert "natural_release_confounded" in tax["taxonomy_label"]


class TestDenominatorPolluted:
    def test_random_fail(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(0,18,0.0006,False)
        tax = classify_bridge_taxonomy(vis, rand, None)
        assert tax["denominator_clean"] is False
        assert "denominator_polluted" in tax["taxonomy_label"]

    def test_random_open(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(5,18,0.0006,True)
        tax = classify_bridge_taxonomy(vis, rand, None)
        assert tax["denominator_clean"] is False


class TestTaxonomyPreservesBoth:
    def test_confounded_and_denom_polluted(self):
        """Both confound and denominator polluted should appear in label."""
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(5,18,0.01,False)  # polluted: OPEN>0, task fails
        clean = _fake_metrics(14,18,0.0,True)  # confounded: high natural OPEN
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["natural_release_confounded"] is True
        assert tax["denominator_clean"] is False
        assert "natural_release_confounded" in tax["taxonomy_label"]
        assert "denominator_polluted" in tax["taxonomy_label"]


class TestQposDirectional:
    def test_directional_opening_detected(self):
        """qpos from 0.039 -> 0.001: opening_delta = 0.038 >= threshold."""
        vis = _fake_metrics(18,18,0.038,False)
        assert vis["qpos_post_start"] == 0.039
        assert vis["qpos_post_min"] == 0.001
        tax = classify_bridge_taxonomy(vis, _fake_metrics(0,18,0.0006,True), _fake_metrics(0,18,0.0,True))
        assert tax["physical_bridge_positive"] is True

    def test_no_opening_when_stable(self):
        """qpos stable at 0.039: opening_delta = 0 < threshold."""
        vis = _fake_metrics(18,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, _fake_metrics(0,18,0.0006,True), _fake_metrics(0,18,0.0,True))
        assert tax["physical_bridge_positive"] is False


class TestGetRawGripper:
    def test_finds_adv_grip(self):
        assert get_raw_gripper({"adv_grip": "0.0"}) == 0.0

    def test_falls_back_to_raw_gripper(self):
        assert get_raw_gripper({"raw_gripper": "0.996"}) == 0.996

    def test_falls_back_to_clean_grip(self):
        assert get_raw_gripper({"clean_grip": "0.0"}) == 0.0

    def test_returns_none_when_missing(self):
        assert get_raw_gripper({"other": "x"}) is None

    def test_clean_open_via_clean_grip(self):
        """clean trace without adv_grip but with clean_grip should count natural OPEN."""
        grip = get_raw_gripper({"clean_grip": "0.0"})
        assert grip == 0.0
        assert raw_gripper_is_open(grip) is True


class TestQposConstants:
    def test_open_max_less_than_closed_min(self):
        assert QPOS_OPEN_MAX < QPOS_CLOSED_MIN

    def test_opening_delta_threshold(self):
        assert QPOS_OPENING_DELTA_THRESH == 0.03
