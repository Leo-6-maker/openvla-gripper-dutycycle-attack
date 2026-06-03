"""Test phase-conditioned VIS audit logic — imports from real audit module."""

import os, sys, csv, io, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'diagnostics'))

from audit_phase_conditioned_vis import (
    classify_bridge_taxonomy, compute_trace_metrics, compute_group_summary,
    get_raw_gripper, QPOS_OPENING_DELTA_THRESH,
)
from gripper_attack.gripper_semantics import raw_gripper_is_open, QPOS_OPEN_MAX, QPOS_CLOSED_MIN


def _fake_metrics(open_cnt=0, total=18, qpos_opening=0.0, done=True, attack_invalid=False):
    return {"generated_OPEN_count":open_cnt,"generated_OPEN_total":total,
            "qpos_opening_delta":qpos_opening,"qpos_abs_delta":abs(qpos_opening),
            "qpos_post_start":0.039,"qpos_post_min":0.039-qpos_opening,
            "done":done,"armL2_max":0.0,"valid":not attack_invalid,
            "attack_invalid":attack_invalid,"claim_excluded":attack_invalid}


def _write_fake_trace(rows, path):
    """Write fake trace CSV to path."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ── P0 REGRESSION: or 0.996 bug ──

class TestOpenCountNoFalsyBug:
    def test_adv_grip_zero_counts_as_open(self):
        """adv_grip=0.0 MUST be OPEN, NOT CLOSE via 'or 0.996' fallback."""
        rows = [{"in_window":"True","adv_grip":"0.0","arm_l2":"0","done":"False",
                 "qpos_post_step":"0.039","condition":"vis_pgd","task":"test","seed":"0",
                 "window_start":"10","window_end":"27"} for _ in range(18)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            _write_fake_trace(rows, f.name)
            m = compute_trace_metrics(f.name)
        os.unlink(f.name)
        assert m["generated_OPEN_count"] == 18, \
            f"OPEN=0.0 was miscounted as CLOSE due to 'or 0.996': got {m['generated_OPEN_count']}"

    def test_adv_grip_0996_counts_as_close(self):
        """adv_grip=0.996 MUST be CLOSE."""
        rows = [{"in_window":"True","adv_grip":"0.996","arm_l2":"0","done":"True",
                 "qpos_post_step":"0.039","condition":"vis_pgd","task":"test","seed":"0",
                 "window_start":"10","window_end":"27"} for _ in range(18)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            _write_fake_trace(rows, f.name)
            m = compute_trace_metrics(f.name)
        os.unlink(f.name)
        assert m["generated_OPEN_count"] == 0

    def test_missing_gripper_counts_missing(self):
        """Row without any gripper field should produce claim_excluded."""
        rows = [{"in_window":"True","arm_l2":"0","done":"True",
                 "qpos_post_step":"0.039","condition":"clean","task":"test","seed":"0",
                 "window_start":"10","window_end":"27"} for _ in range(5)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            _write_fake_trace(rows, f.name)
            m = compute_trace_metrics(f.name)
        os.unlink(f.name)
        assert m["claim_excluded"] is True
        assert "missing_gripper" in m["exclusion_reason"]


# ── P0 REGRESSION: qpos field mismatch ──

class TestGroupSummaryNoKeyError:
    def test_group_summary_fields_exist(self):
        """compute_group_summary must not KeyError — uses qpos_opening_delta, not qpos_delta_post."""
        vis = [_fake_metrics(18,18,0.038,False)]
        rand = [_fake_metrics(0,18,0.0006,True)]
        clean = [_fake_metrics(0,18,0.0,True)]
        s = compute_group_summary(vis, rand, clean, {"task":"test","seed":"0"})
        assert "vis_qpos_opening_delta_mean" in s
        assert "vis_qpos_opening_delta_min" in s
        assert "random_qpos_opening_delta_max" in s
        assert "vis_OPEN_min" in s


# ── Existing taxonomy tests ──

class TestEarlyWindowPositive:
    def test_claim_usable(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(0,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["claim_usable"] is True


class TestLateWindowNegative:
    def test_action_positive_physical_negative(self):
        vis = _fake_metrics(18,18,0.0001,True)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(0,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["action_bridge_positive"] is True
        assert tax["physical_bridge_positive"] is False


class TestNaturalReleaseConfounded:
    def test_confounded(self):
        vis = _fake_metrics(18,18,0.04,False)
        rand = _fake_metrics(0,18,0.0006,True)
        clean = _fake_metrics(14,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["natural_release_confounded"] is True


class TestDenominatorPolluted:
    def test_random_fail(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(0,18,0.0006,False)
        tax = classify_bridge_taxonomy(vis, rand, None)
        assert tax["denominator_clean"] is False


class TestTaxonomyPreservesBoth:
    def test_confounded_and_denom(self):
        vis = _fake_metrics(18,18,0.038,False)
        rand = _fake_metrics(5,18,0.01,False)
        clean = _fake_metrics(14,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, rand, clean)
        assert tax["natural_release_confounded"] is True
        assert tax["denominator_clean"] is False
        assert "natural_release_confounded" in tax["taxonomy_label"]
        assert "denominator_polluted" in tax["taxonomy_label"]


class TestQposDirectional:
    def test_directional_opening(self):
        vis = _fake_metrics(18,18,0.038,False)
        tax = classify_bridge_taxonomy(vis, _fake_metrics(0,18,0.0006,True), _fake_metrics(0,18,0.0,True))
        assert tax["physical_bridge_positive"] is True

    def test_no_opening_when_stable(self):
        vis = _fake_metrics(18,18,0.0,True)
        tax = classify_bridge_taxonomy(vis, _fake_metrics(0,18,0.0006,True), _fake_metrics(0,18,0.0,True))
        assert tax["physical_bridge_positive"] is False


class TestGetRawGripper:
    def test_finds_adv_grip(self):
        assert get_raw_gripper({"adv_grip": "0.0"}) == 0.0

    def test_fallback_order(self):
        assert get_raw_gripper({"raw_gripper": "0.996"}) == 0.996

    def test_returns_none(self):
        assert get_raw_gripper({"x": "1"}) is None

    def test_zero_is_not_none(self):
        """0.0 is a valid raw gripper value (OPEN), NOT None."""
        g = get_raw_gripper({"adv_grip": 0.0})
        assert g == 0.0
        assert g is not None


class TestQposConstants:
    def test_open_max_less_than_closed_min(self):
        assert QPOS_OPEN_MAX < QPOS_CLOSED_MIN


# ── P0 REGRESSION: partial qpos missing ──

class TestPartialQposExcluded:
    def test_missing_1_of_18_qpos_excluded(self):
        """If 1 of 18 in-window rows is missing qpos, trace must be claim_excluded."""
        rows = []
        for i in range(18):
            r = {"in_window":"True","adv_grip":"0.0","arm_l2":"0","done":"False",
                 "condition":"vis_pgd","task":"test","seed":"0",
                 "window_start":"10","window_end":"27"}
            if i != 5:  # row 5 missing qpos
                r["qpos_post_step"] = "0.039"
            rows.append(r)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            _write_fake_trace(rows, f.name)
            m = compute_trace_metrics(f.name)
        os.unlink(f.name)
        assert m["claim_excluded"] is True
        assert m["missing_qpos_count"] == 1


class TestCleanGripFallback:
    def test_clean_grip_only_counts_open(self):
        """clean trace with only clean_grip field must count natural OPEN."""
        rows = [{"in_window":"True","clean_grip":"0.0","arm_l2":"0","done":"True",
                 "qpos_post_step":"0.039","condition":"clean","task":"test","seed":"0",
                 "window_start":"10","window_end":"27"} for _ in range(18)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            _write_fake_trace(rows, f.name)
            m = compute_trace_metrics(f.name)
        os.unlink(f.name)
        assert m["generated_OPEN_count"] == 18, \
            f"clean_grip=0.0 was not counted as OPEN: got {m['generated_OPEN_count']}"


class TestDuplicateDetection:
    def test_duplicate_vis_triggers_duplicate_flag(self):
        """Two valid VIS traces in same group should trigger duplicate detection."""
        vis = [
            _fake_metrics(18,18,0.038,False),
            _fake_metrics(18,18,0.037,False),
        ]
        rand = [_fake_metrics(0,18,0.0006,True)]
        clean = [_fake_metrics(0,18,0.0,True)]
        s = compute_group_summary(vis, rand, clean, {"task":"test","seed":"0"})
        assert s["duplicate_condition_count"] == 1  # vis has duplicate

    def test_no_duplicate_when_single_each(self):
        vis = [_fake_metrics(18,18,0.038,False)]
        rand = [_fake_metrics(0,18,0.0006,True)]
        clean = [_fake_metrics(0,18,0.0,True)]
        s = compute_group_summary(vis, rand, clean, {"task":"test","seed":"0"})
        assert s["duplicate_condition_count"] == 0
        assert s["claim_usable"] is True
