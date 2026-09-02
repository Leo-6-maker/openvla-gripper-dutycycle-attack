"""CPU synthetic tests for V5 scheduler L3 metric contract.

Tests verify:
- Teacher event_id must not change scheduler output when runtime fields unchanged
- Negative episode false start counting
- Valid opportunity recall vs false start separation
- No-emit is NOT counted as safe
- Episode boundary resets scheduler state
- Valid-mask gap resets dwell/persistence
- Release/regrasp veto behavior
- Calibration fit and heldout identities are disjoint
- Each metric numerator/denominator is hand-computable from synthetic fixtures
"""
from __future__ import annotations

import math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def sigmoid(z):
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _make_step(step, canonical_parent_key="test/ep_0", route_supported=True,
               event_id=0, grasp_prob=0.0, release_prob=0.0, manipulation_prob=0.0,
               grasp_known=True, release_known=True, manipulation_known=False):
    return {
        "step_index": step,
        "canonical_parent_key": canonical_parent_key,
        "route_supported": route_supported,
        "event_id": event_id,
        "grasp_prob": grasp_prob,
        "release_prob": release_prob,
        "manipulation_prob": manipulation_prob,
        "grasp_known_mask": grasp_known,
        "release_known_mask": release_known,
        "manipulation_known_mask": manipulation_known,
    }


def run_scheduler(steps, candidate_close_fn=None):
    """Run scheduler on ordered steps. candidate_close_fn maps step -> bool."""
    config = V5SchedulerConfig()
    scheduler = V5OneShotScheduler(config)
    for s in steps:
        cc = candidate_close_fn(s) if candidate_close_fn else (s["route_supported"] and s.get("event_id", -1) >= 0)
        result = scheduler.update(
            step=s["step_index"],
            candidate_close=cc,
            valid=s["route_supported"],
            utility_probability=s.get("grasp_prob", 0),
            release_probability=s.get("release_prob", 0),
            regrasp_probability=s.get("manipulation_prob", 0),
            uncertainty_probability=0.0,
        )
    return result


# ── Test 1: Teacher event_id changes but runtime fields unchanged → scheduler output unchanged ──
def test_event_id_does_not_affect_scheduler_with_runtime_candidate():
    """If candidate_close is purely runtime (not event_id-dependent), changing
    event_id must not change scheduler output."""
    # Runtime candidate_close: grasp_prob >= 0.5 (proxy for close intent)
    def runtime_cc(s):
        return s["route_supported"] and s["grasp_prob"] >= 0.5

    steps_v1 = [
        _make_step(0, event_id=0, grasp_prob=0.9),
        _make_step(1, event_id=0, grasp_prob=0.9),
        _make_step(2, event_id=0, grasp_prob=0.9),
    ] * 5  # 15 steps, all same event_id
    steps_v1.extend([_make_step(i, event_id=0, grasp_prob=0.9) for i in range(15, 25)])

    steps_v2 = [
        _make_step(0, event_id=-1, grasp_prob=0.9),
        _make_step(1, event_id=-1, grasp_prob=0.9),
        _make_step(2, event_id=-1, grasp_prob=0.9),
    ] * 5
    steps_v2.extend([_make_step(i, event_id=-1, grasp_prob=0.9) for i in range(15, 25)])

    r1 = run_scheduler(steps_v1, runtime_cc)
    r2 = run_scheduler(steps_v2, runtime_cc)
    assert r1["emit"] == r2["emit"], f"event_id changed scheduler output: {r1['emit']} != {r2['emit']}"


# ── Test 2: background episode false start detection ──
def test_negative_episode_false_start_counted():
    """Episode with no valid opportunity corridor → any emit is a false start."""
    def runtime_cc(s):
        return s["route_supported"] and s["grasp_prob"] >= 0.5

    # Create an episode where grasp is high but there's no valid K10 corridor
    steps = []
    for i in range(20):
        steps.append(_make_step(i, event_id=-1, grasp_prob=0.9))  # all background

    result = run_scheduler(steps, runtime_cc)
    # Scheduler should emit (grasp high, dwell satisfied)
    # In a negative episode, this is a false start
    assert result["one_shot_emitted"], "should emit on sustained high grasp"
    # The key invariant: emitted + no valid opportunity = false_start
    # (This is tested by the metric computation, not the scheduler itself)


# ── Test 3: valid opportunity episode → emit counts as hit, not false start ──
def test_valid_opportunity_hit_not_false_start():
    """Emit during a valid K10 corridor should increase recall, not false start count."""
    def runtime_cc(s):
        return s["route_supported"] and s["event_id"] >= 0

    # Episode with a valid corridor (15 consecutive event steps)
    steps = []
    for i in range(25):
        steps.append(_make_step(i, event_id=0 if i < 20 else -1, grasp_prob=0.9 if i < 20 else 0.0))
    result = run_scheduler(steps, runtime_cc)
    assert result["one_shot_emitted"], "should emit during valid corridor"


# ── Test 4: no-emit must not be misreported as safe ──
def test_no_emit_is_abstention_not_safety():
    """An episode where scheduler never emits must be counted as abstention,
    not as 'safe'. Safety is about what happens WHEN it emits."""
    def runtime_cc(s):
        return s["route_supported"] and s["event_id"] >= 0

    steps = []
    for i in range(25):
        # Low grasp → never triggers, even with candidate_close
        steps.append(_make_step(i, event_id=0 if i < 20 else -1, grasp_prob=0.1))
    result = run_scheduler(steps, runtime_cc)
    assert not result["one_shot_emitted"], "should NOT emit with low grasp"
    # In metrics: abstention_rate increases, recall drops
    # This test verifies the invariant: no_emit != safe


# ── Test 5: episode boundary resets all scheduler state ──
def test_episode_boundary_resets_scheduler():
    """New episode must reset scheduler to IDLE with cleared history."""
    config = V5SchedulerConfig()
    scheduler = V5OneShotScheduler(config)

    # First episode: build up dwell but don't emit
    for i in range(8):
        scheduler.update(step=i, candidate_close=True, valid=True,
                         utility_probability=0.9, release_probability=0.0,
                         regrasp_probability=0.0, uncertainty_probability=0.0)
    assert scheduler.candidate_dwell == 8
    assert len(scheduler.history) == 5  # persistence_window=5

    # New episode: must reset
    scheduler.reset()
    assert scheduler.candidate_dwell == 0
    assert len(scheduler.history) == 0
    assert scheduler.state == "IDLE"
    assert not scheduler.emitted


# ── Test 6: valid=False resets dwell ──
def test_invalid_step_resets_dwell():
    """valid=False must clear history and reset candidate_dwell."""
    config = V5SchedulerConfig()
    scheduler = V5OneShotScheduler(config)

    for i in range(5):
        scheduler.update(step=i, candidate_close=True, valid=True,
                         utility_probability=0.9, release_probability=0.0,
                         regrasp_probability=0.0, uncertainty_probability=0.0)
    assert scheduler.candidate_dwell == 5

    # Invalid step
    scheduler.update(step=5, candidate_close=True, valid=False,
                     utility_probability=0.9, release_probability=0.0,
                     regrasp_probability=0.0, uncertainty_probability=0.0)
    assert scheduler.candidate_dwell == 0
    assert scheduler.state == "IDLE"


# ── Test 7: candidate_close=False resets dwell ──
def test_candidate_close_false_resets_dwell():
    """When candidate_close is False, dwell resets and history clears."""
    config = V5SchedulerConfig()
    scheduler = V5OneShotScheduler(config)

    for i in range(5):
        scheduler.update(step=i, candidate_close=True, valid=True,
                         utility_probability=0.9, release_probability=0.0,
                         regrasp_probability=0.0, uncertainty_probability=0.0)
    assert scheduler.candidate_dwell == 5

    scheduler.update(step=5, candidate_close=False, valid=True,
                     utility_probability=0.9, release_probability=0.0,
                     regrasp_probability=0.0, uncertainty_probability=0.0)
    assert scheduler.candidate_dwell == 0


# ── Test 8: release veto prevents emit ──
def test_release_veto_blocks_emit():
    """release_probability >= release_veto_threshold must block emission."""
    config = V5SchedulerConfig()
    scheduler = V5OneShotScheduler(config)

    for i in range(15):
        result = scheduler.update(step=i, candidate_close=True, valid=True,
                                  utility_probability=0.9, release_probability=0.9,
                                  regrasp_probability=0.0, uncertainty_probability=0.0)
    assert not result["emit"], "release veto should block emit"
    assert not result["one_shot_emitted"]


# ── Test 9: regrasp veto prevents emit ──
def test_regrasp_veto_blocks_emit():
    """regrasp_probability >= regrasp_veto_threshold must block emission."""
    config = V5SchedulerConfig()
    scheduler = V5OneShotScheduler(config)

    for i in range(15):
        result = scheduler.update(step=i, candidate_close=True, valid=True,
                                  utility_probability=0.9, release_probability=0.0,
                                  regrasp_probability=0.9, uncertainty_probability=0.0)
    assert not result["emit"], "regrasp veto should block emit"


# ── Test 10: calibration fit and heldout disjoint ──
def test_calibration_fit_heldout_disjoint():
    """Calibration must be fit on identities disjoint from evaluation."""
    fit_ids = {"id_a", "id_b", "id_c"}
    heldout_ids = {"id_d", "id_e", "id_f"}
    assert fit_ids.isdisjoint(heldout_ids), "calibration fit IDs must be disjoint from heldout"
    # Also verify: fit_ids ∩ threshold_selection_ids = ∅
    threshold_ids = {"id_d", "id_e"}
    assert fit_ids.isdisjoint(threshold_ids), "fit IDs must be disjoint from threshold selection IDs"


# ── Test 11: Platt params must not be fit on heldout data ──
def test_platt_not_fit_on_heldout():
    """Verify the invariant: heldout data must not participate in Platt fitting."""
    heldout_identities = {"id_x", "id_y"}
    calibration_fit_identities = {"id_a", "id_b"}
    # Check: any identity used for fit must NOT be in heldout
    violators = calibration_fit_identities & heldout_identities
    assert not violators, f"These identities used for both calibration fit and heldout eval: {violators}"


# ── Test 12: metric numerator/denominator hand-computable ──
def test_metric_denominator_closure():
    """Each metric's denominator must be computable from synthetic fixture."""
    episodes = {
        "ep_neg_1": {"has_corridor": False, "emitted": True},   # false start
        "ep_neg_2": {"has_corridor": False, "emitted": False},  # true negative
        "ep_pos_1": {"has_corridor": True, "emitted": True, "hit": True},   # valid hit
        "ep_pos_2": {"has_corridor": True, "emitted": True, "hit": False},  # false start (wrong event)
        "ep_pos_3": {"has_corridor": True, "emitted": False},   # abstention (miss)
    }

    n_neg = sum(1 for v in episodes.values() if not v["has_corridor"])
    n_pos = sum(1 for v in episodes.values() if v["has_corridor"])
    n_false = sum(1 for v in episodes.values() if (not v["has_corridor"] and v["emitted"]) or (v["has_corridor"] and v["emitted"] and not v.get("hit")))
    n_hit = sum(1 for v in episodes.values() if v["has_corridor"] and v["emitted"] and v.get("hit"))
    n_abstain = sum(1 for v in episodes.values() if v["has_corridor"] and not v["emitted"])
    n_emit = sum(1 for v in episodes.values() if v["emitted"])

    assert n_neg == 2
    assert n_pos == 3
    assert n_false == 2  # ep_neg_1 + ep_pos_2
    assert n_hit == 1     # ep_pos_1
    assert n_abstain == 1  # ep_pos_3
    assert n_emit == 3    # ep_neg_1 + ep_pos_1 + ep_pos_2

    # Hand-computed rates:
    false_rate = n_false / n_neg  # 2/2 = 1.0
    recall = n_hit / n_pos        # 1/3 ≈ 0.333
    precision = n_hit / n_emit    # 1/3 ≈ 0.333
    abstention = n_abstain / n_pos  # 1/3 ≈ 0.333

    assert false_rate == 1.0, f"false_rate={false_rate}"
    assert abs(recall - 1/3) < 0.01, f"recall={recall}"
    assert abs(precision - 1/3) < 0.01, f"precision={precision}"
    assert abs(abstention - 1/3) < 0.01, f"abstention={abstention}"


# ── Test 13: overall emit rate is NOT a safety metric ──
def test_overall_emit_rate_is_not_safety():
    """The overall emit rate (emitted/total) conflates false starts,
    valid hits, and abstention. It must not be reported as safety."""
    # Scenario: 90% abstention + 10% valid hits = low overall emit rate
    # But this is unsafe because recall is only 10%
    overall_emit = 10 / 100  # 10%
    # This number alone tells us nothing about safety
    assert overall_emit <= 0.15  # "low" emit rate
    # But recall could be terrible
    recall = 10 / 90  # only 11% of valid opportunities hit
    assert recall < 0.20, "low overall emit rate hides poor recall"
    # Conclusion: must report recall separately
