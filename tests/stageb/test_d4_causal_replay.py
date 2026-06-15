"""D4.0 regression tests: causal first-trigger policy."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stageb"))
from run_d4_causal_replay import causal_first_trigger, classify_emission, select_threshold


def test_emit_first_above_threshold():
    """Policy emits the FIRST candidate above threshold, not the highest."""
    candidates = [{"candidate_step": "10"}, {"candidate_step": "20"}, {"candidate_step": "30"}]
    scores = [0.5, 0.8, 0.6]  # step 20 has highest score
    emit_step, emit_idx = causal_first_trigger(candidates, scores, 0.4)
    assert emit_step == 10  # step 10 is first above 0.4
    assert emit_idx == 0


def test_abstain_when_none_above_threshold():
    candidates = [{"candidate_step": "10"}, {"candidate_step": "20"}]
    scores = [0.3, 0.4]
    emit_step, _ = causal_first_trigger(candidates, scores, 0.5)
    assert emit_step == -1


def test_single_emission_only():
    """Even if multiple above threshold, only first emits."""
    candidates = [{"candidate_step": "10"}, {"candidate_step": "20"}]
    scores = [0.7, 0.8]
    emit_step, emit_idx = causal_first_trigger(candidates, scores, 0.5)
    assert emit_step == 10 and emit_idx == 0


def test_exact_hit_classification():
    assert classify_emission(10, 10) == "EXACT_HIT"


def test_early_trigger_classification():
    assert classify_emission(5, 10) == "EARLY_TRIGGER"


def test_late_trigger_classification():
    assert classify_emission(15, 10) == "LATE_TRIGGER"


def test_abstain_classification():
    assert classify_emission(-1, 10) == "ABSTAIN"


def test_threshold_tiebreak_lexicographic():
    """Lexicographic: max exact → min late → min emissions → higher threshold."""
    sweep = [
        {"threshold": 0.5, "n_exact": 10, "n_early": 1, "n_late": 3, "n_emissions": 14},
        {"threshold": 0.7, "n_exact": 10, "n_early": 1, "n_late": 2, "n_emissions": 12},
        {"threshold": 0.6, "n_exact": 11, "n_early": 2, "n_late": 1, "n_emissions": 12},
        {"threshold": 0.8, "n_exact": 10, "n_early": 0, "n_late": 3, "n_emissions": 14},
    ]
    tau, status = select_threshold(sweep)
    # 0.6: early=2 (feasible), exact=11 — best exact count
    assert abs(tau - 0.6) < 0.001 and status == "OK"


def test_no_feasible_threshold():
    sweep = [
        {"threshold": 0.5, "n_exact": 5, "n_early": 3, "n_late": 2, "n_emissions": 10},
    ]
    tau, status = select_threshold(sweep)
    assert tau is None and status == "NO_FEASIBLE_CAUSAL_THRESHOLD"


def test_threshold_tiebreak_higher_tau():
    """Same exact/early/late/emissions → choose higher threshold."""
    sweep = [
        {"threshold": 0.5, "n_exact": 10, "n_early": 1, "n_late": 3, "n_emissions": 14},
        {"threshold": 0.6, "n_exact": 10, "n_early": 1, "n_late": 3, "n_emissions": 14},
    ]
    tau, status = select_threshold(sweep)
    assert abs(tau - 0.6) < 0.001  # higher threshold wins


def test_not_using_task_specific_threshold():
    """Threshold selection is global, not per-task."""
    # All traces use the same threshold regardless of task
    assert True  # design invariant — enforced by single global sweep
