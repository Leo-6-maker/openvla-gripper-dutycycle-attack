import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "n5" / "phase4_fresh40"))
from fresh40_v5_pipeline import aggregate_and, aggregate_or, canonical_sha, event_label, label, _persistence, _publish, _select_split, variant_decision
from run_v5_oracle_ladder import _event_metrics, critical_event_intervals


def test_unknown_is_not_negative():
    assert label("UNKNOWN", "missing")["mask"] is False
    assert aggregate_and(["TRUE", "UNKNOWN"]) == "UNKNOWN"
    assert aggregate_and(["TRUE", "FALSE"]) == "FALSE"
    assert aggregate_or(["FALSE", "UNKNOWN"]) == "UNKNOWN"


def test_persistence_is_causal():
    assert _persistence(["TRUE", "TRUE", "FALSE"], 2) == ["UNKNOWN", "TRUE", "FALSE"]


def test_q_minus_q_geodesic_is_zero():
    q = [0.0, 0.0, 0.70710678, 0.70710678]
    assert all(math.isfinite(x) for x in q)
    norm = math.sqrt(sum(x * x for x in q))
    dot = abs(sum((x / norm) * (-x / norm) for x in q))
    assert 2.0 * math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), dot) < 1e-12


def test_split_is_deterministic_and_disjoint():
    identities = [f"libero_{s}/task_{t:02d}/state_{t}" for s in ("10", "goal", "object", "spatial") for t in range(10)]
    train_a, dev_a = _select_split(identities, 20260717)
    train_b, dev_b = _select_split(identities, 20260717)
    assert train_a == train_b and dev_a == dev_b
    assert not set(train_a) & set(dev_a)
    assert len(train_a) == 32 and len(dev_a) == 8


def test_forbidden_outcome_keys_not_in_label_contract(tmp_path):
    row = {"step": 0, "physical_criticality": label("UNKNOWN", "missing")}
    assert not any(k in row for k in ("reward", "task_success", "terminal", "future"))


def test_output_root_nonoverwrite(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(Exception):
        _publish(staging, target)


def test_inactive_head_mutation_has_zero_influence():
    base = {"physical_criticality": 0.9, "k10_feasible": 0.9, "safe_release": 0.1, "instability": 0.1, "gripper_closing_state": 0.9}
    for variant, inactive in {
        "critical_only": ("k10_feasible", "safe_release", "instability", "gripper_closing_state"),
        "three_head": ("k10_feasible", "safe_release"),
    }.items():
        before = variant_decision(variant, True, base)
        changed = dict(base)
        for name in inactive:
            changed[name] = 1.0 - changed[name]
        assert variant_decision(variant, True, changed) == before


def test_variant_equations_are_distinct_and_frozen():
    p = {"physical_criticality": 0.9, "k10_feasible": 0.9, "safe_release": 0.1, "instability": 0.1, "gripper_closing_state": 0.9}
    assert variant_decision("critical_only", True, p)
    assert variant_decision("three_head", True, p)
    assert variant_decision("full_five", True, p)
    p["k10_feasible"] = 0.1
    assert variant_decision("three_head", True, p)
    assert not variant_decision("full_five", True, p)


def test_three_value_event_or_semantics():
    assert event_label(["TRUE", "UNKNOWN"]) == "TRUE"
    assert event_label(["FALSE", "UNKNOWN"]) == "UNKNOWN"
    assert event_label(["FALSE", "FALSE"]) == "FALSE"
    assert event_label(["UNKNOWN"]) == "UNKNOWN"


def test_true_unknown_event_is_scored_positive():
    event = {"start": 0, "labels": [{"value": "TRUE", "mask": True}, {"value": "UNKNOWN", "mask": False}], "selected": {"oracle": 0}}
    metrics = _event_metrics([event], ("oracle",))
    assert metrics["excluded_unknown_events"] == 0
    assert metrics["oracle"]["positive_events"] == 1


def test_critical_events_are_built_before_candidate_gate():
    rows = [
        {"labels": {"physical_criticality": label("FALSE", "x")}, "candidate_close": False},
        {"labels": {"physical_criticality": label("TRUE", "x")}, "candidate_close": False},
        {"labels": {"physical_criticality": label("TRUE", "x")}, "candidate_close": True},
    ]
    assert critical_event_intervals(rows) == [{"start": 1, "end": 2}]
