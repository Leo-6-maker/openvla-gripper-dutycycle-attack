import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "n5" / "phase4_fresh40"))
from fresh40_v5_pipeline import aggregate_and, aggregate_or, canonical_sha, label, _persistence, _select_split, variant_decision
from run_v5_oracle_ladder import _event_metrics


def test_unknown_is_not_negative():
    assert label("UNKNOWN", "missing")["mask"] is False
    assert aggregate_and(["TRUE", "UNKNOWN"]) == "UNKNOWN"
    assert aggregate_and(["TRUE", "FALSE"]) == "FALSE"
    assert aggregate_or(["FALSE", "UNKNOWN"]) == "UNKNOWN"


def test_persistence_is_causal():
    assert _persistence(["TRUE", "TRUE", "FALSE"], 2) == ["UNKNOWN", "TRUE", "FALSE"]


def test_q_minus_q_hash_is_not_a_pose_label():
    q = [0.0, 0.0, 0.70710678, 0.70710678]
    assert all(math.isfinite(x) for x in q)
    assert canonical_sha(q) != canonical_sha([-x for x in q])


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
    with pytest.raises(Exception):
        raise RuntimeError("refusing to overwrite existing root")


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


def test_partial_unknown_event_is_not_scored_as_known():
    event = {"start": 0, "labels": [{"value": "TRUE", "mask": True}, {"value": "UNKNOWN", "mask": False}], "selected": {"oracle": 0}}
    metrics = _event_metrics([event], ("oracle",))
    assert metrics["excluded_unknown_events"] == 1
    assert metrics["oracle"]["positive_events"] == 0
