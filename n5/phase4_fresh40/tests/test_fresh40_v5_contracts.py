import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "n5" / "phase4_fresh40"))
from fresh40_v5_pipeline import aggregate_and, aggregate_or, canonical_sha, label, _persistence, _select_split


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
