from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.run_stage_v_m4_exact_40x24_plan_only import _validate_population, _validate_protocol


def _protocol() -> dict:
    return json.loads(Path("configs/STAGE_V_M4_EXACT_40X24_PLAN_ONLY_PROTOCOL_V1.json").read_text(encoding="utf-8"))


def test_plan_protocol_freezes_zero_outcome_boundary() -> None:
    _validate_protocol(_protocol())


def test_plan_protocol_rejects_intervention() -> None:
    protocol = _protocol()
    protocol["operation"]["intervention_executed"] = True
    with pytest.raises(RuntimeError, match="PLAN_OPERATION_BOUNDARY_INVALID:intervention_executed"):
        _validate_protocol(protocol)


def test_successor_attempt_registry_accepts_one_frozen_reserve() -> None:
    suites = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    parents = []
    for suite in suites:
        for index in range(10):
            split = ("TEST" if index < 2 else "VAL" if index < 4 else "TRAIN")
            parents.append({"canonical_parent_key": f"{suite}/task_{index:02d}/state_{index:02d}", "suite": suite, "split": split})
    parents[0] = {"canonical_parent_key": "libero_10/task_08/state_28", "suite": "libero_10", "split": "TEST"}
    final = {
        "schema": "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2",
        "status": "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE",
        "formal_m4_authorized": False,
        "outcomes_read": False,
        "parent_count": 40,
        "parents": parents,
        "split_counts": {"TRAIN": 24, "VAL": 8, "TEST": 8},
        "per_suite_split_counts": {suite: {"TRAIN": 6, "VAL": 2, "TEST": 2} for suite in suites},
        "replacement_binding": {"replacement_parent": "libero_10/task_08/state_28"},
        "source_binding": {"science_commit": "3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2", "science_tree": "2492a075e782a112d1e857248956b2647e751039", "runner_sha256": "26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279"},
    }
    split = {"schema": "STAGE_V_M4_FINAL_PARENT_SPLIT_V2", "status": "FROZEN", "formal_m4_authorized": False, "outcomes_read": False, "final_manifest_sha256": "manifest", "counts": {"TRAIN": 24, "VAL": 8, "TEST": 8}, "parents": parents}
    extra = [{"canonical_parent_key": f"extra/{index:02d}", "suite": "extra"} for index in range(16)]
    registry = {
        "schema": "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_SUCCESSOR_V1",
        "status": "FROZEN_EXACT55_PLUS_RESERVE_CORRIDOR_ATTEMPT_FIREWALL",
        "attempted_identity_count": 56,
        "unique_identity_count": 56,
        "duplicate_count": 0,
        "base_attempted_identity_count": 55,
        "appended_reserve_identity_count": 1,
        "historical_attempt_registry_sha256": "7d5cfd1b3396f6af4ecd6f3de9b9d6ef454bb927c14a6619a90f14b27a273968",
        "attempted_identities": parents + extra,
        "appended_reserve_identities": [{"canonical_parent_key": "libero_10/task_08/state_28"}],
        "outcomes_read": False,
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
    }
    result = _validate_population(final, split, registry, final_sha="manifest", split_sha="split")
    assert len(result) == 40
