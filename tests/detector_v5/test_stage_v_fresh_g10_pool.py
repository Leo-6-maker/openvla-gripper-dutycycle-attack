from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5.build_stage_v_clean_attempt_exclusion import build as build_attempt_exclusion
from scripts.detector_v5.build_stage_v_g10_candidate_pool import build as build_pool
from scripts.detector_v5.stage_v_dynamic_common import sha256_file


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_attempt_exclusion_collects_clean_results_without_marking_exposure(tmp_path: Path) -> None:
    root = tmp_path / "old"
    write_json(root / "qualification" / "QUALIFICATION_ROW.json", {
        "canonical_parent_key": "libero_goal/task_00/state_47",
    })
    write_json(root / "qualification" / "CONTROL_RESULT.json", {
        "canonical_parent_key": "libero_object/task_00/state_47",
    })
    output = tmp_path / "attempts.json"
    report = build_attempt_exclusion([root], output, source_commit="c", source_tree="t")
    assert report["status"] == "PASS"
    assert report["excluded_parent_keys"] == [
        "libero_goal/task_00/state_47",
        "libero_object/task_00/state_47",
    ]


def test_g10_pool_is_deterministic_and_excludes_both_ledgers(tmp_path: Path) -> None:
    g10 = tmp_path / "g10.json"
    identities = [
        f"{suite}/task_{task:02d}/state_{state:02d}"
        for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")
        for task in range(2)
        for state in range(20, 24)
    ]
    write_json(g10, {"identities": identities})
    exposure = tmp_path / "exposure.json"
    write_json(exposure, {
        "schema": "STAGE_V_EXPOSURE_MANIFEST_V2", "status": "PASS",
        "excluded_parent_keys": ["libero_goal/task_00/state_20"],
    })
    attempts = tmp_path / "attempts.json"
    write_json(attempts, {
        "schema": "STAGE_V_CLEAN_QUALIFICATION_ATTEMPT_EXCLUSION_V1", "status": "PASS",
        "excluded_parent_keys": ["libero_object/task_00/state_20"],
    })
    first = tmp_path / "pool1.json"
    second = tmp_path / "pool2.json"
    one = build_pool(g10, exposure, attempts, first, salt="s", candidates_per_suite=2)
    two = build_pool(g10, exposure, attempts, second, salt="s", candidates_per_suite=2)
    assert one["candidates"] == two["candidates"]
    assert len(one["candidates"]) == 8
    keys = {row["canonical_parent_key"] for row in one["candidates"]}
    assert "libero_goal/task_00/state_20" not in keys
    assert "libero_object/task_00/state_20" not in keys
    assert one["gates"]["attack_rollouts"] == 0
    assert sha256_file(first) == sha256_file(second)


def test_g10_pool_refuses_duplicate_identity(tmp_path: Path) -> None:
    g10 = tmp_path / "g10.json"
    write_json(g10, {"identities": ["libero_10/task_00/state_20", "libero_10/task_00/state_20"]})
    exposure = tmp_path / "exposure.json"
    write_json(exposure, {"schema": "STAGE_V_EXPOSURE_MANIFEST_V2", "status": "PASS", "excluded_parent_keys": []})
    attempts = tmp_path / "attempts.json"
    write_json(attempts, {"schema": "STAGE_V_CLEAN_QUALIFICATION_ATTEMPT_EXCLUSION_V1", "status": "PASS", "excluded_parent_keys": []})
    with pytest.raises(ValueError, match="duplicate G10 identity"):
        build_pool(g10, exposure, attempts, tmp_path / "pool.json", salt="s", candidates_per_suite=1)
