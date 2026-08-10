from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.detector_v5.select_stage_v_m3_5_diagnostic_manifest_v2 import COUNTERS, SUITES, select


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(root: Path) -> None:
    files = sorted(path for path in root.iterdir() if path.is_file())
    (root / "SHA256SUMS").write_text("".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{_sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def _corridor_rows() -> list[dict]:
    return [{
        "step": step, "clean_record_valid": True, "clean_terminal": False,
        "phase_eligible": True, "clean_only_phase_label": "CONTACT_MANIPULATION",
        "remaining_horizon": 30, "contact_telemetry_valid": True,
        "object_identity": "cube_1", "object_position": [0.0, 0.0, 0.1],
        "eef_position": [0.0, 0.0, 0.11], "object_eef_distance_m": 0.01,
        "object_gripper_contact": True, "object_support_contact": False,
    } for step in range(43)]


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    keys = [f"{suite}/task_00/state_{state:02d}" for suite in SUITES for state in (47, 48)]
    exposure = tmp_path / "exposure.json"
    _write(exposure, {
        "schema": "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4", "status": "PASS",
        "excluded_parent_keys": keys, "branch_results_read": False, "protected_counters": COUNTERS,
    })
    taxonomy = tmp_path / "taxonomy.json"
    _write(taxonomy, {
        "schema": "STAGE_V_M3_5_SELECTION_TAXONOMY_ELIGIBILITY_AUDIT_V1",
        "status": "PASS_WITH_EXPLICIT_INELIGIBLE_PARENTS", "selected_count": 8,
        "eligible_count": 8, "ineligible_parent_keys": [], "branch_results_read": False,
        "protected_counters": COUNTERS,
    })
    coverage = tmp_path / "coverage"
    for key in keys:
        suite, _task, state = key.split("/")
        root = coverage / key.replace("/", "_")
        _write(root / "PARENT_RESULT.json", {
            "schema": "STAGE_V_M3_5_CLEAN_COVERAGE_RESULT_V1", "status": "PASS",
            "coverage_only": True, "parent_atomic": True, "canonical_parent_key": key,
            "suite": suite, "state_index": int(state.removeprefix("state_")),
            "clean_success": True, "protected_counters": COUNTERS,
        })
        _write(root / "CLEAN_TRAJECTORY.json", {
            "schema": "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1", "outcomes_read": False,
            "rows": _corridor_rows(),
        })
        _seal(root)
    return exposure, taxonomy, coverage, keys


def test_selection_v2_is_deterministic_and_uses_only_sealed_clean_evidence(tmp_path: Path) -> None:
    exposure, taxonomy, coverage, keys = _inputs(tmp_path)
    first = select(exposure, taxonomy, [coverage], tmp_path / "selection-a.json")
    second = select(exposure, taxonomy, [coverage], tmp_path / "selection-b.json")
    assert [row["canonical_parent_key"] for row in first["selected_parents"]] == [row["canonical_parent_key"] for row in second["selected_parents"]]
    assert set(row["canonical_parent_key"] for row in first["selected_parents"]) == set(keys)
    assert first["selection_reads"]["counterfactual_outcomes_read"] is False

    parent = next(coverage.rglob("PARENT_RESULT.json")).parent
    (parent / "COUNTERFACTUAL_BRANCHES.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="COUNTERFACTUAL_BRANCH_FILE_FORBIDDEN"):
        select(exposure, taxonomy, [coverage], tmp_path / "selection-c.json")
