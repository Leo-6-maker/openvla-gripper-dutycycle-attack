from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5.build_stage_v_clean_attempt_union_v2 import build as build_clean_union
from scripts.detector_v5.build_stage_v_exposure_union_v4 import build as build_exposure_union
from scripts.detector_v5.select_stage_v_m3_5_diagnostic_manifest_v1 import select as select_diagnostic


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_exposure_v4_deduplicates_manifest_identities_and_seals(tmp_path: Path) -> None:
    source = tmp_path / "v3.json"
    write_json(source, {
        "schema": "STAGE_V_EXPOSURE_MANIFEST_V2",
        "status": "PASS",
        "excluded_parent_keys": ["libero_goal/task_00/state_20", "libero_goal/task_00/state_21"],
        "attempted_parent_keys": ["libero_goal/task_00/state_20"],
    })
    output = tmp_path / "v4.json"
    report = build_exposure_union([source], output, source_commit="c", source_tree="t")
    assert report["status"] == "PASS"
    assert report["parent_count"] == 2
    assert report["branch_results_read"] is False
    assert report["protected_counters"]["protected_reads"] == 0
    assert output.with_name(output.name + ".sha256").is_file()


def test_exposure_v4_requires_admissible_source(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    write_json(source, {"schema": "X", "status": "FAIL", "excluded_parent_keys": ["libero_goal/task_00/state_20"]})
    with pytest.raises(ValueError, match="not admissible"):
        build_exposure_union([source], tmp_path / "v4.json")


def test_clean_v2_unions_prior_and_v6_without_overlap(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    v6 = tmp_path / "v6.json"
    write_json(prior, {
        "schema": "STAGE_V_CLEAN_QUALIFICATION_ATTEMPT_EXCLUSION_V1",
        "status": "PASS",
        "excluded_parent_keys": ["libero_goal/task_00/state_20", "libero_goal/task_00/state_21"],
    })
    write_json(v6, {
        "schema": "STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_CANDIDATE_MANIFEST_V1",
        "status": "FROZEN",
        "candidates": [{"canonical_parent_key": "libero_goal/task_00/state_22"}],
    })
    report = build_clean_union([prior, v6], tmp_path / "v2.json", source_commit="c", source_tree="t")
    assert report["status"] == "PASS"
    assert report["parent_count"] == 3
    assert report["overlap_count"] == 0
    assert {item["role"] for item in report["source_manifests"]} == {"prior_clean_control_attempts", "v6_clean_qualification_attempts"}


def test_clean_v2_refuses_single_source(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_json(source, {"schema": "X", "status": "PASS", "excluded_parent_keys": ["libero_goal/task_00/state_20"]})
    with pytest.raises(ValueError, match="at least prior"):
        build_clean_union([source], tmp_path / "v2.json")


def test_m35_selection_is_outcome_blind_and_covers_all_suites(tmp_path: Path) -> None:
    source = tmp_path / "v4.json"
    keys = [f"{suite}/task_00/state_{20 + i}" for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial") for i in range(3)]
    write_json(source, {"schema": "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4", "status": "PASS", "excluded_parent_keys": keys})
    output = tmp_path / "diagnostic.json"
    report = select_diagnostic(source, output, per_suite=2, salt="test")
    assert report["status"] == "FROZEN_FOR_VALIDATION"
    assert report["selected_count"] == 8
    assert report["selected_counts_by_suite"] == {suite: 2 for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")}
    assert report["selection_reads"]["outcomes_read"] is False
    assert report["runtime_authorized"] is False
