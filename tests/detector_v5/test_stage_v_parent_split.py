from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.freeze_stage_v_parent_split import freeze


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _manifest() -> dict[str, object]:
    rows = []
    for suite in SUITES:
        for task in range(10):
            rows.append({
                "canonical_parent_key": f"{suite}/task_{task:02d}/state_47",
                "suite": suite,
                "task_index": task,
                "state_index": 47,
                "old_artifacts_reused": False,
                "source_artifact_read": False,
            })
    return {"schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V1", "status": "FROZEN", "selected_parents": rows}


def test_split_is_deterministic_parent_grouped_and_balanced(tmp_path: Path) -> None:
    parent = tmp_path / "parents.json"
    _write(parent, _manifest())
    first = freeze(parent, tmp_path / "one", split_salt="STAGE_V_SPLIT_TEST_V1")
    second = freeze(parent, tmp_path / "two", split_salt="STAGE_V_SPLIT_TEST_V1")

    assert [(row["canonical_parent_key"], row["split"]) for row in first["parents"]] == [
        (row["canonical_parent_key"], row["split"]) for row in second["parents"]
    ]
    assert first["split_counts"] == {"TRAIN": 24, "VAL": 8, "TEST": 8}
    assert first["split_counts_by_suite"] == {
        suite: {"TRAIN": 6, "VAL": 2, "TEST": 2} for suite in SUITES
    }
    assert first["vulnerability_outcomes_read"] is False
    assert first["parent_grouped"] is True


def test_split_refuses_nonclosed_or_wrong_size_manifest(tmp_path: Path) -> None:
    parent = tmp_path / "parents.json"
    value = _manifest()
    value["status"] = "RUNNING"
    _write(parent, value)
    with pytest.raises(ValueError, match="not closed"):
        freeze(parent, tmp_path / "out", split_salt="s")
