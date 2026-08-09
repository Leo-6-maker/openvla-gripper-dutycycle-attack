from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.monitoring.audit_stage_v_live_parents import triage
from scripts.monitoring.audit_stage_v_closure import parent_progress
from scripts.monitoring.inventory_stage_v_roots import inventory


COMMIT = "b300e79bb0e6e754a9d384f8ea1b75034bd1d4b4"
TREE = "96881b4d53f901870dd53ede39d051c0a4c83e34"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def make_parent(root: Path, key: str, *, status: str = "FAIL", branch_count: int = 0, probe_count: int = 1) -> Path:
    parent = root / key
    parent.mkdir(parents=True, exist_ok=True)
    write_json(
        parent / "PARENT_RESULT.json",
        {
            "canonical_parent_key": key,
            "status": status,
            "clean_success": status == "PASS",
            "exact_snapshot_replay": True,
            "current_source_commit": COMMIT,
            "current_source_tree": TREE,
            "current_source_status": "",
            "probe_count": probe_count,
            "branch_count": branch_count,
        },
    )
    (parent / "COUNTERFACTUAL_BRANCHES.jsonl").write_text("{}\n" * branch_count, encoding="utf-8")
    return parent


def make_manifest(path: Path, keys: list[str]) -> None:
    write_json(path, {"selected_parents": [{"canonical_parent_key": key} for key in keys]})


def test_live_parent_triage_distinguishes_active_recent_and_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    keys = ["libero_goal/task_00/state_48", "libero_goal/task_00/state_49", "libero_goal/task_00/state_50"]
    manifest = tmp_path / "manifest.json"
    make_manifest(manifest, keys)
    make_parent(root, keys[0], status="FAIL")
    make_parent(root, keys[1], status="FAIL")
    make_parent(root, keys[2], status="PASS", branch_count=3)
    old = time.time() - 1200
    os.utime(root / keys[1] / "PARENT_RESULT.json", (old, old))
    os.utime(root / keys[1] / "COUNTERFACTUAL_BRANCHES.jsonl", (old, old))
    monkeypatch.setattr(
        "scripts.monitoring.audit_stage_v_live_parents._worker_records",
        lambda _: [{"pid": 11, "gpu": 1, "alive": True, "command": keys[0], "exit_code": None}],
    )
    report = triage(root, manifest, expected_source_commit=COMMIT, expected_source_tree=TREE, only_nonpass=False)
    states = {item["canonical_parent_key"]: item["state"] for item in report["parents"]}
    assert states[keys[0]] == "RUNNING_WRITING"
    assert states[keys[1]] == "TERMINAL_PRODUCER_FAIL"
    assert states[keys[2]] == "BRANCH_COMPLETE_PENDING_AUDIT"


def test_root_inventory_reconciles_six_aborted_roots(tmp_path: Path) -> None:
    active = tmp_path / "STAGE_V_COUNTERFACTUAL_MAP_b300e79b_20260806T005817Z"
    active.mkdir(parents=True)
    write_json(active / "RUN_MANIFEST.json", {"source_commit": COMMIT, "source_tree": TREE})
    for index in range(6):
        root = tmp_path / f"STAGE_V_COUNTERFACTUAL_MAP_b300e79b_20260806T00{index + 1:02d}17Z"
        root.mkdir()
        write_json(root / "ABORTED_INCOMPLETE.json", {"status": "ABORTED_INCOMPLETE", "reason": "control_plane" if index < 4 else "engineering"})
        write_json(root / "RUN_MANIFEST.json", {"source_commit": COMMIT, "source_tree": TREE})
    report = inventory(tmp_path, active)
    assert report["aborted_root_count"] == 6
    assert report["aborted_count_reconciliation"]["delta"] == 2
    assert report["active_root_reuse_of_aborted_roots"] is False


def test_closure_progress_exposes_missing_live_and_terminal_states(tmp_path: Path) -> None:
    root = tmp_path / "root"
    keys = ["libero_goal/task_00/state_48", "libero_goal/task_00/state_49", "libero_goal/task_00/state_50"]
    manifest = tmp_path / "manifest.json"
    make_manifest(manifest, keys)
    make_parent(root, keys[0], status="RUNNING")
    make_parent(root, keys[1], status="FAIL")
    progress = parent_progress(root, parent_manifest=manifest, expected_source_commit=COMMIT, expected_source_tree=TREE, full_audit=False)
    assert progress["live_parent_count"] == 1
    assert progress["terminal_invalid_parent_count"] == 1
    assert progress["missing_parent_count"] == 1
