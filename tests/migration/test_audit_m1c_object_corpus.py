#!/usr/bin/env python3
"""Tests for M1C corpus integrity auditor — duplicate, missing, attack, split leakage."""
import json, tempfile, csv, os
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
sys_path = str(REPO)
import sys; sys.path.insert(0, sys_path)

from scripts.migration.audit_m1c_object_corpus import (
    check_cell_uniqueness, check_file_completeness, check_step_integrity,
    check_safety, check_asset_consistency, check_no_excluded_states,
    cell_key, EXCLUDED_STATES, COMPROMISED_BLIND,
)


def _make_cell(tmpdir, task, state, pool="train", files=None):
    d = Path(tmpdir) / pool / f"task{task}_state{state}"
    d.mkdir(parents=True)
    if files:
        for fname, content in files.items():
            (d / fname).write_text(content)
    return {"task_idx": task, "state_id": state, "pool": pool, "profile": "B0", "_cell_dir": d}


def test_duplicate_detection():
    cells = [
        {"task_idx": 0, "state_id": 3, "profile": "B0"},
        {"task_idx": 0, "state_id": 3, "profile": "B0"},
    ]
    dupes = check_cell_uniqueness(cells, None)
    assert len(dupes) == 1


def test_no_duplicate():
    cells = [
        {"task_idx": 0, "state_id": 3, "profile": "B0"},
        {"task_idx": 0, "state_id": 4, "profile": "B0"},
    ]
    assert check_cell_uniqueness(cells, None) == []


def test_file_completeness_ok():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_cell(tmp, 0, 3, files={
            "step_telemetry.csv": "step,col\n0,a\n",
            "episode_summary.json": '{"ok":true}',
            ".done": '{"exit_code":0}',
        })
        issues = check_file_completeness(c, c["_cell_dir"])
        assert issues == []


def test_file_completeness_missing():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_cell(tmp, 0, 3)
        issues = check_file_completeness(c, c["_cell_dir"])
        assert len(issues) == 3


def test_step_integrity_ok():
    with tempfile.TemporaryDirectory() as tmp:
        tel = "step,corridor_p\n0,0.5\n1,0.6\n2,0.7\n"
        c = _make_cell(tmp, 0, 3, files={"step_telemetry.csv": tel})
        issues = check_step_integrity(c["_cell_dir"])
        assert issues == []


def test_step_integrity_gap():
    with tempfile.TemporaryDirectory() as tmp:
        tel = "step,corridor_p\n0,0.5\n2,0.7\n"
        c = _make_cell(tmp, 0, 3, files={"step_telemetry.csv": tel})
        issues = check_step_integrity(c["_cell_dir"])
        assert len(issues) > 0


def test_safety_attack_frames():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_cell(tmp, 0, 3, files={
            "episode_summary.json": '{"condition":"CLEAN","attack_frames":5}',
        })
        issues = check_safety(c["_cell_dir"])
        assert any("attack_frames" in i for i in issues)


def test_safety_ok():
    with tempfile.TemporaryDirectory() as tmp:
        c = _make_cell(tmp, 0, 3, files={
            "episode_summary.json": '{"condition":"CLEAN","attack_frames":0}',
        })
        issues = check_safety(c["_cell_dir"])
        assert issues == []


def test_excluded_m1b_state():
    cells = [{"task_idx": 0, "state_id": 1}]
    issues = check_no_excluded_states(cells)
    assert len(issues) == 1


def test_excluded_compromised_blind():
    cells = [{"task_idx": 0, "state_id": 40}]
    issues = check_no_excluded_states(cells)
    assert len(issues) == 1


def test_no_excluded():
    cells = [{"task_idx": 0, "state_id": 5}]
    assert check_no_excluded_states(cells) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
