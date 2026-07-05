from __future__ import annotations

import json
from pathlib import Path

from tools.multisuite_detector import run_c6_dry_validation_chain_v1 as m


def test_argv_value():
    argv = ["py", "runner.py", "--task_id", "libero_goal_open_middle_drawer", "--state_ids", "0"]
    assert m.argv_value(argv, "--task_id") == "libero_goal_open_middle_drawer"
    assert m.argv_value(argv, "--state_ids") == "0"
    assert m.argv_value(argv, "--missing") == ""


def test_preview_precheck(tmp_path: Path):
    p = tmp_path / "c6_1l.json"
    p.write_text(json.dumps({"legacy_runner_argv_preview": ["py", "runner.py", "--task_id", "libero_goal_open_middle_drawer", "--dry_run", "--state_ids", "0"]}), encoding="utf-8")
    res = m.preview_precheck(p, "libero_goal_open_middle_drawer", "0")
    assert res["task_id_ok"]
    assert res["state_ids_ok"]
    assert res["dry_run_ok"]


def test_preview_precheck_detects_mismatch(tmp_path: Path):
    p = tmp_path / "c6_1l.json"
    p.write_text(json.dumps({"legacy_runner_argv_preview": ["py", "runner.py", "--task_id", "1", "--dry_run", "--state_ids", "0"]}), encoding="utf-8")
    res = m.preview_precheck(p, "libero_goal_open_middle_drawer", "0")
    assert not res["task_id_ok"]
    assert res["state_ids_ok"]
