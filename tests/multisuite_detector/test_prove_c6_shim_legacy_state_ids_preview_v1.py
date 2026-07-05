from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.multisuite_detector import prove_c6_shim_legacy_state_ids_preview_v1 as m

REPO = Path(__file__).resolve().parents[2]
RESET = "b8812e658e1cf6ce99d648dfbb85e5c65aa83d9b11824dad59a0af2a34c1b8cb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_c6_1j(tmp_path: Path, status=m.INPUT_PASS, state_id=0) -> Path:
    p = tmp_path / "c6_1j.json"
    p.write_text(json.dumps({"status": status, "state_id": state_id, "selected_parent": {"parent_id": "libero_goal/task_01/state_000", "episode_key": "libero_goal/task_01/state_000/clean/attempt_01", "suite": "libero_goal", "task_id": 1, "initial_state_hash": RESET}}), encoding="utf-8")
    return p


def run_tool(tmp_path: Path, c6: Path, expected: str):
    out = tmp_path / "out"
    args = argparse.Namespace(input_c6_1j_json=str(c6), expected_c6_1j_sha256=expected, repo_root=str(REPO), python=sys.executable, output_root=str(out), git_commit="test", files_changed=[], tests=[])
    return m.run(args), out


def test_argv_has_pair():
    assert m.argv_has_pair(["--state_ids", "0"], "--state_ids", 0)
    assert not m.argv_has_pair(["--state_ids", "1"], "--state_ids", 0)


def test_preview_passes(tmp_path):
    c6 = write_c6_1j(tmp_path)
    rc, out = run_tool(tmp_path, c6, sha256(c6))
    report = load(out / "shim_legacy_state_ids_preview_binding.json")
    assert rc == 0
    assert report["status"] == m.PASS
    argv = report["legacy_runner_argv_preview"]
    assert "--state_ids" in argv
    assert "--task_id" in argv
    assert argv[argv.index("--task_id") + 1] == "libero_goal_open_middle_drawer"


def test_hash_mismatch_holds(tmp_path):
    c6 = write_c6_1j(tmp_path)
    rc, out = run_tool(tmp_path, c6, "0" * 64)
    assert rc != 0
    assert load(out / "shim_legacy_state_ids_preview_binding.json")["status"] == "HOLD_C6_1J_HASH_MISMATCH"
