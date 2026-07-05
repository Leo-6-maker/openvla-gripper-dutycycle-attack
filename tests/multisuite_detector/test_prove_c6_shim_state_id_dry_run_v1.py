from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.multisuite_detector import prove_c6_shim_state_id_dry_run_v1 as m

REPO = Path(__file__).resolve().parents[2]
RESET = "b8812e658e1cf6ce99d648dfbb85e5c65aa83d9b11824dad59a0af2a34c1b8cb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_c6_1i(tmp_path: Path, *, handle: str = "state_id:0", status: str = m.INPUT_PASS) -> Path:
    p = tmp_path / "c6_1i.json"
    p.write_text(
        json.dumps(
            {
                "status": status,
                "selected_handle": handle,
                "selected_parent": {
                    "parent_id": "libero_goal/task_01/state_000",
                    "episode_key": "libero_goal/task_01/state_000/clean/attempt_01",
                    "suite": "libero_goal",
                    "task_id": 1,
                    "initial_state_hash": RESET,
                },
            }
        ),
        encoding="utf-8",
    )
    return p


def run_tool(tmp_path: Path, c6: Path, expected: str):
    out = tmp_path / "out"
    args = argparse.Namespace(input_c6_1i_json=str(c6), expected_c6_1i_sha256=expected, output_root=str(out), repo_root=str(REPO), python=sys.executable, git_commit="test", files_changed=[], tests=[])
    return m.run(args), out


def test_parse_state_id():
    assert m.parse_state_id("state_id:0") == 0
    assert m.parse_state_id("episode_idx:0") is None


def test_state_id_dry_run_passes(tmp_path):
    c6 = write_c6_1i(tmp_path)
    rc, out = run_tool(tmp_path, c6, sha256(c6))
    report = load(out / "shim_state_id_static_dry_run_binding.json")
    shim = load(out / "shim_result.json")
    assert rc == 0
    assert report["status"] == m.PASS
    assert report["state_id"] == 0
    assert shim["status"] == m.PASS
    assert shim["state_id"] == 0
    assert shim["state_id_binding"]["binding_mode"] == "DRY_RUN_METADATA_ONLY"
    assert shim["legacy_task_id"] == "libero_goal_open_middle_drawer"
    assert shim["legacy_task_binding"]["legacy_task_id"] == "libero_goal_open_middle_drawer"
    argv = shim["legacy_runner_argv_preview"]
    assert "--task_id" in argv
    assert argv[argv.index("--task_id") + 1] == "libero_goal_open_middle_drawer"


def test_hash_mismatch_holds(tmp_path):
    c6 = write_c6_1i(tmp_path)
    rc, out = run_tool(tmp_path, c6, "0" * 64)
    assert rc != 0
    assert load(out / "shim_state_id_static_dry_run_binding.json")["status"] == "HOLD_C6_1I_HASH_MISMATCH"


def test_non_state_id_handle_holds(tmp_path):
    c6 = write_c6_1i(tmp_path, handle="episode_idx:0")
    rc, out = run_tool(tmp_path, c6, sha256(c6))
    assert rc != 0
    assert load(out / "shim_state_id_static_dry_run_binding.json")["status"] == "HOLD_SELECTED_HANDLE_NOT_STATE_ID"
