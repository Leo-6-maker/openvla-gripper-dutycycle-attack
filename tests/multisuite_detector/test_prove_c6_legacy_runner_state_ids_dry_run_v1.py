from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.multisuite_detector import prove_c6_legacy_runner_state_ids_dry_run_v1 as m


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_runner(tmp_path: Path) -> Path:
    p = tmp_path / "runner.py"
    p.write_text("import sys\nraise SystemExit(0 if '--dry_run' in sys.argv else 2)\n", encoding="utf-8")
    return p


def write_c6_1l(tmp_path: Path, runner: Path, status=m.INPUT_PASS, state_id=0, python="/usr/bin/python3") -> Path:
    p = tmp_path / "c6_1l.json"
    p.write_text(json.dumps({"status": status, "state_id": state_id, "legacy_runner_argv_preview": [python, str(runner), "--dry_run", "--state_ids", str(state_id)]}), encoding="utf-8")
    return p


def run_tool(tmp_path: Path, c6: Path, expected: str, python=sys.executable):
    out = tmp_path / "out"
    args = argparse.Namespace(input_c6_1l_json=str(c6), expected_c6_1l_sha256=expected, repo_root=str(tmp_path), python=python, output_root=str(out), git_commit="test", files_changed=[], tests=[])
    return m.run(args), out


def test_helpers():
    argv = ["x", "--dry_run", "--state_ids", "0"]
    assert m.has_arg(argv, "--dry_run")
    assert m.has_pair(argv, "--state_ids", 0)
    assert m.effective_argv(["old", "runner.py"], "new")[0] == "new"
    assert m.parse_version_tuple("3.10.12") >= (3, 9)


def test_legacy_dry_run_passes_with_python_override(tmp_path):
    runner = write_runner(tmp_path)
    c6 = write_c6_1l(tmp_path, runner, python="/usr/bin/python3")
    rc, out = run_tool(tmp_path, c6, sha256(c6), python=sys.executable)
    report = load(out / "legacy_runner_state_ids_dry_run_invocation.json")
    assert rc == 0
    assert report["status"] == m.PASS
    assert report["executed_command"]["argv"][0] == sys.executable
    assert report["executed_command"]["preview_argv"][0] == "/usr/bin/python3"


def test_missing_dry_run_holds(tmp_path):
    runner = write_runner(tmp_path)
    c6 = tmp_path / "c6_1l.json"
    c6.write_text(json.dumps({"status": m.INPUT_PASS, "state_id": 0, "legacy_runner_argv_preview": [sys.executable, str(runner), "--state_ids", "0"]}), encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, sha256(c6))
    assert rc != 0
    assert load(out / "legacy_runner_state_ids_dry_run_invocation.json")["status"] == "HOLD_LEGACY_ARGV_NOT_DRY_RUN"


def test_bad_python_override_holds(tmp_path):
    runner = write_runner(tmp_path)
    c6 = write_c6_1l(tmp_path, runner, python="/usr/bin/python3")
    rc, out = run_tool(tmp_path, c6, sha256(c6), python=str(tmp_path / "missing-python"))
    assert rc != 0
    assert load(out / "legacy_runner_state_ids_dry_run_invocation.json")["status"] == "HOLD_LEGACY_PYTHON_VERSION_CHECK_FAILED"
