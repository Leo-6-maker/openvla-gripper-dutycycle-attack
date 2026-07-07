from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.multisuite_detector import audit_c6_state_id_source_v1 as m


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_c6(tmp_path: Path, status=m.INPUT_PASS, state_id=0) -> Path:
    p = tmp_path / "c6_1j.json"
    p.write_text(json.dumps({"status": status, "state_id": state_id, "selected_parent": {"parent_id": "p"}}), encoding="utf-8")
    return p


def run_tool(tmp_path: Path, c6: Path, source: Path, expected: str):
    out = tmp_path / "out"
    args = argparse.Namespace(input_c6_1j_json=str(c6), expected_c6_1j_sha256=expected, source_file=str(source), repo_root=str(tmp_path), output_root=str(out), git_commit="test", files_changed=[], tests=[])
    return m.run(args), out


def test_hash_mismatch(tmp_path):
    c6 = write_c6(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("env.reset()\n", encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, source, "0" * 64)
    assert rc != 0
    assert load(out / "state_id_source_static_audit.json")["status"] == "HOLD_C6_1J_HASH_MISMATCH"


def test_direct_state_arg(tmp_path):
    c6 = write_c6(tmp_path)
    source = tmp_path / "source.py"
    source.write_text('ap.add_argument("--state-id")\nenv.reset()\n', encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, source, sha256(c6))
    assert rc == 0
    report = load(out / "state_id_source_static_audit.json")
    assert report["status"] == m.PASS_DIRECT
    assert report["accepted_state_flags"] == ["--state-id"]


def test_direct_legacy_state_ids_arg(tmp_path):
    c6 = write_c6(tmp_path)
    source = tmp_path / "source.py"
    source.write_text('ap.add_argument("--state_ids")\nstate_ids=args.state_ids\nenv.set_init_state(init_states[int(sid)])\n', encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, source, sha256(c6))
    assert rc == 0
    report = load(out / "state_id_source_static_audit.json")
    assert report["status"] == m.PASS_DIRECT
    assert report["accepted_state_flags"] == ["--state_ids"]


def test_patchable_anchor(tmp_path):
    c6 = write_c6(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("env.reset()\n", encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, source, sha256(c6))
    assert rc == 0
    assert load(out / "state_id_source_static_audit.json")["status"] == m.PASS_PATCHABLE


def test_no_anchor_holds(tmp_path):
    c6 = write_c6(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, source, sha256(c6))
    assert rc != 0
    assert load(out / "state_id_source_static_audit.json")["status"] == "HOLD_NO_STATE_RESET_SOURCE_ANCHOR"
