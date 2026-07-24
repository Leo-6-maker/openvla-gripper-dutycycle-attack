from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.multisuite_detector import validate_c6_legacy_dry_run_artifacts_v1 as m


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_c6(tmp_path: Path, status=m.INPUT_PASS, state_id=0) -> Path:
    p = tmp_path / "c6_1m.json"
    p.write_text(json.dumps({"status": status, "state_id": state_id}), encoding="utf-8")
    return p


def write_artifacts(root: Path):
    root.mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    (root / "progress.json").write_text(json.dumps({"status": "done", "model_checkpoint_path": "dry_run"}), encoding="utf-8")
    (root / "summary.csv").write_text("a\n1\n", encoding="utf-8")
    (root / "step_records.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "episode_records.jsonl").write_text("{}\n", encoding="utf-8")


def run_tool(tmp_path: Path, c6: Path, expected: str, artifact_root: Path):
    out = tmp_path / "out"
    args = argparse.Namespace(input_c6_1m_json=str(c6), expected_c6_1m_sha256=expected, artifact_root=str(artifact_root), output_root=str(out), git_commit="test", files_changed=[], tests=[])
    return m.run(args), out


def test_parse_ok_path():
    assert m.parse_ok_path("[ok] v4 dry run -> /tmp/x\n") == "/tmp/x"


def test_artifacts_validate(tmp_path):
    c6 = write_c6(tmp_path)
    artifact = tmp_path / "artifact"
    write_artifacts(artifact)
    rc, out = run_tool(tmp_path, c6, sha256(c6), artifact)
    assert rc == 0
    report = load(out / "legacy_dry_run_artifact_validation.json")
    assert report["status"] == m.PASS
    assert report["progress_model_checkpoint_path"] == "dry_run"


def test_missing_artifact_holds(tmp_path):
    c6 = write_c6(tmp_path)
    rc, out = run_tool(tmp_path, c6, sha256(c6), tmp_path / "missing")
    assert rc != 0
    assert load(out / "legacy_dry_run_artifact_validation.json")["status"] == "HOLD_LEGACY_DRY_RUN_ARTIFACT_ROOT_NOT_FOUND"


def test_empty_records_hold(tmp_path):
    c6 = write_c6(tmp_path)
    artifact = tmp_path / "artifact"
    write_artifacts(artifact)
    (artifact / "step_records.jsonl").write_text("", encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, sha256(c6), artifact)
    assert rc != 0
    assert load(out / "legacy_dry_run_artifact_validation.json")["status"] == "HOLD_LEGACY_DRY_RUN_RECORDS_EMPTY"
