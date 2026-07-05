from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.multisuite_detector import build_c6_state_index_binding_v1 as m

RESET = "b8812e658e1cf6ce99d648dfbb85e5c65aa83d9b11824dad59a0af2a34c1b8cb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_sha256sum(sum_file: Path) -> None:
    for line in sum_file.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split(maxsplit=1)
        assert sha256(sum_file.parent / rel) == expected


def write_c6_1g(tmp_path: Path, reset: str = RESET) -> Path:
    p = tmp_path / "c6_1g.json"
    p.write_text(
        json.dumps(
            {
                "status": "HOLD_RESET_HASH_NOT_RESOLVABLE_TO_STATE_ARTIFACT",
                "selected_parent": {
                    "parent_id": "libero_goal/task_01/state_000",
                    "episode_key": "libero_goal/task_01/state_000/clean/attempt_01",
                    "suite": "libero_goal",
                    "task_id": 1,
                    "initial_state_hash": reset,
                },
            }
        ),
        encoding="utf-8",
    )
    return p


def run_tool(tmp_path: Path, c6: Path, expected: str, root: Path) -> tuple[int, Path]:
    out = tmp_path / "out"
    args = argparse.Namespace(
        input_c6_1g_json=str(c6),
        expected_c6_1g_sha256=expected,
        search_root=[str(root)],
        output_root=str(out),
        max_files=1000,
        max_file_bytes=1024 * 1024,
        git_commit="test",
        files_changed=[],
        tests=[],
    )
    return m.main_from_args(args), out


def test_gate_name():
    assert m.GATE == "C6_1H_STATE_INDEX_BINDING_AUDIT_BUILD"


def test_hash_mismatch_holds(tmp_path):
    c6 = write_c6_1g(tmp_path)
    rc, out = run_tool(tmp_path, c6, "0" * 64, tmp_path)
    assert rc != 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "HOLD_C6_1G_HASH_MISMATCH"


def test_parent_metadata_binds_existing_state_path(tmp_path):
    c6 = write_c6_1g(tmp_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "index.json").write_text(
        json.dumps({"records": [{"parent_id": "libero_goal/task_01/state_000", "state_path": str(artifact)}]}),
        encoding="utf-8",
    )
    rc, out = run_tool(tmp_path, c6, sha256(c6), meta)
    assert rc == 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "PASS_PARENT_METADATA_BINDS_STATE_PATH"


def test_parent_metadata_binds_episode_idx(tmp_path):
    c6 = write_c6_1g(tmp_path)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "index.csv").write_text(
        "episode_key,episode_idx\nlibero_goal/task_01/state_000/clean/attempt_01,7\n",
        encoding="utf-8",
    )
    rc, out = run_tool(tmp_path, c6, sha256(c6), meta)
    assert rc == 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "PASS_PARENT_METADATA_BINDS_STATE_INDEX"
    assert report["binding_summary"]["unique_handles"] == ["episode_idx:7"]


def test_exact_file_hash_match(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("exact artifact", encoding="utf-8")
    c6 = write_c6_1g(tmp_path, reset=sha256(artifact))
    rc, out = run_tool(tmp_path, c6, sha256(c6), tmp_path)
    assert rc == 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "PASS_STATE_HASH_FILE_SHA256_MATCH"
    assert report["binding_summary"]["exact_file_sha256_match_count"] == 1


def test_missing_path_holds(tmp_path):
    c6 = write_c6_1g(tmp_path)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "index.jsonl").write_text(
        json.dumps({"parent_id": "libero_goal/task_01/state_000", "state_path": "missing.json"}) + "\n",
        encoding="utf-8",
    )
    rc, out = run_tool(tmp_path, c6, sha256(c6), meta)
    assert rc != 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "HOLD_BOUND_STATE_PATH_MISSING"


def test_no_binding_holds(tmp_path):
    c6 = write_c6_1g(tmp_path)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "noise.json").write_text(json.dumps({"initial_state_hash": RESET}), encoding="utf-8")
    rc, out = run_tool(tmp_path, c6, sha256(c6), meta)
    assert rc != 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "HOLD_NO_PARENT_STATE_BINDING"


def test_ambiguous_index_holds(tmp_path):
    c6 = write_c6_1g(tmp_path)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "index.csv").write_text(
        "parent_id,episode_idx\n"
        "libero_goal/task_01/state_000,7\n"
        "libero_goal/task_01/state_000,8\n",
        encoding="utf-8",
    )
    rc, out = run_tool(tmp_path, c6, sha256(c6), meta)
    assert rc != 0
    report = load(out / "state_index_binding_audit.json")
    assert report["status"] == "HOLD_AMBIGUOUS_STATE_BINDING"


def test_checksum_report_consistency(tmp_path):
    c6 = write_c6_1g(tmp_path)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "index.csv").write_text(
        "episode_key,episode_idx\nlibero_goal/task_01/state_000/clean/attempt_01,7\n",
        encoding="utf-8",
    )
    rc, out = run_tool(tmp_path, c6, sha256(c6), meta)
    assert rc == 0
    check_sha256sum(out / "SHA256SUMS")
    check_sha256sum(out / "SHA256SUMS.sha256")
    checksum_report = load(out / "checksum_report.json")
    assert checksum_report["self_referential_checksum_fields"] == "ABSENT_BY_DESIGN"
    for rel, expected in checksum_report["reported_files"].items():
        assert sha256(out / rel) == expected
