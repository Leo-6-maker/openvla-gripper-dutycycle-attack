from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "multisuite_detector" / "audit_c6_legacy_runner_source_reset_adapter_v1.py"
RESET = "b8812e658e1cf6ce99d648dfbb85e5c65aa83d9b11824dad59a0af2a34c1b8cb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_sha256sum(sum_file: Path) -> None:
    for line in sum_file.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split(maxsplit=1)
        assert sha256(sum_file.parent / rel) == expected


def write_c6_1f(tmp_path: Path, *, status: str = "PASS_SHIM_DRY_RUN_RESET_ARGS_BOUND") -> tuple[Path, Path]:
    c6 = tmp_path / "c6_1f.json"
    shim = tmp_path / "shim_result.json"
    parent = {
        "parent_id": "libero_goal/task_01/state_000",
        "episode_key": "libero_goal/task_01/state_000/clean/attempt_01",
        "suite": "libero_goal",
        "task_id": 1,
        "initial_state_hash": RESET,
    }
    c6.write_text(
        json.dumps(
            {
                "gate": "C6_1F_RESET_INVOCATION_NO_MODEL_NO_ATTACK_PROOF",
                "status": status,
                "selected_parent": parent,
                "executed_command": {
                    "mode": "SHIM_DRY_RUN_ONLY",
                    "argv": ["python3", "scripts/c6_run_one_condition_openvla_libero.py", "--initial-state-hash", RESET, "--dry-run"],
                    "returncode": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    shim.write_text(json.dumps({**parent, "condition": "CLEAN", "legacy_runner": "scripts/runner.py"}), encoding="utf-8")
    return c6, shim


def write_runner(tmp_path: Path, text: str) -> Path:
    runner = tmp_path / "scripts" / "runner.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(text, encoding="utf-8")
    return runner


def run_audit(tmp_path: Path, c6: Path, shim: Path, expected: str, *, legacy_runner: str = "scripts/runner.py", search_root: Path | None = None):
    out = tmp_path / "audit_out"
    cmd = [
        sys.executable,
        str(TOOL),
        "--input-c6-1f-json",
        str(c6),
        "--expected-c6-1f-sha256",
        expected,
        "--shim-result-json",
        str(shim),
        "--legacy-runner",
        legacy_runner,
        "--repo-root",
        str(tmp_path),
        "--output-root",
        str(out),
        "--git-commit",
        "test",
    ]
    if search_root is not None:
        cmd += ["--search-root", str(search_root)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=REPO)
    return proc, out


def test_c6_1f_hash_mismatch(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    proc, out = run_audit(tmp_path, c6, shim, "0" * 64)
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode != 0
    assert report["status"] == "HOLD_C6_1F_HASH_MISMATCH"


def test_legacy_runner_source_missing(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6))
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode != 0
    assert report["status"] == "HOLD_LEGACY_RUNNER_SOURCE_NOT_FOUND"


def test_runner_accepts_reset_arg_but_never_uses_it(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(tmp_path, 'import argparse\nap=argparse.ArgumentParser()\nap.add_argument("--initial-state-hash")\n')
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6))
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode != 0
    assert report["status"] == "HOLD_LEGACY_RUNNER_ARG_PARSED_BUT_NOT_USED"


def test_hash_exists_but_no_state_artifact_index(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(tmp_path, "def run(env):\n    env.reset()\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "audit.json").write_text(json.dumps({"initial_state_hash": RESET}, indent=2), encoding="utf-8")
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6), search_root=meta)
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode != 0
    assert report["status"] == "HOLD_RESET_HASH_NOT_RESOLVABLE_TO_STATE_ARTIFACT"
    assert report["reset_resolution"]["initial_state_hash_found"] is True
    assert report["reset_resolution"]["resolves_to_state_artifact"] is False


def test_patchable_adapter_case(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(tmp_path, "def run(env):\n    env.reset()\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "states.csv").write_text(f"initial_state_hash,state_path\n{RESET},/tmp/state.pkl\n", encoding="utf-8")
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6), search_root=meta)
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode == 0
    assert report["status"] == "PASS_STATIC_RESET_ADAPTER_PATCHABLE"
    assert report["adapter_classification"] == "ADAPTER_NEEDED_HASH_TO_STATE_PATH"


def test_multiline_json_state_path_resolves(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(tmp_path, "def run(env):\n    env.reset()\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "state_index.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "parent_id": "libero_goal/task_01/state_000",
                        "initial_state_hash": RESET,
                        "state_path": "/tmp/state.pkl",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6), search_root=meta)
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode == 0
    assert report["status"] == "PASS_STATIC_RESET_ADAPTER_PATCHABLE"
    assert "state_path" in report["reset_resolution"]["candidate_fields"]
    assert any(row["source_kind"] == "json_object" for row in report["reset_resolution"]["candidate_artifacts"])


def test_jsonl_nested_episode_idx_resolves(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(tmp_path, "def run(env):\n    env.reset()\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "state_index.jsonl").write_text(
        json.dumps({"record": {"initial_state_hash": RESET, "episode_idx": 7}}) + "\n",
        encoding="utf-8",
    )
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6), search_root=meta)
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode == 0
    assert report["status"] == "PASS_STATIC_RESET_ADAPTER_PATCHABLE"
    assert "episode_idx" in report["reset_resolution"]["candidate_fields"]


def test_no_adapter_needed_case(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(
        tmp_path,
        'import argparse\nap=argparse.ArgumentParser()\nap.add_argument("--initial-state-hash")\ndef run(env, args):\n    env.reset(args.initial_state_hash)\n',
    )
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6))
    report = load(out / "legacy_runner_source_reset_adapter_static_audit.json")
    assert proc.returncode == 0
    assert report["status"] == "PASS_STATIC_RESET_ADAPTER_NOT_REQUIRED"
    assert report["legacy_runner"]["uses_reset_arg_for_env_reset"] is True


def test_checksum_report_final_consistency(tmp_path):
    c6, shim = write_c6_1f(tmp_path)
    write_runner(tmp_path, "def run(env):\n    env.reset()\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "states.csv").write_text(f"initial_state_hash,state_path\n{RESET},/tmp/state.pkl\n", encoding="utf-8")
    proc, out = run_audit(tmp_path, c6, shim, sha256(c6), search_root=meta)
    assert proc.returncode == 0
    report = load(out / "checksum_report.json")
    assert report["self_referential_checksum_fields"] == "ABSENT_BY_DESIGN"
    for rel, expected in report["reported_files"].items():
        assert sha256(out / rel) == expected
    check_sha256sum(out / "SHA256SUMS")
    check_sha256sum(out / "SHA256SUMS.sha256")
