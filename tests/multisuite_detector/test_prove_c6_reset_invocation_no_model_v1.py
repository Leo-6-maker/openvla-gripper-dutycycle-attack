from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SHIM = REPO / "scripts" / "c6_run_one_condition_openvla_libero.py"
PROOF = REPO / "tools" / "multisuite_detector" / "prove_c6_reset_invocation_no_model_v1.py"
RESET = "b8812e658e1cf6ce99d648dfbb85e5c65aa83d9b11824dad59a0af2a34c1b8cb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def shim_cmd(tmp_path: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(SHIM),
        "--parent-id",
        "libero_goal/task_01/state_000",
        "--episode-key",
        "libero_goal/task_01/state_000/clean/attempt_01",
        "--suite",
        "libero_goal",
        "--task-id",
        "1",
        "--condition",
        "CLEAN",
        "--output-json",
        str(tmp_path / "shim.json"),
        "--work-dir",
        str(tmp_path / "work"),
        *extra,
    ]


def write_c6_1e(tmp_path: Path, *, dry_run: bool = True) -> Path:
    argv = [
        sys.executable,
        "scripts/c6_run_one_condition_openvla_libero.py",
        "--parent-id",
        "libero_goal/task_01/state_000",
        "--episode-key",
        "libero_goal/task_01/state_000/clean/attempt_01",
        "--suite",
        "libero_goal",
        "--task-id",
        "1",
        "--condition",
        "CLEAN",
        "--output-json",
        "{legacy_result_json}",
        "--work-dir",
        "{work_dir}",
        "--initial-state-hash",
        RESET,
    ]
    if dry_run:
        argv.append("--dry-run")
    path = tmp_path / "c6_1e.json"
    path.write_text(
        json.dumps(
            {
                "status": "PASS_STATIC_SHIM_ARG_BINDING",
                "selected_parent": {
                    "parent_id": "libero_goal/task_01/state_000",
                    "episode_key": "libero_goal/task_01/state_000/clean/attempt_01",
                    "suite": "libero_goal",
                    "task_id": "1",
                },
                "reset_binding": {"field": "initial_state_hash", "value": RESET},
                "constructed_invocation": {"argv": argv},
            }
        ),
        encoding="utf-8",
    )
    return path


def run_proof(tmp_path: Path, c6_1e: Path, expected_sha: str):
    out = tmp_path / "proof"
    return subprocess.run(
        [
            sys.executable,
            str(PROOF),
            "--input-c6-1e-json",
            str(c6_1e),
            "--expected-c6-1e-sha256",
            expected_sha,
            "--output-root",
            str(out),
            "--repo-root",
            str(REPO),
            "--git-commit",
            "test",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    ), out


def test_shim_dry_run_success(tmp_path):
    proc = subprocess.run(shim_cmd(tmp_path, "--initial-state-hash", RESET, "--dry-run"), text=True, capture_output=True, check=False)
    obj = load(tmp_path / "shim.json")
    assert proc.returncode == 0
    assert obj["status"] == "PASS_SHIM_DRY_RUN_RESET_ARGS_BOUND"
    assert obj["initial_state_hash"] == RESET
    assert obj["boundaries"]["OpenVLA"] == "NOT_PERFORMED"
    assert obj["boundaries"]["LIBERO"] == "NOT_PERFORMED"


def test_shim_dry_run_missing_initial_state_hash(tmp_path):
    proc = subprocess.run(shim_cmd(tmp_path, "--dry-run"), text=True, capture_output=True, check=False)
    obj = load(tmp_path / "shim.json")
    assert proc.returncode != 0
    assert obj["status"] == "HOLD_RESET_FIELD_MISSING"


def test_shim_without_dry_run_still_fail_closed(tmp_path):
    proc = subprocess.run(shim_cmd(tmp_path, "--initial-state-hash", RESET), text=True, capture_output=True, check=False)
    obj = load(tmp_path / "shim.json")
    assert proc.returncode == 2
    assert obj["status"] == "HOLD_PARENT_RESET_UNBOUND"


def test_c6_1f_uses_c6_1e_constructed_argv(tmp_path):
    c6_1e = write_c6_1e(tmp_path)
    proc, out = run_proof(tmp_path, c6_1e, sha256(c6_1e))
    proof = load(out / "reset_invocation_no_model_no_attack_proof.json")
    shim = load(out / "shim_result.json")
    assert proc.returncode == 0
    assert proof["status"] == "PASS_SHIM_DRY_RUN_RESET_ARGS_BOUND"
    assert proof["executed_command"]["returncode"] == 0
    assert "{legacy_result_json}" not in proof["executed_command"]["argv"]
    assert shim["initial_state_hash"] == RESET
    assert (out / "SHA256SUMS").exists()
    assert (out / "SHA256SUMS.sha256").exists()


def test_c6_1f_rejects_hash_mismatch(tmp_path):
    c6_1e = write_c6_1e(tmp_path)
    proc, out = run_proof(tmp_path, c6_1e, "0" * 64)
    proof = load(out / "reset_invocation_no_model_no_attack_proof.json")
    assert proc.returncode != 0
    assert proof["status"] == "HOLD_C6_1E_HASH_MISMATCH"


def test_c6_1f_rejects_missing_dry_run(tmp_path):
    c6_1e = write_c6_1e(tmp_path, dry_run=False)
    proc, out = run_proof(tmp_path, c6_1e, sha256(c6_1e))
    proof = load(out / "reset_invocation_no_model_no_attack_proof.json")
    assert proc.returncode != 0
    assert proof["status"] == "HOLD_DRY_RUN_FLAG_MISSING"
