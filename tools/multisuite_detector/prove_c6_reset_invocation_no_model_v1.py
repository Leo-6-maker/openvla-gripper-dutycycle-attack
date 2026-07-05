#!/usr/bin/env python3
"""Prove the C6 reset invocation reaches the shim in dry-run mode only."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

GATE = "C6_1F_RESET_INVOCATION_NO_MODEL_NO_ATTACK_PROOF"
PASS = "PASS_SHIM_DRY_RUN_RESET_ARGS_BOUND"
SHIM = "scripts/c6_run_one_condition_openvla_libero.py"
RESET_ARG = "--initial-state-hash"
BOUNDARY_KEYS = [
    "legacy_runner_execution",
    "OpenVLA",
    "LIBERO",
    "rollout",
    "intervention",
    "attack_condition",
]


class Hold(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_shim_arg(arg: str) -> bool:
    normalized = arg.replace("\\", "/")
    return normalized == SHIM or normalized.endswith("/" + SHIM)


def substitute_placeholders(argv: list[str], output_root: Path) -> list[str]:
    result_json = output_root / "shim_result.json"
    work_dir = output_root / "work"
    out = []
    for arg in argv:
        if arg == "{legacy_result_json}":
            out.append(str(result_json))
        elif arg == "{work_dir}":
            out.append(str(work_dir))
        elif "{" in arg or "}" in arg:
            raise Hold("HOLD_PLACEHOLDER_SUBSTITUTION_FAILED", f"unresolved placeholder in argv: {arg}")
        else:
            out.append(arg)
    if str(result_json) not in out or str(work_dir) not in out:
        raise Hold("HOLD_PLACEHOLDER_SUBSTITUTION_FAILED", "required output placeholders were not present")
    return out


def value_after(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def verify_argv(argv: list[str], selected_parent: dict[str, Any]) -> None:
    if len(argv) < 3:
        raise Hold("HOLD_C6_1E_CONSTRUCTED_ARGV_MISSING", "constructed argv is too short")
    script_args = [arg for arg in argv if is_shim_arg(arg)]
    if script_args != [SHIM] and len(script_args) != 1:
        raise Hold("HOLD_UNAUTHORIZED_LEGACY_RUNNER_EXECUTION_ATTEMPT", "argv does not target exactly the C6 shim")
    if not any(is_shim_arg(arg) for arg in argv):
        raise Hold("HOLD_UNAUTHORIZED_LEGACY_RUNNER_EXECUTION_ATTEMPT", "C6 shim path missing from argv")
    if "--dry-run" not in argv:
        raise Hold("HOLD_DRY_RUN_FLAG_MISSING", "constructed argv lacks --dry-run")
    expected_hash = str(selected_parent.get("initial_state_hash", ""))
    if not expected_hash or value_after(argv, RESET_ARG) != expected_hash:
        raise Hold("HOLD_AUDITED_RESET_ARG_MISSING_FROM_ARGV", "audited initial_state_hash is not bound in argv")


def boundaries_from(obj: dict[str, Any]) -> dict[str, Any]:
    b = obj.get("boundaries")
    if isinstance(b, dict):
        return b
    b = obj.get("boundary")
    if isinstance(b, dict):
        return b
    return {}


def verify_shim_result(path: Path, selected_parent: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise Hold("HOLD_SHIM_RESULT_JSON_MISSING", "shim_result.json was not written")
    obj = read_json(path)
    expected = {
        "parent_id": selected_parent.get("parent_id"),
        "episode_key": selected_parent.get("episode_key"),
        "suite": selected_parent.get("suite"),
        "task_id": selected_parent.get("task_id"),
        "initial_state_hash": selected_parent.get("initial_state_hash"),
    }
    for key, expected_value in expected.items():
        if str(obj.get(key)) != str(expected_value):
            raise Hold("HOLD_SHIM_RESULT_PARENT_MISMATCH", f"{key} mismatch")
    if obj.get("condition") != "CLEAN":
        raise Hold("HOLD_SHIM_RESULT_PARENT_MISMATCH", "condition mismatch")
    if obj.get("status") != PASS:
        raise Hold("HOLD_STATIC_ARTIFACT_INCOMPLETE", f"shim status is {obj.get('status')!r}")
    boundaries = boundaries_from(obj)
    for key in BOUNDARY_KEYS:
        if boundaries.get(key) != "NOT_PERFORMED":
            raise Hold("HOLD_BOUNDARY_VIOLATION", f"{key} boundary is {boundaries.get(key)!r}")
    return obj


def selected_parent_with_hash(c6_1e: dict[str, Any]) -> dict[str, Any]:
    parent = dict(c6_1e.get("selected_parent") or {})
    reset = c6_1e.get("reset_binding") or {}
    parent["initial_state_hash"] = reset.get("value", "")
    return parent


def write_checksums(output_root: Path) -> tuple[str, str]:
    files = [
        "reset_invocation_no_model_no_attack_proof.json",
        "shim_result.json",
        "stdout.txt",
        "stderr.txt",
    ]
    existing = [name for name in files if (output_root / name).exists()]
    sums = output_root / "SHA256SUMS"
    lines = [f"{sha256_file(output_root / name)}  {name}\n" for name in existing]
    sums.write_text("".join(lines), encoding="utf-8")
    sidecar = output_root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(sums), sha256_file(sidecar)


def run_proof(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "stdout.txt"
    stderr_path = output_root / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    observed = sha256_file(args.input_c6_1e_json)
    if observed != args.expected_c6_1e_sha256:
        raise Hold("HOLD_C6_1E_HASH_MISMATCH", "C6_1E artifact hash mismatch")
    c6_1e = read_json(args.input_c6_1e_json)
    if c6_1e.get("status") != "PASS_STATIC_SHIM_ARG_BINDING":
        raise Hold("HOLD_C6_1E_STATUS_NOT_PASS", f"C6_1E status is {c6_1e.get('status')!r}")
    raw_argv = (c6_1e.get("constructed_invocation") or {}).get("argv")
    if not isinstance(raw_argv, list) or not raw_argv:
        raise Hold("HOLD_C6_1E_CONSTRUCTED_ARGV_MISSING", "constructed argv missing")
    selected_parent = selected_parent_with_hash(c6_1e)
    verify_argv([str(a) for a in raw_argv], selected_parent)
    argv = substitute_placeholders([str(a) for a in raw_argv], output_root)
    verify_argv(argv, selected_parent)

    proc = subprocess.run(argv, text=True, capture_output=True, cwd=args.repo_root, check=False)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise Hold("HOLD_SHIM_DRY_RUN_RETURNS_NONZERO", f"shim returned {proc.returncode}")
    shim_path = output_root / "shim_result.json"
    shim = verify_shim_result(shim_path, selected_parent)
    return {
        "gate": GATE,
        "status": PASS,
        "input_c6_1e_json": str(args.input_c6_1e_json),
        "input_c6_1e_json_sha256": observed,
        "expected_c6_1e_json_sha256": args.expected_c6_1e_sha256,
        "selected_parent": selected_parent,
        "executed_command": {"mode": "SHIM_DRY_RUN_ONLY", "argv": argv, "returncode": proc.returncode},
        "shim_result": {"path": str(shim_path), "sha256": sha256_file(shim_path), "status": shim.get("status")},
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "boundaries": {
            "legacy_runner_execution": "NOT_PERFORMED",
            "OpenVLA": "NOT_PERFORMED",
            "LIBERO": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
            "artifact_mutation": "NOT_PERFORMED",
        },
        "files_changed": args.files_changed,
        "git_commit": args.git_commit,
        "tests": args.tests,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-c6-1e-json", required=True)
    p.add_argument("--expected-c6-1e-sha256", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    args = p.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        report = run_proof(args)
        rc = 0
    except Hold as exc:
        report = {
            "gate": GATE,
            "status": exc.status,
            "reason": exc.reason,
            "input_c6_1e_json": str(args.input_c6_1e_json),
            "expected_c6_1e_json_sha256": args.expected_c6_1e_sha256,
            "boundaries": {
                "legacy_runner_execution": "NOT_PERFORMED",
                "OpenVLA": "NOT_PERFORMED",
                "LIBERO": "NOT_PERFORMED",
                "rollout": "NOT_PERFORMED",
                "intervention": "NOT_PERFORMED",
                "attack_condition": "NOT_PERFORMED",
                "artifact_mutation": "NOT_PERFORMED",
            },
            "files_changed": args.files_changed,
            "git_commit": args.git_commit,
            "tests": args.tests,
        }
        rc = 2
    write_json(output_root / "reset_invocation_no_model_no_attack_proof.json", report)
    sums_sha, sidecar_sha = write_checksums(output_root)
    report["SHA256SUMS_sha256"] = sums_sha
    report["SHA256SUMS_sidecar_sha256"] = sidecar_sha
    write_json(output_root / "reset_invocation_no_model_no_attack_proof.json", report)
    write_checksums(output_root)
    print(json.dumps(report, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
