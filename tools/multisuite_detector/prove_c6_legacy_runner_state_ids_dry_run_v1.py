#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

GATE = "C6_1M_LEGACY_RUNNER_STATE_IDS_DRY_RUN_INVOCATION"
INPUT_PASS = "PASS_SHIM_DRY_RUN_LEGACY_STATE_IDS_PREVIEW_BOUND"
PASS = "PASS_LEGACY_RUNNER_STATE_IDS_DRY_RUN_RETURNS_ZERO"
OUT_FILES = ["legacy_runner_state_ids_dry_run_invocation.json", "stdout.txt", "stderr.txt", "checksum_report.json"]
MIN_PYTHON = (3, 9)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def has_arg(argv, flag):
    return flag in list(argv or [])


def has_pair(argv, flag, value):
    argv = list(argv or [])
    for i, x in enumerate(argv[:-1]):
        if x == flag and str(argv[i + 1]) == str(value):
            return True
    return False


def value_after(argv, flag):
    argv = list(argv or [])
    for i, x in enumerate(argv[:-1]):
        if x == flag:
            return str(argv[i + 1])
    return ""


def effective_argv(argv, python_override):
    out = list(argv or [])
    if out and str(python_override or "").strip():
        out[0] = str(python_override)
    return out


def interpreter_version(python_exe):
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return "", proc.stderr.strip() or proc.stdout.strip()
    return proc.stdout.strip(), ""


def parse_version_tuple(text):
    try:
        parts = [int(x) for x in str(text).split(".")[:2]]
    except Exception:
        return (0, 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts[:2])


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    observed = sha256_file(args.input_c6_1l_json)
    stdout_path = out / "stdout.txt"
    stderr_path = out / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    preview_argv = []
    argv = []
    rc = None
    state_id = None
    legacy_task_id = ""
    python_version = ""
    python_error = ""
    status = PASS
    if observed != args.expected_c6_1l_sha256:
        status = "HOLD_C6_1L_HASH_MISMATCH"
    else:
        c6 = read_json(args.input_c6_1l_json)
        state_id = c6.get("state_id")
        preview_argv = list(c6.get("legacy_runner_argv_preview") or [])
        argv = effective_argv(preview_argv, args.python)
        legacy_task_id = value_after(argv, "--task_id")
        if c6.get("status") != INPUT_PASS:
            status = "HOLD_C6_1L_STATUS_NOT_PASS"
        elif state_id is None:
            status = "HOLD_STATE_ID_MISSING"
        elif not argv:
            status = "HOLD_LEGACY_ARGV_EMPTY"
        elif not has_arg(argv, "--dry_run"):
            status = "HOLD_LEGACY_ARGV_NOT_DRY_RUN"
        elif not has_pair(argv, "--state_ids", state_id):
            status = "HOLD_LEGACY_ARGV_STATE_IDS_MISSING"
        elif not legacy_task_id:
            status = "HOLD_LEGACY_ARGV_TASK_ID_MISSING"
        elif legacy_task_id.isdigit():
            status = "HOLD_LEGACY_ARGV_TASK_ID_NUMERIC"
        else:
            python_version, python_error = interpreter_version(argv[0])
            if python_error:
                status = "HOLD_LEGACY_PYTHON_VERSION_CHECK_FAILED"
            elif parse_version_tuple(python_version) < MIN_PYTHON:
                status = "HOLD_LEGACY_PYTHON_VERSION_TOO_OLD"
            else:
                proc = subprocess.run(argv, cwd=args.repo_root, text=True, capture_output=True, check=False)
                rc = proc.returncode
                stdout_path.write_text(proc.stdout, encoding="utf-8")
                stderr_path.write_text(proc.stderr, encoding="utf-8")
                if rc != 0:
                    status = "HOLD_LEGACY_DRY_RUN_RETURNS_NONZERO"
    report = {
        "gate": GATE,
        "status": status,
        "input_c6_1l_json_sha256": observed,
        "expected_c6_1l_json_sha256": args.expected_c6_1l_sha256,
        "state_id": state_id,
        "legacy_task_id": legacy_task_id,
        "python_override": str(args.python or ""),
        "python_version": python_version,
        "python_error": python_error,
        "executed_command": {"mode": "LEGACY_RUNNER_DRY_RUN_ONLY", "preview_argv": preview_argv, "argv": argv, "returncode": rc},
        "boundaries": {"legacy_runner_execution": "DRY_RUN_ONLY", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED"},
        "files_changed": args.files_changed,
        "git_commit": args.git_commit,
        "tests": args.tests,
    }
    write_json(out / "legacy_runner_state_ids_dry_run_invocation.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1l-json", required=True)
    p.add_argument("--expected-c6-1l-sha256", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--python", default="", help="Optional Python >=3.9 interpreter override for the legacy dry-run argv[0].")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
