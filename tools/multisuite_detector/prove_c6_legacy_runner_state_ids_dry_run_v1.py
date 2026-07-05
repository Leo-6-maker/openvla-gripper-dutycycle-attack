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
    argv = []
    rc = None
    state_id = None
    status = PASS
    if observed != args.expected_c6_1l_sha256:
        status = "HOLD_C6_1L_HASH_MISMATCH"
    else:
        c6 = read_json(args.input_c6_1l_json)
        state_id = c6.get("state_id")
        argv = list(c6.get("legacy_runner_argv_preview") or [])
        if c6.get("status") != INPUT_PASS:
            status = "HOLD_C6_1L_STATUS_NOT_PASS"
        elif state_id is None:
            status = "HOLD_STATE_ID_MISSING"
        elif not has_arg(argv, "--dry_run"):
            status = "HOLD_LEGACY_ARGV_NOT_DRY_RUN"
        elif not has_pair(argv, "--state_ids", state_id):
            status = "HOLD_LEGACY_ARGV_STATE_IDS_MISSING"
        else:
            proc = subprocess.run(argv, cwd=args.repo_root, text=True, capture_output=True, check=False)
            rc = proc.returncode
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            if rc != 0:
                status = "HOLD_LEGACY_DRY_RUN_RETURNS_NONZERO"
    report = {"gate": GATE, "status": status, "input_c6_1l_json_sha256": observed, "expected_c6_1l_json_sha256": args.expected_c6_1l_sha256, "state_id": state_id, "executed_command": {"mode": "LEGACY_RUNNER_DRY_RUN_ONLY", "argv": argv, "returncode": rc}, "boundaries": {"legacy_runner_execution": "DRY_RUN_ONLY", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED"}, "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "legacy_runner_state_ids_dry_run_invocation.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1l-json", required=True)
    p.add_argument("--expected-c6-1l-sha256", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
