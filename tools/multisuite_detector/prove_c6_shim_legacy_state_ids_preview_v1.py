#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

GATE = "C6_1L_SHIM_LEGACY_STATE_IDS_ARGV_PREVIEW_BINDING"
INPUT_PASS = "PASS_SHIM_DRY_RUN_STATE_ID_BOUND"
PASS = "PASS_SHIM_DRY_RUN_LEGACY_STATE_IDS_PREVIEW_BOUND"
OUT_FILES = ["shim_legacy_state_ids_preview_binding.json", "shim_result.json", "stdout.txt", "stderr.txt", "checksum_report.json"]


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


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def argv_has_pair(argv, key, value):
    for i, x in enumerate(argv[:-1]):
        if x == key and str(argv[i + 1]) == str(value):
            return True
    return False


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stdout.txt").write_text("", encoding="utf-8")
    (out / "stderr.txt").write_text("", encoding="utf-8")
    observed = sha256_file(args.input_c6_1j_json)
    parent = {}
    state_id = None
    shim_obj = {}
    rc = None
    cmd = []
    status = PASS
    if observed != args.expected_c6_1j_sha256:
        status = "HOLD_C6_1J_HASH_MISMATCH"
    else:
        c6 = read_json(args.input_c6_1j_json)
        parent = dict(c6.get("selected_parent") or {})
        state_id = c6.get("state_id")
        if c6.get("status") != INPUT_PASS:
            status = "HOLD_C6_1J_STATUS_NOT_PASS"
        elif state_id is None:
            status = "HOLD_STATE_ID_MISSING"
        else:
            result = out / "shim_result.json"
            cmd = [args.python, "scripts/c6_run_one_condition_openvla_libero.py", "--parent-id", str(parent["parent_id"]), "--episode-key", str(parent["episode_key"]), "--suite", str(parent["suite"]), "--task-id", str(parent["task_id"]), "--condition", "CLEAN", "--output-json", str(result), "--work-dir", str(out / "work"), "--initial-state-hash", str(parent["initial_state_hash"]), "--state-id", str(state_id), "--dry-run"]
            proc = subprocess.run(cmd, cwd=args.repo_root, text=True, capture_output=True, check=False)
            rc = proc.returncode
            (out / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
            if rc != 0 or not result.exists():
                status = "HOLD_SHIM_DRY_RUN_FAILED"
            else:
                shim_obj = read_json(result)
                preview = shim_obj.get("legacy_runner_argv_preview") or []
                if shim_obj.get("status") != INPUT_PASS:
                    status = "HOLD_SHIM_STATUS_NOT_STATE_ID_BOUND"
                elif shim_obj.get("legacy_runner_argv_preview_mode") != "NOT_EXECUTED_DRY_RUN_METADATA_ONLY":
                    status = "HOLD_PREVIEW_MODE_MISMATCH"
                elif not argv_has_pair(preview, "--state_ids", state_id):
                    status = "HOLD_LEGACY_STATE_IDS_PREVIEW_MISSING"
    report = {"gate": GATE, "status": status, "input_c6_1j_json_sha256": observed, "expected_c6_1j_json_sha256": args.expected_c6_1j_sha256, "selected_parent": parent, "state_id": state_id, "executed_command": {"mode": "SHIM_DRY_RUN_ONLY", "argv": cmd, "returncode": rc}, "legacy_runner_argv_preview": shim_obj.get("legacy_runner_argv_preview"), "boundaries": {"legacy_runner_execution": "NOT_PERFORMED", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED"}, "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "shim_legacy_state_ids_preview_binding.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1j-json", required=True)
    p.add_argument("--expected-c6-1j-sha256", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--python", default="python3")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
