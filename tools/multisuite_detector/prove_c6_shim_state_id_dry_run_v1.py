#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

GATE = "C6_1J_SHIM_STATE_ID_STATIC_DRY_RUN_BINDING"
PASS = "PASS_SHIM_DRY_RUN_STATE_ID_BOUND"
INPUT_PASS = "PASS_PARENT_SUFFIX_SELECTS_STATE_INDEX_CANDIDATE"
SHIM = "scripts/c6_run_one_condition_openvla_libero.py"
OUT_FILES = ["shim_state_id_static_dry_run_binding.json", "shim_result.json", "stdout.txt", "stderr.txt", "checksum_report.json"]


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


def parse_state_id(handle):
    if not str(handle).startswith("state_id:"):
        return None
    try:
        return int(str(handle).split(":", 1)[1])
    except ValueError:
        return None


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
    (out / "stdout.txt").write_text("", encoding="utf-8")
    (out / "stderr.txt").write_text("", encoding="utf-8")
    observed = sha256_file(args.input_c6_1i_json)
    status = PASS
    reason = ""
    parent = {}
    argv = []
    rc = None
    state_id = None
    shim_obj = {}
    if observed != args.expected_c6_1i_sha256:
        status = "HOLD_C6_1I_HASH_MISMATCH"
    else:
        c6 = read_json(args.input_c6_1i_json)
        parent = dict(c6.get("selected_parent") or {})
        state_id = parse_state_id(c6.get("selected_handle"))
        if c6.get("status") != INPUT_PASS:
            status = "HOLD_C6_1I_STATUS_NOT_PASS"
        elif state_id is None:
            status = "HOLD_SELECTED_HANDLE_NOT_STATE_ID"
        elif not all(parent.get(k) not in (None, "") for k in ["parent_id", "episode_key", "suite", "task_id", "initial_state_hash"]):
            status = "HOLD_SELECTED_PARENT_INCOMPLETE"
        else:
            result = out / "shim_result.json"
            argv = [args.python, SHIM, "--parent-id", str(parent["parent_id"]), "--episode-key", str(parent["episode_key"]), "--suite", str(parent["suite"]), "--task-id", str(parent["task_id"]), "--condition", "CLEAN", "--output-json", str(result), "--work-dir", str(out / "work"), "--initial-state-hash", str(parent["initial_state_hash"]), "--state-id", str(state_id), "--dry-run"]
            proc = subprocess.run(argv, cwd=args.repo_root, text=True, capture_output=True, check=False)
            rc = proc.returncode
            (out / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
            if rc != 0:
                status = "HOLD_SHIM_DRY_RUN_RETURNS_NONZERO"
            elif not result.exists():
                status = "HOLD_SHIM_RESULT_JSON_MISSING"
            else:
                shim_obj = read_json(result)
                if shim_obj.get("status") != PASS:
                    status = "HOLD_SHIM_STATUS_NOT_STATE_ID_BOUND"
                elif shim_obj.get("state_id") != state_id:
                    status = "HOLD_SHIM_STATE_ID_MISMATCH"
                elif (shim_obj.get("state_id_binding") or {}).get("binding_mode") != "DRY_RUN_METADATA_ONLY":
                    status = "HOLD_SHIM_STATE_ID_MODE_MISMATCH"
    report = {"gate": GATE, "status": status, "reason": reason, "input_c6_1i_json": str(args.input_c6_1i_json), "input_c6_1i_json_sha256": observed, "expected_c6_1i_json_sha256": args.expected_c6_1i_sha256, "selected_parent": parent, "state_id": state_id, "executed_command": {"mode": "SHIM_DRY_RUN_ONLY", "argv": argv, "returncode": rc}, "shim_result": {"path": str(out / "shim_result.json"), "status": shim_obj.get("status"), "sha256": sha256_file(out / "shim_result.json") if (out / "shim_result.json").exists() else ""}, "boundaries": {"runtime_execution": "NOT_PERFORMED", "env_execution": "NOT_PERFORMED", "rollout": "NOT_PERFORMED"}, "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "shim_state_id_static_dry_run_binding.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1i-json", required=True)
    p.add_argument("--expected-c6-1i-sha256", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--python", default="python3")
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
