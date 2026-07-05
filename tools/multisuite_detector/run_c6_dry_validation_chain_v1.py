#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

GATE = "C6_DRY_VALIDATION_CHAIN_RUNNER"
PASS = "PASS_C6_DRY_CHAIN_THROUGH_1N"
OUT_FILES = ["c6_dry_validation_chain_report.json", "checksum_report.json"]


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


def run_cmd(cmd, cwd, stdout_path, stderr_path):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    Path(stdout_path).write_text(proc.stdout, encoding="utf-8")
    Path(stderr_path).write_text(proc.stderr, encoding="utf-8")
    return proc.returncode


def status_of(path):
    try:
        return str(read_json(path).get("status", ""))
    except Exception:
        return ""


def argv_value(argv, flag):
    argv = list(argv or [])
    for i, x in enumerate(argv[:-1]):
        if x == flag:
            return str(argv[i + 1])
    return ""


def preview_precheck(c6_1l_json, expected_task_id, expected_state_id):
    obj = read_json(c6_1l_json)
    argv = list(obj.get("legacy_runner_argv_preview") or [])
    return {
        "preview_argv": argv,
        "task_id": argv_value(argv, "--task_id"),
        "state_ids": argv_value(argv, "--state_ids"),
        "task_id_ok": argv_value(argv, "--task_id") == str(expected_task_id),
        "state_ids_ok": argv_value(argv, "--state_ids") == str(expected_state_id),
        "dry_run_ok": "--dry_run" in argv,
    }


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported})
    sums = out / "SHA256SUMS"
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def step_record(name, cmd, rc, json_path):
    return {"name": name, "returncode": rc, "json_path": str(json_path), "json_sha256": sha256_file(json_path) if Path(json_path).exists() else "", "status": status_of(json_path), "cmd": cmd}


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo_root)
    py = str(args.python)
    steps = []
    status = PASS
    reason = ""

    if not Path(args.input_c6_1i_json).exists():
        status = "HOLD_C6_1I_JSON_NOT_FOUND"
        reason = str(args.input_c6_1i_json)
    elif sha256_file(args.input_c6_1i_json) != args.expected_c6_1i_sha256:
        status = "HOLD_C6_1I_HASH_MISMATCH"
        reason = sha256_file(args.input_c6_1i_json)

    env_dir = out / "env_validation"
    if status == PASS and args.validate_env:
        env_dir.mkdir(parents=True, exist_ok=True)
        env_json = env_dir / "libero_official_env_validation.json"
        cmd = [py, "tools/multisuite_detector/validate_c6_libero_official_env_v1.py", "--python", py, "--repo-root", str(repo), "--output-root", str(env_dir), "--git-commit", args.git_commit]
        if args.require_cuda:
            cmd.append("--require-cuda")
        rc = run_cmd(cmd, repo, env_dir / "stdout.txt", env_dir / "stderr.txt")
        steps.append(step_record("env", cmd, rc, env_json))
        if rc != 0:
            status = "HOLD_ENV_VALIDATION_FAILED"
            reason = status_of(env_json)

    j_dir = out / "c6_1j"
    if status == PASS:
        j_json = j_dir / "shim_state_id_static_dry_run_binding.json"
        cmd = [py, "tools/multisuite_detector/prove_c6_shim_state_id_dry_run_v1.py", "--input-c6-1i-json", str(args.input_c6_1i_json), "--expected-c6-1i-sha256", args.expected_c6_1i_sha256, "--repo-root", str(repo), "--output-root", str(j_dir), "--git-commit", args.git_commit]
        rc = run_cmd(cmd, repo, j_dir / "chain_stdout.txt", j_dir / "chain_stderr.txt")
        steps.append(step_record("c6_1j", cmd, rc, j_json))
        if rc != 0:
            status = "HOLD_C6_1J_FAILED"
            reason = status_of(j_json)

    k_dir = out / "c6_1k"
    if status == PASS and args.run_c6_1k:
        k_json = k_dir / "state_id_source_static_audit.json"
        j_sha = sha256_file(j_json)
        cmd = [py, "tools/multisuite_detector/audit_c6_state_id_source_v1.py", "--input-c6-1j-json", str(j_json), "--expected-c6-1j-sha256", j_sha, "--source-file", "scripts/v4_run_eval_openvla.py", "--repo-root", str(repo), "--output-root", str(k_dir), "--git-commit", args.git_commit]
        rc = run_cmd(cmd, repo, k_dir / "chain_stdout.txt", k_dir / "chain_stderr.txt")
        steps.append(step_record("c6_1k", cmd, rc, k_json))
        if rc != 0:
            status = "HOLD_C6_1K_FAILED"
            reason = status_of(k_json)

    l_dir = out / "c6_1l"
    if status == PASS:
        l_json = l_dir / "shim_legacy_state_ids_preview_binding.json"
        j_sha = sha256_file(j_json)
        cmd = [py, "tools/multisuite_detector/prove_c6_shim_legacy_state_ids_preview_v1.py", "--input-c6-1j-json", str(j_json), "--expected-c6-1j-sha256", j_sha, "--repo-root", str(repo), "--python", py, "--output-root", str(l_dir), "--git-commit", args.git_commit]
        rc = run_cmd(cmd, repo, l_dir / "chain_stdout.txt", l_dir / "chain_stderr.txt")
        steps.append(step_record("c6_1l", cmd, rc, l_json))
        if rc != 0:
            status = "HOLD_C6_1L_FAILED"
            reason = status_of(l_json)

    precheck = {}
    if status == PASS:
        precheck = preview_precheck(l_json, args.expected_legacy_task_id, args.expected_state_id)
        if not precheck["dry_run_ok"]:
            status = "HOLD_C6_1L_PREVIEW_NOT_DRY_RUN"
        elif not precheck["task_id_ok"]:
            status = "HOLD_C6_1L_PREVIEW_TASK_ID_MISMATCH"
        elif not precheck["state_ids_ok"]:
            status = "HOLD_C6_1L_PREVIEW_STATE_ID_MISMATCH"
        if status != PASS:
            reason = json.dumps(precheck, sort_keys=True)

    m_dir = out / "c6_1m"
    if status == PASS:
        m_json = m_dir / "legacy_runner_state_ids_dry_run_invocation.json"
        l_sha = sha256_file(l_json)
        cmd = [py, "tools/multisuite_detector/prove_c6_legacy_runner_state_ids_dry_run_v1.py", "--input-c6-1l-json", str(l_json), "--expected-c6-1l-sha256", l_sha, "--python", py, "--repo-root", str(repo), "--output-root", str(m_dir), "--git-commit", args.git_commit]
        rc = run_cmd(cmd, repo, m_dir / "chain_stdout.txt", m_dir / "chain_stderr.txt")
        steps.append(step_record("c6_1m", cmd, rc, m_json))
        if rc != 0:
            status = "HOLD_C6_1M_FAILED"
            reason = status_of(m_json)

    n_dir = out / "c6_1n"
    if status == PASS and args.run_c6_1n:
        n_json = n_dir / "legacy_dry_run_artifact_validation.json"
        m_sha = sha256_file(m_json)
        cmd = [py, "tools/multisuite_detector/validate_c6_legacy_dry_run_artifacts_v1.py", "--input-c6-1m-json", str(m_json), "--expected-c6-1m-sha256", m_sha, "--output-root", str(n_dir), "--git-commit", args.git_commit]
        rc = run_cmd(cmd, repo, n_dir / "chain_stdout.txt", n_dir / "chain_stderr.txt")
        steps.append(step_record("c6_1n", cmd, rc, n_json))
        if rc != 0:
            status = "HOLD_C6_1N_FAILED"
            reason = status_of(n_json)

    report = {"gate": GATE, "status": status, "reason": reason, "output_root": str(out), "python": py, "git_commit": args.git_commit, "steps": steps, "preview_precheck": precheck, "boundaries": {"runner": "DRY_RUN_ONLY", "model": "NOT_PERFORMED", "sim_env": "NOT_PERFORMED", "rollout": "NOT_PERFORMED"}}
    write_json(out / "c6_dry_validation_chain_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--input-c6-1i-json", required=True)
    p.add_argument("--expected-c6-1i-sha256", required=True)
    p.add_argument("--expected-legacy-task-id", default="libero_goal_open_middle_drawer")
    p.add_argument("--expected-state-id", default="0")
    p.add_argument("--validate-env", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--run-c6-1k", action="store_true")
    p.add_argument("--run-c6-1n", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
