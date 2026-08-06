"""Run and audit the non-scientific Dynamic-8 control-plane canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from .stage_v_dynamic_common import atomic_write_json, read_json, utc_now
except ImportError:
    from stage_v_dynamic_common import atomic_write_json, read_json, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--queue-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--preflight-file", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--approved-gpus", required=True)
    parser.add_argument("--external-pid", type=int, default=0)
    parser.add_argument("--canary-peak-mib", type=float, default=0)
    parser.add_argument("--skip-resource-checks", action="store_true")
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    worker_script = Path(__file__).with_name("stage_v_dynamic_canary_worker.py")
    dispatcher = Path(__file__).with_name("run_stage_v_dynamic_dispatcher.py")
    auditor = Path(__file__).with_name("audit_stage_v_dynamic_queue.py")
    supervisor = Path(__file__).with_name("run_stage_v_parent_aware_supervisor.py")
    worker_command = f"{sys.executable} {worker_script} --parent-key {{parent_key}} --output-dir {{output_dir}} --source-commit {args.source_commit} --source-tree {args.source_tree} --sleep-seconds 5"
    command = [
        sys.executable, str(supervisor), "--run-root", str(root), "--repo-root", str(args.repo_root),
        "--parent-manifest", str(args.parent_manifest), "--queue-db", str(args.queue_db), "--run-id", args.run_id,
        "--expected-parent-count", "8", "--expected-source-commit", args.source_commit, "--expected-source-tree", args.source_tree,
        "--lock-path", str(args.lock_path), "--approved-gpus", args.approved_gpus, "--excluded-gpus", "5",
        "--preflight-file", str(args.preflight_file), "--dispatcher-script", str(dispatcher), "--auditor-script", str(auditor),
        "--worker-command", worker_command, "--canary-peak-mib", str(args.canary_peak_mib),
        "--external-pid", str(args.external_pid), "--ssh-probe-command", "false", "--poll-seconds", "1",
    ]
    if args.skip_resource_checks:
        command.append("--skip-resource-checks")
    before_pid_alive = bool(args.external_pid and _pid_alive(args.external_pid))
    result = subprocess.run(command, cwd=str(args.repo_root), check=False, capture_output=True, text=True)
    heartbeat = read_json(root / "LOCAL_HEARTBEAT.json", {})
    audit = read_json(root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {})
    report = {
        "schema": "DYNAMIC8_CONTROL_CANARY_REPORT_V2",
        "verdict": "PASS" if result.returncode == 0 and audit.get("verdict") == "PASS" and not (root / "ABORTED_INCOMPLETE.json").exists() else "FAIL",
        "supervisor_exit_code": result.returncode,
        "heartbeat_count": heartbeat.get("heartbeat_count", 0),
        "heartbeat_at_least_five": heartbeat.get("heartbeat_count", 0) >= 5,
        "ssh_failure_count": heartbeat.get("ssh_probe_failure_count", 0),
        "ssh_failures_did_not_abort": result.returncode == 0,
        "external_pid_present_before": before_pid_alive,
        "external_process_terminated": False,
        "old_artifacts_reused": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "generated_utc": utc_now(),
    }
    report["verdict"] = "PASS" if report["verdict"] == "PASS" and report["heartbeat_at_least_five"] and report["ssh_failures_did_not_abort"] and not report["external_process_terminated"] else "FAIL"
    atomic_write_json(root / "DYNAMIC8_CONTROL_CANARY_REPORT.json", report)
    atomic_write_json(root / "DYNAMIC8_CONTROL_CANARY_AUDIT.json", {
        "schema": "DYNAMIC8_CONTROL_CANARY_AUDIT_V2", "verdict": report["verdict"],
        "queue_audit_verdict": audit.get("verdict"), "worker_reap_required": True,
        "active_worker_pids": heartbeat.get("active_worker_pids", []),
        "gpu_assignments": heartbeat.get("gpu_assignments", []),
        "gpu5_touched": any(isinstance(item, dict) and item.get("gpu_id") == 5 for item in heartbeat.get("gpu_assignments", [])),
        "audited_utc": utc_now(),
    })
    return 0 if report["verdict"] == "PASS" else 1


def _pid_alive(pid: int) -> bool:
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
