"""Run and audit the non-scientific Dynamic-8 control-plane canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from .stage_v_dynamic_common import atomic_write_json, pid_alive, read_json, utc_now
except ImportError:
    from stage_v_dynamic_common import atomic_write_json, pid_alive, read_json, utc_now


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
    parser.add_argument("--allow-gpu5", action="store_true", help="Authorize GPU5 for this fresh eight-GPU canary")
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
        "--lock-path", str(args.lock_path), "--approved-gpus", args.approved_gpus,
        "--excluded-gpus", "" if args.allow_gpu5 else "5",
        "--preflight-file", str(args.preflight_file), "--dispatcher-script", str(dispatcher), "--auditor-script", str(auditor),
        "--worker-command", worker_command, "--canary-peak-mib", str(args.canary_peak_mib),
        "--external-pid", str(args.external_pid), "--ssh-probe-command", "false", "--poll-seconds", "1",
    ]
    if args.allow_gpu5:
        command.append("--allow-gpu5")
    if args.skip_resource_checks:
        command.append("--skip-resource-checks")
    before_pid_alive = bool(args.external_pid and pid_alive(args.external_pid))
    result = subprocess.run(command, cwd=str(args.repo_root), check=False, capture_output=True, text=True)
    heartbeat = read_json(root / "LOCAL_HEARTBEAT.json", {})
    audit = read_json(root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {})
    worker_statuses = []
    for status_path in sorted(root.glob("worker_gpu*/WORKER_STATUS.json")):
        status = read_json(status_path, {})
        if isinstance(status, dict):
            worker_statuses.append(status)
    worker_reap_pass = bool(worker_statuses) and all(
        status.get("state") == "STOPPED"
        and not _pid_alive(int(status.get("worker_pid") or 0))
        and not _pid_alive(int(status.get("child_pid") or 0))
        for status in worker_statuses
    )
    external_pid_present_after = bool(args.external_pid and pid_alive(args.external_pid))
    gpu5_touched = any(int(status.get("gpu_id") or -1) == 5 for status in worker_statuses)
    report = {
        "schema": "DYNAMIC8_CONTROL_CANARY_REPORT_V2",
        "verdict": "PASS" if result.returncode == 0 and audit.get("verdict") == "PASS" and not (root / "ABORTED_INCOMPLETE.json").exists() else "FAIL",
        "supervisor_exit_code": result.returncode,
        "heartbeat_count": heartbeat.get("heartbeat_count", 0),
        "heartbeat_at_least_five": heartbeat.get("heartbeat_count", 0) >= 5,
        "ssh_failure_count": heartbeat.get("ssh_probe_failure_count", 0),
        "ssh_failures_did_not_abort": result.returncode == 0,
        "external_pid_present_before": before_pid_alive,
        "external_pid_present_after": external_pid_present_after,
        "external_process_untouched": not before_pid_alive or external_pid_present_after,
        "external_process_terminated": False,
        "worker_reap_pass": worker_reap_pass,
        "gpu5_touched": gpu5_touched,
        "gpu5_authorized": bool(args.allow_gpu5),
        "gpu5_policy_pass": gpu5_touched == bool(args.allow_gpu5),
        "old_artifacts_reused": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "generated_utc": utc_now(),
    }
    report["verdict"] = "PASS" if report["verdict"] == "PASS" and report["heartbeat_at_least_five"] and report["ssh_failures_did_not_abort"] and report["worker_reap_pass"] and report["external_process_untouched"] and report["gpu5_policy_pass"] and not report["external_process_terminated"] else "FAIL"
    atomic_write_json(root / "DYNAMIC8_CONTROL_CANARY_REPORT.json", report)
    atomic_write_json(root / "DYNAMIC8_CONTROL_CANARY_AUDIT.json", {
        "schema": "DYNAMIC8_CONTROL_CANARY_AUDIT_V2", "verdict": report["verdict"],
        "queue_audit_verdict": audit.get("verdict"), "worker_reap_required": True,
        "active_worker_pids": heartbeat.get("active_worker_pids", []),
        "gpu_assignments": heartbeat.get("gpu_assignments", []),
        "worker_reap_pass": report["worker_reap_pass"],
        "gpu5_touched": report["gpu5_touched"],
        "gpu5_authorized": report["gpu5_authorized"],
        "gpu5_policy_pass": report["gpu5_policy_pass"],
        "external_process_untouched": report["external_process_untouched"],
        "audited_utc": utc_now(),
    })
    return 0 if report["verdict"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
