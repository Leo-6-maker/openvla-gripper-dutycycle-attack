from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "detector_v5"
sys.path.insert(0, str(SCRIPTS))

import audit_stage_v_local_supervisor as auditor
import run_stage_v_local_supervisor as supervisor


def _source_identity() -> tuple[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _base_run(root: Path, *, supervisor_pid: int | None = None, dispatcher_pid: int | None = None) -> None:
    commit, tree = _source_identity()
    _write(
        root / "SUPERVISOR_START.json",
        {
            "schema": supervisor.SCHEMA,
            "source_commit": commit,
            "source_tree": tree,
            "planned_parents": 1,
            "supervisor_pid": supervisor_pid,
            "dispatcher_pid": dispatcher_pid,
        },
    )
    _write(
        root / "LOCAL_HEARTBEAT.json",
        {
            "schema": supervisor.SCHEMA,
            "control_plane_mode": "LOCAL_AUTONOMOUS",
            "ssh_is_hard_stop": False,
            "source_commit": commit,
            "source_tree": tree,
            "supervisor_pid": supervisor_pid,
            "dispatcher_pid": dispatcher_pid,
            "active_worker_pids": [],
            "gpu_assignments": [],
            "heartbeat_count": 1,
            "accepted_parent_results": 0,
            "updated_utc": supervisor.utc_now(),
            "gpu_xid_status": "CLEAR",
            "resource_errors": [],
        },
    )


def _run_cli(tmp_path: Path, *, dispatcher_code: str, ssh_probe: str = "false") -> tuple[subprocess.CompletedProcess[str], Path]:
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        pytest.skip("integration supervisor test requires a clean source worktree")
    commit, tree = _source_identity()
    root = tmp_path / "run"
    lock = tmp_path / "stage_v.lock"
    logs = tmp_path / "logs"
    logs.mkdir()
    command = [
        sys.executable,
        str(SCRIPTS / "run_stage_v_local_supervisor.py"),
        "--run-root", str(root),
        "--repo-root", str(ROOT),
        "--expected-source-commit", commit,
        "--expected-source-tree", tree,
        "--lock-path", str(lock),
        "--approved-gpus", "0",
        "--planned-parents", "1",
        "--dispatcher-command", shlex.join([sys.executable]),
        "--dispatcher-arg=-c",
        "--dispatcher-arg=" + dispatcher_code,
        "--audit-command", shlex.join([
            sys.executable,
            str(SCRIPTS / "audit_stage_v_local_supervisor.py"),
            "--run-root", "{run_root}",
            "--planned-parents", "1",
            "--approved-gpus", "0",
            "--live",
            "--no-write",
        ]),
        "--poll-interval", "0.05",
        "--heartbeat-interval", "0.05",
        "--min-available-ram-gib", "0",
        "--gpu-query-command", "false",
        "--kernel-log-command", "false",
        "--ssh-probe-command", ssh_probe,
        "--stdout-log", str(logs / "stdout.log"),
        "--stderr-log", str(logs / "stderr.log"),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
    return result, root


def _success_dispatcher_code() -> str:
    return (
        "import json,time; "
        "json.dump({'planned_parents':1,'completed_parents':1,'accepted_parent_results':1,"
        "'failed_parents':0,'accepted_parent_artifacts':[{'artifact_audit_verdict':'PASS'}],"
        "'eval160_reads':0,'protected_eval_reads':0,'vis_pgd_attack_rollouts':0},"
        "open('RUN_SUMMARY.json','w')); time.sleep(0.35)"
    )


def test_parent_shell_exit_does_not_stop_detached_supervisor(tmp_path: Path) -> None:
    result, root = _run_cli(tmp_path, dispatcher_code=_success_dispatcher_code())
    assert result.returncode == 0, result.stderr
    assert (root / "SUPERVISOR_COMPLETE.json").is_file()


def test_ssh_probe_failure_is_telemetry_only_and_heartbeat_repeats(tmp_path: Path) -> None:
    result, root = _run_cli(tmp_path, dispatcher_code=_success_dispatcher_code(), ssh_probe="false")
    assert result.returncode == 0, result.stderr
    heartbeat = json.loads((root / "LOCAL_HEARTBEAT.json").read_text(encoding="utf-8"))
    assert heartbeat["ssh_is_hard_stop"] is False
    assert heartbeat["ssh_probe_failure_count"] > 0
    assert heartbeat["heartbeat_count"] >= 3


@pytest.mark.skipif(os.name != "posix", reason="flock semantics are production Linux behavior")
def test_duplicate_launcher_is_rejected_by_flock(tmp_path: Path) -> None:
    lock = supervisor.ExclusiveLock(tmp_path / "run.lock", {"supervisor_pid": os.getpid()})
    lock.acquire(tmp_path / "one")
    try:
        duplicate = supervisor.ExclusiveLock(tmp_path / "run.lock", {"supervisor_pid": os.getpid()})
        with pytest.raises(supervisor.SupervisorError, match="DUPLICATE_SUPERVISOR"):
            duplicate.acquire(tmp_path / "two")
    finally:
        lock.close()


@pytest.mark.skipif(os.name != "posix", reason="flock semantics are production Linux behavior")
def test_stale_lock_is_audited_before_new_run(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    lock_path.write_text(json.dumps({"supervisor_pid": 99999999}) + "\n", encoding="utf-8")
    lock = supervisor.ExclusiveLock(lock_path, {"supervisor_pid": os.getpid()})
    run_root = tmp_path / "new"
    lock.acquire(run_root)
    try:
        assert (run_root / "STALE_LOCK_AUDIT.json").is_file()
    finally:
        lock.close()


def test_dispatcher_pid_loss_aborts(tmp_path: Path) -> None:
    code = "import sys; sys.exit(23)"
    result, root = _run_cli(tmp_path, dispatcher_code=code)
    assert result.returncode != 0
    marker = json.loads((root / "ABORTED_INCOMPLETE.json").read_text(encoding="utf-8"))
    assert marker["accepted_parent_results"] == 0
    assert "DISPATCHER_EXIT" in marker["control_plane_abort_reason"]


def test_stale_heartbeat_with_dead_pid_is_not_accepted(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _base_run(root, supervisor_pid=99999999, dispatcher_pid=99999998)
    report = auditor.audit_run(root, planned_parents=1, approved_gpus=[0], final=True, write_report=False)
    assert report["verdict"] == "FAIL"
    assert "completion_marker_missing" in report["errors"]


def test_stale_heartbeat_with_healthy_pid_is_not_killed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _base_run(root, supervisor_pid=os.getpid(), dispatcher_pid=None)
    report = auditor.audit_run(root, planned_parents=1, approved_gpus=[0], final=False, write_report=False)
    assert report["verdict"] == "PASS"
    assert str(os.getpid()) in report["warnings"][0]
    assert supervisor.pid_alive(os.getpid())


def test_oom_delta_swap_and_xid_are_hard_stops() -> None:
    assert "OOM_KILL_COUNTER_INCREASED" in supervisor.hard_stop_errors(
        available_ram_bytes=10**12,
        min_available_ram_bytes=1,
        baseline_oom=3,
        oom_kill=4,
        swap_bad_streak=0,
        swap_bad_samples=2,
        xid_status="CLEAR",
    )
    assert "SWAP_HARD_STOP" in supervisor.hard_stop_errors(
        available_ram_bytes=10**12,
        min_available_ram_bytes=1,
        baseline_oom=3,
        oom_kill=3,
        swap_bad_streak=2,
        swap_bad_samples=2,
        xid_status="CLEAR",
    )
    assert "NVIDIA_XID" in supervisor.hard_stop_errors(
        available_ram_bytes=10**12,
        min_available_ram_bytes=1,
        baseline_oom=3,
        oom_kill=3,
        swap_bad_streak=0,
        swap_bad_samples=2,
        xid_status="XID_DETECTED",
    )


def test_artifact_validation_failure_is_not_pass(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _base_run(root)
    _write(root / "SUPERVISOR_COMPLETE.json", {
        "status": "PASS", "accepted_parent_results": 1,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
    })
    report = auditor.audit_run(root, planned_parents=1, approved_gpus=[0], final=True, write_report=False)
    assert report["verdict"] == "FAIL"
    assert "accepted_parent_artifact_audit_missing" in report["errors"]


def test_each_gpu_has_at_most_one_worker_and_gpu5_is_forbidden() -> None:
    errors = supervisor.validate_workers(
        [{"pid": 1, "gpu": 0}, {"pid": 2, "gpu": 0}, {"pid": 3, "gpu": 5}],
        [0, 1],
        require_live=False,
    )
    assert "MULTIPLE_WORKERS_ON_GPU:0" in errors
    assert "GPU5_FORBIDDEN" in errors


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are production Linux behavior")
def test_abort_reaps_dispatcher_process_group() -> None:
    child_code = "import subprocess,time; subprocess.Popen(['sleep','30']); time.sleep(30)"
    process = subprocess.Popen([sys.executable, "-c", child_code], start_new_session=True)
    time.sleep(0.15)
    supervisor.terminate_process_group(process, grace_seconds=1)
    assert process.poll() is not None


def test_accepted_parent_requires_complete_artifact_audit(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _base_run(root)
    _write(root / "SUPERVISOR_COMPLETE.json", {
        "status": "PASS", "accepted_parent_results": 1,
        "accepted_parent_artifacts": [{"artifact_audit_verdict": "FAIL"}],
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
    })
    report = auditor.audit_run(root, planned_parents=1, approved_gpus=[0], final=True, write_report=False)
    assert report["verdict"] == "FAIL"
    assert "accepted_parent_artifact_audit_failed:0" in report["errors"]


def test_aborted_root_keeps_accepted_parent_count_zero(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _base_run(root)
    _write(root / "ABORTED_INCOMPLETE.json", {
        "status": "ABORTED_INCOMPLETE", "accepted_parent_results": 0,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
    })
    report = auditor.audit_run(root, planned_parents=1, approved_gpus=[0], final=True, write_report=False)
    assert report["accepted_parent_results"] == 0
    assert "aborted_root_accepted_parent_count_not_zero" not in report["errors"]
