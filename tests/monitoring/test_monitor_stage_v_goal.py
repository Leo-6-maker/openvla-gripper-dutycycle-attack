from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.monitoring.audit_stage_v_closure import (
    audit_closure,
    parent_progress,
    write_root_seal,
)
from scripts.monitoring.monitor_stage_v_goal import ExclusiveMonitorLock, Gatekeeper, MonitorError


COMMIT = "b300e79bb0e6e754a9d384f8ea1b75034bd1d4b4"
TREE = "96881b4d53f901870dd53ede39d051c0a4c83e34"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_dir(path: Path) -> None:
    lines = []
    for child in sorted(path.iterdir()):
        if child.name in {"SHA256SUMS", "SHA256SUMS.sha256"} or not child.is_file():
            continue
        lines.append(f"{sha(child)}  {child.name}\n")
    (path / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    (path / "SHA256SUMS.sha256").write_text(f"{sha(path / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def make_root(tmp_path: Path, *, clean_success: bool = True, parent_key: str = "libero_goal/task_00/state_48") -> tuple[Path, Path, Path]:
    root = tmp_path / "stage-v"
    manifest = tmp_path / "parent-manifest.json"
    write_json(manifest, {"schema": "D8_STAGE_V_CLEAN_SUCCESS_PARENT_MANIFEST_V1", "selected_parents": [{"canonical_parent_key": parent_key}]})
    parent = root / parent_key
    parent.mkdir(parents=True)
    result = {
        "schema": "D8_STAGE_V_COUNTERFACTUAL_PARENT_RESULT_V1",
        "status": "PASS",
        "canonical_parent_key": parent_key,
        "suite": "libero_goal",
        "task_index": 0,
        "state_index": 48,
        "clean_success": clean_success,
        "exact_snapshot_replay": True,
        "current_source_commit": COMMIT,
        "current_source_tree": TREE,
        "current_source_status": "",
        "probe_count": 1,
        "branch_count": 3,
        "branch_arms": ["CLEAN", "OPEN_T3", "OPEN_T5", "OPEN_T10", "NOOP_T10_REPLAY"],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    write_json(parent / "PARENT_RESULT.json", result)
    rows = []
    for arm in ("OPEN_T3", "OPEN_T5", "OPEN_T10"):
        rows.append(
            {
                "arm": arm,
                "probe_step": 4,
                "control": {"status": "PASS"},
                "opened": {"status": "PASS"},
                "comparison": {
                    "label_status": "VALID",
                    "control_task_success": True,
                    "local_vulnerability": False,
                    "task_vulnerability": False,
                },
            }
        )
    (parent / "COUNTERFACTUAL_BRANCHES.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    seal_dir(parent)
    write_json(root / "SUPERVISOR_START.json", {"source_commit": COMMIT, "source_tree": TREE, "planned_parents": 1, "parent_manifest": str(manifest)})
    write_json(root / "LOCAL_HEARTBEAT.json", {"source_commit": COMMIT, "source_tree": TREE, "heartbeat_count": 2, "oom_kill": 0, "gpu_xid_status": "CLEAR"})
    write_json(root / "RUN_MANIFEST.json", {"source_commit": COMMIT, "source_tree": TREE, "gpus": [1], "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0})
    write_json(root / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {"verdict": "PASS", "parent_count": 1, "source_commit": COMMIT, "source_tree": TREE})
    write_json(root / "SUPERVISOR_COMPLETE.json", {"status": "PASS", "source_commit": COMMIT, "source_tree": TREE, "planned_parents": 1, "completed_parents": 1, "accepted_parent_results": 1, "accepted_parent_artifacts": [{"artifact_audit_verdict": "PASS"}]})
    return root, manifest, parent


def test_closure_success_and_root_seal(tmp_path: Path) -> None:
    root, manifest, _ = make_root(tmp_path)
    pre = audit_closure(root, parent_manifest=manifest, expected_source_commit=COMMIT, expected_source_tree=TREE, expected_parent_count=1, require_root_seal=False)
    assert pre["verdict"] == "PASS"
    write_root_seal(root)
    final = audit_closure(root, parent_manifest=manifest, expected_source_commit=COMMIT, expected_source_tree=TREE, expected_parent_count=1, require_root_seal=True)
    assert final["verdict"] == "PASS"
    assert final["accepted_parent_count"] == 1


def test_clean_success_failure_is_not_accepted(tmp_path: Path) -> None:
    root, manifest, _ = make_root(tmp_path, clean_success=False)
    progress = parent_progress(root, parent_manifest=manifest, expected_source_commit=COMMIT, expected_source_tree=TREE)
    assert progress["accepted_parent_count"] == 0
    assert progress["invalid_parent_count"] == 1


def test_duplicate_identity_and_missing_branch_are_fail_closed(tmp_path: Path) -> None:
    root, manifest, parent = make_root(tmp_path)
    duplicate = root / "libero_goal" / "task_99" / "state_99"
    duplicate.mkdir(parents=True)
    for name in ("PARENT_RESULT.json", "COUNTERFACTUAL_BRANCHES.jsonl", "SHA256SUMS", "SHA256SUMS.sha256"):
        (duplicate / name).write_bytes((parent / name).read_bytes())
    progress = parent_progress(root, parent_manifest=manifest, expected_source_commit=COMMIT, expected_source_tree=TREE)
    assert progress["duplicate_identity_count"] == 1
    assert progress["accepted_parent_count"] == 0


def test_bad_source_and_bad_seal_fail_closed(tmp_path: Path) -> None:
    root, manifest, parent = make_root(tmp_path)
    result = json.loads((parent / "PARENT_RESULT.json").read_text())
    result["current_source_tree"] = "bad"
    write_json(parent / "PARENT_RESULT.json", result)
    report = audit_closure(root, parent_manifest=manifest, expected_source_commit=COMMIT, expected_source_tree=TREE, expected_parent_count=1, require_root_seal=False)
    assert report["verdict"] == "FAIL"
    assert report["accepted_parent_count"] == 0


def test_single_instance_flock_and_stale_lock(tmp_path: Path) -> None:
    path = tmp_path / "monitor.lock"
    first = ExclusiveMonitorLock(path, {"monitor_pid": 1})
    first.acquire(tmp_path / "monitor")
    second = ExclusiveMonitorLock(path, {"monitor_pid": 2})
    with pytest.raises(MonitorError, match="DUPLICATE_MONITOR"):
        second.acquire(tmp_path / "monitor")
    first.close()
    third = ExclusiveMonitorLock(path, {"monitor_pid": 3})
    third.acquire(tmp_path / "monitor")
    assert (tmp_path / "monitor" / "STAGE_V_MONITOR_STALE_LOCK_AUDIT.json").is_file()
    third.close()


def test_gpu5_is_never_accepted() -> None:
    from scripts.detector_v5.run_stage_v_local_supervisor import validate_workers

    assert "GPU5_FORBIDDEN" in validate_workers([{"pid": os.getpid(), "gpu": 5}], [1, 2], require_live=False)


def test_stale_heartbeat_with_healthy_pids_is_degraded_not_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, manifest, _ = make_root(tmp_path)
    args = type("Args", (), {})()
    args.stage_v_root = root
    args.goal_root = tmp_path / "goal"
    args.goal_root.mkdir()
    args.expected_source_commit = COMMIT
    args.expected_source_tree = TREE
    args.expected_parent_count = 1
    args.expected_gpus = [1]
    args.reserved_gpus = [5]
    args.protected_pid = 0
    args.lock_path = tmp_path / "lock"
    args.kill_grace_seconds = 0.01
    args.stage_v2_command_file = args.goal_root / "STAGE_V2_COMMAND.json"
    monitor = Gatekeeper(args)
    (root / "SUPERVISOR_COMPLETE.json").unlink()
    monkeypatch.setattr(monitor, "_resource_snapshot", lambda: {
        "supervisor_pid": 11,
        "dispatcher_pid": 12,
        "supervisor_alive": True,
        "dispatcher_alive": True,
        "active_worker_pids": [],
        "gpu_assignments": [],
        "hard_stop_errors": [],
        "heartbeat_age_seconds": 1000,
        "heartbeat_warning_seconds": 90,
        "boundary_counters": {},
    })
    killed = []
    monkeypatch.setattr(monitor, "_hard_stop", lambda *args: killed.append(args) or "STOP")
    monitor._write_parent_progress = lambda: {"planned_parent_count": 1, "branch_complete_parent_count": 0, "accepted_parent_count": 0, "parents": []}
    monitor._write_resource_sample = lambda resource: None
    monitor._write_monitor_heartbeat = lambda resource, progress: None
    monitor.tick()
    assert not killed
    assert json.loads((monitor.monitor_root / "STAGE_V_MONITOR_STATE.json").read_text())["status"] == "DEGRADED"


def test_oom_delta_is_a_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _, _ = make_root(tmp_path)
    args = type("Args", (), {
        "stage_v_root": root,
        "goal_root": tmp_path / "goal",
        "expected_source_commit": COMMIT,
        "expected_source_tree": TREE,
        "expected_parent_count": 1,
        "expected_gpus": [1],
        "reserved_gpus": [5],
        "protected_pid": 0,
        "lock_path": tmp_path / "lock",
        "kill_grace_seconds": 0.01,
        "stage_v2_command_file": tmp_path / "goal" / "STAGE_V2_COMMAND.json",
    })()
    args.goal_root.mkdir()
    monitor = Gatekeeper(args)
    (root / "SUPERVISOR_COMPLETE.json").unlink()
    monkeypatch.setattr(monitor, "_resource_snapshot", lambda: {"hard_stop_errors": ["OOM_KILL_COUNTER_INCREASED"], "active_worker_pids": [], "gpu_assignments": [], "supervisor_alive": True, "dispatcher_alive": True})
    monkeypatch.setattr(monitor, "_write_parent_progress", lambda: {"planned_parent_count": 1, "branch_complete_parent_count": 0, "accepted_parent_count": 0, "parents": []})
    monkeypatch.setattr(monitor, "_write_resource_sample", lambda resource: None)
    monkeypatch.setattr(monitor, "_write_monitor_heartbeat", lambda resource, progress: None)
    called = []
    monkeypatch.setattr(monitor, "_hard_stop", lambda reasons, resource: called.append(reasons) or "STOP")
    assert monitor.tick() == "STOP"
    assert called == [["OOM_KILL_COUNTER_INCREASED"]]


def test_v2_command_forbidden_boundary(tmp_path: Path) -> None:
    root, manifest, _ = make_root(tmp_path)
    goal = tmp_path / "goal"
    goal.mkdir()
    command_file = goal / "STAGE_V2_COMMAND.json"
    closure = root / "STAGE_V_CLOSURE_RECEIPT.json"
    write_json(closure, {"status": "STAGE_V_FORMAL_MAP_CLOSED"})
    runner = tmp_path / "runner.py"
    auditor = tmp_path / "auditor.py"
    config = tmp_path / "config.json"
    for path in (runner, auditor, config):
        path.write_text(path.name, encoding="utf-8")
    write_json(
        command_file,
        {
            "schema": "STAGE_V2_COMMAND_V2",
            "stage": "V2_TEACHER_ENRICHMENT",
            "read_only": True,
            "stage_v_root": str(root),
            "stage_v_source_commit": COMMIT,
            "stage_v_source_tree": TREE,
            "stage_v2_source_commit": "stage-v2-test-commit",
            "stage_v2_source_tree": "stage-v2-test-tree",
            "expected_stage_v_closure_receipt_sha256": sha(closure),
            "expected_parent_manifest_sha256": sha(manifest),
            "parent_manifest_sha256": sha(manifest),
            "expected_run_manifest_sha256": sha(root / "RUN_MANIFEST.json"),
            "stage_v2_runner_path": str(runner),
            "stage_v2_runner_sha256": sha(runner),
            "stage_v2_auditor_path": str(auditor),
            "stage_v2_auditor_sha256": sha(auditor),
            "stage_v2_config_path": str(config),
            "stage_v2_config_sha256": sha(config),
            "output_root_template": str(goal / "STAGE_V2_TEACHER_ENRICHMENT_{commit8}_{utc}"),
            "lock_path": str(goal / ".stage_v2_teacher_enrichment.lock"),
            "env": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
            "command": ["python", "run_attack.py"],
        },
    )
    args = type("Args", (), {"stage_v_root": root, "goal_root": goal, "expected_source_commit": COMMIT, "expected_source_tree": TREE, "expected_parent_count": 1, "expected_gpus": [1], "reserved_gpus": [5], "protected_pid": 0, "lock_path": tmp_path / "lock", "kill_grace_seconds": 0.01, "stage_v2_command_file": command_file})()
    monitor = Gatekeeper(args)
    assert monitor._load_v2_spec()[1] == "STAGE_V2_COMMAND_FORBIDDEN_BOUNDARY"


def test_v2_command_plan_waits_for_closure_then_materializes(tmp_path: Path) -> None:
    root, manifest, _ = make_root(tmp_path)
    goal = tmp_path / "goal"
    goal.mkdir()
    runner = tmp_path / "runner.py"
    auditor = tmp_path / "auditor.py"
    config = tmp_path / "config.json"
    for path in (runner, auditor, config):
        path.write_text(path.name, encoding="utf-8")
    command_file = goal / "MONITOR" / "STAGE_V2_COMMAND.json"
    args = type("Args", (), {"stage_v_root": root, "goal_root": goal, "expected_source_commit": COMMIT, "expected_source_tree": TREE, "expected_parent_count": 1, "expected_gpus": [1], "reserved_gpus": [5], "protected_pid": 0, "lock_path": tmp_path / "lock", "kill_grace_seconds": 0.01, "stage_v2_command_file": command_file})()
    monitor = Gatekeeper(args)
    plan = {
        "schema": "STAGE_V2_COMMAND_PLAN_V1",
        "stage": "V2_TEACHER_ENRICHMENT",
        "read_only": True,
        "stage_v_root": str(root),
        "stage_v_source_commit": COMMIT,
        "stage_v_source_tree": TREE,
        "stage_v2_source_commit": "stage-v2-commit",
        "stage_v2_source_tree": "stage-v2-tree",
        "expected_parent_manifest_sha256": sha(manifest),
        "expected_run_manifest_sha256": sha(root / "RUN_MANIFEST.json"),
        "stage_v2_runner_path": str(runner),
        "stage_v2_runner_sha256": sha(runner),
        "stage_v2_auditor_path": str(auditor),
        "stage_v2_auditor_sha256": sha(auditor),
        "stage_v2_config_path": str(config),
        "stage_v2_config_sha256": sha(config),
        "output_root_template": str(goal / "STAGE_V2_TEACHER_ENRICHMENT_{commit8}_{utc}"),
        "lock_path": str(goal / ".stage_v2_teacher_enrichment.lock"),
        "env": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
        "command_template": ["python", "run_v2.py", "--stage-v-root", "{stage_v_root}", "--output-root", "{output_root}"],
    }
    write_json(monitor.monitor_root / "STAGE_V2_COMMAND_PLAN.json", plan)
    assert monitor._materialize_v2_command() == "STAGE_V_CLOSURE_RECEIPT_NOT_READY"
    write_json(root / "STAGE_V_CLOSURE_RECEIPT.json", {"status": "STAGE_V_FORMAL_MAP_CLOSED", "manifest_sha256": sha(manifest)})
    assert monitor._materialize_v2_command() is None
    final = json.loads(command_file.read_text())
    assert final["schema"] == "STAGE_V2_COMMAND_V2"
    assert final["expected_stage_v_closure_receipt_sha256"] == sha(root / "STAGE_V_CLOSURE_RECEIPT.json")
    assert monitor._load_v2_spec()[1] is None
