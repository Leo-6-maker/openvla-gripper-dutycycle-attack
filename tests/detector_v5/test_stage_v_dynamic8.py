from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5 import run_stage_v_parent_aware_supervisor as sup
from scripts.detector_v5.audit_stage_v_abort_postmortem import build_postmortem, build_timeout_policy
from scripts.detector_v5 import run_stage_v_control_qualification as control_qualification
from scripts.detector_v5.run_stage_v_control_qualification import ranked
from scripts.detector_v5.stage_v_dynamic_common import (
    atomic_write_json, gpu_preflight, project_queue, science_artifact_status, sha256_file,
)
from scripts.detector_v5.stage_v_science_core_provenance import build as build_provenance, verify as verify_provenance
from scripts.detector_v5.run_stage_v_local_supervisor import ExclusiveLock, SupervisorError, terminate_process_group
from scripts.fec.atomic_task_queue import AtomicTaskQueue


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def make_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=root, approved_gpus=[0, 1, 2, 3, 4, 6, 7, 8], excluded_gpus=[5],
        expected_source_commit="commit", expected_source_tree="tree", skip_resource_checks=False,
        gpu_query_command="fake", min_available_ram_gib=0, external_pid=0, heartbeat_stale_seconds=600,
        ssh_probe_command="", ssh_probe_timeout=1, timeout_policy=None,
    )


def fake_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sup, "_mem_snapshot", lambda: {"available_ram_bytes": 1024**4, "available_ram_gib": 1024, "swap_used_bytes": 0, "swap_in": 0, "swap_out": 0, "oom_kill": 0})
    monkeypatch.setattr(sup, "gpu_snapshot", lambda command: ([{"gpu_id": gpu, "memory_used_mib": 0, "memory_free_mib": 100000, "utilization_gpu_percent": 0} for gpu in [0, 1, 2, 3, 4, 6, 7, 8]], None))
    monkeypatch.setattr(sup, "_xid_status", lambda start: None)


def test_atomic_queue_claim_is_single_owner(tmp_path: Path) -> None:
    queue = AtomicTaskQueue(str(tmp_path / "q.sqlite"), run_id="r")
    queue.init_run(state="ACTIVE", manifest_sha="m", source_sha="s")
    queue.register_tasks([{"cell_id": "p", "parent_id": "p", "suite": "libero_goal", "task_index": 0, "state_index": 48, "arm": "PARENT"}])
    first = queue.claim_task("w1", expected_manifest_sha="m", expected_source_sha="s")
    second = queue.claim_task("w2", expected_manifest_sha="m", expected_source_sha="s")
    assert first and second is None
    queue.close()


def test_control_qualification_uses_queue_and_seals_two_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert control_qualification.DEFAULT_SALT == "STAGE_V_R2_CONTROL_QUALIFICATION_20260807"
    manifest = tmp_path / "candidates.json"
    write_json(manifest, {"parents": [{
        "canonical_parent_key": "libero_goal/task_00/state_48",
        "suite": "libero_goal", "task_index": 0, "state_index": 48,
        "audit_status": "PASS", "remaining_policy_steps": 1,
    }]})

    def fake_run_once(template: str, *, candidate_path: Path, output_dir: Path, replicate: str, source_commit: str, source_tree: str, gpu: int = 0):
        result = {
            "status": "PASS", "exit_code": 0, "clean_success": True, "snapshot_restore_valid": True,
            "runtime_valid": True, "task_identity_valid": True, "metrics_finite": True,
            "artifact_validation_pass": True, "old_artifacts_reused": False,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
            "source_commit": source_commit,
            "source_tree": source_tree, "remaining_horizon_complete": True,
            "terminal_outcome": "SUCCESS", "terminal_state_sha256": "state",
            "key_state_identity_sha256": "identity", "canonical_parent_key": "libero_goal/task_00/state_48",
        }
        atomic_write_json(output_dir / "CONTROL_RESULT.json", result)
        return 0, result

    monkeypatch.setattr(control_qualification, "_run_once", fake_run_once)
    args = SimpleNamespace(
        candidate_manifest=manifest, output_dir=tmp_path / "qualification",
        runner_command="clean-only", source_commit="commit", source_tree="tree",
        source_clean_root=tmp_path / "clean",
        salt="test", initial_per_suite=1, batch_size=1, target_per_suite=1, suites="libero_goal", gpus="0,1",
    )
    report, rows, extras = control_qualification.qualify(args)
    assert report["status"] == "PASS"
    assert report["queue_progress"]["done"] == 2
    assert extras["audit"]["queue_states"] == {"DONE_VALID": 2}
    assert rows[0]["qualified"] is True
    assert extras["manifest"]["schema"] == "STAGE_V_FORMAL_PARENT_MANIFEST_V2"


def test_science_manifest_is_fresh_v1_identity_binding(tmp_path: Path) -> None:
    rows = [
        {
            "canonical_parent_key": f"{suite}/task_{index:02d}/state_48",
            "suite": suite,
            "task_index": index,
            "state_index": 48,
            "source_artifact_root": f"/clean/{suite}/task_{index:02d}/state_48",
            "old_artifacts_reused": False,
            "source_artifact_read": False,
        }
        for suite in control_qualification.EXPECTED_SUITES
        for index in range(10)
    ]
    manifest = control_qualification.build_science_parent_manifest(
        {
            "schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V2",
            "status": "PASS",
            "source_commit": "control-commit",
            "source_tree": "control-tree",
            "selected_parents": rows,
            "candidate_manifest_sha256": "candidate",
        },
        source_clean_root="/clean",
    )
    assert manifest["schema"] == "STAGE_V_FORMAL_PARENT_MANIFEST_V1"
    assert manifest["selected_count"] == 40
    assert manifest["source_commit"] == "control-commit"
    assert manifest["source_tree"] == "control-tree"
    assert all(row["old_artifacts_reused"] is False and row["source_artifact_read"] is False for row in manifest["selected_parents"])


def test_queue_projection_has_pending_running_complete_failed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    project_queue(root, [{"cell_id": "a", "state": "PENDING"}, {"cell_id": "b", "state": "RUNNING"}, {"cell_id": "c", "state": "DONE_VALID"}, {"cell_id": "d", "state": "FAILED_FATAL_POST_ACTION"}])
    assert (root / "QUEUE_PENDING" / "a.json").is_file()
    assert (root / "QUEUE_RUNNING" / "b.json").is_file()
    assert (root / "QUEUE_COMPLETE" / "c.json").is_file()
    assert (root / "QUEUE_FAILED" / "d.json").is_file()


def test_science_core_provenance_detects_tamper(tmp_path: Path) -> None:
    source = tmp_path / "runner.py"
    source.write_text("pass\n", encoding="utf-8")
    frozen = build_provenance([source], source_commit="b300", source_tree="968")
    receipt = tmp_path / "PROVENANCE.json"
    write_json(receipt, frozen)
    assert verify_provenance(receipt, expected_commit="b300", expected_tree="968")[0]
    source.write_text("tampered\n", encoding="utf-8")
    assert not verify_provenance(receipt, expected_commit="b300", expected_tree="968")[0]


def test_gpu_preflight_excludes_gpu5_and_waits_when_capacity_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.detector_v5.stage_v_dynamic_common.gpu_snapshot", lambda command: ([{"gpu_id": gpu, "memory_free_mib": 100000} for gpu in range(8)], None))
    result = gpu_preflight(required_count=8, excluded_gpus=[5], canary_peak_mib=1000)
    assert result["status"] == "PRELAUNCH_WAITING_FOR_8_GPUS"
    assert 5 not in result["safe_gpus"]


def test_gpu_preflight_selects_exactly_eight_when_more_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.detector_v5.stage_v_dynamic_common.gpu_snapshot", lambda command: ([{"gpu_id": gpu, "memory_free_mib": 100000} for gpu in range(9)], None))
    result = gpu_preflight(required_count=8, excluded_gpus=[5], canary_peak_mib=1000)
    assert result["status"] == "PASS"
    assert len(result["all_safe_gpus"]) == 8
    assert result["safe_gpus"] == result["all_safe_gpus"]
    assert 5 not in result["safe_gpus"]


@pytest.mark.skipif(os.name == "nt", reason="fcntl process locking is Linux production behavior")
def test_duplicate_launcher_lock_rejected(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock"
    first = ExclusiveLock(lock_path, {"supervisor_pid": os.getpid()})
    first.acquire(tmp_path / "run")
    second = ExclusiveLock(lock_path, {"supervisor_pid": os.getpid()})
    with pytest.raises(SupervisorError, match="DUPLICATE_SUPERVISOR"):
        second.acquire(tmp_path / "run")
    first.close()


@pytest.mark.skipif(os.name == "nt", reason="fcntl process locking is Linux production behavior")
def test_stale_lock_is_audited(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock"
    lock_path.write_text(json.dumps({"supervisor_pid": 999999}) + "\n", encoding="utf-8")
    lock = ExclusiveLock(lock_path, {"supervisor_pid": os.getpid()})
    lock.acquire(tmp_path / "run")
    assert (tmp_path / "run" / "STALE_LOCK_AUDIT.json").is_file()
    lock.close()


def test_ssh_probe_failure_is_telemetry_only(tmp_path: Path) -> None:
    args = make_args(tmp_path / "run")
    args.ssh_probe_command = "false"
    worker = sup.DynamicSupervisor(args)
    worker._probe_ssh()
    assert worker.ssh_failure_count == 1
    assert not (args.run_root / "ABORTED_INCOMPLETE.json").exists()


def test_local_heartbeat_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "LOCAL_HEARTBEAT.json"
    atomic_write_json(path, {"schema": "test", "heartbeat_count": 1})
    assert json.loads(path.read_text(encoding="utf-8"))["heartbeat_count"] == 1


def test_worker_heartbeat_file_is_preferred_over_legacy_status(tmp_path: Path) -> None:
    root = tmp_path / "run"
    worker_root = root / "worker_gpu0"
    worker_root.mkdir(parents=True)
    write_json(worker_root / "WORKER_STATUS.json", {"state": "RUNNING", "worker_pid": 1})
    write_json(worker_root / "WORKER_HEARTBEAT.json", {"state": "IDLE", "worker_pid": 2})
    worker = sup.DynamicSupervisor(make_args(root))
    rows = worker._worker_statuses()
    assert rows[0]["state"] == "IDLE"
    assert rows[0]["_heartbeat_file_present"] is True


def test_dead_worker_timeout_is_parent_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)).isoformat()
    write_json(root / "worker_gpu0" / "WORKER_STATUS.json", {
        "state": "RUNNING", "worker_pid": 999999, "child_pid": 999998, "worker_pgid": 999999,
        "gpu_id": 0, "current_parent": "libero_goal/task_00/state_48", "current_branch": "OPEN_T10",
        "updated_utc": old, "parent_started_epoch": 1, "last_progress_epoch": 1, "last_artifact_epoch": 1,
        "simulator_step": 0, "branch_progress": 0, "gpu_utilization_percent": 0,
    })
    worker = sup.DynamicSupervisor(make_args(root))
    _, errors = worker._resource_snapshot()
    assert "WORKER_HEARTBEAT_LOST_BOUND" in errors
    assert list((root / "TIMEOUT_RECEIPTS").glob("*.json"))


def test_branch_timeout_receipt_is_parent_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    now = time_now()
    old = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(root / "worker_gpu0" / "WORKER_HEARTBEAT.json", {
        "state": "RUNNING", "worker_pid": 999999, "child_pid": 999998,
        "worker_pgid": 999999, "gpu_id": 0,
        "worker_id": "stage-v-r2-gpu0", "current_parent": "libero_goal/task_00/state_48",
        "current_branch": "OPEN_T10", "updated_utc": old,
        "parent_started_epoch": now - 7200, "branch_started_epoch": now - 7200,
        "last_simulator_progress_epoch": now - 7200,
        "last_branch_progress_epoch": now - 7200, "last_artifact_epoch": now - 7200,
        "gpu_utilization_percent": 80,
    })
    worker = sup.DynamicSupervisor(make_args(root))
    worker.timeout_policy = {"branch_hard_seconds": 1, "parent_hard_seconds": 999999}
    _, errors = worker._resource_snapshot()
    assert "BRANCH_WATCHDOG_TIMEOUT_BOUND" in errors
    receipt = next((root / "TIMEOUT_RECEIPTS").glob("*.json"))
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["canonical_parent_key"] == "libero_goal/task_00/state_48"
    assert value["branch"] == "OPEN_T10"
    assert value["threshold_seconds"] == 1


def test_healthy_worker_with_fresh_heartbeat_is_not_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "worker_gpu0" / "WORKER_STATUS.json", {
        "state": "RUNNING", "worker_pid": os.getpid(), "child_pid": os.getpid(), "worker_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
        "gpu_id": 0, "current_parent": "libero_goal/task_00/state_48", "current_branch": "OPEN_T10",
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "parent_started_epoch": time_now() - 60,
        "last_progress_epoch": time_now() - 5, "last_artifact_epoch": time_now() - 5,
        "simulator_step": 10, "branch_progress": 3, "gpu_utilization_percent": 50,
    })
    worker = sup.DynamicSupervisor(make_args(root))
    _, errors = worker._resource_snapshot()
    assert not any("TIMEOUT" in item or "HEARTBEAT_LOST" in item for item in errors)


def test_stale_heartbeat_with_live_worker_is_not_heartbeat_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)).isoformat()
    now = time_now()
    write_json(root / "worker_gpu0" / "WORKER_HEARTBEAT.json", {
        "state": "RUNNING", "worker_pid": os.getpid(), "child_pid": os.getpid(),
        "worker_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(), "gpu_id": 0,
        "worker_id": "stage-v-r2-gpu0", "current_parent": "libero_goal/task_00/state_48",
        "current_branch": "OPEN_T10", "updated_utc": old,
        "parent_started_epoch": now - 60, "last_simulator_progress_epoch": now - 5,
        "last_branch_progress_epoch": now - 5, "last_artifact_epoch": now - 5,
    })
    worker = sup.DynamicSupervisor(make_args(root))
    _, errors = worker._resource_snapshot()
    assert "WORKER_HEARTBEAT_LOST_BOUND" not in errors


def test_dispatcher_pid_manifest_mismatch_is_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "RUN_MANIFEST.json", {"dispatcher_pid": 12345})
    worker = sup.DynamicSupervisor(make_args(root))
    worker.dispatcher = SimpleNamespace(pid=54321)
    _, errors = worker._resource_snapshot()
    assert "DISPATCHER_PID_MISMATCH" in errors


def time_now() -> float:
    import time
    return time.time()


def test_oom_counter_delta_is_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    monkeypatch.setattr(sup, "_mem_snapshot", lambda: {"available_ram_bytes": 1024**4, "available_ram_gib": 1024, "swap_used_bytes": 0, "swap_in": 0, "swap_out": 0, "oom_kill": 2})
    worker = sup.DynamicSupervisor(make_args(tmp_path / "run"))
    worker.baseline_oom = 1
    _, errors = worker._resource_snapshot()
    assert "OOM_KILL_COUNTER_INCREASED" in errors


def test_swap_two_samples_is_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    monkeypatch.setattr(sup, "_mem_snapshot", lambda: {"available_ram_bytes": 1024**4, "available_ram_gib": 1024, "swap_used_bytes": 1, "swap_in": 1, "swap_out": 1, "oom_kill": 0})
    worker = sup.DynamicSupervisor(make_args(tmp_path / "run"))
    worker.baseline_oom = 0
    assert "SWAP_NONZERO_TWO_SAMPLES" not in worker._resource_snapshot()[1]
    assert "SWAP_NONZERO_TWO_SAMPLES" in worker._resource_snapshot()[1]


def test_xid_is_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    monkeypatch.setattr(sup, "_xid_status", lambda start: "NVRM: Xid 31")
    worker = sup.DynamicSupervisor(make_args(tmp_path / "run"))
    _, errors = worker._resource_snapshot()
    assert "NVIDIA_XID" in errors


def test_gpu5_and_duplicate_worker_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_resources(monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for gpu in (0, 0, 5):
        write_json(root / f"worker_gpu{gpu}_{len(list(root.glob('worker*')))}" / "WORKER_STATUS.json", {
            "state": "RUNNING", "worker_pid": os.getpid(), "child_pid": os.getpid(), "gpu_id": gpu,
            "updated_utc": now, "current_parent": None,
        })
    worker = sup.DynamicSupervisor(make_args(root))
    _, errors = worker._resource_snapshot()
    assert "MULTIPLE_PROJECT_WORKERS_PER_GPU" in errors
    assert "UNAPPROVED_OR_GPU5_WORKER" in errors


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are Linux production behavior")
def test_abort_reaps_owned_process_group() -> None:
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    terminate_process_group(child, grace_seconds=1)
    assert child.poll() is not None


def test_science_artifact_validation_requires_complete_parent(tmp_path: Path) -> None:
    output = tmp_path / "attempt"
    parent = output / "libero_goal" / "task_00" / "state_48"
    parent.mkdir(parents=True)
    write_json(parent / "PARENT_RESULT.json", {"status": "PASS", "clean_success": True, "branch_count": 72, "canonical_parent_key": "libero_goal/task_00/state_48"})
    (parent / "COUNTERFACTUAL_BRANCHES.jsonl").write_text("{}\n", encoding="utf-8")
    assert science_artifact_status(output, "libero_goal/task_00/state_48")["valid"]
    (parent / "PARENT_RESULT.json").write_text("{}\n", encoding="utf-8")
    assert not science_artifact_status(output, "libero_goal/task_00/state_48")["valid"]


def test_strict_science_artifact_requires_lineage_branches_and_parent_seal(tmp_path: Path) -> None:
    output = tmp_path / "attempt"
    parent = output / "libero_goal" / "task_00" / "state_48"
    parent.mkdir(parents=True)
    key = "libero_goal/task_00/state_48"
    branches = []
    for arm in ("OPEN_T3", "OPEN_T5", "OPEN_T10"):
        for probe_step in range(24):
            branches.append({
                "canonical_parent_key": key, "probe_step": probe_step, "arm": arm,
                "control_arm": "NOOP_T10_REPLAY", "prefix_replay_exact": True,
                "comparison": {"control_status": "PASS", "open_status": "PASS", "label_status": "VALID"},
            })
    (parent / "COUNTERFACTUAL_BRANCHES.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in branches), encoding="utf-8",
    )
    write_json(parent / "PARENT_RESULT.json", {
        "status": "PASS", "clean_success": True, "branch_count": 72, "canonical_parent_key": key,
        "current_source_commit": "science-commit", "current_source_tree": "science-tree", "current_source_status": "",
        "suite": "libero_goal", "task_idx": 0, "state_id": 48,
    })
    sums = parent / "SHA256SUMS"
    sums.write_text(
        f"{sha256_file(parent / 'COUNTERFACTUAL_BRANCHES.jsonl')}  COUNTERFACTUAL_BRANCHES.jsonl\n"
        f"{sha256_file(parent / 'PARENT_RESULT.json')}  PARENT_RESULT.json\n",
        encoding="utf-8",
    )
    (parent / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    row = {"canonical_parent_key": key, "suite": "libero_goal", "task_index": 0, "state_index": 48}
    assert science_artifact_status(
        output, key, expected_source_commit="science-commit", expected_source_tree="science-tree", expected_row=row,
    )["valid"]
    sums.unlink()
    assert not science_artifact_status(
        output, key, expected_source_commit="science-commit", expected_source_tree="science-tree", expected_row=row,
    )["valid"]


def test_postmortem_is_read_only_and_falls_back_timeout_policy(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    write_json(old / "ABORTED_INCOMPLETE.json", {"status": "ABORTED_INCOMPLETE", "reason": "PARENT_WATCHDOG_TIMEOUT"})
    write_json(old / "LOCAL_HEARTBEAT.json", {"active_worker_pids": [1], "current_parent": None, "current_branch": None})
    before = (old / "ABORTED_INCOMPLETE.json").read_bytes()
    report = build_postmortem(old)
    policy = build_timeout_policy(report)
    assert policy["source"] == "fallback_due_to_insufficient_timestamps"
    assert (old / "ABORTED_INCOMPLETE.json").read_bytes() == before


def test_postmortem_reads_external_manifest_and_worker_summary(tmp_path: Path) -> None:
    old = tmp_path / "old"
    old.mkdir()
    manifest = tmp_path / "parent_manifest.json"
    write_json(manifest, {"parents": [
        {"canonical_parent_key": "p0"}, {"canonical_parent_key": "p1"},
    ]})
    write_json(old / "RUN_MANIFEST.json", {
        "parent_manifest": str(manifest), "parent_manifest_sha256": sha256_file(manifest),
        "map_layout": "6", "gpus": [1, 2, 3, 4, 6, 7],
    })
    write_json(old / "WORKER_GPU1_SUMMARY.json", {
        "parents": [{"canonical_parent_key": "p0", "clean_success": True, "branch_count": 72}],
    })
    report = build_postmortem(old, expected_parent_count=2)
    assert report["parent_counts"]["manifest_discovered"] == 2
    assert report["parent_counts"]["parent_results"] == 1
    assert report["parent_counts"]["missing_parent_keys"] == ["p1"]
    assert report["parent_counts"]["manifest"]["sha256_verified"] is True
    assert report["failure_mode_assessment"]["static_layout6_gpu_idle_tail"] is True


def test_timeout_policy_uses_separate_branch_and_parent_p95() -> None:
    policy = build_timeout_policy({"runtime_seconds": {
        "branch": {"p95": 2 * 3600}, "parent": {"p95": 5 * 3600},
    }})
    assert policy["source"] == "old_root_branch_and_parent_runtime_p95"
    assert policy["branch_soft_seconds"] == 4 * 3600
    assert policy["parent_soft_seconds"] == 10 * 3600
    assert policy["parent_hard_seconds"] == 16 * 3600


def test_stage_v2_primary_unit_is_open_t10() -> None:
    from scripts.detector_v5.stage_v2_teacher_enrichment import compute_report
    report = compute_report(
        [
            {"canonical_parent_key": "p", "arm": "OPEN_T10", "candidate_step": 1, "group": "teacher_corridor", "local_vulnerability": True, "task_vulnerability": True},
            {"canonical_parent_key": "p", "arm": "OPEN_T10", "candidate_step": 1, "group": "teacher_corridor", "local_vulnerability": True, "task_vulnerability": True},
            {"canonical_parent_key": "q", "arm": "OPEN_T10", "candidate_step": 1, "group": "background_random", "local_vulnerability": False, "task_vulnerability": False},
        ], config={"bootstrap": {"repetitions": 10, "seed": 1}}, execution_class="FORMAL", binding={}, input_summary={"invalid_branch_rows": 0},
    )
    assert report["primary_arm"] == "OPEN_T10"
    assert report["primary_summary"]["candidate_state_count"] == 2
    assert report["primary_summary"]["duplicate_source_rows"] == 1


def test_dynamic8_end_to_end_accepts_all_parents(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    manifest = tmp_path / "manifest.json"
    rows = [{"canonical_parent_key": f"libero_goal/task_{index:02d}/state_48", "suite": "libero_goal", "task_index": index, "state_index": 48} for index in range(8)]
    write_json(manifest, {"parents": rows})
    preflight = tmp_path / "preflight.json"
    write_json(preflight, {"status": "PASS", "safe_gpus": [0, 1, 2, 3, 4, 6, 7, 8]})
    root = tmp_path / "run"
    queue_db = root / "QUEUE.sqlite"
    lock = tmp_path / "lock"
    supervisor = repo / "scripts/detector_v5/run_stage_v_parent_aware_supervisor.py"
    dispatcher = repo / "scripts/detector_v5/run_stage_v_dynamic_dispatcher.py"
    auditor = repo / "scripts/detector_v5/audit_stage_v_dynamic_queue.py"
    canary_worker = repo / "scripts/detector_v5/stage_v_dynamic_canary_worker.py"
    worker_command = f"{sys.executable} {canary_worker} --parent-key {{parent_key}} --output-dir {{output_dir}} --source-commit {commit} --source-tree {tree}"
    command = [sys.executable, str(supervisor), "--run-root", str(root), "--repo-root", str(repo), "--parent-manifest", str(manifest), "--queue-db", str(queue_db), "--run-id", "e2e", "--expected-parent-count", "8", "--expected-source-commit", commit, "--expected-source-tree", tree, "--lock-path", str(lock), "--approved-gpus", "0,1,2,3,4,6,7,8", "--preflight-file", str(preflight), "--dispatcher-script", str(dispatcher), "--auditor-script", str(auditor), "--worker-command", worker_command, "--skip-resource-checks", "--poll-seconds", "0.1", "--worker-heartbeat-seconds", "0.1", "--heartbeat-stale-seconds", "5", "--min-available-ram-gib", "0"]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((root / "SUPERVISOR_COMPLETE.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads((root / "STAGE_V_COUNTERFACTUAL_AUDIT.json").read_text(encoding="utf-8"))["verdict"] == "PASS"


def test_abort_receipt_zeroes_accepted_count(tmp_path: Path) -> None:
    worker = sup.DynamicSupervisor(make_args(tmp_path / "run"))
    worker.root.mkdir()
    assert worker._abort("TEST_ABORT") == 1
    receipt = json.loads((worker.root / "ABORTED_INCOMPLETE.json").read_text(encoding="utf-8"))
    assert receipt["accepted_parent_results"] == 0


def test_r2b_preparation_is_hash_ordered_and_never_reuses_r2a(tmp_path: Path) -> None:
    from scripts.detector_v5 import prepare_stage_v_r2b_manifest as r2b

    r2a = tmp_path / "r2a"
    for index in range(40):
        suite = r2b.SUITES[index // 10]
        key = f"{suite}/task_{index % 10:02d}/state_48"
        parent = r2a / suite / f"task_{index % 10:02d}" / "state_48"
        parent.mkdir(parents=True)
        write_json(parent / "PARENT_RESULT.json", {"canonical_parent_key": key, "local_vulnerability": True})
    write_json(r2a / "STAGE_V_CLOSURE_RECEIPT.json", {
        "status": "STAGE_V_FORMAL_MAP_CLOSED", "accepted_parents": 40, "completed_branches": 2880,
    })
    write_json(r2a / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {"verdict": "PASS"})
    sums = r2a / "SHA256SUMS"
    files = sorted(path for path in r2a.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    sums.write_text("".join(f"{sha256_file(path)}  {path.relative_to(r2a).as_posix()}\n" for path in files), encoding="utf-8")
    (r2a / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")

    candidates = []
    for suite in r2b.SUITES:
        for index in range(20):
            candidates.append({"canonical_parent_key": f"{suite}/task_{index:02d}/state_48", "suite": suite, "task_index": index, "state_index": 48})
    candidate_manifest = tmp_path / "candidates.json"
    write_json(candidate_manifest, {"parents": candidates})
    r2a_manifest = tmp_path / "r2a-manifest.json"
    write_json(r2a_manifest, {"parents": [row for row in candidates if int(row["canonical_parent_key"].split("/")[1].removeprefix("task_")) < 10]})
    args = SimpleNamespace(
        r2a_root=r2a, r2a_manifest=r2a_manifest, candidate_manifest=candidate_manifest,
        output_root=tmp_path / "r2b", source_commit="commit", source_tree="tree",
        salt="test", parents_per_suite=10,
    )
    decision = r2b.prepare(args)
    assert decision["status"] == "R2B_REQUIRED"
    selected = decision["selected_parents"]
    assert len(selected) == 40
    assert not ({row["canonical_parent_key"] for row in selected} & {row["canonical_parent_key"] for row in json.loads(r2a_manifest.read_text(encoding="utf-8"))["parents"]})
