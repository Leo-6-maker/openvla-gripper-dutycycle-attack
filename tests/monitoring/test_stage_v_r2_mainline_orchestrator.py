from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.monitoring import run_stage_v_r2_mainline_orchestrator as orch


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(repo: Path) -> dict[str, str]:
    return {"commit": "commit", "tree": "tree", "status_porcelain": ""}


def make_plan(tmp_path: Path) -> tuple[dict[str, object], Path]:
    runner = tmp_path / "runner.py"
    auditor = tmp_path / "auditor.py"
    config = tmp_path / "config.json"
    receipt = tmp_path / "receipt.json"
    parent = tmp_path / "parent.json"
    runner.write_text("runner\n", encoding="utf-8")
    auditor.write_text("auditor\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    write_json(receipt, {"status": "PASS"})
    write_json(parent, {"parents": []})
    plan = {
        "schema": orch.PLAN_SCHEMA,
        "stage": "C0",
        "source_commit": "commit",
        "source_tree": "tree",
        "cwd": str(tmp_path),
        "runner_path": str(runner),
        "runner_sha256": sha(runner),
        "auditor_path": str(auditor),
        "auditor_sha256": sha(auditor),
        "config_path": str(config),
        "config_sha256": sha(config),
        "input_receipts": [{"name": "qualification", "path": str(receipt), "sha256": sha(receipt)}],
        "parent_manifest": {"path": str(parent), "sha256": sha(parent)},
        "output_root_template": str(tmp_path / "formal_{commit8}_{utc}"),
        "gpu_policy": {"required_count": 8, "excluded_gpus": [5], "protected_pids": [], "canary_peak_mib": 1000},
        "lock_path": str(tmp_path / "stage.lock"),
        "env": {"OMP_NUM_THREADS": "1"},
        "command_template": [sys.executable, "-c", "import time; time.sleep(5)", "{output_root}"],
        "audit_command_template": [sys.executable, "-c", "raise SystemExit(0)"],
        "completion_receipts": ["COMPLETE.json"],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
    }
    path = tmp_path / "C0_PLAN.json"
    write_json(path, plan)
    plan["_path"] = str(path)
    plan["_sha256"] = sha(path)
    return plan, path


def test_observer_is_not_a_launcher() -> None:
    text = Path(orch.__file__).read_text(encoding="utf-8")
    assert "run_stage_v_r2_mainline_monitor.py" not in text
    assert "STAGE_V_R2_MAINLINE_ORCHESTRATOR" in text


def test_unregistered_plan_is_wait_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    monkeypatch.setattr(orch, "source_binding", source)
    args = type("Args", (), {
        "repo_root": tmp_path, "state_root": tmp_path / "state", "lock_path": tmp_path / "orchestrator.lock",
        "qualification_root": qualification, "plan_registry": tmp_path / "missing.json", "external_pid": 0,
        "poll_seconds": 1, "once": True,
    })()
    instance = orch.Orchestrator(args)
    assert instance.tick() == orch.WAIT_QUALIFICATION
    state = json.loads((tmp_path / "state" / "STAGE_V_R2_ORCHESTRATOR_STATE.json").read_text(encoding="utf-8"))
    assert state["reason"] == "QUALIFICATION_INCOMPLETE"
    assert state["ssh_is_hard_stop"] is False


def test_bad_plan_input_sha_is_rejected(tmp_path: Path) -> None:
    plan, path = make_plan(tmp_path)
    plan["input_receipts"][0]["sha256"] = "0" * 64  # type: ignore[index]
    write_json(path, plan)
    with pytest.raises(orch.OrchestratorError, match="input_receipt_0_SHA256_MISMATCH"):
        orch.validate_plan(path, source=source(tmp_path))


def test_registry_requires_exact_source_and_registered_stage(tmp_path: Path) -> None:
    plan, plan_path = make_plan(tmp_path)
    registry = tmp_path / "registry.json"
    write_json(registry, {"schema": orch.REGISTRY_SCHEMA, "plans": [{"stage": "C0", "path": str(plan_path), "sha256": sha(plan_path)}]})
    plans, registry_sha = orch.load_registry(registry, source=source(tmp_path))
    assert set(plans) == {"C0"}
    assert registry_sha == sha(registry)


def test_duplicate_lock_is_rejected_and_stale_is_audited(tmp_path: Path) -> None:
    lock_path = tmp_path / "orchestrator.lock"
    first = orch.FileLock(lock_path, tmp_path, {"pid": os.getpid()})
    first.acquire()
    second = orch.FileLock(lock_path, tmp_path, {"pid": os.getpid()})
    with pytest.raises(orch.DuplicateOrchestrator):
        second.acquire()
    first.close()
    write_json(lock_path, {"pid": 2**31 - 1})
    stale = orch.FileLock(lock_path, tmp_path, {"pid": os.getpid()})
    stale.acquire()
    assert (tmp_path / "STALE_LOCK_AUDIT.json").is_file()
    stale.close()


def test_process_identity_rejects_pid_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orch, "_proc_identity", lambda pid: {"pid": pid, "start_ticks": 22, "cwd": str(tmp_path), "cmdline": ["runner", str(tmp_path)]})
    receipt = {"pid": 7, "start_ticks": 21, "cwd": str(tmp_path), "cmdline": ["runner", str(tmp_path)], "output_root": str(tmp_path)}
    assert not orch.process_identity_matches(receipt, expected_cwd=tmp_path, expected_root=tmp_path)


def test_gpu_preflight_strict_eight_and_gpu5_exclusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = "\n".join(f"{gpu}, GPU-{gpu}, A800, 80000, 1000, 79000" for gpu in range(9))
    monkeypatch.setattr(orch, "_query", lambda command: (rows if "query-gpu" in command else "", None))
    monkeypatch.setattr(orch, "_process_info", lambda pid: {"pid": pid, "cmdline": [], "cwd": None, "start_ticks": 1, "uid": 0})
    result = orch.gpu_preflight(stage="C0", required_gpus=8, excluded_gpus=[5], protected_pids=[], canary_peak_mib=1000, project_root=tmp_path)
    assert result["status"] == "PASS"
    assert len(result["safe_gpus"]) == 8
    assert 5 not in result["safe_gpus"]


def test_gpu_preflight_protected_process_is_local_exclusion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = "\n".join(f"{gpu}, GPU-{gpu}, A800, 80000, 1000, 79000" for gpu in range(9))
    apps = "GPU-0, 1895889, 1000"
    monkeypatch.setattr(orch, "_query", lambda command: (rows if "query-gpu" in command else apps, None))
    monkeypatch.setattr(orch, "_process_info", lambda pid: {"pid": pid, "cmdline": ["foreign"], "cwd": "/tmp", "start_ticks": 1, "uid": 0})
    result = orch.gpu_preflight(stage="C0", required_gpus=8, excluded_gpus=[5], protected_pids=[1895889], canary_peak_mib=1000, project_root=tmp_path)
    assert result["status"] == "PRELAUNCH_WAITING_FOR_8_GPUS"
    gpu0 = next(row for row in result["gpu_rows"] if row["index"] == 0)
    assert "PROTECTED_PROCESS_PRESENT" in gpu0["reasons"]
    assert 1895889 not in result["safe_gpus"]


def test_dependency_blocks_until_receipt_passes(tmp_path: Path) -> None:
    plan, _ = make_plan(tmp_path)
    root = tmp_path / "formal"
    root.mkdir()
    assert orch._completed(root, plan) is False
    write_json(root / "COMPLETE.json", {"status": "PASS"})
    assert orch._completed(root, plan) is True


def test_duplicate_dispatcher_blocks_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, _ = make_plan(tmp_path)
    monkeypatch.setattr(orch, "active_project_dispatcher_pids", lambda _: [760179])
    monkeypatch.setattr(orch, "source_binding", source)
    args = type("Args", (), {
        "repo_root": tmp_path, "state_root": tmp_path / "state", "lock_path": tmp_path / "lock",
        "qualification_root": tmp_path, "plan_registry": tmp_path / "missing", "external_pid": 0,
        "poll_seconds": 1, "once": True,
    })()
    instance = orch.Orchestrator(args)
    with pytest.raises(orch.OrchestratorError, match="DUPLICATE_DISPATCHER_PRESENT"):
        instance._materialize("C0", plan, {"safe_gpus": list(range(8))})


def test_source_dirty_is_hard_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    monkeypatch.setattr(orch, "source_binding", lambda _: {"commit": "commit", "tree": "tree", "status_porcelain": " M dirty"})
    args = type("Args", (), {
        "repo_root": tmp_path, "state_root": tmp_path / "state", "lock_path": tmp_path / "lock",
        "qualification_root": qualification, "plan_registry": tmp_path / "missing", "external_pid": 0,
        "poll_seconds": 1, "once": True,
    })()
    instance = orch.Orchestrator(args)
    assert instance.tick() == orch.HARD_STOP
    state = json.loads((tmp_path / "state" / "STAGE_V_R2_ORCHESTRATOR_STATE.json").read_text(encoding="utf-8"))
    assert state["reason"] == "SOURCE_WORKTREE_DIRTY"


def test_exactly_once_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, _ = make_plan(tmp_path)
    monkeypatch.setattr(orch, "source_binding", source)
    args = type("Args", (), {
        "repo_root": tmp_path, "state_root": tmp_path / "state", "lock_path": tmp_path / "lock",
        "qualification_root": tmp_path, "plan_registry": tmp_path / "missing", "external_pid": 0,
        "poll_seconds": 1, "once": True,
    })()
    instance = orch.Orchestrator(args)
    resource = {"safe_gpus": [0, 1, 2, 3, 4, 6, 7, 8]}
    instance._materialize("C0", plan, resource)
    with pytest.raises(orch.OrchestratorError, match="COMMAND_ALREADY_MATERIALIZED"):
        instance._materialize("C0", plan, resource)
    launch = json.loads((tmp_path / "state" / "C0_LAUNCH.json").read_text(encoding="utf-8"))
    if orch.pid_alive(int(launch["pid"])):
        try:
            os.kill(int(launch["pid"]), signal.SIGTERM)
        except OSError:
            pass


def test_reconcile_existing_formal_root_writes_reattach_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, _ = make_plan(tmp_path)
    plan["stage"] = "R2A"
    plan["output_root_template"] = str(tmp_path / "STAGE_V_R2A_COUNTERFACTUAL_MAP_{commit8}_{utc}")
    plan["gpu_policy"] = {"required_count": 8, "excluded_gpus": [], "gpu5_authorized": True, "protected_pids": []}
    root = tmp_path / "STAGE_V_R2A_COUNTERFACTUAL_MAP_commit_20260808T000000Z"
    root.mkdir()
    write_json(root / "SUPERVISOR_START.json", {
        "run_root": str(root), "source_commit": "commit", "source_tree": "tree",
        "parent_manifest": plan["parent_manifest"]["path"], "parent_manifest_sha256": plan["parent_manifest"]["sha256"],
        "supervisor_pid": 123, "supervisor_pgid": 123, "started_utc": "now",
    })
    write_json(root / "RUN_MANIFEST.json", {"source_commit": "commit", "source_tree": "tree"})
    write_json(root / "LOCAL_HEARTBEAT.json", {"supervisor_pid": 123, "dispatcher_pid": 124, "gpu_assignments": []})
    monkeypatch.setattr(orch, "source_binding", source)
    monkeypatch.setattr(orch, "pid_alive", lambda pid: pid in {123, 124})
    monkeypatch.setattr(orch, "_proc_identity", lambda pid: {
        "pid": pid, "start_ticks": 9, "cwd": str(tmp_path),
        "cmdline": ["python", "supervisor", str(root)],
    })
    args = type("Args", (), {
        "repo_root": tmp_path, "state_root": tmp_path / "state", "lock_path": tmp_path / "lock",
        "qualification_root": tmp_path, "plan_registry": tmp_path / "missing", "external_pid": 0,
        "poll_seconds": 1, "once": True,
    })()
    instance = orch.Orchestrator(args)
    instance.registry_version = 2
    instance.plan_sha = "registry-sha"
    assert instance._reconcile_existing_root("R2A", plan) == "RUNNING"
    launch = json.loads((tmp_path / "state" / "R2A_LAUNCH.json").read_text(encoding="utf-8"))
    assert launch["reconciled_from_existing_root"] is True
    assert launch["output_root"] == str(root.resolve())
