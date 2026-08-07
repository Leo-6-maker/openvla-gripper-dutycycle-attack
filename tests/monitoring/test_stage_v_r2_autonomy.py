from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.monitoring import materialize_stage_v_r2_next_plan as materializer
from scripts.monitoring import run_stage_v_r2_mainline_orchestrator as orchestrator
from scripts.monitoring import run_stage_v_r2_plan_controller as controller


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_foreign_compute_process_blocks_only_its_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gpu_rows = "\n".join(f"{i}, GPU-{i}, A800, 80000, 1000, 79000, 0" for i in range(8))
    apps = "GPU-2, 1234, 1000"
    monkeypatch.setattr(orchestrator, "_query", lambda command: (gpu_rows if "query-gpu" in command else apps, None))
    monkeypatch.setattr(orchestrator, "_process_info", lambda pid: {
        "pid": pid, "cmdline": ["foreign"], "cwd": "/foreign", "start_ticks": 1, "uid": 1000,
    })
    result = orchestrator.gpu_preflight(
        stage="C0", required_gpus=6, excluded_gpus=[5], protected_pids=[], canary_peak_mib=1000,
        project_root=tmp_path,
    )
    assert result["status"] == "PASS"
    assert 2 not in result["safe_gpus"]
    assert 5 not in result["safe_gpus"]
    gpu2 = next(row for row in result["gpu_rows"] if row["index"] == 2)
    assert "FOREIGN_PROCESS_PRESENT" in gpu2["reasons"]


def test_stage_specific_boundary_allows_vis_but_not_r2a(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    auditor = tmp_path / "auditor.py"
    config = tmp_path / "config.json"
    receipt = tmp_path / "receipt.json"
    parent = tmp_path / "parent.json"
    for path in (runner, auditor, config, receipt, parent):
        path.write_text("{}\n", encoding="utf-8")

    def plan(stage: str, command: list[str]) -> Path:
        value = {
            "schema": orchestrator.PLAN_SCHEMA, "stage": stage, "source_commit": "c", "source_tree": "t",
            "cwd": str(tmp_path), "runner_path": str(runner), "runner_sha256": orchestrator.sha256_file(runner),
            "auditor_path": str(auditor), "auditor_sha256": orchestrator.sha256_file(auditor),
            "config_path": str(config), "config_sha256": orchestrator.sha256_file(config),
            "input_receipts": [{"path": str(receipt), "sha256": orchestrator.sha256_file(receipt)}],
            "parent_manifest": {"path": str(parent), "sha256": orchestrator.sha256_file(parent)},
            "output_root_template": str(tmp_path / f"{stage}_{{commit8}}_{{utc}}"), "command_template": command,
            "audit_command_template": ["python", "audit.py"], "completion_receipts": ["DONE.json"],
            "resource_policy": {"resource_kind": "GPU", "required_gpu_count": 8, "minimum_gpu_count": 8, "maximum_gpu_count": 8, "strict_gpu_count": True, "excluded_gpus": [5]},
            "lock_path": str(tmp_path / "lock"), "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
        }
        path = tmp_path / f"{stage}.json"
        write_json(path, value)
        return path

    with pytest.raises(orchestrator.OrchestratorError, match="PLAN_STAGE_FORBIDDEN_COMMAND"):
        orchestrator.validate_plan(plan("R2A", ["python", "run_pgd.py"]), source={"commit": "c", "tree": "t"})
    assert orchestrator.validate_plan(
        plan("R2A", ["python", "/data/gripper_attack_detector_goal_v2.py"]),
        source={"commit": "c", "tree": "t"},
    )["stage"] == "R2A"
    with pytest.raises(orchestrator.OrchestratorError, match="PLAN_STAGE_FORBIDDEN_COMMAND"):
        orchestrator.validate_plan(plan("R2A", ["python", "run_attack.py"]), source={"commit": "c", "tree": "t"})
    assert orchestrator.validate_plan(plan("VIS_SMALL_MATRIX", ["python", "run_pgd.py"]), source={"commit": "c", "tree": "t"})["stage"] == "VIS_SMALL_MATRIX"


def test_append_only_registry_chain_rejects_rewrite(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    auditor = tmp_path / "auditor.py"
    config = tmp_path / "config.json"
    receipt = tmp_path / "receipt.json"
    parent = tmp_path / "parent.json"
    for path in (runner, auditor, config, receipt, parent):
        path.write_text("{}\n", encoding="utf-8")

    def make_plan(path: Path, stage: str) -> None:
        write_json(path, {
            "schema": orchestrator.PLAN_SCHEMA, "stage": stage, "source_commit": "c", "source_tree": "t",
            "cwd": str(tmp_path), "runner_path": str(runner), "runner_sha256": orchestrator.sha256_file(runner),
            "auditor_path": str(auditor), "auditor_sha256": orchestrator.sha256_file(auditor),
            "config_path": str(config), "config_sha256": orchestrator.sha256_file(config),
            "input_receipts": [{"path": str(receipt), "sha256": orchestrator.sha256_file(receipt)}],
            "parent_manifest": {"path": str(parent), "sha256": orchestrator.sha256_file(parent)},
            "output_root_template": str(tmp_path / f"{stage}_{{commit8}}_{{utc}}"),
            "command_template": ["python", "run.py"], "audit_command_template": ["python", "audit.py"],
            "completion_receipts": ["DONE.json"], "resource_policy": {"resource_kind": "CPU_ONLY", "required_gpu_count": 0, "minimum_gpu_count": 0, "maximum_gpu_count": 0, "strict_gpu_count": False},
            "lock_path": str(tmp_path / f"{stage}.lock"), "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
        })

    plan = tmp_path / "plan.json"
    make_plan(plan, "STAGE_V2")
    first = materializer.append_registry(
        state_root=tmp_path, source_commit="c", source_tree="t", stage="STAGE_V2", plan_path=plan,
        upstream_receipts=[],
    )
    second_plan = tmp_path / "plan2.json"
    make_plan(second_plan, "STAGE_O")
    second = materializer.append_registry(
        state_root=tmp_path, source_commit="c", source_tree="t", stage="STAGE_O", plan_path=second_plan,
        upstream_receipts=[],
    )
    assert first.name == "PLAN_REGISTRY_V0001.json"
    assert second.name == "PLAN_REGISTRY_V0002.json"
    plans, version, digest, latest = orchestrator.verify_registry_chain(second, source={"commit": "c", "tree": "t"})
    assert version == 2 and digest == orchestrator.sha256_file(second) and latest == second.resolve()
    assert set(plans) == {"STAGE_V2", "STAGE_O"}
    tampered = json.loads(second.read_text(encoding="utf-8"))
    rewritten_plan = tmp_path / "rewritten-plan.json"
    rewritten_plan.write_bytes(plan.read_bytes())
    tampered["plans"][0]["path"] = str(rewritten_plan)
    tampered["plans"][0]["sha256"] = orchestrator.sha256_file(rewritten_plan)
    write_json(second, tampered)
    second.with_suffix(second.suffix + ".sha256").write_text(
        f"{orchestrator.sha256_file(second)}  {second.name}\n", encoding="utf-8"
    )
    with pytest.raises(orchestrator.OrchestratorError, match="PLAN_REGISTRY_APPEND_PREFIX_MISMATCH"):
        orchestrator.verify_registry_chain(second, source={"commit": "c", "tree": "t"})
    second.write_text(second.read_text(encoding="utf-8").replace('"version": 2', '"version": 1'), encoding="utf-8")
    with pytest.raises(orchestrator.OrchestratorError):
        orchestrator.verify_registry_chain(second, source={"commit": "c", "tree": "t"})


def test_q_pass_materializes_fresh_c0_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    source = {"commit": "new-source", "tree": "new-tree", "status_porcelain": ""}
    monkeypatch.setattr(controller, "source_binding", lambda _: dict(source))
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    rows = [{
        "canonical_parent_key": f"libero_goal/task_{i:02d}/state_48", "suite": "libero_goal",
        "task_index": i, "state_index": 48, "qualification_rank_sha256": f"{i:064x}",
    } for i in range(80)]
    candidate = tmp_path / "candidate.json"
    write_json(candidate, {"selected_parents": rows, "old_artifacts_reused": False, "source_artifacts_modified": False})
    formal_rows = rows[:40]
    formal = qualification / "STAGE_V_R2_PARENT_MANIFEST_A.json"
    write_json(formal, {"schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V2", "status": "PASS", "source_commit": "q", "source_tree": "qt", "selected_count": 40, "selected_parents": formal_rows})
    write_json(qualification / "CONTROL_QUALIFICATION_REPORT.json", {
        "status": "PASS", "source_commit": "q", "source_tree": "qt", "evaluated_rows": 160,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
    })
    write_json(qualification / "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", {"verdict": "PASS"})
    science = tmp_path / "science.json"
    write_json(science, {"verdict": "PASS"})
    state = tmp_path / "state"
    args = SimpleNamespace(
        repo_root=repo, state_root=state, qualification_root=qualification, candidate_manifest=candidate,
        science_provenance=science, lock_path=tmp_path / "controller.lock", python_executable="python",
        external_pid=1895889, poll_seconds=0.01, once=True,
    )
    instance = controller.PlanController(args)
    assert instance.tick() == "C0_PLAN_READY"
    registry = state / "PLAN_REGISTRY_V0001.json"
    assert registry.is_file()
    assert (state / "C0_DIAGNOSTIC_PARENT_MANIFEST.json").is_file()
    plans, version, _, _ = orchestrator.verify_registry_chain(registry, source=source)
    assert version == 1 and set(plans) == {"C0"}


def test_controller_materializes_full_chain_and_stops_after_direct_open_no_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    source = {"commit": "chain-source", "tree": "chain-tree", "status_porcelain": ""}
    monkeypatch.setattr(controller, "source_binding", lambda _: dict(source))
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    rows = [{
        "canonical_parent_key": f"libero_{('10' if i < 20 else 'goal' if i < 40 else 'object' if i < 60 else 'spatial')}/task_{i:02d}/state_48",
        "suite": "libero_10" if i < 20 else "libero_goal" if i < 40 else "libero_object" if i < 60 else "libero_spatial",
        "task_index": i, "state_index": 48, "qualification_rank_sha256": f"{i:064x}",
    } for i in range(80)]
    candidate = tmp_path / "candidate.json"
    write_json(candidate, {
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1", "candidates": rows,
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0},
    })
    formal = qualification / "Q2_PARENT_MANIFEST_A.json"
    write_json(formal, {
        "schema": "STAGE_Q2_PARENT_MANIFEST_A_V1", "status": "PASS", "source_commit": source["commit"],
        "source_tree": source["tree"], "selected_count": 40, "selected_parents": rows[:40],
    })
    write_json(qualification / "Q2_CONTROL_QUALIFICATION_REPORT.json", {
        "status": "PASS", "source_commit": source["commit"], "source_tree": source["tree"],
        "qualified_by_suite": {suite: 10 for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")},
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
    })
    write_json(qualification / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", {"verdict": "PASS"})
    science = tmp_path / "science.json"
    write_json(science, {"schema": "SCIENCE_RECEIPT", "status": "PASS"})
    state = tmp_path / "state"
    args = SimpleNamespace(
        repo_root=repo, state_root=state, qualification_root=qualification, candidate_manifest=candidate,
        science_provenance=science, lock_path=tmp_path / "controller.lock", python_executable="python",
        external_pid=1895889, poll_seconds=0.01, once=True, allow_gpu5=True,
    )
    instance = controller.PlanController(args)
    assert instance.tick() == "C0_PLAN_READY"
    write_json(state / "C0_AUDIT.json", {"status": "PASS"})

    def add_spec(stage: str, *, gpu: bool = False, decision: str | None = None) -> None:
        binding_dir = tmp_path / "bindings" / stage
        binding_dir.mkdir(parents=True, exist_ok=True)
        runner = binding_dir / "runner.py"
        auditor = binding_dir / "auditor.py"
        config = binding_dir / "config.json"
        parent = binding_dir / "parent.json"
        receipt = binding_dir / "receipt.json"
        for path in (runner, auditor, config, parent, receipt):
            path.write_text("{}\n", encoding="utf-8")
        policy = {
            "resource_kind": "GPU" if gpu else "CPU_ONLY", "required_gpu_count": 8 if gpu else 0,
            "minimum_gpu_count": 8 if gpu else 0, "maximum_gpu_count": 8 if gpu else 0,
            "strict_gpu_count": gpu, "excluded_gpus": [] if gpu else [], "gpu5_authorized": gpu,
            "protected_pids": [1895889], "canary_peak_mib": 0,
        }
        spec = {
            "schema": materializer.STAGE_SPEC_SCHEMA, "stage": stage,
            "source_commit": source["commit"], "source_tree": source["tree"],
            "runner_path": str(runner), "runner_path_sha256": materializer.sha256_file(runner),
            "auditor_path": str(auditor), "auditor_path_sha256": materializer.sha256_file(auditor),
            "config_path": str(config), "config_path_sha256": materializer.sha256_file(config),
            "parent_manifest": {"path": str(parent), "sha256": materializer.sha256_file(parent)},
            "input_receipts": [{"name": "receipt", "path": str(receipt), "sha256": materializer.sha256_file(receipt)}],
            "cwd": str(tmp_path), "python_executable": "python",
            "output_root_template": str(tmp_path / f"{stage}_{{commit8}}_{{utc}}"),
            "command_template": ["python", str(runner), "--output-root", "{output_root}"],
            "audit_command_template": ["python", str(auditor), "--output-root", "{output_root}"],
            "completion_receipts": ["DONE.json"], "resource_policy": policy,
            "lock_path": str(tmp_path / f".{stage}.lock"),
            "decision_receipt_names": ["STAGE_V_R2B_DECISION.json"] if decision else (["DIRECT_OPEN_TIMING_REPORT.json"] if stage == "DIRECT_OPEN_PILOT" else []),
        }
        write_json(state / f"{stage}_SPEC.json", spec)

    def complete(stage: str, receipt_name: str | None = None, status: str | None = None) -> None:
        write_json(state / f"{stage}_AUDIT.json", {"status": "PASS"})
        if receipt_name:
            root = state / f"{stage}_output"
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / receipt_name, {"status": status})
            write_json(state / f"{stage}_LAUNCH.json", {"output_root": str(root)})

    add_spec("R2A", gpu=True)
    assert instance.tick() == "R2A_PLAN_READY"
    complete("R2A")
    add_spec("R2B_DECISION", decision="R2B_NOT_REQUIRED")
    assert instance.tick() == "R2B_DECISION_PLAN_READY"
    complete("R2B_DECISION", "STAGE_V_R2B_DECISION.json", "R2B_NOT_REQUIRED")
    for stage in ("STAGE_V2", "STAGE_O", "STUDENT_FREEZE", "PILOT_QUALIFICATION"):
        add_spec(stage)
        assert instance.tick() == f"{stage}_PLAN_READY"
        complete(stage)
    add_spec("DIRECT_OPEN_PILOT")
    assert instance.tick() == "DIRECT_OPEN_PILOT_PLAN_READY"
    complete("DIRECT_OPEN_PILOT", "DIRECT_OPEN_TIMING_REPORT.json", "NO_GO")
    assert instance.tick() == controller.PIPELINE_COMPLETE_NO_VIS
