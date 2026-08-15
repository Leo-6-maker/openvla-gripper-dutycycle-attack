from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5 import run_stage_v_m4_formal_parent_with_resource_gate as gate
from scripts.detector_v5.stage_v_gpu_resource_contract import MIN_FREE_MEMORY_MIB, GpuLeaseStore, ResourceContractError, combine_inventory


UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _inventory(*, free: int = MIN_FREE_MEMORY_MIB + 1, gpu_id: int = 2) -> list[dict[str, object]]:
    return combine_inventory([{
        "gpu_id": gpu_id,
        "uuid": f"GPU-{UUID}",
        "memory_used_mib": 1,
        "memory_free_mib": free,
        "utilization_gpu_percent": 0,
    }])


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        gpu=2,
        minimum_free_mib=MIN_FREE_MEMORY_MIB,
        source_worktree=tmp_path,
        runner=tmp_path / "runner.py",
        output_root=tmp_path / "run",
        source_commit="commit",
        source_tree="tree",
    )


def test_parent_receipts_are_namespaced_and_create_only(tmp_path: Path) -> None:
    _, gate_a, _ = gate._parent_paths(tmp_path, 0, "libero_goal/task_01/state_48")
    _, gate_b, _ = gate._parent_paths(tmp_path, 1, "libero_goal/task_02/state_48")
    assert gate_a != gate_b
    receipt = gate_a / "JOB.json"
    assert gate._create_only(receipt, {"status": "first"}) is True
    assert gate._create_only(receipt, {"status": "second"}) is False
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "first"


def test_resource_hold_happens_before_claim_and_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "query_inventory", lambda: (_inventory(free=MIN_FREE_MEMORY_MIB - 1), None))
    monkeypatch.setattr(gate, "_legacy_controller_rows", lambda: [])
    gate_root = tmp_path / "parent" / "gate"
    store = GpuLeaseStore(tmp_path / "leases.sqlite")
    with pytest.raises(ResourceContractError, match="GPU_NOT_ADMITTED"):
        gate._admit_before_claim(_args(tmp_path), gate_root, store, "job", 0, "p")
    assert not (gate_root / "CLAIM.json").exists()
    assert store.active() == []


def test_legacy_controller_hold_happens_before_claim_and_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "query_inventory", lambda: (_inventory(), None))
    monkeypatch.setattr(gate, "_legacy_controller_rows", lambda: [{"pid": 7, "command": "legacy-controller"}])
    gate_root = tmp_path / "parent" / "gate"
    store = GpuLeaseStore(tmp_path / "leases.sqlite")
    with pytest.raises(ResourceContractError, match="LEGACY_CONTROLLER_PRESENT"):
        gate._admit_before_claim(_args(tmp_path), gate_root, store, "job", 0, "p")
    assert not (gate_root / "CLAIM.json").exists()
    assert store.active() == []


def test_legacy_scan_ignores_shared_tmux_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0,
        stdout="101 tmux new-session monitor_stage_v_goal.py\n202 python -u run_stage_v_r2_plan_controller.py\n",
    ))
    real_readlink = gate.os.readlink
    monkeypatch.setattr(gate.os, "readlink", lambda path: "/usr/bin/tmux" if "/101/" in path else "/usr/bin/python")
    rows = gate._legacy_controller_rows()
    monkeypatch.setattr(gate.os, "readlink", real_readlink)
    assert rows == [{"pid": 202, "command": "python -u run_stage_v_r2_plan_controller.py"}]


def test_two_workers_use_independent_parent_receipts_and_atomic_leases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "query_inventory", lambda: (_inventory(), None))
    monkeypatch.setattr(gate, "_legacy_controller_rows", lambda: [])
    store = GpuLeaseStore(tmp_path / "leases.sqlite")
    args = _args(tmp_path)
    claims = []
    for index in (0, 1):
        _, gate_root, _ = gate._parent_paths(tmp_path / "run", index, f"suite/task_{index}/state_48")
        _, lease = gate._admit_before_claim(args, gate_root, store, f"job-{index}", index, f"p-{index}")
        claims.append(gate._claim(gate_root, index, f"p-{index}"))
        assert store.release(lease, reason="TEST_RELEASE") is True
    assert claims[0] != claims[1]
    assert all(path.is_file() for path in claims)
    assert store.active() == []


def test_intervention_provenance_survives_independent_audit_failure(tmp_path: Path) -> None:
    output = tmp_path / "science"
    output.mkdir()
    (output / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl").write_text("{}\n", encoding="utf-8")
    evidence = gate._runtime_evidence(output, child_process_started=True, child_process_completed=True, independent_audit_pass=False)
    assert evidence["intervention_started"] is True
    assert evidence["intervention_completed"] is True
    assert evidence["outcomes_read"] is True
    assert evidence["independent_audit_pass"] is False


def test_formal_child_requires_exact_official_environment_entrypoint() -> None:
    gate._verify_official_environment_python(Path(gate.OFFICIAL_ENVIRONMENT_PYTHON))
    with pytest.raises(ValueError, match="OFFICIAL_ENVIRONMENT_PYTHON_PATH_MISMATCH"):
        gate._verify_official_environment_python(Path("/home/sz/miniconda3/envs/hallo/bin/python3.10"))


def test_launch_binding_rejects_authorization_hash_mismatch(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    authorization = tmp_path / "authorization.json"
    binding = tmp_path / "binding.json"
    protocol.write_text("{}\n", encoding="utf-8")
    authorization.write_text(json.dumps({
        "repository_head": "auth-head", "repository_tree": "auth-tree", "protected_counters": dict(gate.COUNTERS),
    }) + "\n", encoding="utf-8")
    for relative in gate.LAUNCH_GATE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    binding.write_text(json.dumps({
        "schema": "STAGE_V_M4_FORMAL_LAUNCH_GATE_BINDING_V1", "status": "PASS", "launch_gate_authorized": True,
        "formal_m4_authorized": True, "runtime_authorized": True,
        "authorization_sha256": "wrong", "protocol_sha256": gate.sha256(protocol),
        "minimum_free_memory_mib": MIN_FREE_MEMORY_MIB,
        "protected_counters": dict(gate.COUNTERS), "intervention_executed": False, "outcomes_read": False,
        "v_phys_generated": False, "repository_head": "head", "repository_tree": "tree",
        "authorization_repository_head": "auth-head", "authorization_repository_tree": "auth-tree",
        "runtime_file_sha256": {relative: gate.sha256(tmp_path / relative) for relative in gate.LAUNCH_GATE_FILES},
    }) + "\n", encoding="utf-8")
    binding.with_name(binding.name + ".sha256").write_text(
        f"{hashlib.sha256(binding.read_bytes()).hexdigest()}  {binding.name}\n", encoding="utf-8",
    )
    monkey = SimpleNamespace(
        launch_gate_binding=binding, authorization=authorization, protocol=protocol, source_worktree=tmp_path,
        minimum_free_mib=MIN_FREE_MEMORY_MIB,
    )
    original_git = gate._git
    gate._git = lambda _root, *args: "head" if args[-1] == "HEAD" else "tree"
    try:
        with pytest.raises(ValueError, match="M4_OUTER_AUTHORITY_SHA_MISMATCH"):
            gate._verify_launch_gate_binding(monkey, json.loads(authorization.read_text(encoding="utf-8")))
    finally:
        gate._git = original_git
