from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5.stage_v_gpu_resource_contract import (
    MODE_B,
    MODE_M35,
    MIN_FREE_MEMORY_MIB,
    GpuLeaseStore,
    ResourceContractError,
    admit_mode_b_or_c,
    combine_inventory,
    resolve_cuda_physical_uuid,
    verify_recheck,
    write_resource_receipt,
)


UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def inventory(*, free: int = MIN_FREE_MEMORY_MIB, process_rows=()):
    return combine_inventory(
        [{
            "gpu_id": 2,
            "uuid": f"GPU-{UUID}",
            "memory_used_mib": 123,
            "memory_free_mib": free,
            "utilization_gpu_percent": 97,
        }],
        process_rows,
        process_identity=lambda pid: {"owner": "foreign", "command": f"proc-{pid}"},
    )


def test_mode_b_allows_foreign_compute_when_memory_is_admitted() -> None:
    decision = admit_mode_b_or_c(
        inventory(process_rows=[{"gpu_uuid": UUID, "pid": 77, "process_name": "foreign.py", "used_memory_mib": 4096}]),
        mode=MODE_B,
    )
    assert decision["status"] == "PASS"
    assert decision["eligible_gpu_ids"] == [2]
    row = decision["gpu_decisions"][0]
    assert row["foreign_workload_present"] is True
    assert row["foreign_processes"][0]["owner"] == "foreign"
    assert row["utilization_gpu_percent"] == 97


def test_m35_mode_allows_registered_foreign_compute_when_memory_is_admitted() -> None:
    decision = admit_mode_b_or_c(
        inventory(process_rows=[{"gpu_uuid": UUID, "pid": 77, "process_name": "foreign.py", "used_memory_mib": 4096}]),
        mode=MODE_M35,
    )
    assert decision["status"] == "PASS"
    assert decision["eligible_gpu_ids"] == [2]
    assert decision["gpu_decisions"][0]["foreign_workload_present"] is True


def test_mode_b_rejects_memory_below_contract_even_with_idle_utilization() -> None:
    decision = admit_mode_b_or_c(inventory(free=MIN_FREE_MEMORY_MIB - 1), mode=MODE_B)
    assert decision["status"] == "HOLD_NO_ELIGIBLE_GPU"
    assert decision["eligible_gpu_ids"] == []
    assert "INSUFFICIENT_FREE_MEMORY" in decision["gpu_decisions"][0]["reasons"]


def test_project_lease_blocks_gpu_but_unknown_foreign_work_is_only_telemetry() -> None:
    decision = admit_mode_b_or_c(inventory(process_rows=[{"gpu_uuid": UUID, "pid": 77}]), mode=MODE_B,
                                 leased_gpu_ids=[2], project_pids=[77])
    row = decision["gpu_decisions"][0]
    assert row["safe"] is False
    assert row["reasons"] == ["PROJECT_LEASE_PRESENT", "PROJECT_WORKER_PRESENT"]
    assert row["foreign_processes"] == []


def test_gpu_lease_is_atomic_and_single_owner(tmp_path: Path) -> None:
    store = GpuLeaseStore(tmp_path / "leases.sqlite")
    args = {
        "gpu_id": 2, "gpu_uuid": UUID, "worker_id": "worker-a", "worker_pid": 100,
        "stage": "STAGE_O", "atomic_job_id": "parent-a", "source_commit": "commit",
        "source_tree": "tree", "runtime_root": tmp_path, "launch_snapshot": inventory()[0],
    }
    lease = store.acquire(**args)
    with pytest.raises(ResourceContractError, match="GPU_LEASE_BUSY:2"):
        store.acquire(**{**args, "worker_id": "worker-b", "worker_pid": 101})
    assert store.release(lease) is True
    assert store.active() == []


def test_stale_recovery_requires_identity_and_keeps_audit_row(tmp_path: Path) -> None:
    store = GpuLeaseStore(tmp_path / "leases.sqlite")
    lease = store.acquire(
        gpu_id=2, gpu_uuid=UUID, worker_id="worker-a", worker_pid=100, stage="STAGE_O",
        atomic_job_id="parent-a", source_commit="commit", source_tree="tree",
        runtime_root=tmp_path, launch_snapshot=inventory()[0],
    )
    with pytest.raises(ResourceContractError, match="IDENTITY_UNVERIFIED"):
        store.recover_stale(lease["lease_id"], pid_alive=False, identity_verified=False)
    assert store.recover_stale(lease["lease_id"], pid_alive=False, identity_verified=True)
    assert store.active() == []
    with store._connect() as conn:
        row = conn.execute("SELECT state FROM gpu_leases WHERE lease_id=?", (lease["lease_id"],)).fetchone()
    assert row["state"] == "RECOVERED_STALE"


def test_recheck_requires_same_physical_uuid_and_memory() -> None:
    with pytest.raises(ResourceContractError, match="UUID_MISMATCH"):
        verify_recheck({"gpu_id": 2, "gpu_uuid": "other", "memory_free_mib": MIN_FREE_MEMORY_MIB},
                       expected_gpu_id=2, expected_gpu_uuid=UUID)


def test_cuda_uuid_falls_back_to_bound_physical_inventory_when_torch_omits_uuid() -> None:
    value, source = resolve_cuda_physical_uuid(2, torch_device_uuid=None, cuda_visible_devices="2", inventory=inventory())
    assert value == UUID
    assert source == "CUDA_VISIBLE_DEVICES[0]+nvidia-smi_physical_index"


def test_cuda_uuid_rejects_direct_torch_mismatch() -> None:
    with pytest.raises(ResourceContractError, match="TORCH_GPU_UUID_MISMATCH"):
        resolve_cuda_physical_uuid(2, torch_device_uuid="GPU-other", cuda_visible_devices="2", inventory=inventory())
    with pytest.raises(ResourceContractError, match="MEMORY_INSUFFICIENT"):
        verify_recheck({"gpu_id": 2, "gpu_uuid": UUID, "memory_free_mib": MIN_FREE_MEMORY_MIB - 1},
                       expected_gpu_id=2, expected_gpu_uuid=UUID)


def test_resource_receipt_binds_lease_and_foreign_telemetry(tmp_path: Path) -> None:
    store = GpuLeaseStore(tmp_path / "leases.sqlite")
    lease = store.acquire(
        gpu_id=2, gpu_uuid=UUID, worker_id="worker-a", worker_pid=100, stage="STAGE_O",
        atomic_job_id="parent-a", source_commit="commit", source_tree="tree",
        runtime_root=tmp_path, launch_snapshot=inventory()[0],
    )
    path = tmp_path / "RESOURCE_PRE.json"
    write_resource_receipt(path, phase="PRE", gpu_snapshot=inventory(process_rows=[{"gpu_uuid": UUID, "pid": 77}])[0],
                           lease=lease, atomic_job_id="parent-a")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["gpu_uuid"] == UUID
    assert value["lease"]["source_commit"] == "commit"
    assert value["foreign_processes"][0]["pid"] == 77
