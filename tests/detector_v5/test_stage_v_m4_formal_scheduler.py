from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5 import run_stage_v_m4_formal_parent_with_resource_gate as parent_gate
from scripts.detector_v5 import run_stage_v_m4_formal_scheduler as scheduler
from scripts.detector_v5.stage_v_gpu_resource_contract import MIN_FREE_MEMORY_MIB, combine_inventory


UUID0 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
UUID1 = "11111111-2222-3333-4444-555555555555"


def _metadata() -> dict[str, object]:
    return {
        "worker_id": "m4_gpu0_test",
        "physical_gpu_index": 0,
        "gpu_id": 0,
        "gpu_uuid": UUID0,
        "cuda_visible_devices": "0",
        "worker_pid": 1234,
        "source_commit": "commit",
        "source_tree": "tree",
        "authority_sha256": "authority",
        "protocol_sha256": "protocol",
        "runtime_provenance_sha256": "provenance",
        "attempt_ordinal": 1,
    }


def _queue() -> dict[str, object]:
    return {"parent_keys": [f"suite/task_{index:02d}/state_48" for index in range(40)], "parent_count": 40}


def test_frozen_queue_has_exactly_40_unique_parent_keys() -> None:
    manifest = {
        "schema": "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2",
        "status": "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE",
        "parent_count": 40,
        "parents": [{"canonical_parent_key": f"suite/task_{index:02d}/state_48"} for index in range(40)],
    }
    queue = parent_gate._frozen_queue(manifest, manifest_sha="manifest", split_sha="split", exact_sha="exact", protocol_sha="protocol", authorization_sha="authority")
    assert queue["parent_count"] == 40
    assert len(queue["parent_keys"]) == 40
    assert len(set(queue["parent_keys"])) == 40


def test_concurrent_claim_race_has_one_winner_and_full_binding(tmp_path: Path) -> None:
    gate_root = tmp_path / "parent" / "gate"

    def claim() -> str:
        try:
            return str(parent_gate._claim(gate_root, 0, "suite/task_00/state_48", claim_metadata=_metadata()))
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sum(result.endswith("CLAIM.json") for result in results) == 1
    value = json.loads((gate_root / "CLAIM.json").read_text(encoding="utf-8"))
    for key, expected in _metadata().items():
        assert value[key] == expected
    assert value["claim_timestamp"] == value["claimed_utc"]


def test_claim_rejects_incomplete_identity_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CLAIM_IDENTITY_BINDING_MISSING"):
        parent_gate._claim(tmp_path / "gate", 0, "key", claim_metadata={"worker_id": "worker"})


def test_completed_and_hold_parent_cannot_be_reclaimed(tmp_path: Path) -> None:
    queue = {"parent_keys": ["suite/task_00/state_48"], "parent_count": 1}
    completed_root, completed_gate, _ = parent_gate._parent_paths(tmp_path / "completed", 0, queue["parent_keys"][0])
    parent_gate._claim(completed_gate, 0, queue["parent_keys"][0], claim_metadata=_metadata())
    (completed_gate / "PARENT_STATUS.json").write_text(json.dumps({"status": "PASS_FORMAL_M4_PARENT_ATOMIC"}), encoding="utf-8")
    assert scheduler._pending(queue, tmp_path / "completed", set()) == []

    hold_root, hold_gate, _ = parent_gate._parent_paths(tmp_path / "hold", 0, queue["parent_keys"][0])
    hold_metadata = _metadata() | {"worker_id": "m4_gpu1_test", "physical_gpu_index": 1, "gpu_id": 1, "gpu_uuid": UUID1, "cuda_visible_devices": "1"}
    parent_gate._claim(hold_gate, 0, queue["parent_keys"][0], claim_metadata=hold_metadata)
    (hold_gate / "PARENT_STATUS.json").write_text(json.dumps({"status": "HOLD_FORMAL_M4_STRUCTURAL_FAILURE"}), encoding="utf-8")
    with pytest.raises(scheduler.ResourceContractError, match="PARENT_HOLD_NOT_RECLAIMABLE"):
        scheduler._pending(queue, tmp_path / "hold", set())
    assert completed_root.is_dir() and hold_root.is_dir()


def test_global_hold_prevents_new_claims(tmp_path: Path) -> None:
    queue = _queue()
    (tmp_path / "GLOBAL_HOLD.json").write_text("{}", encoding="utf-8")
    assert scheduler._pending(queue, tmp_path, set()) == []


def test_resource_admission_excludes_leases_reservations_and_active_workers(tmp_path: Path) -> None:
    args = SimpleNamespace(source_worktree=tmp_path, runner=tmp_path / "runner.py")
    inventory = combine_inventory([
        {"gpu_id": 0, "uuid": f"GPU-{UUID0}", "memory_used_mib": 1, "memory_free_mib": MIN_FREE_MEMORY_MIB + 1, "utilization_gpu_percent": 0},
        {"gpu_id": 1, "uuid": f"GPU-{UUID1}", "memory_used_mib": 1, "memory_free_mib": MIN_FREE_MEMORY_MIB + 1, "utilization_gpu_percent": 0},
    ])
    assert scheduler._eligible_gpus(args, inventory, leased={0}, reserved=set(), assigned={1}) == []


def test_preexecution_failure_is_requeueable_but_claimed_parent_is_not(tmp_path: Path) -> None:
    queue = _queue()
    assert scheduler._pending(queue, tmp_path, set())[0] == (0, queue["parent_keys"][0])
    _, gate_root, _ = parent_gate._parent_paths(tmp_path, 0, queue["parent_keys"][0])
    parent_gate._claim(gate_root, 0, queue["parent_keys"][0], claim_metadata=_metadata())
    with pytest.raises(scheduler.ResourceContractError, match="ORPHANED_PARENT_CLAIM"):
        scheduler._pending(queue, tmp_path, set())


def test_completed_parent_releases_next_atomic_claim(tmp_path: Path) -> None:
    queue = _queue()
    _, gate_root, _ = parent_gate._parent_paths(tmp_path, 0, queue["parent_keys"][0])
    parent_gate._claim(gate_root, 0, queue["parent_keys"][0], claim_metadata=_metadata())
    (gate_root / "PARENT_STATUS.json").write_text(json.dumps({"status": "PASS_FORMAL_M4_PARENT_ATOMIC"}), encoding="utf-8")
    assert scheduler._pending(queue, tmp_path, set())[0] == (1, queue["parent_keys"][1])


def test_queue_order_does_not_depend_on_outcomes(tmp_path: Path) -> None:
    queue = _queue()
    first = scheduler._pending(queue, tmp_path, set())
    (tmp_path / "outcome-like.json").write_text(json.dumps({"status": "PASS", "label": True}), encoding="utf-8")
    assert scheduler._pending(queue, tmp_path, set()) == first


def test_model_path_is_bound_by_frozen_parent_suite(tmp_path: Path) -> None:
    model = tmp_path / "libero-10"
    model.mkdir()
    args = SimpleNamespace(model_path=tmp_path / "fallback")
    protocol = {"inputs": {"model_paths": {"libero_10": str(model)}}}
    assert scheduler._model_path_for_parent(protocol, "libero_10/task_00/state_48", args) == model.resolve()
    with pytest.raises(scheduler.ResourceContractError, match="MODEL_PATH_BINDING_MISSING"):
        scheduler._model_path_for_parent(protocol, "libero_goal/task_00/state_48", args)
