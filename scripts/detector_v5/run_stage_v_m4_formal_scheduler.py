"""Dispatch frozen formal-M4 parent bundles from one atomic global queue."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - the runtime target is Linux
    fcntl = None  # type: ignore[assignment]

try:
    from . import run_stage_v_m4_formal_parent_with_resource_gate as parent_gate
    from .stage_v_gpu_resource_contract import MODE_B, MIN_FREE_MEMORY_MIB, ResourceContractError, admit_mode_b_or_c, query_inventory
except ImportError:  # direct server execution
    import run_stage_v_m4_formal_parent_with_resource_gate as parent_gate  # type: ignore
    from stage_v_gpu_resource_contract import MODE_B, MIN_FREE_MEMORY_MIB, ResourceContractError, admit_mode_b_or_c  # type: ignore
    from stage_v_gpu_resource_contract import query_inventory  # type: ignore


MAX_PHYSICAL_GPUS = 8
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_create_only(path: Path, value: Mapping[str, Any]) -> bool:
    return parent_gate._create_only(path, value)


def _reservation_gpu_ids(paths: list[Path]) -> set[int]:
    reserved: set[int] = set()
    for path in paths:
        if not path.is_file():
            raise ResourceContractError(f"EXTERNAL_RESERVATION_ROOT_MISSING:{path}")
        value = parent_gate._load(path)
        if value.get("status") != "ACTIVE":
            continue
        leases = value.get("leases")
        if isinstance(leases, list):
            reserved.update(int(row["gpu_id"]) for row in leases if isinstance(row, Mapping) and row.get("state") == "ACTIVE")
        else:
            reserved.update(int(gpu) for gpu in value.get("gpu_ids", []))
    return reserved


def _queue_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    contract = protocol.get("resource_contract")
    if not isinstance(contract, Mapping):
        raise ResourceContractError("RESOURCE_CONTRACT_MISSING")
    required_true = ("atomic_global_queue", "dynamic_gpu_admission", "rolling_replenishment", "formal_parent_atomicity", "retry_only_preexecution")
    if any(contract.get(key) is not True for key in required_true):
        raise ResourceContractError("GLOBAL_QUEUE_CONTRACT_NOT_FROZEN")
    if int(contract.get("minimum_free_memory_mib", -1)) != MIN_FREE_MEMORY_MIB or contract.get("strict_comparison") != "free_memory_mib > minimum_free_memory_mib":
        raise ResourceContractError("GLOBAL_QUEUE_THRESHOLD_CONTRACT_INVALID")
    if int(contract.get("maximum_project_workers_per_gpu", -1)) != 1 or int(contract.get("max_concurrent_project_workers", -1)) > MAX_PHYSICAL_GPUS:
        raise ResourceContractError("GLOBAL_QUEUE_WORKER_CAP_INVALID")
    return dict(contract)


def _reservation_paths(protocol: Mapping[str, Any], args: argparse.Namespace) -> list[Path]:
    explicit = [Path(item).resolve() for item in args.reservation_root]
    if explicit:
        return explicit
    contract = protocol.get("resource_contract", {})
    values = contract.get("external_project_reservation_roots", []) if isinstance(contract, Mapping) else []
    return [Path(str(item)).resolve() for item in values]


def _model_path_for_parent(protocol: Mapping[str, Any], parent_key: str, args: argparse.Namespace) -> Path:
    mapping = protocol.get("inputs", {}).get("model_paths") if isinstance(protocol.get("inputs"), Mapping) else None
    if not isinstance(mapping, Mapping):
        return args.model_path
    suite = str(parent_key).split("/", 1)[0]
    value = mapping.get(suite)
    if not isinstance(value, str) or not value:
        raise ResourceContractError(f"MODEL_PATH_BINDING_MISSING:{suite}")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ResourceContractError(f"MODEL_PATH_MISSING:{suite}:{path}")
    return path


@contextmanager
def _scheduler_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise ResourceContractError("SCHEDULER_FLOCK_UNAVAILABLE")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ResourceContractError("GLOBAL_SCHEDULER_ALREADY_ACTIVE") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _attempt_ordinal(gate_root: Path) -> int:
    attempts = []
    for path in gate_root.glob("DISPATCH_*.json"):
        try:
            attempts.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return max(attempts, default=0) + 1


def _pending(queue: Mapping[str, Any], root: Path, assigned: set[int]) -> list[tuple[int, str]]:
    if (root / "GLOBAL_HOLD.json").exists():
        return []
    result = []
    for index, key in enumerate(queue["parent_keys"]):
        if index in assigned:
            continue
        _, gate_root, _ = parent_gate._parent_paths(root, index, str(key))
        claim = gate_root / "CLAIM.json"
        status = gate_root / "PARENT_STATUS.json"
        if claim.exists() and not status.exists():
            raise ResourceContractError(f"ORPHANED_PARENT_CLAIM:{index}")
        if claim.exists():
            if parent_gate._load(status).get("status") != "PASS_FORMAL_M4_PARENT_ATOMIC":
                raise ResourceContractError(f"PARENT_HOLD_NOT_RECLAIMABLE:{index}")
            continue
        result.append((index, str(key)))
    return result


def _eligible_gpus(args: argparse.Namespace, inventory: list[dict[str, Any]], *, leased: set[int], reserved: set[int], assigned: set[int]) -> list[int]:
    excluded = leased | reserved | assigned
    admission = admit_mode_b_or_c(
        inventory,
        mode=MODE_B,
        leased_gpu_ids=excluded,
        project_process_tokens=(str(args.source_worktree.resolve()), str(args.runner.resolve()), str(Path(__file__).resolve())),
        minimum_free_mib=MIN_FREE_MEMORY_MIB,
    )
    return [int(row["gpu_id"]) for row in admission["gpu_decisions"] if row.get("safe") and row.get("memory_free_mib") is not None and float(row["memory_free_mib"]) > MIN_FREE_MEMORY_MIB]


def _child_command(args: argparse.Namespace, index: int, gpu: int, attempt: int, runtime_provenance_sha: str, model_path: Path) -> list[str]:
    return [
        str(args.python), str(args.parent_gate),
        "--protocol", str(args.protocol), "--authorization", str(args.authorization),
        "--launch-gate-binding", str(args.launch_gate_binding), "--final-manifest", str(args.final_manifest),
        "--final-split", str(args.final_split), "--exact-plan-root", str(args.exact_plan_root),
        "--source-worktree", str(args.source_worktree), "--runner", str(args.runner),
        "--python", str(args.python), "--official-snapshot-root", str(args.official_snapshot_root),
        "--upstream-root", str(args.upstream_root), "--model-path", str(model_path),
        "--output-root", str(args.output_root), "--source-commit", args.source_commit,
        "--source-tree", args.source_tree, "--parent-index", str(index), "--gpu", str(gpu),
        "--attempt-ordinal", str(attempt), "--minimum-free-mib", str(MIN_FREE_MEMORY_MIB),
        "--runtime-provenance-sha256", runtime_provenance_sha,
    ]


def _verify_claim(path: Path, *, index: int, key: str, gpu: int, gpu_uuid: str, pid: int, args: argparse.Namespace, authority_sha: str, protocol_sha: str, runtime_provenance_sha: str) -> bool:
    if not path.is_file():
        return False
    claim = parent_gate._load(path)
    return (
        claim.get("parent_index") == index and claim.get("canonical_parent_key") == key
        and claim.get("worker_id") == f"formal-m4-parent-{index:02d}"
        and int(claim.get("physical_gpu_index", -1)) == gpu and str(claim.get("gpu_uuid")) == gpu_uuid
        and claim.get("cuda_visible_devices") == str(gpu) and int(claim.get("worker_pid", -1)) == pid
        and claim.get("source_commit") == args.source_commit and claim.get("source_tree") == args.source_tree
        and claim.get("authority_sha256") == authority_sha and claim.get("protocol_sha256") == protocol_sha
        and claim.get("runtime_provenance_sha256") == runtime_provenance_sha and int(claim.get("attempt_ordinal", 0)) >= 1
        and isinstance(claim.get("claim_timestamp"), str) and bool(claim.get("claim_timestamp"))
        and claim.get("outcomes_read") is False and claim.get("protected_counters") == COUNTERS
    )


def _write_progress(root: Path, queue: Mapping[str, Any], active: Mapping[int, Mapping[str, Any]], eligible: list[int]) -> None:
    claimed = completed = hold = unclaimed = 0
    parent_states: dict[str, str] = {}
    active_indices = {int(row["parent_index"]) for row in active.values()}
    for index, key in enumerate(queue["parent_keys"]):
        _, gate_root, _ = parent_gate._parent_paths(root, index, str(key))
        claim = gate_root / "CLAIM.json"
        status = gate_root / "PARENT_STATUS.json"
        if status.is_file():
            try:
                state = parent_gate._load(status).get("status")
            except (OSError, ValueError, json.JSONDecodeError):
                state = "HOLD"
            if state == "PASS_FORMAL_M4_PARENT_ATOMIC":
                completed += 1
                parent_states[str(key)] = "COMPLETED"
            else:
                hold += 1
                parent_states[str(key)] = "HOLD"
        elif claim.is_file() or index in active_indices:
            claimed += 1
            parent_states[str(key)] = "CLAIMED"
        else:
            unclaimed += 1
            parent_states[str(key)] = "UNCLAIMED"
    parent_gate._write(root / "PROGRESS.json", {
        "schema": "STAGE_V_M4_FORMAL_GLOBAL_PROGRESS_V1", "total_parents": len(queue["parent_keys"]),
        "unclaimed": unclaimed, "claimed": claimed, "completed": completed, "hold": hold,
        "parent_states": parent_states,
        "active_workers": [{key: value for key, value in row.items() if key != "process"} for row in active.values()],
        "eligible_gpus": eligible, "completed_branches": completed * 96,
        "accounted_treatment_branches": completed * 72, "outcomes_read": False,
        "protected_counters": dict(COUNTERS), "updated_utc": _utc(),
    })


def _global_hold(root: Path, *, reason: str, active: Mapping[int, Mapping[str, Any]]) -> None:
    _write_create_only(root / "GLOBAL_HOLD.json", {
        "schema": "STAGE_V_M4_FORMAL_GLOBAL_HOLD_V3",
        "status": "HOLD_STOP_GLOBAL_SCHEDULING",
        "reason": reason,
        "active_workers": [{key: value for key, value in row.items() if key != "process"} for row in active.values()],
        "outcomes_read": False,
        "intervention_executed": False,
        "protected_counters": dict(COUNTERS),
        "created_utc": _utc(),
    })


def run(args: argparse.Namespace) -> int:
    for name in ("protocol", "authorization", "launch_gate_binding", "final_manifest", "final_split", "exact_plan_root", "source_worktree", "runner", "python", "official_snapshot_root", "upstream_root", "model_path", "output_root", "parent_gate"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.source_commit == "" or args.source_tree == "":
        raise ValueError("SOURCE_BINDING_REQUIRED")
    protocol = parent_gate._load(args.protocol)
    authority = parent_gate._load(args.authorization)
    _queue_contract(protocol)
    parent_gate._verify_runtime_snapshot(args, authority)
    parent_gate._verify_launch_gate_binding(args, authority)
    formal = parent_gate.validate_formal_m4_v2_authority(
        protocol, protocol_path=args.protocol, split_path=args.final_split,
        source_commit=args.source_commit, source_tree=args.source_tree, authorization=authority,
    )
    runtime_provenance_sha = str(
        authority.get("runtime_provenance_sha256", "")
        or authority.get("authority_bindings", {}).get("runtime_provenance_sha256", "")
        or authority.get("authority_bindings", {}).get("successor_runtime_provenance_sha256", "")
    )
    if not runtime_provenance_sha:
        raise ResourceContractError("RUNTIME_PROVENANCE_BINDING_MISSING")
    protocol_sha = parent_gate.sha256(args.protocol)
    authority_sha = parent_gate.sha256(args.authorization)
    manifest = parent_gate._load(args.final_manifest)
    queue = parent_gate._frozen_queue(
        manifest,
        manifest_sha=parent_gate.sha256(args.final_manifest),
        split_sha=parent_gate.sha256(args.final_split),
        exact_sha=formal["exact_plan_manifest_sha256"],
        protocol_sha=parent_gate.sha256(args.protocol),
        authorization_sha=parent_gate.sha256(args.authorization),
    )
    root = args.output_root
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("RUNTIME_ROOT_NOT_DIRECTORY")
    parent_gate._ensure_queue(root, queue)
    contract = _queue_contract(protocol)
    max_in_flight = min(MAX_PHYSICAL_GPUS, int(contract["max_concurrent_project_workers"]))
    reservation_paths = _reservation_paths(protocol, args)
    if not 0 <= max_in_flight <= MAX_PHYSICAL_GPUS:
        raise ValueError("MAX_IN_FLIGHT_INVALID")

    with _scheduler_lock(root / "GLOBAL_SCHEDULER.lock"):
        _write_create_only(root / "SCHEDULER_START.json", {
            "schema": "STAGE_V_M4_FORMAL_GLOBAL_SCHEDULER_START_V1",
            "status": "RUNNING", "pid": os.getpid(), "max_in_flight": max_in_flight,
            "source_commit": args.source_commit, "source_tree": args.source_tree,
            "protocol_sha256": parent_gate.sha256(args.protocol), "authority_sha256": parent_gate.sha256(args.authorization),
            "reservation_roots": [str(path) for path in reservation_paths], "outcomes_read": False,
            "protected_counters": dict(COUNTERS), "created_utc": _utc(),
        })
        active: dict[int, dict[str, Any]] = {}
        assigned: set[int] = set()
        while True:
            if (root / "GLOBAL_HOLD.json").exists():
                return 75
            for gpu, worker in list(active.items()):
                process = worker["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                index, key = worker["parent_index"], worker["parent_key"]
                _, gate_root, _ = parent_gate._parent_paths(root, index, key)
                claim_path = gate_root / "CLAIM.json"
                if return_code == 75 and not claim_path.exists():
                    active.pop(gpu)
                    assigned.discard(index)
                    continue
                if return_code != 0 or not _verify_claim(
                    claim_path, index=index, key=key, gpu=gpu, gpu_uuid=str(worker["gpu_uuid"]), pid=process.pid,
                    args=args, authority_sha=authority_sha, protocol_sha=protocol_sha,
                    runtime_provenance_sha=runtime_provenance_sha,
                ):
                    _global_hold(root, reason=f"WORKER_STRUCTURAL_FAILURE:{index}:{return_code}", active=active)
                    return 2
                active.pop(gpu)
                assigned.discard(index)

            pending = _pending(queue, root, assigned)
            _write_progress(root, queue, active, [])
            if not pending and not active:
                _write_create_only(root / "SCHEDULER_STATUS.json", {
                    "schema": "STAGE_V_M4_FORMAL_GLOBAL_SCHEDULER_STATUS_V1", "status": "PASS_QUEUE_COMPLETE",
                    "parent_count": queue["parent_count"], "source_commit": args.source_commit, "source_tree": args.source_tree,
                    "outcomes_read": False, "intervention_executed": False, "protected_counters": dict(COUNTERS), "completed_utc": _utc(),
                })
                return 0
            reserved = _reservation_gpu_ids(reservation_paths)
            inventory, query_error = query_inventory()
            if query_error:
                time.sleep(args.poll_seconds)
                continue
            lease_store = parent_gate.GpuLeaseStore(root / "GPU_LEASES.sqlite")
            leased = {int(row["gpu_id"]) for row in lease_store.active()}
            available = _eligible_gpus(args, inventory, leased=leased, reserved=reserved, assigned=set(active))
            _write_progress(root, queue, active, available)
            for gpu, (index, key) in zip(available, pending[: max(0, max_in_flight - len(active))]):
                _, gate_root, _ = parent_gate._parent_paths(root, index, key)
                gate_root.mkdir(parents=True, exist_ok=True)
                attempt = _attempt_ordinal(gate_root)
                slot = gate_root / f"DISPATCH_SLOT_{attempt:03d}.json"
                if not _write_create_only(slot, {"schema": "STAGE_V_M4_FORMAL_GLOBAL_DISPATCH_SLOT_V1", "parent_index": index, "canonical_parent_key": key, "gpu": gpu, "attempt_ordinal": attempt, "outcomes_read": False, "protected_counters": dict(COUNTERS)}):
                    _global_hold(root, reason=f"DISPATCH_SLOT_CONFLICT:{index}", active=active)
                    return 2
                log_path = gate_root / f"SCHEDULER_CHILD_{attempt:03d}.log"
                gpu_row = next((row for row in inventory if int(row.get("gpu_id", -1)) == gpu), {})
                gpu_uuid = str(gpu_row.get("gpu_uuid", ""))
                model_path = _model_path_for_parent(protocol, key, args)
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(_child_command(args, index, gpu, attempt, runtime_provenance_sha, model_path), cwd=args.source_worktree, stdout=log, stderr=subprocess.STDOUT, text=True)
                dispatch = gate_root / f"DISPATCH_{attempt:03d}.json"
                dispatched_at = _utc()
                if not _write_create_only(dispatch, {
                    "schema": "STAGE_V_M4_FORMAL_GLOBAL_TASK_DISPATCH_V1", "status": "ASSIGNED",
                    "parent_index": index, "canonical_parent_key": key, "worker_id": f"formal-m4-parent-{index:02d}",
                    "physical_gpu_index": gpu, "gpu_uuid": gpu_uuid, "cuda_visible_devices": str(gpu), "gpu_id": gpu,
                    "model_path": str(model_path),
                    "worker_pid": process.pid, "source_commit": args.source_commit, "source_tree": args.source_tree,
                    "authority_sha256": authority_sha, "protocol_sha256": protocol_sha, "runtime_provenance_sha256": runtime_provenance_sha,
                    "attempt_ordinal": attempt, "outcomes_read": False, "protected_counters": dict(COUNTERS), "created_utc": dispatched_at,
                    "dispatch_timestamp": dispatched_at,
                }):
                    _global_hold(root, reason=f"DISPATCH_CLAIM_CONFLICT:{index}", active=active)
                    return 2
                active[gpu] = {"process": process, "parent_index": index, "parent_key": key, "gpu_id": gpu, "gpu_uuid": gpu_uuid, "worker_pid": process.pid, "attempt_ordinal": attempt, "model_path": str(model_path)}
                assigned.add(index)
            time.sleep(args.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "authorization", "launch_gate_binding", "final_manifest", "final_split", "exact_plan_root", "source_worktree", "runner", "python", "official_snapshot_root", "upstream_root", "model_path", "output_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parent-gate", type=Path, default=Path(__file__))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--reservation-root", type=Path, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-free-mib", type=int, default=MIN_FREE_MEMORY_MIB)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (parent_gate.M4GovernanceError, ResourceContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD_FORMAL_M4_SCHEDULER", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
