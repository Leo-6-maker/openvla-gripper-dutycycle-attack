#!/usr/bin/env python3
"""Run one frozen formal-M4 parent behind the current authority/resource gate."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


OFFICIAL_ENVIRONMENT_PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"

try:
    from .stage_v_gpu_resource_contract import (
        MODE_B,
        MIN_FREE_MEMORY_MIB,
        GpuLeaseStore,
        ResourceContractError,
        admit_mode_b_or_c,
        query_inventory,
        verify_recheck,
        write_resource_receipt,
    )
    from .stage_v_m4_governance import (
        COUNTERS,
        M4GovernanceError,
        sha256,
        validate_formal_m4_v2_authority,
    )
except ImportError:  # direct server execution
    from stage_v_gpu_resource_contract import (  # type: ignore
        MODE_B,
        MIN_FREE_MEMORY_MIB,
        GpuLeaseStore,
        ResourceContractError,
        admit_mode_b_or_c,
        query_inventory,
        verify_recheck,
        write_resource_receipt,
    )
    from stage_v_m4_governance import (  # type: ignore
        COUNTERS,
        M4GovernanceError,
        sha256,
        validate_formal_m4_v2_authority,
    )


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_key(value: str) -> str:
    return value.replace("/", "__")


def _parent_paths(root: Path, parent_index: int, parent_key: str) -> tuple[Path, Path, Path]:
    """Return isolated parent, gate-receipt, and science-output roots."""
    parent_root = root / "parents" / f"{parent_index:02d}_{_safe_key(parent_key)}"
    return parent_root, parent_root / "gate", parent_root / "science"


def _create_only(path: Path, value: Mapping[str, Any]) -> bool:
    """Create a receipt without allowing a concurrent worker to clobber it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return True


def _next_receipt_path(root: Path, name: str) -> Path:
    """Keep retries append-only while retaining a stable first-attempt name."""
    target = root / f"{name}.json"
    if not target.exists():
        return target
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return root / f"{name}_{stamp}_{os.getpid()}.json"


def _runtime_evidence(output: Path, *, child_process_started: bool, child_process_completed: bool,
                      independent_audit_pass: bool) -> dict[str, Any]:
    """Derive execution provenance from materialized artifacts, never from final status."""
    branches = (output / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl").is_file()
    labels = (output / "M4_V_PHYS_LABELS_V1.jsonl").is_file()
    observations = (output / "M4_TREATMENT_OBSERVATIONS_V1.jsonl").is_file()
    materialized = branches or labels or observations
    if not child_process_started:
        state = "NOT_STARTED"
    elif not materialized:
        state = "CHILD_STARTED_NO_INTERVENTION_MARKER"
    elif child_process_completed:
        state = "INTERVENTION_COMPLETED_WITH_ARTIFACTS"
    else:
        state = "INTERVENTION_STARTED_PARTIAL"
    return {
        "child_process_started": child_process_started,
        "intervention_started": materialized,
        "intervention_completed": child_process_completed and materialized,
        "branch_artifacts_materialized": branches,
        "v_phys_artifacts_materialized": labels,
        "m4_outcomes_materialized": materialized,
        "outcomes_read": materialized,
        "outcomes_used_for_selection": False,
        "scheduler_outcomes_read": False,
        "investigator_outcomes_read": False,
        "outcomes_read_uncertain": child_process_started and not materialized,
        "independent_audit_pass": independent_audit_pass,
        "intervention_state": state,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _strict_free(row: Mapping[str, Any], minimum: int) -> bool:
    value = row.get("memory_free_mib")
    return value is not None and float(value) > minimum


def _verify_official_environment_python(path: Path) -> None:
    """Require the exact user-authorized environment entrypoint, not its target."""
    if str(path).replace("\\", "/") != OFFICIAL_ENVIRONMENT_PYTHON:
        raise ValueError("OFFICIAL_ENVIRONMENT_PYTHON_PATH_MISMATCH")


def _verify_runtime_snapshot(args: argparse.Namespace, authorization: Mapping[str, Any]) -> None:
    worktree = args.source_worktree.resolve()
    if _git(worktree, "status", "--porcelain"):
        raise ValueError("RUNTIME_WORKTREE_NOT_CLEAN")
    repository_head = str(authorization.get("repository_head", ""))
    if not repository_head or subprocess.run(
        ["git", "merge-base", "--is-ancestor", repository_head, _git(worktree, "rev-parse", "HEAD")],
        cwd=worktree,
        check=False,
    ).returncode != 0:
        raise ValueError("RUNTIME_REPOSITORY_HEAD_NOT_DESCENDANT_OF_AUTHORIZATION")
    if _git(worktree, "rev-parse", "HEAD^{tree}") == "":
        raise ValueError("RUNTIME_REPOSITORY_TREE_UNAVAILABLE")
    runtime_files = authorization.get("runtime_file_sha256", {})
    if not isinstance(runtime_files, Mapping):
        raise ValueError("RUNTIME_FILE_BINDING_MISSING")
    for relative, expected in runtime_files.items():
        path = worktree / str(relative)
        if not path.is_file() or sha256(path) != str(expected):
            raise ValueError(f"RUNTIME_FILE_SHA_MISMATCH:{relative}")
    protocol_path = args.protocol.resolve()
    if sha256(protocol_path) != authorization.get("protocol_sha256"):
        raise ValueError("M4_PROTOCOL_SHA_MISMATCH")


def _verify_launch_gate_binding(args: argparse.Namespace, authorization: Mapping[str, Any]) -> None:
    binding_path = args.launch_gate_binding.resolve()
    binding = _load(binding_path)
    if binding.get("schema") != "STAGE_V_M4_FORMAL_LAUNCH_GATE_BINDING_V1" or binding.get("status") != "PASS" or binding.get("launch_gate_authorized") is not True or binding.get("formal_m4_authorized") is not True or binding.get("runtime_authorized") is not True:
        raise ValueError("M4_OUTER_LAUNCH_GATE_BINDING_INVALID")
    sidecar = binding_path.with_name(binding_path.name + ".sha256")
    tokens = sidecar.read_text(encoding="utf-8").split() if sidecar.is_file() else []
    if not tokens or tokens[0] != sha256(binding_path):
        raise ValueError("M4_OUTER_LAUNCH_GATE_BINDING_SEAL_INVALID")
    if binding.get("authorization_sha256") != sha256(args.authorization) or binding.get("protocol_sha256") != sha256(args.protocol):
        raise ValueError("M4_OUTER_AUTHORITY_SHA_MISMATCH")
    if binding.get("protected_counters") != COUNTERS or authorization.get("protected_counters") != COUNTERS:
        raise ValueError("M4_OUTER_PROTECTED_BOUNDARY_INVALID")
    minimum_free_mib = int(binding.get("minimum_free_memory_mib", -1))
    protocol_contract = _load(args.protocol).get("resource_contract", {})
    if (minimum_free_mib != MIN_FREE_MEMORY_MIB
            or int(protocol_contract.get("minimum_free_memory_mib", -1)) != MIN_FREE_MEMORY_MIB
            or args.minimum_free_mib != minimum_free_mib):
        raise ValueError("M4_OUTER_RESOURCE_THRESHOLD_MISMATCH")
    if binding.get("intervention_executed") is not False or binding.get("outcomes_read") is not False or binding.get("v_phys_generated") is not False:
        raise ValueError("M4_OUTER_OUTCOME_BOUNDARY_INVALID")
    worktree = args.source_worktree.resolve()
    if binding.get("repository_head") != _git(worktree, "rev-parse", "HEAD") or binding.get("repository_tree") != _git(worktree, "rev-parse", "HEAD^{tree}"):
        raise ValueError("M4_OUTER_REPOSITORY_SNAPSHOT_MISMATCH")
    runtime_files = binding.get("runtime_file_sha256")
    if not isinstance(runtime_files, Mapping) or set(runtime_files) != set(LAUNCH_GATE_FILES):
        raise ValueError("M4_OUTER_RUNTIME_FILE_BINDING_INVALID")
    for relative in LAUNCH_GATE_FILES:
        path = worktree / relative
        if not path.is_file() or sha256(path) != str(runtime_files.get(relative)):
            raise ValueError(f"M4_OUTER_RUNTIME_FILE_SHA_MISMATCH:{relative}")
    if binding.get("authorization_repository_head") != authorization.get("repository_head") or binding.get("authorization_repository_tree") != authorization.get("repository_tree"):
        raise ValueError("M4_OUTER_AUTHORIZATION_SNAPSHOT_MISMATCH")


def _frozen_queue(manifest: Mapping[str, Any], *, manifest_sha: str, split_sha: str, exact_sha: str, protocol_sha: str, authorization_sha: str) -> dict[str, Any]:
    rows = manifest.get("parents")
    if manifest.get("schema") != "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2" or manifest.get("status") != "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE" or manifest.get("parent_count") != 40 or not isinstance(rows, list):
        raise ValueError("FINAL40_QUEUE_SOURCE_INVALID")
    keys = [str(row.get("canonical_parent_key", "")) for row in rows if isinstance(row, Mapping)]
    if len(keys) != 40 or len(set(keys)) != 40 or any(not key for key in keys):
        raise ValueError("FINAL40_QUEUE_KEYS_INVALID")
    return {
        "schema": "STAGE_V_M4_FORMAL_PARENT_QUEUE_V1",
        "status": "FROZEN_NOT_EXECUTED",
        "selection_rule": "FINAL_MANIFEST_ORDER_ONLY",
        "parent_count": 40,
        "parent_keys": keys,
        "formal_parent_manifest_sha256": manifest_sha,
        "formal_parent_split_sha256": split_sha,
        "exact_plan_manifest_sha256": exact_sha,
        "protocol_sha256": protocol_sha,
        "authorization_sha256": authorization_sha,
        "outcomes_read": False,
        "intervention_executed": False,
        "protected_counters": dict(COUNTERS),
        "created_utc": _utc(),
    }


def _ensure_queue(root: Path, queue: Mapping[str, Any]) -> None:
    path = root / "FORMAL_PARENT_QUEUE.json"
    if path.exists():
        existing = _load(path)
        compare = {key: value for key, value in existing.items() if key != "created_utc"}
        expected = {key: value for key, value in queue.items() if key != "created_utc"}
        if compare != expected:
            raise ValueError("FROZEN_PARENT_QUEUE_MISMATCH")
        return
    if any(root.iterdir()):
        raise ValueError("RUNTIME_ROOT_NONEMPTY_WITHOUT_FROZEN_QUEUE")
    _write(path, queue)


def _claim(root: Path, parent_index: int, parent_key: str, *, claim_metadata: Mapping[str, Any] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata = dict(claim_metadata or {})
    if metadata:
        required = ("worker_id", "physical_gpu_index", "gpu_uuid", "cuda_visible_devices", "worker_pid", "source_commit", "source_tree", "authority_sha256", "protocol_sha256", "runtime_provenance_sha256", "attempt_ordinal")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"CLAIM_IDENTITY_BINDING_MISSING:{','.join(missing)}")
        if int(metadata["attempt_ordinal"]) < 1:
            raise ValueError("CLAIM_ATTEMPT_ORDINAL_INVALID")
    path = root / "CLAIM.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            claimed_at = _utc()
            payload = {
                "schema": "STAGE_V_M4_FORMAL_PARENT_CLAIM_V1",
                "parent_index": parent_index,
                "canonical_parent_key": parent_key,
                "status": "CLAIMED_FOR_LAUNCH",
                "claimed_utc": claimed_at,
                "claim_timestamp": claimed_at,
                "outcomes_read": False,
                "outcomes_used_for_selection": False,
                "protected_counters": dict(COUNTERS),
            }
            payload.update(metadata)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"PARENT_ALREADY_CLAIMED_NO_RERUN:{parent_index}") from exc
    return path


LEGACY_CONTROLLER_TOKENS = (
    "monitor_stage_v_goal.py",
    "run_stage_v_r2_plan_controller.py",
    "run_stage_v_dynamic_dispatcher.py",
)

LAUNCH_GATE_FILES = (
    "scripts/detector_v5/run_stage_v_m4_formal_parent_with_resource_gate.py",
    "scripts/detector_v5/run_stage_v_m4_formal_scheduler.py",
    "scripts/detector_v5/stage_v_gpu_resource_contract.py",
    "scripts/detector_v5/run_stage_v_m4_matched_parent.py",
    "scripts/detector_v5/stage_v_m4_governance.py",
    "scripts/detector_v5/audit_stage_v_m4_matched_parent.py",
    "scripts/detector_v5/audit_stage_v_m4_static.py",
    "scripts/detector_v5/issue_stage_v_m4_runtime_authorization_v2.py",
)


def _project_tokens(args: argparse.Namespace) -> tuple[str, ...]:
    return (str(args.source_worktree.resolve()), str(args.runner.resolve()), str(Path(__file__).resolve()))


def _legacy_controller_rows() -> list[dict[str, Any]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False, timeout=10)
    if result.returncode != 0:
        raise ResourceContractError(f"CONTROLLER_QUERY_FAILED:{result.returncode}")
    rows: list[dict[str, Any]] = []
    for raw in result.stdout.splitlines():
        values = raw.strip().split(None, 1)
        if len(values) != 2:
            continue
        try:
            pid = int(values[0])
        except ValueError:
            continue
        command = values[1]
        if pid == os.getpid() or not any(token in command for token in LEGACY_CONTROLLER_TOKENS):
            continue
        try:
            executable = Path(os.readlink(f"/proc/{pid}/exe")).name
        except OSError:
            executable = ""
        # ponytail: shared tmux/bash wrappers retain historical argv; only a live
        # Python controller process is an authority conflict.
        if executable and not executable.startswith("python"):
            continue
        if any(token in command for token in LEGACY_CONTROLLER_TOKENS):
            rows.append({"pid": pid, "command": command})
    return rows


def _target(inventory: list[dict[str, Any]], gpu: int, *, minimum: int, leased: list[int], tokens: tuple[str, ...]) -> dict[str, Any]:
    admission = admit_mode_b_or_c(
        inventory,
        mode=MODE_B,
        leased_gpu_ids=leased,
        project_process_tokens=tokens,
        minimum_free_mib=minimum,
    )
    row = next((item for item in admission["gpu_decisions"] if int(item["gpu_id"]) == gpu), None)
    if row is None or not row.get("safe") or not _strict_free(row, minimum):
        raise ResourceContractError(f"GPU_NOT_ADMITTED_STRICT_FREE_GT_{minimum}:{gpu}")
    return {"admission": admission, "row": row}


def _verify_parent(output: Path, return_code: int) -> list[str]:
    reasons: list[str] = []
    result_path = output / "PARENT_RESULT.json"
    audit_path = output / "M4_INDEPENDENT_AUDIT.json"
    sums_path = output / "SHA256SUMS"
    seal_path = output / "SHA256SUMS.sha256"
    if return_code != 0:
        reasons.append(f"RUNNER_RETURN_CODE_{return_code}")
    if not result_path.is_file() or not audit_path.is_file() or not sums_path.is_file() or not seal_path.is_file():
        reasons.append("PARENT_SEAL_OR_AUDIT_MISSING")
        return reasons
    result, audit = _load(result_path), _load(audit_path)
    if result.get("status") != "PASS" or result.get("independent_audit_status") != "PASS_M4_PARENT_INDEPENDENT":
        reasons.append("PARENT_RESULT_NOT_PASS")
    if audit.get("status") != "PASS_M4_PARENT_INDEPENDENT":
        reasons.append("INDEPENDENT_AUDIT_NOT_PASS")
    if result.get("probe_count") != 24 or result.get("branch_count") != 96 or result.get("treatment_label_count") != 72:
        reasons.append("PARENT_ATOMIC_COUNTS_INVALID")
    if result.get("selection_outcomes_read") is not False or result.get("protected_counters") != COUNTERS:
        reasons.append("PARENT_BOUNDARY_INVALID")
    tokens = seal_path.read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != sha256(sums_path):
        reasons.append("PARENT_ROOT_SEAL_INVALID")
    return reasons


def _admit_before_claim(args: argparse.Namespace, gate_root: Path, lease_store: GpuLeaseStore,
                        job_id: str, parent_index: int, parent_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Complete controller/admission/lease/recheck before creating a claim."""
    lease: dict[str, Any] | None = None
    try:
        inventory, query_error = query_inventory()
        if query_error:
            raise ResourceContractError(f"GPU_INVENTORY_QUERY_FAILED:{query_error}")
        controllers = _legacy_controller_rows()
        _write(_next_receipt_path(gate_root, "CONTROLLER_AUDIT"), {
            "schema": "STAGE_V_M4_FORMAL_CONTROLLER_AUDIT_V1",
            "status": "HOLD_LEGACY_CONTROLLER_PRESENT" if controllers else "PASS_NO_CONFLICTING_LEGACY_CONTROLLER",
            "controllers": controllers, "outcomes_read": False,
            "outcomes_used_for_selection": False, "protected_counters": dict(COUNTERS),
        })
        if controllers:
            raise ResourceContractError("LEGACY_CONTROLLER_PRESENT")
        target = _target(inventory, args.gpu, minimum=args.minimum_free_mib,
                         leased=[int(row["gpu_id"]) for row in lease_store.active()],
                         tokens=_project_tokens(args))
        _write(_next_receipt_path(gate_root, "RESOURCE_ADMISSION"), {
            "schema": "STAGE_V_M4_FORMAL_RESOURCE_ADMISSION_V1", "status": "PASS",
            "gpu": target["row"], "admission": target["admission"],
            "partial_fleet_allowed": True, "foreign_workload_allowed": True,
            "outcomes_read": False, "outcomes_used_for_selection": False,
            "protected_counters": dict(COUNTERS),
        })
        lease = lease_store.acquire(
            gpu_id=args.gpu, gpu_uuid=str(target["row"]["gpu_uuid"]),
            worker_id=f"formal-m4-parent-{parent_index:02d}", worker_pid=os.getpid(),
            stage="FORMAL_M4_V2", atomic_job_id=job_id,
            source_commit=args.source_commit, source_tree=args.source_tree,
            runtime_root=args.output_root, launch_snapshot=target["row"],
        )
        _write(_next_receipt_path(gate_root, "LEASE_ACQUIRED"), {
            "schema": "STAGE_V_M4_FORMAL_GPU_LEASE_ACQUIRED_V1", "status": "PASS",
            "lease_id": lease["lease_id"], "gpu": target["row"],
            "atomic_job_id": job_id, "protected_counters": dict(COUNTERS),
        })
        rechecked_inventory, query_error = query_inventory()
        if query_error:
            raise ResourceContractError(f"GPU_RECHECK_QUERY_FAILED:{query_error}")
        rechecked = next((item for item in rechecked_inventory if int(item.get("gpu_id", -1)) == args.gpu), None)
        if rechecked is None:
            raise ResourceContractError("GPU_RECHECK_ID_MISSING")
        verify_recheck(rechecked, expected_gpu_id=args.gpu,
                       expected_gpu_uuid=str(target["row"]["gpu_uuid"]),
                       minimum_free_mib=args.minimum_free_mib)
        if not _strict_free(rechecked, args.minimum_free_mib):
            raise ResourceContractError(f"GPU_RECHECK_FREE_NOT_STRICTLY_GREATER_THAN_{args.minimum_free_mib}")
        rechecked_target = _target(rechecked_inventory, args.gpu, minimum=args.minimum_free_mib,
                                   leased=[], tokens=_project_tokens(args))
        _write(_next_receipt_path(gate_root, "RESOURCE_RECHECK"), {
            "schema": "STAGE_V_M4_FORMAL_RESOURCE_RECHECK_V1", "status": "PASS",
            "gpu": rechecked_target["row"], "admission": rechecked_target["admission"],
            "outcomes_read": False, "outcomes_used_for_selection": False,
            "protected_counters": dict(COUNTERS),
        })
        write_resource_receipt(_next_receipt_path(gate_root, "RESOURCE_PRE"), phase="PRE_LAUNCH_RECHECK",
                               gpu_snapshot=rechecked, lease=lease, atomic_job_id=job_id)
        return target, lease
    except Exception:
        if lease is not None:
            released = lease_store.release(lease, reason="PRELAUNCH_HOLD")
            _write(_next_receipt_path(gate_root, "RESOURCE_RELEASE"), {
                "schema": "STAGE_V_M4_FORMAL_RESOURCE_RELEASE_V1",
                "status": "PASS" if released else "HOLD_RELEASE_FAILED", "release_ok": released,
                "lease_id": lease["lease_id"], "atomic_job_id": job_id,
                "reason": "PRELAUNCH_HOLD", "outcomes_read": False,
                "outcomes_used_for_selection": False, "protected_counters": dict(COUNTERS),
            })
        raise


def run(args: argparse.Namespace) -> int:
    args.protocol = args.protocol.resolve()
    args.authorization = args.authorization.resolve()
    args.launch_gate_binding = args.launch_gate_binding.resolve()
    args.final_manifest = args.final_manifest.resolve()
    args.final_split = args.final_split.resolve()
    args.exact_plan_root = args.exact_plan_root.resolve()
    args.source_worktree = args.source_worktree.resolve()
    args.runner = args.runner.resolve()
    _verify_official_environment_python(args.python)
    args.python = args.python.resolve()
    args.output_root = args.output_root.resolve()
    if not 0 <= args.gpu <= 7:
        raise ValueError("GPU_INDEX_OUT_OF_AUTHORIZED_POOL")
    protocol, authorization = _load(args.protocol), _load(args.authorization)
    if authorization.get("formal_m4_authorized") is not True or authorization.get("owner_authorized") is not True:
        raise ValueError("FORMAL_M4_AUTHORIZATION_FLAG_FALSE")
    if authorization.get("runtime_authorized") is not True or authorization.get("outcomes_read") is not False or authorization.get("intervention_executed") is not False:
        raise ValueError("M4_AUTHORIZATION_BOUNDARY_INVALID")
    _verify_runtime_snapshot(args, authorization)
    _verify_launch_gate_binding(args, authorization)
    authority = validate_formal_m4_v2_authority(
        protocol,
        protocol_path=args.protocol,
        split_path=args.final_split,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
        authorization=authorization,
    )
    if Path(authority["manifest_path"]).resolve() != args.final_manifest or Path(authority["exact_plan_root"]).resolve() != args.exact_plan_root:
        raise ValueError("CURRENT_AUTHORITY_PATH_MISMATCH")
    manifest = _load(args.final_manifest)
    manifest_sha, split_sha, exact_sha = sha256(args.final_manifest), sha256(args.final_split), authority["exact_plan_manifest_sha256"]
    protocol_sha, authorization_sha = sha256(args.protocol), sha256(args.authorization)
    runtime_provenance_sha = str(
        getattr(args, "runtime_provenance_sha256", "")
        or authorization.get("runtime_provenance_sha256", "")
        or authorization.get("authority_bindings", {}).get("runtime_provenance_sha256", "")
        or authorization.get("authority_bindings", {}).get("successor_runtime_provenance_sha256", "")
    )
    if not runtime_provenance_sha:
        raise ValueError("RUNTIME_PROVENANCE_BINDING_MISSING")
    queue = _frozen_queue(manifest, manifest_sha=manifest_sha, split_sha=split_sha, exact_sha=exact_sha, protocol_sha=protocol_sha, authorization_sha=authorization_sha)
    if not 0 <= args.parent_index < 40:
        raise ValueError("PARENT_INDEX_OUT_OF_QUEUE")
    parent_key = queue["parent_keys"][args.parent_index]
    root_existed = args.output_root.exists()
    if not root_existed:
        args.output_root.mkdir(parents=True, exist_ok=False)
    elif not args.output_root.is_dir():
        raise ValueError("RUNTIME_ROOT_NOT_DIRECTORY")
    _ensure_queue(args.output_root, queue)
    if (args.output_root / "GLOBAL_HOLD.json").exists():
        raise ValueError("GLOBAL_M4_HOLD_ALREADY_SEALED")

    parent_root, gate_root, parent_output = _parent_paths(args.output_root, args.parent_index, parent_key)
    gate_root.mkdir(parents=True, exist_ok=True)
    claim_path = gate_root / "CLAIM.json"
    if claim_path.exists():
        raise ValueError(f"PARENT_ALREADY_CLAIMED_NO_RERUN:{args.parent_index}")
    job_id = f"FORMAL_M4_V2:{args.parent_index:02d}:{parent_key}"
    lease_store = GpuLeaseStore(args.output_root / "GPU_LEASES.sqlite")
    lease: dict[str, Any] | None = None

    # Admission is intentionally completed before _claim.  A resource/controller
    # hold therefore leaves no attempted-parent identity behind and can be retried
    # after the external condition changes.
    try:
        target, lease = _admit_before_claim(args, gate_root, lease_store, job_id,
                                            args.parent_index, parent_key)
    except Exception as exc:
        hold = {
            "schema": "STAGE_V_M4_FORMAL_PRELAUNCH_HOLD_V1",
            "status": "HOLD_PRELAUNCH_RETRYABLE", "parent_index": args.parent_index,
            "canonical_parent_key": parent_key, "claim_created": False,
            "child_process_started": False, "intervention_started": False,
            "reason": f"{type(exc).__name__}:{exc}", "no_rerun_to_pass": False,
            "outcomes_read": False, "outcomes_used_for_selection": False,
            "protected_counters": dict(COUNTERS), "created_utc": _utc(),
        }
        _write(_next_receipt_path(gate_root, "PRELAUNCH_HOLD"), hold)
        print(json.dumps(hold, sort_keys=True))
        return 75

    try:
        claim_path = _claim(
            gate_root,
            args.parent_index,
            parent_key,
            claim_metadata={
                "worker_id": lease["worker_id"],
                "physical_gpu_index": lease["gpu_id"],
                "gpu_id": lease["gpu_id"],
                "gpu_uuid": lease["gpu_uuid"],
                "cuda_visible_devices": str(args.gpu),
                "worker_pid": lease["worker_pid"],
                "source_commit": args.source_commit,
                "source_tree": args.source_tree,
                "authority_sha256": authorization_sha,
                "protocol_sha256": protocol_sha,
                "authorization_sha256": authorization_sha,
                "runtime_provenance_sha256": runtime_provenance_sha,
                "attempt_ordinal": args.attempt_ordinal,
            },
        )
    except Exception as exc:
        released = lease_store.release(lease, reason="CLAIM_CONFLICT")
        _write(_next_receipt_path(gate_root, "RESOURCE_RELEASE"), {
            "schema": "STAGE_V_M4_FORMAL_RESOURCE_RELEASE_V1",
            "status": "PASS" if released else "HOLD_RELEASE_FAILED", "release_ok": released,
            "lease_id": lease["lease_id"], "atomic_job_id": job_id,
            "reason": "CLAIM_CONFLICT", "outcomes_read": False,
            "outcomes_used_for_selection": False, "protected_counters": dict(COUNTERS),
        })
        hold = {
            "schema": "STAGE_V_M4_FORMAL_CLAIM_CONFLICT_HOLD_V1",
            "status": "HOLD_CLAIM_CONFLICT", "parent_index": args.parent_index,
            "canonical_parent_key": parent_key, "claim_created": False,
            "child_process_started": False, "intervention_started": False,
            "reason": f"{type(exc).__name__}:{exc}", "lease_released": released,
            "outcomes_read": False, "outcomes_used_for_selection": False,
            "protected_counters": dict(COUNTERS), "created_utc": _utc(),
        }
        _write(_next_receipt_path(gate_root, "CLAIM_CONFLICT_HOLD"), hold)
        print(json.dumps(hold, sort_keys=True))
        return 75
    _write(gate_root / "CLAIM_METADATA.json", {
        "schema": "STAGE_V_M4_FORMAL_PARENT_CLAIM_METADATA_V1",
        "status": "CLAIMED_FOR_LAUNCH", "claim": str(claim_path),
        "parent_index": args.parent_index, "canonical_parent_key": parent_key,
        "outcomes_read": False, "outcomes_used_for_selection": False,
        "protected_counters": dict(COUNTERS),
    })
    command = [
        str(args.python), str(args.runner), "--protocol", str(args.protocol),
        "--output-dir", str(parent_output), "--official-snapshot-root", str(args.official_snapshot_root),
        "--upstream-root", str(args.upstream_root), "--model-path", str(args.model_path),
        "--authorization-receipt", str(args.authorization), "--exact-plan-root", str(args.exact_plan_root),
        "--parent-key", parent_key, "--gpu", str(args.gpu), "--source-commit", args.source_commit,
        "--source-tree", args.source_tree, "--exact-plan-manifest-sha256", exact_sha, "--enable-runtime",
    ]
    _write(gate_root / "JOB.json", {
        "schema": "STAGE_V_M4_FORMAL_PARENT_JOB_V1", "status": "LAUNCH_PENDING",
        "parent_index": args.parent_index, "canonical_parent_key": parent_key,
        "command": command, "claim": str(claim_path), "outcomes_read": False,
        "outcomes_used_for_selection": False, "protected_counters": dict(COUNTERS),
    })
    log_path = gate_root / "SCIENCE_RUNNER.log"
    failure_reasons: list[str] = []
    child_process_started = False
    child_process_completed = False
    return_code: int | None = None
    process: subprocess.Popen[str] | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            process = subprocess.Popen(command, cwd=args.source_worktree, stdout=log,
                                       stderr=subprocess.STDOUT, text=True, env=child_env)
            child_process_started = True
            _write(gate_root / "CHILD_STARTED.json", {
                "schema": "STAGE_V_M4_FORMAL_CHILD_STARTED_V1", "status": "PASS",
                "pid": process.pid, "parent_index": args.parent_index,
                "canonical_parent_key": parent_key, "outcomes_read": False,
                "outcomes_used_for_selection": False, "protected_counters": dict(COUNTERS),
            })
            return_code = process.wait()
            child_process_completed = True
    except OSError as exc:
        failure_reasons.append(f"CHILD_NOT_STARTED:{type(exc).__name__}:{exc}")
    if child_process_started and not child_process_completed and process is not None:
        try:
            return_code = process.wait()
            child_process_completed = True
        except OSError as exc:
            failure_reasons.append(f"CHILD_WAIT_FAILED:{type(exc).__name__}:{exc}")
    evidence = _runtime_evidence(parent_output, child_process_started=child_process_started,
                                 child_process_completed=child_process_completed,
                                 independent_audit_pass=False)
    independent_audit_pass = False
    try:
        if child_process_started:
            failure_reasons.extend(_verify_parent(parent_output, return_code if return_code is not None else -1))
            audit_path = parent_output / "M4_INDEPENDENT_AUDIT.json"
            audit = _load(audit_path) if audit_path.is_file() else {}
            independent_audit_pass = audit.get("status") == "PASS_M4_PARENT_INDEPENDENT"
            evidence = _runtime_evidence(parent_output, child_process_started=True,
                                         child_process_completed=child_process_completed,
                                         independent_audit_pass=independent_audit_pass)
            post_inventory, query_error = query_inventory()
            if query_error:
                failure_reasons.append(f"GPU_POST_QUERY_FAILED:{query_error}")
            else:
                post = next((item for item in post_inventory if int(item.get("gpu_id", -1)) == args.gpu), None)
                if post is None or str(post.get("gpu_uuid")) != str(lease["gpu_uuid"]):
                    failure_reasons.append("GPU_POST_IDENTITY_INVALID")
                else:
                    write_resource_receipt(gate_root / "RESOURCE_POST.json", phase="POST_PARENT",
                                           gpu_snapshot=post, lease=lease, atomic_job_id=job_id)
        else:
            evidence["intervention_state"] = "NOT_STARTED"
    except Exception as exc:
        failure_reasons.append(f"POST_CLAIM_GATE_EXCEPTION:{type(exc).__name__}:{exc}")
        evidence = _runtime_evidence(parent_output, child_process_started=child_process_started,
                                     child_process_completed=child_process_completed,
                                     independent_audit_pass=independent_audit_pass)
    try:
        release_ok = lease_store.release(lease, reason="PARENT_PASS" if not failure_reasons else "STRUCTURAL_FAILURE")
    except Exception as exc:
        release_ok = False
        failure_reasons.append(f"GPU_LEASE_RELEASE_EXCEPTION:{type(exc).__name__}:{exc}")
    _write(gate_root / "RESOURCE_RELEASE.json", {
        "schema": "STAGE_V_M4_FORMAL_RESOURCE_RELEASE_V1", "status": "PASS" if release_ok else "HOLD_RELEASE_FAILED",
        "release_ok": release_ok, "lease_id": lease["lease_id"], "atomic_job_id": job_id,
        "reason": "PARENT_PASS" if not failure_reasons else "STRUCTURAL_FAILURE",
        "outcomes_read": evidence["outcomes_read"], "outcomes_used_for_selection": False,
        "protected_counters": dict(COUNTERS),
    })
    if not release_ok:
        failure_reasons.append("GPU_LEASE_RELEASE_FAILED")
    status = "PASS_FORMAL_M4_PARENT_ATOMIC" if not failure_reasons else "HOLD_FORMAL_M4_STRUCTURAL_FAILURE"
    status_payload = {
        "schema": "STAGE_V_M4_FORMAL_PARENT_STATUS_V2", "status": status,
        "parent_index": args.parent_index, "canonical_parent_key": parent_key,
        "gpu": args.gpu, "failure_reasons": failure_reasons, "claim_created": True,
        **evidence, "intervention_executed": evidence["intervention_started"],
        "v_phys_generated": evidence["v_phys_artifacts_materialized"],
        "no_rerun_to_pass": evidence["child_process_started"],
        "protected_counters": dict(COUNTERS), "completed_utc": _utc(),
    }
    _write(gate_root / "PARENT_STATUS.json", status_payload)
    if failure_reasons:
        _create_only(args.output_root / "GLOBAL_HOLD.json", {
            "schema": "STAGE_V_M4_FORMAL_GLOBAL_HOLD_V2", "status": "HOLD_STOP_GLOBAL_SCHEDULING",
            "trigger_parent_index": args.parent_index, "trigger_parent_key": parent_key,
            "reasons": failure_reasons, "no_rerun_to_pass": status_payload["no_rerun_to_pass"],
            "claim_created": True, **evidence, "protected_counters": dict(COUNTERS), "created_utc": _utc(),
        })
    print(json.dumps(status_payload, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--launch-gate-binding", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--final-split", type=Path, required=True)
    parser.add_argument("--exact-plan-root", type=Path, required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--parent-index", type=int, default=0)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--attempt-ordinal", type=int, default=1)
    parser.add_argument("--runtime-provenance-sha256", default="")
    parser.add_argument("--minimum-free-mib", type=int, default=MIN_FREE_MEMORY_MIB)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (M4GovernanceError, ResourceContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "HOLD_FORMAL_M4_GATE", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
