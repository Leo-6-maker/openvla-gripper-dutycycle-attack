"""Run deterministic clean A/B qualification before Stage V R2."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from typing import Any, Mapping

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ImportError:  # direct server execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fec.atomic_task_queue import AtomicTaskQueue

try:
    from .stage_v_dynamic_common import atomic_write_json, canonical_parent_key, load_rows, normalize_parent, sha256_file, sha256_text, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, canonical_parent_key, load_rows, normalize_parent, sha256_file, sha256_text, utc_now

try:
    from .stage_v_gpu_resource_contract import (
        MODE_B, MIN_FREE_MEMORY_MIB, GpuLeaseStore, admit_mode_b_or_c, query_inventory,
        verify_recheck, write_resource_receipt,
    )
except ImportError:  # direct server execution
    from stage_v_gpu_resource_contract import (
        MODE_B, MIN_FREE_MEMORY_MIB, GpuLeaseStore, admit_mode_b_or_c, query_inventory,
        verify_recheck, write_resource_receipt,
    )


FORBIDDEN = re.compile(r"(?<![A-Za-z0-9_])(?:OPEN(?:_T[0-9]+)?|VIS|PGD|ATTACK|EVAL160|PROTECTED|TEACHER)(?![A-Za-z0-9_])", re.IGNORECASE)
EXPECTED_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_SALT = "STAGE_V_R2_CONTROL_QUALIFICATION_20260807"


def _render_command(template: str, **values: object) -> str:
    """Replace only our placeholders; preserve shell syntax such as ``${IFS}``."""
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def ranked(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        normalized = normalize_parent(row)
        key = normalized["canonical_parent_key"]
        normalized["qualification_rank_sha256"] = hashlib.sha256(f"{salt}::{key}".encode()).hexdigest()
        output.append(normalized)
    return sorted(output, key=lambda item: (item["qualification_rank_sha256"], item["canonical_parent_key"]))


def _result_from_directory(directory: Path) -> Mapping[str, Any] | None:
    for name in ("CONTROL_RESULT.json", "RESULT.json", "PARENT_RESULT.json"):
        value = json.loads((directory / name).read_text(encoding="utf-8")) if (directory / name).is_file() else None
        if isinstance(value, Mapping):
            return value
    return None


def _run_once(template: str, *, candidate_path: Path, output_dir: Path, replicate: str, source_commit: str, source_tree: str, gpu: int = 0) -> tuple[int, dict[str, Any]]:
    command_text = _render_command(
        template,
        candidate_path=str(candidate_path), output_dir=str(output_dir), replicate=replicate,
        source_commit=source_commit, source_tree=source_tree, gpu=gpu,
    )
    if FORBIDDEN.search(command_text):
        return 2, {"status": "FAIL", "reason": "FORBIDDEN_COMMAND_TOKEN"}
    command = command_text if isinstance(command_text, str) else str(command_text)
    completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
    result = _result_from_directory(output_dir)
    if result is None:
        for line in reversed(completed.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                result = candidate
                break
    payload = dict(result) if result else {}
    payload.update({
        "replicate": replicate,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    })
    return completed.returncode, payload


def qualifies(row: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], source_commit: str, source_tree: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for name, result in (("A", a), ("B", b)):
        if result.get("exit_code") != 0 or result.get("status") not in ("PASS", "DONE", "QUALIFIED"):
            errors.append(f"{name}_NOT_COMPLETE")
        for field, error in (
            ("clean_success", "CLEAN_SUCCESS_FALSE"),
            ("task_identity_valid", "TASK_IDENTITY_INVALID"),
            ("snapshot_restore_valid", "SNAPSHOT_RESTORE_INVALID"),
            ("runtime_valid", "RUNTIME_INVALID"),
            ("metrics_finite", "NONFINITE"),
            ("artifact_validation_pass", "ARTIFACT_VALIDATION_FAIL"),
        ):
            if result.get(field) is not True:
                errors.append(f"{name}_{error}")
        if result.get("old_artifacts_reused") is not False:
            errors.append(f"{name}_OLD_ARTIFACT_REUSE")
        for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts"):
            if result.get(field, 0) != 0:
                errors.append(f"{name}_BOUNDARY_VIOLATION:{field}")
        if result.get("source_commit") != source_commit or result.get("source_tree") != source_tree:
            errors.append(f"{name}_PROVENANCE_MISMATCH")
        if result.get("remaining_horizon_complete") is not True:
            errors.append(f"{name}_HORIZON_INCOMPLETE")
        if row.get("assigned_gpu") is not None and result.get("worker_gpu") != int(row["assigned_gpu"]):
            errors.append(f"{name}_GPU_AFFINITY_MISMATCH")
    # Terminal outcome/state are descriptive for fresh clean qualification.
    # Only the initial identity binds A and B to the same causal parent.
    if (
        not a.get("key_state_identity_sha256")
        or not b.get("key_state_identity_sha256")
        or a.get("key_state_identity_sha256") != b.get("key_state_identity_sha256")
    ):
        errors.append("AB_MISMATCH:key_state_identity_sha256")
    if a.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("A_PARENT_IDENTITY_MISMATCH")
    if b.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("B_PARENT_IDENTITY_MISMATCH")
    return not errors, sorted(set(errors))


def audit_qualification_row(row: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], source_commit: str, source_tree: str) -> tuple[bool, list[str]]:
    """Independent qualification decision; do not call the producer helper."""
    errors: list[str] = []
    for name, result in (("A", a), ("B", b)):
        if result.get("exit_code") != 0 or result.get("status") not in {"PASS", "DONE", "QUALIFIED"}:
            errors.append(f"{name}_NOT_COMPLETE")
        required_true = ("clean_success", "task_identity_valid", "snapshot_restore_valid", "runtime_valid", "metrics_finite", "artifact_validation_pass", "remaining_horizon_complete")
        errors.extend(f"{name}_{field.upper()}_FALSE" for field in required_true if result.get(field) is not True)
        if result.get("old_artifacts_reused") is not False:
            errors.append(f"{name}_OLD_ARTIFACT_REUSE")
        for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts"):
            if result.get(field, 0) != 0:
                errors.append(f"{name}_BOUNDARY_VIOLATION:{field}")
        if result.get("source_commit") != source_commit or result.get("source_tree") != source_tree:
            errors.append(f"{name}_PROVENANCE_MISMATCH")
        if result.get("canonical_parent_key") != row.get("canonical_parent_key"):
            errors.append(f"{name}_PARENT_IDENTITY_MISMATCH")
        if row.get("assigned_gpu") is not None and result.get("worker_gpu") != int(row["assigned_gpu"]):
            errors.append(f"{name}_GPU_AFFINITY_MISMATCH")
    # Keep the auditor aligned with the frozen contract: terminal fields are
    # descriptive; initial identity remains an exact hard gate.
    if (
        not a.get("key_state_identity_sha256")
        or not b.get("key_state_identity_sha256")
        or a.get("key_state_identity_sha256") != b.get("key_state_identity_sha256")
    ):
        errors.append("AB_MISMATCH:key_state_identity_sha256")
    return not errors, sorted(set(errors))


def build_science_parent_manifest(formal_manifest: Mapping[str, Any], *, source_clean_root: str) -> dict[str, Any]:
    """Build a fresh V1 identity manifest for the frozen external science runner."""
    if formal_manifest.get("schema") != "STAGE_V_FORMAL_PARENT_MANIFEST_V2" or formal_manifest.get("status") != "PASS":
        raise ValueError("formal qualification manifest is not PASS")
    rows = formal_manifest.get("selected_parents")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("formal qualification manifest must contain 40 parents")
    root = str(source_clean_root).rstrip("/")
    if not root:
        raise ValueError("source clean root is missing")
    selected: list[dict[str, Any]] = []
    keys: set[str] = set()
    for raw in rows:
        row = normalize_parent(raw)
        key = str(row["canonical_parent_key"])
        if key in keys or row.get("old_artifacts_reused") is not False or row.get("source_artifact_read") is not False:
            raise ValueError(f"invalid qualified parent identity: {key}")
        keys.add(key)
        selected.append({
            "canonical_parent_key": key,
            "suite": row["suite"],
            "task_index": row["task_index"],
            "state_index": row["state_index"],
            "source_artifact_root": str(row.get("source_artifact_root") or f"{root}/{key}"),
            "artifact_recursive_sha256": str(row.get("artifact_recursive_sha256") or ""),
            "selection_role": "qualified_clean_control_parent_only",
            "qualification_mode": "FRESH_CLEAN_AB_REPLAY",
            "source_artifact_read": False,
            "old_artifacts_reused": False,
        })
    return {
        "schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V1",
        "status": "FROZEN",
        "source_clean_root": root,
        "source_commit": formal_manifest.get("source_commit"),
        "source_tree": formal_manifest.get("source_tree"),
        "selected_parents": selected,
        "selected_count": len(selected),
        "parents_by_suite": {suite: sum(row["suite"] == suite for row in selected) for suite in EXPECTED_SUITES},
        "candidate_manifest_sha256": formal_manifest.get("candidate_manifest_sha256"),
        "control_qualification_report_sha256": formal_manifest.get("control_qualification_report_sha256"),
        "control_qualification_rows_sha256": formal_manifest.get("control_qualification_rows_sha256"),
        "control_qualification_audit_sha256": formal_manifest.get("control_qualification_audit_sha256"),
        "old_artifacts_reused": False,
        "source_artifacts_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "generated_utc": utc_now(),
    }


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()] if value else [0]
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"invalid GPU list: {value!r}")
    return gpus


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_clean_root = str(Path(args.source_clean_root).resolve())
    raw_rows = load_rows(args.candidate_manifest)
    rows = ranked(
        [row for row in raw_rows if row.get("audit_status", "PASS") == "PASS" and int(row.get("remaining_policy_steps", 1) or 0) > 0],
        args.salt,
    )
    if not rows:
        raise ValueError("candidate manifest is empty")
    suites = sorted({str(row["suite"]) for row in rows})
    if args.suites:
        suites = [suite for suite in suites if suite in set(args.suites.split(","))]
    if not suites:
        raise ValueError("candidate manifest has no requested suites")
    if not args.suites and tuple(suites) != EXPECTED_SUITES:
        raise ValueError(f"candidate manifest suites must be {EXPECTED_SUITES}, got {tuple(suites)}")
    keys = [str(row["canonical_parent_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate manifest contains duplicate canonical parent keys")
    gpus = _parse_gpus(args.gpus)
    resource_mode = str(getattr(args, "resource_mode", "LEGACY"))
    minimum_free_mib = int(getattr(args, "minimum_free_mib", MIN_FREE_MEMORY_MIB))
    if resource_mode not in {"LEGACY", MODE_B}:
        raise ValueError(f"unsupported resource mode: {resource_mode}")
    if resource_mode == MODE_B:
        parent_gpu_map = {
            key: gpus[int(hashlib.sha256(f"{args.salt}::{key}".encode()).hexdigest(), 16) % len(gpus)]
            for key in keys
        }
        rows = [
            {**row, "assigned_gpu": parent_gpu_map[str(row["canonical_parent_key"])], "parent_gpu_affinity": "FROZEN_HASH_SALT"}
            for row in rows
        ]
    else:
        parent_gpu_map = {}
    by_suite = {suite: [row for row in rows if str(row["suite"]) == suite] for suite in suites}
    rows_out: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {suite: [] for suite in suites}
    resource_lease_store = None
    resource_preflight: dict[str, Any] | None = None
    resource_lease_db = Path(getattr(args, "resource_lease_db", args.output_dir / "GPU_LEASES.sqlite")).resolve()
    if resource_mode == MODE_B:
        resource_lease_store = GpuLeaseStore(resource_lease_db)
        inventory, inventory_error = query_inventory()
        if inventory_error:
            raise RuntimeError(f"RESOURCE_PREFLIGHT_FAIL:{inventory_error}")
        active = resource_lease_store.active()
        admission = admit_mode_b_or_c(
            inventory, mode=MODE_B, leased_gpu_ids=[item["gpu_id"] for item in active],
            minimum_free_mib=minimum_free_mib,
        )
        missing = sorted(set(gpus) - set(admission.get("eligible_gpu_ids", [])))
        if missing:
            raise RuntimeError(f"RESOURCE_PREFLIGHT_GPU_NOT_ELIGIBLE:{missing}")
        resource_preflight = {
            "schema": "STAGE_V_DYNAMIC_CLEAN_QUALIFICATION_RESOURCE_PREFLIGHT_V1",
            "status": "PASS", "resource_mode": MODE_B, "requested_gpu_ids": gpus,
            "minimum_free_memory_mib": minimum_free_mib,
            "maximum_project_workers_per_gpu": 1, "partial_fleet_allowed": True,
            "foreign_workload_allowed": True, "admission": admission,
            "inventory": inventory, "active_project_leases": active,
            "lease_db": str(resource_lease_db), "source_commit": args.source_commit,
            "source_tree": args.source_tree, "captured_utc": utc_now(),
        }
        atomic_write_json(args.output_dir / "PRE_QUALIFICATION_RESOURCE_RECEIPT.json", resource_preflight)
    queue_db = args.output_dir / "CONTROL_QUALIFICATION.sqlite"
    queue = AtomicTaskQueue(str(queue_db), run_id=args.salt)
    manifest_sha = sha256_file(args.candidate_manifest)
    source_sha = f"{args.source_commit}:{args.source_tree}"
    queue.init_run(
        state="ACTIVE", manifest_sha=manifest_sha, source_sha=source_sha,
        config_sha=sha256_text(args.runner_command),
        capacity_policy={
            "mode": "atomic_dynamic_workers", "gpus": gpus, "worker_count": len(gpus),
            "resource_mode": resource_mode, "minimum_free_memory_mib": minimum_free_mib,
            "maximum_project_workers_per_gpu": 1,
            "parent_gpu_affinity": "FROZEN_HASH_SALT" if resource_mode == MODE_B else None,
            "lease_db": str(resource_lease_db) if resource_lease_store else None,
            "old_artifacts_reused": False,
        },
    )
    rows_by_key = {str(row["canonical_parent_key"]): row for row in rows}

    def qualification_worker(gpu: int, batch_keys: set[str], batch_index: int) -> list[tuple[str, str, int, dict[str, Any]]]:
        worker_id = f"stage-v-control-qualifier-gpu{gpu}-pid{os.getpid()}"
        outcomes: list[tuple[str, str, int, dict[str, Any]]] = []
        allowed_parent_ids = (
            {key for key in batch_keys if parent_gpu_map.get(key) == gpu}
            if resource_mode == MODE_B else None
        )
        lease = None
        try:
            if resource_lease_store is not None and allowed_parent_ids:
                inventory, inventory_error = query_inventory()
                if inventory_error:
                    raise RuntimeError(f"RESOURCE_JOB_PREFLIGHT_FAIL:{inventory_error}")
                active = resource_lease_store.active()
                admission = admit_mode_b_or_c(
                    inventory, mode=MODE_B, leased_gpu_ids=[item["gpu_id"] for item in active],
                    minimum_free_mib=minimum_free_mib,
                )
                decision = next((item for item in admission.get("gpu_decisions", []) if int(item.get("gpu_id", -1)) == gpu), None)
                if gpu not in set(admission.get("eligible_gpu_ids", [])) or not decision:
                    raise RuntimeError(f"RESOURCE_JOB_GPU_NOT_ELIGIBLE:{gpu}")
                lease = resource_lease_store.acquire(
                    gpu_id=gpu, gpu_uuid=str(decision["gpu_uuid"]), worker_id=worker_id,
                    worker_pid=os.getpid(), stage="STAGE_V_DYNAMIC_CLEAN_QUALIFICATION",
                    atomic_job_id=f"{args.salt}:batch_{batch_index}:gpu_{gpu}",
                    source_commit=args.source_commit, source_tree=args.source_tree,
                    runtime_root=args.output_dir, launch_snapshot=decision,
                )
                rechecked_inventory, recheck_error = query_inventory()
                if recheck_error:
                    raise RuntimeError(f"RESOURCE_JOB_RECHECK_FAIL:{recheck_error}")
                rechecked = next((item for item in rechecked_inventory if int(item.get("gpu_id", -1)) == gpu), None)
                if rechecked is None:
                    raise RuntimeError(f"RESOURCE_JOB_RECHECK_GPU_MISSING:{gpu}")
                verify_recheck(rechecked, expected_gpu_id=gpu, expected_gpu_uuid=str(decision["gpu_uuid"]), minimum_free_mib=minimum_free_mib)
                write_resource_receipt(
                    args.output_dir / f"PRE_JOB_RESOURCE_RECEIPT_BATCH{batch_index}_GPU{gpu}.json",
                    phase="PRE_JOB_RESOURCE_RECEIPT", gpu_snapshot=rechecked, lease=lease,
                    atomic_job_id=f"{args.salt}:batch_{batch_index}:gpu_{gpu}",
                )
            while True:
                task = queue.claim_task(
                    worker_id, hostname=socket.gethostname(), pid=os.getpid(), gpu_id=gpu,
                    expected_manifest_sha=manifest_sha, expected_source_sha=source_sha,
                    allowed_parent_ids=allowed_parent_ids,
                )
                if task is None:
                    return outcomes
                key = str(task["parent_id"])
                if key not in batch_keys or key not in rows_by_key:
                    raise RuntimeError(f"CONTROL_QUALIFICATION_QUEUE_IDENTITY_FAIL:{key}")
                row = rows_by_key[key]
                replicate = str(task["arm"])
                base = args.output_dir / "qualification" / str(row["suite"]) / key.replace("/", "__")
                code, result = _run_once(
                    args.runner_command, candidate_path=base / "CANDIDATE.json", output_dir=base / replicate,
                    replicate=replicate, source_commit=args.source_commit, source_tree=args.source_tree, gpu=gpu,
                )
                outcome = "DONE_VALID" if code == 0 and result else "FAILED_FATAL_POST_ACTION"
                receipt = base / replicate / "CONTROL_RESULT.json"
                receipt_sha = sha256_file(receipt) if receipt.is_file() else None
                if not queue.commit_result(
                    task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"],
                    exit_code=code, error_class=None if result else "MISSING_CONTROL_RESULT",
                    exposure_status="CLEAN_ONLY", task_outcome=outcome, output_dir=str(base / replicate), receipt_sha=receipt_sha,
                ):
                    raise RuntimeError(f"CONTROL_QUALIFICATION_QUEUE_COMMIT_FAIL:{key}:{replicate}")
                outcomes.append((key, replicate, code, dict(result)))
        finally:
            if lease is not None and resource_lease_store is not None:
                resource_lease_store.release(lease, reason="QUALIFICATION_BATCH_FINISHED")
            queue.close()

    cursor = {suite: 0 for suite in suites}
    batch_index = 0
    while True:
        batch_index += 1
        batch_rows: list[dict[str, Any]] = []
        for suite in suites:
            if len(selected[suite]) >= args.target_per_suite:
                continue
            suite_rows = by_suite[suite]
            start = cursor[suite]
            take = args.initial_per_suite if start == 0 else args.batch_size
            end = min(start + take, len(suite_rows))
            batch_rows.extend(suite_rows[start:end])
            cursor[suite] = end
        if not batch_rows:
            break
        batch_keys = {str(row["canonical_parent_key"]) for row in batch_rows}
        for row in batch_rows:
            key = str(row["canonical_parent_key"])
            base = args.output_dir / "qualification" / str(row["suite"]) / key.replace("/", "__")
            base.mkdir(parents=True, exist_ok=False)
            candidate_path = base / "CANDIDATE.json"
            atomic_write_json(candidate_path, {
                **row,
                "old_artifacts_reused": False,
                "source_artifact_read": False,
                "qualification_mode": "FRESH_CLEAN_AB_REPLAY",
            })
            queue.register_tasks([
                {
                    "cell_id": f"CONTROL|{key.replace('/', '__')}|{replicate}",
                    "parent_id": key, "suite": str(row["suite"]), "task_index": int(row["task_index"]),
                    "state_index": int(row["state_index"]), "arm": replicate,
                    "task_kind": "CONTROL_QUALIFICATION",
                }
                for replicate in ("A", "B")
            ])
            for replicate in ("A", "B"):
                (base / replicate).mkdir()
        outcomes: list[tuple[str, str, int, dict[str, Any]]] = []
        worker_errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(gpus), thread_name_prefix="stage-v-control") as pool:
            futures = [pool.submit(qualification_worker, gpu, batch_keys, batch_index) for gpu in gpus]
            for future in as_completed(futures):
                try:
                    outcomes.extend(future.result())
                except Exception as exc:
                    worker_errors.append(f"{type(exc).__name__}:{exc}")
        if worker_errors:
            queue.set_run_state("HOLD")
            atomic_write_json(args.output_dir / "CONTROL_QUALIFICATION_WORKER_ERRORS.json", {"errors": worker_errors, "updated_utc": utc_now()})
            raise RuntimeError("CONTROL_QUALIFICATION_WORKER_ERROR")
        result_map = {(key, replicate): (code, result) for key, replicate, code, result in outcomes}
        for row in batch_rows:
            key = str(row["canonical_parent_key"])
            base = args.output_dir / "qualification" / str(row["suite"]) / key.replace("/", "__")
            replicate_rows: dict[str, dict[str, Any]] = {}
            for replicate in ("A", "B"):
                code, result = result_map.get((key, replicate), (1, {"status": "FAIL", "reason": "MISSING_QUEUE_OUTCOME"}))
                result = dict(result)
                result.setdefault("exit_code", code)
                if code != 0:
                    result.setdefault("status", "FAIL")
                replicate_rows[replicate] = result
            ok, errors = qualifies(row, replicate_rows["A"], replicate_rows["B"], args.source_commit, args.source_tree)
            record = {
                "schema": "STAGE_V_CONTROL_QUALIFICATION_ROW_V3",
                "canonical_parent_key": key,
                "suite": str(row["suite"]),
                "task_index": int(row["task_index"]),
                "state_index": int(row["state_index"]),
                "assigned_gpu": int(row["assigned_gpu"]) if row.get("assigned_gpu") is not None else None,
                "replicate_worker_gpus": {replicate: replicate_rows[replicate].get("worker_gpu") for replicate in ("A", "B")},
                "qualification_rank_sha256": row["qualification_rank_sha256"],
                "candidate_sha256": sha256_file(base / "CANDIDATE.json"),
                "replicates": replicate_rows,
                "qualified": ok,
                "errors": errors,
                "evaluated_utc": utc_now(),
            }
            atomic_write_json(base / "QUALIFICATION_ROW.json", record)
            rows_out.append(record)
            if ok and len(selected[str(row["suite"])]) < args.target_per_suite:
                selected[str(row["suite"])].append({
                    **row,
                    "old_artifacts_reused": False,
                    "source_artifact_read": False,
                    "qualification_mode": "FRESH_CLEAN_AB_REPLAY",
                })
    report = {
        "schema": "STAGE_V_CONTROL_QUALIFICATION_REPORT_V2",
        "status": "PASS" if all(len(selected[suite]) >= args.target_per_suite for suite in suites) and (bool(args.suites) or len(suites) == len(EXPECTED_SUITES)) else "FAIL",
        "salt": args.salt,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "candidate_manifest_sha256": manifest_sha,
        "source_clean_root": source_clean_root,
        "gpus": gpus,
        "worker_count": len(gpus),
        "resource_mode": resource_mode,
        "minimum_free_memory_mib": minimum_free_mib,
        "resource_lease_db": str(resource_lease_db) if resource_lease_store else None,
        "resource_preflight": resource_preflight,
        "parent_gpu_assignment": parent_gpu_map,
        "parent_gpu_assignment_sha256": sha256_text(json.dumps(parent_gpu_map, sort_keys=True, separators=(",", ":"))),
        "initial_per_suite": args.initial_per_suite,
        "batch_size": args.batch_size,
        "target_per_suite": args.target_per_suite,
        "suites": suites,
        "qualified_by_suite": {suite: len(selected[suite]) for suite in suites},
        "evaluated_rows": len(rows_out),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "generated_utc": utc_now(),
        "queue_db": str(queue_db),
        "queue_progress": queue.get_progress(),
    }
    manifest_rows = [row for suite in suites for row in selected[suite][:args.target_per_suite]]
    manifest = {
        # ponytail: retain the frozen science runner's manifest contract and
        # add the R2 qualification bindings around it.
        "schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V2",
        "status": report["status"],
        "salt": args.salt,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "candidate_manifest_sha256": manifest_sha,
        "source_clean_root": source_clean_root,
        "parents": manifest_rows,
        "selected_parents": manifest_rows,
        "selected_count": len(manifest_rows),
        "planned_parent_count": len(manifest_rows),
        "parents_by_suite": {suite: args.target_per_suite for suite in suites},
        "resource_mode": resource_mode,
        "minimum_free_memory_mib": minimum_free_mib,
        "parent_gpu_assignment": parent_gpu_map,
        "parent_gpu_assignment_sha256": sha256_text(json.dumps(parent_gpu_map, sort_keys=True, separators=(",", ":"))),
        "old_artifacts_reused": False,
        "generated_utc": utc_now(),
    }
    independent_errors: list[str] = []
    recomputed: dict[str, int] = {suite: 0 for suite in suites}
    for record in rows_out:
        recomputed_ok, recomputed_errors = audit_qualification_row(
            record, record.get("replicates", {}).get("A", {}), record.get("replicates", {}).get("B", {}),
            args.source_commit, args.source_tree,
        )
        if recomputed_ok:
            recomputed[str(record["suite"])] += 1
        if recomputed_ok != bool(record.get("qualified")):
            independent_errors.append(f"ROW_RECOMPUTE_MISMATCH:{record.get('canonical_parent_key')}")
    selected_keys = [str(row["canonical_parent_key"]) for suite in suites for row in selected[suite][:args.target_per_suite]]
    if len(selected_keys) != len(set(selected_keys)):
        independent_errors.append("SELECTED_DUPLICATE_PARENT_KEYS")
    audit = {
        "schema": "STAGE_V_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT_V2",
        "verdict": "PASS" if report["status"] == "PASS" and not independent_errors else "FAIL",
        "recomputed_qualified_by_suite": recomputed,
        "manifest_parent_count": len(manifest_rows),
        "duplicate_parent_keys": sorted({key for key in selected_keys if selected_keys.count(key) > 1}),
        "errors": sorted(set(independent_errors)),
        "boundaries": {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0},
        "audited_utc": utc_now(),
        "queue_states": {state: sum(1 for item in queue.list_tasks() if item["state"] == state) for state in sorted({str(task["state"]) for task in queue.list_tasks()})},
    }
    report["independent_audit_verdict"] = audit["verdict"]
    if audit["verdict"] != "PASS":
        report["status"] = "FAIL"
        manifest["status"] = "FAIL"
    queue.set_run_state("COMPLETE" if not any(item["state"] != "DONE_VALID" for item in queue.list_tasks()) else "HOLD")
    queue.close()
    return report, rows_out, {"manifest": manifest, "audit": audit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--source-clean-root", type=Path, required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--initial-per-suite", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--target-per-suite", type=int, default=10)
    parser.add_argument("--suites", default="")
    parser.add_argument("--resource-mode", default="LEGACY")
    parser.add_argument("--resource-lease-db", type=Path)
    parser.add_argument("--minimum-free-mib", type=int, default=MIN_FREE_MEMORY_MIB)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            parser.error(f"qualification output must be new/empty: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
    report, rows, extras = qualify(args)
    atomic_write_json(args.output_dir / "CONTROL_QUALIFICATION_REPORT.json", report)
    with (args.output_dir / "CONTROL_QUALIFICATION_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    atomic_write_json(args.output_dir / "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", extras["audit"])
    manifest_path = args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V2.json"
    if report["status"] == "PASS" and extras["audit"]["verdict"] == "PASS":
        extras["manifest"].update({
            "control_qualification_report_sha256": sha256_file(args.output_dir / "CONTROL_QUALIFICATION_REPORT.json"),
            "control_qualification_rows_sha256": sha256_file(args.output_dir / "CONTROL_QUALIFICATION_ROWS.jsonl"),
            "control_qualification_audit_sha256": sha256_file(args.output_dir / "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json"),
        })
        atomic_write_json(manifest_path, extras["manifest"])
        science_manifest = build_science_parent_manifest(
            extras["manifest"], source_clean_root=str(args.source_clean_root.resolve()),
        )
        science_path = args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.json"
        atomic_write_json(science_path, science_manifest)
        science_sha = sha256_file(science_path)
        (args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.sha256").write_text(
            science_sha + "  STAGE_V_FORMAL_PARENT_MANIFEST_V1.json\n", encoding="utf-8",
        )
        extras["manifest"]["science_parent_manifest"] = str(science_path)
        extras["manifest"]["science_parent_manifest_sha256"] = science_sha
        atomic_write_json(manifest_path, extras["manifest"])
        (args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V2.sha256").write_text(
            sha256_file(manifest_path) + "  STAGE_V_FORMAL_PARENT_MANIFEST_V2.json\n", encoding="utf-8",
        )
        write_manifest_a(args.output_dir, manifest_path)
    return 0 if report["status"] == "PASS" else 1


def write_manifest_a(output_dir: Path, manifest_path: Path) -> Path:
    """Write the frozen R2 qualification manifest under its plan-level name."""
    alias = output_dir / "STAGE_V_R2_PARENT_MANIFEST_A.json"
    alias.write_bytes(manifest_path.read_bytes())
    (output_dir / "STAGE_V_R2_PARENT_MANIFEST_A.sha256").write_text(
        sha256_file(alias) + "  STAGE_V_R2_PARENT_MANIFEST_A.json\n", encoding="utf-8",
    )
    return alias


if __name__ == "__main__":
    raise SystemExit(main())
