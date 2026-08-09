"""Run fresh clean A/B qualification with parent-atomic GPU leases."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from typing import Any, Mapping

try:
    from .run_stage_v_r2_q2_control_qualification import engineering_valid, qualify_pair
    from .stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, utc_now
    from .stage_v_gpu_resource_contract import (
        MODE_B, MIN_FREE_MEMORY_MIB, GpuLeaseStore, ResourceContractError,
        admit_mode_b_or_c, query_inventory, verify_recheck, write_resource_receipt,
    )
except ImportError:  # direct execution on the server
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.detector_v5.run_stage_v_r2_q2_control_qualification import engineering_valid, qualify_pair
    from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, utc_now
    from scripts.detector_v5.stage_v_gpu_resource_contract import (
        MODE_B, MIN_FREE_MEMORY_MIB, GpuLeaseStore, ResourceContractError,
        admit_mode_b_or_c, query_inventory, verify_recheck, write_resource_receipt,
    )


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
BOUNDARIES = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _load_manifest(path: Path, salt: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(path)
    if manifest.get("schema") != "STAGE_V_R2_QUALIFICATION_CANDIDATE_MANIFEST_V1" or manifest.get("status") != "FROZEN":
        raise ValueError("qualification candidate manifest is not frozen")
    rows = [normalize_parent(row) for row in manifest.get("selected_parents", []) if isinstance(row, Mapping)]
    if len(rows) != int(manifest.get("selected_count", -1)) or not rows:
        raise ValueError("qualification candidate manifest count mismatch")
    keys = [str(row["canonical_parent_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("qualification candidate manifest contains duplicate parents")
    if any(row.get("old_artifacts_reused") is not False or row.get("source_artifact_read") is not False for row in rows):
        raise ValueError("qualification candidate manifest reuses artifacts")
    for row in rows:
        expected = __import__("hashlib").sha256(f"{salt}::{row['canonical_parent_key']}".encode()).hexdigest()
        if row.get("qualification_rank_sha256") != expected:
            raise ValueError(f"qualification rank mismatch:{row['canonical_parent_key']}")
    counts = {suite: sum(row["suite"] == suite for row in rows) for suite in SUITES}
    if any(count < 10 for count in counts.values()):
        raise ValueError("qualification manifest cannot satisfy suite quotas")
    return manifest, rows


def _run_clean(args: argparse.Namespace, row: Mapping[str, Any], parent_root: Path, replicate: str, gpu: int) -> tuple[int, dict[str, Any]]:
    output = parent_root / replicate
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.python_executable), str(args.clean_runner),
        "--candidate-path", str(parent_root / "CANDIDATE.json"),
        "--output-dir", str(output), "--replicate", replicate, "--gpu", str(gpu),
        "--worker-id", f"stage-v-fresh-qual-gpu{gpu}-{replicate}",
        "--worker-script", str(args.official_worker),
        "--provenance-source", str(args.provenance_source),
        "--upstream-root", str(args.upstream_root),
        "--source-commit", args.source_commit, "--source-tree", args.source_tree,
        "--min-remaining-steps", str(args.min_remaining_steps),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    log = (parent_root / f"{replicate}.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=str(args.repo_root), env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
        code = process.wait()
    finally:
        log.close()
    result_path = output / "CONTROL_RESULT.json"
    result = _json(result_path) if result_path.is_file() else {"status": "FAIL", "exit_code": code}
    return code, result


def _resource_lease(args: argparse.Namespace, store: GpuLeaseStore, gpu: int, worker_id: str,
                    parent_key: str, attempt: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory, error = query_inventory()
    if error:
        raise ResourceContractError(error)
    admission = admit_mode_b_or_c(
        inventory, mode=MODE_B, leased_gpu_ids=[row["gpu_id"] for row in store.active()],
        project_pids=[row["worker_pid"] for row in store.active()],
        project_process_tokens=(str(args.run_root), "run_stage_v_r2_fresh_qualification_atomic.py"),
        excluded_gpu_ids=(), minimum_free_mib=args.minimum_free_mib,
    )
    decision = next((row for row in admission["gpu_decisions"] if int(row["gpu_id"]) == gpu), None)
    if not decision or decision.get("safe") is not True:
        raise ResourceContractError(f"GPU_NOT_ELIGIBLE:{gpu}:{decision and decision.get('reasons')}")
    lease = store.acquire(
        gpu_id=gpu, gpu_uuid=decision["gpu_uuid"], worker_id=worker_id, worker_pid=os.getpid(),
        stage="FRESH_QUALIFICATION", atomic_job_id=f"{args.run_id}:{parent_key}:{attempt}",
        source_commit=args.source_commit, source_tree=args.source_tree,
        runtime_root=args.run_root, launch_snapshot=decision,
    )
    rechecked = next((row for row in query_inventory()[0] if int(row["gpu_id"]) == gpu), None)
    if rechecked is None:
        store.release(lease, reason="RECHECK_QUERY_FAILED")
        raise ResourceContractError("GPU_RECHECK_NOT_FOUND")
    try:
        verify_recheck(rechecked, expected_gpu_id=gpu, expected_gpu_uuid=decision["gpu_uuid"], minimum_free_mib=args.minimum_free_mib)
    except Exception:
        store.release(lease, reason="RECHECK_FAILED")
        raise
    return lease, decision, rechecked


def _write_resource(path: Path, *, phase: str, decision: Mapping[str, Any], snapshot: Mapping[str, Any],
                    lease: Mapping[str, Any], run_id: str, parent_key: str) -> None:
    atomic_write_json(path, {
        "schema": "STAGE_V_PRE_JOB_RESOURCE_RECEIPT_V2",
        "phase": phase,
        "mode": MODE_B,
        "run_id": run_id,
        "parent_id": parent_key,
        "gpu_id": decision["gpu_id"],
        "gpu_uuid": decision["gpu_uuid"],
        "minimum_free_memory_mib": MIN_FREE_MEMORY_MIB,
        "memory_free_mib": snapshot.get("memory_free_mib"),
        "utilization_gpu_percent": snapshot.get("utilization_gpu_percent"),
        "foreign_workload_allowed": True,
        "foreign_processes": decision.get("foreign_processes", []),
        "lease": {key: lease.get(key) for key in ("lease_id", "gpu_id", "gpu_uuid", "worker_id", "worker_pid", "atomic_job_id", "source_commit", "source_tree", "runtime_root", "acquired_utc")},
        "captured_utc": utc_now(),
    })


def _heartbeat(queue: Any, task: Mapping[str, Any], worker_id: str, stop: threading.Event) -> None:
    while not stop.wait(30.0):
        queue.heartbeat(task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"])


def _worker(args: argparse.Namespace, gpu: int, rows_by_key: Mapping[str, Mapping[str, Any]], manifest_sha: str,
            source_sha: str, queue: Any, store: GpuLeaseStore) -> None:
    worker_id = f"fresh-qualification-gpu{gpu}-pid{os.getpid()}-tid{threading.get_ident()}"
    while True:
        task = queue.claim_task(worker_id, hostname=socket.gethostname(), pid=os.getpid(), gpu_id=gpu,
                                expected_manifest_sha=manifest_sha, expected_source_sha=source_sha)
        if task is None:
            return
        key = str(task["parent_id"])
        row = rows_by_key[key]
        attempt = int(task["attempt_count"])
        parent_root = args.run_root / "qualification" / str(row["suite"]) / key.replace("/", "__") / f"attempt_{attempt:02d}"
        parent_root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(parent_root / "CANDIDATE.json", dict(row))
        lease: dict[str, Any] | None = None
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=lambda: _heartbeat(queue, task, worker_id, heartbeat_stop),
            name=f"heartbeat-gpu{gpu}", daemon=True,
        )
        heartbeat.start()
        valid = False
        errors: list[str] = []
        result_a: dict[str, Any] = {"status": "FAIL", "exit_code": 1}
        result_b: dict[str, Any] = {"status": "FAIL", "exit_code": 1}
        code_a = code_b = 1
        try:
            lease, decision, snapshot = _resource_lease(args, store, gpu, worker_id, key, attempt)
            _write_resource(parent_root / "PRE_JOB_RESOURCE_RECEIPT.json", phase="PRE_JOB", decision=decision, snapshot=snapshot, lease=lease, run_id=args.run_id, parent_key=key)
            code_a, result_a = _run_clean(args, row, parent_root, "A", gpu)
            code_b, result_b = _run_clean(args, row, parent_root, "B", gpu)
            valid_a, errors_a = engineering_valid(row, result_a, code_a, args.source_commit, args.source_tree)
            valid_b, errors_b = engineering_valid(row, result_b, code_b, args.source_commit, args.source_tree)
            valid = valid_a and valid_b
            qualified, classification, pair_errors = qualify_pair(row, result_a, result_b, valid_a, valid_b, args.source_commit, args.source_tree)
            errors = sorted(set(errors_a + errors_b + pair_errors))
            parent_status = "PASS" if valid else "ENGINEERING_INVALID"
            task_outcome = "DONE_VALID" if valid else ("FAILED_RETRYABLE_INFRA" if attempt <= args.max_infrastructure_retries else "FAILED_FATAL_POST_ACTION")
            parent_result = {
                "schema": "STAGE_V_R2_FRESH_QUALIFICATION_PARENT_RESULT_V2",
                "status": parent_status, "qualified": qualified, "classification": classification,
                "canonical_parent_key": key, "suite": row["suite"], "task_index": row["task_index"], "state_index": row["state_index"],
                "replicates": {"A": result_a, "B": result_b}, "replicate_exit_codes": {"A": code_a, "B": code_b},
                "engineering_valid": {"A": valid_a, "B": valid_b}, "errors": errors,
                "worker_gpu": gpu, "source_commit": args.source_commit, "source_tree": args.source_tree,
                "old_artifacts_reused": False, "source_artifact_read": False,
                **{field: 0 for field in BOUNDARIES}, "evaluated_utc": utc_now(),
            }
            atomic_write_json(parent_root / "PARENT_RESULT.json", parent_result)
            if task_outcome == "FAILED_FATAL_POST_ACTION":
                queue.set_run_state("HOLD")
        except Exception as exc:
            errors = [f"{type(exc).__name__}:{exc}"]
            task_outcome = "FAILED_RETRYABLE_INFRA" if attempt <= args.max_infrastructure_retries else "FAILED_FATAL_POST_ACTION"
            atomic_write_json(parent_root / "PARENT_RESULT.json", {
                "schema": "STAGE_V_R2_FRESH_QUALIFICATION_PARENT_RESULT_V2", "status": "INFRASTRUCTURE_RESOURCE_FAILURE",
                "qualified": False, "classification": "INFRASTRUCTURE_RESOURCE_FAILURE", "canonical_parent_key": key,
                "suite": row["suite"], "task_index": row["task_index"], "state_index": row["state_index"],
                "errors": errors, "worker_gpu": gpu, "source_commit": args.source_commit, "source_tree": args.source_tree,
                "old_artifacts_reused": False, "source_artifact_read": False, **{field: 0 for field in BOUNDARIES}, "evaluated_utc": utc_now(),
            })
            if task_outcome == "FAILED_FATAL_POST_ACTION":
                queue.set_run_state("HOLD")
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2.0)
            if lease is not None:
                try:
                    post_inventory, _ = query_inventory()
                    post = next((item for item in post_inventory if int(item["gpu_id"]) == gpu), snapshot if "snapshot" in locals() else {})
                    decision_for_post = locals().get("decision", {"gpu_id": gpu, "gpu_uuid": lease["gpu_uuid"], "foreign_processes": []})
                    _write_resource(parent_root / "POST_JOB_RESOURCE_RECEIPT.json", phase="POST_JOB", decision=decision_for_post, snapshot=post, lease=lease, run_id=args.run_id, parent_key=key)
                finally:
                    store.release(lease, reason="PARENT_FINISHED")
        receipt = parent_root / "PARENT_RESULT.json"
        committed = queue.commit_result(
            task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"],
            exit_code=0 if valid else 1, error_class=None if valid else ";".join(errors), exposure_status="CLEAN_ONLY",
            task_outcome=task_outcome, output_dir=str(parent_root), receipt_sha=sha256_file(receipt) if receipt.is_file() else None,
        )
        if not committed:
            queue.set_run_state("HOLD")
            raise RuntimeError(f"QUALIFICATION_QUEUE_COMMIT_FAILED:{key}")


def _rows_report(args: argparse.Namespace, rows: list[dict[str, Any]], queue: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in queue.list_tasks():
        key = str(task["parent_id"])
        row = next(item for item in rows if item["canonical_parent_key"] == key)
        parent = Path(str(task.get("accepted_output_dir") or ""))
        result = _json(parent / "PARENT_RESULT.json") if (parent / "PARENT_RESULT.json").is_file() else {}
        reps = result.get("replicates") if isinstance(result.get("replicates"), Mapping) else {}
        output.append({
            "canonical_parent_key": key, "suite": row["suite"], "task_index": row["task_index"], "state_index": row["state_index"],
            "qualification_rank_sha256": row["qualification_rank_sha256"], "candidate": dict(row),
            "replicates": dict(reps),
            "replicate_exit_codes": result.get("replicate_exit_codes", {"A": 1, "B": 1}),
            "replicate_output_dirs": {replicate: str(parent / replicate) for replicate in ("A", "B")},
            "replicate_attempts": {replicate: [{"attempt": int(task["attempt_count"]), "output_dir": str(parent / replicate)}] for replicate in ("A", "B")},
            "engineering_valid": result.get("engineering_valid", {"A": False, "B": False}),
            "qualified": result.get("qualified") is True, "classification": result.get("classification", "MISSING_RESULT"),
            "errors": result.get("errors", ["MISSING_RESULT"]), "old_artifacts_reused": False,
            "source_commit": args.source_commit, "source_tree": args.source_tree,
        })
    return sorted(output, key=lambda item: item["canonical_parent_key"])


def run(args: argparse.Namespace) -> int:
    protocol = _json(args.protocol)
    if protocol.get("schema") != "STAGE_V_R2_FRESH_QUALIFICATION_PROTOCOL_V2" or protocol.get("status") != "FROZEN_THROUGHPUT_PARENT_ATOMIC":
        raise ValueError("fresh qualification protocol is not frozen")
    manifest, rows = _load_manifest(args.candidate_manifest, protocol["salt"])
    if args.target_per_suite != int(protocol["target_per_suite"]):
        raise ValueError("target per suite is not frozen")
    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise ValueError("fresh qualification root must be new/empty")
    args.run_root.mkdir(parents=True, exist_ok=True)
    inventory, error = query_inventory()
    if error:
        atomic_write_json(args.run_root / "PRE_QUALIFICATION_HOLD.json", {"schema": "STAGE_V_FRESH_QUALIFICATION_HOLD_V1", "reason": error, "updated_utc": utc_now()})
        return 2
    admission = admit_mode_b_or_c(inventory, mode=MODE_B, minimum_free_mib=args.minimum_free_mib, excluded_gpu_ids=())
    eligible = sorted(admission["eligible_gpu_ids"], key=lambda gpu: (-float(next(item.get("memory_free_mib") or 0 for item in admission["gpu_decisions"] if int(item["gpu_id"]) == gpu)), float(next(item.get("utilization_gpu_percent") or 100 for item in admission["gpu_decisions"] if int(item["gpu_id"]) == gpu)), int(gpu)))[:args.maximum_project_workers]
    atomic_write_json(args.run_root / "PRE_QUALIFICATION_RESOURCE_RECEIPT.json", {
        "schema": "STAGE_V_FRESH_QUALIFICATION_PRE_RESOURCE_V2", "status": "PASS" if eligible else "HOLD_NO_ELIGIBLE_GPU",
        "mode": MODE_B, "minimum_free_memory_mib": args.minimum_free_mib, "eligible_gpu_ids": eligible,
        "maximum_project_workers": args.maximum_project_workers, "maximum_project_workers_per_gpu": 1,
        "partial_fleet_allowed": True, "foreign_workload_allowed": True, "gpu_decisions": admission["gpu_decisions"], "captured_utc": utc_now(),
    })
    if not eligible:
        return 2
    manifest_sha = sha256_file(args.candidate_manifest)
    source_sha = f"{args.source_commit}:{args.source_tree}"
    queue = __import__("scripts.fec.atomic_task_queue", fromlist=["AtomicTaskQueue"]).AtomicTaskQueue(str(args.run_root / "FRESH_QUALIFICATION.sqlite"), run_id=args.run_id)
    queue.init_run(state="ACTIVE", manifest_sha=manifest_sha, source_sha=source_sha, config_sha=sha256_file(args.protocol), capacity_policy={"mode": MODE_B, "eligible_gpu_ids": eligible, "maximum_project_workers_per_gpu": 1, "partial_fleet_allowed": True, "parent_atomic": True})
    queue.register_tasks([{"cell_id": row["canonical_parent_key"], "parent_id": row["canonical_parent_key"], "suite": row["suite"], "task_index": row["task_index"], "state_index": row["state_index"], "arm": "PARENT_AB", "task_kind": "FRESH_CLEAN_AB_PARENT", "priority": index} for index, row in enumerate(rows)])
    store = GpuLeaseStore(args.run_root / "GPU_LEASES.sqlite")
    rows_by_key = {str(row["canonical_parent_key"]): row for row in rows}
    with ThreadPoolExecutor(max_workers=len(eligible), thread_name_prefix="fresh-qualification") as workers:
        futures = [workers.submit(_worker, args, int(gpu), rows_by_key, manifest_sha, source_sha, queue, store) for gpu in eligible]
        for future in futures:
            future.result()
    if queue.get_run_state() != "HOLD":
        queue.set_run_state("COMPLETE")
    report_rows = _rows_report(args, rows, queue)
    atomic_write_json(args.run_root / "Q2_CONTROL_QUALIFICATION_REPORT.json", {
        "schema": "STAGE_V_R2_FRESH_QUALIFICATION_REPORT_V2", "status": "PASS" if all(sum(item["qualified"] and item["suite"] == suite for item in report_rows) >= args.target_per_suite for suite in SUITES) and queue.get_run_state() != "HOLD" else "FAIL",
        "producer_verdict": "PASS" if queue.get_run_state() != "HOLD" else "FAIL", "source_commit": args.source_commit, "source_tree": args.source_tree,
        "protocol_sha256": sha256_file(args.protocol), "candidate_manifest_sha256": manifest_sha, "candidate_manifest": str(args.candidate_manifest.resolve()),
        "gpus": eligible, "worker_count": len(eligible), "eligible_gpu_ids": eligible, "maximum_project_workers_per_gpu": 1,
        "parent_atomic": True, "old_artifacts_reused": False, "evaluated_rows": len(report_rows),
        "evaluated_by_suite": {suite: sum(item["suite"] == suite for item in report_rows) for suite in SUITES},
        "qualified_by_suite": {suite: sum(item["qualified"] and item["suite"] == suite for item in report_rows) for suite in SUITES},
        **{field: 0 for field in BOUNDARIES}, "queue_state": queue.get_run_state(), "generated_utc": utc_now(),
    })
    atomic_write_json(args.run_root / "Q2_CONTROL_QUALIFICATION_ROWS.json", {"schema": "STAGE_V_R2_FRESH_QUALIFICATION_ROWS_V2", "rows": report_rows})
    queue.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--clean-runner", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--provenance-source", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-per-suite", type=int, default=10)
    parser.add_argument("--maximum-project-workers", type=int, default=8)
    parser.add_argument("--minimum-free-mib", type=int, default=MIN_FREE_MEMORY_MIB)
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    parser.add_argument("--min-remaining-steps", type=int, default=10)
    args = parser.parse_args(argv)
    args.run_root = args.run_root.resolve()
    args.candidate_manifest = args.candidate_manifest.resolve()
    args.protocol = args.protocol.resolve()
    args.repo_root = args.repo_root.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
