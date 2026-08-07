"""Run the fresh Q2 clean A/B qualification on the frozen candidate universe."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any, Mapping

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ImportError:  # direct server execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fec.atomic_task_queue import AtomicTaskQueue

try:
    from .stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, sha256_text, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, normalize_parent, sha256_file, sha256_text, utc_now


EXPECTED_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_SALT = "STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807"
POOL_SCHEMA = "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1"
VALID_RESULT_STATUSES = {"PASS", "DONE", "QUALIFIED", "TASK_FAILURE"}
FORBIDDEN = re.compile(r"(?<![A-Za-z0-9_])(?:OPEN(?:_T[0-9]+)?|VIS|PGD|ATTACK|EVAL160|PROTECTED)(?![A-Za-z0-9_])", re.IGNORECASE)


def _render_command(template: str, **values: object) -> str:
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _ranked(rows: list[Mapping[str, Any]], salt: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = normalize_parent(raw)
        key = str(row["canonical_parent_key"])
        row["qualification_rank_sha256"] = hashlib.sha256(f"{salt}::{key}".encode()).hexdigest()
        output.append(row)
    return sorted(output, key=lambda item: (str(item["qualification_rank_sha256"]), str(item["canonical_parent_key"])))


def load_candidate_universe(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = _load_object(path)
    if pool.get("schema") != POOL_SCHEMA:
        raise ValueError("Q2 candidate universe is not the frozen clean-only pool")
    candidates = pool.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Q2 candidate universe has no candidates")
    expected_count = int(pool.get("candidate_count", len(candidates)))
    if expected_count != len(candidates):
        raise ValueError("Q2 candidate universe count mismatch")
    if pool.get("selection_frozen_before_new_rollouts") is not True:
        raise ValueError("Q2 candidate selection was not frozen before new rollouts")
    gates = pool.get("gates")
    if not isinstance(gates, Mapping) or any(int(gates.get(key, 1) or 0) != 0 for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts")):
        raise ValueError("Q2 candidate universe boundary gate is nonzero")
    rows = [normalize_parent(row) for row in candidates if isinstance(row, Mapping)]
    if len(rows) != len(candidates):
        raise ValueError("Q2 candidate universe contains a non-object row")
    keys = [str(row["canonical_parent_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Q2 candidate universe contains duplicate parent keys")
    if {str(row["suite"]) for row in rows} != set(EXPECTED_SUITES):
        raise ValueError("Q2 candidate universe must contain all four suites")
    if any(row.get("legacy_g10_test_only") is not True for row in rows):
        raise ValueError("Q2 candidate universe contains a non-frozen candidate")
    counts = {suite: sum(str(row["suite"]) == suite for row in rows) for suite in EXPECTED_SUITES}
    configured = pool.get("candidates_per_suite")
    if isinstance(configured, int) and any(count != configured for count in counts.values()):
        raise ValueError("Q2 candidate universe is not balanced by suite")
    pool["counts_by_suite"] = counts
    return pool, rows


def _result_from_directory(directory: Path) -> Mapping[str, Any] | None:
    for name in ("CONTROL_RESULT.json", "RESULT.json", "PARENT_RESULT.json"):
        path = directory / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                return value
    return None


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _run_once(template: str, *, candidate_path: Path, output_dir: Path, replicate: str,
              source_commit: str, source_tree: str, gpu: int) -> tuple[int, dict[str, Any]]:
    command_text = _render_command(
        template, candidate_path=candidate_path, output_dir=output_dir, replicate=replicate,
        source_commit=source_commit, source_tree=source_tree, gpu=gpu,
    )
    if FORBIDDEN.search(command_text):
        return 2, {"status": "FAIL", "reason": "FORBIDDEN_COMMAND_TOKEN", "exit_code": 2}
    completed = subprocess.run(command_text, shell=True, check=False, capture_output=True, text=True)
    result = _result_from_directory(output_dir)
    payload = dict(result) if result else {
        "status": "FAIL", "reason": "MISSING_CONTROL_RESULT", "exit_code": completed.returncode,
    }
    payload.update({
        "process_exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    })
    return completed.returncode, payload


def engineering_valid(row: Mapping[str, Any], result: Mapping[str, Any], process_exit_code: int,
                      source_commit: str, source_tree: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if process_exit_code != 0 or result.get("exit_code") != 0:
        errors.append("PROCESS_EXIT_NONZERO")
    if result.get("status") not in VALID_RESULT_STATUSES:
        errors.append("RESULT_STATUS_INVALID")
    for field in ("snapshot_restore_valid", "task_identity_valid", "runtime_valid", "metrics_finite", "artifact_validation_pass"):
        if result.get(field) is not True:
            errors.append(f"{field.upper()}_FALSE")
    if result.get("old_artifacts_reused") is not False:
        errors.append("OLD_ARTIFACT_REUSE")
    if result.get("source_commit") != source_commit or result.get("source_tree") != source_tree:
        errors.append("SOURCE_PROVENANCE_MISMATCH")
    if result.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("PARENT_IDENTITY_MISMATCH")
    if not result.get("key_state_identity_sha256"):
        errors.append("INITIAL_STATE_IDENTITY_MISSING")
    for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts"):
        if result.get(field, 0) != 0:
            errors.append(f"BOUNDARY_VIOLATION:{field}")
    return not errors, sorted(set(errors))


def qualify_pair(row: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any],
                 a_valid: bool, b_valid: bool, source_commit: str, source_tree: str) -> tuple[bool, str, list[str]]:
    errors: list[str] = []
    if not a_valid:
        errors.append("A_ENGINEERING_INVALID")
    if not b_valid:
        errors.append("B_ENGINEERING_INVALID")
    if errors:
        return False, "ENGINEERING_INVALID", sorted(errors)
    a_success = a.get("clean_success") is True
    b_success = b.get("clean_success") is True
    if a.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("A_PARENT_IDENTITY_MISMATCH")
    if b.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("B_PARENT_IDENTITY_MISMATCH")
    if not a.get("key_state_identity_sha256") or not b.get("key_state_identity_sha256"):
        errors.append("INITIAL_STATE_IDENTITY_MISSING")
    elif a.get("key_state_identity_sha256") != b.get("key_state_identity_sha256"):
        errors.append("AB_INITIAL_STATE_IDENTITY_MISMATCH")
    if errors:
        return False, "CLEAN_REPEATABILITY_FAIL_IDENTITY", sorted(set(errors))
    if a_success and b_success:
        return True, "QUALIFIED", []
    if a_success and not b_success:
        return False, "CLEAN_REPEATABILITY_FAIL_A_SUCCESS_B_FAIL", ["B_CLEAN_SUCCESS_FALSE"]
    if not a_success and b_success:
        return False, "CLEAN_REPEATABILITY_FAIL_A_FAIL_B_SUCCESS", ["A_CLEAN_SUCCESS_FALSE"]
    return False, "CLEAN_REPEATABILITY_FAIL_BOTH_FAIL", ["A_CLEAN_SUCCESS_FALSE", "B_CLEAN_SUCCESS_FALSE"]


def _heartbeat_loop(queue: AtomicTaskQueue, task: Mapping[str, Any], worker_id: str, stop: threading.Event) -> None:
    while not stop.wait(30.0):
        queue.heartbeat(task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"])


def _run_task(queue: AtomicTaskQueue, task: Mapping[str, Any], *, worker_id: str, args: argparse.Namespace,
              rows_by_key: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    key = str(task["parent_id"])
    row = rows_by_key[key]
    replicate = str(task["arm"])
    base = args.output_dir / "qualification" / str(row["suite"]) / key.replace("/", "__")
    attempt_number = int(task.get("attempt_count") or 1)
    output_dir = base / replicate / f"attempt_{attempt_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    queue.heartbeat(task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"])
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(queue, task, worker_id, stop), daemon=True)
    heartbeat.start()
    try:
        code, result = _run_once(
            args.runner_command, candidate_path=base / "CANDIDATE.json", output_dir=output_dir,
            replicate=replicate, source_commit=args.source_commit, source_tree=args.source_tree, gpu=int(task["gpu_id"]),
        )
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
    valid, errors = engineering_valid(row, result, code, args.source_commit, args.source_tree)
    if valid:
        outcome = "DONE_VALID"
    elif attempt_number <= args.max_infrastructure_retries:
        outcome = "FAILED_RETRYABLE_INFRA"
    else:
        outcome = "FAILED_FATAL_POST_ACTION"
    receipt = output_dir / "CONTROL_RESULT.json"
    committed = queue.commit_result(
        task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"],
        exit_code=code, error_class=None if valid else ";".join(errors), exposure_status="CLEAN_ONLY",
        task_outcome=outcome, output_dir=str(output_dir), receipt_sha=sha256_file(receipt) if receipt.is_file() else None,
    )
    if not committed:
        raise RuntimeError(f"Q2_QUEUE_COMMIT_FAILED:{key}:{replicate}:{attempt_number}")
    if outcome == "FAILED_FATAL_POST_ACTION":
        queue.set_run_state("HOLD")
    attempt_dirs = []
    for path in sorted((base / replicate).glob("attempt_*")):
        if path.is_dir():
            try:
                attempt = int(path.name.removeprefix("attempt_"))
            except ValueError:
                continue
            attempt_dirs.append({"attempt": attempt, "output_dir": str(path)})
    return {
        "canonical_parent_key": key, "replicate": replicate, "output_dir": str(output_dir), "attempt_number": attempt_number,
        "process_exit_code": code, "result": result, "engineering_valid": valid,
        "engineering_errors": errors, "attempts": attempt_dirs, "final": outcome != "FAILED_RETRYABLE_INFRA",
    }


def _worker(queue: AtomicTaskQueue, gpu: int, batch_keys: set[str], *, args: argparse.Namespace,
            rows_by_key: Mapping[str, Mapping[str, Any]], manifest_sha: str, source_sha: str) -> list[dict[str, Any]]:
    worker_id = f"stage-v-q2-gpu{gpu}-pid{os.getpid()}-tid{threading.get_ident()}"
    results: list[dict[str, Any]] = []
    try:
        while True:
            task = queue.claim_task(
                worker_id, hostname=socket.gethostname(), pid=os.getpid(), gpu_id=gpu,
                expected_manifest_sha=manifest_sha, expected_source_sha=source_sha,
            )
            if task is None:
                return results
            key = str(task["parent_id"])
            if key not in batch_keys or key not in rows_by_key:
                queue.set_run_state("HOLD")
                raise RuntimeError(f"Q2_QUEUE_IDENTITY_FAIL:{key}")
            outcome = _run_task(queue, {**task, "gpu_id": gpu}, worker_id=worker_id, args=args, rows_by_key=rows_by_key)
            if outcome["final"]:
                results.append(outcome)
    finally:
        queue.close()


def _build_parent_record(row: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]],
                         source_commit: str, source_tree: str) -> dict[str, Any]:
    reps: dict[str, dict[str, Any]] = {}
    for replicate in ("A", "B"):
        item = results.get(replicate)
        if item is None:
            reps[replicate] = {"status": "FAIL", "exit_code": 1, "process_exit_code": 1, "reason": "NOT_EVALUATED"}
        else:
            reps[replicate] = dict(item.get("result") or {})
            reps[replicate]["process_exit_code"] = item.get("process_exit_code")
    a_valid = bool((results.get("A") or {}).get("engineering_valid"))
    b_valid = bool((results.get("B") or {}).get("engineering_valid"))
    qualified, classification, errors = qualify_pair(row, reps["A"], reps["B"], a_valid, b_valid, source_commit, source_tree)
    a_hash = reps["A"].get("terminal_state_sha256")
    b_hash = reps["B"].get("terminal_state_sha256")
    a_horizon = reps["A"].get("remaining_horizon_complete")
    b_horizon = reps["B"].get("remaining_horizon_complete")
    return {
        "schema": "STAGE_Q2_CONTROL_QUALIFICATION_ROW_V1",
        "canonical_parent_key": row["canonical_parent_key"], "suite": row["suite"],
        "task_index": int(row["task_index"]), "state_index": int(row["state_index"]),
        "qualification_rank_sha256": row["qualification_rank_sha256"],
        "candidate": dict(row), "replicates": reps,
        "replicate_output_dirs": {replicate: (results.get(replicate) or {}).get("output_dir") for replicate in ("A", "B")},
        "replicate_attempts": {replicate: (results.get(replicate) or {}).get("attempts", []) for replicate in ("A", "B")},
        "engineering_valid": {"A": a_valid, "B": b_valid},
        "terminal_state_sha256_equal": bool(a_hash and b_hash and a_hash == b_hash),
        "remaining_horizon_complete_equal": a_horizon == b_horizon,
        "remaining_horizon_complete": {"A": a_horizon, "B": b_horizon},
        "qualified": qualified, "classification": classification, "errors": errors,
        "old_artifacts_reused": False, "source_commit": source_commit, "source_tree": source_tree,
        "evaluated_utc": utc_now(),
    }


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = _load_object(args.protocol)
    if protocol.get("schema") != "STAGE_Q2_PROTOCOL_V1" or protocol.get("status") != "FROZEN":
        raise ValueError("Q2 protocol is not frozen")
    if protocol.get("source_commit") != args.source_commit or protocol.get("source_tree") != args.source_tree:
        raise ValueError("Q2 protocol/source binding mismatch")
    pool, raw_rows = load_candidate_universe(args.candidate_universe)
    manifest_sha = sha256_file(args.candidate_universe)
    if protocol.get("candidate_universe_sha256") != manifest_sha:
        raise ValueError("Q2 candidate universe SHA256 mismatch")
    rows = _ranked(raw_rows, args.salt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError(f"Q2 output root must be new/empty: {args.output_dir}")
    source_sha = f"{args.source_commit}:{args.source_tree}"
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or 5 in gpus:
        raise ValueError("Q2 GPU list is invalid or includes excluded GPU5")
    queue = AtomicTaskQueue(str(args.output_dir / "Q2_CONTROL_QUALIFICATION.sqlite"), run_id=args.salt)
    queue.init_run(
        state="ACTIVE", manifest_sha=manifest_sha, source_sha=source_sha,
        config_sha=sha256_text(args.runner_command),
        capacity_policy={"mode": "atomic_dynamic_workers", "gpus": gpus, "worker_count": len(gpus), "gpu5_excluded": True, "old_artifacts_reused": False},
    )
    by_suite = {suite: [row for row in rows if str(row["suite"]) == suite] for suite in EXPECTED_SUITES}
    cursors = {suite: 0 for suite in EXPECTED_SUITES}
    selected = {suite: [] for suite in EXPECTED_SUITES}
    rows_out: list[dict[str, Any]] = []
    expansion_history: list[dict[str, Any]] = []
    engineering_hard_stop = False
    while True:
        batch_rows: list[dict[str, Any]] = []
        batch_by_suite: dict[str, int] = {}
        for suite in EXPECTED_SUITES:
            if len(selected[suite]) >= args.target_per_suite:
                continue
            start = cursors[suite]
            take = args.initial_per_suite if start == 0 else args.batch_size
            end = min(start + take, len(by_suite[suite]))
            if end > start:
                batch_rows.extend(by_suite[suite][start:end])
                batch_by_suite[suite] = end - start
                cursors[suite] = end
        if not batch_rows:
            break
        batch_keys = {str(row["canonical_parent_key"]) for row in batch_rows}
        rows_by_key = {str(row["canonical_parent_key"]): row for row in batch_rows}
        for row in batch_rows:
            base = args.output_dir / "qualification" / str(row["suite"]) / str(row["canonical_parent_key"]).replace("/", "__")
            base.mkdir(parents=True, exist_ok=False)
            atomic_write_json(base / "CANDIDATE.json", {**row, "old_artifacts_reused": False, "source_artifact_read": False, "qualification_mode": "FRESH_CLEAN_AB_REPLAY"})
            queue.register_tasks([
                {"cell_id": f"Q2|{row['canonical_parent_key'].replace('/', '__')}|{replicate}", "parent_id": row["canonical_parent_key"],
                 "suite": row["suite"], "task_index": int(row["task_index"]), "state_index": int(row["state_index"]),
                 "arm": replicate, "task_kind": "Q2_CONTROL_QUALIFICATION"}
                for replicate in ("A", "B")
            ])
        outcomes: list[dict[str, Any]] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(gpus), thread_name_prefix="stage-v-q2") as pool_executor:
            futures = [pool_executor.submit(_worker, queue, gpu, batch_keys, args=args, rows_by_key=rows_by_key, manifest_sha=manifest_sha, source_sha=source_sha) for gpu in gpus]
            for future in as_completed(futures):
                try:
                    outcomes.extend(future.result())
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}:{exc}")
        if errors:
            queue.set_run_state("HOLD")
            engineering_hard_stop = True
            atomic_write_json(args.output_dir / "Q2_ENGINEERING_HARD_STOP.json", {"schema": "STAGE_Q2_ENGINEERING_HARD_STOP_V1", "errors": errors, "updated_utc": utc_now()})
        by_key_rep = {(str(item.get("canonical_parent_key")), str(item.get("replicate"))): item for item in outcomes}
        batch_rows_count = 0
        for row in batch_rows:
            result_pair = {replicate: by_key_rep.get((str(row["canonical_parent_key"]), replicate)) for replicate in ("A", "B")}
            if any(value is not None for value in result_pair.values()):
                record = _build_parent_record(row, result_pair, args.source_commit, args.source_tree)
                rows_out.append(record)
                batch_rows_count += 1
                if record["classification"] == "ENGINEERING_INVALID":
                    engineering_hard_stop = True
                if record["qualified"] and len(selected[str(row["suite"])]) < args.target_per_suite:
                    selected[str(row["suite"])].append(dict(row))
            else:
                rows_out.append(_build_parent_record(row, {}, args.source_commit, args.source_tree))
        expansion_history.append({"evaluated_by_suite": batch_by_suite, "qualified_by_suite": {suite: len(selected[suite]) for suite in EXPECTED_SUITES}, "evaluated_rows": batch_rows_count, "updated_utc": utc_now()})
        if engineering_hard_stop:
            queue.set_run_state("HOLD")
            break
    report = {
        "schema": "STAGE_Q2_CONTROL_QUALIFICATION_REPORT_V1",
        "status": "PASS" if all(len(selected[suite]) >= args.target_per_suite for suite in EXPECTED_SUITES) and not engineering_hard_stop else "FAIL",
        "producer_verdict": "PASS" if all(len(selected[suite]) >= args.target_per_suite for suite in EXPECTED_SUITES) and not engineering_hard_stop else "FAIL",
        "salt": args.salt, "source_commit": args.source_commit, "source_tree": args.source_tree,
        "protocol_sha256": sha256_file(args.protocol), "candidate_universe_sha256": manifest_sha,
        "candidate_universe_count": len(rows), "candidate_universe_counts_by_suite": pool["counts_by_suite"],
        "source_clean_root": args.source_clean_root, "gpus": gpus, "worker_count": len(gpus),
        "initial_per_suite": args.initial_per_suite, "batch_size": args.batch_size, "target_per_suite": args.target_per_suite,
        "evaluated_by_suite": {suite: sum(record["suite"] == suite for record in rows_out) for suite in EXPECTED_SUITES},
        "qualified_by_suite": {suite: len(selected[suite]) for suite in EXPECTED_SUITES},
        "selected_parents_by_suite": {suite: [row["canonical_parent_key"] for row in selected[suite]] for suite in EXPECTED_SUITES},
        "expansion_history": expansion_history, "engineering_hard_stop": engineering_hard_stop,
        "engineering_invalid_rows": sum(record["classification"] == "ENGINEERING_INVALID" for record in rows_out),
        "clean_repeatability_failure_rows": sum(str(record["classification"]).startswith("CLEAN_REPEATABILITY_FAIL") for record in rows_out),
        "evaluated_rows": len(rows_out), "old_artifacts_reused": False,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
        "queue_db": str(args.output_dir / "Q2_CONTROL_QUALIFICATION.sqlite"), "queue_progress": queue.get_progress(),
        "independent_audit_verdict": "PENDING", "generated_utc": utc_now(),
    }
    queue.set_run_state("HOLD" if engineering_hard_stop else "COMPLETE")
    queue.close()
    return report, rows_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--source-clean-root", required=True)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--initial-per-suite", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--target-per-suite", type=int, default=10)
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    args = parser.parse_args(argv)
    if args.salt != DEFAULT_SALT or args.max_infrastructure_retries != 1:
        parser.error("Q2 salt and retry policy are frozen")
    args.protocol = args.protocol.resolve()
    args.candidate_universe = args.candidate_universe.resolve()
    args.output_dir = args.output_dir.resolve()
    report, rows = qualify(args)
    atomic_write_json(args.output_dir / "Q2_CONTROL_QUALIFICATION_REPORT.json", report)
    with (args.output_dir / "Q2_CONTROL_QUALIFICATION_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
