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


FORBIDDEN = re.compile(r"(?<![A-Za-z0-9_])(?:OPEN(?:_T[0-9]+)?|VIS|PGD|ATTACK|EVAL160|PROTECTED|TEACHER)(?![A-Za-z0-9_])", re.IGNORECASE)
EXPECTED_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


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
    command_text = template.format(
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
    for field in ("terminal_outcome", "terminal_state_sha256", "key_state_identity_sha256"):
        if a.get(field) is None or b.get(field) is None or a.get(field) != b.get(field):
            errors.append(f"AB_MISMATCH:{field}")
    if a.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("A_PARENT_IDENTITY_MISMATCH")
    if b.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("B_PARENT_IDENTITY_MISMATCH")
    return not errors, sorted(set(errors))


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()] if value else [0]
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"invalid GPU list: {value!r}")
    return gpus


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    by_suite = {suite: [row for row in rows if str(row["suite"]) == suite] for suite in suites}
    keys = [str(row["canonical_parent_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate manifest contains duplicate canonical parent keys")
    rows_out: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {suite: [] for suite in suites}
    gpus = _parse_gpus(args.gpus)
    queue_db = args.output_dir / "CONTROL_QUALIFICATION.sqlite"
    queue = AtomicTaskQueue(str(queue_db), run_id="STAGE_V_R2_CONTROL_QUALIFICATION_20260807")
    manifest_sha = sha256_file(args.candidate_manifest)
    source_sha = f"{args.source_commit}:{args.source_tree}"
    queue.init_run(
        state="ACTIVE", manifest_sha=manifest_sha, source_sha=source_sha,
        config_sha=sha256_text(args.runner_command),
        capacity_policy={"mode": "atomic_dynamic_workers", "gpus": gpus, "worker_count": len(gpus), "old_artifacts_reused": False},
    )
    rows_by_key = {str(row["canonical_parent_key"]): row for row in rows}

    def qualification_worker(gpu: int, batch_keys: set[str]) -> list[tuple[str, str, int, dict[str, Any]]]:
        worker_id = f"stage-v-control-qualifier-gpu{gpu}-pid{os.getpid()}-tid{__import__('threading').get_ident()}"
        outcomes: list[tuple[str, str, int, dict[str, Any]]] = []
        try:
            while True:
                task = queue.claim_task(
                    worker_id, hostname=socket.gethostname(), pid=os.getpid(), gpu_id=gpu,
                    expected_manifest_sha=manifest_sha, expected_source_sha=source_sha,
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
            queue.close()

    cursor = {suite: 0 for suite in suites}
    while True:
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
            futures = [pool.submit(qualification_worker, gpu, batch_keys) for gpu in gpus]
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
        "gpus": gpus,
        "worker_count": len(gpus),
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
        "parents": manifest_rows,
        "selected_parents": manifest_rows,
        "selected_count": len(manifest_rows),
        "planned_parent_count": len(manifest_rows),
        "parents_by_suite": {suite: args.target_per_suite for suite in suites},
        "old_artifacts_reused": False,
        "generated_utc": utc_now(),
    }
    independent_errors: list[str] = []
    recomputed: dict[str, int] = {suite: 0 for suite in suites}
    for record in rows_out:
        recomputed_ok, recomputed_errors = qualifies(
            record, record.get("replicates", {}).get("A", {}), record.get("replicates", {}).get("B", {}),
            args.source_commit, args.source_tree,
        )
        if recomputed_ok:
            recomputed[str(record["suite"])] += 1
        if recomputed_ok != bool(record.get("qualified")) or recomputed_errors != list(record.get("errors", [])):
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
    parser.add_argument("--salt", default="STAGE_V_R2_CONTROL_QUALIFICATION_20260807")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--initial-per-suite", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--target-per-suite", type=int, default=10)
    parser.add_argument("--suites", default="")
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
        (args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V2.sha256").write_text(
            sha256_file(manifest_path) + "  STAGE_V_FORMAL_PARENT_MANIFEST_V2.json\n", encoding="utf-8",
        )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
