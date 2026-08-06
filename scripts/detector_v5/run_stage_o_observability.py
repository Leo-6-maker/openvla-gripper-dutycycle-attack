"""Run the bounded, parent-grouped Stage O observability study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, load_rows, sha256_file, utc_now
except ImportError:
    from stage_v_dynamic_common import atomic_write_json, load_rows, sha256_file, utc_now

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ImportError:
    from ..fec.atomic_task_queue import AtomicTaskQueue


FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:VIS|PGD|ATTACK|EVAL160|STUDENT|SCHEDULER|FINAL[_-]?DETECTOR)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MODES = ("O1_CAUSAL25D", "O2_NONCAUSAL25D_UPPER", "O3_PRIVILEGED_CLEAN_STATE_UPPER", "O4_RGB_CAUSAL25D")
SEEDS = (2026080711, 2026080712, 2026080713)
BOUNDARY_FIELDS = ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts")


def _select(rows: list[dict[str, Any]], suite: str, salt: str, count: int, offset: int = 0) -> list[dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("suite")) == suite]
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(f"{salt}::{row.get('canonical_parent_key')}".encode()).hexdigest(),
    )
    return ranked[offset:offset + count]


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _job_specs(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if set(str(row.get("suite")) for row in rows) != set(SUITES):
        raise RuntimeError("STAGE_O_REQUIRES_ALL_FOUR_SUITES")
    for suite in SUITES:
        suite_rows = _select(rows, suite, salt, 10)
        if len(suite_rows) != 10:
            raise RuntimeError(f"INSUFFICIENT_SUITE_ROWS:{suite}:{len(suite_rows)}/10")
        selected[suite] = {
            "train": suite_rows[:6],
            "validation": suite_rows[6:8],
            "untouched_test": suite_rows[8:10],
        }
    jobs: list[dict[str, Any]] = []
    for suite in SUITES:
        for split, split_rows in selected[suite].items():
            for row in split_rows:
                parent = str(row.get("canonical_parent_key"))
                for seed in SEEDS:
                    for mode in MODES:
                        cell_id = f"{suite}::{split}::{parent}::{seed}::{mode}"
                        jobs.append({
                            "cell_id": cell_id,
                            "suite": suite,
                            "split": split,
                            "canonical_parent_key": parent,
                            "task_index": _int(row.get("task_index", row.get("task_idx"))),
                            "state_index": _int(row.get("state_index", row.get("state_id"))),
                            "seed": seed,
                            "mode": mode,
                        })
    if len(jobs) != 480 or len({job["cell_id"] for job in jobs}) != len(jobs):
        raise RuntimeError("STAGE_O_JOB_IDENTITY_CLOSURE_FAIL")
    return jobs


def _argv(template: str, values: Mapping[str, Any]) -> list[str]:
    try:
        command = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"STAGE_O_COMMAND_TEMPLATE_INVALID:{exc}") from exc
    if FORBIDDEN.search(command):
        raise RuntimeError("FORBIDDEN_RUNNER_COMMAND")
    argv = shlex.split(command, posix=os.name != "nt")
    if not argv:
        raise RuntimeError("STAGE_O_EMPTY_RUNNER_COMMAND")
    return argv


def _run_one(
    *,
    queue_db: Path,
    run_id: str,
    manifest_sha: str,
    source_commit: str,
    worker_id: str,
    gpu: int,
    task: Mapping[str, Any],
    job: Mapping[str, Any],
    job_dir: Path,
    runner_command: str,
) -> dict[str, Any]:
    queue = AtomicTaskQueue(str(queue_db), run_id=run_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "JOB.json"
    atomic_write_json(job_path, dict(job))
    values = {**dict(job), "job_path": str(job_path), "output_dir": str(job_dir), "gpu_id": gpu}
    result: dict[str, Any] = {**dict(job), "gpu_id": gpu, "worker_id": worker_id}
    try:
        argv = _argv(runner_command, values)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        completed = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
        result_path = job_dir / "RESULT.json"
        value = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        if isinstance(value, Mapping):
            result.update(dict(value))
        else:
            result["error"] = "STAGE_O_RESULT_MISSING_OR_INVALID"
        result["exit_code"] = completed.returncode
        if any(result.get(field) != 0 for field in BOUNDARY_FIELDS):
            result["error"] = "STAGE_O_BOUNDARY_NONZERO_OR_MISSING"
        result["status"] = "PASS" if completed.returncode == 0 and result.get("status") == "PASS" and not result.get("error") else "FAIL"
        if completed.stdout:
            (job_dir / "STDOUT.log").write_text(completed.stdout, encoding="utf-8")
        if completed.stderr:
            (job_dir / "STDERR.log").write_text(completed.stderr, encoding="utf-8")
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        result.update({"status": "FAIL", "exit_code": 1, "error": str(exc)})
    result_path = job_dir / "JOB_RESULT.json"
    atomic_write_json(result_path, result)
    outcome = "DONE_VALID" if result.get("status") == "PASS" and result.get("exit_code", 1) == 0 else "FAILED_FATAL_POST_ACTION"
    committed = queue.commit_result(
        task["cell_id"], task["attempt_id"], worker_id, task["lease_token"], task["lease_epoch"],
        exit_code=int(result.get("exit_code", 1)), error_class=None if outcome == "DONE_VALID" else str(result.get("error", "STAGE_O_JOB_FAIL")),
        task_outcome=outcome, output_dir=str(job_dir), receipt_sha=sha256_file(result_path),
    )
    result["queue_commit"] = committed
    atomic_write_json(result_path, result)
    queue.close()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        raise RuntimeError("STAGE_O_REQUIRES_EIGHT_UNIQUE_GPUS")
    rows = load_rows(args.parent_manifest)
    jobs = _job_specs(rows, args.salt)
    if FORBIDDEN.search(args.runner_command):
        raise RuntimeError("FORBIDDEN_RUNNER_COMMAND")
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "STAGE_O_MANIFEST.json", {
        "schema": "STAGE_O_OBSERVABILITY_MANIFEST_V2",
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "salt": args.salt,
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "planned_jobs": len(jobs),
        "jobs": len(jobs),
        "job_specs": jobs,
        "split_counts": {suite: {split: sum(job["suite"] == suite and job["split"] == split for job in jobs) // (len(SEEDS) * len(MODES)) for split in ("train", "validation", "untouched_test")} for suite in SUITES},
        "gpus": list(args.gpus),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "generated_utc": utc_now(),
    })
    manifest_sha = sha256_file(args.output_root / "STAGE_O_MANIFEST.json")
    queue_db = args.output_root / "QUEUE.sqlite"
    run_id = f"STAGE_O_{args.output_root.name}"
    queue = AtomicTaskQueue(str(queue_db), run_id=run_id)
    queue.init_run(state="ACTIVE", manifest_sha=manifest_sha, source_sha=args.source_commit)
    queue.register_tasks([
        {"cell_id": job["cell_id"], "parent_id": job["canonical_parent_key"], "suite": job["suite"], "task_index": job["task_index"], "state_index": job["state_index"], "arm": job["mode"], "task_kind": "STAGE_O_OBSERVABILITY"}
        for job in jobs
    ])
    queue.close()
    jobs_by_id = {job["cell_id"]: job for job in jobs}
    job_dirs = {job["cell_id"]: args.output_root / "jobs" / f"{index:05d}" for index, job in enumerate(jobs)}

    def worker(gpu: int) -> list[dict[str, Any]]:
        worker_id = f"stage_o_gpu{gpu}_{os.getpid()}"
        results: list[dict[str, Any]] = []
        while True:
            queue_view = AtomicTaskQueue(str(queue_db), run_id=run_id)
            task = queue_view.claim_task(worker_id, hostname=socket.gethostname(), pid=os.getpid(), gpu_id=gpu, expected_manifest_sha=manifest_sha, expected_source_sha=args.source_commit)
            queue_view.close()
            if task is None:
                return results
            result = _run_one(
                queue_db=queue_db, run_id=run_id, manifest_sha=manifest_sha, source_commit=args.source_commit,
                worker_id=worker_id, gpu=gpu, task=task, job=jobs_by_id[task["cell_id"]],
                job_dir=job_dirs[task["cell_id"]], runner_command=args.runner_command,
            )
            results.append(result)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        worker_results = [item for group in pool.map(worker, args.gpus) for item in group]
    final_queue = AtomicTaskQueue(str(queue_db), run_id=run_id)
    tasks = final_queue.list_tasks()
    progress = final_queue.get_progress()
    failed_tasks = [task for task in tasks if task.get("state") != "DONE_VALID"]
    final_queue.set_run_state("CLOSED" if not failed_tasks and len(tasks) == len(jobs) else "ABORTED")
    final_queue.close()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.output_root / "jobs").glob("*/JOB_RESULT.json"))]
    errors = [item for item in results if item.get("status") != "PASS" or item.get("exit_code", 1) != 0 or item.get("queue_commit") is not True]
    missing = len(jobs) - len(results)
    report = {
        "schema": "STAGE_O_OBSERVABILITY_REPORT_V2",
        "status": "PASS" if not errors and missing == 0 and progress["done"] == len(jobs) else "FAIL",
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "jobs": len(jobs),
        "completed_jobs": len(results) - len(errors),
        "failed_jobs": len(errors),
        "missing_jobs": missing,
        "queue_progress": progress,
        "modes": list(MODES),
        "seeds": list(SEEDS),
        "errors": errors[:50],
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "generated_utc": utc_now(),
    }
    atomic_write_json(args.output_root / "STAGE_O_REPORT.json", report)
    job_ids = [item.get("cell_id") for item in jobs]
    result_ids = [item.get("cell_id") for item in results]
    audit = {
        "schema": "STAGE_O_INDEPENDENT_AUDIT_V2",
        "verdict": "PASS" if report["status"] == "PASS" and len(job_ids) == len(set(job_ids)) and set(job_ids) == set(result_ids) else "FAIL",
        "manifest_sha256": manifest_sha,
        "report_sha256": sha256_file(args.output_root / "STAGE_O_REPORT.json"),
        "planned_job_count": len(job_ids),
        "completed_job_count": len(result_ids),
        "missing_job_count": len(set(job_ids) - set(result_ids)),
        "duplicate_job_ids": sorted({item for item in result_ids if result_ids.count(item) > 1}),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "audited_utc": utc_now(),
    }
    atomic_write_json(args.output_root / "STAGE_O_AUDIT.json", audit)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--salt", default="STAGE_O_OBSERVABILITY_V1_20260807")
    parser.add_argument("--gpus", type=lambda value: [int(item) for item in value.split(",") if item], default=list(range(8)))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gpus:
        raise SystemExit("at least one GPU is required")
    report = run(args)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
