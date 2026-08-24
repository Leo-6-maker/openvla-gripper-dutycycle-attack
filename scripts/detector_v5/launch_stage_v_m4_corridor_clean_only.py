#!/usr/bin/env python3
"""Launch one clean-only A/B bundle per final parent with bounded GPU fan-out."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_key(key: str) -> str:
    return key.replace("/", "__")


def _jobs(split: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = split.get("parents")
    if split.get("schema") != "STAGE_V_TRAIN_VAL_TEST_PARENT_SPLIT_V1" or split.get("status") != "FROZEN" or not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("FINAL_SPLIT_MUST_BE_FROZEN_EXACT_40")
    jobs: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["canonical_parent_key"])
        for replicate in ("A", "B"):
            output = args.output_root / "parents" / _safe_key(key) / replicate
            log = args.output_root / "logs" / f"{_safe_key(key)}__{replicate}.log"
            command = [
                str(args.python), str(args.runner),
                "--protocol", str(args.protocol),
                "--authorization", str(args.authorization),
                "--parent-key", key,
                "--output-dir", str(output),
                "--official-snapshot-root", str(args.official_snapshot_root),
                "--upstream-root", str(args.upstream_root),
                "--model-root", str(args.model_root),
                "--gpu", "{gpu}",
                "--source-commit", args.source_commit,
                "--source-tree", args.source_tree,
                "--replicate", replicate,
            ]
            jobs.append({
                "canonical_parent_key": key,
                "replicate": replicate,
                "command_template": command,
                "output_dir": str(output),
                "log": str(log),
                "state": "PENDING",
                "outcomes_read": False,
                "protected_counters": dict(COUNTERS),
            })
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated admitted physical GPU indices")
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    args.protocol = args.protocol.resolve()
    args.authorization = args.authorization.resolve()
    args.split = args.split.resolve()
    args.runner = args.runner.resolve()
    args.official_snapshot_root = args.official_snapshot_root.resolve()
    args.upstream_root = args.upstream_root.resolve()
    args.model_root = args.model_root.resolve()
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("ADMITTED_GPU_LIST_INVALID")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"REFUSE_NONEMPTY_OUTPUT_ROOT:{args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    split = _load(args.split)
    jobs = _jobs(split, args)
    manifest: dict[str, Any] = {
        "schema": "STAGE_V_M4_FINAL_CORRIDOR_CLEAN_PREFLIGHT_LAUNCH_V2",
        "status": "RUNNING",
        "started_utc": _utc(),
        "protocol": str(args.protocol),
        "authorization": str(args.authorization),
        "formal_split": str(args.split),
        "official_python": str(args.python),
        "runner": str(args.runner),
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "admitted_gpus": gpus,
        "maximum_project_workers_per_gpu": 1,
        "replicates": ["A", "B"],
        "scope": "ALL_FINAL_40_PARENTS_CURRENT_SOURCE_A_B_REQUALIFICATION",
        "outcomes_read": False,
        "intervention_executed": False,
        "labels_generated": False,
        "protected_counters": dict(COUNTERS),
        "runner_failure_count": 0,
        "completed_count": 0,
        "tasks": jobs,
    }
    manifest_path = args.output_root / "LAUNCH_MANIFEST.json"
    _write(manifest_path, manifest)

    pending = list(range(len(jobs)))
    running: dict[int, tuple[int, subprocess.Popen[str], Any]] = {}
    # ponytail: fixed FIFO queue; one worker per admitted GPU is the whole resource policy.
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            index = pending.pop(0)
            job = jobs[index]
            log_path = Path(job["log"])
            log_handle = log_path.open("w", encoding="utf-8")
            command = [str(item).replace("{gpu}", str(gpu)) for item in job["command_template"]]
            job.update({"state": "RUNNING", "gpu": gpu, "start_utc": _utc(), "command": command})
            try:
                process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            except OSError as exc:
                log_handle.close()
                job.update({"state": "DONE", "return_code": -1, "error": f"{type(exc).__name__}:{exc}", "end_utc": _utc()})
                manifest["runner_failure_count"] += 1
                manifest["completed_count"] += 1
                continue
            job["pid"] = process.pid
            running[gpu] = (index, process, log_handle)
        for gpu, (index, process, log_handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            job = jobs[index]
            job.update({"state": "DONE", "return_code": return_code, "end_utc": _utc()})
            receipt = Path(job["output_dir"]) / "M4_CORRIDOR_PREFLIGHT.json"
            if receipt.is_file():
                try:
                    value = _load(receipt)
                    job.update({"receipt_status": value.get("status"), "probe_count": value.get("probe_count", 0), "outcomes_read": value.get("outcomes_read")})
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    job["receipt_error"] = f"{type(exc).__name__}:{exc}"
            if return_code != 0:
                manifest["runner_failure_count"] += 1
            manifest["completed_count"] += 1
            del running[gpu]
        _write(manifest_path, manifest)
        if running:
            time.sleep(2)

    manifest.update({
        "status": "COMPLETED_NO_M4_OUTCOMES" if manifest["runner_failure_count"] == 0 else "COMPLETED_WITH_RUNNER_FAILURE",
        "completed_utc": _utc(),
        "task_count": len(jobs),
    })
    _write(manifest_path, manifest)
    return 0 if manifest["runner_failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
