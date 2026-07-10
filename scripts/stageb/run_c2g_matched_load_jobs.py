#!/usr/bin/env python3
"""Execute the frozen C2g matched-load job manifest sequentially.

The launcher is intentionally simple and provenance-heavy.  Each job is delegated
to run_c2g_clean_window_vis_pgd.py; resume accepts an existing job only when its
metadata is runtime-valid and identity/protocol fields match the frozen row.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from src.gripper_attack.c2g_matched_load_manifest import (
    CORE_CONDITIONS,
    DETECTOR_TIMING_CONDITIONS,
    validate_core_2x2_manifest,
)

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "scripts" / "stageb" / "run_c2g_clean_window_vis_pgd.py"
PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete(metadata_path: Path, job: dict[str, Any], expected_commit: str) -> bool:
    step_path = metadata_path.with_name("step_records.jsonl")
    if not metadata_path.is_file() or not step_path.is_file() or step_path.stat().st_size == 0:
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for line in step_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
    except Exception:
        return False
    return bool(
        metadata.get("runtime_valid") is True
        and metadata.get("parent_key") == job["parent_key"]
        and metadata.get("condition") == job["condition"]
        and metadata.get("suite") == job["suite"]
        and int(metadata.get("task_index", -1)) == int(job["task_index"])
        and int(metadata.get("state_id", -1)) == int(job["state_id"])
        and metadata.get("protocol_name") == PROTOCOL_NAME
        and metadata.get("protocol_version") == PROTOCOL_VERSION
        and metadata.get("git_commit") == expected_commit
    )


def command_for_job(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    load = job["load_spec"]
    command = [
        sys.executable,
        str(WORKER),
        "--parent-key", str(job["parent_key"]),
        "--condition", str(job["condition"]),
        "--checkpoint", str(job.get("checkpoint_path") or args.checkpoint),
        "--output-dir", str(args.output_root),
        "--expected-git-commit", str(args.expected_git_commit),
        "--device", str(args.device),
        "--max-steps", str(job.get("max_steps", args.max_steps)),
        "--burst-length", str(load["burst_length"]),
        "--objective-seed", str(job["objective_seed"]),
        "--epsilon", str(load["epsilon"]),
        "--step-size", str(load["step_size"]),
        "--pgd-steps", str(load["pgd_steps"]),
        "--temporal-init", str(load["temporal_init_policy"]),
        "--resize-size", str(load["image_height"]),
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path.format(suite=job["suite"])])
    if args.policy_model_manifest:
        command.extend(["--policy-model-manifest", args.policy_model_manifest])
    if bool(load.get("random_start_policy")):
        command.append("--random-start")
    else:
        command.append("--no-random-start")
    if job["condition"] not in DETECTOR_TIMING_CONDITIONS and job["condition"] != "CLEAN":
        command.extend(["--planned-start-step", str(job["planned_start_step"])])
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default="", help="optional format string containing {suite}")
    parser.add_argument("--policy-model-manifest", default="")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--condition", choices=("", *CORE_CONDITIONS), default="")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs.resolve())
    validate_core_2x2_manifest(jobs)
    if args.condition:
        jobs = [job for job in jobs if job["condition"] == args.condition]
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, 1):
        metadata_path = args.output_root / job["parent_key"] / job["condition"] / "episode_metadata.json"
        if args.resume and complete(metadata_path, job, args.expected_git_commit):
            results.append({"parent_key": job["parent_key"], "condition": job["condition"], "status": "SKIP_COMPLETE"})
            continue
        command = command_for_job(job, args)
        print(f"[{index}/{len(jobs)}] " + " ".join(command), flush=True)
        if args.dry_run:
            results.append({"parent_key": job["parent_key"], "condition": job["condition"], "status": "DRY_RUN", "command": command})
            continue
        completed = subprocess.run(command, cwd=REPO)
        status = "PASS" if completed.returncode == 0 and complete(metadata_path, job, args.expected_git_commit) else "HOLD"
        results.append({"parent_key": job["parent_key"], "condition": job["condition"], "status": status, "returncode": completed.returncode})
        if status != "PASS":
            print(json.dumps({"status": "HOLD_C2G_JOB_LAUNCH", "results": results}, indent=2), file=sys.stderr)
            return 2
    report = {"status": "PASS_C2G_JOB_LAUNCH", "job_count": len(jobs), "results": results}
    report_path = args.output_root / "c2g_job_launcher_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
