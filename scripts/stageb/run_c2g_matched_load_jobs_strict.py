#!/usr/bin/env python3
"""Strict matched-load launcher that never rewrites the frozen CLEAN parent.

The CLEAN detector-only rollout is the parent artifact bound into every matched job.
This wrapper validates it through the base launcher, then executes only the four
attack conditions. This prevents a malformed CLEAN row from being regenerated after
timing and parent hashes have already been frozen.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.stageb.run_c2g_matched_load_jobs import (
    complete,
    read_jsonl,
    validate_parent_bindings,
)
from src.gripper_attack.c2g_matched_load_manifest import (
    ATTACK_CONDITIONS,
    validate_core_2x2_manifest,
)

REPO = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = REPO / "scripts" / "stageb" / "run_c2g_matched_load_jobs.py"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--policy-model-manifest", default="")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs.resolve())
    manifest = validate_core_2x2_manifest(jobs, strict_objective_seed_pairing=True)
    output_root = args.output_root.resolve()
    parent_binding = validate_parent_bindings(
        jobs,
        output_root=output_root,
        fallback_checkpoint=args.checkpoint,
    )

    clean_jobs = [job for job in jobs if job["condition"] == "CLEAN"]
    clean_failures: list[dict[str, Any]] = []
    for job in clean_jobs:
        metadata = output_root / job["parent_key"] / "CLEAN" / "episode_metadata.json"
        if not complete(metadata, job, args.expected_git_commit):
            clean_failures.append(
                {
                    "parent_key": job["parent_key"],
                    "reason": "FROZEN_CLEAN_PARENT_NOT_COMPLETE",
                    "metadata": str(metadata),
                }
            )
    if clean_failures:
        print(
            json.dumps(
                {
                    "status": "HOLD_C2G_STRICT_JOB_LAUNCH",
                    "clean_failures": clean_failures,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    results: list[dict[str, Any]] = []
    launched = 0
    for condition in sorted(ATTACK_CONDITIONS):
        condition_count = sum(job["condition"] == condition for job in jobs)
        if condition_count <= 0:
            raise RuntimeError(f"manifest contains no jobs for {condition}")
        remaining = 0 if args.max_jobs <= 0 else max(0, args.max_jobs - launched)
        if args.max_jobs > 0 and remaining == 0:
            break
        command = [
            sys.executable,
            str(BASE_LAUNCHER),
            "--jobs", str(args.jobs.resolve()),
            "--output-root", str(output_root),
            "--checkpoint", args.checkpoint,
            "--expected-git-commit", args.expected_git_commit,
            "--device", args.device,
            "--max-steps", str(args.max_steps),
            "--condition", condition,
        ]
        if args.model_path:
            command.extend(["--model-path", args.model_path])
        if args.policy_model_manifest:
            command.extend(["--policy-model-manifest", args.policy_model_manifest])
        command.append("--resume" if args.resume else "--no-resume")
        if args.dry_run:
            command.append("--dry-run")
        if remaining > 0:
            command.extend(["--max-jobs", str(remaining)])
        completed = subprocess.run(command, cwd=REPO)
        results.append(
            {
                "condition": condition,
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "HOLD",
                "command": command if args.dry_run else None,
            }
        )
        if completed.returncode != 0:
            break
        launched += min(condition_count, remaining) if remaining > 0 else condition_count

    status = (
        "PASS_C2G_STRICT_JOB_LAUNCH"
        if results and all(row["status"] == "PASS" for row in results)
        else "HOLD_C2G_STRICT_JOB_LAUNCH"
    )
    report = {
        "gate": "C2G_STRICT_MATCHED_LOAD_JOB_LAUNCH",
        "status": status,
        "manifest_validation": manifest,
        "parent_binding": parent_binding,
        "clean_parent_count": len(clean_jobs),
        "clean_parent_rewritten": False,
        "attack_condition_results": results,
    }
    report_path = output_root / "c2g_strict_job_launcher_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
