#!/usr/bin/env python3
"""Release matched-load launcher for one suite or one exact job manifest.

The frozen CLEAN rollout is validated as a detector-only parent and is never passed
to the worker for regeneration. Only the four attack rows are delegated to the base
launcher. This preserves the clean-parent hash used for timing and matched controls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.run_c2g_matched_load_jobs import (
    read_jsonl,
    validate_parent_bindings,
)
from src.gripper_attack.c2g_matched_load_manifest import (
    ATTACK_CONDITIONS,
    validate_core_2x2_manifest,
)

REPO = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = REPO / "scripts" / "stageb" / "run_c2g_matched_load_jobs.py"
PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"


def validate_frozen_clean_parent(
    output_root: Path,
    job: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    metadata_path = output_root / str(job["parent_key"]) / "CLEAN" / "episode_metadata.json"
    steps_path = metadata_path.with_name("step_records.jsonl")
    if not metadata_path.is_file() or not steps_path.is_file() or steps_path.stat().st_size == 0:
        raise FileNotFoundError(f"frozen CLEAN parent is incomplete: {metadata_path.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"frozen CLEAN parent has no step rows: {metadata_path.parent}")
    expected_fields = {
        "parent_key": job["parent_key"],
        "condition": "CLEAN",
        "suite": job["suite"],
        "task_index": job["task_index"],
        "state_id": job["state_id"],
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": expected_commit,
        "detector_checkpoint_sha256": job["detector_checkpoint_sha256"],
    }
    mismatches = {
        field: {"expected": expected, "actual": metadata.get(field)}
        for field, expected in expected_fields.items()
        if metadata.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"frozen CLEAN parent metadata mismatch: {mismatches}")
    if metadata.get("runtime_valid") is not True:
        raise ValueError("frozen CLEAN parent runtime_valid is not true")
    if int(metadata.get("attack_delivery_count", -1)) != 0:
        raise ValueError("frozen CLEAN parent reports attacked frames")
    if any(bool(row.get("attack_delivered")) for row in rows):
        raise ValueError("frozen CLEAN step records contain an attacked frame")
    starts = [int(row["step"]) for row in rows if bool(row.get("trigger_started"))]
    if len(starts) > 1:
        raise ValueError(f"frozen CLEAN parent contains multiple detector starts: {starts}")
    return {
        "parent_key": job["parent_key"],
        "step_count": len(rows),
        "detector_start_step": starts[0] if starts else None,
        "success": metadata.get("success"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--policy-model-manifest", default="")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    jobs = read_jsonl(args.jobs.resolve())
    manifest = validate_core_2x2_manifest(
        jobs,
        strict_objective_seed_pairing=True,
    )
    output_root = args.output_root.resolve()
    parent_binding = validate_parent_bindings(
        jobs,
        output_root=output_root,
        fallback_checkpoint=args.checkpoint,
    )
    clean_jobs = [job for job in jobs if job["condition"] == "CLEAN"]
    if len(clean_jobs) != int(manifest["parent_count"]):
        raise ValueError("manifest does not contain exactly one CLEAN row per parent")
    clean_summaries = [
        validate_frozen_clean_parent(output_root, job, args.expected_git_commit)
        for job in clean_jobs
    ]

    results: list[dict[str, Any]] = []
    launched = 0
    for condition in sorted(ATTACK_CONDITIONS):
        condition_jobs = [job for job in jobs if job["condition"] == condition]
        if not condition_jobs:
            raise ValueError(f"manifest contains no jobs for {condition}")
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
            "--model-path", args.model_path,
            "--max-steps", str(args.max_steps),
            "--condition", condition,
            "--resume" if args.resume else "--no-resume",
        ]
        if args.policy_model_manifest:
            command.extend(["--policy-model-manifest", args.policy_model_manifest])
        if remaining > 0:
            command.extend(["--max-jobs", str(remaining)])
        if args.dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, cwd=REPO)
        results.append(
            {
                "condition": condition,
                "job_count": len(condition_jobs),
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "HOLD",
            }
        )
        if completed.returncode != 0:
            break
        launched += min(len(condition_jobs), remaining) if remaining > 0 else len(condition_jobs)

    status = (
        "PASS_C2G_RELEASE_JOB_LAUNCH"
        if results and all(row["status"] == "PASS" for row in results)
        else "HOLD_C2G_RELEASE_JOB_LAUNCH"
    )
    report = {
        "gate": "C2G_RELEASE_MATCHED_LOAD_JOB_LAUNCH",
        "status": status,
        "manifest_validation": manifest,
        "parent_binding": parent_binding,
        "clean_parent_rewritten": False,
        "clean_parents": clean_summaries,
        "attack_condition_results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "c2g_release_job_launcher_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
