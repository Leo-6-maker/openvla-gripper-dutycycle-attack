#!/usr/bin/env python3
"""Execute the frozen C2g matched-load job manifest sequentially.

Each job is delegated to run_c2g_clean_window_vis_pgd.py. Before any attack, this
launcher independently rebinds the frozen manifest to the CLEAN parent artifacts,
official LIBERO init state, detector checkpoint, and detector config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.gripper_attack.c2g_matched_load_manifest import (
    CONTROL_OBJECTIVE_CONDITIONS,
    CORE_CONDITIONS,
    DETECTOR_TIMING_CONDITIONS,
    validate_core_2x2_manifest,
)

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "scripts" / "stageb" / "run_c2g_clean_window_vis_pgd.py"
PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"
SUPPORTED_CONTROL_OBJECTIVE = "SHUFFLED_GRIPPER_GRADIENT"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_file_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        content_sha = sha256_file(path)
        digest.update(f"{path.name}|{path.stat().st_size}|{content_sha}\n".encode("utf-8"))
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_parent_bindings(
    jobs: Sequence[dict[str, Any]],
    *,
    output_root: Path,
    fallback_checkpoint: str,
) -> dict[str, Any]:
    """Recompute every immutable parent/checkpoint binding before execution."""

    from libero.libero import benchmark

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_parent.setdefault(str(job["parent_key"]), []).append(job)
    suite_cache: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for parent_key, group in sorted(by_parent.items()):
        exemplar = group[0]
        clean_metadata = output_root / parent_key / "CLEAN" / "episode_metadata.json"
        clean_steps = clean_metadata.with_name("step_records.jsonl")
        if not clean_metadata.is_file() or not clean_steps.is_file() or clean_steps.stat().st_size == 0:
            raise FileNotFoundError(f"frozen CLEAN parent artifacts missing for {parent_key}")
        clean_sha = combined_file_sha256((clean_metadata, clean_steps))
        if clean_sha != exemplar["clean_parent_sha256"]:
            raise RuntimeError(
                f"clean_parent_sha256 mismatch for {parent_key}: "
                f"{clean_sha} != {exemplar['clean_parent_sha256']}"
            )

        checkpoint = Path(exemplar.get("checkpoint_path") or fallback_checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"detector checkpoint missing: {checkpoint}")
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != exemplar["detector_checkpoint_sha256"]:
            raise RuntimeError(
                f"detector checkpoint hash mismatch for {parent_key}: "
                f"{checkpoint_sha} != {exemplar['detector_checkpoint_sha256']}"
            )
        config_path = Path(exemplar["detector_config_path"]).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"detector config missing: {config_path}")
        config_sha = sha256_file(config_path)
        if config_sha != exemplar["detector_config_sha256"]:
            raise RuntimeError(
                f"detector config hash mismatch for {parent_key}: "
                f"{config_sha} != {exemplar['detector_config_sha256']}"
            )

        suite = str(exemplar["suite"])
        task_index = int(exemplar["task_index"])
        state_id = int(exemplar["state_id"])
        if suite not in suite_cache:
            suite_cache[suite] = benchmark.get_benchmark_dict()[suite]()
        states = suite_cache[suite].get_task_init_states(task_index)
        if state_id < 0 or state_id >= len(states):
            raise IndexError(f"state_id outside official init-state range for {parent_key}")
        init_sha = array_sha256(states[state_id])
        if init_sha != exemplar["initial_state_sha256"]:
            raise RuntimeError(
                f"initial_state_sha256 mismatch for {parent_key}: "
                f"{init_sha} != {exemplar['initial_state_sha256']}"
            )
        summaries.append(
            {
                "parent_key": parent_key,
                "clean_parent_sha256": clean_sha,
                "initial_state_sha256": init_sha,
                "detector_checkpoint_sha256": checkpoint_sha,
                "detector_config_sha256": config_sha,
            }
        )
    return {"status": "PASS_C2G_PARENT_BINDINGS", "parent_count": len(summaries), "parents": summaries}


def complete(metadata_path: Path, job: dict[str, Any], expected_commit: str) -> bool:
    step_path = metadata_path.with_name("step_records.jsonl")
    if not metadata_path.is_file() or not step_path.is_file() or step_path.stat().st_size == 0:
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return False
    expected_frames = int(job["expected_attacked_frames"])
    delivered = sum(bool(row.get("attack_delivered")) for row in rows)
    return bool(
        rows
        and metadata.get("runtime_valid") is True
        and metadata.get("parent_key") == job["parent_key"]
        and metadata.get("condition") == job["condition"]
        and metadata.get("suite") == job["suite"]
        and int(metadata.get("task_index", -1)) == int(job["task_index"])
        and int(metadata.get("state_id", -1)) == int(job["state_id"])
        and metadata.get("protocol_name") == PROTOCOL_NAME
        and metadata.get("protocol_version") == PROTOCOL_VERSION
        and metadata.get("git_commit") == expected_commit
        and metadata.get("objective_family") == job["objective_family"]
        and int(metadata.get("objective_seed", -1)) == int(job["objective_seed"])
        and metadata.get("detector_checkpoint_sha256") == job["detector_checkpoint_sha256"]
        and int(metadata.get("attack_delivery_count", -1)) == expected_frames
        and delivered == expected_frames
    )


def random_start_flag(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized in {"uniform_linf_seeded", "uniform_linf", "random", "enabled"}:
        return "--random-start"
    if normalized in {"zero", "none", "disabled"}:
        return "--no-random-start"
    raise ValueError(f"unsupported random_start_policy: {policy}")


def command_for_job(job: dict[str, Any], args: argparse.Namespace) -> list[str]:
    load = job["load_spec"]
    if job["condition"] in CONTROL_OBJECTIVE_CONDITIONS and job["objective_family"] != SUPPORTED_CONTROL_OBJECTIVE:
        raise ValueError(
            f"runtime currently supports only {SUPPORTED_CONTROL_OBJECTIVE}; "
            f"manifest requested {job['objective_family']}"
        )
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
        random_start_flag(load["random_start_policy"]),
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path.format(suite=job["suite"])])
    if args.policy_model_manifest:
        command.extend(["--policy-model-manifest", args.policy_model_manifest])
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
    manifest_validation = validate_core_2x2_manifest(
        jobs,
        strict_objective_seed_pairing=True,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    parent_binding = validate_parent_bindings(
        jobs,
        output_root=args.output_root.resolve(),
        fallback_checkpoint=args.checkpoint,
    )
    if args.condition:
        jobs = [job for job in jobs if job["condition"] == args.condition]
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
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
    report = {
        "status": "PASS_C2G_JOB_LAUNCH",
        "job_count": len(jobs),
        "manifest_validation": manifest_validation,
        "parent_binding": parent_binding,
        "results": results,
    }
    report_path = args.output_root / "c2g_job_launcher_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
