#!/usr/bin/env python3
"""Manifest-driven CLEAN-only cross-suite census queue.

Runs the suite-agnostic clean collector over the preregistered 300-episode
matrix:
  Wave A: all suites, tasks 0-9, states 0-4
  Wave B: all suites, tasks 0-9, states 5-9

No VIS, RAND, shuffled control, PGD, attack, or threshold tuning is reachable
from this queue.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

POLICIES = {
    "libero_spatial": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial",
        "unnorm_key": "libero_spatial",
    },
    "libero_goal": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal",
        "unnorm_key": "libero_goal",
    },
    "libero_10": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-10",
        "unnorm_key": "libero_10",
    },
}
SUITE_ORDER = ["libero_spatial", "libero_goal", "libero_10"]
WAVES = {"A": range(0, 5), "B": range(5, 10)}


@dataclass(frozen=True)
class CleanJob:
    job_id: str
    wave: str
    suite: str
    task_idx: int
    state_id: int
    eval_seed: int
    model_path: str
    unnorm_key: str


def build_jobs(eval_seed: int = 0) -> list[CleanJob]:
    jobs: list[CleanJob] = []
    for wave, states in WAVES.items():
        for suite in SUITE_ORDER:
            spec = POLICIES[suite]
            for task_idx in range(10):
                for state_id in states:
                    jobs.append(CleanJob(
                        job_id=f"wave{wave}_{suite}_t{task_idx:02d}_s{state_id:02d}",
                        wave=wave,
                        suite=suite,
                        task_idx=task_idx,
                        state_id=state_id,
                        eval_seed=int(eval_seed),
                        model_path=spec["model_path"],
                        unnorm_key=spec["unnorm_key"],
                    ))
    return jobs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_text(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, cwd=str(REPO)).decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def require_clean_worktree(expected_commit: str) -> None:
    head = run_text(["git", "rev-parse", "HEAD"])
    if head != expected_commit:
        raise SystemExit(f"HEAD_MISMATCH: got {head}, expected {expected_commit}")
    status = run_text(["git", "status", "--short"])
    if status.strip():
        raise SystemExit(f"DIRTY_WORKTREE:\n{status}")


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def command_for_job(args: argparse.Namespace, job: CleanJob, output_dir: Path) -> list[str]:
    return [
        args.python,
        "scripts/stageb/run_sc5_cross_suite_clean.py",
        "--suite", job.suite,
        "--model_path", job.model_path,
        "--unnorm_key", job.unnorm_key,
        "--task_idx", str(job.task_idx),
        "--state_id", str(job.state_id),
        "--eval_seed", str(job.eval_seed),
        "--detector_path", args.detector_path,
        "--source_commit", args.source_commit,
        "--output_dir", str(output_dir),
        "--render_gpu", str(args.render_gpu),
        "--save_video",
    ]


def audit_episode(output_dir: Path) -> tuple[str, str]:
    required = [
        "episode_manifest.json",
        "episode_summary.json",
        "step_telemetry.csv",
        "detector_telemetry.csv",
        "frame_index.csv",
        "privileged_sidecar.json",
        "sim_state_stream.npz",
        "sim_state_manifest.json",
        "artifact_sha256.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        return "SCIENTIFIC_INVALID", "missing_artifacts:" + "|".join(missing)
    try:
        summary = json.loads((output_dir / "episode_summary.json").read_text(encoding="utf-8"))
        sidecar = json.loads((output_dir / "privileged_sidecar.json").read_text(encoding="utf-8"))
        if summary.get("vis_or_rand_run") is not False:
            return "SCIENTIFIC_INVALID", "vis_or_rand_flag_not_false"
        if sidecar.get("privileged_valid") is not False or sidecar.get("teacher_abstain") is not True:
            return "SCIENTIFIC_INVALID", "privileged_sidecar_not_abstain"
    except Exception as exc:
        return "SCIENTIFIC_INVALID", f"audit_exception:{type(exc).__name__}:{exc}"
    return "COMPLETE", "audit_pass"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--detector_path", required=True)
    ap.add_argument("--source_commit", required=True)
    ap.add_argument("--python", default="/data/aviary/envs/openvla_official_libero_20260525/bin/python")
    ap.add_argument("--cuda_visible_devices", default="2,6")
    ap.add_argument("--render_gpu", type=int, default=6)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=200.0)
    ap.add_argument("--plan_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices != "2,6" or int(args.render_gpu) != 6:
        raise SystemExit("ONLY_ORDERED_GPU_PAIR_2_6_AUTHORIZED")
    require_clean_worktree(args.source_commit)

    root = Path(args.output_root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"OUTPUT_ROOT_NONEMPTY:{root}")
    root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.eval_seed)
    planned = [asdict(j) | {"status": "PLANNED", "output_dir": str(root / j.wave / j.job_id)} for j in jobs]
    write_csv(root / "queue_manifest.csv", planned)
    (root / "queue_config.json").write_text(json.dumps({
        "source_commit": args.source_commit,
        "cuda_visible_devices": args.cuda_visible_devices,
        "render_gpu": args.render_gpu,
        "detector_path": args.detector_path,
        "job_count": len(jobs),
        "waves": {"A": "states_0_4_all_suites", "B": "states_5_9_all_suites"},
        "vis_rand_attack": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.plan_only:
        return

    rows: list[dict[str, Any]] = []
    consecutive_audit_fail = 0
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "OPENVLA_ATTN_IMPLEMENTATION": "eager",
        "TOKENIZERS_PARALLELISM": "false",
    })
    for job in jobs:
        if disk_free_gb(root) < float(args.min_free_gb):
            raise SystemExit("LOW_DISK_SPACE_STOP")
        output_dir = root / job.wave / job.job_id
        if output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(f"OUTPUT_DIR_NONEMPTY:{output_dir}")
        cmd = command_for_job(args, job, output_dir)
        log_path = root / "logs" / f"{job.job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(job) | {
            "output_dir": str(output_dir),
            "status": "RUNNING",
            "attempt": 0,
            "command": " ".join(cmd),
            "log_path": str(log_path),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        rows.append(row)
        write_csv(root / "queue_status.csv", rows)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            row["status"] = "INFRA_FAILED"
            row["returncode"] = proc.returncode
            row["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            row["reason"] = "collector_nonzero"
            write_csv(root / "queue_status.csv", rows)
            if "illegal memory access" in log_text or "CUDA error" in log_text:
                raise SystemExit(f"GPU_INFRA_STOP:{job.job_id}")
            continue
        status, reason = audit_episode(output_dir)
        row["status"] = status
        row["reason"] = reason
        row["returncode"] = proc.returncode
        row["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if status != "COMPLETE":
            consecutive_audit_fail += 1
        else:
            consecutive_audit_fail = 0
        write_csv(root / "queue_status.csv", rows)
        if consecutive_audit_fail >= 2:
            raise SystemExit("CONSECUTIVE_ARTIFACT_AUDIT_FAILURE_STOP")


if __name__ == "__main__":
    main()
