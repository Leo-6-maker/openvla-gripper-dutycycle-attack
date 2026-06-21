#!/usr/bin/env python3
"""Execute one assigned worker from the provisional two-suite Layer3 job manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROVISIONAL_SENTINEL = "PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def command_for_job(job: dict[str, str], *, python_bin: str) -> list[str]:
    return [
        python_bin,
        "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py",
        "--condition",
        job["condition"],
        "--suite",
        job["suite"],
        "--model_path",
        job["model_path"],
        "--unnorm_key",
        job["unnorm_key"],
        "--task_idx",
        job["task_idx"],
        "--state_id",
        job["state_id"],
        "--anchor",
        job["teacher_anchor"],
        "--seed_id",
        job["attack_seed"],
        "--output_dir",
        job["output_dir"],
        "--render_gpu",
        job["render_gpu"],
        "--mlp_path",
        job["detector_path"],
        "--write_video",
    ]


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    jobs = [row for row in read_csv(Path(args.job_manifest)) if row["assigned_worker"] == args.assigned_worker]
    if not jobs:
        raise SystemExit(f"no jobs for worker {args.assigned_worker}")
    out = Path(args.worker_output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"worker_output_dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / PROVISIONAL_SENTINEL).write_text("Provisional two-suite Layer3 worker output. Not final paper evidence.\n", encoding="utf-8")

    ledger: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "eager")

    for job in jobs:
        cmd = command_for_job(job, python_bin=args.python_bin)
        started = time.time()
        row = {
            "job_id": job["job_id"],
            "parent_key": job["parent_key"],
            "condition": job["condition"],
            "assigned_worker": args.assigned_worker,
            "cuda_visible_devices": args.cuda_visible_devices,
            "output_dir": job["output_dir"],
            "command": " ".join(cmd),
            "started_at": started,
            "returncode": "",
            "status": "RUNNING",
        }
        log_path = out / f"{job['job_id']}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        row["finished_at"] = time.time()
        row["duration_sec"] = row["finished_at"] - started
        row["returncode"] = proc.returncode
        row["log_path"] = str(log_path)
        row["status"] = "COMPLETE" if proc.returncode == 0 else "FAILED"
        ledger.append(row)
        write_csv(out / "worker_command_ledger.csv", ledger)
        if proc.returncode != 0 and not args.continue_on_failure:
            break

    summary = {
        "assigned_worker": args.assigned_worker,
        "cuda_visible_devices": args.cuda_visible_devices,
        "job_count": len(jobs),
        "complete_count": sum(1 for row in ledger if row["status"] == "COMPLETE"),
        "failed_count": sum(1 for row in ledger if row["status"] == "FAILED"),
        "ledger": str(out / "worker_command_ledger.csv"),
    }
    write_json(out / "worker_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-manifest", required=True)
    ap.add_argument("--assigned-worker", required=True, choices=["PAIR_A", "PAIR_B"])
    ap.add_argument("--cuda-visible-devices", required=True)
    ap.add_argument("--worker-output-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--continue-on-failure", action="store_true")
    return ap.parse_args()


def main() -> None:
    run_worker(parse_args())


if __name__ == "__main__":
    main()

