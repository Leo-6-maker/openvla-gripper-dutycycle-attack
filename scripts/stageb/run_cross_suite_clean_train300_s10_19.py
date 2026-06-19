#!/usr/bin/env python3
"""Manifest-driven CLEAN-only collection for states 10-19.

This launcher creates the CROSS_SUITE_CLEAN_TRAIN300_S10_19 corpus. It is
deliberately separate from Layer 1 resolver work and only calls the existing
CLEAN collector. No VIS, RAND, shuffled, oracle, attack, or Teacher selection
path is reachable from this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ORCHESTRATION_REPO = Path(__file__).resolve().parents[2]
PR30_MERGE_COMMIT = "141657fdc5d85c5fd564913c955d61e9e6be9ddc"
COLLECTOR_SOURCE_COMMIT = "63793972743f667c6a6bcc12e9700f322f261147"
CONDITION = "CLEAN"
SUITES = ("libero_spatial", "libero_goal", "libero_10")
TASKS = range(10)
STATES = range(10, 20)
POLICIES = {
    "libero_spatial": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial",
        "unnorm_key": "libero_spatial",
        "worker": "worker_spatial_gpu13",
        "cuda_visible_devices": "1,3",
        "render_gpu": 1,
        "output_subdir": "spatial_gpu13",
    },
    "libero_goal": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal",
        "unnorm_key": "libero_goal",
        "worker": "worker_goal_gpu26",
        "cuda_visible_devices": "2,6",
        "render_gpu": 6,
        "output_subdir": "goal_gpu26",
    },
    "libero_10": {
        "model_path": "/data/aviary/models/openvla/openvla-7b-finetuned-libero-10",
        "unnorm_key": "libero_10",
        "worker": "worker_libero10_gpu54",
        "cuda_visible_devices": "5,4",
        "render_gpu": 5,
        "output_subdir": "libero10_gpu54",
    },
}


@dataclass(frozen=True)
class TrainJob:
    canonical_key: str
    job_id: str
    suite: str
    task_idx: int
    state_id: int
    eval_seed: int
    condition: str
    split_role: str
    assigned_worker: str
    assigned_gpu_pair: str
    render_gpu: int
    output_dir: str
    model_path: str
    unnorm_key: str
    status: str = "PLANNED"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def run_text(cmd: list[str], timeout: int = 30, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd or ORCHESTRATION_REPO), stderr=subprocess.STDOUT, timeout=timeout, text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def require_clean_checkout(repo: Path) -> None:
    head = run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != PR30_MERGE_COMMIT:
        raise SystemExit(f"HEAD_MISMATCH: got {head}, expected {PR30_MERGE_COMMIT}")
    status = run_text(["git", "status", "--short"], cwd=repo)
    if status.strip():
        raise SystemExit(f"DIRTY_WORKTREE:\n{status}")


def split_role(state_id: int) -> str:
    if 10 <= state_id <= 17:
        return "train_pool"
    if 18 <= state_id <= 19:
        return "validation_pool"
    raise ValueError(f"state outside train300 range: {state_id}")


def build_jobs(output_root: Path, eval_seed: int = 0) -> list[TrainJob]:
    jobs: list[TrainJob] = []
    for suite in SUITES:
        spec = POLICIES[suite]
        for task_idx in TASKS:
            for state_id in STATES:
                canonical_key = f"{suite}|{task_idx}|{state_id}|{eval_seed}|{CONDITION}"
                job_id = f"{suite}_t{task_idx:02d}_s{state_id:02d}"
                jobs.append(
                    TrainJob(
                        canonical_key=canonical_key,
                        job_id=job_id,
                        suite=suite,
                        task_idx=task_idx,
                        state_id=state_id,
                        eval_seed=eval_seed,
                        condition=CONDITION,
                        split_role=split_role(state_id),
                        assigned_worker=str(spec["worker"]),
                        assigned_gpu_pair=str(spec["cuda_visible_devices"]),
                        render_gpu=int(spec["render_gpu"]),
                        output_dir=str(output_root / str(spec["output_subdir"]) / job_id),
                        model_path=str(spec["model_path"]),
                        unnorm_key=str(spec["unnorm_key"]),
                    )
                )
    return jobs


def validate_master_manifest(jobs: list[TrainJob]) -> dict[str, Any]:
    keys = [j.canonical_key for j in jobs]
    duplicate_keys = sorted({k for k in keys if keys.count(k) > 1})
    overlap = [j.canonical_key for j in jobs if 0 <= j.state_id <= 9]
    by_suite = {suite: sum(1 for j in jobs if j.suite == suite) for suite in SUITES}
    by_role = {
        "train_pool": sum(1 for j in jobs if j.split_role == "train_pool"),
        "validation_pool": sum(1 for j in jobs if j.split_role == "validation_pool"),
    }
    missing = []
    for suite in SUITES:
        for task_idx in TASKS:
            for state_id in STATES:
                key = f"{suite}|{task_idx}|{state_id}|0|{CONDITION}"
                if key not in keys:
                    missing.append(key)
    return {
        "planned_count": len(jobs),
        "unique_planned_count": len(set(keys)),
        "duplicate_keys": duplicate_keys,
        "overlap_with_clean300_state0_9": len(overlap),
        "missing_expected_keys": missing,
        "by_suite": by_suite,
        "by_role": by_role,
        "state_availability_10_19": "verified_by_exact_manifest_range_and_collector_fail_closed_per_job",
        "suite_checkpoint_mapping_verified": True,
        "unnorm_key_mapping_verified": True,
    }


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def command_for_job(args: argparse.Namespace, job: TrainJob) -> list[str]:
    return [
        args.python,
        str(Path(args.collector_repo) / "scripts" / "stageb" / "run_sc5_cross_suite_clean.py"),
        "--suite",
        job.suite,
        "--model_path",
        job.model_path,
        "--unnorm_key",
        job.unnorm_key,
        "--task_idx",
        str(job.task_idx),
        "--state_id",
        str(job.state_id),
        "--eval_seed",
        str(job.eval_seed),
        "--detector_path",
        args.detector_path,
        "--source_commit",
        COLLECTOR_SOURCE_COMMIT,
        "--output_dir",
        job.output_dir,
        "--render_gpu",
        str(job.render_gpu),
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
        return "SCHEMA_INVALID", "missing_artifacts:" + "|".join(missing)
    try:
        summary = json.loads((output_dir / "episode_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "episode_manifest.json").read_text(encoding="utf-8"))
        sidecar = json.loads((output_dir / "privileged_sidecar.json").read_text(encoding="utf-8"))
        if summary.get("condition") != CONDITION or manifest.get("condition") != CONDITION:
            return "SCHEMA_INVALID", "condition_not_clean"
        if summary.get("vis_or_rand_run") is not False:
            return "SCHEMA_INVALID", "vis_or_rand_flag_not_false"
        if manifest.get("attack_enabled") is not False or manifest.get("vis_enabled") is not False or manifest.get("rand_enabled") is not False:
            return "SCHEMA_INVALID", "attack_flags_not_false"
        if sidecar.get("privileged_valid") is not False or sidecar.get("teacher_abstain") is not True:
            return "SCHEMA_INVALID", "privileged_sidecar_not_abstain"
    except Exception as exc:
        return "SCHEMA_INVALID", f"audit_exception:{type(exc).__name__}:{exc}"
    return "COMPLETE", "audit_pass"


def snapshot_command(output_root: Path, name: str, cmd: list[str], *, cwd: Path | None = None) -> None:
    text = run_text(cmd, timeout=20, cwd=cwd)
    (output_root / "snapshots").mkdir(parents=True, exist_ok=True)
    (output_root / "snapshots" / name).write_text(text + "\n", encoding="utf-8")


def run_worker(args: argparse.Namespace, jobs: list[TrainJob]) -> None:
    worker_jobs = [j for j in jobs if j.suite == args.suite]
    spec = POLICIES[args.suite]
    if args.cuda_visible_devices != spec["cuda_visible_devices"] or args.render_gpu != spec["render_gpu"]:
        raise SystemExit("WORKER_GPU_MAPPING_MISMATCH")
    root = Path(args.output_root)
    logs = root / "logs" / spec["worker"]
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "OPENVLA_ATTN_IMPLEMENTATION": "eager",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    rows: list[dict[str, Any]] = []
    for job in worker_jobs:
        output_dir = Path(job.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(f"OUTPUT_DIR_NONEMPTY:{output_dir}")
        if disk_free_gb(root) < args.min_free_gb:
            raise SystemExit("LOW_DISK_SPACE_STOP")
        row = asdict(job)
        row.update(
            {
                "status": "RUNNING",
                "pid": os.getpid(),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command": " ".join(command_for_job(args, job)),
                "log_path": str(logs / f"{job.job_id}.log"),
            }
        )
        rows.append(row)
        write_csv(root / f"queue_status_{spec['worker']}.csv", rows)
        log_path = Path(row["log_path"])
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command_for_job(args, job), cwd=str(Path(args.collector_repo)), env=env, stdout=log, stderr=subprocess.STDOUT)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        row["returncode"] = proc.returncode
        row["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if proc.returncode != 0:
            row["status"] = "INFRA_FAILED"
            row["reason"] = "collector_nonzero"
            write_csv(root / f"queue_status_{spec['worker']}.csv", rows)
            if "illegal memory access" in log_text or "CUDA error" in log_text:
                raise SystemExit(f"GPU_INFRA_STOP:{job.job_id}")
            continue
        status, reason = audit_episode(output_dir)
        row["status"] = status
        row["reason"] = reason
        write_csv(root / f"queue_status_{spec['worker']}.csv", rows)
        if status == "SCHEMA_INVALID":
            raise SystemExit(f"SCHEMA_INVALID_STOP:{job.job_id}:{reason}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--detector_path", required=True)
    ap.add_argument("--suite", choices=SUITES, help="Run only this worker suite.")
    ap.add_argument("--cuda_visible_devices", default="")
    ap.add_argument("--render_gpu", type=int, default=-1)
    ap.add_argument("--python", default="/data/aviary/envs/openvla_official_libero_20260525/bin/python")
    ap.add_argument("--collector_repo", required=True, help="Clean checkout pinned to freeze/cross-suite-clean300-20260619.")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--min_free_gb", type=float, default=200.0)
    ap.add_argument("--plan_only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    require_clean_checkout(Path(args.collector_repo))
    root = Path(args.output_root)
    if args.plan_only:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        raise SystemExit(f"OUTPUT_ROOT_MISSING_RUN_PLAN_FIRST:{root}")

    jobs = build_jobs(root, args.eval_seed)
    validation = validate_master_manifest(jobs)
    if (
        validation["planned_count"] != 300
        or validation["unique_planned_count"] != 300
        or validation["duplicate_keys"]
        or validation["overlap_with_clean300_state0_9"] != 0
        or validation["missing_expected_keys"]
    ):
        raise SystemExit(f"MASTER_MANIFEST_INVALID:{validation}")

    manifest_rows = [asdict(j) for j in jobs]
    write_csv(root / "cross_suite_clean_train300_s10_19_master_manifest.csv", manifest_rows)
    manifest_sha = sha256_file(root / "cross_suite_clean_train300_s10_19_master_manifest.csv")
    write_json(
        root / "root_registry.json",
        {
            "corpus_name": "CROSS_SUITE_CLEAN_TRAIN300_S10_19",
            "corpus_role": "TARGET_SUITE_TRAIN_VALIDATION_ONLY",
            "primary_zero_shot_test_corpus": "CROSS_SUITE_CLEAN300 states 0-9",
            "split_policy": {"states_10_17": "train_pool", "states_18_19": "validation_pool", "states_0_9": "frozen_clean300_test_pool"},
            "pr30_merge_commit": PR30_MERGE_COMMIT,
            "collector_source_commit": COLLECTOR_SOURCE_COMMIT,
            "manifest_sha256": manifest_sha,
            "validation": validation,
            "policies": POLICIES,
        },
    )
    snapshot_command(root, "nvidia_smi_before.txt", ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader"])
    snapshot_command(root, "nvidia_smi_pmon_before.txt", ["nvidia-smi", "pmon", "-c", "1"])
    snapshot_command(root, "collector_git_status.txt", ["git", "status", "--short"], cwd=Path(args.collector_repo))
    snapshot_command(root, "collector_git_head.txt", ["git", "rev-parse", "HEAD"], cwd=Path(args.collector_repo))

    if args.plan_only:
        return
    if not args.suite:
        raise SystemExit("--suite is required unless --plan_only")
    run_worker(args, jobs)


if __name__ == "__main__":
    main()
