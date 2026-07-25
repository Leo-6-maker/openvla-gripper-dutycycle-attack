#!/usr/bin/env python3
"""Launch and aggregate 16 fail-closed FEC smoke workers from a frozen identity manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GPU_LAYOUT = {2: "libero_10", 3: "libero_goal", 6: "libero_object", 7: "libero_spatial"}
WORKERS_PER_GPU = 4
TOTAL_WORKERS = 16


class LaunchError(RuntimeError):
    pass


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity-manifest", type=Path, required=True)
    p.add_argument("--output-base", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/fec_attack_v3.yaml"))
    p.add_argument("--n4-module", type=Path, required=True)
    p.add_argument("--n4-provider-name", default=None)
    p.add_argument("--n4-norm-data", type=Path, required=True)
    p.add_argument("--expected-attacker-sha256", required=True)
    p.add_argument("--wave-stagger-seconds", type=float, default=20.0)
    return p.parse_args()


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != TOTAL_WORKERS:
        raise LaunchError(f"identity manifest must contain {TOTAL_WORKERS} workers, got {len(rows)}")
    ids = [int(r["worker_id"]) for r in rows]
    if len(set(ids)) != TOTAL_WORKERS:
        raise LaunchError("worker_id values must be unique")
    counts = Counter(int(r["gpu_id"]) for r in rows)
    if counts != Counter({gpu: WORKERS_PER_GPU for gpu in GPU_LAYOUT}):
        raise LaunchError(f"GPU counts mismatch: {counts}")
    for row in rows:
        gpu = int(row["gpu_id"])
        suite = str(row["suite"])
        if gpu not in GPU_LAYOUT or GPU_LAYOUT[gpu] != suite:
            raise LaunchError(f"worker {row['worker_id']} has invalid GPU/suite mapping: {gpu}/{suite}")
        if "task_index" not in row or "model_path" not in row:
            raise LaunchError(f"worker {row['worker_id']} missing task_index/model_path")
        if ("state_index" in row) == ("init_state_npy" in row):
            raise LaunchError(f"worker {row['worker_id']} must provide exactly one of state_index/init_state_npy")
        for key in ("seed", "rand_direction_seed", "random_time_seed"):
            if key not in row:
                raise LaunchError(f"worker {row['worker_id']} missing {key}")
        row["slot"] = int(row.get("slot", 0))
    per_gpu = defaultdict(list)
    for row in rows:
        per_gpu[int(row["gpu_id"])].append(row)
    for gpu, gpu_rows in per_gpu.items():
        gpu_rows.sort(key=lambda r: (int(r.get("slot", 0)), int(r["worker_id"])))
        for slot, row in enumerate(gpu_rows):
            row["slot"] = slot
    return sorted(rows, key=lambda r: (int(r["slot"]), int(r["gpu_id"])))


def worker_command(args: argparse.Namespace, row: dict[str, Any], run_root: Path) -> list[str]:
    script = args.repo_root / "scripts/fec/run_gpu_smoke.py"
    cmd = [
        str(args.python), str(script),
        "--gpu-id", str(row["gpu_id"]),
        "--suite", str(row["suite"]),
        "--task-index", str(row["task_index"]),
        "--output-root", str(run_root),
        "--model-path", str(row["model_path"]),
        "--config", str(args.config),
        "--repo-root", str(args.repo_root),
        "--n4-module", str(args.n4_module),
        "--n4-norm-data", str(args.n4_norm_data),
        "--expected-attacker-sha256", args.expected_attacker_sha256,
        "--seed", str(row["seed"]),
        "--rand-direction-seed", str(row["rand_direction_seed"]),
        "--random-time-seed", str(row["random_time_seed"]),
    ]
    if args.n4_provider_name:
        cmd += ["--n4-provider-name", args.n4_provider_name]
    if "state_index" in row:
        cmd += ["--state-index", str(row["state_index"])]
    else:
        cmd += ["--init-state-npy", str(row["init_state_npy"])]
    if row.get("random_time_start") is not None:
        cmd += ["--random-time-start", str(row["random_time_start"])]
    return cmd


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    payload = load_json(args.identity_manifest)
    rows = payload.get("workers", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise LaunchError("identity manifest must be a list or a mapping with workers")
    rows = validate_rows([dict(r) for r in rows])
    if args.output_base.exists() and any(args.output_base.iterdir()):
        raise LaunchError(f"output base is not empty: {args.output_base}")
    args.output_base.mkdir(parents=True, exist_ok=True)

    processes = []
    launch_rows = []
    for slot in range(WORKERS_PER_GPU):
        for row in [r for r in rows if int(r["slot"]) == slot]:
            worker_id = int(row["worker_id"])
            worker_root = args.output_base / f"gpu_{int(row['gpu_id'])}" / f"worker_{worker_id:02d}"
            worker_root.mkdir(parents=True, exist_ok=False)
            run_root = worker_root / "run"
            log_path = worker_root / "worker.log"
            cmd = worker_command(args, row, run_root)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(row["gpu_id"])
            env["MUJOCO_GL"] = "egl"
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((row, proc, log, run_root, log_path, cmd))
            launch_rows.append({**row, "pid": proc.pid, "run_root": str(run_root), "log": str(log_path), "cmd": cmd})
            print(f"[launch] worker={worker_id:02d} gpu={row['gpu_id']} suite={row['suite']} pid={proc.pid}", flush=True)
        if slot < WORKERS_PER_GPU - 1:
            time.sleep(max(args.wave_stagger_seconds, 0.0))

    atomic_json(args.output_base / "launch_manifest.json", {
        "scientific_role": "SMOKE_ONLY",
        "formal_matrix_execution": False,
        "cs200_access": False,
        "workers": launch_rows,
        "launched_unix": time.time(),
    })

    completed = []
    for row, proc, log, run_root, log_path, cmd in processes:
        exit_code = proc.wait()
        log.close()
        summary_path = run_root / "smoke_summary.json"
        summary = load_json(summary_path) if summary_path.is_file() else None
        completed.append({
            "worker_id": int(row["worker_id"]),
            "gpu_id": int(row["gpu_id"]),
            "suite": row["suite"],
            "exit_code": exit_code,
            "summary_path": str(summary_path),
            "summary": summary,
        })
        print(f"[done] worker={int(row['worker_id']):02d} exit={exit_code} valid={bool(summary and summary.get('valid'))}", flush=True)

    worker_pass = all(x["exit_code"] == 0 and x["summary"] and x["summary"].get("valid") for x in completed)
    true_exec = sum(int(x["summary"]["results"]["TRUE_T10"]["attack_executed_frames"]) for x in completed if x["summary"])
    rand_exec = sum(int(x["summary"]["results"]["RAND_T10"]["attack_executed_frames"]) for x in completed if x["summary"])
    oracle_exec = sum(int(x["summary"]["results"]["COMMAND_OPEN_ORACLE"]["attack_executed_frames"]) for x in completed if x["summary"])
    random_exec = sum(int(x["summary"]["results"]["RANDOM_TIME_T10"]["attack_executed_frames"]) for x in completed if x["summary"])
    natural_attack_coverage = true_exec >= 10 and rand_exec >= 10 and oracle_exec >= 10
    random_time_coverage = random_exec >= 10
    valid = worker_pass and natural_attack_coverage and random_time_coverage
    receipt = {
        "status": "PASS_AT_16" if valid else "FAIL",
        "valid": valid,
        "worker_pass": worker_pass,
        "natural_attack_coverage": natural_attack_coverage,
        "random_time_coverage": random_time_coverage,
        "attack_frames": {
            "TRUE_T10": true_exec,
            "RAND_T10": rand_exec,
            "COMMAND_OPEN_ORACLE": oracle_exec,
            "RANDOM_TIME_T10": random_exec,
        },
        "completed": completed,
        "formal_matrix_execution": False,
        "cs200_access": False,
        "completed_unix": time.time(),
    }
    atomic_json(args.output_base / "FEC_GPU_SMOKE_RECEIPT_V2.json", receipt)
    if valid:
        atomic_json(args.output_base / "FEC_GPU_SMOKE_PASS_AT_16.json", {"status": "PASS_AT_16"})
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
