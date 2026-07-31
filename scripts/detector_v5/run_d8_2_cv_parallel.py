"""D8-2 CV: Parallel launcher — distributes 25 training units across GPUs.

Each (config, fold) is an independent job. GPU assignment via CUDA_VISIBLE_DEVICES.
Launches all jobs concurrently, waits for completion, then aggregates results.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FOLDS = [0, 1, 2, 3, 4]
CONFIGS = ["B0", "B1", "B2", "B3", "B4"]
SEED = 20260717
EPOCHS = 20
GPU_IDS = [0, 1, 2, 3, 6, 7]  # available GPUs (4,5 in use)


def launch_unit(config: str, fold: int, gpu: int, cache_root: str, output_root: str, python: str, epochs: int) -> subprocess.Popen:
    unit_dir = Path(output_root) / f"{config}_fold{fold}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    log_path = unit_dir / "train.log"

    cmd = [
        python, "-u", str(ROOT / "scripts" / "detector_v5" / "run_d8_2_cv_unit.py"),
        "--cache-root", cache_root,
        "--config", config,
        "--fold", str(fold),
        "--seed", str(SEED),
        "--epochs", str(epochs),
        "--output-dir", str(unit_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{ROOT / 'scripts' / 'detector_v5'}:{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"  [{config} fold {fold}] GPU {gpu} PID {proc.pid} -> {unit_dir}")
    return proc, log_fh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,6,7")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    cache_root = str(args.cache_root.resolve())
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(str(output_root))
    output_root.mkdir(parents=True)

    # Build job list: (config, fold, gpu_index)
    jobs = []
    for i, config in enumerate(CONFIGS):
        for j, fold in enumerate(FOLDS):
            gpu = gpu_ids[(i * len(FOLDS) + j) % len(gpu_ids)]
            jobs.append((config, fold, gpu))

    print(f"Launching {len(jobs)} jobs on GPUs {gpu_ids}")
    for config, fold, gpu in jobs:
        print(f"  {config} fold {fold} -> GPU {gpu}")

    procs = []
    for config, fold, gpu in jobs:
        proc, fh = launch_unit(config, fold, gpu, cache_root, str(output_root), args.python, args.epochs)
        procs.append((proc, fh, config, fold))
        time.sleep(1)  # stagger starts

    # Wait for all
    print(f"\nWaiting for {len(procs)} jobs...")
    results = {}
    for proc, fh, config, fold in procs:
        proc.wait()
        fh.close()
        unit_dir = output_root / f"{config}_fold{fold}"
        metrics_path = unit_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            results[f"{config}_fold{fold}"] = m
            status = "OK" if m.get("balanced_accuracy", 0) > 0 else "WARN"
        else:
            status = f"FAIL (code {proc.returncode})"
            results[f"{config}_fold{fold}"] = {"status": status}
        print(f"  [{config} fold {fold}] {status}")

    # Aggregate
    summary = {}
    for config in CONFIGS:
        fm = [v for k, v in results.items() if k.startswith(f"{config}_")]
        baccs = [m.get("balanced_accuracy", 0) for m in fm if "balanced_accuracy" in m]
        mccs = [m.get("mcc", 0) for m in fm if "mcc" in m]
        summary[config] = {
            "mean_bacc": float(np.mean(baccs)) if baccs else 0,
            "mean_mcc": float(np.mean(mccs)) if mccs else 0,
            "folds_completed": len([m for m in fm if "balanced_accuracy" in m]),
        }

    (output_root / "D8_2_CV_SUMMARY.json").write_text(json.dumps({
        "schema": "D8_2_CV_SUMMARY_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "folds": FOLDS, "configs": CONFIGS, "epochs": EPOCHS,
        "summary": summary,
    }, indent=2, sort_keys=True) + "\n")

    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
