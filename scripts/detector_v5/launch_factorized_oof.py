#!/usr/bin/env python3
"""Launch Factorized Student OOF training on GPU 1 and 3.

3 workers per GPU, cycling through fold×seed combinations.
Primary 25D9D runs first, then ablation 25D runs.
Each worker trains sequentially to avoid OOM.
"""
import subprocess, sys, time
from pathlib import Path

PY = "/home/sz/miniconda3/envs/hallo/bin/python"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_factorized_oof.py"
OPS = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops")
S1 = OPS / "OFFICIAL_V3_S1_FIT_V1_d31187f"
TEACHER = OPS / "OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721"
FOLDS = OPS / "OFFICIAL_V3_FIT_FOLDS_V1_d31187f"
OUT_BASE = OPS / "OFFICIAL_V3_FACTORIZED_STUDENT_OOF"

SEEDS = [42, 123, 456]
FOLDS = [0, 1, 2, 3]
GPUS = [1, 3]
MODEL_TYPES = ["25D9D", "25D"]

jobs = []
for mt in MODEL_TYPES:
    for fold in range(4):
        for seed in SEEDS:
            out = OUT_BASE / mt / f"fold{fold}_seed{seed}"
            jobs.append((mt, fold, seed, out))

print(f"Total jobs: {len(jobs)} ({len(MODEL_TYPES)} types × 4 folds × 3 seeds)")
print(f"GPUs: {GPUS}")

def launch_worker(gpu, job_list, worker_id):
    """Train jobs sequentially on one GPU."""
    env = {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTHONPATH": "/mnt/sdc/dty_user/openvla_attack/src",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }
    for mt, fold, seed, out in job_list:
        if out.exists():
            print(f"  [GPU{gpu} W{worker_id}] SKIP {mt} fold{fold} seed{seed} (exists)")
            continue
        print(f"  [GPU{gpu} W{worker_id}] START {mt} fold{fold} seed{seed}")
        start = time.time()
        r = subprocess.run(
            [PY, str(TRAIN_SCRIPT),
             "--model-type", mt, "--fold-id", str(fold), "--seed", str(seed),
             "--gpu", "0",  # CUDA_VISIBLE_DEVICES remaps
             "--output-root", str(out),
             "--s1-root", str(S1), "--teacher-root", str(TEACHER),
             "--fold-root", str(FOLDS)],
            env={**__import__("os").environ, **env},
            capture_output=True, text=True,
        )
        elapsed = time.time() - start
        if r.returncode == 0:
            print(f"  [GPU{gpu} W{worker_id}] DONE  {mt} fold{fold} seed{seed} ({elapsed:.0f}s)")
        else:
            print(f"  [GPU{gpu} W{worker_id}] FAIL  {mt} fold{fold} seed{seed}: {r.stderr[-200:]}")
            # Continue with next job — don't kill other workers

# Split jobs into worker queues
primary = [j for j in jobs if j[0] == "25D9D"]
ablation = [j for j in jobs if j[0] == "25D"]

# Round-robin: each GPU gets 3 workers, each worker gets fold×seed for primary then ablation
workers = []
for gpu in GPUS:
    for w in range(3):
        # Worker w on GPU gpu gets a subset of jobs
        worker_jobs = []
        for mt in MODEL_TYPES:
            job_pool = primary if mt == "25D9D" else ablation
            for i, job in enumerate(job_pool):
                if i % (len(GPUS) * 3) == (gpu - GPUS[0]) * 3 + w:
                    worker_jobs.append(job)
        if worker_jobs:
            workers.append((gpu, w, worker_jobs))

print(f"Workers: {len(workers)}")
for gpu, w, jobs_w in workers:
    print(f"  GPU{gpu} W{w}: {len(jobs_w)} jobs")

# Launch workers (staggered to avoid OOM)
import threading
threads = []
for i, (gpu, w, jobs_w) in enumerate(workers):
    t = threading.Thread(target=launch_worker, args=(gpu, jobs_w, w))
    threads.append(t)
    t.start()
    time.sleep(5)  # stagger launches to avoid simultaneous CUDA init

for t in threads:
    t.join()

print("ALL OOF RUNS COMPLETE")
