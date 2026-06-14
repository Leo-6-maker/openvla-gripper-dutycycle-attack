#!/usr/bin/env python3
"""D2: Fresh clean rollout launcher — parallel GPU workers for missing task-state groups.
Clean inference only. NO attack. NO training.
"""

import argparse, csv, os, subprocess, sys, time, json
from datetime import datetime
from pathlib import Path

REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
RUNNER = f"{REPO}/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py"
MODEL = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
GPUS = [0, 1, 2, 4, 5, 6, 7]  # 3 excluded
RENDER_GPU = "7"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index for this worker")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.manifest, newline="") as f:
        all_jobs = list(csv.DictReader(f))

    # Assign to this GPU (round-robin)
    jobs = [j for i, j in enumerate(all_jobs) if i % len(GPUS) == GPUS.index(args.gpu)]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {args.gpu}: {len(jobs)} jobs assigned")

    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OPENVLA_ATTN_IMPLEMENTATION": "eager",
           "CUDA_VISIBLE_DEVICES": str(args.gpu), "DISPLAY": ""}

    done = 0; failed = 0
    for i, job in enumerate(jobs):
        tag = f"{job['task_key']}_s{job['state_id']}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU{args.gpu} [{i+1}/{len(jobs)}] CLEAN {tag}")

        t0 = time.time()
        cmd = [PYTHON, "-u", RUNNER, "--task", job["task_key"], "--state_ids", job["state_id"],
               "--condition", "clean", "--window_start", "0", "--window_end", "10",
               "--max_steps_override", "280", "--success_metric", "check_success",
               "--num_steps_wait", "10", "--model_path", MODEL,
               "--render_gpu_device_id", RENDER_GPU, "--model_gpu_device_id", str(args.gpu),
               "--output_dir", args.output_dir, "--job_id", job["job_id"], "--seed", job["seed"]]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

        dt = time.time() - t0
        status = "done" if r.returncode == 0 else "failed"
        if status == "done": done += 1
        else: failed += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   {status} ({dt:.0f}s)")

        # Update manifest
        job["status"] = status
        job["runtime_sec"] = str(round(dt, 1))
        log_path = Path(args.output_dir) / f"worker_gpu{args.gpu}_progress.csv"
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
            w.writeheader(); w.writerows(jobs)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU{args.gpu} DONE: {done}/{len(jobs)} ({failed} failed)")


if __name__ == "__main__":
    main()
