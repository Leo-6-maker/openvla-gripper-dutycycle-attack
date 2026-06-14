#!/usr/bin/env python3
"""D2: Fresh clean rollout — sequential multi-GPU worker.
OpenVLA-7B needs ~16-22GB. Uses device_map=auto across all visible GPUs.
Clean inference only. NO attack.
"""

import argparse, csv, os, subprocess, sys, time
from datetime import datetime

REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
RUNNER = f"{REPO}/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py"
MODEL = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
GPU_VISIBLE = "0,1,2,4,5,6,7"  # 3 excluded
RENDER_GPU = "0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.manifest, newline="") as f:
        all_jobs = list(csv.DictReader(f))

    jobs = all_jobs[args.start:args.end] if args.end > 0 else all_jobs
    print(f"[{datetime.now().strftime('%H:%M:%S')}] D2 clean rollout: {len(jobs)} jobs, GPUs: {GPU_VISIBLE}")

    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OPENVLA_ATTN_IMPLEMENTATION": "eager",
           "CUDA_VISIBLE_DEVICES": GPU_VISIBLE, "DISPLAY": ""}

    done = 0; failed = 0
    for i, job in enumerate(jobs):
        tag = f"{job['task_key']}_s{job['state_id']}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{i+1}/{len(jobs)}] CLEAN {tag}")

        t0 = time.time()
        cmd = [PYTHON, "-u", RUNNER, "--task", job["task_key"], "--state_ids", job["state_id"],
               "--condition", "clean", "--window_start", "0", "--window_end", "10",
               "--max_steps_override", "280", "--success_metric", "check_success",
               "--num_steps_wait", "10", "--model_path", MODEL,
               "--render_gpu_device_id", RENDER_GPU, "--model_gpu_device_id", "-1",
               "--output_dir", args.output_dir, "--job_id", job["job_id"], "--seed", job["seed"]]
        r = subprocess.run(cmd, env=env, timeout=600)

        dt = time.time() - t0
        status = "done" if r.returncode == 0 else "failed"
        if status == "done": done += 1
        else: failed += 1
        job["status"] = status
        job["runtime_sec"] = str(round(dt, 1))
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   {status} ({dt:.0f}s) [{done}/{i+1} ok]")

        # Save progress
        with open(args.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_jobs[0].keys()))
            w.writeheader(); w.writerows(all_jobs)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE: {done}/{len(jobs)} ({failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
