#!/usr/bin/env python3
"""D2: Fresh clean rollout — task-batched subprocess launcher.
Uses comma-separated state_ids per task (1 model load per task).
CUDA_LAUNCH_BLOCKING=1 to prevent async memory errors.
"""

import argparse, csv, os, subprocess, sys, time, glob
from collections import defaultdict
from datetime import datetime

REPO = "/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607"
RUNNER = f"{REPO}/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py"
MODEL = "/data/aviary/models/openvla/openvla-7b-finetuned-libero-object"
PYTHON = "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python"
GPU_VISIBLE = "0,1,2,4,5,6,7"
RENDER_GPU = "0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.manifest, newline="") as f:
        all_jobs = list(csv.DictReader(f))

    by_task = defaultdict(list)
    for j in all_jobs:
        by_task[j["task_key"]].append(j["state_id"])

    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "OPENVLA_ATTN_IMPLEMENTATION": "eager",
           "CUDA_VISIBLE_DEVICES": GPU_VISIBLE, "DISPLAY": "",
           "CUDA_LAUNCH_BLOCKING": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    total = len(all_jobs); completed = 0; failed = 0; job_id = 500000

    print(f"[{datetime.now().strftime('%H:%M:%S')}] D2 rollout: {total} states, {len(by_task)} tasks")
    print(f"GPUs: {GPU_VISIBLE}")

    for task in sorted(by_task):
        states = sorted(by_task[task], key=int)
        n = len(states)
        states_str = ",".join(states)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] TASK {task}: {n} states ({states_str[:50]}...)")

        t0 = time.time()
        cmd = [PYTHON, "-u", RUNNER, "--task", task, "--state_ids", states_str,
               "--condition", "clean", "--window_start", "0", "--window_end", "10",
               "--max_steps_override", "280", "--success_metric", "check_success",
               "--num_steps_wait", "10", "--model_path", MODEL,
               "--render_gpu_device_id", RENDER_GPU, "--model_gpu_device_id", "-1",
               "--output_dir", args.output_dir, "--job_id", str(job_id), "--seed", "0"]

        try:
            r = subprocess.run(cmd, env=env, timeout=5400, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 90min")
            failed += n; job_id += n; continue

        dt = time.time() - t0
        n_traces = len(glob.glob(os.path.join(args.output_dir, f"trace_{task}_s*_job{job_id}*.csv")))
        n_failed = n - n_traces
        completed += n_traces; failed += n_failed
        status = "OK" if n_failed == 0 else f"PARTIAL ({n_failed}/{n} missing)"

        print(f"[{datetime.now().strftime('%H:%M:%S')}]   {status} ({dt:.0f}s, {n_traces} traces) [{completed}/{total} total]")
        if r.returncode != 0:
            print(f"  stderr: {r.stderr[-200:] if r.stderr else 'none'}")

        job_id += n

    print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE: {completed}/{total} ({failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
