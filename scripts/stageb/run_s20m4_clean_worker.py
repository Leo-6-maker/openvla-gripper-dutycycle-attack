#!/usr/bin/env python3
"""S20M4 clean scan worker: run clean (no attack) rollouts, record summaries/traces."""
import csv, os, subprocess, sys, time
from datetime import datetime

QUEUE = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else '6,7'
RENDER = sys.argv[3] if len(sys.argv) > 3 else '6'
OUT = sys.argv[4] if len(sys.argv) > 4 else '/data/liuyu/outputs/stageb_s20m4_clean_scan_20260613'

PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
os.makedirs(OUT, exist_ok=True)

ENV = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
       'OPENVLA_ATTN_IMPLEMENTATION': 'eager', 'CUDA_VISIBLE_DEVICES': GPU, 'DISPLAY': ''}

def run_clean(job):
    tag = '%s_s%s_clean_seed%s' % (job['task'], job['state_id'], job['seed'])
    cmd = [PY, '-u', RUNNER, '--task', job['task'], '--state_ids', job['state_id'],
           '--condition', 'clean', '--window_start', '0', '--window_end', '10',
           '--max_steps_override', '280', '--success_metric', 'check_success',
           '--num_steps_wait', '10', '--model_path', MODEL,
           '--render_gpu_device_id', RENDER, '--model_gpu_device_id', '-1',
           '--output_dir', OUT, '--job_id', job['job_id'], '--seed', job['seed']]
    return subprocess.run(cmd, env=ENV)

def main():
    with open(QUEUE, newline='') as f:
        jobs = list(csv.DictReader(f))

    print('[%s] S20M4 clean worker: %s (%d jobs, GPU %s)' % (
        datetime.now().strftime('%H:%M:%S'), QUEUE, len(jobs), GPU))

    for i, job in enumerate(jobs):
        if job.get('status') in ('done', 'failed'):
            continue
        tag = '%s_s%s_clean' % (job['task'], job['state_id'])
        print('[%s] CLEAN %d/%d: %s' % (datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag))
        t0 = time.time()
        r = run_clean(job)
        dt = time.time() - t0
        job['status'] = 'done' if r.returncode == 0 else 'failed'
        job['runtime_sec'] = str(round(dt, 1))
        print('[%s]   %s (%.0fs)' % (datetime.now().strftime('%H:%M:%S'),
              'PASS' if r.returncode == 0 else 'FAIL', dt))

        with open(QUEUE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
            w.writeheader(); w.writerows(jobs)

    done = sum(1 for j in jobs if j.get('status') in ('done', 'failed'))
    failed = sum(1 for j in jobs if j.get('status') == 'failed')
    print('[%s] DONE: %d/%d (%d failed)' % (datetime.now().strftime('%H:%M:%S'), done, len(jobs), failed))

if __name__ == '__main__':
    main()
