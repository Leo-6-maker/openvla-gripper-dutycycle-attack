#!/usr/bin/env python3
"""S20f GPU worker: processes a job queue sequentially, marks completion.
Each job: {task, state_id, window_start, window_end, condition, attack_seed, random_control_seed, job_id}
Reads queue CSV, runs jobs in order, writes progress after each job."""
import csv, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime

QUEUE = sys.argv[1] if len(sys.argv) > 1 else '/data/liuyu/outputs/stageb_s20f_queues_20260611/queue_gpu10.csv'
GPU_PAIR = sys.argv[2] if len(sys.argv) > 2 else '1,0'
RENDER_ID = sys.argv[3] if len(sys.argv) > 3 else '0'
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else '/data/liuyu/outputs/stageb_s20f_queues_20260611/output'
PROGRESS = QUEUE.replace('.csv', '_progress.json')

PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'

os.makedirs(OUT_DIR, exist_ok=True)

def load_queue():
    if not Path(QUEUE).exists():
        return []
    with open(QUEUE, newline='') as f:
        return list(csv.DictReader(f))

def save_progress(jobs):
    done = [j for j in jobs if j.get('status') == 'done']
    with open(PROGRESS, 'w') as f:
        json.dump({'updated': datetime.now().isoformat(), 'total': len(jobs),
                   'done': len(done), 'done_ids': [j['job_id'] for j in done]}, f, indent=2)

def run_job(job):
    vid_dir = os.path.join(OUT_DIR, 'videos', 'job%s_%s_s%s_w%s_%s_%s_seed%s' % (
        job['job_id'], job['task'], job['state_id'],
        job['window_start'], job['window_end'],
        job['condition'], job.get('attack_seed', job.get('seed', '0'))))
    cmd = [PY, '-u', RUNNER,
           '--task', job['task'], '--state_ids', job['state_id'],
           '--condition', job['condition'],
           '--window_start', job['window_start'], '--window_end', job['window_end'],
           '--max_steps_override', '280', '--success_metric', 'done',
           '--num_steps_wait', '10',
           '--model_path', MODEL,
           '--render_gpu_device_id', RENDER_ID, '--model_gpu_device_id', '-1',
           '--output_dir', OUT_DIR,
           '--save_video_dir', vid_dir,
           '--job_id', job['job_id'],
           '--seed', job.get('seed', '0')]
    if job['condition'] in ('random_linf', 'vis_pgd'):
        cmd += ['--eps_raw_pixels', '6',
                '--attack_seed', job.get('attack_seed', '80')]
    if job['condition'] == 'random_linf':
        cmd += ['--random_control_seed', job.get('random_control_seed', job.get('attack_seed', '80'))]
    if job['condition'] == 'vis_pgd':
        cmd += ['--pgd_steps', '20']
    return subprocess.run(cmd, env={**os.environ,
        'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
        'OPENVLA_ATTN_IMPLEMENTATION': 'eager',
        'CUDA_VISIBLE_DEVICES': GPU_PAIR})

def main():
    env = os.environ.copy()
    env.update({'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                'OPENVLA_ATTN_IMPLEMENTATION': 'eager',
                'CUDA_VISIBLE_DEVICES': GPU_PAIR, 'DISPLAY': ''})
    os.environ.update(env)

    jobs = load_queue()
    if not jobs:
        print('[%s] Empty queue: %s' % (datetime.now().strftime('%H:%M:%S'), QUEUE))
        return

    print('[%s] GPU worker started: %s (%d jobs, GPU %s render=%s)' % (
        datetime.now().strftime('%H:%M:%S'), QUEUE, len(jobs), GPU_PAIR, RENDER_ID))
    print('[%s] Output: %s' % (datetime.now().strftime('%H:%M:%S'), OUT_DIR))

    for i, job in enumerate(jobs):
        if job.get('status') == 'done':
            continue
        tag = '%s_s%s_w%s_%s_%s' % (job['task'], job['state_id'],
                                      job['window_start'], job['window_end'], job['condition'])
        print('[%s] Job %d/%d: %s (job=%s)' % (
            datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag, job['job_id']))

        t0 = time.time()
        result = run_job(job)
        dt = time.time() - t0

        job['status'] = 'done' if result.returncode == 0 else 'failed'
        job['runtime_sec'] = round(dt, 1)
        job['exit_code'] = result.returncode

        # Write back to queue
        with open(QUEUE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
            w.writeheader(); w.writerows(jobs)
        save_progress(jobs)

        status = 'PASS' if result.returncode == 0 else 'FAIL (code=%d)' % result.returncode
        print('[%s]   %s (%.0fs)' % (datetime.now().strftime('%H:%M:%S'), status, dt))

    done = sum(1 for j in jobs if j.get('status') == 'done')
    print('[%s] GPU worker DONE: %d/%d jobs completed' % (
        datetime.now().strftime('%H:%M:%S'), done, len(jobs)))

if __name__ == '__main__':
    main()
