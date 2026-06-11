#!/usr/bin/env python3
"""S20H multiseed confirmation worker: RAND → check → VIS (conditional)."""
import csv, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime

QUEUE = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else '1,0'
RENDER = sys.argv[3] if len(sys.argv) > 3 else '1'
OUT = sys.argv[4] if len(sys.argv) > 4 else '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612'

PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
os.makedirs(OUT, exist_ok=True)

ENV = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
       'OPENVLA_ATTN_IMPLEMENTATION': 'eager', 'CUDA_VISIBLE_DEVICES': GPU, 'DISPLAY': ''}

def run_job(job):
    tag = '%s_s%s_w%s_%s_%s_seed%s' % (job['task'], job['state_id'],
        job['window_start'], job['window_end'], job['condition'], job['attack_seed'])
    vid = os.path.join(OUT, 'videos', tag)
    cmd = [PY, '-u', RUNNER, '--task', job['task'], '--state_ids', job['state_id'],
           '--condition', job['condition'], '--window_start', job['window_start'],
           '--window_end', job['window_end'], '--max_steps_override', '280',
           '--success_metric', 'check_success', '--num_steps_wait', '10',
           '--model_path', MODEL, '--render_gpu_device_id', RENDER,
           '--model_gpu_device_id', '-1', '--output_dir', OUT,
           '--save_video_dir', vid, '--job_id', job['job_id'], '--seed', '0']
    if job['condition'] in ('random_linf', 'vis_pgd'):
        cmd += ['--eps_raw_pixels', '6', '--attack_seed', job['attack_seed']]
    if job['condition'] == 'random_linf':
        cmd += ['--random_control_seed', job['random_control_seed']]
    if job['condition'] == 'vis_pgd':
        cmd += ['--pgd_steps', '20']
    return subprocess.run(cmd, env=ENV)

def check_rand_pass(job):
    """Check if RAND summary shows pass (success, open<=5, no timeout)."""
    summary_files = list(Path(OUT).glob('summary_*_random_linf_seed%s_job%s.json' % (job['attack_seed'], job['job_id'])))
    if not summary_files:
        return False
    s = json.load(open(summary_files[0]))
    return (s['success_done_any'] and not s.get('timeout', False) and
            s['decoded_open_count'] <= 5 and s['max_open_streak'] <= 5)

def main():
    with open(QUEUE, newline='') as f:
        jobs = list(csv.DictReader(f))

    print('[%s] S20H worker: %s (%d jobs, GPU %s)' % (
        datetime.now().strftime('%H:%M:%S'), QUEUE, len(jobs), GPU))

    for i, job in enumerate(jobs):
        if job.get('status') == 'done': continue
        cond = job['condition']
        tag = '%s_s%s_w%s_%s_seed%s' % (job['task'], job['state_id'],
              job['window_start'], job['condition'], job['attack_seed'])

        if cond == 'random_linf':
            print('[%s] RAND %d/%d: %s' % (datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag))
            t0 = time.time()
            r = run_job(job)
            dt = time.time() - t0
            job['status'] = 'done' if r.returncode == 0 else 'failed'
            job['runtime_sec'] = str(round(dt, 1))
            job['rand_pass'] = str(check_rand_pass(job))
            print('[%s]   %s (%.0fs) rand_pass=%s' % (
                datetime.now().strftime('%H:%M:%S'),
                'PASS' if r.returncode == 0 else 'FAIL', dt, job['rand_pass']))

        elif cond == 'vis_pgd':
            # Check if paired RAND passed
            rand_job = [j for j in jobs if j['candidate_id'] == job['candidate_id'] and j['condition'] == 'random_linf' and j['attack_seed'] == job['attack_seed']]
            skip = False
            if rand_job and rand_job[0].get('rand_pass') == 'False':
                print('[%s] SKIP VIS %d/%d: %s (RAND not passed)' % (
                    datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag))
                job['status'] = 'skipped_rand_confounded'
                skip = True

            if not skip:
                print('[%s] VIS  %d/%d: %s' % (datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag))
                t0 = time.time()
                r = run_job(job)
                dt = time.time() - t0
                job['status'] = 'done' if r.returncode == 0 else 'failed'
                job['runtime_sec'] = str(round(dt, 1))
                print('[%s]   %s (%.0fs)' % (
                    datetime.now().strftime('%H:%M:%S'),
                    'PASS' if r.returncode == 0 else 'FAIL', dt))

        # Save progress
        with open(QUEUE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
            w.writeheader(); w.writerows(jobs)

    done = sum(1 for j in jobs if j.get('status') in ('done', 'skipped_rand_confounded'))
    print('[%s] DONE: %d/%d completed' % (datetime.now().strftime('%H:%M:%S'), done, len(jobs)))

if __name__ == '__main__':
    main()
