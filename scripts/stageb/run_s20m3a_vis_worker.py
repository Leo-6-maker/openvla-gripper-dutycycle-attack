#!/usr/bin/env python3
"""S20M3a VIS-fill worker: check matched RAND gate from S20M2, run VIS only if gate passes."""
import csv, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime

QUEUE = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else '0,1'
RENDER = sys.argv[3] if len(sys.argv) > 3 else '0'
OUT = sys.argv[4] if len(sys.argv) > 4 else '/data/liuyu/outputs/stageb_s20m3a_vis_fill_20260613'
RAND_REF = '/data/liuyu/outputs/stageb_s20m2_frozen_forward_20260613'

PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
os.makedirs(OUT, exist_ok=True)

ENV = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
       'OPENVLA_ATTN_IMPLEMENTATION': 'eager', 'CUDA_VISIBLE_DEVICES': GPU, 'DISPLAY': ''}

def find_matched_rand(job):
    """Find matched RAND summary from S20M2. Uses candidate_id (task_sid_ws_we) to match."""
    cid = job['candidate_id']  # e.g. alphabet_soup_s2_w110_120
    rand_seed = job.get('rand_seed', '92')
    pattern = 'summary_{}_s20d_random_linf_seed{}_job*.json'.format(cid, rand_seed)
    matches = list(Path(RAND_REF).glob(pattern))
    if matches:
        return json.load(open(matches[0]))
    # Try unified glob
    for f in Path(RAND_REF).glob('summary_*.json'):
        s = json.load(open(f))
        key = '{task}_s{sid}_w{ws}_{we}'.format(
            task=s['task'], sid=s['state_id'],
            ws=s['window_start'], we=s['window_end'])
        if key == cid and str(s.get('attack_seed','')) == rand_seed:
            return s
    return None

def check_rand_gate(rand_summary):
    """Gate: RAND must be STRICT or USABLE (clean), no timeout, success done."""
    if rand_summary is None:
        return False, 'missing_rand'
    o = rand_summary['decoded_open_count']
    st = rand_summary['max_open_streak']
    d = rand_summary['success_done_any']
    to = rand_summary.get('timeout', False)
    if to or not d:
        return False, 'rand_not_clean'
    if o > 5 or st > 5:
        return False, 'rand_borderline'
    return True, 'rand_{}'.format('STRICT' if o<=3 and st<=3 else 'USABLE')

def run_vis(job):
    tag = '%s_s%s_w%s_%s_%s_seed%s' % (job['task'], job['state_id'],
        job['window_start'], job['window_end'], job['condition'], job['attack_seed'])
    vid = os.path.join(OUT, 'videos', tag)
    cmd = [PY, '-u', RUNNER, '--task', job['task'], '--state_ids', job['state_id'],
           '--condition', 'vis_pgd', '--window_start', job['window_start'],
           '--window_end', job['window_end'], '--max_steps_override', '280',
           '--success_metric', 'check_success', '--num_steps_wait', '10',
           '--model_path', MODEL, '--render_gpu_device_id', RENDER,
           '--model_gpu_device_id', '-1', '--output_dir', OUT,
           '--save_video_dir', vid, '--job_id', job['job_id'], '--seed', '0',
           '--eps_raw_pixels', '6', '--attack_seed', job['attack_seed'],
           '--pgd_steps', '20']
    return subprocess.run(cmd, env=ENV)

def main():
    with open(QUEUE, newline='') as f:
        jobs = list(csv.DictReader(f))

    print('[%s] S20M3a VIS worker: %s (%d jobs, GPU %s)' % (
        datetime.now().strftime('%H:%M:%S'), QUEUE, len(jobs), GPU))
    print('[%s] RAND reference: %s' % (datetime.now().strftime('%H:%M:%S'), RAND_REF))

    for i, job in enumerate(jobs):
        if job.get('status') in ('done', 'skipped_rand_confounded', 'skipped_no_rand'):
            continue

        tag = '%s_s%s_w%s_%s_seed%s' % (job['task'], job['state_id'],
              job['window_start'], job['condition'], job['attack_seed'])

        # Gate check
        rand_s = find_matched_rand(job)
        gate_ok, gate_reason = check_rand_gate(rand_s)

        if not gate_ok:
            print('[%s] SKIP VIS %d/%d: %s (gate=%s)' % (
                datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag, gate_reason))
            job['status'] = 'skipped_' + gate_reason
            job['gate_ok'] = 'False'
            job['gate_reason'] = gate_reason
        else:
            print('[%s] VIS  %d/%d: %s (gate=%s rand_open=%d rand_streak=%d)' % (
                datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag,
                gate_reason, rand_s['decoded_open_count'], rand_s['max_open_streak']))
            t0 = time.time()
            r = run_vis(job)
            dt = time.time() - t0
            job['status'] = 'done' if r.returncode == 0 else 'failed'
            job['runtime_sec'] = str(round(dt, 1))
            job['gate_ok'] = 'True'
            job['gate_reason'] = gate_reason
            print('[%s]   %s (%.0fs)' % (
                datetime.now().strftime('%H:%M:%S'),
                'PASS' if r.returncode == 0 else 'FAIL', dt))

        # Save progress
        with open(QUEUE, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
            w.writeheader(); w.writerows(jobs)

    done = sum(1 for j in jobs if j.get('status') in ('done', 'skipped_rand_confounded', 'skipped_no_rand', 'skipped_rand_not_clean', 'skipped_rand_borderline', 'skipped_missing_rand'))
    failed = sum(1 for j in jobs if j.get('status') == 'failed')
    print('[%s] DONE: %d/%d completed (%d failed)' % (
        datetime.now().strftime('%H:%M:%S'), done, len(jobs), failed))

if __name__ == '__main__':
    main()
