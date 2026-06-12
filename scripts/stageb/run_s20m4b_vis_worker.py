#!/usr/bin/env python3
"""S20M4b VIS discovery worker: read S20M4a RAND baseline, run VIS seed99.
Discovery only — does NOT auto-launch confirmation seeds."""
import csv, json, os, subprocess, sys, time, numpy as np
from pathlib import Path
from datetime import datetime

QUEUE = sys.argv[1]
GPU = sys.argv[2] if len(sys.argv) > 2 else '4,5'
RENDER = sys.argv[3] if len(sys.argv) > 3 else '4'
OUT = sys.argv[4] if len(sys.argv) > 4 else '/data/liuyu/outputs/stageb_s20m4b_vis_discovery_20260613'
# Cross-directory baseline lookup across all stages
RAND_DIRS = [
    '/data/liuyu/outputs/stageb_s20m4_rand_stability_20260613',
    '/data/liuyu/outputs/stageb_s20m3b_multiseed_confirmation_20260613',
    '/data/liuyu/outputs/stageb_s20m3a_vis_fill_20260613',
    '/data/liuyu/outputs/stageb_s20m2_frozen_forward_20260613',
    '/data/liuyu/outputs/stageb_s20m1_randonly_calibration_20260613',
    '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
    '/data/liuyu/outputs/stageb_s20l_v2_randonly_20260613',
]

PY = '/data/aviary/envs/openvla_official_libero_20260525/bin/python'
RUNNER = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py'
MODEL = '/data/aviary/models/openvla/openvla-7b-finetuned-libero-object'
os.makedirs(OUT, exist_ok=True)

ENV = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
       'OPENVLA_ATTN_IMPLEMENTATION': 'eager', 'CUDA_VISIBLE_DEVICES': GPU, 'DISPLAY': ''}

def find_rand_baseline(job):
    """Load RAND summaries across all stage directories for this candidate."""
    cid = job['candidate_id']
    # Get preferred seeds from queue or use common set
    pref_seeds = job.get('baseline_seeds', '96|97|98').split('|')
    results = []
    for d in RAND_DIRS:
        if not os.path.exists(d): continue
        for f in Path(d).glob('summary_*.json'):
            try:
                s = json.load(open(f))
            except: continue
            if s.get('condition') != 'random_linf': continue
            key = '{task}_s{sid}_w{ws}_{we}'.format(
                task=s['task'], sid=s['state_id'],
                ws=s['window_start'], we=s['window_end'])
            if key == cid:
                seed = str(s.get('attack_seed',''))
                if pref_seeds and pref_seeds[0] and seed not in pref_seeds: continue
                # Deduplicate by seed
                if not any(str(r.get('attack_seed','')) == seed for r in results):
                    results.append(s)
    return results

def rand_median_open_streak(rand_summaries):
    """Compute median open/streak from RAND baseline."""
    if not rand_summaries:
        return 0, 0
    opens = [s['decoded_open_count'] for s in rand_summaries]
    streaks = [s['max_open_streak'] for s in rand_summaries]
    return int(np.median(opens)), int(np.median(streaks))

def run_vis(job):
    tag = '%s_s%s_w%s_%s_%s_seed%s' % (job['task'], job['state_id'],
        job['window_start'], job['window_end'], 'vis_pgd', job['attack_seed'])
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

def classify(r_med_open, r_med_streak, vis_open, vis_streak, vis_done, vis_timeout, infra_ok):
    """Classify VIS discovery result."""
    if not infra_ok or vis_timeout is None:
        return 'CONFOUNDED_OR_INFRA'
    open_gap = vis_open - r_med_open
    streak_gap = vis_streak - r_med_streak
    if open_gap >= 3 or streak_gap >= 3:
        return 'DISCOVERY_CMD_POSITIVE'
    return 'NO_EFFECT'

def main():
    with open(QUEUE, newline='') as f:
        jobs = list(csv.DictReader(f))

    print('[%s] S20M4b VIS worker: %s (%d jobs, GPU %s, seed99)' % (
        datetime.now().strftime('%H:%M:%S'), QUEUE, len(jobs), GPU))

    for i, job in enumerate(jobs):
        if job.get('status') in ('done','failed','skipped'):
            continue
        tag = '%s_s%s_w%s_vis_seed99' % (job['task'], job['state_id'], job['window_start'])
        priority = job.get('claim_priority','P0')

        # Load RAND baseline
        rand_baseline = find_rand_baseline(job)
        r_med_open, r_med_streak = rand_median_open_streak(rand_baseline)
        n_rand = len(rand_baseline)

        if n_rand < 1:
            print('[%s] SKIP %d/%d: %s (no RAND baseline)' % (
                datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag))
            job['status'] = 'skipped_no_rand'
            job['rand_median_open'] = ''
            job['rand_median_streak'] = ''
            continue

        print('[%s] VIS %d/%d: %s [%s] (RAND_median open=%d streak=%d n=%d)' % (
            datetime.now().strftime('%H:%M:%S'), i+1, len(jobs), tag, priority,
            r_med_open, r_med_streak, n_rand))
        t0 = time.time()
        r = run_vis(job)
        dt = time.time() - t0

        job['runtime_sec'] = str(round(dt, 1))
        job['rand_median_open'] = str(r_med_open)
        job['rand_median_streak'] = str(r_med_streak)
        job['rand_n_seeds'] = str(n_rand)

        if r.returncode != 0:
            job['status'] = 'failed'
            print('[%s]   FAIL (%.0fs)' % (datetime.now().strftime('%H:%M:%S'), dt))
        else:
            job['status'] = 'done'
            # Read summary to classify
            summary_files = list(Path(OUT).glob('summary_*vis_pgd_seed99_job%s.json' % job['job_id']))
            if summary_files:
                vs = json.load(open(summary_files[0]))
                vis_open = vs['decoded_open_count']
                vis_streak = vs['max_open_streak']
                vis_done = vs['success_done_any']
                vis_timeout = vs.get('timeout', False)
                vis_steps = vs['n_steps']
                infra_ok = vs.get('infra_status','') == 'ok'

                cls = classify(r_med_open, r_med_streak, vis_open, vis_streak,
                              vis_done, vis_timeout, infra_ok)
                open_gap = vis_open - r_med_open
                streak_gap = vis_streak - r_med_streak

                job['vis_open'] = str(vis_open)
                job['vis_streak'] = str(vis_streak)
                job['vis_done'] = str(vis_done)
                job['vis_timeout'] = str(vis_timeout)
                job['vis_steps'] = str(vis_steps)
                job['open_gap'] = str(open_gap)
                job['streak_gap'] = str(streak_gap)
                job['discovery_class'] = cls

                print('[%s]   DONE (%.0fs) VIS_open=%d streak=%d gap=%+d/%+d → %s' % (
                    datetime.now().strftime('%H:%M:%S'), dt,
                    vis_open, vis_streak, open_gap, streak_gap, cls))
            else:
                job['discovery_class'] = 'CONFOUNDED_OR_INFRA'
                print('[%s]   DONE (%.0fs) but no summary found' % (
                    datetime.now().strftime('%H:%M:%S'), dt))

        # Save progress
        with open(QUEUE, 'w', newline='') as f:
            fields = [k for k in jobs[0].keys()] + [
                k for k in job.keys() if k not in jobs[0].keys()]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(jobs)

    done = sum(1 for j in jobs if j.get('status') in ('done','failed','skipped'))
    positive = sum(1 for j in jobs if j.get('discovery_class') == 'DISCOVERY_CMD_POSITIVE')
    print('[%s] DONE: %d/%d (%d CMD_positive)' % (
        datetime.now().strftime('%H:%M:%S'), done, len(jobs), positive))

if __name__ == '__main__':
    main()
