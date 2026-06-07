#!/usr/bin/env python3
"""P2: Clean-Control 3R Confirmation
Runs 3R VIS on 12 clean-control candidates (6 types).
Goal: >=6 confirmed control/hard negatives for detector v3 training.

Usage:
  python scripts/run_clean_control_3r_confirmation.py --gpu_pair 0,1
"""

import os, sys, time, csv, subprocess, argparse
from datetime import datetime

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
CAND_CSV = os.path.join(REPO, 'tables/clean_rollout_control_negative_candidates.csv')
OUT_DIR = '/data/liuyu/outputs/clean_control_3r_confirmation_20260606'
SUMMARY_CSV = os.path.join(REPO, 'tables/clean_control_3r_confirmation_summary.csv')
REPORT_MD = os.path.join(REPO, 'reports/CLEAN_CONTROL_3R_CONFIRMATION.md')

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    print('%s %s' % (t, msg))

def run_vis3r(task, state_id, ws, we, gpu_pair, output_dir):
    ep_name = '%s_s%d' % (task, state_id)
    ep_dir = os.path.join(output_dir, ep_name)
    os.makedirs(ep_dir, exist_ok=True)
    log_path = os.path.join(ep_dir, 'vis_3r_control.log')

    cmd = [PY, '-u', os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3.py'),
        '--task', task,
        '--state-id', str(state_id),
        '--condition', 'vis_pgd',
        '--gpu_pair', gpu_pair,
        '--perturb_start', str(ws),
        '--perturb_end', str(we),
        '--eps_raw_pixels', '6',
        '--pgd_steps', '40',
        '--pgd_restarts', '3',
        '--objective', 'prefix_locked_gripper_open_margin',
        '--seed', '0',
    ]

    log('  Running 3R: %s s%d [%d,%d] -> %s' % (task, state_id, ws, we, log_path))
    t0 = time.time()
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
    rt = time.time() - t0

    # Parse result
    vis_open = '?'
    qpos_delta = '?'
    task_failure = '?'
    try:
        with open(log_path) as f:
            for line in f:
                if 'VIS_OPEN' in line:
                    vis_open = line.strip()
                if 'qpos_opening_delta' in line:
                    qpos_delta = line.strip()
                if 'done' in line.lower() and 'True' in line:
                    task_failure = 'True'
    except:
        pass

    return {'rc': rc, 'runtime_sec': round(rt, 1), 'vis_open': vis_open,
            'qpos_delta': qpos_delta, 'task_failure': task_failure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_pair', required=True)
    args = parser.parse_args()
    gpu_pair = args.gpu_pair

    with open(CAND_CSV) as f:
        candidates = list(csv.DictReader(f))
    log('Loaded %d clean-control candidates' % len(candidates))

    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    confirmed_negatives = 0

    for i, c in enumerate(candidates):
        task = c['task_key'].strip()
        sid = int(c['state_id'])
        ws = int(c['window_start'])
        we = int(c['window_end'])
        control_type = c.get('control_type', '?')
        reason = c.get('reason', '?')

        log('=== Control %d/%d: %s s%d [%d,%d] type=%s ===' % (i+1, len(candidates), task, sid, ws, we, control_type))

        r = run_vis3r(task, sid, ws, we, gpu_pair, OUT_DIR)
        time.sleep(5)

        result = {
            'task_key': task, 'state_id': str(sid),
            'window_start': str(ws), 'window_end': str(we),
            'control_type': control_type, 'reason': reason,
            'run_3r_rc': str(r['rc']),
            'vis_open': r['vis_open'],
            'qpos_delta': r['qpos_delta'],
            'task_failure': r['task_failure'],
            'runtime_sec': str(r['runtime_sec']),
            'gpu_pair': gpu_pair,
        }

        # Determine confirmation
        if r['rc'] == 0:
            # Parse outcome
            if 'VIS_OPEN=0' in str(r['vis_open']) or '0/' in str(r['vis_open']):
                result['confirmed_as'] = 'confirmed_negative'
                confirmed_negatives += 1
                log('  CONFIRMED NEGATIVE (%d/6 target)' % confirmed_negatives)
            else:
                result['confirmed_as'] = 'needs_review_opening_detected'
                log('  NEEDS REVIEW: opening detected in control window')
        else:
            result['confirmed_as'] = 'infra_failed'
            log('  INFRA FAILED: rc=%d' % r['rc'])

        results.append(result)

        # Save incremental
        with open(SUMMARY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

        if confirmed_negatives >= 6:
            log('GOAL REACHED: %d confirmed negatives >= 6!' % confirmed_negatives)

    # Final
    with open(SUMMARY_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log('Summary: %s' % SUMMARY_CSV)

    with open(REPORT_MD, 'w') as f:
        f.write('# Clean-Control 3R Confirmation\n\n')
        f.write('**Date**: %s\n\n' % datetime.now().strftime('%Y-%m-%d %H:%M'))
        f.write('**GPU**: %s\n\n' % gpu_pair)
        f.write('**Goal**: >=6 confirmed negatives\n\n')
        f.write('**Result**: %d/%d confirmed negatives\n\n' % (confirmed_negatives, len(results)))
        f.write('## Results\n\n')
        f.write('| # | Task | State | Window | Type | RC | Confirmed |\n')
        f.write('|---|------|-------|--------|------|----|----------|\n')
        for i, r in enumerate(results):
            f.write('| %d | %s | %s | [%s,%s] | %s | %s | %s |\n' % (
                i+1, r['task_key'], r['state_id'], r['window_start'], r['window_end'],
                r['control_type'], r['run_3r_rc'], r['confirmed_as']))
    log('Report: %s' % REPORT_MD)

if __name__ == '__main__':
    main()
