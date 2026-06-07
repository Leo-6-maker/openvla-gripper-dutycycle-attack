#!/usr/bin/env python3
"""P1: Config-Matched Calibration v2 — 1R vs 3R
Runs matched 1R and 3R VIS on 10 candidates (5 pos + 5 neg).
Same candidate, same config, only pgd_restarts differs.

Usage:
  python scripts/run_calibration_v2_matched.py --gpu_pair 0,1
"""

import os, sys, time, csv, subprocess, argparse
from datetime import datetime

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
CAND_CSV = os.path.join(REPO, 'tables/vis_1r_vs_3r_calibration_v2_candidates.csv')
OUT_1R = '/data/liuyu/outputs/vis_calibration_matched_v2_1r_20260606'
OUT_3R = '/data/liuyu/outputs/vis_calibration_matched_v2_3r_20260606'
SUMMARY_CSV = os.path.join(REPO, 'tables/vis_1r_vs_3r_calibration_v2_results.csv')
REPORT_MD = os.path.join(REPO, 'reports/VIS_1R_VS_3R_CALIBRATION_V2.md')

COMMON_ARGS = [
    '--eps_raw_pixels', '6',
    '--pgd_steps', '40',
    '--objective', 'prefix_locked_gripper_open_margin',
    '--seed', '0',
]

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    print('%s %s' % (t, msg))

def run_vis(task, state_id, ws, we, gpu_pair, restart_count, output_dir):
    """Run a single VIS rollout."""
    ep_name = '%s_s%s' % (task, state_id)
    ep_dir = os.path.join(output_dir, ep_name)
    os.makedirs(ep_dir, exist_ok=True)

    log_name = 'vis_%dR.log' % restart_count
    log_path = os.path.join(ep_dir, log_name)

    cmd = [PY, '-u', os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3.py'),
        '--task', task,
        '--state-id', str(state_id),
        '--condition', 'vis_pgd',
        '--gpu_pair', gpu_pair,
        '--perturb_start', str(ws),
        '--perturb_end', str(we),
        '--pgd_restarts', str(restart_count),
    ] + COMMON_ARGS

    log('  Running %dR: %s s%d [%s,%s] -> %s' % (restart_count, task, state_id, ws, we, log_path))
    t0 = time.time()
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
    rt = time.time() - t0

    # Parse result: check for VIS_OPEN count in log
    vis_open = '?'
    try:
        with open(log_path) as f:
            for line in f:
                if 'VIS_OPEN' in line:
                    vis_open = line.strip().split('VIS_OPEN')[-1].strip().split()[0]
                    break
                if 'vis_open_count' in line:
                    vis_open = line.strip()
    except:
        pass

    return {'rc': rc, 'runtime_sec': round(rt, 1), 'vis_open': vis_open, 'log_path': log_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_pair', required=True, help='GPU pair e.g. 0,1')
    args = parser.parse_args()
    gpu_pair = args.gpu_pair

    # Load candidates
    with open(CAND_CSV) as f:
        candidates = list(csv.DictReader(f))
    log('Loaded %d calibration v2 candidates' % len(candidates))

    os.makedirs(OUT_1R, exist_ok=True)
    os.makedirs(OUT_3R, exist_ok=True)

    results = []

    for i, c in enumerate(candidates):
        task = c['task_key'].strip()
        sid = int(c['state_id'])
        ws = int(c['window_start'])
        we = int(c['window_end'])
        expected = c.get('expected_label', '?')
        reason = c.get('reason', '?')

        log('=== Candidate %d/%d: %s s%d [%d,%d] expected=%s ===' % (i+1, len(candidates), task, sid, ws, we, expected))

        # Run 1R first (faster)
        r1 = run_vis(task, sid, ws, we, gpu_pair, 1, OUT_1R)
        time.sleep(5)

        # Run 3R
        r3 = run_vis(task, sid, ws, we, gpu_pair, 3, OUT_3R)
        time.sleep(5)

        result = {
            'task_key': task, 'state_id': str(sid),
            'window_start': str(ws), 'window_end': str(we),
            'expected_label': expected, 'reason': reason,
            'run_1r_rc': str(r1['rc']), 'run_1r_vis_open': r1['vis_open'],
            'run_1r_runtime_sec': str(r1['runtime_sec']),
            'run_3r_rc': str(r3['rc']), 'run_3r_vis_open': r3['vis_open'],
            'run_3r_runtime_sec': str(r3['runtime_sec']),
            'gpu_pair': gpu_pair,
        }

        # Compare
        if r1['rc'] == 0 and r3['rc'] == 0:
            result['paired_status'] = 'both_ok'
        elif r1['rc'] != 0 and r3['rc'] != 0:
            result['paired_status'] = 'both_infra_failed'
        elif r1['rc'] != 0:
            result['paired_status'] = '1r_infra_failed'
        else:
            result['paired_status'] = '3r_infra_failed'

        results.append(result)

        # Save incremental summary
        with open(SUMMARY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

        log('  Result: %s | 1R: rc=%d open=%s | 3R: rc=%d open=%s' % (
            result['paired_status'], r1['rc'], r1['vis_open'], r3['rc'], r3['vis_open']))

    # ── Final summary ──
    log('=== Calibration v2 Complete ===')
    both_ok = [r for r in results if r['paired_status'] == 'both_ok']
    log('Both OK: %d/%d' % (len(both_ok), len(results)))

    # Write summary CSV
    with open(SUMMARY_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log('Summary: %s' % SUMMARY_CSV)

    # Write report
    with open(REPORT_MD, 'w') as f:
        f.write('# VIS 1R vs 3R Calibration v2\n\n')
        f.write('**Date**: %s\n\n' % datetime.now().strftime('%Y-%m-%d %H:%M'))
        f.write('**GPU**: %s\n\n' % gpu_pair)
        f.write('## Results\n\n')
        f.write('| # | Task | State | Window | Expected | 1R RC | 1R Open | 3R RC | 3R Open | Status |\n')
        f.write('|---|------|-------|--------|----------|-------|---------|-------|---------|--------|\n')
        for i, r in enumerate(results):
            f.write('| %d | %s | %s | [%s,%s] | %s | %s | %s | %s | %s | %s |\n' % (
                i+1, r['task_key'], r['state_id'], r['window_start'], r['window_end'],
                r['expected_label'], r['run_1r_rc'], r['run_1r_vis_open'],
                r['run_3r_rc'], r['run_3r_vis_open'], r['paired_status']))

    log('Report: %s' % REPORT_MD)

if __name__ == '__main__':
    main()
