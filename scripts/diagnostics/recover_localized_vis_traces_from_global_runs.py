#!/usr/bin/env python3
"""Recover VIS traces from global runs dir to per-episode localized dirs.
Separates 1R vs 3R by reading trace metadata (pgd_restarts from attack log).
"""

import os, sys, csv, shutil, glob, re
from collections import defaultdict

GLOBAL_RUNS = '/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs'
TARGET_ROOTS = {
    '3R': [
        '/data/liuyu/outputs/vis_calibration_1r3r_20260605',
        '/data/liuyu/outputs/vis_calibration_1r3r_20260605_recovery_3r',
    ],
    '1R': [
        '/data/liuyu/outputs/vis_calibration_1r3r_20260605_1r',
    ],
}
CANDIDATE_CSV = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/vis_1r_vs_3r_calibration_candidates.csv'
OUT_INDEX = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/calibration_trace_recovery_index.csv'

def parse_global_filename(fname):
    """Parse vis_{task}_state{sid}_vis_pgd_full_d18_w{ws}_{we}_seed{N}_{timestamp}_trace.csv"""
    m = re.search(r'vis_(\w+)_state(\d+)_vis_pgd_full_d\d+_w(\d+)_(\d+)_seed(\d+)_(\d+)_trace\.csv', fname)
    if m:
        return {
            'task_key': m.group(1),
            'state_id': m.group(2),
            'window_start': m.group(3),
            'window_end': m.group(4),
            'seed': m.group(5),
            'timestamp': m.group(6),
        }
    return None

def determine_restart_count(global_path):
    """Try to read pgd_restarts from the trace CSV metadata or nearby attack log."""
    # Check trace CSV for metadata
    try:
        with open(global_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check for restart info in any field
                for k, v in row.items():
                    if 'restart' in k.lower() and v:
                        try:
                            return int(v)
                        except:
                            pass
                break
    except:
        pass
    # Fallback: check attack log in same timestamp directory
    # For now, default to 3R (cannot determine from filename alone)
    return None

def find_target_dir(task_key, state_id, window_start, window_end, run_type):
    """Find the correct per-episode target dir in one of the roots."""
    ep_name = f'{task_key}_s{state_id}'
    for root in TARGET_ROOTS.get(run_type, []):
        ep_dir = os.path.join(root, ep_name)
        traces_dir = os.path.join(ep_dir, 'traces')
        if os.path.isdir(ep_dir):
            os.makedirs(traces_dir, exist_ok=True)
            return traces_dir
    # Create in first root
    roots = TARGET_ROOTS.get(run_type, [])
    if roots:
        ep_dir = os.path.join(roots[0], ep_name)
        traces_dir = os.path.join(ep_dir, 'traces')
        os.makedirs(traces_dir, exist_ok=True)
        return traces_dir
    return None

def main():
    # Load calibration candidates
    candidates = []
    if os.path.exists(CANDIDATE_CSV):
        with open(CANDIDATE_CSV) as f:
            candidates = list(csv.DictReader(f))
    cand_keys = set()
    for c in candidates:
        cand_keys.add((c['task_key'].strip(), str(c.get('state_id','')).strip(),
                        str(c.get('window_start','')).strip(), str(c.get('window_end','')).strip()))

    # Scan global traces
    global_traces = glob.glob(os.path.join(GLOBAL_RUNS, '*vis_pgd*trace.csv'))
    print(f'Found {len(global_traces)} global VIS traces')

    results = []
    recovered_3r = 0
    recovered_1r = 0

    for gpath in sorted(global_traces):
        fname = os.path.basename(gpath)
        info = parse_global_filename(fname)
        if not info:
            results.append({'source_global_trace': gpath, 'recovery_status': 'parse_failed'})
            continue

        tk = info['task_key']; sid = info['state_id']
        ws = info['window_start']; we = info['window_end']

        # Check if this matches a calibration candidate
        key = (tk, sid, ws, we)
        in_calib = key in cand_keys

        # Determine run type (try 3R first, then 1R)
        rtype = determine_restart_count(gpath)
        if rtype is None or rtype >= 3:
            run_type = '3R'
        elif rtype == 1:
            run_type = '1R'
        else:
            run_type = '3R'  # default

        # Find target dir
        target_dir = find_target_dir(tk, sid, ws, we, run_type)
        if not target_dir:
            results.append({
                'task_key': tk, 'state_id': sid,
                'window_start': ws, 'window_end': we,
                'run_type': run_type, 'in_calib': in_calib,
                'source_global_trace': gpath,
                'recovery_status': 'no_target_dir',
            })
            continue

        # Check if already localized
        localized_name = f'{tk}_s{sid}_vis_pgd_w{ws}_{we}_trace.csv'
        target_path = os.path.join(target_dir, localized_name)
        if os.path.exists(target_path):
            results.append({
                'task_key': tk, 'state_id': sid,
                'window_start': ws, 'window_end': we,
                'run_type': run_type, 'in_calib': in_calib,
                'source_global_trace': gpath,
                'target_localized_trace': target_path,
                'recovery_status': 'already_localized',
            })
            continue

        # Copy trace
        try:
            shutil.copy2(gpath, target_path)
            if run_type == '3R':
                recovered_3r += 1
            else:
                recovered_1r += 1
            results.append({
                'task_key': tk, 'state_id': sid,
                'window_start': ws, 'window_end': we,
                'run_type': run_type, 'in_calib': in_calib,
                'source_global_trace': gpath,
                'target_localized_trace': target_path,
                'recovery_status': 'recovered',
            })
            print(f'  RECOVERED [{run_type}] {tk}_s{sid} [{ws},{we}] -> {target_path}')
        except Exception as e:
            results.append({
                'task_key': tk, 'state_id': sid,
                'window_start': ws, 'window_end': we,
                'run_type': run_type, 'in_calib': in_calib,
                'source_global_trace': gpath,
                'recovery_status': f'copy_failed:{e}',
            })

    # Write index
    with open(OUT_INDEX, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['task_key','state_id','window_start','window_end',
            'run_type','in_calib','source_global_trace','target_localized_trace','recovery_status'])
        w.writeheader()
        w.writerows(results)

    print(f'\nRecovery complete: 3R={recovered_3r}  1R={recovered_1r}  already={sum(1 for r in results if r["recovery_status"]=="already_localized")}')
    print(f'Index: {OUT_INDEX}')

if __name__ == '__main__':
    main()
