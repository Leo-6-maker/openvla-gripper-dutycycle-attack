#!/usr/bin/env python3
"""Phase 3 command-hold auditor: arm delta, contact loss, recovery metrics."""
import csv, json, os, sys, numpy as np
from pathlib import Path

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('tables')
RPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('reports')
BASE = sys.argv[3] if len(sys.argv) > 3 else '/data/liuyu/outputs/v2_phase3_pilot'

for d in [OUT_DIR, RPT_DIR]: d.mkdir(parents=True, exist_ok=True)

def sf(v, default=float('nan')):
    try: return float(v)
    except: return default

results = []
for state in ['s0', 's2']:
    for cond in ['CLEAN', 'HOLD_D5_K10', 'HOLD_SG5_K10', 'HOLD_SC5_K10', 'HOLD_MID_K10']:
        path = os.path.join(BASE, state, cond)
        tele = os.path.join(path, 'step_telemetry.csv')
        summ = os.path.join(path, 'episode_summary.json')
        if not os.path.isfile(tele):
            results.append({'state': state, 'condition': cond, 'present': False})
            continue

        rows = list(csv.DictReader(open(tele)))
        summary = json.load(open(summ)) if os.path.isfile(summ) else {}
        n_steps = len(rows)
        anchor = int(summary.get('anchor', -1))
        K = int(summary.get('K', 10))

        # Arm delta
        arm_ok = all(r.get('arm_delta_ok') in ('True', 'true', '1', True) for r in rows)
        arm_delta_max = max(sf(r.get('arm_delta_max', 0)) for r in rows)

        # Hold window
        hold_rows = [r for r in rows if r.get('in_window') in ('True', 'true', '1')]
        n_hold = len(hold_rows)
        hold_ok = (n_hold == K) if cond != 'CLEAN' else (n_hold == 0)
        hold_open_ok = all(sf(r.get('executed_env_grip', 0)) < 0 for r in hold_rows) if hold_rows else True

        # Object lift check
        obj_z0 = sf(rows[0].get('obj_z0', rows[0].get('obj_z', 0)))
        obj_z_vals = [(sf(r['obj_z']), sf(r['eef_obj_dist']))
                      for r in rows if str(r.get('obj_z','')) not in ('nan','')]

        # Pre-window baseline (steps ANCHOR-10 to ANCHOR-1)
        pre_rows = [r for r in rows if anchor - 10 <= int(r['step']) < anchor]
        pre_eef_dists = [sf(r['eef_obj_dist']) for r in pre_rows if str(r.get('eef_obj_dist','')) not in ('nan','')]
        baseline_eef_median = np.median(pre_eef_dists) if pre_eef_dists else float('nan')

        # Post-window object tracking
        post_rows = [r for r in rows if anchor <= int(r['step']) < min(n_steps, anchor + 30)]
        post_eef = [(int(r['step']), sf(r['eef_obj_dist']), sf(r['obj_z']))
                    for r in post_rows if str(r.get('eef_obj_dist','')) not in ('nan','')]

        # Contact loss: eef_obj_dist increase >= 0.04 sustained >= 2 steps
        contact_loss = False
        contact_loss_start = -1
        consec = 0
        for s, d, z in post_eef:
            if not np.isnan(baseline_eef_median) and d > baseline_eef_median + 0.04:
                consec += 1
                if consec >= 2 and not contact_loss:
                    contact_loss = True; contact_loss_start = s
            else:
                consec = 0

        # Object z drop >= 0.03 sustained >= 2
        obj_drop = False
        obj_drop_start = -1
        consec_z = 0
        pre_z_vals = [sf(r['obj_z']) for r in pre_rows if str(r.get('obj_z','')) not in ('nan','')]
        pre_z_median = np.median(pre_z_vals) if pre_z_vals else float('nan')
        for s, d, z in post_eef:
            if not np.isnan(pre_z_median) and z < pre_z_median - 0.03:
                consec_z += 1
                if consec_z >= 2 and not obj_drop:
                    obj_drop = True; obj_drop_start = s
            else:
                consec_z = 0

        # Peak eef_distance increase
        peak_eef = max([d for _, d, _ in post_eef]) if post_eef else float('nan')
        peak_eef_delta = peak_eef - baseline_eef_median if not np.isnan(peak_eef) and not np.isnan(baseline_eef_median) else float('nan')

        # Recovery: eef_dist returns within baseline + 0.02
        recovery_step = -1
        for s, d, z in post_eef:
            if s > anchor + 2 and not np.isnan(baseline_eef_median) and d <= baseline_eef_median + 0.02:
                recovery_step = s; break
        recovery_latency = recovery_step - anchor if recovery_step > 0 else -1

        # Extra steps vs CLEAN
        success = bool(summary.get('task_success', False))

        results.append({
            'state': state, 'condition': cond, 'present': True,
            'n_steps': n_steps, 'anchor': anchor, 'K': K,
            'arm_delta_max': arm_delta_max, 'arm_delta_ok': arm_ok,
            'hold_count': n_hold, 'hold_count_ok': hold_ok,
            'hold_open_all': hold_open_ok,
            'baseline_eef_median': round(baseline_eef_median, 6) if not np.isnan(baseline_eef_median) else None,
            'peak_eef_delta': round(peak_eef_delta, 6) if not np.isnan(peak_eef_delta) else None,
            'contact_loss': contact_loss, 'contact_loss_start': contact_loss_start,
            'obj_drop': obj_drop, 'obj_drop_start': obj_drop_start,
            'recovery_latency': recovery_latency,
            'task_success': success,
        })

# Write table
with open(OUT_DIR / 'v2_phase3_pilot_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)

# Summary
print('=== Phase 3 Pilot Results ===')
for r in results:
    if not r['present']:
        print('%s/%s: MISSING' % (r['state'], r['condition']))
    else:
        print('%s/%s: steps=%d arm_ok=%s peak_eef=%.4f contact=%s drop=%s rec_lat=%d success=%s' % (
            r['state'], r['condition'], r['n_steps'], r['arm_delta_ok'],
            r['peak_eef_delta'] or 0, r['contact_loss'], r['obj_drop'],
            r['recovery_latency'], r['task_success']))
