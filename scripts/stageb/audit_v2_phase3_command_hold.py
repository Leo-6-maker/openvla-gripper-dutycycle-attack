#!/usr/bin/env python3
"""Phase 3 command-hold auditor: full contract verification + contact/recovery metrics."""
import csv, json, os, sys, numpy as np
from pathlib import Path

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('tables')
RPT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('reports')
BASE = sys.argv[3] if len(sys.argv) > 3 else '/data/liuyu/outputs/v2_phase3_pilot'
MANIFEST = sys.argv[4] if len(sys.argv) > 4 else '/data/liuyu/audit_v3/v2_phase3_anchor_manifest.json'

for d in [OUT_DIR, RPT_DIR]: d.mkdir(parents=True, exist_ok=True)

manifest = json.load(open(MANIFEST))
K = manifest['K']

def sf(v, default=float('nan')):
    try: return float(v)
    except: return default

results = []
for state in ['s0']:
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

        # ── G0: Contract checks ──
        is_hold = cond != 'CLEAN'
        hold_rows = [r for r in rows if r.get('in_window') in ('True', 'true', '1')]
        n_hold = len(hold_rows)
        hold_steps_actual = [int(r['step']) for r in hold_rows]

        # Exact window check
        if is_hold:
            expected_steps = list(range(anchor, anchor + K))
            steps_exact = (hold_steps_actual == expected_steps)
            hold_complete = (n_hold == K)
            window_truncated = n_hold < K and n_hold > 0
        else:
            steps_exact = (n_hold == 0)
            hold_complete = True
            window_truncated = False

        # Arm delta check
        arm_deltas = [sf(r.get('arm_delta_max', 0)) for r in rows]
        arm_delta_max = max(arm_deltas) if arm_deltas else 0
        arm_ok = (arm_delta_max == 0.0)

        # Hold OPEN check
        hold_open_ok = all(sf(r.get('executed_env_grip', 0) if 'executed_env_grip' in r else r.get('env_gripper', 1)) < 0
                          for r in hold_rows) if hold_rows else True

        # Anchor vs manifest
        state_key = 'state_0'
        manifest_anchor = None
        for wname, wcond in [('W0_D5', 'HOLD_D5_K10'), ('W1_SG5', 'HOLD_SG5_K10'),
                              ('W2_SC5', 'HOLD_SC5_K10'), ('W3_MID', 'HOLD_MID_K10')]:
            if cond == wcond:
                wi = manifest['states'][state_key]['windows'].get(wname, {})
                manifest_anchor = wi.get('anchor', None)
        anchor_match = (anchor == manifest_anchor) if is_hold else True

        # ── Physical metrics ──
        obj_z0 = sf(rows[0].get('obj_z0', rows[0].get('obj_z', 0)))

        # Pre-window baseline (anchor-10 to anchor-1)
        pre_rows = [r for r in rows if max(0, anchor - 10) <= int(r['step']) < anchor]
        pre_eef = [sf(r['eef_obj_dist']) for r in pre_rows if str(r.get('eef_obj_dist','')) not in ('nan','')]
        baseline_eef_median = np.median(pre_eef) if pre_eef else float('nan')
        pre_z = [sf(r['obj_z']) for r in pre_rows if str(r.get('obj_z','')) not in ('nan','')]
        pre_z_median = np.median(pre_z) if pre_z else float('nan')

        # Object lifted before anchor?
        obj_lifted_pre = any((sf(r['obj_z']) - obj_z0) > 0.02 for r in pre_rows) if pre_rows else False

        # Post-window object tracking (anchor to anchor+30)
        post_rows = [r for r in rows if anchor <= int(r['step']) < min(n_steps, anchor + 30)]
        post_eef = [(int(r['step']), sf(r['eef_obj_dist']), sf(r['obj_z']))
                    for r in post_rows if str(r.get('eef_obj_dist','')) not in ('nan','')]

        # Contact loss: eef_obj_dist increase >= 0.04 sustained >= 2 steps
        contact_loss = False; contact_loss_start = -1; consec = 0
        for s, d, z in post_eef:
            if not np.isnan(baseline_eef_median) and d > baseline_eef_median + 0.04:
                consec += 1
                if consec >= 2 and not contact_loss:
                    contact_loss = True; contact_loss_start = s
            else:
                consec = 0

        # Object z drop >= 0.03 sustained >= 2
        obj_drop = False; obj_drop_start = -1; consec_z = 0
        for s, d, z in post_eef:
            if not np.isnan(pre_z_median) and z < pre_z_median - 0.03:
                consec_z += 1
                if consec_z >= 2 and not obj_drop:
                    obj_drop = True; obj_drop_start = s
            else:
                consec_z = 0

        # Peak eef increase
        peak_eef = max([d for _, d, _ in post_eef]) if post_eef else float('nan')
        peak_eef_delta = peak_eef - baseline_eef_median if not np.isnan(peak_eef) and not np.isnan(baseline_eef_median) else float('nan')

        # Recovery: eef returns within baseline + 0.02
        recovery_step = -1
        for s, d, z in post_eef:
            if s > anchor + 2 and not np.isnan(baseline_eef_median) and d <= baseline_eef_median + 0.02:
                recovery_step = s; break
        recovery_latency = recovery_step - anchor if recovery_step > 0 else -1

        # Window crosses release_safe?
        sc_start = manifest['states'][state_key].get('sc_start')
        rs_start = manifest['states'][state_key].get('rs_start')
        crosses_release = (rs_start is not None and is_hold and anchor + K - 1 >= rs_start)

        success = bool(summary.get('task_success', False))

        # ── Gate verdict ──
        gates = []
        gates.append(('G0_DATA', os.path.isfile(tele)))
        gates.append(('G1_ARM_DELTA_ZERO', arm_ok))
        gates.append(('G2_ANCHOR_MATCH_MANIFEST', anchor_match))
        if is_hold:
            gates.append(('G3_HOLD_COUNT_EXACT', n_hold == K))
            gates.append(('G4_STEPS_EXACT', steps_exact))
            gates.append(('G5_HOLD_OPEN_ALL', hold_open_ok))
            gates.append(('G6_WINDOW_NOT_TRUNCATED', not window_truncated))
            gates.append(('G7_NOT_CROSSES_RELEASE', not crosses_release))
        all_gates_pass = all(p for _, p in gates)

        results.append({
            'state': state, 'condition': cond, 'anchor': anchor,
            'n_steps': n_steps, 'n_hold': n_hold,
            'hold_steps_actual': str(hold_steps_actual),
            'steps_exact': steps_exact,
            'window_truncated_by_done': window_truncated,
            'arm_ok': arm_ok, 'arm_delta_max': arm_delta_max,
            'anchor_ok': anchor_match,
            'hold_ok': hold_complete, 'hold_open_ok': hold_open_ok,
            'crosses_release': crosses_release,
            'obj_lifted_pre': obj_lifted_pre,
            'baseline_eef_median': round(baseline_eef_median, 6) if not np.isnan(baseline_eef_median) else None,
            'peak_eef_delta': round(peak_eef_delta, 6) if not np.isnan(peak_eef_delta) else None,
            'contact_loss': contact_loss, 'contact_loss_start': contact_loss_start,
            'obj_drop': obj_drop, 'obj_drop_start': obj_drop_start,
            'recovery_latency': recovery_latency,
            'task_success': success,
            'all_gates_pass': all_gates_pass,
            'gates': str(gates),
        })

# ── Write ──
with open(OUT_DIR / 'v2_phase3_s0_audit_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)

print('=== State 0 Audit ===')
all_pass = True
for r in results:
    if not r['present']:
        print('%s: MISSING' % r['condition'])
        all_pass = False; continue

    flags = []
    if not r['anchor_ok']: flags.append('ANCHOR_MISMATCH')
    if not r['arm_ok']: flags.append('ARM_DELTA_FAIL')
    if not r['hold_ok'] and r['condition'] != 'CLEAN': flags.append('HOLD_COUNT_FAIL')
    if r['window_truncated_by_done']: flags.append('WINDOW_TRUNCATED_BY_DONE')
    if r['crosses_release']: flags.append('CROSSES_RELEASE')
    if not flags: flags.append('CONTRACT_PASS')

    print('%s a=%d: steps=%d hold=%d/%d arm_ok=%s succ=%s lifted=%s eef_peak=%.4f contact=%s recovery=%d | %s' % (
        r['condition'], r['anchor'], r['n_steps'], r['n_hold'],
        10 if r['condition'] != 'CLEAN' else 0,
        r['arm_ok'], r['task_success'], r['obj_lifted_pre'],
        r['peak_eef_delta'] or 0, r['contact_loss'], r['recovery_latency'],
        ' '.join(flags)))
    if not r['all_gates_pass'] and r['condition'] != 'CLEAN':
        all_pass = False

print('\nCONTRACT: %s' % ('PASS' if all_pass else 'FAIL'))
print('HOLD_MID: %s' % ('WINDOW_TRUNCATED_BY_DONE' if any(r['window_truncated_by_done'] for r in results) else 'OK'))
