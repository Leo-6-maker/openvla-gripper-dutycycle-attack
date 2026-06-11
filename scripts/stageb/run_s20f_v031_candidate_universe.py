#!/usr/bin/env python3
"""S20f: Build v0.3.1 phase-aware candidate universe from S20d V4 clean traces.
Sliding windows with phase features: approach/grasp/lift/transport/preplace.
Window convention: half-open [ws, we), window_len=10, stride=5."""
import csv, numpy as np, os, sys

OUT_DIR = '/data/liuyu/outputs/stageb_s20f_v031_repair_20260611'
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_LEN = 10
STRIDE = 5
MIN_WS = 5

TRACES = {
    ('ketchup', '1'): '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_ketchup_s1_w0_10_s20d_clean_seed0_job960100.csv',
    ('tomato_sauce', '3'): '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_tomato_sauce_s3_w0_10_s20d_clean_seed0_job960101.csv',
    ('tomato_sauce', '5'): '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_tomato_sauce_s5_w0_10_s20d_clean_seed0_job960102.csv',
}

# ── Phase detection from clean trace ──
def detect_phases(rows):
    """Detect grasp, lift, transport, preplace phases from clean gripper + eef data."""
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d

    is_open = [g(r, 'decoded_open_bool') for r in rows]
    eef_z_vals = [g(r, 'eef_z') for r in rows]
    obj_z_vals = [g(r, 'obj_z') for r in rows]
    grip_vals = [g(r, 'clean_gripper_env') for r in rows]

    # Find first stable CLOSE streak (>=3)
    first_close = None
    streak = 0
    for i, o in enumerate(is_open):
        if o == 0:  # CLOSE
            if streak == 0: first_close = i
            streak += 1
            if streak >= 3 and first_close is not None:
                first_close = first_close
                break
        else:
            streak = 0
            first_close = None
    if streak < 3:
        first_close = None

    # Find lift: object_z increase from baseline
    base_obj = float(np.median([g(r, 'obj_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    base_eef = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    lift_step = None
    for i in range(first_close or 0, len(rows)):
        if g(rows[i], 'obj_z') - base_obj >= 0.015 or g(rows[i], 'eef_z') - base_eef >= 0.03:
            lift_step = i
            break

    # Find preplace: eef_z starts descending from peak (after lift)
    preplace_step = None
    peak_step = None
    peak_z = None
    if lift_step is not None:
        for i in range(lift_step, len(rows)):
            z = g(rows[i], 'eef_z')
            if peak_z is None or z > peak_z:
                peak_z = z
                peak_step = i
            if peak_step is not None and i > peak_step + 3 and peak_z - z >= 0.005:
                preplace_step = i
                break

    # Find done step
    done_step = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary', '0') or '0') == 1 or int(r.get('success_done', '0') or '0') == 1:
            done_step = i
            break
    if done_step is None:
        done_step = len(rows) - 1

    return {
        'first_close_step': first_close,
        'lift_step': lift_step,
        'preplace_step': preplace_step,
        'done_step': done_step,
        'base_obj_z': base_obj,
        'base_eef_z': base_eef,
        'max_steps': len(rows),
    }


def classify_phase(ws, we, phases):
    """Classify a window into a phase bucket."""
    fc = phases['first_close_step']
    lift = phases['lift_step']
    pp = phases['preplace_step']
    wc = (ws + we) / 2.0
    done = phases['done_step']

    if wc >= done - 5:
        return 'place_or_done'
    if pp is not None and wc >= pp:
        return 'preplace'
    if lift is not None and wc >= lift + 5:
        return 'transport'
    if lift is not None and wc >= lift:
        return 'early_transport'
    if fc is not None and wc >= fc:
        return 'grasp_transition'
    return 'approach'


def compute_features(trace_path, ws, we, phases):
    with open(trace_path) as f:
        rows = list(csv.DictReader(f))

    window_rows = [r for r in rows if ws <= int(r['step']) < we]
    pre_rows = [r for r in rows if int(r['step']) < ws]

    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d

    # Clean OPEN features
    clean_open_count = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)
    clean_open_frac = clean_open_count / max(len(window_rows), 1)

    # Gripper command features
    gripper_vals = [g(r, 'clean_gripper_env') for r in window_rows]
    raw_gripper_mean = float(np.mean(gripper_vals)) if gripper_vals else 0.0
    raw_gripper_max = float(np.max(gripper_vals)) if gripper_vals else 0.0

    # Qpos features
    qpos_pre_vals = [g(r, 'gripper_qpos_before') for r in pre_rows[-5:]] if pre_rows else [0.0]
    qpos_pre = float(np.median(qpos_pre_vals)) if qpos_pre_vals else 0.0
    qpos_vals = [g(r, 'gripper_qpos_before') for r in window_rows]
    qpos_mean = float(np.mean(qpos_vals)) if qpos_vals else 0.0
    qpos_max = float(np.max(qpos_vals)) if qpos_vals else 0.0
    if len(qpos_vals) >= 2:
        qpos_slope = float(np.polyfit(np.arange(len(qpos_vals)), qpos_vals, 1)[0])
    else:
        qpos_slope = 0.0

    # EEF displacement
    eef_z_vals = [g(r, 'eef_z') for r in window_rows]
    eef_disp = float(np.max(eef_z_vals) - np.min(eef_z_vals)) if len(eef_z_vals) >= 2 else 0.0

    # Timing
    done_step = phases['done_step']
    wc = (ws + we) / 2.0
    rel_timing = wc / max(done_step, 1)

    # Phase-aware features
    fc = phases['first_close_step']
    lift = phases['lift_step']
    after_first_close = fc is not None and ws >= fc
    after_lift = lift is not None and ws >= lift

    # OPEN count before/after first close
    open_before_fc = sum(1 for r in window_rows
                         if g(r, 'decoded_open_bool') == 1 and int(r['step']) < (fc or 999))
    open_after_fc = sum(1 for r in window_rows
                        if g(r, 'decoded_open_bool') == 1 and int(r['step']) >= (fc or 0))
    # post-grasp TRANSPORT OPENs (after first_close, not in preplace)
    pp = phases['preplace_step']
    post_grasp_open = sum(1 for r in window_rows
                          if g(r, 'decoded_open_bool') == 1
                          and int(r['step']) >= (fc or 0)
                          and (pp is None or int(r['step']) < pp))

    phase_id = classify_phase(ws, we, phases)

    return {
        'clean_open_count': clean_open_count,
        'clean_open_frac': clean_open_frac,
        'raw_gripper_mean': raw_gripper_mean,
        'raw_gripper_max': raw_gripper_max,
        'qpos_pre': qpos_pre,
        'qpos_mean': qpos_mean,
        'qpos_max': qpos_max,
        'qpos_slope': qpos_slope,
        'eef_disp': eef_disp,
        'wc': wc,
        'rel_timing': rel_timing,
        'phase_id': phase_id,
        'open_before_first_close': open_before_fc,
        'open_after_first_close': open_after_fc,
        'post_grasp_open_count': post_grasp_open,
        'after_first_close': after_first_close,
        'after_lift': after_lift,
        'first_close_step': fc,
        'lift_step': lift,
        'preplace_step': pp,
        'done_step': done_step,
        'actual_max_step': phases['max_steps'],
    }


# ── Build universe ──
universe = []
for (task, sid), tpath in TRACES.items():
    with open(tpath) as f:
        rows = list(csv.DictReader(f))
    phases = detect_phases(rows)
    clean_success = any(r.get('success_primary') == '1' for r in rows)

    print('[%s_s%s] first_close=%s lift=%s preplace=%s done=%s' % (
        task, sid, phases['first_close_step'], phases['lift_step'],
        phases['preplace_step'], phases['done_step']))

    max_step = phases['max_steps']
    for ws in range(MIN_WS, max_step - WINDOW_LEN, STRIDE):
        we = ws + WINDOW_LEN
        # Skip windows that end after done
        if we > phases['done_step'] + 5:
            continue
        cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)
        feats = compute_features(tpath, ws, we, phases)
        phase = feats['phase_id']

        # Stratum
        if phase == 'transport' and feats['clean_open_count'] <= 1:
            stratum = 'high_opportunity'
        elif phase in ('early_transport', 'transport', 'preplace') and feats['clean_open_count'] <= 3:
            stratum = 'medium_opportunity'
        elif phase in ('grasp_transition', 'early_transport'):
            stratum = 'medium_opportunity'
        else:
            stratum = 'hard_or_idle'

        universe.append({
            'candidate_id': cid, 'task': task, 'state_id': sid,
            'window_start': ws, 'window_end': we, 'window_len': WINDOW_LEN,
            'window_indexing': 'half_open',
            'phase_id': phase, 'stratum': stratum,
            **feats,
            'clean_success_done': clean_success, 'clean_success_check': clean_success,
            'source_trace': tpath, 'source_runner': 's20d_v4_fixed_window_l3',
        })

# Write
u_fields = ['candidate_id', 'task', 'state_id', 'window_start', 'window_end',
            'window_len', 'window_indexing', 'phase_id', 'stratum',
            'clean_open_count', 'clean_open_frac', 'raw_gripper_mean', 'raw_gripper_max',
            'qpos_pre', 'qpos_mean', 'qpos_max', 'qpos_slope', 'eef_disp',
            'wc', 'rel_timing',
            'open_before_first_close', 'open_after_first_close', 'post_grasp_open_count',
            'after_first_close', 'after_lift',
            'first_close_step', 'lift_step', 'preplace_step', 'done_step', 'actual_max_step',
            'clean_success_done', 'clean_success_check',
            'source_trace', 'source_runner']

out_csv = os.path.join(OUT_DIR, 's20f_v031_candidate_universe.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=u_fields, extrasaction='ignore')
    w.writeheader(); w.writerows(universe)

# Stats
from collections import Counter
phase_counts = Counter(u['phase_id'] for u in universe)
stratum_counts = Counter(u['stratum'] for u in universe)
print('\nPhase distribution: %s' % dict(phase_counts))
print('Stratum distribution: %s' % dict(stratum_counts))
print('Total: %d windows' % len(universe))
print('Output: %s' % out_csv)

# Suggest labeling batch
print('\n=== SUGGESTED LABELING QUEUE (12 windows) ===')
# Sample by phase
selected = []
for phase in ['approach', 'grasp_transition', 'early_transport', 'transport', 'preplace']:
    phase_wins = sorted([u for u in universe if u['phase_id'] == phase], key=lambda u: u['rel_timing'])
    if phase_wins:
        # Pick one early, one middle from each phase
        n = len(phase_wins)
        for idx in [n//4, 3*n//4]:
            if 0 <= idx < n:
                selected.append(phase_wins[min(idx, n-1)])

# Deduplicate
seen = set()
label_queue = []
for u in selected:
    if u['candidate_id'] not in seen:
        label_queue.append(u)
        seen.add(u['candidate_id'])

label_csv = os.path.join(OUT_DIR, 's20f_v031_labeling_queue_seed80.csv')
with open(label_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=u_fields, extrasaction='ignore')
    w.writeheader(); w.writerows(label_queue)

print('\nLabeling queue:')
for u in label_queue:
    print('  %s phase=%-18s open=%d/%d after_fc=%s after_lift=%s' % (
        u['candidate_id'], u['phase_id'],
        int(u['clean_open_count']), int(u['open_after_first_close']),
        u['after_first_close'], u['after_lift']))
print('Output: %s' % label_csv)
