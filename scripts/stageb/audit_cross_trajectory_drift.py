#!/usr/bin/env python3
"""Audit K10 containment + 25D parity from saved rollout telemetry."""
import csv, json, sys, numpy as np

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]

canon_path = sys.argv[1] if len(sys.argv) > 1 else 'tables/v2_sc5_canonical_dataset.csv'
tel_paths = sys.argv[2:] if len(sys.argv) > 2 else [
    '/data/liuyu/outputs/rollout_e2e/step_telemetry.csv',
    '/data/liuyu/outputs/rollout_e2e_s0/step_telemetry.csv',
]

canon = {}
with open(canon_path) as f:
    for r in csv.DictReader(f):
        sid = r.get('state_id','')
        t = r.get('task_name','').lower()
        if 'butter' not in t: continue
        key = 's' + sid
        if key not in canon: canon[key] = []
        canon[key].append(r)

for tp in tel_paths:
    tel = []
    with open(tp) as f:
        for r in csv.DictReader(f):
            tel.append(r)
    if not tel: print(tp + ': EMPTY'); continue

    # Find emit
    emit = None; sid = None
    for r in tel:
        if int(r.get('mlp_emit', -1)) >= 0:
            emit = int(r['mlp_emit']); break
    if emit is None: print(tp + ': NO EMIT'); continue

    # Find state_id from telemetry
    sid = 's' + tel[0].get('step','0')  # rough, use path
    summary_path = tp.replace('step_telemetry.csv', 'episode_summary.json')
    try:
        with open(summary_path) as f: s = json.load(f)
        sid = 's' + str(s['state_id'])
        anchor = s.get('teacher_anchor', -1)
    except: anchor = -1

    can_rows = canon.get(sid, [])
    if not can_rows:
        print(sid + ': NO CANONICAL'); continue

    k10_map = {int(r['step_idx']): int(r.get('teacher_full_k10_valid_at_t', 0)) for r in can_rows}
    k10_at_emit = k10_map.get(emit, 0)

    # 25D parity
    max_errs = {}; n_compared = 0
    for r_tel in tel:
        step = int(r_tel['step'])
        if step >= emit: break
        can_at_step = [cr for cr in can_rows if int(cr['step_idx']) == step]
        if not can_at_step: continue
        cr = can_at_step[0]; n_compared += 1
        for fn in SC5_FEATURES:
            tv = float(r_tel.get('f_'+fn, 'nan'))
            cv = float(cr.get(fn, 'nan'))
            if np.isnan(tv) or np.isnan(cv): continue
            err = abs(tv - cv)
            if fn not in max_errs or err > max_errs[fn]: max_errs[fn] = err

    print('=== %s ===' % sid)
    print('  emit=%d anchor=%d error=%d K10_at_emit=%s' % (emit, anchor, emit-anchor, bool(k10_at_emit)))
    print('  compared=%d steps' % n_compared)
    if max_errs:
        worst = sorted(max_errs.items(), key=lambda x: -x[1])[:5]
        print('  worst_5: %s' % str([(f, round(e,6)) for f,e in worst]))
        print('  max_abs_err=%.6f' % max(max_errs.values()))
    print('  invalid=%d first_valid=%d' % (s.get('invalid_feature_steps', -1), s.get('first_valid_step', -1)))
