#!/usr/bin/env python3
"""Generate Batch3c control candidates from NPZ — stable/post_lock, far_too_early, pre_lock."""
import csv, os, numpy as np
from collections import defaultdict

data = np.load('data/detector/object_clean_sequences_v3.npz', allow_pickle=True)
Xr = data['X_raw']; mask = data['mask']; ep_ids = list(data['episode_ids'])

meta = {}
with open('data/detector/object_clean_sequences_v3_meta.csv') as f:
    for r in csv.DictReader(f): meta[r['episode_id']] = r

tested = set()
for fn in ['tables/object_teacher_delay50_vis_smoke_batch1.csv',
           'tables/object_phase_response_batch2b_balanced_candidates.csv',
           'tables/object_phase_response_batch3_candidates.csv',
           'tables/object_phase_response_batch3b_candidates.csv']:
    if os.path.exists(fn):
        with open(fn) as f:
            for r in csv.DictReader(f):
                tested.add((r['task_key'], r.get('state_id','0'), r.get('window_start',''), r.get('window_end','')))

TASK_KEY_MAP = {
    'pick_up_the_alphabet_soup_and_place_it_in_the_basket': 'alphabet_soup',
    'pick_up_the_cream_cheese_and_place_it_in_the_basket': 'cream_cheese',
    'pick_up_the_salad_dressing_and_place_it_in_the_basket': 'salad_dressing',
    'pick_up_the_bbq_sauce_and_place_it_in_the_basket': 'bbq_sauce',
    'pick_up_the_ketchup_and_place_it_in_the_basket': 'ketchup',
    'pick_up_the_tomato_sauce_and_place_it_in_the_basket': 'tomato_sauce',
    'pick_up_the_butter_and_place_it_in_the_basket': 'butter',
    'pick_up_the_milk_and_place_it_in_the_basket': 'milk',
    'pick_up_the_chocolate_pudding_and_place_it_in_the_basket': 'chocolate_pudding',
    'pick_up_the_orange_juice_and_place_it_in_the_basket': 'orange_juice',
}

WINDOW_LEN = 18
PRIORITY = ['tomato_sauce','orange_juice','cream_cheese','salad_dressing','milk','bbq_sauce','alphabet_soup','butter','ketchup']
controls = {role: [] for role in ['stable_post_lock','far_too_early','pre_lock']}

for orig_idx, eid in enumerate(ep_ids):
    eid_str = str(eid)
    ep = meta.get(eid_str, {})
    tg_str = ep.get('T_gform','')
    if not tg_str: continue
    tg = int(tg_str)
    T = int(mask[orig_idx].sum())
    task_full = ep.get('task_name','?')
    tk = TASK_KEY_MAP.get(task_full, task_full[:20])
    st = ep.get('state_id','?')

    # stable/post_lock: after T_gform, gripper naturally open
    for wt in range(tg + 20, min(T - WINDOW_LEN, tg + 120), 5):
        key = (tk, st, str(wt), str(wt+WINDOW_LEN-1))
        if key in tested: continue
        gc = Xr[orig_idx, wt:wt+WINDOW_LEN, 0]
        oratio = float((gc < 0.5).sum()) / WINDOW_LEN
        if oratio > 0.3:
            qs = float(Xr[orig_idx, wt, 1])
            qm = float(Xr[orig_idx, wt:wt+WINDOW_LEN, 1].min())
            es = float(np.sqrt((Xr[orig_idx, wt:wt+WINDOW_LEN, 7:10]**2).sum(axis=1)).mean())
            controls['stable_post_lock'].append(dict(task_key=tk, state_id=st, window_start=wt, window_end=wt+WINDOW_LEN-1, phase_bin_proxy='stable_grasp_or_lift_proxy', candidate_role='stable_post_lock_control', clean_open_ratio=round(oratio,4), T_gform=tg, relative_lead=wt-tg, qpos_start=round(qs,6), qpos_min=round(qm,6), eef_speed_mean=round(es,6), actual_window_len=WINDOW_LEN, manual_selection_flag=True))
            break

    # far_too_early: far before grasp, CLOSED
    for wt in range(0, max(tg - 60, 5), 5):
        if wt + WINDOW_LEN >= tg: break
        key = (tk, st, str(wt), str(wt+WINDOW_LEN-1))
        if key in tested: continue
        gc = Xr[orig_idx, wt:wt+WINDOW_LEN, 0]
        oratio = float((gc < 0.5).sum()) / WINDOW_LEN
        if oratio <= 0.1:
            qs = float(Xr[orig_idx, wt, 1])
            qm = float(Xr[orig_idx, wt:wt+WINDOW_LEN, 1].min())
            es = float(np.sqrt((Xr[orig_idx, wt:wt+WINDOW_LEN, 7:10]**2).sum(axis=1)).mean())
            controls['far_too_early'].append(dict(task_key=tk, state_id=st, window_start=wt, window_end=wt+WINDOW_LEN-1, phase_bin_proxy='approach_far_closed_proxy', candidate_role='far_too_early_control', clean_open_ratio=round(oratio,4), T_gform=tg, relative_lead=wt-tg, qpos_start=round(qs,6), qpos_min=round(qm,6), eef_speed_mean=round(es,6), actual_window_len=WINDOW_LEN, manual_selection_flag=True))
            break

    # pre_lock: before grasp, CLOSED
    if tg >= 25:
        wt = tg - 20
        key = (tk, st, str(wt), str(wt+WINDOW_LEN-1))
        if key not in tested and wt + WINDOW_LEN < T:
            gc = Xr[orig_idx, wt:wt+WINDOW_LEN, 0]
            oratio = float((gc < 0.5).sum()) / WINDOW_LEN
            if oratio <= 0.1:
                qs = float(Xr[orig_idx, wt, 1])
                qm = float(Xr[orig_idx, wt:wt+WINDOW_LEN, 1].min())
                es = float(np.sqrt((Xr[orig_idx, wt:wt+WINDOW_LEN, 7:10]**2).sum(axis=1)).mean())
                controls['pre_lock'].append(dict(task_key=tk, state_id=st, window_start=wt, window_end=wt+WINDOW_LEN-1, phase_bin_proxy='pre_lock_closed_proxy', candidate_role='pre_lock_control', clean_open_ratio=round(oratio,4), T_gform=tg, relative_lead=wt-tg, qpos_start=round(qs,6), qpos_min=round(qm,6), eef_speed_mean=round(es,6), actual_window_len=WINDOW_LEN, manual_selection_flag=True))

# Select diverse per task
batch3c = []
for role in ['stable_post_lock','far_too_early','pre_lock']:
    pool = controls[role]
    pool.sort(key=lambda d: PRIORITY.index(d['task_key']) if d['task_key'] in PRIORITY else 99)
    seen, picked = set(), []
    for d in pool:
        if d['task_key'] not in seen: seen.add(d['task_key']); picked.append(d)
    targets = {'stable_post_lock':6,'far_too_early':4,'pre_lock':4}
    batch3c.extend(picked[:targets[role]])

rc = defaultdict(int); tc = defaultdict(int)
for d in batch3c: rc[d['candidate_role']] += 1; tc[d['task_key']] += 1
print('Batch3c: %d controls, %d tasks' % (len(batch3c), len(tc)))
print('Roles: %s' % dict(rc))
print('Tasks: %s' % dict(tc))

bfields = ['task_key','state_id','window_start','window_end','phase_bin_proxy','candidate_role',
           'clean_open_ratio','T_gform','relative_lead','qpos_start','qpos_min','eef_speed_mean',
           'actual_window_len','manual_selection_flag','reason_selected']
with open('tables/object_phase_response_batch3c_control_candidates.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=bfields, extrasaction='ignore')
    w.writeheader()
    for d in batch3c:
        d['reason_selected'] = 'batch3c_' + d['candidate_role']
        w.writerow(d)
print('Written batch3c_control_candidates.csv')
for d in batch3c:
    print('  %-25s %-15s s%-3s [%s,%s] lead=%s open=%s' % (d['candidate_role'], d['task_key'], d['state_id'], d['window_start'], d['window_end'], d['relative_lead'], d['clean_open_ratio']))
