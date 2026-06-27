#!/usr/bin/env python3
import json, os, csv, hashlib, shutil, random

random.seed(42)

SRC = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_table1/cqfr_blind'
os.makedirs(OUT, exist_ok=True)

runs = []
for cond in sorted(os.listdir(SRC)):
    cp = os.path.join(SRC, cond)
    if not os.path.isdir(cp):
        continue
    for run_dir in sorted(os.listdir(cp)):
        rp = os.path.join(cp, run_dir)
        video_path = os.path.join(rp, 'rollout_raw.mp4')
        summ_path = os.path.join(rp, 'episode_summary.json')
        if os.path.isfile(video_path) and os.path.isfile(summ_path):
            with open(summ_path) as f:
                summ = json.load(f)
            runs.append({
                'condition': cond,
                'run_dir': run_dir,
                'video_path': video_path,
                'task_success': summ.get('task_success', None),
                'task_name': summ.get('task_name', summ.get('task', '')),
                'state_id': summ.get('state_id', ''),
                'perturbation_seed': summ.get('perturbation_seed', ''),
                'arm_lock': summ.get('arm_lock', False),
                'objective_id': summ.get('objective_id', ''),
                'attack_frames': summ.get('attack_frames', 0),
            })

print('Total available: {}'.format(len(runs)))

# Stratified sampling: ensure coverage
strata = {}
for r in runs:
    key = (r['condition'], r['task_success'])
    strata.setdefault(key, []).append(r)

selected = []
# Target: 14 per condition = 56 total (trim to 55)
target_per_cond = 14
for cond_name in ['tma_nolock', 'tma_armlock', 'prefix_nolock', 'prefix_armlock']:
    pool = [r for r in runs if r['condition'] == cond_name]
    n = min(target_per_cond, len(pool))
    # Ensure balance of success/failure within each condition
    succ_pool = [r for r in pool if r['task_success']]
    fail_pool = [r for r in pool if not r['task_success']]
    n_succ = min(n // 3, len(succ_pool))
    n_fail = n - n_succ
    n_fail = min(n_fail, len(fail_pool))
    n_succ = n - n_fail
    n_succ = min(n_succ, len(succ_pool))
    selected.extend(random.sample(succ_pool, n_succ) if n_succ > 0 else [])
    selected.extend(random.sample(fail_pool, n_fail) if n_fail > 0 else [])

# Fill remaining slots to reach 55
if len(selected) < 55:
    already = {(r['run_dir'], r['condition']) for r in selected}
    remaining = [r for r in runs if (r['run_dir'], r['condition']) not in already]
    n_extra = 55 - len(selected)
    selected.extend(random.sample(remaining, min(n_extra, len(remaining))))

# Exactly 55
if len(selected) > 55:
    random.shuffle(selected)
    selected = selected[:55]

selected.sort(key=lambda r: (r['condition'], r['run_dir']))
print('Selected: {} videos'.format(len(selected)))

# Copy with opaque names, save blind key
blind_key = []
for i, r in enumerate(selected):
    blind_id = 'B{:04d}'.format(i + 1)
    src_video = r['video_path']
    dst_video = os.path.join(OUT, blind_id + '.mp4')

    shutil.copy2(src_video, dst_video)

    sha = hashlib.sha256()
    with open(dst_video, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)

    blind_key.append({
        'blind_id': blind_id,
        'source_run_key': r['run_dir'],
        'condition': r['condition'],
        'objective_id': r['objective_id'],
        'arm_lock': r['arm_lock'],
        'task': r['task_name'],
        'state_id': r['state_id'],
        'perturbation_seed': r['perturbation_seed'],
        'task_success': r['task_success'],
        'attack_frames': r['attack_frames'],
        'blind_video_sha256': sha.hexdigest(),
    })
    print('  {} -> {} ({}, succ={}, lock={})'.format(
        blind_id, r['run_dir'][:45], r['condition'], r['task_success'], r['arm_lock']))

# PRIVATE blind key
key_path = os.path.join(OUT, 'CQFR_BLIND_KEY_PRIVATE.csv')
with open(key_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(blind_key[0].keys()))
    w.writeheader()
    w.writerows(blind_key)
print('Private key: {}'.format(key_path))

# PUBLIC reviewer template — NO LEAKS
reviewer_path = os.path.join(OUT, 'CQFR_REVIEWER_TEMPLATE.csv')
with open(reviewer_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['blind_id', 'video_path', 'review_label', 'confidence', 'notes'])
    for bk in blind_key:
        w.writerow([bk['blind_id'], bk['blind_id'] + '.mp4', '', '', ''])
print('Reviewer template: {}'.format(reviewer_path))

# Summary
succ = sum(1 for bk in blind_key if bk['task_success'])
fail = sum(1 for bk in blind_key if not bk['task_success'])
al = sum(1 for bk in blind_key if bk['arm_lock'])
nl = sum(1 for bk in blind_key if not bk['arm_lock'])
print()
print('=== CQFR PACKAGE ===')
print('Videos: {} ({} success, {} failure, {} armlock, {} nolock)'.format(len(blind_key), succ, fail, al, nl))
for cond in sorted(set(bk['condition'] for bk in blind_key)):
    c = sum(1 for bk in blind_key if bk['condition'] == cond)
    print('  {}: {}'.format(cond, c))
print()
print('IMPORTANT: CQFR_BLIND_KEY_PRIVATE.csv must NOT be shared with reviewers.')
print('Reviewers receive ONLY: B0001.mp4 ... B{:04d}.mp4 + CQFR_REVIEWER_TEMPLATE.csv'.format(len(blind_key)))
