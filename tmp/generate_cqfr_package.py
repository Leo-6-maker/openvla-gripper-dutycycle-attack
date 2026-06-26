#!/usr/bin/env python3
"""Generate CQFR blind review package from metric refresh v2 runs."""
import json, os, csv, hashlib, random
from pathlib import Path

random.seed(42)

METRIC_DIR = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2')
OUT_DIR = Path('/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/cqfr')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect all runs
runs = []
for tag_dir in sorted(METRIC_DIR.iterdir()):
    if not tag_dir.is_dir(): continue
    condition = tag_dir.name
    for cell_dir in sorted(tag_dir.iterdir()):
        if not cell_dir.is_dir(): continue
        summary_f = cell_dir / 'episode_summary.json'
        complete_f = cell_dir / 'COMPLETE.json'
        video_f = cell_dir / 'rollout_raw.mp4'
        if not summary_f.exists() or not complete_f.exists():
            continue
        with open(summary_f) as f:
            s = json.load(f)

        parts = condition.split('_')
        objective = 'TMA' if 'tma' in condition else 'Prefix'
        arm_lock = 'armlock' in condition

        task_success = s.get('task_success', None)
        token_duty = s.get('token_open_duty', 0)
        tasr_episode = 1 if token_duty >= 0.8 else 0

        runs.append({
            'condition': condition, 'objective': objective, 'arm_lock': arm_lock,
            'cell': cell_dir.name, 'task_success': task_success,
            'token_duty': token_duty, 'tasr_episode': tasr_episode,
            'video_path': str(video_f) if video_f.exists() else '',
            'video_sha': s.get('video', {}).get('video_sha256', '')[:16],
        })

print(f'Total runs: {len(runs)}')

# Statistics
all_success = [r for r in runs if r['task_success'] is True]
all_failure = [r for r in runs if r['task_success'] is False]
print(f'Success: {len(all_success)}, Failure: {len(all_failure)}')

# Per-condition FR
from collections import defaultdict
cond_counts = defaultdict(lambda: {'total': 0, 'fail': 0})
for r in runs:
    cond_counts[r['condition']]['total'] += 1
    if r['task_success'] is False:
        cond_counts[r['condition']]['fail'] += 1

print('\nPer-condition FR:')
for cond in sorted(cond_counts):
    c = cond_counts[cond]
    print(f'  {cond}: {c["fail"]}/{c["total"]} = {c["fail"]/c["total"]:.3f}')

# Selection
selected_keys = set()

# 1. All successes with TASR=1
for r in runs:
    if r['task_success'] is True and r['tasr_episode'] == 1:
        selected_keys.add(r['cell'] + '|' + r['condition'])
print(f'\n1. Success + TASR=1: {len(selected_keys)}')

# 2. TMA-vs-Prefix discordant pairs
for r_tma in [r for r in runs if r['objective'] == 'TMA']:
    r_prefix = next((r for r in runs if r['cell'] == r_tma['cell']
                     and r['objective'] == 'Prefix' and r['arm_lock'] == r_tma['arm_lock']), None)
    if r_prefix and r_tma['task_success'] != r_prefix['task_success']:
        selected_keys.add(r_tma['cell'] + '|' + r_tma['condition'])
        selected_keys.add(r_prefix['cell'] + '|' + r_prefix['condition'])
print(f'2. TMA-vs-Prefix discordant: total {len(selected_keys)}')

# 3. Lock-vs-no-lock discordant pairs
for r_lock in [r for r in runs if r['arm_lock']]:
    r_nolock = next((r for r in runs if r['cell'] == r_lock['cell']
                     and r['objective'] == r_lock['objective'] and not r['arm_lock']), None)
    if r_nolock and r_lock['task_success'] != r_nolock['task_success']:
        selected_keys.add(r_lock['cell'] + '|' + r_lock['condition'])
        selected_keys.add(r_nolock['cell'] + '|' + r_nolock['condition'])
print(f'3. Lock-vs-no-lock discordant: total {len(selected_keys)}')

# 4. 20% random sample per method
for obj in ['TMA', 'Prefix']:
    obj_runs = [r for r in runs if r['objective'] == obj]
    remaining = [r for r in obj_runs if r['cell'] + '|' + r['condition'] not in selected_keys]
    sample_n = max(2, int(len(obj_runs) * 0.20))
    sampled = random.sample(remaining, min(sample_n, len(remaining)))
    for r in sampled:
        selected_keys.add(r['cell'] + '|' + r['condition'])
print(f'4. 20% sample: total {len(selected_keys)}')

# Build package
package = []
for r in runs:
    key = r['cell'] + '|' + r['condition']
    if key in selected_keys:
        blind_id = hashlib.sha256(key.encode()).hexdigest()[:8]
        r['blind_id'] = blind_id
        package.append(r)

# Write CSV
csv_path = OUT_DIR / 'CQFR_REVIEW_PACKAGE.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['blind_id', 'video_path', 'cell', 'condition', 'objective', 'arm_lock', 'task_success', 'token_duty', 'video_sha'], extrasaction='ignore')
    w.writeheader()
    for r in sorted(package, key=lambda x: x['blind_id']):
        w.writerow(r)
print(f'\nReview package: {len(package)} videos -> {csv_path}')

# Blind mapping (for later unmasking)
blind_map_path = OUT_DIR / 'CQFR_BLIND_MAP.json'
with open(blind_map_path, 'w') as f:
    mapping = {r['blind_id']: {'cell': r['cell'], 'condition': r['condition'], 'objective': r['objective'], 'arm_lock': r['arm_lock'], 'task_success': r['task_success']} for r in package}
    json.dump(mapping, f, indent=2)
print(f'Blind map: {blind_map_path}')

# Statistics
print(f'\nPackage breakdown:')
for obj in ['TMA', 'Prefix']:
    n = len([r for r in package if r['objective'] == obj])
    print(f'  {obj}: {n}')
for lock_val, label in [(False, 'no_lock'), (True, 'arm_lock')]:
    n = len([r for r in package if r['arm_lock'] == lock_val])
    print(f'  {label}: {n}')

print(f'\nBlind map (keep secret from reviewers!): {blind_map_path}')
print('Reviewers use: CQFR_REVIEW_PACKAGE.csv (blind_id, video_path, cell, token_duty only)')
print('Done.')
