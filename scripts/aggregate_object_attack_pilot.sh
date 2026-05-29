#!/bin/bash
# Aggregate Object attack pilot results
set -e
OUT=/data/liuyu/outputs/milestone_2f_object_detector_matched_attack_pilot_20260529
PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python

echo "=== Aggregating attack pilot ==="

# 1. Aggregate artifact-rich manifests
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
${PY} scripts/aggregate_artifact_rich_manifests.py \
  --output_root ${OUT} \
  --overwrite

# 2. Per-condition summary
${PY} -c "
import csv, json, os, glob
from collections import defaultdict

OUT = '${OUT}'
runs_dir = os.path.join(OUT, 'runs', 'libero_object')
if not os.path.isdir(runs_dir):
    print('No runs dir found')
    exit(0)

# Collect per-episode results
results = []
for ep_dir in sorted(glob.glob(os.path.join(runs_dir, '*'))):
    manifest_path = os.path.join(ep_dir, 'run_manifest.json')
    step_path = os.path.join(ep_dir, 'step_records.jsonl')
    if not os.path.exists(manifest_path):
        continue
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Extract condition from run_id
    run_id = manifest.get('run_id', '')
    condition = 'unknown'
    for cond in ['clean', 'oracle_open', 'random_control', 'VIS_targeted']:
        if cond in run_id:
            condition = cond; break

    # Count triggers and attacks from step_records
    n_triggers = 0; n_attacks = 0; n_steps = 0
    if os.path.exists(step_path):
        with open(step_path) as f:
            for line in f:
                n_steps += 1
                try:
                    rec = json.loads(line)
                    if rec.get('detector_trigger_now'): n_triggers += 1
                    if rec.get('attack_applied'): n_attacks += 1
                except: pass

    results.append({
        'task_name': manifest.get('task_name', ''),
        'state_id': manifest.get('state_id', ''),
        'condition': condition,
        'success': manifest.get('success', False),
        'n_steps': n_steps,
        'n_triggers': n_triggers,
        'n_attacks': n_attacks,
    })

# Write summary
with open(os.path.join(OUT, 'tables', 'attack_pilot_per_episode_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)

# Per-condition aggregate
by_cond = defaultdict(lambda: {'n': 0, 'success': 0, 'total_triggers': 0, 'total_attacks': 0})
for r in results:
    c = by_cond[r['condition']]
    c['n'] += 1; c['success'] += int(r['success'])
    c['total_triggers'] += r['n_triggers']
    c['total_attacks'] += r['n_attacks']

print()
print('=== PER-CONDITION SUMMARY ===')
for cond in ['clean', 'oracle_open', 'random_control', 'VIS_targeted']:
    c = by_cond.get(cond, {'n':0,'success':0,'total_triggers':0,'total_attacks':0})
    print(f'{cond}: SR={c[\"success\"]}/{c[\"n\"]} ({c[\"success\"]/max(1,c[\"n\"]):.2f}) triggers={c[\"total_triggers\"]} attacks={c[\"total_attacks\"]}')

with open(os.path.join(OUT, 'tables', 'attack_pilot_condition_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['condition', 'n', 'success', 'sr', 'total_triggers', 'total_attacks'])
    w.writeheader()
    for cond in ['clean', 'oracle_open', 'random_control', 'VIS_targeted']:
        c = by_cond.get(cond, {'n':0,'success':0,'total_triggers':0,'total_attacks':0})
        w.writerow({'condition': cond, 'n': c['n'], 'success': c['success'],
            'sr': c['success']/max(1,c['n']), 'total_triggers': c['total_triggers'],
            'total_attacks': c['total_attacks']})

print(f'\\nOutput: {OUT}/tables/')
"

echo "Aggregation complete"
