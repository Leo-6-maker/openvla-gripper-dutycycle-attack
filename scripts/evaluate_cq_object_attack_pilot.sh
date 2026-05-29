#!/bin/bash
# CQ evaluation for Object attack pilot
set -e
OUT=/data/liuyu/outputs/milestone_2f_object_detector_matched_attack_pilot_20260529
PY=/data/aviary/envs/openvla_official_libero_20260525/bin/python

echo "=== CQ Evaluation ==="

${PY} -c "
import csv, json, os, glob
from collections import defaultdict

OUT = '${OUT}'
runs_dir = os.path.join(OUT, 'runs', 'libero_object')
if not os.path.isdir(runs_dir):
    print('No runs dir'); exit(0)

# For each episode, compute CQ metrics from step_records
# Using existing infer_failure_phase from grasp.py
import sys
sys.path.insert(0, '/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524')
sys.path.insert(0, '/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524')
from gripper_attack.grasp import infer_failure_phase

results = []
for ep_dir in sorted(glob.glob(os.path.join(runs_dir, '*'))):
    manifest_path = os.path.join(ep_dir, 'run_manifest.json')
    step_path = os.path.join(ep_dir, 'step_records.jsonl')
    if not os.path.exists(manifest_path):
        continue

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load step records
    steps = []
    if os.path.exists(step_path):
        with open(step_path) as f:
            for line in f:
                try: steps.append(json.loads(line))
                except: pass

    run_id = manifest.get('run_id', '')
    condition = 'unknown'
    for cond in ['clean', 'oracle_open', 'random_control', 'VIS_targeted']:
        if cond in run_id: condition = cond; break

    success = manifest.get('success', False)
    failure_phase = infer_failure_phase(steps, success)

    # Gripper stats
    gripper_qpos_vals = [s.get('gripper_qpos', 0) for s in steps if s.get('gripper_qpos') not in ('', None)]
    max_qpos = max(gripper_qpos_vals) if gripper_qpos_vals else 0
    max_open_streak = 0; current_streak = 0
    for q in gripper_qpos_vals:
        if q > 0.5: current_streak += 1; max_open_streak = max(max_open_streak, current_streak)
        else: current_streak = 0

    # Trigger/attack stats
    n_triggers = sum(1 for s in steps if s.get('detector_trigger_now'))
    n_attacks = sum(1 for s in steps if s.get('attack_applied'))

    results.append({
        'task_name': manifest.get('task_name', ''),
        'state_id': manifest.get('state_id', ''),
        'condition': condition,
        'official_success': success,
        'failure_phase': failure_phase,
        'cq_failure': failure_phase not in ('success_libero', 'no_grasp', 'unknown'),
        'max_gripper_qpos': max_qpos,
        'max_open_streak': max_open_streak,
        'n_steps': len(steps),
        'n_triggers': n_triggers,
        'n_attacks': n_attacks,
    })

with open(os.path.join(OUT, 'tables', 'cq_evaluation_per_episode.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)

# Per-condition CQ metrics
by_cond = defaultdict(lambda: {'n': 0, 'official_sr': 0, 'cq_failures': 0, 'cq_successes': 0})
for r in results:
    c = by_cond[r['condition']]
    c['n'] += 1
    c['official_sr'] += int(r['official_success'])
    if r['cq_failure']: c['cq_failures'] += 1
    if not r['cq_failure']: c['cq_successes'] += 1

print()
print('=== CQ BY CONDITION ===')
print(f"{"Condition":<20} {"SR":>5} {"CQFR":>6} {"CQSR":>6} {"Mismatch":>9} {"Gripper":>8} {"Streak":>6}")
for cond in ['clean', 'oracle_open', 'random_control', 'VIS_targeted']:
    c = by_cond.get(cond)
    if not c or c['n'] == 0: continue
    sr = c['official_sr'] / c['n']
    cqfr = c['cq_failures'] / c['n']
    cqsr = c['cq_successes'] / c['n']
    mismatch = sr - cqsr
    cond_results = [r for r in results if r['condition'] == cond]
    avg_qpos = sum(r['max_gripper_qpos'] for r in cond_results) / len(cond_results)
    avg_streak = sum(r['max_open_streak'] for r in cond_results) / len(cond_results)
    print(f'{cond:<20} {sr:>5.2f} {cqfr:>6.3f} {cqsr:>6.3f} {mismatch:>9.3f} {avg_qpos:>8.4f} {avg_streak:>6.1f}')

with open(os.path.join(OUT, 'tables', 'cq_condition_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['condition', 'n', 'official_sr', 'CQFR', 'CQSR', 'sr_cq_mismatch', 'avg_max_gripper_qpos', 'avg_max_open_streak'])
    w.writeheader()
    for cond in ['clean', 'oracle_open', 'random_control', 'VIS_targeted']:
        c = by_cond.get(cond)
        if not c or c['n'] == 0: continue
        cond_results = [r for r in results if r['condition'] == cond]
        avg_qpos = sum(r['max_gripper_qpos'] for r in cond_results) / len(cond_results)
        avg_streak = sum(r['max_open_streak'] for r in cond_results) / len(cond_results)
        w.writerow({'condition': cond, 'n': c['n'], 'official_sr': c['official_sr']/c['n'],
            'CQFR': c['cq_failures']/c['n'], 'CQSR': c['cq_successes']/c['n'],
            'sr_cq_mismatch': c['official_sr']/c['n'] - c['cq_successes']/c['n'],
            'avg_max_gripper_qpos': avg_qpos, 'avg_max_open_streak': avg_streak})

print(f'\\nOutput: {OUT}/tables/')
"

echo "CQ evaluation complete"
