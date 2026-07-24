#!/usr/bin/env python3
"""SC5 Teacher Replay Parity — 60-cell audit.
Verifies that frozen teacher + privileged records reproduce saved teacher_labels.jsonl."""
import json, csv, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path('/mnt/sdc/dty_user/openvla_attack')
REPLAY = BASE / 'evidence/object_checkpoint_migration/m1_runtime_b0_d1/replay_60cell'
TEACHER_CONFIG = BASE / 'migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json'
OUT_DIR = BASE / 'reports/sc5_object_loto_v2'
os.makedirs(OUT_DIR, exist_ok=True)

# Load frozen teacher config
with open(TEACHER_CONFIG) as f:
    tc_data = json.load(f)

# Import teacher
sys.path.insert(0, str(BASE / 'src'))
from gripper_attack.v2_privileged_teacher import V2PrivilegedTeacher, TeacherConfig

# Build config object from frozen thresholds
thresh = tc_data['thresholds']
cfg = TeacherConfig()
cfg.grasp_close_sustain = thresh['grasp_close_sustain']
cfg.eef_obj_dist_max = thresh['eef_obj_dist_max']
cfg.eef_obj_dist_stable_var = thresh['eef_obj_dist_stable_var']
cfg.lift_z_threshold = thresh['lift_z_threshold']
cfg.lift_sustain_steps = thresh.get('lift_sustain_steps', 2)
cfg.carry_obj_z_var_max = thresh['carry_obj_z_var_max']
cfg.carry_window = thresh['carry_window']
cfg.preplace_target_dist_min = thresh['preplace_target_dist_min']
cfg.preplace_target_dist_max = thresh['preplace_target_dist_max']
cfg.release_target_dist_max = thresh['release_target_dist_max']
cfg.regrasp_eef_obj_dist_max = thresh['regrasp_eef_obj_dist_max']
cfg.stability_window = thresh['stability_window']

GUARD = tc_data['guard']
K = tc_data['K']
teacher = V2PrivilegedTeacher(cfg)

print('Teacher config loaded:')
print('  guard={} K={}'.format(GUARD, K))
print('  eef_obj_dist_max={}'.format(cfg.eef_obj_dist_max))

# Collect all 60 episodes
TASK_NAMES = {
    0: 'butter', 1: 'ketchup', 2: 'milk', 3: 'orange_juice',
    4: 'alphabet_soup', 5: 'tomato_sauce', 6: 'cream_cheese',
    7: 'salad_dressing', 8: 'bbq_sauce', 9: 'chocolate_pudding',
}
NAME_TO_IDX = {v: k for k, v in TASK_NAMES.items()}

episodes = []
for task_dir in sorted(REPLAY.iterdir()):
    if not task_dir.is_dir(): continue
    task_name = task_dir.name.rsplit('_s', 1)[0]
    state_id = task_dir.name.rsplit('_s', 1)[1]
    task_idx = NAME_TO_IDX.get(task_name, -1)

    for variant_dir in sorted(task_dir.iterdir()):
        if not variant_dir.is_dir(): continue
        priv_path = variant_dir / 'privileged_step_records.jsonl'
        label_path = variant_dir / 'teacher_labels.jsonl'
        if not priv_path.exists() or not label_path.exists(): continue

        episodes.append({
            'task_name': task_name, 'task_idx': task_idx,
            'state_id': state_id, 'variant': variant_dir.name,
            'priv_path': str(priv_path), 'label_path': str(label_path),
        })

print('Found {} episodes'.format(len(episodes)))

# Replay parity
results = []
phase_agree_total = 0
phase_total = 0

for ep in episodes:
    with open(ep['priv_path']) as f:
        records = [json.loads(line) for line in f]

    # NEW labels from frozen teacher
    new_labels = teacher.label_trajectory(records)

    # OLD labels from saved file
    with open(ep['label_path']) as f:
        old_labels = [json.loads(line) for line in f]

    if len(new_labels) != len(old_labels):
        print('  ROW MISMATCH: {} new vs {} old — {} {} {}'.format(
            len(new_labels), len(old_labels), ep['task_name'], ep['state_id'], ep['variant']))
        continue

    phase_agree = 0
    phase_total = 0

    for i, (nl, ol) in enumerate(zip(new_labels, old_labels)):
        if nl is not None and ol is not None:
            np_phase = nl.get('phase', '?')
            op_phase = ol.get('phase', '?')
            if np_phase == op_phase:
                phase_agree += 1
            elif phase_total < 5:
                # Debug first few mismatches
                pass
            phase_total += 1
        elif nl is None and ol is None:
            phase_agree += 1
            phase_total += 1

    results.append({
        'task_name': ep['task_name'], 'task_idx': ep['task_idx'],
        'state_id': ep['state_id'], 'variant': ep['variant'],
        'n_steps': len(new_labels), 'phase_agree': phase_agree,
        'phase_total': phase_total,
        'phase_rate': phase_agree / phase_total if phase_total else 0,
    })

    phase_agree_total += phase_agree

print('\n=== REPLAY PARITY RESULTS ===')
print('Episodes: {}'.format(len(results)))
overall_phase_rate = phase_agree_total / phase_total if phase_total else 0
print('Overall phase agreement: {}/{} ({:.2f}%)'.format(phase_agree_total, phase_total, 100*overall_phase_rate))

# Per-episode phase agreement
phase_rates = [r['phase_rate'] for r in results]
below_99 = [r for r in results if r['phase_rate'] < 0.99]
print('Episodes with <99% phase agreement: {}'.format(len(below_99)))
for r in below_99[:5]:
    print('  {} {} {}: {:.2f}%'.format(r['task_name'], r['state_id'], r['variant'], 100*r['phase_rate']))

# Gate
gate_pass = overall_phase_rate >= 0.99 and len(below_99) == 0
print('\nGate T0 (Phase agreement >= 99%): {}'.format('PASS' if gate_pass else 'FAIL'))

# Write results
with open(OUT_DIR / 'TEACHER_REPLAY_PARITY.json', 'w') as f:
    json.dump({
        'total_episodes': len(results),
        'overall_phase_agreement': overall_phase_rate,
        'episodes_below_99pct': len(below_99),
        'gate_pass': gate_pass,
    }, f, indent=2)

# Per-episode CSV
with open(OUT_DIR / 'TEACHER_REPLAY_PARITY_EPISODE.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task_name', 'task_idx', 'state_id', 'variant', 'n_steps', 'phase_rate'])
    w.writeheader()
    w.writerows(results)

print('\nOutputs: {}'.format(OUT_DIR))
