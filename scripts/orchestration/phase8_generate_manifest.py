#!/usr/bin/env python3
"""Generate immutable Phase 8 job manifest for cross-suite generalization."""
import json, hashlib, os, sys
from pathlib import Path

OUT = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase8_cross_suite_v1/manifests')
OUT.mkdir(parents=True, exist_ok=True)

SUITES = {
    'libero_spatial': {'n_tasks': 10, 'max_steps': 400, 'available': True},
    'libero_goal':    {'n_tasks': 10, 'max_steps': 400, 'available': False},
    'libero_long':    {'n_tasks': 10, 'max_steps': 500, 'available': False},
}

CONDITIONS = [
    ('CLEAN',          False, '',                                               False),
    ('RANDOM',         True,  '',                                               False),
    ('UNTARGETED_CE',  True,  'untargeted_clean_token_ce',                      False),
    ('TMA_NOLOCK',     True,  'vanilla_tma_gripper_open_ce',                    False),
    ('TMA_ARMLOCK',    True,  'vanilla_tma_gripper_open_ce',                    True),
    ('PREFIX_NOLOCK',  True,  'autoregressive_prefix_gripper_target_token_logratio_arm_v3', False),
    ('PREFIX_ARMLOCK', True,  'autoregressive_prefix_gripper_target_token_logratio_arm_v3', True),
]

SEEDS = [42, 123, 456]
PROTOCOL = {
    'epsilon': 0.023529411764705882, 'pgd_steps': 20, 'K': 10,
    'target_token': 31744, 'eval_seed': 0, 'teacher_anchor_valid': False,
}

jobs = []
job_id = 0

for suite_name, suite_cfg in SUITES.items():
    if not suite_cfg['available']:
        continue
    for task_idx in range(suite_cfg['n_tasks']):
        for seed in SEEDS:
            # CLEAN job first
            clean_jid = f'p8_{suite_name}_t{task_idx:02d}_e{seed}_clean'
            job_id += 1
            jobs.append({
                'job_id': clean_jid, 'phase': 'P2', 'suite': suite_name,
                'task_idx': task_idx, 'evaluation_seed': seed,
                'condition': 'CLEAN', 'arm_lock': False,
                'attack_enabled': False, 'objective_id': '',
                'max_env_steps': suite_cfg['max_steps'],
                'parent_clean_job': None, **PROTOCOL,
            })

            # 6 attack conditions
            for cond_name, attack_enabled, objective, arm_lock in CONDITIONS[1:]:
                jid = f'p8_{suite_name}_t{task_idx:02d}_e{seed}_{cond_name.lower()}'
                job_id += 1
                jobs.append({
                    'job_id': jid, 'phase': 'P3' if 'ARMLOCK' not in cond_name else 'P4',
                    'suite': suite_name, 'task_idx': task_idx, 'evaluation_seed': seed,
                    'condition': cond_name, 'arm_lock': arm_lock,
                    'attack_enabled': attack_enabled, 'objective_id': objective,
                    'max_env_steps': suite_cfg['max_steps'],
                    'parent_clean_job': clean_jid, **PROTOCOL,
                })

# Write manifest
manifest_path = OUT / 'ALL_SPATIAL_210_JOBS.jsonl'
with open(manifest_path, 'w') as f:
    for job in jobs:
        f.write(json.dumps(job) + '\n')

manifest_sha = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
with open(OUT / 'ALL_SPATIAL_210_JOBS.sha256', 'w') as f:
    f.write(f'{manifest_sha}  ALL_SPATIAL_210_JOBS.jsonl\n')

# Phase sub-manifests
for phase in ['P2', 'P3', 'P4']:
    phase_jobs = [j for j in jobs if j['phase'] == phase]
    phase_path = OUT / f'{phase}_SPATIAL_{len(phase_jobs)}.jsonl'
    with open(phase_path, 'w') as f:
        for job in phase_jobs:
            f.write(json.dumps(job) + '\n')

# P1 smoke: 1 task × 1 seed × 7 conditions = 7 per suite
smoke_jobs = [j for j in jobs if j['task_idx'] == 0 and j['evaluation_seed'] == 42]
smoke_path = OUT / 'P1_SMOKE_7.jsonl'
with open(smoke_path, 'w') as f:
    for job in smoke_jobs:
        f.write(json.dumps(job) + '\n')

print(f'Total jobs: {len(jobs)} (Spatial only)')
print(f'  P1 smoke: {len(smoke_jobs)}')
print(f'  P2 CLEAN: {len([j for j in jobs if j["condition"]=="CLEAN"])}')
print(f'  P3 core: {len([j for j in jobs if j["phase"]=="P3"])}')
print(f'  P4 armlock: {len([j for j in jobs if j["phase"]=="P4"])}')
print(f'Manifest SHA256: {manifest_sha}')
print(f'Manifest: {manifest_path}')
