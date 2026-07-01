"""Fix COMMAND_OPEN_ORACLE manifest: condition labels + generate 9-fold canary."""
import json, os
from collections import defaultdict

ORACLE_MF = '/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/COMMAND_OPEN_ORACLE_T10/launch'
OUT_MF = '/mnt/sdc/dty_user/table1_sota_execution_v1/manifests'
CANARY_DIR = '/mnt/sdc/dty_user/table1_sota_execution_v1/canary/oracle'

os.makedirs(OUT_MF, exist_ok=True)
os.makedirs(CANARY_DIR, exist_ok=True)

# Read all existing jobs
all_jobs = []
for mf in sorted(os.listdir(ORACLE_MF)):
    if not mf.endswith('.jsonl'): continue
    for line in open(os.path.join(ORACLE_MF, mf)):
        j = json.loads(line.strip())
        # Fix condition labels
        j['condition'] = 'COMMAND_OPEN_ORACLE'
        j['condition_id'] = 'COMMAND_OPEN_ORACLE'
        j['attack_objective'] = 'oracle_env_gripper_open'
        j['objective_id'] = 'oracle_env_gripper_open'
        # Remove oracle flag (not needed now that condition is explicit)
        j.pop('oracle', None)
        all_jobs.append(j)

print(f'Total jobs: {len(all_jobs)}')

# Split into 8 GPU manifests
gpu_jobs = defaultdict(list)
for i, j in enumerate(all_jobs):
    gpu_jobs[i % 8].append(j)

for gpu in range(8):
    mf_path = os.path.join(OUT_MF, f'manifest_gpu{gpu}.jsonl')
    with open(mf_path, 'w') as f:
        for j in gpu_jobs[gpu]:
            f.write(json.dumps(j) + '\n')
    print(f'GPU {gpu}: {len(gpu_jobs[gpu])} jobs -> {mf_path}')

# Generate 9-fold canary (one job per fold, state_id=0)
canary = []
for fold_int in range(1, 10):
    fold = f'{fold_int:02d}'
    for j in all_jobs:
        if j['fold'] == fold:
            canary.append(j)
            break
    else:
        print(f'WARNING: No canary candidate for fold {fold}')

print(f'\nCanary: {len(canary)} jobs across 9 folds')

# Write canary manifest as single manifest for GPU 6
canary_path = os.path.join(CANARY_DIR, 'manifest_canary.jsonl')
with open(canary_path, 'w') as f:
    for j in canary:
        f.write(json.dumps(j) + '\n')
print(f'Canary manifest: {canary_path}')

# Verify fold coverage
folds_covered = {j['fold'] for j in canary}
print(f'Folds covered: {sorted(folds_covered)}')
assert len(folds_covered) == 9, f'Expected 9 folds, got {len(folds_covered)}'

# Write manifest SHA
manifest_files = {}
for fn in sorted(os.listdir(OUT_MF)):
    if fn.endswith('.jsonl'):
        import hashlib
        sha = hashlib.sha256(open(os.path.join(OUT_MF, fn), 'rb').read()).hexdigest()[:16]
        manifest_files[fn] = sha

print(f'\nManifest SHAs: {json.dumps(manifest_files, indent=2)}')
print('Done. Manifests ready for execution.')
