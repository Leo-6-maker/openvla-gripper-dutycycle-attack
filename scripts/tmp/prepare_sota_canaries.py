"""Prepare 9-fold canary manifests for TMA Student, TMA Random-Time, UMA, SHUFFLED.

Each canary = 9 jobs (one per fold, state_id=0 when available, else first state in fold).
"""
import json, os, sys
from collections import defaultdict

EVID = '/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1'
SOTA_MF = '/mnt/sdc/dty_user/table1_sota_execution_v1/manifests'
CANARY = '/mnt/sdc/dty_user/table1_sota_execution_v1/canary'
os.makedirs(SOTA_MF, exist_ok=True)
os.makedirs(CANARY, exist_ok=True)

# ── Load TRUE_T10 emission data ──
TRUE_T10 = EVID + '/TRUE_T10/formal_v1'
tt_emissions = {}  # key -> emit_step
tt_valid = {}  # key -> n_valid_steps
for fold in sorted(os.listdir(TRUE_T10)):
    fp = os.path.join(TRUE_T10, fold)
    if not os.path.isdir(fp): continue
    fold_id = fold.split('_')[1]
    for sd in sorted(os.listdir(fp)):
        sid = int(sd.split('_')[1])
        sp = os.path.join(fp, sd)
        for dd in sorted(os.listdir(sp)):
            did = int(dd.split('_')[2])
            dp = os.path.join(sp, dd)
            for pd in sorted(os.listdir(dp)):
                pid = int(pd.split('_')[2])
                ep = os.path.join(dp, pd, 'episode_summary.json')
                if os.path.exists(ep):
                    d = json.load(open(ep))
                    key = (fold_id, sid, did, pid)
                    tt_emissions[key] = d.get('mlp_emit_step', -1)
                    tt_valid[key] = d.get('n_steps', 0) - d.get('invalid_feature_steps', 0)

print(f"TRUE_T10: {len(tt_emissions)} episodes ({sum(1 for v in tt_emissions.values() if v>=0)} emission)")

# ── Helper: generate random triggers ──
import numpy as np
def random_triggers(valid_steps_list, K=10, guard=5, seed=42):
    rng = np.random.RandomState(seed)
    triggers = []
    for ns in valid_steps_list:
        if ns >= guard + K:
            triggers.append(int(rng.randint(guard, ns - K + 1)))
        else:
            triggers.append(None)
    return triggers

# ── Build condition specs ──
CONDITIONS = {
    "TMA_STUDENT": {
        "condition_id": "TRUE_T10",
        "condition": "TRUE_T10",
        "output_namespace": "TMA",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "timing": "student_trigger",
        "description": "Adapted TMA + Student Trigger",
    },
    "TMA_RANDOM_TIME": {
        "condition_id": "TRUE_T10",
        "condition": "TRUE_T10",
        "output_namespace": "TMA_RANDOM_TIME",
        "attack_objective": "vanilla_tma_gripper_open_ce",
        "timing": "random_trigger",
        "description": "Adapted TMA + Random-Time",
    },
    "UMA_STUDENT": {
        "condition_id": "TRUE_T10",
        "condition": "TRUE_T10",
        "output_namespace": "UMA",
        "attack_objective": "untargeted_clean_token_ce",
        "timing": "student_trigger",
        "description": "UMA Untargeted CE-PGD + Student Trigger",
    },
    "SHUFFLED_STUDENT": {
        "condition_id": "SHUFFLED_T10",
        "condition": "SHUFFLED_T10",
        "output_namespace": "SHUFFLED",
        "attack_objective": None,
        "timing": "student_trigger",
        "description": "Shuffled Gradient + Student Trigger",
    },
}

# ── Load source TRUE_T10 manifest for reference ──
src_mf = EVID + '/TRUE_T10/formal_v1/manifest.jsonl'
if not os.path.exists(src_mf):
    # No single manifest — reconstruct from individual jobs
    src_mf = EVID + '/RANDOM_TIME_INVALID_V1/original_manifest.jsonl'
if not os.path.exists(src_mf):
    print(f"ERROR: No source manifest found")
    sys.exit(1)

src_jobs = [json.loads(l) for l in open(src_mf)]
print(f"Source manifest: {len(src_jobs)} jobs")

# Verify 162 jobs
if len(src_jobs) != 162:
    print(f"ERROR: Expected 162 source jobs, got {len(src_jobs)}")
    sys.exit(1)

# ── Generate manifests for each SOTA condition ──
for cond_name, spec in CONDITIONS.items():
    print(f"\n{'='*50}")
    print(f"{cond_name}: {spec['description']}")

    output_ns = spec['output_namespace']
    cond_id = spec['condition_id']

    # Compute triggers
    timing = spec['timing']
    if timing == 'student_trigger':
        # Use TRUE_T10 emission steps
        triggers = {}
        for j in src_jobs:
            key = (j['fold'], j['state_id'], j['detector_seed'], j['perturbation_seed'])
            triggers[key] = tt_emissions.get(key, -1)
    elif timing == 'random_trigger':
        # Generate random windows
        keys_ordered = [(j['fold'], j['state_id'], j['detector_seed'], j['perturbation_seed']) for j in src_jobs]
        valid = [tt_valid.get(k, 400) for k in keys_ordered]
        raw_triggers = random_triggers(valid, K=10, guard=5, seed=42)
        triggers = {k: (t if t is not None else -1) for k, t in zip(keys_ordered, raw_triggers)}
    else:
        triggers = {}

    # Build jobs
    jobs = []
    for j in src_jobs:
        key = (j['fold'], j['state_id'], j['detector_seed'], j['perturbation_seed'])
        ts = triggers.get(key, -1)

        nj = dict(j)
        nj['condition_id'] = cond_id
        nj['condition'] = spec.get('condition', cond_id)
        if spec.get('attack_objective'):
            nj['attack_objective'] = spec['attack_objective']
            nj['objective_id'] = spec['attack_objective']
        nj['trigger_step_override'] = ts
        nj['job_key'] = j['job_key'].replace('RANDOM_TIME', output_ns).replace('TRUE_T10', output_ns)
        nj['output_dir'] = j['output_dir'].replace('RANDOM_TIME', output_ns).replace('TRUE_T10', output_ns)
        nj['bridge_condition'] = spec.get('bridge_condition', cond_id)
        if 'source_true_t10_job_key' not in nj:
            nj['source_true_t10_job_key'] = j.get('job_key', '')
        if 'n_valid_steps' not in nj:
            nj['n_valid_steps'] = tt_valid.get(key, 400)
        jobs.append(nj)

    n_trigger = sum(1 for j in jobs if j['trigger_step_override'] >= 0)
    n_noemit = sum(1 for j in jobs if j['trigger_step_override'] < 0)
    print(f"  Jobs: {len(jobs)}, Trigger: {n_trigger}, No-emission: {n_noemit}")

    # Split into 8 GPU manifests
    gpu_jobs = defaultdict(list)
    for i, j in enumerate(jobs):
        gpu_jobs[i % 8].append(j)

    out_dir = os.path.join(SOTA_MF, output_ns)
    os.makedirs(out_dir, exist_ok=True)
    for gpu in range(8):
        mf_path = os.path.join(out_dir, f'manifest_gpu{gpu}.jsonl')
        with open(mf_path, 'w') as f:
            for j in gpu_jobs[gpu]:
                f.write(json.dumps(j) + '\n')

    # Generate 9-fold canary
    canary = []
    folds_seen = set()
    for fold_int in range(1, 10):
        fold = f'{fold_int:02d}'
        for j in jobs:
            if j['fold'] == fold and fold not in folds_seen:
                canary.append(j)
                folds_seen.add(fold)
                break

    canary_dir = os.path.join(CANARY, output_ns)
    os.makedirs(canary_dir, exist_ok=True)
    canary_path = os.path.join(canary_dir, 'manifest_canary.jsonl')
    with open(canary_path, 'w') as f:
        for j in canary:
            f.write(json.dumps(j) + '\n')

    print(f"  Canary: {len(canary)} jobs, folds: {sorted(folds_seen)}")
    print(f"  Manifests: {out_dir}/")
    print(f"  Canary: {canary_path}")

    # Verify first job
    j0 = jobs[0]
    print(f"  Sample: fold={j0['fold']} s{j0['state_id']} det={j0['detector_seed']} pert={j0['perturbation_seed']} trigger={j0['trigger_step_override']} obj={spec.get('attack_objective','N/A')}")

# Write condition registry
import hashlib, time
registry_lines = ["condition_id,authorization_status,objective_semantics,timing_policy,epsilon,optimization_steps,K,attack_objective,output_root"]
for cond_name, spec in CONDITIONS.items():
    status = "CANARY_READY"
    registry_lines.append(
        f"{cond_name},{status},{spec['description']},{spec['timing']},"
        f"0.02353,20,10,{spec.get('attack_objective','null')},"
        f"{EVID}/{spec['output_namespace']}/formal_v1"
    )

registry_path = os.path.join('/mnt/sdc/dty_user/table1_sota_execution_v1/condition_specs', 'TABLE1_SOTA_CONDITION_REGISTRY_V1.csv')
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
with open(registry_path, 'w') as f:
    f.write('\n'.join(registry_lines) + '\n')
print(f"\nRegistry: {registry_path}")
print("Done. All SOTA manifests ready.")
