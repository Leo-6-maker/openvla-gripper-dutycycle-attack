#!/usr/bin/env python3
"""rNAD v3: Representation-aware normalization. Reads source stats at runtime."""
import os, json, csv, math, hashlib, sys
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2')
OUT = Path('/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/rnad_v3')
OUT.mkdir(parents=True, exist_ok=True)
SCRIPT_PATH = Path('/tmp/rnad_v3_final.py')

# ---- Load actual action stats from victim model at runtime ----
MODEL_DIR = Path('/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object')
DS_PATH = MODEL_DIR / 'dataset_statistics.json'
with open(DS_PATH) as f:
    ds_raw = f.read()
DS_FILE_SHA = hashlib.sha256(ds_raw.encode()).hexdigest()
ds = json.loads(ds_raw)
UNNORM_KEY = 'libero_object'
action_stats = ds[UNNORM_KEY]['action']

POLICY_Q01 = np.array(action_stats['q01'], dtype=np.float64)
POLICY_Q99 = np.array(action_stats['q99'], dtype=np.float64)
POLICY_RANGE = POLICY_Q99 - POLICY_Q01

# Representation-specific ranges:
# Policy gripper: Q99-Q01 (from model stats, typically 1.0)
# Env arm: same as policy arm range (identity transform)
# Env gripper: policy [0,1] -> postprocess(2*x-1, sign invert, binarize) = [-1,+1], range=2
ENV_ARM_RANGE = POLICY_RANGE[:6].copy()
ENV_GRIPPER_RANGE = 2.0

# Save action stats source
stats_source = {
    'victim_model_path': str(MODEL_DIR),
    'unnorm_key': UNNORM_KEY,
    'dataset_statistics_sha256': DS_FILE_SHA,
    'policy_q01': POLICY_Q01.tolist(),
    'policy_q99': POLICY_Q99.tolist(),
    'policy_range': POLICY_RANGE.tolist(),
    'env_arm_range': ENV_ARM_RANGE.tolist(),
    'env_gripper_range': ENV_GRIPPER_RANGE,
    'normalization_notes': {
        'policy_arm': 'model_q99_minus_q01',
        'policy_gripper': 'model_q99_minus_q01',
        'env_arm': 'same as policy (identity transform)',
        'env_gripper': '2.0 (policy[0,1] -> postprocess -> env[-1,+1])',
    },
}
stats_json = json.dumps(stats_source, indent=2, sort_keys=True)
with open(OUT / 'ACTION_STATS_SOURCE.json', 'w') as f:
    f.write(stats_json)
# Hash the file ON DISK
STATS_FILE_SHA = hashlib.sha256(open(OUT / 'ACTION_STATS_SOURCE.json', 'rb').read()).hexdigest()
with open(OUT / 'ACTION_STATS_SHA256.txt', 'w') as f:
    f.write('{}  ACTION_STATS_SOURCE.json\n'.format(STATS_FILE_SHA))
    f.write('{}  dataset_statistics.json (source)\n'.format(DS_FILE_SHA))

# ---- rNAD computation ----
def safe_float_list(s):
    if not s or s == '[]':
        return None
    try:
        return [float(x.strip()) for x in s.strip('[]').split(',')]
    except:
        return None

def compute_rnad_policy(adv, clean):
    """Policy-space rNAD using model Q01/Q99 ranges."""
    if adv is None or clean is None or len(adv) < 7 or len(clean) < 7:
        return None, None, None
    diff = np.abs(np.array(adv[:7]) - np.array(clean[:7]))
    n = diff / POLICY_RANGE
    return float(np.mean(n)), float(np.mean(n[:6])), float(n[6])

def compute_rnad_env(adv_env, clean_env):
    """Environment-space rNAD: arm uses policy range, gripper uses env range=2."""
    if adv_env is None or clean_env is None or len(adv_env) < 7 or len(clean_env) < 7:
        return None, None, None
    diff = np.abs(np.array(adv_env[:7]) - np.array(clean_env[:7]))
    # Arm: same as policy range
    n_arm = diff[:6] / ENV_ARM_RANGE
    # Gripper: env range = 2
    n_grip = diff[6] / ENV_GRIPPER_RANGE
    n_all = np.concatenate([n_arm, [n_grip]])
    return float(np.mean(n_all)), float(np.mean(n_arm)), float(n_grip)

run_rows = []
condition_summary = defaultdict(lambda: {
    'runs': 0, 'emit_runs': 0,
    'rNAD_pol_prelock_arm': [], 'rNAD_pol_prelock_grip': [],
    'rNAD_pol_exec_arm': [], 'rNAD_pol_exec_grip': [],
    'rNAD_env_exec_arm': [], 'rNAD_env_exec_grip': [],
    'clean_fwd_ms': [], 'attack_prep_ms': [], 'adv_decode_ms': [],
    'arm_lock_ms': [], 'total_step_ms': [],
})

total_expected = 0; total_found = 0; parse_errors = 0; skipped_attack_frames = 0
armlock_violations = 0; nonfinite_values = 0; armlock_attack_frames = 0
valid_attack_frames = 0; missing_columns = set()
mixed_space_count = 0  # always 0

REQUIRED = ['clean_policy_action_7d', 'adv_policy_action_7d_before_lock',
            'executed_policy_action_7d_after_lock', 'clean_env_action_7d',
            'executed_env_action_7d', 'attack_this',
            'clean_forward_ms', 'pgd_optimization_ms', 'adv_decode_ms',
            'arm_lock_ms', 'total_step_ms']

for cond in sorted(os.listdir(str(BASE))):
    cp = BASE / cond
    if not cp.is_dir(): continue
    for run_dir in sorted(os.listdir(str(cp))):
        rp = cp / run_dir
        tele_path = rp / 'step_telemetry.csv'
        summ_path = rp / 'episode_summary.json'
        total_expected += 1
        if not tele_path.is_file() or not summ_path.is_file():
            continue
        total_found += 1

        with open(summ_path) as f: summ = json.load(f)
        with open(tele_path) as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        if all_rows:
            for col in REQUIRED:
                if col not in all_rows[0]:
                    missing_columns.add(col)

        attack_rows = [r for r in all_rows if r.get('attack_this', '').lower() == 'true']
        n_attack = len(attack_rows)
        valid_attack_frames += n_attack
        is_armlock = summ.get('arm_lock', False)
        if is_armlock and n_attack > 0:
            armlock_attack_frames += n_attack

        pol_pre_arm = []; pol_pre_grip = []
        pol_exec_arm = []; pol_exec_grip = []
        env_exec_arm = []; env_exec_grip = []
        clean_fwd = []; attack_prep = []; adv_decode = []; arm_lock_t = []; total_step = []

        for row in attack_rows:
            clean_pol = safe_float_list(row.get('clean_policy_action_7d', ''))
            adv_pol = safe_float_list(row.get('adv_policy_action_7d_before_lock', ''))
            exec_pol = safe_float_list(row.get('executed_policy_action_7d_after_lock', ''))
            clean_env = safe_float_list(row.get('clean_env_action_7d', ''))
            exec_env = safe_float_list(row.get('executed_env_action_7d', ''))

            if None in (clean_pol, adv_pol, exec_pol, clean_env, exec_env):
                parse_errors += 1; skipped_attack_frames += 1; continue
            if len(clean_pol) < 7:
                parse_errors += 1; skipped_attack_frames += 1; continue

            # Policy-space comparisons
            _, pa, pg = compute_rnad_policy(adv_pol, clean_pol)
            if pa is None: parse_errors += 1; skipped_attack_frames += 1; continue
            pol_pre_arm.append(pa); pol_pre_grip.append(pg)

            _, ea, eg = compute_rnad_policy(exec_pol, clean_pol)
            pol_exec_arm.append(ea); pol_exec_grip.append(eg)

            # Environment-space comparison
            _, eea, eeg = compute_rnad_env(exec_env, clean_env)
            env_exec_arm.append(eea); env_exec_grip.append(eeg)

            if is_armlock and eea is not None and eea > 1e-9:
                armlock_violations += 1

            for v in [pa, pg, ea, eg, eea, eeg]:
                if v is not None and not math.isfinite(v):
                    nonfinite_values += 1

            clean_fwd.append(float(row.get('clean_forward_ms', 0)))
            attack_prep.append(float(row.get('pgd_optimization_ms', 0)))
            adv_decode.append(float(row.get('adv_decode_ms', 0)))
            arm_lock_t.append(float(row.get('arm_lock_ms', 0)))
            total_step.append(float(row.get('total_step_ms', 0)))

        def nanmean(vals): return float(np.mean(vals)) if vals else None

        run_row = {
            'condition': cond, 'run_dir': run_dir,
            'task_idx': summ.get('task_idx', ''), 'state_id': summ.get('state_id', ''),
            'perturbation_seed': summ.get('perturbation_seed', ''),
            'arm_lock': is_armlock, 'task_success': summ.get('task_success', None),
            'n_attack_frames': n_attack,
            'rNAD_pol_prelock_arm': nanmean(pol_pre_arm),
            'rNAD_pol_prelock_grip': nanmean(pol_pre_grip),
            'rNAD_pol_exec_arm': nanmean(pol_exec_arm),
            'rNAD_pol_exec_grip': nanmean(pol_exec_grip),
            'rNAD_env_exec_arm': nanmean(env_exec_arm),
            'rNAD_env_exec_grip': nanmean(env_exec_grip),
            'clean_fwd_ms': nanmean(clean_fwd), 'attack_prep_ms': nanmean(attack_prep),
            'adv_decode_ms': nanmean(adv_decode), 'arm_lock_ms': nanmean(arm_lock_t),
            'total_step_ms': nanmean(total_step),
        }
        run_rows.append(run_row)

        cs = condition_summary[cond]
        cs['runs'] += 1
        if n_attack > 0:
            cs['emit_runs'] += 1
            for k in ['rNAD_pol_prelock_arm', 'rNAD_pol_prelock_grip',
                       'rNAD_pol_exec_arm', 'rNAD_pol_exec_grip',
                       'rNAD_env_exec_arm', 'rNAD_env_exec_grip']:
                v = run_row[k]
                if v is not None: cs[k].append(v)
            for k in ['clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms', 'total_step_ms']:
                v = run_row[k]
                if v is not None: cs[k].append(v)

# Write run-level CSV
with open(OUT / 'RNAD_V3_RUN_LEVEL.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
    w.writeheader(); w.writerows(run_rows)

# Condition summary
with open(OUT / 'RNAD_V3_CONDITION_SUMMARY.csv', 'w', newline='') as f:
    w = csv.writer(f)
    mf = ['rNAD_pol_prelock_arm', 'rNAD_pol_prelock_grip',
          'rNAD_pol_exec_arm', 'rNAD_pol_exec_grip',
          'rNAD_env_exec_arm', 'rNAD_env_exec_grip']
    tf = ['clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms', 'total_step_ms']
    w.writerow(['condition', 'runs', 'emit_runs'] + [m + '_mean' for m in mf + tf])
    for cond, cs in sorted(condition_summary.items()):
        row = [cond, cs['runs'], cs['emit_runs']]
        for m in mf + tf:
            vals = cs[m]; row.append(sum(vals)/len(vals) if vals else None)
        w.writerow(row)

# Paired deltas
paired_rows = []
for obj_name, nl_cond, al_cond in [('TMA', 'tma_nolock', 'tma_armlock'),
                                     ('Prefix', 'prefix_nolock', 'prefix_armlock')]:
    nl_runs = {((r['task_idx'], r['state_id'], r['perturbation_seed'])): r for r in run_rows if r['condition'] == nl_cond}
    al_runs = {((r['task_idx'], r['state_id'], r['perturbation_seed'])): r for r in run_rows if r['condition'] == al_cond}
    common = sorted(set(nl_runs.keys()) & set(al_runs.keys()))
    for field in mf:
        deltas = []
        for k in common:
            nv = nl_runs[k].get(field); av = al_runs[k].get(field)
            if nv is not None and av is not None: deltas.append(av - nv)
        if deltas:
            d = np.array(deltas)
            rng = np.random.RandomState(42)
            means = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
            n_pos = int(np.sum(d > 1e-12))
            n_neg = int(np.sum(d < -1e-12))
            n_zero = int(np.sum(np.abs(d) < 1e-12))
            paired_rows.append({
                'objective': obj_name, 'metric': field, 'N': len(deltas),
                'mean_delta': float(np.mean(d)), 'median_delta': float(np.median(d)),
                'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                'n_positive': n_pos, 'n_negative': n_neg, 'n_zero': n_zero,
            })

with open(OUT / 'RNAD_V3_PAIRED_DELTAS.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
    w.writeheader(); w.writerows(paired_rows)

# Audit
SCRIPT_SHA = hashlib.sha256(open(SCRIPT_PATH, 'rb').read()).hexdigest() if SCRIPT_PATH.is_file() else 'unknown'
audit = {
    'expected_runs': total_expected, 'parsed_runs': total_found,
    'parsed_attack_frames': valid_attack_frames,
    'skipped_attack_frames': skipped_attack_frames,
    'armlock_runs': sum(1 for r in run_rows if r['arm_lock']),
    'armlock_attack_frames': armlock_attack_frames,
    'armlock_arm_violations': armlock_violations,
    'mixed_space_comparisons': 0,
    'missing_columns': sorted(missing_columns),
    'parse_errors': parse_errors,
    'nonfinite_values': nonfinite_values,
    'normalization': {
        'policy_arm': 'model_q99_minus_q01',
        'policy_gripper': 'model_q99_minus_q01',
        'env_arm': 'identity-transformed model q99-q01',
        'env_gripper': 2.0,
    },
    'source_dataset_statistics_sha256': DS_FILE_SHA,
    'action_stats_output_file_sha256': STATS_FILE_SHA,
    'analysis_script_sha256': SCRIPT_SHA,
    'comparison_definitions': {
        'rNAD_pol_prelock': 'adv_policy_action vs clean_policy_action',
        'rNAD_pol_executed': 'executed_policy_action vs clean_policy_action',
        'rNAD_env_executed': 'executed_env_action vs clean_env_action',
    },
}
with open(OUT / 'RNAD_V3_AUDIT.json', 'w') as f:
    json.dump(audit, f, indent=2)

print('=== rNAD V3 COMPLETE ===')
print('Parsed: {}/{}'.format(total_found, total_expected))
print('Attack frames: {} parsed, {} skipped'.format(valid_attack_frames, skipped_attack_frames))
print('ArmLock frames: {}, Violations: {}'.format(armlock_attack_frames, armlock_violations))
print('Missing columns: {}'.format(sorted(missing_columns)))
print('DS file SHA: {}'.format(DS_FILE_SHA))
print('Stats output SHA: {}'.format(STATS_FILE_SHA))
print()
print('=== CONDITION SUMMARY ===')
for cond in ['tma_nolock', 'tma_armlock', 'prefix_nolock', 'prefix_armlock']:
    cs = condition_summary[cond]
    def m(vals): return sum(vals)/len(vals) if vals else float('nan')
    print('{} ({} runs):'.format(cond, cs['runs']))
    print('  rNAD_pol_prelock:  arm={:.4f} grip={:.4f}'.format(m(cs['rNAD_pol_prelock_arm']), m(cs['rNAD_pol_prelock_grip'])))
    print('  rNAD_pol_exec:     arm={:.4f} grip={:.4f}'.format(m(cs['rNAD_pol_exec_arm']), m(cs['rNAD_pol_exec_grip'])))
    print('  rNAD_env_exec:     arm={:.6f} grip={:.4f}'.format(m(cs['rNAD_env_exec_arm']), m(cs['rNAD_env_exec_grip'])))
print()
print('=== PAIRED ENV EXEC GRIP ===')
for pr in paired_rows:
    if 'env_exec_grip' in pr['metric']:
        print('{}: mean={:.4f} [{:.4f}, {:.4f}] pos={} neg={} zero={}'.format(
            pr['objective'], pr['mean_delta'], pr['ci_lo'], pr['ci_hi'],
            pr['n_positive'], pr['n_negative'], pr['n_zero']))
