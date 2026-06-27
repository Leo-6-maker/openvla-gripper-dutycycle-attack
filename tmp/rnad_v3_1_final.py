#!/usr/bin/env python3
"""rNAD v3.1: All fixes — read_bytes, assertions, gate checks, full output SHAs."""
import os, json, csv, math, hashlib, sys
import numpy as np
from collections import defaultdict
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BASE = Path('/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2')
OUT = Path('/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/rnad_v3')
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load dataset_statistics.json from victim model ----
MODEL_DIR = Path('/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object')
DS_PATH = MODEL_DIR / 'dataset_statistics.json'
ds_bytes = DS_PATH.read_bytes()
DS_FILE_SHA = hashlib.sha256(ds_bytes).hexdigest()
ds = json.loads(ds_bytes.decode('utf-8'))
UNNORM_KEY = 'libero_object'

# Assertions
assert UNNORM_KEY in ds, 'Missing unnorm_key: ' + UNNORM_KEY
assert 'action' in ds[UNNORM_KEY], 'Missing action stats'
action_stats = ds[UNNORM_KEY]['action']
POLICY_Q01 = np.array(action_stats['q01'], dtype=np.float64)
POLICY_Q99 = np.array(action_stats['q99'], dtype=np.float64)
assert POLICY_Q01.shape == (7,), 'q01 not 7-dim'
assert POLICY_Q99.shape == (7,), 'q99 not 7-dim'
assert np.isfinite(POLICY_Q01).all(), 'non-finite q01'
assert np.isfinite(POLICY_Q99).all(), 'non-finite q99'
POLICY_RANGE = POLICY_Q99 - POLICY_Q01
assert (POLICY_RANGE > 0).all(), 'non-positive range'

# Environment gripper range: raw_gripper in [0,1] -> env_gripper = -(2*raw - 1) in [-1,+1]
# Range = abs(+1 - (-1)) = 2
def raw_gripper_to_env(g):
    return -(2.0 * g - 1.0)
env_lo = raw_gripper_to_env(POLICY_Q01[6])
env_hi = raw_gripper_to_env(POLICY_Q99[6])
ENV_GRIPPER_RANGE = abs(env_hi - env_lo)
assert abs(ENV_GRIPPER_RANGE - 2.0) < 1e-9, 'env gripper range != 2'
ENV_ARM_RANGE = POLICY_RANGE[:6].copy()

# Save action stats source
stats_source = {
    'victim_model_path': str(MODEL_DIR),
    'unnorm_key': UNNORM_KEY,
    'dataset_statistics_sha256': DS_FILE_SHA,
    'policy_q01': POLICY_Q01.tolist(),
    'policy_q99': POLICY_Q99.tolist(),
    'policy_range': POLICY_RANGE.tolist(),
    'env_arm_range': ENV_ARM_RANGE.tolist(),
    'env_gripper_range': float(ENV_GRIPPER_RANGE),
    'env_gripper_range_derivation': 'raw_gripper in [0,1] -> env = -(2*raw-1) in [-1,+1]; range = 2',
    'normalization': {
        'policy_arm': 'model_q99_minus_q01',
        'policy_gripper': 'model_q99_minus_q01',
        'env_arm': 'identity-transformed model q99-q01',
        'env_gripper': '2.0 (derived from raw->env postprocess)',
    },
}
stats_json = json.dumps(stats_source, indent=2)
with open(OUT / 'ACTION_STATS_SOURCE.json', 'w') as f:
    f.write(stats_json)
STATS_FILE_SHA = hashlib.sha256((OUT / 'ACTION_STATS_SOURCE.json').read_bytes()).hexdigest()

with open(OUT / 'ACTION_STATS_FILE_SHA256.txt', 'w') as f:
    f.write('{}  dataset_statistics.json (source)\n'.format(DS_FILE_SHA))
    f.write('{}  ACTION_STATS_SOURCE.json (output)\n'.format(STATS_FILE_SHA))

# ---- rNAD computation ----
def safe_float_list(s):
    if not s or s == '[]': return None
    try: return [float(x.strip()) for x in s.strip('[]').split(',')]
    except: return None

def compute_rnad_policy(adv, clean):
    if adv is None or clean is None or len(adv) < 7 or len(clean) < 7:
        return None, None, None
    diff = np.abs(np.array(adv[:7]) - np.array(clean[:7]))
    n = diff / POLICY_RANGE
    return float(np.mean(n)), float(np.mean(n[:6])), float(n[6])

def compute_rnad_env(adv_env, clean_env):
    if adv_env is None or clean_env is None or len(adv_env) < 7 or len(clean_env) < 7:
        return None, None, None
    diff = np.abs(np.array(adv_env[:7]) - np.array(clean_env[:7]))
    n_arm = diff[:6] / ENV_ARM_RANGE
    n_grip = diff[6] / ENV_GRIPPER_RANGE
    return float(np.mean(np.concatenate([n_arm, [n_grip]]))), float(np.mean(n_arm)), float(n_grip)

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
successfully_parsed_attack_frames = 0; missing_columns = set()
armlock_run_count = 0

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
            skipped_attack_frames += 10; continue
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
        is_armlock = summ.get('arm_lock', False)
        if is_armlock:
            armlock_run_count += 1

        pol_pre_arm = []; pol_pre_grip = []
        pol_exec_arm = []; pol_exec_grip = []
        env_exec_arm = []; env_exec_grip = []
        clean_fwd = []; attack_prep = []; adv_decode = []; arm_lock_t = []; total_step = []

        n_skipped_this_run = 0
        for row in attack_rows:
            clean_pol = safe_float_list(row.get('clean_policy_action_7d', ''))
            adv_pol = safe_float_list(row.get('adv_policy_action_7d_before_lock', ''))
            exec_pol = safe_float_list(row.get('executed_policy_action_7d_after_lock', ''))
            clean_env = safe_float_list(row.get('clean_env_action_7d', ''))
            exec_env = safe_float_list(row.get('executed_env_action_7d', ''))

            if None in (clean_pol, adv_pol, exec_pol, clean_env, exec_env):
                parse_errors += 1; n_skipped_this_run += 1; continue
            if len(clean_pol) < 7:
                parse_errors += 1; n_skipped_this_run += 1; continue

            _, pa, pg = compute_rnad_policy(adv_pol, clean_pol)
            if pa is None: parse_errors += 1; n_skipped_this_run += 1; continue

            successfully_parsed_attack_frames += 1
            pol_pre_arm.append(pa); pol_pre_grip.append(pg)

            _, ea, eg = compute_rnad_policy(exec_pol, clean_pol)
            pol_exec_arm.append(ea); pol_exec_grip.append(eg)

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

        skipped_attack_frames += n_skipped_this_run
        n_attack_parsed = len(attack_rows) - n_skipped_this_run
        if is_armlock and n_attack_parsed > 0:
            armlock_attack_frames += n_attack_parsed

        def nanmean(vals): return float(np.mean(vals)) if vals else None

        run_row = {
            'condition': cond, 'run_dir': run_dir,
            'task_idx': summ.get('task_idx', ''), 'state_id': summ.get('state_id', ''),
            'perturbation_seed': summ.get('perturbation_seed', ''),
            'arm_lock': is_armlock, 'task_success': summ.get('task_success', None),
            'n_attack_frames': n_attack_parsed,
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
        if n_attack_parsed > 0:
            cs['emit_runs'] += 1
            for k in ['rNAD_pol_prelock_arm', 'rNAD_pol_prelock_grip',
                       'rNAD_pol_exec_arm', 'rNAD_pol_exec_grip',
                       'rNAD_env_exec_arm', 'rNAD_env_exec_grip']:
                v = run_row[k]
                if v is not None: cs[k].append(v)
            for k in ['clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms', 'total_step_ms']:
                v = run_row[k]
                if v is not None: cs[k].append(v)

# ---- GATE CHECKS ----
gate_errors = []
if total_expected != 108: gate_errors.append('expected_runs != 108: {}'.format(total_expected))
if total_found != 108: gate_errors.append('parsed_runs != 108: {}'.format(total_found))
if successfully_parsed_attack_frames != 1080: gate_errors.append('parsed_attack_frames != 1080: {}'.format(successfully_parsed_attack_frames))
if skipped_attack_frames != 0: gate_errors.append('skipped_attack_frames != 0: {}'.format(skipped_attack_frames))
if armlock_run_count != 54: gate_errors.append('armlock_runs != 54: {}'.format(armlock_run_count))
if armlock_attack_frames != 540: gate_errors.append('armlock_attack_frames != 540: {}'.format(armlock_attack_frames))
if armlock_violations != 0: gate_errors.append('armlock_violations != 0: {}'.format(armlock_violations))
if nonfinite_values != 0: gate_errors.append('nonfinite_values != 0: {}'.format(nonfinite_values))
if len(missing_columns) > 0: gate_errors.append('missing_columns: {}'.format(sorted(missing_columns)))
if gate_errors:
    print('GATE FAILED:')
    for e in gate_errors: print('  ' + e)
    sys.exit(1)

# ---- Write outputs ----
run_csv = OUT / 'RNAD_V3_RUN_LEVEL.csv'
with open(run_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
    w.writeheader(); w.writerows(run_rows)

cond_csv = OUT / 'RNAD_V3_CONDITION_SUMMARY.csv'
mf = ['rNAD_pol_prelock_arm', 'rNAD_pol_prelock_grip', 'rNAD_pol_exec_arm', 'rNAD_pol_exec_grip',
      'rNAD_env_exec_arm', 'rNAD_env_exec_grip']
tf = ['clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms', 'total_step_ms']
with open(cond_csv, 'w', newline='') as f:
    w = csv.writer(f)
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
        deltas = [al_runs[k][field] - nl_runs[k][field] for k in common
                  if nl_runs[k].get(field) is not None and al_runs[k].get(field) is not None]
        if deltas:
            d = np.array(deltas)
            rng = np.random.RandomState(42)
            means = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(10000)]
            ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
            n_pos = int(np.sum(d > 1e-12)); n_neg = int(np.sum(d < -1e-12)); n_zero = int(np.sum(np.abs(d) < 1e-12))
            paired_rows.append({'objective': obj_name, 'metric': field, 'N': len(deltas),
                'mean_delta': float(np.mean(d)), 'median_delta': float(np.median(d)),
                'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                'n_positive': n_pos, 'n_negative': n_neg, 'n_zero': n_zero})

paired_csv = OUT / 'RNAD_V3_PAIRED_DELTAS.csv'
with open(paired_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
    w.writeheader(); w.writerows(paired_rows)

# Definitions
with open(OUT / 'RNAD_V3_DEFINITIONS.md', 'w') as f:
    f.write("""# rNAD V3 Definitions

## Range-Normalized Action Discrepancy
rNAD_i = |a_adv_i - a_clean_i| / range_i

## Normalization per representation
- Policy arm (DoF 0-5): range = Q99_i - Q01_i (from model dataset_statistics.json)
- Policy gripper (DoF 6): range = Q99_6 - Q01_6
- Environment arm (DoF 0-5): same as policy (identity transform in OpenVLA→LIBERO)
- Environment gripper (DoF 6): range = 2.0
  (raw_gripper in [0,1] → env = -(2*raw-1) in [-1,+1]; range = |1 - (-1)| = 2)

## Aggregation
- all: mean over 7 DoF
- arm: mean over DoF 0-5
- gripper: DoF 6 value

## Comparison spaces (all same-space)
- rNAD_pol_prelock: adv_policy_action vs clean_policy_action
- rNAD_pol_executed: executed_policy_action vs clean_policy_action
- rNAD_env_executed: executed_env_action vs clean_env_action
""")

# Audit
SCRIPT_SHA = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
audit = {
    'expected_runs': total_expected, 'parsed_runs': total_found,
    'successfully_parsed_attack_frames': successfully_parsed_attack_frames,
    'skipped_attack_frames': skipped_attack_frames,
    'armlock_runs': armlock_run_count,
    'armlock_attack_frames': armlock_attack_frames,
    'armlock_arm_violations': armlock_violations,
    'mixed_space_comparisons': 0,
    'missing_columns': sorted(missing_columns),
    'parse_errors': parse_errors,
    'nonfinite_values': nonfinite_values,
    'gate_passed': len(gate_errors) == 0,
    'normalization': stats_source['normalization'],
    'source_dataset_statistics_sha256': DS_FILE_SHA,
    'action_stats_output_file_sha256': STATS_FILE_SHA,
    'analysis_script_sha256': SCRIPT_SHA,
}
with open(OUT / 'RNAD_V3_AUDIT.json', 'w') as f:
    json.dump(audit, f, indent=2)

# Output SHA256SUMS
sha_out = OUT / 'RNAD_V3_OUTPUT_SHA256SUMS.txt'
with open(sha_out, 'w') as f:
    for fn in ['RNAD_V3_RUN_LEVEL.csv', 'RNAD_V3_CONDITION_SUMMARY.csv',
               'RNAD_V3_PAIRED_DELTAS.csv', 'RNAD_V3_AUDIT.json',
               'RNAD_V3_DEFINITIONS.md', 'ACTION_STATS_SOURCE.json',
               'ACTION_STATS_FILE_SHA256.txt']:
        fp = OUT / fn
        if fp.is_file():
            f.write('{}  {}\n'.format(hashlib.sha256(fp.read_bytes()).hexdigest(), fn))

print('=== rNAD V3.1 GATE PASSED ===')
print('Runs: {}/{}'.format(total_found, total_expected))
print('Attack frames: {} parsed, {} skipped'.format(successfully_parsed_attack_frames, skipped_attack_frames))
print('ArmLock: {} runs, {} frames, {} violations'.format(armlock_run_count, armlock_attack_frames, armlock_violations))
print('DS SHA: {}'.format(DS_FILE_SHA))
print('Script SHA: {}'.format(SCRIPT_SHA))
print()
for cond in ['tma_nolock', 'tma_armlock', 'prefix_nolock', 'prefix_armlock']:
    cs = condition_summary[cond]
    def m(vals): return sum(vals)/len(vals) if vals else float('nan')
    print('{}: env_exec_grip={:.4f} arm={:.6f}'.format(cond, m(cs['rNAD_env_exec_grip']), m(cs['rNAD_env_exec_arm'])))
print()
for pr in paired_rows:
    if 'env_exec_grip' in pr['metric']:
        print('{}: delta={:.4f} [{:.4f}, {:.4f}] pos={} neg={} zero={}'.format(
            pr['objective'], pr['mean_delta'], pr['ci_lo'], pr['ci_hi'],
            pr['n_positive'], pr['n_negative'], pr['n_zero']))
