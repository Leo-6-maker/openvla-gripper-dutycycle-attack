#!/usr/bin/env python3
"""rNAD v2: Same-space comparisons only, actual model action stats."""
import os, json, csv, math, hashlib, sys
import numpy as np
from collections import defaultdict

BASE = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/rnad_v2'
os.makedirs(OUT, exist_ok=True)

# Actual LIBERO-Object action statistics from victim model dataset_statistics.json
MODEL_PATH = '/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object'
UNNORM_KEY = 'libero_object'
ACTION_Q01 = np.array([-0.5383928418159485, -0.8758928775787354, -0.9375,
                       -0.06964285671710968, -0.11678571254014969, -0.15964286029338837, 0.0],
                      dtype=np.float64)
ACTION_Q99 = np.array([0.8464285731315613, 0.84375, 0.9375,
                       0.08142857253551483, 0.14892856776714325, 0.0867857113480568, 1.0],
                      dtype=np.float64)
ACTION_RANGE = ACTION_Q99 - ACTION_Q01

# Save action stats source
stats_source = {
    'victim_model_path': MODEL_PATH,
    'unnorm_key': UNNORM_KEY,
    'source_file': MODEL_PATH + '/dataset_statistics.json',
    'q01': ACTION_Q01.tolist(),
    'q99': ACTION_Q99.tolist(),
    'range': ACTION_RANGE.tolist(),
}
with open(os.path.join(OUT, 'ACTION_STATS_SOURCE.json'), 'w') as f:
    json.dump(stats_source, f, indent=2)
stats_sha = hashlib.sha256(json.dumps(stats_source, sort_keys=True).encode()).hexdigest()
with open(os.path.join(OUT, 'ACTION_STATS_SHA256.txt'), 'w') as f:
    f.write(stats_sha + '  ACTION_STATS_SOURCE.json\n')

def safe_float_list(s):
    if not s or s == '[]':
        return None
    try:
        return [float(x.strip()) for x in s.strip('[]').split(',')]
    except:
        return None

def compute_rnad(adv_action, clean_action):
    """Same-space Range-Normalized Action Discrepancy.
    Only valid when adv and clean are in the SAME action space.
    """
    if adv_action is None or clean_action is None:
        return None, None, None
    if len(adv_action) < 7 or len(clean_action) < 7:
        return None, None, None
    diff = np.abs(np.array(adv_action[:7], dtype=np.float64) -
                  np.array(clean_action[:7], dtype=np.float64))
    nad_per_dof = diff / ACTION_RANGE
    nad_all = float(np.mean(nad_per_dof))
    nad_arm = float(np.mean(nad_per_dof[:6]))
    nad_gripper = float(nad_per_dof[6])
    return nad_all, nad_arm, nad_gripper

run_rows = []
condition_summary = defaultdict(lambda: {
    'runs': 0, 'emit_runs': 0,
    'rNAD_pol_prelock_arm': [], 'rNAD_pol_prelock_grip': [],
    'rNAD_pol_exec_arm': [], 'rNAD_pol_exec_grip': [],
    'rNAD_env_exec_arm': [], 'rNAD_env_exec_grip': [],
    'clean_fwd_ms': [], 'attack_prep_ms': [], 'adv_decode_ms': [],
    'arm_lock_ms': [], 'total_step_ms': [],
})

total_expected = 0; total_found = 0; parse_errors = 0
armlock_violations = 0; nonfinite_values = 0
mixed_space_count = 0  # should always be 0 in v2
armlock_attack_frames = 0

REQUIRED = ['clean_policy_action_7d', 'adv_policy_action_7d_before_lock',
            'executed_policy_action_7d_after_lock', 'clean_env_action_7d',
            'executed_env_action_7d', 'attack_this',
            'clean_forward_ms', 'pgd_optimization_ms', 'adv_decode_ms',
            'arm_lock_ms', 'total_step_ms']

for cond in sorted(os.listdir(BASE)):
    cp = os.path.join(BASE, cond)
    if not os.path.isdir(cp):
        continue
    for run_dir in sorted(os.listdir(cp)):
        rp = os.path.join(cp, run_dir)
        tele_path = os.path.join(rp, 'step_telemetry.csv')
        summ_path = os.path.join(rp, 'episode_summary.json')
        total_expected += 1
        if not os.path.isfile(tele_path) or not os.path.isfile(summ_path):
            continue
        total_found += 1

        with open(summ_path) as f: summ = json.load(f)
        with open(tele_path) as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        if all_rows:
            for col in REQUIRED:
                if col not in all_rows[0]:
                    parse_errors += 1

        attack_rows = [r for r in all_rows if r.get('attack_this', '').lower() == 'true']
        n_attack = len(attack_rows)
        is_armlock = summ.get('arm_lock', False)
        if is_armlock and n_attack > 0:
            armlock_attack_frames += n_attack

        # Accumulators
        pol_pre_arm = []; pol_pre_grip = []
        pol_exec_arm = []; pol_exec_grip = []
        env_exec_arm = []; env_exec_grip = []
        clean_fwd = []; attack_prep = []; adv_decode = []
        arm_lock_t = []; total_step = []

        for row in attack_rows:
            clean_pol = safe_float_list(row.get('clean_policy_action_7d', ''))
            adv_pol = safe_float_list(row.get('adv_policy_action_7d_before_lock', ''))
            exec_pol = safe_float_list(row.get('executed_policy_action_7d_after_lock', ''))
            clean_env = safe_float_list(row.get('clean_env_action_7d', ''))
            exec_env = safe_float_list(row.get('executed_env_action_7d', ''))

            if None in (clean_pol, adv_pol, exec_pol, clean_env, exec_env):
                parse_errors += 1; continue
            if len(clean_pol) < 7:
                parse_errors += 1; continue

            # --- SAME-SPACE COMPARISONS ONLY ---
            # 1. Policy pre-lock: adv_policy vs clean_policy (both policy space)
            _, pa, pg = compute_rnad(adv_pol, clean_pol)
            if pa is None: parse_errors += 1; continue
            pol_pre_arm.append(pa); pol_pre_grip.append(pg)

            # 2. Policy executed: exec_policy vs clean_policy (both policy space)
            _, ea, eg = compute_rnad(exec_pol, clean_pol)
            pol_exec_arm.append(ea); pol_exec_grip.append(eg)

            # 3. Environment executed: exec_env vs clean_env (both env space)
            _, eea, eeg = compute_rnad(exec_env, clean_env)
            env_exec_arm.append(eea); env_exec_grip.append(eeg)

            # ArmLock invariant: env-executed arm must be exactly 0
            if is_armlock and eea is not None and eea > 1e-9:
                armlock_violations += 1

            for v in [pa, pg, ea, eg, eea, eeg]:
                if v is not None and not math.isfinite(v):
                    nonfinite_values += 1

            # Timing
            clean_fwd.append(float(row.get('clean_forward_ms', 0)))
            attack_prep.append(float(row.get('pgd_optimization_ms', 0)))
            adv_decode.append(float(row.get('adv_decode_ms', 0)))
            arm_lock_t.append(float(row.get('arm_lock_ms', 0)))
            total_step.append(float(row.get('total_step_ms', 0)))

        def nanmean(vals):
            return float(np.mean(vals)) if vals else None
        def nanmax(vals):
            return float(np.max(vals)) if vals else None

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
                if v is not None:
                    cs[k].append(v)
            for k in ['clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms', 'total_step_ms']:
                v = run_row[k]
                if v is not None:
                    cs[k].append(v)

# Write run-level CSV
with open(os.path.join(OUT, 'RNAD_V2_RUN_LEVEL.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
    w.writeheader(); w.writerows(run_rows)

# Write condition summary
with open(os.path.join(OUT, 'RNAD_V2_CONDITION_SUMMARY.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    fields = ['condition', 'runs', 'emit_runs']
    metric_fields = ['rNAD_pol_prelock_arm', 'rNAD_pol_prelock_grip',
                     'rNAD_pol_exec_arm', 'rNAD_pol_exec_grip',
                     'rNAD_env_exec_arm', 'rNAD_env_exec_grip']
    timing_fields = ['clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms', 'total_step_ms']
    w.writerow(fields + [m + '_mean' for m in metric_fields + timing_fields])
    for cond, cs in sorted(condition_summary.items()):
        row = [cond, cs['runs'], cs['emit_runs']]
        for m in metric_fields + timing_fields:
            vals = cs[m]
            row.append(sum(vals)/len(vals) if vals else None)
        w.writerow(row)

# Write paired deltas
paired_rows = []
for obj_name, nl_cond, al_cond in [('TMA', 'tma_nolock', 'tma_armlock'),
                                     ('Prefix', 'prefix_nolock', 'prefix_armlock')]:
    nl_runs = {((r['task_idx'], r['state_id'], r['perturbation_seed'])): r for r in run_rows if r['condition'] == nl_cond}
    al_runs = {((r['task_idx'], r['state_id'], r['perturbation_seed'])): r for r in run_rows if r['condition'] == al_cond}
    common = sorted(set(nl_runs.keys()) & set(al_runs.keys()))

    for field in ['rNAD_pol_prelock_arm', 'rNAD_pol_prelock_grip',
                   'rNAD_pol_exec_arm', 'rNAD_pol_exec_grip',
                   'rNAD_env_exec_arm', 'rNAD_env_exec_grip']:
        deltas = []
        for k in common:
            nv = nl_runs[k].get(field)
            av = al_runs[k].get(field)
            if nv is not None and av is not None:
                deltas.append(av - nv)
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

with open(os.path.join(OUT, 'RNAD_V2_PAIRED_DELTAS.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
    w.writeheader(); w.writerows(paired_rows)

# Write audit
audit = {
    'expected_runs': total_expected, 'parsed_runs': total_found,
    'attack_frames_total': sum(int(r['n_attack_frames']) for r in run_rows),
    'armlock_runs': sum(1 for r in run_rows if r['arm_lock']),
    'armlock_attack_frames': armlock_attack_frames,
    'armlock_arm_violations': armlock_violations,
    'mixed_space_comparisons': 0,
    'parse_errors': parse_errors,
    'nonfinite_values': nonfinite_values,
    'action_stats_sha256': stats_sha,
    'action_q01': ACTION_Q01.tolist(),
    'action_q99': ACTION_Q99.tolist(),
    'action_range': ACTION_RANGE.tolist(),
    'same_space_only': True,
    'comparison_definitions': {
        'rNAD_pol_prelock': 'adv_policy_action vs clean_policy_action',
        'rNAD_pol_executed': 'executed_policy_action vs clean_policy_action',
        'rNAD_env_executed': 'executed_env_action vs clean_env_action',
    },
}
with open(os.path.join(OUT, 'RNAD_V2_AUDIT.json'), 'w') as f:
    json.dump(audit, f, indent=2)

# Print summary
print('=== rNAD V2 COMPLETE ===')
print('Expected: {}, Parsed: {}'.format(total_expected, total_found))
print('ArmLock frames: {}, Violations: {}'.format(armlock_attack_frames, armlock_violations))
print('Parse errors: {}, Non-finite: {}'.format(parse_errors, nonfinite_values))
print('Mixed-space comparisons: 0 (all same-space)')
print()
print('Action stats Q01:', ACTION_Q01.tolist())
print('Action stats Q99:', ACTION_Q99.tolist())
print('Action range:', ACTION_RANGE.tolist())
print()
print('=== CONDITION SUMMARY ===')
for cond in ['tma_nolock', 'tma_armlock', 'prefix_nolock', 'prefix_armlock']:
    cs = condition_summary[cond]
    def m(vals): return sum(vals)/len(vals) if vals else float('nan')
    print('{} ({} runs):'.format(cond, cs['runs']))
    print('  rNAD_pol_prelock:  arm={:.4f} grip={:.4f}'.format(
        m(cs['rNAD_pol_prelock_arm']), m(cs['rNAD_pol_prelock_grip'])))
    print('  rNAD_pol_exec:     arm={:.4f} grip={:.4f}'.format(
        m(cs['rNAD_pol_exec_arm']), m(cs['rNAD_pol_exec_grip'])))
    print('  rNAD_env_exec:     arm={:.6f} grip={:.4f}'.format(
        m(cs['rNAD_env_exec_arm']), m(cs['rNAD_env_exec_grip'])))
    print('  latency: clean={:.0f} prep={:.0f} decode={:.0f} lock={:.3f} total={:.0f}'.format(
        m(cs['clean_fwd_ms']), m(cs['attack_prep_ms']), m(cs['adv_decode_ms']),
        m(cs['arm_lock_ms']), m(cs['total_step_ms'])))
print()
print('=== PAIRED DELTAS ===')
for pr in paired_rows:
    print('{} {}: mean={:.4f} [{:.4f}, {:.4f}] pos={} neg={} zero={}'.format(
        pr['objective'], pr['metric'], pr['mean_delta'], pr['ci_lo'], pr['ci_hi'],
        pr['n_positive'], pr['n_negative'], pr['n_zero']))
