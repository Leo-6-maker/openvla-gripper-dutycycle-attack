#!/usr/bin/env python3
import os, json, csv, math, sys
from collections import defaultdict

BASE = '/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/metric_refresh_v2'
OUT = '/mnt/sdc/dty_user/openvla_attack/reports/phase7_table1/nad'
os.makedirs(OUT, exist_ok=True)

# LIBERO action space approximate bounds (from model action_stats)
# Q01 and Q99 for each of 7 DoF, used for NAD normalization
import numpy as np
ACTION_Q01 = np.array([-0.18, -0.20, -0.15, -0.35, -0.30, -0.25, -0.99], dtype=np.float64)
ACTION_Q99 = np.array([ 0.18,  0.20,  0.15,  0.35,  0.30,  0.25,  0.99], dtype=np.float64)
ACTION_RANGE = ACTION_Q99 - ACTION_Q01  # per-DoF range for normalization

def safe_float_list(s):
    if not s or s == '[]':
        return None
    try:
        return [float(x.strip()) for x in s.strip('[]').split(',')]
    except:
        return None

def compute_nad(adv_action, clean_action):
    """Normalized Action Discrepancy per DoF.
    NAD_i = |adv_i - clean_i| / range_i
    Returns (nad_all, nad_arm, nad_gripper) or (None,None,None).
    """
    if adv_action is None or clean_action is None or len(adv_action) < 7 or len(clean_action) < 7:
        return None, None, None
    diff = np.abs(np.array(adv_action[:7], dtype=np.float64) -
                  np.array(clean_action[:7], dtype=np.float64))
    nad_per_dof = diff / ACTION_RANGE
    nad_all = float(np.mean(nad_per_dof))
    nad_arm = float(np.mean(nad_per_dof[:6]))
    nad_gripper = float(nad_per_dof[6])
    return nad_all, nad_arm, nad_gripper

run_rows = []
armlock_audit = []
condition_summary = defaultdict(lambda: {
    'runs': 0, 'emit_runs': 0, 'no_emit_runs': 0,
    'nad_pol_arm_mean': [], 'nad_pol_arm_max': [],
    'nad_pol_grip_mean': [], 'nad_pol_grip_max': [],
    'nad_exec_arm_mean': [], 'nad_exec_arm_max': [],
    'nad_exec_grip_mean': [], 'nad_exec_grip_max': [],
    'grip_cmd_diff_mean': [], 'grip_qpos_diff_mean': [],
    'width_diff_mean': [], 'height_diff': [],
    'clean_fwd_ms': [], 'attack_prep_ms': [], 'adv_decode_ms': [],
    'arm_lock_ms': [], 'total_step_ms': [],
    'eef_disp': [], 'obj_eef_dist': [],
})

total_expected = 0
total_found = 0
missing_columns = set()
nonfinite_values = 0
armlock_violations = 0
armlock_attack_frames = 0
parse_errors = 0

REQUIRED_COLS = [
    'clean_policy_action_7d', 'adv_policy_action_7d_before_lock',
    'executed_policy_action_7d_after_lock', 'clean_env_action_7d',
    'executed_env_action_7d', 'attack_this',
    'clean_forward_ms', 'pgd_optimization_ms', 'adv_decode_ms',
    'arm_lock_ms', 'total_step_ms',
    'gripper_qpos_sum', 'gripper_width', 'raw_gripper', 'env_gripper',
    'eef_x', 'eef_y', 'eef_z', 'object_x', 'object_y', 'object_z',
    'condition', 'objective_id', 'arm_lock',
    'task', 'state_id', 'perturbation_seed',
]

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

        with open(summ_path) as f:
            summ = json.load(f)

        with open(tele_path) as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        if all_rows:
            for col in REQUIRED_COLS:
                if col not in all_rows[0]:
                    missing_columns.add(col)

        attack_rows = [r for r in all_rows if r.get('attack_this', '').lower() == 'true']
        n_attack = len(attack_rows)

        nad_pol_arm_vals = []
        nad_pol_grip_vals = []
        nad_exec_arm_vals = []
        nad_exec_grip_vals = []
        grip_cmd_diffs = []
        grip_qpos_diffs = []
        width_diffs = []
        clean_fwd_times = []
        attack_prep_times = []
        adv_decode_times = []
        arm_lock_times = []
        total_step_times = []
        eef_disps = []
        obj_eef_dists = []

        is_armlock = summ.get('arm_lock', False)

        for row in attack_rows:
            clean_pol = safe_float_list(row.get('clean_policy_action_7d', ''))
            adv_pol = safe_float_list(row.get('adv_policy_action_7d_before_lock', ''))
            exec_pol = safe_float_list(row.get('executed_policy_action_7d_after_lock', ''))
            clean_env = safe_float_list(row.get('clean_env_action_7d', ''))
            exec_env = safe_float_list(row.get('executed_env_action_7d', ''))

            if None in (clean_pol, adv_pol, exec_pol, clean_env, exec_env):
                parse_errors += 1
                continue

            if len(clean_pol) < 7:
                parse_errors += 1
                continue

            # NAD policy (adv_policy vs clean_policy)
            nad_pol_all, nad_pol_arm, nad_pol_grip = compute_nad(adv_pol, clean_pol)
            if nad_pol_arm is not None:
                nad_pol_arm_vals.append(nad_pol_arm)
                nad_pol_grip_vals.append(nad_pol_grip)

            # NAD exec (executed vs clean_env)
            nad_exec_all, nad_exec_arm, nad_exec_grip = compute_nad(exec_pol, clean_env)
            if nad_exec_arm is not None:
                nad_exec_arm_vals.append(nad_exec_arm)
                nad_exec_grip_vals.append(nad_exec_grip)

            # ArmLock invariant: exec_arm NAD must be exactly 0
            if is_armlock and nad_exec_arm is not None and nad_exec_arm > 1e-9:
                armlock_violations += 1
                armlock_audit.append({
                    'condition': cond, 'run': run_dir,
                    'step': row.get('step'), 'nad_exec_arm': nad_exec_arm,
                })

            grip_cmd_diffs.append(abs(float(row.get('raw_gripper', 0)) - float(row.get('env_gripper', 0))))
            qpos = float(row.get('gripper_qpos_sum', 0))
            width = float(row.get('gripper_width', 0))
            grip_qpos_diffs.append(abs(qpos))
            width_diffs.append(width)

            clean_fwd_times.append(float(row.get('clean_forward_ms', 0)))
            attack_prep_times.append(float(row.get('pgd_optimization_ms', 0)))
            adv_decode_times.append(float(row.get('adv_decode_ms', 0)))
            arm_lock_times.append(float(row.get('arm_lock_ms', 0)))
            total_step_times.append(float(row.get('total_step_ms', 0)))

            try:
                eef_disp = ((float(row.get('eef_x', 0)))**2 +
                           (float(row.get('eef_y', 0)))**2 +
                           (float(row.get('eef_z', 0)))**2) ** 0.5
                eef_disps.append(eef_disp)
            except:
                pass

            obj_eef_dists.append(float(row.get('object_eef_distance', 0)))

            for v in [nad_pol_arm, nad_pol_grip, nad_exec_arm, nad_exec_grip]:
                if v is not None and not math.isfinite(v):
                    nonfinite_values += 1

        if is_armlock and n_attack > 0:
            armlock_attack_frames += n_attack

        run_row = {
            'condition': cond,
            'run_dir': run_dir,
            'task': summ.get('task_name', summ.get('task', '')),
            'task_idx': summ.get('task_idx', ''),
            'state_id': summ.get('state_id', ''),
            'perturbation_seed': summ.get('perturbation_seed', ''),
            'arm_lock': is_armlock,
            'task_success': summ.get('task_success', None),
            'n_attack_frames': n_attack,
            'mlp_triggered': summ.get('mlp_triggered', False),
            'nad_pol_all_mean': sum(nad_pol_arm_vals)/len(nad_pol_arm_vals) if nad_pol_arm_vals else None,
            'nad_pol_arm_mean': sum(nad_pol_arm_vals)/len(nad_pol_arm_vals) if nad_pol_arm_vals else None,
            'nad_pol_arm_max': max(nad_pol_arm_vals) if nad_pol_arm_vals else None,
            'nad_pol_grip_mean': sum(nad_pol_grip_vals)/len(nad_pol_grip_vals) if nad_pol_grip_vals else None,
            'nad_pol_grip_max': max(nad_pol_grip_vals) if nad_pol_grip_vals else None,
            'nad_exec_all_mean': sum(nad_exec_arm_vals)/len(nad_exec_arm_vals) if nad_exec_arm_vals else None,
            'nad_exec_arm_mean': sum(nad_exec_arm_vals)/len(nad_exec_arm_vals) if nad_exec_arm_vals else None,
            'nad_exec_arm_max': max(nad_exec_arm_vals) if nad_exec_arm_vals else None,
            'nad_exec_grip_mean': sum(nad_exec_grip_vals)/len(nad_exec_grip_vals) if nad_exec_grip_vals else None,
            'nad_exec_grip_max': max(nad_exec_grip_vals) if nad_exec_grip_vals else None,
            'grip_cmd_diff_mean': sum(grip_cmd_diffs)/len(grip_cmd_diffs) if grip_cmd_diffs else None,
            'grip_qpos_diff_mean': sum(grip_qpos_diffs)/len(grip_qpos_diffs) if grip_qpos_diffs else None,
            'width_diff_mean': sum(width_diffs)/len(width_diffs) if width_diffs else None,
            'clean_fwd_ms_mean': sum(clean_fwd_times)/len(clean_fwd_times) if clean_fwd_times else None,
            'attack_prep_ms_mean': sum(attack_prep_times)/len(attack_prep_times) if attack_prep_times else None,
            'adv_decode_ms_mean': sum(adv_decode_times)/len(adv_decode_times) if adv_decode_times else None,
            'arm_lock_ms_mean': sum(arm_lock_times)/len(arm_lock_times) if arm_lock_times else None,
            'total_step_ms_mean': sum(total_step_times)/len(total_step_times) if total_step_times else None,
            'eef_disp_mean': sum(eef_disps)/len(eef_disps) if eef_disps else None,
            'obj_eef_dist_mean': sum(obj_eef_dists)/len(obj_eef_dists) if obj_eef_dists else None,
        }
        run_rows.append(run_row)

        cs = condition_summary[cond]
        cs['runs'] += 1
        if n_attack > 0:
            cs['emit_runs'] += 1
            if run_row['nad_pol_arm_mean'] is not None:
                cs['nad_pol_arm_mean'].append(run_row['nad_pol_arm_mean'])
                cs['nad_pol_arm_max'].append(run_row['nad_pol_arm_max'])
                cs['nad_pol_grip_mean'].append(run_row['nad_pol_grip_mean'])
                cs['nad_pol_grip_max'].append(run_row['nad_pol_grip_max'])
                cs['nad_exec_arm_mean'].append(run_row['nad_exec_arm_mean'])
                cs['nad_exec_arm_max'].append(run_row['nad_exec_arm_max'])
                cs['nad_exec_grip_mean'].append(run_row['nad_exec_grip_mean'])
                cs['nad_exec_grip_max'].append(run_row['nad_exec_grip_max'])
                cs['grip_cmd_diff_mean'].append(run_row['grip_cmd_diff_mean'])
                cs['grip_qpos_diff_mean'].append(run_row['grip_qpos_diff_mean'])
                cs['width_diff_mean'].append(run_row['width_diff_mean'])
                cs['clean_fwd_ms'].append(run_row['clean_fwd_ms_mean'])
                cs['attack_prep_ms'].append(run_row['attack_prep_ms_mean'])
                cs['adv_decode_ms'].append(run_row['adv_decode_ms_mean'])
                cs['arm_lock_ms'].append(run_row['arm_lock_ms_mean'])
                cs['total_step_ms'].append(run_row['total_step_ms_mean'])
                cs['eef_disp'].append(run_row['eef_disp_mean'])
                cs['obj_eef_dist'].append(run_row['obj_eef_dist_mean'])
        else:
            cs['no_emit_runs'] += 1

# Write run-level CSV
run_csv_path = os.path.join(OUT, 'NAD_RUN_LEVEL.csv')
fieldnames = list(run_rows[0].keys()) if run_rows else []
with open(run_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(run_rows)
print('Written: {} ({} rows)'.format(run_csv_path, len(run_rows)))

# Write condition summary CSV
cond_csv_path = os.path.join(OUT, 'NAD_CONDITION_SUMMARY.csv')
with open(cond_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['condition', 'runs', 'emit_runs', 'no_emit_runs',
                     'nad_pol_arm_mean', 'nad_pol_arm_max',
                     'nad_pol_grip_mean', 'nad_pol_grip_max',
                     'nad_exec_arm_mean', 'nad_exec_arm_max',
                     'nad_exec_grip_mean', 'nad_exec_grip_max',
                     'clean_fwd_ms', 'attack_prep_ms', 'adv_decode_ms', 'arm_lock_ms',
                     'total_step_ms', 'eef_disp', 'obj_eef_dist'])
    for cond, cs in sorted(condition_summary.items()):
        def mean(vals):
            return sum(vals)/len(vals) if vals else float('nan')
        writer.writerow([
            cond, cs['runs'], cs['emit_runs'], cs['no_emit_runs'],
            mean(cs['nad_pol_arm_mean']), mean(cs['nad_pol_arm_max']),
            mean(cs['nad_pol_grip_mean']), mean(cs['nad_pol_grip_max']),
            mean(cs['nad_exec_arm_mean']), mean(cs['nad_exec_arm_max']),
            mean(cs['nad_exec_grip_mean']), mean(cs['nad_exec_grip_max']),
            mean(cs['clean_fwd_ms']), mean(cs['attack_prep_ms']),
            mean(cs['adv_decode_ms']), mean(cs['arm_lock_ms']),
            mean(cs['total_step_ms']), mean(cs['eef_disp']), mean(cs['obj_eef_dist']),
        ])
print('Written: {}'.format(cond_csv_path))

# Write ArmLock audit
armlock_csv_path = os.path.join(OUT, 'ARMLOCK_INVARIANT_FULL.csv')
if armlock_audit:
    with open(armlock_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'run', 'step', 'nad_exec_arm'])
        writer.writeheader()
        for a in armlock_audit:
            writer.writerow(a)
print('Written: {} ({} violations)'.format(armlock_csv_path, len(armlock_audit)))

# Write audit JSON
audit = {
    'expected_runs': total_expected,
    'found_runs': total_found,
    'emit_runs': sum(cs['emit_runs'] for cs in condition_summary.values()),
    'no_emit_runs': sum(cs['no_emit_runs'] for cs in condition_summary.values()),
    'armlock_attack_frames_audited': armlock_attack_frames,
    'armlock_violations': armlock_violations,
    'missing_columns': sorted(missing_columns),
    'nonfinite_values': nonfinite_values,
    'parse_errors': parse_errors,
    'condition_summary': {
        cond: {'runs': cs['runs'], 'emit_runs': cs['emit_runs'], 'no_emit_runs': cs['no_emit_runs']}
        for cond, cs in condition_summary.items()
    },
}
with open(os.path.join(OUT, 'NAD_AUDIT.json'), 'w') as f:
    json.dump(audit, f, indent=2)
print('Written: {}'.format(os.path.join(OUT, 'NAD_AUDIT.json')))

# Print summary
print()
print('=== NAD AGGREGATION COMPLETE ===')
print('Expected: {}, Found: {}'.format(total_expected, total_found))
print('Emit runs: {}, No-emit: {}'.format(audit['emit_runs'], audit['no_emit_runs']))
print('ArmLock frames audited: {}, violations: {}'.format(armlock_attack_frames, armlock_violations))
print('Missing columns: {}'.format(sorted(missing_columns)))
print('Non-finite: {}, Parse errors: {}'.format(nonfinite_values, parse_errors))
print()
print('=== CONDITION SUMMARY ===')
for cond, cs in sorted(condition_summary.items()):
    def mean(vals):
        return sum(vals)/len(vals) if vals else float('nan')
    print('{}: {} runs ({} emit, {} no-emit)'.format(cond, cs['runs'], cs['emit_runs'], cs['no_emit_runs']))
    print('  NAD_pol_arm:  {:.4f} / max {:.4f}'.format(mean(cs['nad_pol_arm_mean']), mean(cs['nad_pol_arm_max'])))
    print('  NAD_pol_grip: {:.4f} / max {:.4f}'.format(mean(cs['nad_pol_grip_mean']), mean(cs['nad_pol_grip_max'])))
    print('  NAD_exec_arm: {:.6f} / max {:.6f}'.format(mean(cs['nad_exec_arm_mean']), mean(cs['nad_exec_arm_max'])))
    print('  NAD_exec_grip: {:.4f} / max {:.4f}'.format(mean(cs['nad_exec_grip_mean']), mean(cs['nad_exec_grip_max'])))
    print('  Latency: clean={:.0f} prep={:.0f} decode={:.0f} lock={:.3f} total={:.0f}ms'.format(
        mean(cs['clean_fwd_ms']), mean(cs['attack_prep_ms']),
        mean(cs['adv_decode_ms']), mean(cs['arm_lock_ms']), mean(cs['total_step_ms'])))
    print('  EEF disp: {:.4f}  obj-EEF dist: {:.4f}'.format(mean(cs['eef_disp']), mean(cs['obj_eef_dist'])))
