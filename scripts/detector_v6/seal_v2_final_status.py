"""Seal V2 Pipeline Final Status: FAILED_BEFORE_H2."""
import json, os, hashlib, time

OUT_DIR = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_final_status'
os.makedirs(OUT_DIR, exist_ok=True)

def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

# C2 Distribution Shift
c2_shift = {
    'schema': 'V2_C2_DISTRIBUTION_SHIFT_V1',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'finding': 'Student ranking degrades from FIT_DEV to C2. Per-split calibrator cannot fix ranking.',
    'metrics': {
        'FIT_DEV': {'step_auroc': 0.982, 'episode_auroc': 0.982},
        'C2': {
            'step_auroc_pooled': 0.755, 'step_auroc_per_split_mean': 0.790,
            'step_auroc_per_split_median': 0.815, 'step_auroc_range': [0.495, 0.972],
            'episode_auroc_pooled': 0.566, 'episode_auroc_per_split_mean': 0.680,
            'n_steps': 26903, 'n_episodes': 116, 'positive_rate': 0.199,
        },
    },
    'per_split_step_auroc': {
        'o0_i0': 0.972, 'o0_i1': 0.711, 'o0_i2': 0.926,
        'o1_i0': 0.742, 'o1_i1': 0.896, 'o1_i2': 0.529,
        'o2_i0': 0.772, 'o2_i1': 0.798, 'o2_i2': 0.916,
        'o3_i0': 0.495, 'o3_i1': 0.832, 'o3_i2': 0.891,
    },
    'simpson_paradox': False,
    'calibrator': {'method': 'POOLED_MONOTONIC_PLATT', 'a': 0.096592, 'b': -0.452978, 'ranking_preserved': True},
}
with open(os.path.join(OUT_DIR, 'V2_C2_DISTRIBUTION_SHIFT_V1.json'), 'w') as f: json.dump(c2_shift, f, indent=2)

# P2 Coverage Failure
p2_cov = {
    'schema': 'P2_SCIENTIFIC_COVERAGE_FAILURE_V1',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'finding': 'P2 absent coverage insufficient for FS <= 10% threshold selection.',
    'required': {'valid_absent': 40, 'F3': 10, 'F4': 10, 'opportunity': 40, 'parser_invalid': 0},
    'actual': {'valid_absent': 26, 'F3': 5, 'F4': 12, 'opportunity': 86, 'parser_invalid': 11},
    'gates': {'valid_absent_ge_40': False, 'F3_ge_10': False, 'F4_ge_10': True, 'opportunity_ge_40': True, 'parser_invalid_eq_0': False},
    'note': '26 absent episodes total. FS<=10% allows at most 2 FP. Insufficient for stable threshold.',
}
with open(os.path.join(OUT_DIR, 'P2_SCIENTIFIC_COVERAGE_FAILURE_V1.json'), 'w') as f: json.dump(p2_cov, f, indent=2)

# P2 No Feasible Policy
p2_nfp = {
    'schema': 'P2_NO_FEASIBLE_POLICY_V1',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'finding': 'No (tau, persistence) pair satisfies FS<=10% with recall>0 on P2.',
    'search_space': {'tau_start': 'linspace(0.1, 0.95, 86)', 'persistence': [1,2,3], 'total_configs': 258},
    'root_causes': [
        'C2 episode AUROC=0.566: Student ranking does not separate opp/abs on C2',
        'P2 only 26 absent episodes: at most 2 FP allowed',
        'Combined: no threshold achieves both recall>0 and FS<=10%',
    ],
}
with open(os.path.join(OUT_DIR, 'P2_NO_FEASIBLE_POLICY_V1.json'), 'w') as f: json.dump(p2_nfp, f, indent=2)

# Final Status
final = {
    'schema': 'FINAL_DETECTOR_V2_STATUS_V1',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'status': 'FAILED_BEFORE_H2',
    'pipeline': {
        'STUDENT_TRAINING': 'PASS (12/12, AUROC 0.982)',
        'STUDENT_FREEZE': 'PASS',
        'C2_CALIBRATOR_INTEGRITY': 'PASS (a>0, ranking preserved)',
        'C2_GENERALIZATION': 'FAIL_WEAK (0.982 -> 0.566 episode AUROC)',
        'P2_COVERAGE': 'FAIL (26 absent, need >=40)',
        'P2_FEASIBLE_POLICY': 'FAIL',
        'P2_SCHEDULER_FREEZE': 'FAIL',
    },
    'authorization': {
        'H2_AUTHORIZED': False, 'H2_READ': False,
        'FINAL_DETECTOR_V2': 'FAILED_BEFORE_H2', 'FORMAL_ATTACK': 'HOLD',
    },
    'primary_findings': [
        'Startability supervision dramatically improved H1: recall 33%->99%, FS 29%->12%',
        'Improvement did not stably generalize to C2: episode AUROC 0.982->0.566',
        'Distribution shift from FIT (states 0-23) to C2 (states 24-29)',
        'P2 has insufficient absent coverage for formal threshold selection',
        'Per-split calibrator cannot fix ranking (a>0 preserves AUROC)',
    ],
    'next_steps': [
        'Diagnose state-level distribution shift',
        'Fix parser root cause',
        'Redesign splits with state stratification',
        'Increase F3/F4 hard-negative coverage',
        'Establish V2.1 with improved generalization',
        'Keep H2 unread',
    ],
    'h1_status': 'DEVELOPMENT_ONLY',
    'h2_status': 'UNREAD',
}
with open(os.path.join(OUT_DIR, 'FINAL_DETECTOR_V2_STATUS_V1.json'), 'w') as f: json.dump(final, f, indent=2)

# Seal
all_files = []
for root, dirs, fns in os.walk(OUT_DIR):
    for fn in sorted(fns):
        if fn in ('SHA256SUMS','SHA256SUMS.sha256'): continue
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_DIR)
        all_files.append((rel, sha256_file(fp)))
with open(os.path.join(OUT_DIR, 'SHA256SUMS'), 'w') as f:
    for rel, h in sorted(all_files): f.write(f'{h}  {rel}\n')
sh = sha256_file(os.path.join(OUT_DIR, 'SHA256SUMS'))
with open(os.path.join(OUT_DIR, 'SHA256SUMS.sha256'), 'w') as f:
    f.write(f'{sh}  SHA256SUMS\n')

print('V2 Pipeline Final Status: FAILED_BEFORE_H2')
print(f'Seal: {sh[:16]}')
for k,v in final['pipeline'].items():
    print(f'  {k}: {v}')
