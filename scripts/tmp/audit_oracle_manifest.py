"""Audit COMMAND_OPEN_ORACLE manifest against TRUE_T10 emission keys."""
import os, json, sys, time

TRUE = '/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/TRUE_T10/formal_v1'
ORACLE_MF = '/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/COMMAND_OPEN_ORACLE_T10/launch'

# Collect TRUE_T10 emission/no-emission keys
tt_emit = set()
tt_noemit = set()
for fold in sorted(os.listdir(TRUE)):
    fp = os.path.join(TRUE, fold)
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
                    if d.get('mlp_emit_step', -1) >= 0:
                        tt_emit.add(key)
                    else:
                        tt_noemit.add(key)

# Collect oracle manifest keys (with duplicate detection)
oracle_keys = {}
oracle_emit_steps = {}
oracle_raw_lines = 0
duplicates = []
for mf in sorted(os.listdir(ORACLE_MF)):
    if not mf.endswith('.jsonl'): continue
    for line in open(os.path.join(ORACLE_MF, mf)):
        oracle_raw_lines += 1
        j = json.loads(line.strip())
        k = (j['fold'], j['state_id'], j['detector_seed'], j['perturbation_seed'])
        if k in oracle_keys:
            duplicates.append(str(k))
        oracle_keys[k] = j
        oracle_emit_steps[k] = j.get('trigger_step_override', -1)

# Audit results
print(f'TRUE_T10 total: {len(tt_emit) + len(tt_noemit)}')
print(f'TRUE_T10 emission keys: {len(tt_emit)}')
print(f'TRUE_T10 no-emission keys: {len(tt_noemit)}')
print(f'Oracle manifest keys: {len(oracle_keys)}')
print(f'Match (oracle ∩ emit): {len(set(oracle_keys.keys()) & tt_emit)}')
print(f'Extra in oracle (not in TT): {len(set(oracle_keys.keys()) - tt_emit)}')
print(f'Missing from oracle (emit not covered): {len(tt_emit - set(oracle_keys.keys()))}')
print(f'No-emit wrongly in oracle: {len(set(oracle_keys.keys()) & tt_noemit)}')
print(f'PASS: {len(set(oracle_keys.keys()) & tt_emit) == 141 and len(set(oracle_keys.keys()) - tt_emit) == 0}')

# Check condition labeling
conditions = set()
for k, j in oracle_keys.items():
    conditions.add((j.get('condition'), j.get('condition_id'), j.get('oracle')))
print(f'\nCondition labels in manifest: {conditions}')

# Verify trigger_step_override matches TRUE_T10 emit
mismatches = 0
for k in oracle_keys:
    if k in tt_emit:
        # Get TT emit step
        fold_id, sid, did, pid = k
        ep_path = os.path.join(TRUE, f'fold_{fold_id}', f'state_{sid}',
                               f'det_seed_{did}', f'pert_seed_{pid}', 'episode_summary.json')
        if os.path.exists(ep_path):
            tt = json.load(open(ep_path))
            tt_emit_step = tt['mlp_emit_step']
            oracle_trigger = oracle_emit_steps[k]
            if tt_emit_step != oracle_trigger:
                mismatches += 1
                if mismatches <= 5:
                    print(f'  MISMATCH {k}: TT emit={tt_emit_step}, oracle trigger={oracle_trigger}')

print(f'Emit step mismatches: {mismatches}')

# Fold distribution
from collections import Counter
fold_dist = Counter(k[0] for k in oracle_keys)
print(f'\nFold distribution: {dict(sorted(fold_dist.items()))}')

# No-emission disposition
noemit_21 = sorted(tt_noemit)
print(f'\nNo-emission episodes (21 for ITT denominator):')
for k in noemit_21[:5]:
    print(f'  {k}')
print(f'  ... ({len(noemit_21)} total)')

# Compute manifest SHA
import hashlib
manifest_sha = hashlib.sha256()
for mf in sorted(os.listdir(ORACLE_MF)):
    if not mf.endswith('.jsonl'): continue
    with open(os.path.join(ORACLE_MF, mf), 'rb') as f:
        manifest_sha.update(f.read())

# Write output
OUT = '/mnt/sdc/dty_user/table1_sota_execution_v1/audits'
os.makedirs(OUT, exist_ok=True)

verdict_pass = (len(set(oracle_keys.keys()) & tt_emit) == 141
                and len(set(oracle_keys.keys()) - tt_emit) == 0
                and mismatches == 0
                and len(duplicates) == 0
                and oracle_raw_lines == 141)

with open(os.path.join(OUT, 'ORACLE_MANIFEST_AUDIT.json'), 'w') as f:
    json.dump({
        'audit_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'manifest_path': ORACLE_MF,
        'manifest_sha256': manifest_sha.hexdigest(),
        'true_t10_path': TRUE,
        'raw_line_count': oracle_raw_lines,
        'unique_key_count': len(oracle_keys),
        'duplicate_keys': duplicates,
        'emission_keys_match': len(set(oracle_keys.keys()) & tt_emit),
        'extra_in_oracle': len(set(oracle_keys.keys()) - tt_emit),
        'missing_from_oracle': len(tt_emit - set(oracle_keys.keys())),
        'no_emit_in_oracle': len(set(oracle_keys.keys()) & tt_noemit),
        'emit_step_mismatches': mismatches,
        'fold_distribution': dict(sorted(fold_dist.items())),
        'no_emission_keys': [f'{k[0]}_{k[1]}_{k[2]}_{k[3]}' for k in sorted(tt_noemit)],
        'condition_labels': [list(c) for c in conditions],
        'verdict': 'PASS' if verdict_pass else 'FAIL'
    }, f, indent=2)

print(f'\nAudit saved to {OUT}/ORACLE_MANIFEST_AUDIT.json')
print(f'VERDICT: {"PASS" if verdict_pass else "FAIL"}')
if not verdict_pass:
    print('Audit FAILED — exiting with code 1')
    sys.exit(1)
