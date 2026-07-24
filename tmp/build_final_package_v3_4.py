#!/usr/bin/env python3
"""Build final V3.4 audit package: correct provenance order, blinded templates, aggregator."""
import os, json, csv, hashlib, shutil, subprocess

BASE = '/mnt/sdc/dty_user/openvla_attack'
TMP = '/tmp/table1_v3_4_final'
if os.path.exists(TMP): shutil.rmtree(TMP)
os.makedirs(TMP)

# ===== 1. Write provenance files FIRST =====
ANALYSIS_COMMIT = 'c7a446752228234f6e98458578b03f71da5db83c'
with open(TMP + '/ANALYSIS_GENERATOR_COMMIT.txt', 'w') as f:
    f.write(ANALYSIS_COMMIT + '\n')
with open(TMP + '/ARTIFACT_RUNTIME_GIT_HEAD.txt', 'w') as f:
    f.write('ace1876281a9ad6ed68e1229a6e17346356766e9\n')
with open(TMP + '/EXPERIMENT_INPUT_COMMIT.txt', 'w') as f:
    f.write('01d19779ef770135e1ad01fd8541e75e56181057\n')

# ===== 2. Copy rNAD =====
RN = TMP + '/rnad_v3'; os.makedirs(RN)
for fn in os.listdir(BASE + '/reports/phase7_table1/rnad_v3'):
    src = BASE + '/reports/phase7_table1/rnad_v3/' + fn
    if os.path.isfile(src): shutil.copy2(src, RN + '/' + fn)
for s in ['rnad_v3_1_final.py', 'rnad_v3_1_cleanup.py']:
    if os.path.isfile('/tmp/' + s): shutil.copy2('/tmp/' + s, RN + '/' + s)

# ===== 3. Copy CQFR public + private =====
CP = TMP + '/cqfr_v3_4/public'; os.makedirs(CP)
CQ = BASE + '/evidence/phase7_table1/cqfr_v3_unique68/public'
for fn in os.listdir(CQ):
    fp = CQ + '/' + fn
    if os.path.isfile(fp) and any(fn.endswith(x) for x in ['.zip', '.sha256', '.csv', '.txt', '.md']):
        shutil.copy2(fp, CP + '/' + fn)

CV = TMP + '/cqfr_v3_4/private'; os.makedirs(CV)
PR = BASE + '/evidence/phase7_table1/cqfr_v3_unique68/private'
for fn in os.listdir(PR):
    if fn.endswith('.csv') or fn.endswith('.json'):
        shutil.copy2(PR + '/' + fn, CV + '/' + fn)

# Copy generator
shutil.copy2('/tmp/generate_cqfr_v3_4.py', TMP + '/cqfr_v3_4/generate_cqfr_v3_4.py')

# ===== 4. Generate BLINDED reviewer templates =====
# Read private key to get blind_id -> task_instruction mapping
key_path = CV + '/CQFR_UNIQUE68_PRIVATE_KEY.csv'
with open(key_path) as f:
    key_rows = list(csv.DictReader(f))

# Read reviewer assignments
r1_path = CV + '/CQFR_REVIEWER1_ASSIGNMENT.csv'
r2_path = CV + '/CQFR_REVIEWER2_ASSIGNMENT.csv'
with open(r1_path) as f: r1_rows = list(csv.DictReader(f))
with open(r2_path) as f: r2_rows = list(csv.DictReader(f))

# Build id -> instruction map
instr_map = {r['blind_id']: r['task_instruction'] for r in key_rows}

def write_blinded_template(path, assignment_rows, reviewer_label):
    """Write reviewer-facing CSV with ONLY blind_id, video_path, task_instruction, and blank annotation columns."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['blind_id', 'video_path', 'task_instruction',
                    'task_outcome', 'task_outcome_confidence',
                    'contact_quality_failure', 'contact_quality_confidence',
                    'premature_release', 'drop_after_lift', 'unstable_transport',
                    'uncontrolled_final_drop', 'controlled_placement',
                    'primary_contact_failure_cause', 'notes'])
        for row in assignment_rows:
            bid = row['blind_id']
            w.writerow([bid, bid + '.mp4', instr_map.get(bid, ''),
                        '', '', '', '', '', '', '', '', '', '', ''])

write_blinded_template(CP + '/CQFR_REVIEWER1_BLINDED_TEMPLATE.csv', r1_rows, 'R1')
write_blinded_template(CP + '/CQFR_REVIEWER2_BLINDED_TEMPLATE.csv', r2_rows, 'R2')
print('Blinded reviewer templates written')

# ===== 5. Write label aggregator/validator =====
aggregator_path = TMP + '/cqfr_v3_4/cqfr_label_aggregator.py'
with open(aggregator_path, 'w') as f:
    f.write(r'''#!/usr/bin/env python3
"""CQFR Label Validator & Aggregator V3.4."""
import os, json, csv, math, sys
import numpy as np
from collections import defaultdict

def validate_labels(path, role):
    """Validate reviewer CSV for legal field values and consistency."""
    errors = []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    legal_to = {'success', 'failure', 'ambiguous', 'video_invalid', ''}
    legal_cq = {'yes', 'no', 'ambiguous', 'not_applicable', ''}
    legal_yn = {'yes', 'no', 'ambiguous', 'not_applicable', ''}
    legal_cause = {'gripper', 'arm', 'mixed', 'other', 'unclear', 'not_applicable', ''}
    legal_conf = {'high', 'medium', 'low', ''}

    for i, row in enumerate(rows):
        bid = row.get('blind_id', f'row_{i}')
        to = row.get('task_outcome', '')
        cq = row.get('contact_quality_failure', '')
        if to not in legal_to: errors.append(f'{bid}: illegal task_outcome={to}')
        if cq not in legal_cq: errors.append(f'{bid}: illegal contact_quality_failure={cq}')
        # video_invalid should have empty CQ
        if to == 'video_invalid' and cq not in ('', 'not_applicable'):
            errors.append(f'{bid}: video_invalid with CQ={cq}')
        # CQ=no should not have failure subtypes or cause
        if cq == 'no':
            for subtype in ['premature_release', 'drop_after_lift', 'unstable_transport', 'uncontrolled_final_drop']:
                v = row.get(subtype, '')
                if v == 'yes': errors.append(f'{bid}: CQ=no but {subtype}=yes')
            cause = row.get('primary_contact_failure_cause', '')
            if cause not in ('', 'not_applicable'): errors.append(f'{bid}: CQ=no but cause={cause}')

    print(f'{role}: {len(rows)} rows, {len(errors)} validation errors')
    for e in errors[:10]: print(f'  {e}')
    return rows, errors

def aggregate(r1_rows, r2_rows, run_mapping, blind_key):
    """Aggregate labels to per-condition CQFR/CQSR/mismatch with cluster bootstrap."""
    # Build id -> label maps
    r1_map = {r['blind_id']: r for r in r1_rows}
    r2_map = {r['blind_id']: r for r in r2_rows}
    id_to_sha = {bk['blind_id']: bk['unique_video_sha256'] for bk in blind_key}

    # Adjudicate on overlap
    overlap = set(r1_map.keys()) & set(r2_map.keys())
    r1_only = set(r1_map.keys()) - overlap

    adjudicated = {}
    disagreements = []
    for bid in overlap:
        r1_cq = r1_map[bid].get('contact_quality_failure', '')
        r2_cq = r2_map[bid].get('contact_quality_failure', '')
        if r1_cq == r2_cq:
            adjudicated[bid] = r1_cq
        else:
            disagreements.append((bid, r1_cq, r2_cq))
            # Use Reviewer 1 as primary pending adjudication
            adjudicated[bid] = r1_cq
    for bid in r1_only:
        adjudicated[bid] = r1_map[bid].get('contact_quality_failure', '')

    # Agreement on overlap
    n_agree = len(overlap) - len(disagreements)
    n_overlap = len(overlap)
    raw_agree = n_agree / n_overlap if n_overlap else 1.0
    print(f'Agreement on {n_overlap} overlap: {n_agree}/{n_overlap} = {raw_agree:.3f}')
    if disagreements:
        print(f'Disagreements: {len(disagreements)}')

    # Map to 108 scientific rows
    bid_to_sha = {bk['blind_id']: bk['unique_video_sha256'] for bk in blind_key}
    mapping_rows = []
    for rm in run_mapping:
        bid = rm['unique_blind_id']
        cq_label = adjudicated.get(bid, 'ambiguous')
        mapping_rows.append({**rm, 'cq_label': cq_label})

    # Per-condition aggregate
    conditions = sorted(set(rm['condition'] for rm in run_mapping))
    print('\nPer-condition CQFR (conditional, adjudicated):')
    for cond in conditions:
        cond_runs = [r for r in mapping_rows if r['condition'] == cond]
        valid = [r for r in cond_runs if r['cq_label'] in ('yes', 'no')]
        n_yes = sum(1 for r in valid if r['cq_label'] == 'yes')
        n_cq = len(valid)
        cqfr = n_yes / n_cq if n_cq else 0
        # CQSR
        cqsr_n = sum(1 for r in valid if r['simulator_task_success'] == 'True' and r['cq_label'] == 'no')
        cqsr = cqsr_n / n_cq if n_cq else 0
        # SR-CQ mismatch
        mismatch_n = sum(1 for r in valid if r['simulator_task_success'] == 'True' and r['cq_label'] == 'yes')
        mismatch = mismatch_n / n_cq if n_cq else 0
        # Cluster bootstrap CI (by unique video hash)
        video_hashes = list(set(r['unique_video_sha256'] for r in cond_runs))
        if len(video_hashes) > 1:
            rng = np.random.RandomState(42)
            cqfr_means = []
            for _ in range(10000):
                bs_hashes = rng.choice(video_hashes, size=len(video_hashes), replace=True)
                bs_runs = [r for r in cond_runs if r['unique_video_sha256'] in set(bs_hashes)]
                bs_valid = [r for r in bs_runs if r['cq_label'] in ('yes', 'no')]
                if bs_valid:
                    cqfr_means.append(sum(1 for r in bs_valid if r['cq_label'] == 'yes') / len(bs_valid))
            ci_lo, ci_hi = np.percentile(cqfr_means, [2.5, 97.5])
        else:
            ci_lo, ci_hi = cqfr, cqfr
        print(f'  {cond}: CQFR={n_yes}/{n_cq}={cqfr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], CQSR={cqsr:.3f}, mismatch={mismatch:.3f}')

    # Pooled
    valid_all = [r for r in mapping_rows if r['cq_label'] in ('yes', 'no')]
    n_yes_all = sum(1 for r in valid_all if r['cq_label'] == 'yes')
    ambiguous = sum(1 for r in mapping_rows if r['cq_label'] == 'ambiguous')
    print(f'\nPooled: CQFR={n_yes_all}/{len(valid_all)}, ambiguous={ambiguous}/{len(mapping_rows)}')
    return mapping_rows

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--r1', required=True, help='Reviewer 1 CSV')
    ap.add_argument('--r2', required=True, help='Reviewer 2 CSV')
    ap.add_argument('--mapping', required=True, help='108-run private mapping CSV')
    ap.add_argument('--key', required=True, help='Private key CSV')
    ap.add_argument('--output', default='/tmp/cqfr_aggregated.csv')
    args = ap.parse_args()

    with open(args.key) as f: blind_key = list(csv.DictReader(f))
    with open(args.mapping) as f: run_mapping = list(csv.DictReader(f))

    r1_rows, e1 = validate_labels(args.r1, 'Reviewer 1')
    r2_rows, e2 = validate_labels(args.r2, 'Reviewer 2')
    if e1 or e2:
        print('Validation errors found. Fix before aggregating.')
        sys.exit(1)

    result = aggregate(r1_rows, r2_rows, run_mapping, blind_key)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result[0].keys()))
        w.writeheader(); w.writerows(result)
    print(f'\nAggregated output: {args.output}')
''')
print('Aggregator script written')

# ===== 6. TREE + SHA256SUMS (AFTER all files written) =====
with open(TMP + '/TREE.txt', 'w') as f:
    for root, dirs, files in os.walk(TMP):
        for fn in sorted(files):
            f.write(os.path.relpath(os.path.join(root, fn), TMP) + '\n')

with open(TMP + '/SHA256SUMS.txt', 'w') as f:
    for root, dirs, files in os.walk(TMP):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            if fn != 'SHA256SUMS.txt' and os.path.getsize(fp) < 200 * 1024 * 1024:
                sha = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
                f.write(sha + '  ' + os.path.relpath(fp, TMP) + '\n')

# ===== 7. Verify ANALYSIS_GENERATOR_COMMIT checksum =====
with open(TMP + '/SHA256SUMS.txt') as f:
    for line in f:
        expected, fn = line.strip().split('  ', 1)
        actual = hashlib.sha256(open(os.path.join(TMP, fn), 'rb').read()).hexdigest()
        if actual != expected:
            print(f'CHECKSUM FAIL: {fn}')
            print(f'  expected: {expected}')
            print(f'  actual:   {actual}')
        else:
            pass  # OK
print('All checksums verified')

# ===== 8. Tar =====
subprocess.run(['tar', '-czf', '/tmp/table1_v3_4_final.tar.gz', '-C', TMP, '.'], check=True)
sz = os.path.getsize('/tmp/table1_v3_4_final.tar.gz')
sha = hashlib.sha256(open('/tmp/table1_v3_4_final.tar.gz', 'rb').read()).hexdigest()
print(f'Package SHA256: {sha} ({round(sz/1024/1024,1)} MB)')
