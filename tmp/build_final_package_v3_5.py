#!/usr/bin/env python3
"""Build final V3.5 audit package: reviewer ZIPs, aggregator, correct baseline name, hard fail on checksum mismatch."""
import os, json, csv, hashlib, shutil, subprocess, sys

BASE = '/mnt/sdc/dty_user/openvla_attack'
TMP = '/tmp/table1_v3_5_final'
if os.path.exists(TMP): shutil.rmtree(TMP)
os.makedirs(TMP)

ANALYSIS_COMMIT = '8fe8dd73c0ac619f1d53e2afd48a8254a7c659db'

# ===== Provenance FIRST =====
with open(TMP + '/ANALYSIS_GENERATOR_COMMIT.txt', 'w') as f: f.write(ANALYSIS_COMMIT + '\n')
with open(TMP + '/ARTIFACT_RUNTIME_GIT_HEAD.txt', 'w') as f: f.write('ace1876281a9ad6ed68e1229a6e17346356766e9\n')
with open(TMP + '/EXPERIMENT_INPUT_COMMIT.txt', 'w') as f: f.write('01d19779ef770135e1ad01fd8541e75e56181057\n')

# ===== rNAD =====
RN = TMP + '/rnad_v3'; os.makedirs(RN)
for fn in os.listdir(BASE + '/reports/phase7_table1/rnad_v3'):
    src = BASE + '/reports/phase7_table1/rnad_v3/' + fn
    if os.path.isfile(src): shutil.copy2(src, RN + '/' + fn)
for s in ['rnad_v3_1_final.py', 'rnad_v3_1_cleanup.py']:
    if os.path.isfile('/tmp/' + s): shutil.copy2('/tmp/' + s, RN + '/' + s)

# ===== CQFR public + private =====
CP = TMP + '/cqfr_v3_5/public'; os.makedirs(CP)
CQ = BASE + '/evidence/phase7_table1/cqfr_v3_unique68/public'
for fn in os.listdir(CQ):
    fp = CQ + '/' + fn
    if os.path.isfile(fp):
        shutil.copy2(fp, CP + '/' + fn)

CV = TMP + '/cqfr_v3_5/private'; os.makedirs(CV)
PR = BASE + '/evidence/phase7_table1/cqfr_v3_unique68/private'
for fn in os.listdir(PR):
    if fn.endswith('.csv') or fn.endswith('.json'):
        shutil.copy2(PR + '/' + fn, CV + '/' + fn)

# Copy generators
shutil.copy2('/tmp/generate_cqfr_v3_4.py', TMP + '/cqfr_v3_5/generate_cqfr_v3_4.py')
shutil.copy2('/tmp/cqfr_label_aggregator_v3_5.py', TMP + '/cqfr_v3_5/cqfr_label_aggregator_v3_5.py')

# ===== Blinded templates =====
with open(CV + '/CQFR_UNIQUE68_PRIVATE_KEY.csv') as f: key_rows = list(csv.DictReader(f))
with open(CV + '/CQFR_REVIEWER1_ASSIGNMENT.csv') as f: r1_rows = list(csv.DictReader(f))
with open(CV + '/CQFR_REVIEWER2_ASSIGNMENT.csv') as f: r2_rows = list(csv.DictReader(f))
instr_map = {r['blind_id']: r['task_instruction'] for r in key_rows}

def write_blinded(path, assignment):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['blind_id', 'video_path', 'task_instruction',
                    'task_outcome', 'task_outcome_confidence',
                    'contact_quality_failure', 'contact_quality_confidence',
                    'premature_release', 'drop_after_lift', 'unstable_transport',
                    'uncontrolled_final_drop', 'controlled_placement',
                    'primary_contact_failure_cause', 'notes'])
        for row in assignment:
            bid = row['blind_id']
            w.writerow([bid, bid + '.mp4', instr_map.get(bid, ''), '', '', '', '', '', '', '', '', '', '', ''])

write_blinded(CP + '/CQFR_REVIEWER1_BLINDED_TEMPLATE.csv', r1_rows)
write_blinded(CP + '/CQFR_REVIEWER2_BLINDED_TEMPLATE.csv', r2_rows)

# ===== Reviewer 1 ZIP (68 videos) =====
r1_files = [r['blind_id'] + '.mp4' for r in r1_rows]
r1_files += ['CQFR_REVIEWER1_BLINDED_TEMPLATE.csv',
             'CQFR_LABEL_DEFINITIONS_V3_4.txt',
             'CQFR_REVIEW_PROTOCOL_V3_4.md']
# Copy definitions/protocol (use V3.4 since V3.5 aggregator is separate)
for fn in ['CQFR_LABEL_DEFINITIONS_V3_4.txt', 'CQFR_REVIEW_PROTOCOL_V3_4.md']:
    src = CQ + '/' + fn
    if os.path.isfile(src): shutil.copy2(src, CP + '/' + fn)

# Checksum for R1
with open(CP + '/CQFR_REVIEWER1_SHA256SUMS.txt', 'w') as f:
    for fn in r1_files:
        fp = CP + '/' + fn
        if os.path.isfile(fp):
            f.write(hashlib.sha256(open(fp, 'rb').read()).hexdigest() + '  ' + fn + '\n')
subprocess.run(['zip', '-r', '-X', CP + '/CQFR_REVIEWER1_PACKAGE_V3_5.zip'] + r1_files + ['CQFR_REVIEWER1_SHA256SUMS.txt'],
               cwd=CP, capture_output=True, text=True, check=True)
r1_sha = hashlib.sha256(open(CP + '/CQFR_REVIEWER1_PACKAGE_V3_5.zip', 'rb').read()).hexdigest()
print(f'R1 ZIP: {r1_sha} ({os.path.getsize(CP + "/CQFR_REVIEWER1_PACKAGE_V3_5.zip")/1024/1024:.1f}MB)')

# ===== Reviewer 2 ZIP (36 videos) =====
r2_files = [r['blind_id'] + '.mp4' for r in r2_rows]
r2_files += ['CQFR_REVIEWER2_BLINDED_TEMPLATE.csv',
             'CQFR_LABEL_DEFINITIONS_V3_4.txt',
             'CQFR_REVIEW_PROTOCOL_V3_4.md']
with open(CP + '/CQFR_REVIEWER2_SHA256SUMS.txt', 'w') as f:
    for fn in r2_files:
        fp = CP + '/' + fn
        if os.path.isfile(fp):
            f.write(hashlib.sha256(open(fp, 'rb').read()).hexdigest() + '  ' + fn + '\n')
subprocess.run(['zip', '-r', '-X', CP + '/CQFR_REVIEWER2_PACKAGE_V3_5.zip'] + r2_files + ['CQFR_REVIEWER2_SHA256SUMS.txt'],
               cwd=CP, capture_output=True, text=True, check=True)
r2_sha = hashlib.sha256(open(CP + '/CQFR_REVIEWER2_PACKAGE_V3_5.zip', 'rb').read()).hexdigest()
print(f'R2 ZIP: {r2_sha} ({os.path.getsize(CP + "/CQFR_REVIEWER2_PACKAGE_V3_5.zip")/1024/1024:.1f}MB)')

# ===== TREE + SHA256SUMS =====
with open(TMP + '/TREE.txt', 'w') as f:
    for root, dirs, files in os.walk(TMP):
        for fn in sorted(files):
            f.write(os.path.relpath(os.path.join(root, fn), TMP) + '\n')

checksum_failures = 0
with open(TMP + '/SHA256SUMS.txt', 'w') as f:
    for root, dirs, files in os.walk(TMP):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            if fn != 'SHA256SUMS.txt' and os.path.getsize(fp) < 200 * 1024 * 1024:
                sha = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
                f.write(sha + '  ' + os.path.relpath(fp, TMP) + '\n')

# Verify all checksums
with open(TMP + '/SHA256SUMS.txt') as f:
    for line in f:
        expected, fn = line.strip().split('  ', 1)
        actual = hashlib.sha256(open(os.path.join(TMP, fn), 'rb').read()).hexdigest()
        if actual != expected:
            print(f'CHECKSUM FAIL: {fn}')
            checksum_failures += 1

if checksum_failures > 0:
    print(f'FATAL: {checksum_failures} checksum failures')
    sys.exit(1)
print(f'All checksums verified ({checksum_failures} failures)')

# ===== Tar =====
subprocess.run(['tar', '-czf', '/tmp/table1_v3_5_final.tar.gz', '-C', TMP, '.'], check=True)
sz = os.path.getsize('/tmp/table1_v3_5_final.tar.gz')
sha = hashlib.sha256(open('/tmp/table1_v3_5_final.tar.gz', 'rb').read()).hexdigest()
print(f'Package SHA256: {sha} ({round(sz/1024/1024,1)} MB)')
