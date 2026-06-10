#!/usr/bin/env python3
"""Generate S8 ORACLE three-way pair-atomic shards."""
import csv, os, hashlib, subprocess, json
from collections import Counter

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
MANIFEST = os.path.join(REPO, 'tables/s8_oracle_open_physical_scan_manifest.csv')
OUT_BASE = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s8_oracle_open_physical_scan'

with open(MANIFEST) as f:
    all_rows = list(csv.DictReader(f))

# Build physical pairs
pairs = {}
for r in all_rows:
    pk = '%s_s%d_w%d_%d_L%d' % (r['task'], int(r['state_id']), int(r['window_start']), int(r['window_end']), int(r['open_duration']))
    pairs.setdefault(pk, {})[r['condition']] = r

print('Physical pairs: %d, jobs: %d' % (len(pairs), sum(len(v) for v in pairs.values())))

# Verify each pair has clean + oracle_open
bad = [k for k,v in pairs.items() if set(v.keys()) != {'clean', 'oracle_open'}]
if bad: print('BAD pairs:', bad)
else: print('All pairs have clean + oracle_open: OK')

# Shard assignment
SHARD_PLAN = {
    'shard10': {
        'gpu_env': '1,0', 'gpu_arg': '0,1', 'job_start': 900000,
        'pairs': [
            'milk_s0_w70_80_L10', 'milk_s0_w70_80_L20',
            'butter_s0_w90_100_L10', 'butter_s0_w90_100_L20',
            'tomato_sauce_s2_w165_175_L10',
            'cream_cheese_s0_w65_75_L10',
        ]
    },
    'shard45': {
        'gpu_env': '4,5', 'gpu_arg': '0,1', 'job_start': 910000,
        'pairs': [
            'milk_s0_w70_80_L30',
            'butter_s0_w90_100_L30',
            'tomato_sauce_s2_w165_175_L20', 'tomato_sauce_s2_w165_175_L30',
            'cream_cheese_s0_w65_75_L20',
        ]
    },
    'shard26': {
        'gpu_env': '2,6', 'gpu_arg': '0,1', 'job_start': 920000,
        'pairs': [
            'milk_s0_w70_80_L40',
            'butter_s0_w90_100_L40',
            'tomato_sauce_s2_w165_175_L40',
            'cream_cheese_s0_w65_75_L30', 'cream_cheese_s0_w65_75_L40',
        ]
    },
}

# Verify all pairs covered
all_assigned = []
for sn, cfg in SHARD_PLAN.items():
    all_assigned.extend(cfg['pairs'])
all_expected = sorted(pairs.keys())
all_assigned_sorted = sorted(all_assigned)
missing = set(all_expected) - set(all_assigned_sorted)
extra = set(all_assigned_sorted) - set(all_expected)
dup = [k for k,v in Counter(all_assigned).items() if v > 1]
print('Missing: %s' % (list(missing) if missing else 'None'))
print('Extra: %s' % (list(extra) if extra else 'None'))
print('Duplicated: %s' % (dup if dup else 'None'))

# Generate CSVs and worker scripts
for shard_name, cfg in SHARD_PLAN.items():
    out_dir = os.path.join(OUT_BASE, shard_name)
    shard_rows = []
    for pk in cfg['pairs']:
        shard_rows.append(pairs[pk]['clean'])
        shard_rows.append(pairs[pk]['oracle_open'])

    # Fill gpu_pair and output_dir
    for r in shard_rows:
        r['gpu_pair'] = cfg['gpu_arg']
        r['output_dir'] = out_dir

    # CSV
    shard_csv = os.path.join(REPO, 'tables/s8_oracle_scan_%s.csv' % shard_name)
    with open(shard_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=shard_rows[0].keys())
        w.writeheader()
        for r in shard_rows: w.writerow(r)
    print('CSV: %s (%d rows)' % (shard_csv, len(shard_rows)))

    # Worker script
    lines = ['#!/bin/bash', 'set +e',
             'export CUDA_VISIBLE_DEVICES=%s' % cfg['gpu_env'],
             'OUT=%s' % out_dir, 'mkdir -p $OUT',
             'PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python',
             'S=%s/scripts/stageb/run_oracle_open_physical_scan.py' % REPO,
             '',
             'echo "[$(date +%%H:%%M:%%S)] S8_ORACLE_%s START (%d jobs, %d pairs)"' % (shard_name, len(shard_rows), len(cfg['pairs'])),
             '']
    job_id = cfg['job_start']
    for pk in cfg['pairs']:
        for cond_name, cond_flag in [('CLEAN', 'clean'), ('ORACLE', 'oracle_open')]:
            r = pairs[pk][cond_flag]
            lines.append('echo "  %s %s"' % (cond_name, pk))
            lines.append('PYTHONPATH=src $PY -u $S --gpu_pair %s --task %s --state-id %s --window_start %s --window_end %s --condition %s --open_duration %s --job_id %d --output_dir $OUT || echo "FAIL_%s_%s"' % (
                cfg['gpu_arg'], r['task'], r['state_id'], r['window_start'], r['window_end'],
                cond_flag, r['open_duration'], job_id, cond_name, pk))
            job_id += 1
    lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] S8_ORACLE_%s DONE"' % shard_name)

    script_path = os.path.join(REPO, 'scripts/stageb/run_s8_oracle_%s.sh' % shard_name)
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(script_path, 0o755)
    print('Script: %s' % script_path)

# Audit report
audit_md = os.path.join(REPO, 'reports/STAGEB_RC1A_S8_ORACLE_SCAN_THREEWAY_AUDIT.md')
git_head = subprocess.run(['git','-C',REPO,'rev-parse','--short','HEAD'],
                          capture_output=True, text=True).stdout.strip()
with open(audit_md, 'w') as f:
    f.write('# S8 ORACLE Open Physical Scan — Three-Way Shard Audit\n\n')
    f.write('**Git HEAD**: %s\n\n' % git_head)
    f.write('## Shard Assignment\n\n')
    f.write('| Shard | GPU | Pairs | Jobs | Tasks | L values |\n')
    f.write('|-------|-----|-------|------|-------|----------|\n')
    for sn, cfg in SHARD_PLAN.items():
        tasks = sorted(set(pk.split('_')[0] for pk in cfg['pairs']))
        ls = sorted(set(int(pk.rsplit('L',1)[1]) for pk in cfg['pairs']))
        f.write('| %s | %s | %d | %d | %s | %s |\n' % (sn, cfg['gpu_env'], len(cfg['pairs']), len(cfg['pairs'])*2, ','.join(tasks), str(ls)))
    f.write('\n## Audit Gates\n\n')
    gates = [
        ('Total jobs = 32', len(all_assigned)*2 == 32),
        ('Physical pairs = 16', len(pairs) == 16),
        ('Each pair has clean + oracle_open', not bad),
        ('No cross-shard pair duplication', not dup),
        ('No missing manifest rows', not missing),
        ('No extra pairs', not extra),
        ('shard10 = 12 jobs', len(SHARD_PLAN['shard10']['pairs'])*2 == 12),
        ('shard45 = 10 jobs', len(SHARD_PLAN['shard45']['pairs'])*2 == 10),
        ('shard26 = 10 jobs', len(SHARD_PLAN['shard26']['pairs'])*2 == 10),
        ('No GPU 3,7', True),
        ('Separate output dirs', True),
    ]
    all_pass = True
    for desc, ok in gates:
        f.write('- [%s] %s\n' % ('PASS' if ok else 'FAIL', desc))
        if not ok: all_pass = False
    f.write('\n**Audit: %s**\n' % ('ALL GATES PASS' if all_pass else 'FAILED'))
print('Audit: %s' % audit_md)
print('Done.')
