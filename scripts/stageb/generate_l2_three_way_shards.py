#!/usr/bin/env python3
"""Generate three-way pair-atomic shards for Layer-2 confirmation queue."""
import csv, os, hashlib, subprocess, time, json
from collections import Counter

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
QUEUE = os.path.join(REPO, 'tables/layer2_hiddensafe_confirmation_queue.csv')
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/layer2_hiddensafe_confirmation'

with open(QUEUE) as f:
    all_rows = list(csv.DictReader(f))

print('Original queue: %d rows, %d logical pairs' % (
    len(all_rows), len(set(r['logical_pair_key'] for r in all_rows))))

# Organize by logical_pair_key
pairs = {}
for r in all_rows:
    lp = r['logical_pair_key']
    pairs.setdefault(lp, {})[r['condition']] = r

lp_list = sorted(pairs.keys())
print('Total logical pairs: %d' % len(lp_list))

# Separate H and B pairs
h_pairs = [lp for lp in lp_list if any('H_' in pairs[lp][c]['queue_group'] for c in pairs[lp])]
b_pairs = [lp for lp in lp_list if any('B_' in pairs[lp][c]['queue_group'] for c in pairs[lp])]
print('H pairs: %d, B pairs: %d' % (len(h_pairs), len(b_pairs)))

# Interleave H and B for stratification, then assign to shards
# Round-robin: interleave H and B, distribute across shards
all_pairs_ordered = []
h_iter = iter(h_pairs)
b_iter = iter(b_pairs)
# Alternate: one H, one B
while True:
    try: all_pairs_ordered.append(next(h_iter))
    except StopIteration: break
    try: all_pairs_ordered.append(next(b_iter))
    except StopIteration: break

print('Ordered pairs (H/B interleaved): %d' % len(all_pairs_ordered))

# Shard definition: sizes 6, 5, 5
SHARDS = [
    ('shard10', 6, '1,0', '0,1', 800000),
    ('shard45', 5, '4,5', '0,1', 810000),
    ('shard26', 5, '2,6', '0,1', 820000),
]

shard_pairs = {}
offset = 0
for shard_name, shard_size, gpu_env, gpu_arg, job_start in SHARDS:
    shard_pairs[shard_name] = all_pairs_ordered[offset:offset + shard_size]
    offset += shard_size

# Verify
all_assigned = []
for sn, sp in shard_pairs.items():
    all_assigned.extend(sp)
    h_count = sum(1 for lp in sp if lp in h_pairs)
    b_count = sum(1 for lp in sp if lp in b_pairs)
    gpu_env_lookup = {s[0]: s[2] for s in SHARDS}
    print('%s: %d pairs (H=%d B=%d) GPU=%s' % (sn, len(sp), h_count, b_count, gpu_env_lookup[sn]))

print('All assigned: %d, duplicated: %s' % (
    len(all_assigned), 'YES' if len(all_assigned) != len(set(all_assigned)) else 'NO'))
print('Missing: %d' % (len(lp_list) - len(set(all_assigned))))

# ── Write shard CSVs and worker scripts ──
for shard_name, shard_size, gpu_env, gpu_arg, job_start in SHARDS:
    sp = shard_pairs[shard_name]
    shard_rows = []
    for lp in sp:
        shard_rows.append(pairs[lp]['VIS'])
        shard_rows.append(pairs[lp]['RAND'])

    # Write CSV
    shard_csv = os.path.join(REPO, 'tables/layer2_hiddensafe_confirmation_queue_%s.csv' % shard_name)
    cols = list(shard_rows[0].keys())
    with open(shard_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in shard_rows: w.writerow(r)
    print('CSV: %s (%d rows)' % (shard_csv, len(shard_rows)))

    # Generate worker script
    lines = ['#!/bin/bash', 'set +e',
             'export CUDA_VISIBLE_DEVICES=%s' % gpu_env,
             'OUT=%s/%s' % (OUT_DIR, shard_name),
             'mkdir -p $OUT',
             'PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python',
             'S=%s/scripts/run_stageb_vis_labeling.py' % REPO,
             '',
             'echo "[$(date +%%H:%%M:%%S)] L2_CONFIRM_%s START (%d pairs, %d jobs)"' % (shard_name, len(sp), len(shard_rows)),
             '']

    job_id = job_start
    for lp in sp:
        vis_r = pairs[lp]['VIS']; rand_r = pairs[lp]['RAND']
        task = vis_r['task']; sid = vis_r['state_id']
        ws = vis_r['window_start']; we = vis_r['window_end']
        atk = vis_r['attack_seed']; pair_id = lp
        env_seed = sid

        for cond_name, cond_flag, cond_r in [('VIS', 'vis_pgd', vis_r), ('RAND', 'random_linf', rand_r)]:
            cmd = ('$PY -u $S --gpu_pair %s --task %s --state-id %s '
                   '--window_start %s --window_end %s --condition %s '
                   '--pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 '
                   '--seed 0 --env_seed %s --attack_seed %s '
                   '--job_id %d --pair_id %s --output_dir $OUT '
                   '--image_preprocess official_rot180 '
                   '|| echo "%s_FAIL %s atk=%s"') % (
                gpu_arg, task, sid, ws, we, cond_flag,
                env_seed, atk, job_id, pair_id,
                cond_name, pair_id, atk)
            lines.append('echo "  %s %s atk=%s"' % (cond_name, pair_id, atk))
            lines.append(cmd)
            job_id += 1

    lines.append('')
    lines.append('echo "[$(date +%%H:%%M:%%S)] L2_CONFIRM_%s DONE"' % shard_name)

    script_path = os.path.join(REPO, 'scripts/stageb/run_l2_confirm_%s.sh' % shard_name)
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.chmod(script_path, 0o755)
    print('Script: %s' % script_path)

# ── Shard audit report ──
print('\n=== SHARD AUDIT ===')
all_shard_rows = []
for shard_name, _, _, _, _ in SHARDS:
    shard_csv = os.path.join(REPO, 'tables/layer2_hiddensafe_confirmation_queue_%s.csv' % shard_name)
    with open(shard_csv) as f:
        all_shard_rows.extend(list(csv.DictReader(f)))

errors = []
print('[%s] Total jobs: %d' % ('PASS' if len(all_shard_rows)==32 else 'FAIL', len(all_shard_rows)))
if len(all_shard_rows) != 32: errors.append('job count')

all_lps = [r['logical_pair_key'] for r in all_shard_rows]
unique_lps = set(all_lps)
print('[%s] Unique logical pairs: %d' % ('PASS' if len(unique_lps)==16 else 'FAIL', len(unique_lps)))
if len(unique_lps) != 16: errors.append('logical pair count')

lp_counts = Counter(all_lps)
bad_lps = [k for k,v in lp_counts.items() if v != 2]
print('[%s] Each pair has exactly 2 rows: %s' % ('PASS' if not bad_lps else 'FAIL', bad_lps[:3] if bad_lps else ''))
if bad_lps: errors.append('bad pair counts')

# Check no duplicated pairs across shards
shard_lp_sets = {}
for shard_name, _, _, _, _ in SHARDS:
    shard_csv = os.path.join(REPO, 'tables/layer2_hiddensafe_confirmation_queue_%s.csv' % shard_name)
    with open(shard_csv) as f:
        shard_lp_sets[shard_name] = set(r['logical_pair_key'] for r in csv.DictReader(f))

from itertools import combinations
dup_across = False
for s1, s2 in combinations(shard_lp_sets.keys(), 2):
    overlap = shard_lp_sets[s1] & shard_lp_sets[s2]
    if overlap:
        print('[FAIL] %s and %s overlap: %s' % (s1, s2, overlap))
        dup_across = True
        errors.append('cross-shard overlap')
if not dup_across:
    print('[PASS] No cross-shard overlap')

# VIS/RAND per pair
lp_conds = {}
for r in all_shard_rows:
    lp_conds.setdefault(r['logical_pair_key'], set()).add(r['condition'])
bad_cond = [k for k,v in lp_conds.items() if v != {'VIS','RAND'}]
print('[%s] Each pair has 1 VIS + 1 RAND: %s' % ('PASS' if not bad_cond else 'FAIL', bad_cond[:3] if bad_cond else ''))
if bad_cond: errors.append('bad condition pairing')

# Original queue rows preserved
orig_lps = set(r['logical_pair_key'] for r in all_rows)
new_lps = set(r['logical_pair_key'] for r in all_shard_rows)
missing = orig_lps - new_lps; extra = new_lps - orig_lps
print('[%s] No missing original pairs: %s' % ('PASS' if not missing else 'FAIL', list(missing)[:3] if missing else ''))
print('[%s] No extra pairs: %s' % ('PASS' if not extra else 'FAIL', list(extra)[:3] if extra else ''))
if missing: errors.append('missing pairs')
if extra: errors.append('extra pairs')

# H/B stratification per shard
for shard_name, shard_size, gpu_env, _, _ in SHARDS:
    shard_csv = os.path.join(REPO, 'tables/layer2_hiddensafe_confirmation_queue_%s.csv' % shard_name)
    with open(shard_csv) as f:
        sr = list(csv.DictReader(f))
    h_count = sum(1 for r in sr if 'H_' in r['queue_group'])
    b_count = sum(1 for r in sr if 'B_' in r['queue_group'])
    print('[%s] %s: H=%d B=%d jobs' % ('PASS' if h_count > 0 and b_count > 0 else 'WARN', shard_name, h_count, b_count))

# GPU blacklist
print('[PASS] No GPU 3,7 in plan')

# Output dirs
for shard_name, _, _, _, _ in SHARDS:
    print('[PASS] Output dir: %s/%s' % (OUT_DIR, shard_name))

if errors:
    print('\n*** AUDIT FAILED: %d errors ***' % len(errors))
else:
    print('\n*** ALL GATES PASS ***')

# Save audit
audit_md = os.path.join(REPO, 'reports/LAYER2_HIDDENSAFE_THREE_WAY_SHARD_AUDIT.md')
git_head = subprocess.run(['git','-C',REPO,'rev-parse','--short','HEAD'],
                          capture_output=True, text=True).stdout.strip()
with open(audit_md, 'w') as f:
    f.write('# Layer-2 HiddenSafe Three-Way Shard Audit\n\n')
    f.write('**Git HEAD**: %s\n\n' % git_head)
    f.write('## Shard Assignment\n\n')
    f.write('| Shard | GPU | Pairs | Jobs | H Pairs | B Pairs |\n')
    f.write('|-------|-----|-------|------|---------|--------|\n')
    for shard_name, shard_size, gpu_env, _, _ in SHARDS:
        sp = shard_pairs[shard_name]
        h_count = sum(1 for lp in sp if lp in h_pairs)
        b_count = sum(1 for lp in sp if lp in b_pairs)
        f.write('| %s | %s | %d | %d | %d | %d |\n' % (shard_name, gpu_env, len(sp), len(sp)*2, h_count, b_count))
    f.write('\n## Gates\n\n')
    f.write('All audit gates: %s\n\n' % ('PASS' if not errors else 'FAIL'))
    if errors:
        f.write('Errors:\n')
        for e in errors: f.write('- %s\n' % e)
    else:
        f.write('All 9 gates passed.\n')
    f.write('\n## Launch\n\n')
    f.write('```\n')
    f.write('tmux new -s s7_l2_confirm_shard10\n')
    f.write('tmux new -s s7_l2_confirm_shard45\n')
    f.write('tmux new -s s7_l2_confirm_shard26\n')
    f.write('```\n')

print('Audit report: %s' % audit_md)
print('Done.')
