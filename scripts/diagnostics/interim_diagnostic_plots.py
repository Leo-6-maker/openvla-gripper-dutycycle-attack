#!/usr/bin/env python3
"""CPU-only diagnostic plots from interim label data. No torch, no CUDA."""
import csv, json, os, sys
from collections import Counter

SMOKE_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827'
EXP_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827'
INTERIM_DIR = '/data/liuyu/outputs/stageb_v1_1_targeted_expansion_rc1a_d4a3827_interim'

# Load all pair summaries
all_pairs = []
for sdir in [SMOKE_DIR, EXP_DIR]:
    if not os.path.isdir(sdir):
        continue
    pairs = {}
    for f in sorted(os.listdir(sdir)):
        if not f.startswith('summary_') or not f.endswith('.json'):
            continue
        with open(os.path.join(sdir, f)) as fp:
            d = json.load(fp)
        pid = d.get('pair_id', '?')
        c = 'VIS' if '_vis_pgd_' in f else 'RAND'
        pairs.setdefault(pid, {})[c] = d
    for pid, p in pairs.items():
        if 'VIS' in p and 'RAND' in p:
            all_pairs.append({
                'pair_id': pid,
                'source': 'smoke' if 'smoke_' in pid else 'expansion',
                'task': p['VIS'].get('task_key', '?'),
                'vis_open': p['VIS'].get('decoded_open_count', 0),
                'vis_streak': p['VIS'].get('decoded_longest_open_streak', 0),
                'vis_qpos': p['VIS'].get('qpos_delta', 0),
                'rand_open': p['RAND'].get('decoded_open_count', 0),
                'rand_streak': p['RAND'].get('decoded_longest_open_streak', 0),
                'rand_qpos': p['RAND'].get('qpos_delta', 0),
                'vis_steps': p['VIS'].get('n_total_steps', 0),
            })

if not all_pairs:
    print('No complete pairs yet.')
    sys.exit(0)

# Text-based diagnostic output (no matplotlib needed)
print('=== DIAGNOSTIC PLOTS (text) ===')
print('Pairs: %d (smoke=%d expansion=%d)' % (
    len(all_pairs),
    sum(1 for p in all_pairs if p['source'] == 'smoke'),
    sum(1 for p in all_pairs if p['source'] == 'expansion'),
))

# 1. VIS vs RAND open_count scatter
print('\n--- VIS vs RAND open_count ---')
for p in sorted(all_pairs, key=lambda x: -max(x['vis_open'], x['rand_open'])):
    bar = 'V' * min(p['vis_open'], 15) + '  |  ' + 'R' * min(p['rand_open'], 15)
    print('%-50s VIS=%2d RAND=%2d  %s' % (
        '%s_%s' % (p['task'][:12], p['pair_id'].split('_s')[1][:15] if '_s' in p['pair_id'] else '?'),
        p['vis_open'], p['rand_open'], bar))

# 2. VIS vs RAND qpos
print('\n--- VIS vs RAND |qpos| ---')
for p in sorted(all_pairs, key=lambda x: -max(abs(x['vis_qpos']), abs(x['rand_qpos']))):
    vq = abs(p['vis_qpos']); rq = abs(p['rand_qpos'])
    vc = '!' if vq >= 0.01 else '.'
    rc = '!' if rq >= 0.01 else '.'
    print('%-50s VIS=%.4f %s  RAND=%.4f %s' % (
        '%s_%s' % (p['task'][:12], p['pair_id'].split('_s')[1][:15] if '_s' in p['pair_id'] else '?'),
        vq, vc, rq, rc))

# 3. Category distribution by task
print('\n--- Label category by task ---')
task_labels = {}
for p in all_pairs:
    vo = p['vis_open']; ro = p['rand_open']
    vq = abs(p['vis_qpos']); rq = abs(p['rand_qpos'])
    vs = p['vis_steps']

    if vs <= 50: label = 'unstable'
    elif vo >= 6 and ro < 6: label = 'cmd_specific'
    elif ro >= 6 and vo < 6: label = 'rand_cmd'
    elif vo >= 6 and ro >= 6: label = 'confounded'
    elif vq >= 0.01 and rq < 0.01: label = 'vis_phys'
    elif rq >= 0.01 and vq < 0.01: label = 'rand_phys'
    elif vq >= 0.01 and rq >= 0.01: label = 'shared_qpos'
    else: label = 'negative'

    task_labels.setdefault(p['task'], Counter())[label] += 1

for tk in sorted(task_labels):
    parts = ' '.join('%s=%d' % (l, c) for l, c in task_labels[tk].most_common())
    print('  %-20s %s' % (tk, parts))

# 4. Hard_neg_candidate conversion
print('\n--- Hard_neg_candidate conversions ---')
for p in all_pairs:
    if 'hard_neg' in p['pair_id']:
        vo = p['vis_open']; ro = p['rand_open']
        vq = abs(p['vis_qpos']); rq = abs(p['rand_qpos'])
        if vo >= 6 or ro >= 6:
            print('  SURPRISE: %s VIS=%d RAND=%d Vq=%.4f Rq=%.4f' %
                  (p['pair_id'][:50], vo, ro, vq, rq))
        elif vo == 0 and ro == 0 and vq < 0.005 and rq < 0.005:
            print('  CONFIRMED: %s (zero VIS, zero RAND)' % p['pair_id'][:50])
        else:
            print('  UNCERTAIN: %s VIS=%d RAND=%d Vq=%.4f Rq=%.4f' %
                  (p['pair_id'][:50], vo, ro, vq, rq))

# 5. Summary stats
print('\n--- Summary Stats ---')
print('Total complete pairs: %d' % len(all_pairs))
print('cmd_specific (VIS>=6,RAND<6): %d' %
      sum(1 for p in all_pairs if p['vis_open'] >= 6 and p['rand_open'] < 6))
print('rand_cmd (RAND>=6,VIS<6): %d' %
      sum(1 for p in all_pairs if p['rand_open'] >= 6 and p['vis_open'] < 6))
print('confounded (both>=6): %d' %
      sum(1 for p in all_pairs if p['vis_open'] >= 6 and p['rand_open'] >= 6))
print('vis_specific_phys: %d' %
      sum(1 for p in all_pairs if abs(p['vis_qpos']) >= 0.01 and abs(p['rand_qpos']) < 0.01))
print('rand_phys: %d' %
      sum(1 for p in all_pairs if abs(p['rand_qpos']) >= 0.01 and abs(p['vis_qpos']) < 0.01))
print('shared_qpos (both>=0.01): %d' %
      sum(1 for p in all_pairs if abs(p['vis_qpos']) >= 0.01 and abs(p['rand_qpos']) >= 0.01))
print('unstable (<=50 steps): %d' %
      sum(1 for p in all_pairs if p['vis_steps'] <= 50))
print('RAND opens still 0: %d/%d' %
      (sum(1 for p in all_pairs if p['rand_open'] == 0), len(all_pairs)))

# Save to text file
report_path = os.path.join(INTERIM_DIR, 'diagnostic_summary.txt')
with open(report_path, 'w') as f:
    f.write('Interim diagnostic summary — %d pairs\n' % len(all_pairs))
    f.write('Generated CPU-only, no GPU\n\n')
    for p in sorted(all_pairs, key=lambda x: -x['vis_open']):
        f.write('%-60s VIS=%d/%d q=%.5f  RAND=%d/%d q=%.5f  steps=%d\n' %
                (p['pair_id'], p['vis_open'], p['vis_streak'], p['vis_qpos'],
                 p['rand_open'], p['rand_streak'], p['rand_qpos'], p['vis_steps']))
print('\nReport: %s' % report_path)
