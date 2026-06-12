#!/usr/bin/env python3
"""S20M4a RAND-stability audit: classify parents, write stability tables.
Part of Freeze A — execution/provenance freeze. Not a claim freeze."""
import csv, json, glob, os, sys
from collections import Counter, defaultdict

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
OUT = '/data/liuyu/outputs/stageb_s20m4_rand_stability_20260613'
MANIFEST = T + '/s20m4a_rand_stability_manifest.csv'

# Load manifest
manifest = {}
with open(MANIFEST) as f:
    for r in csv.DictReader(f):
        manifest[r['candidate_id']] = r

# Load all summaries, group by candidate
candidates = defaultdict(list)
for f in sorted(glob.glob(OUT+'/summary_*.json')):
    try:
        s = json.load(open(f))
    except:
        continue
    cid = '{task}_s{sid}_w{ws}_{we}'.format(
        task=s['task'], sid=s['state_id'],
        ws=s['window_start'], we=s['window_end'])
    o = s['decoded_open_count']; st = s['max_open_streak']
    dflag = s['success_done_any']; to = s.get('timeout', False)
    if to or not dflag:
        label = 'RANDOM_SENSITIVE'
    elif o <= 3 and st <= 3:
        label = 'RAND_STRICT'
    elif o <= 5 and st <= 5:
        label = 'RAND_USABLE'
    else:
        label = 'BORDERLINE'
    candidates[cid].append({
        'seed': str(s.get('attack_seed','?')), 'open': o, 'streak': st,
        'done': dflag, 'timeout': to, 'label': label, 'steps': s['n_steps'],
        'job_id': s.get('job_id','?'),
    })

# ── Classification ──
# Protocol-strict: 3/3 seeds STRICT (open<=3), no timeout
# Protocol-usable: >=2/3 STRICT or USABLE, no RS, no timeout
# Unstable: everything else

protocol_strict = []
protocol_usable = []
unstable = []
incomplete = []
total_completed = 0

for cid, seeds in sorted(candidates.items()):
    m = manifest.get(cid, {})
    n = len(seeds)
    total_completed += n
    labels = [s['label'] for s in seeds]

    if n < 3:
        incomplete.append((cid, m, seeds))
        continue

    n_strict = sum(1 for l in labels if l == 'RAND_STRICT')
    n_usable = sum(1 for l in labels if l == 'RAND_USABLE')
    n_clean = n_strict + n_usable
    n_rs = sum(1 for l in labels if l == 'RANDOM_SENSITIVE')
    n_border = sum(1 for l in labels if l == 'BORDERLINE')

    opens = [s['open'] for s in seeds]
    streaks = [s['streak'] for s in seeds]
    max_open = max(opens)
    max_streak = max(streaks)
    timeouts = sum(1 for s in seeds if s['timeout'])

    if n_strict >= 3 and timeouts == 0:
        protocol_strict.append((cid, m, seeds, max_open, max_streak))
    elif n_clean >= 2 and n_rs == 0 and timeouts == 0:
        protocol_usable.append((cid, m, seeds, n_clean, max_open, max_streak, n_strict, n_usable, n_border))
    else:
        unstable.append((cid, m, seeds, n_clean, n_rs, n_border, timeouts, max_open))

# ── Write stability outcome audit ──
with open(T+'/s20m4a_rand_stability_outcome_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'candidate_id','task','phase','tier','n_seeds',
        'seed96_open','seed96_streak','seed96_label',
        'seed97_open','seed97_streak','seed97_label',
        'seed98_open','seed98_streak','seed98_label',
        'max_open','max_streak','n_strict','n_usable','n_border','n_rs','n_timeouts',
        'stability_class','vis_eligible'])
    w.writeheader()

    all_classified = protocol_strict + protocol_usable + unstable + incomplete
    for item in all_classified:
        if len(item) == 4:  # incomplete: (cid, m, seeds)
            cid, m, seeds = item
            n_seeds = len(seeds)
            row = {'candidate_id': cid, 'task': m.get('task',''), 'phase': m.get('phase',''),
                   'tier': m.get('tier',''), 'n_seeds': n_seeds, 'stability_class': 'INCOMPLETE'}
            for s in seeds:
                row['seed%s_open'%s['seed']] = s['open']
                row['seed%s_streak'%s['seed']] = s['streak']
                row['seed%s_label'%s['seed']] = s['label']
            row['vis_eligible'] = False
            w.writerow(row)
        elif len(item) == 5 and isinstance(item[3], int):  # protocol_strict: (cid, m, seeds, max_open, max_streak)
            cid, m, seeds, max_open, max_streak = item
            row = {'candidate_id': cid, 'task': m.get('task',''), 'phase': m.get('phase',''),
                   'tier': m.get('tier',''), 'n_seeds': 3,
                   'max_open': max_open, 'max_streak': max_streak,
                   'n_strict': 3, 'n_usable': 0, 'n_border': 0, 'n_rs': 0, 'n_timeouts': 0,
                   'stability_class': 'PROTOCOL_STRICT', 'vis_eligible': True}
            for s in seeds:
                row['seed%s_open'%s['seed']] = s['open']
                row['seed%s_streak'%s['seed']] = s['streak']
                row['seed%s_label'%s['seed']] = s['label']
            w.writerow(row)
        elif len(item) == 9:  # protocol_usable
            cid, m, seeds, n_clean, max_open, max_streak, n_strict, n_usable, n_border = item
            row = {'candidate_id': cid, 'task': m.get('task',''), 'phase': m.get('phase',''),
                   'tier': m.get('tier',''), 'n_seeds': 3,
                   'max_open': max_open, 'max_streak': max_streak,
                   'n_strict': n_strict, 'n_usable': n_usable, 'n_border': n_border,
                   'n_rs': 0, 'n_timeouts': 0,
                   'stability_class': 'PROTOCOL_USABLE', 'vis_eligible': True}
            for s in seeds:
                row['seed%s_open'%s['seed']] = s['open']
                row['seed%s_streak'%s['seed']] = s['streak']
                row['seed%s_label'%s['seed']] = s['label']
            w.writerow(row)
        elif len(item) == 8:  # unstable
            cid, m, seeds, n_clean, n_rs, n_border, timeouts, max_open = item
            n_strict = sum(1 for s in seeds if s['label']=='RAND_STRICT')
            n_usable = sum(1 for s in seeds if s['label']=='RAND_USABLE')
            max_streak = max(s['streak'] for s in seeds)
            row = {'candidate_id': cid, 'task': m.get('task',''), 'phase': m.get('phase',''),
                   'tier': m.get('tier',''), 'n_seeds': 3,
                   'max_open': max_open, 'max_streak': max_streak,
                   'n_strict': n_strict, 'n_usable': n_usable, 'n_border': n_border,
                   'n_rs': n_rs, 'n_timeouts': timeouts,
                   'stability_class': 'UNSTABLE', 'vis_eligible': False}
            for s in seeds:
                row['seed%s_open'%s['seed']] = s['open']
                row['seed%s_streak'%s['seed']] = s['streak']
                row['seed%s_label'%s['seed']] = s['label']
            w.writerow(row)

# ── Write parent stability summary ──
n_total = len(protocol_strict) + len(protocol_usable) + len(unstable)
with open(T+'/s20m4a_parent_stability_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['stability_class','count','pct','vis_eligible_count'])
    w.writeheader()
    for cls, items, eligible in [
        ('PROTOCOL_STRICT', protocol_strict, True),
        ('PROTOCOL_USABLE', protocol_usable, True),
        ('UNSTABLE', unstable, False),
        ('INCOMPLETE', incomplete, False)]:
        w.writerow({'stability_class': cls, 'count': len(items),
                    'pct': round(len(items)/max(n_total,1)*100, 1) if cls != 'INCOMPLETE' else '',
                    'vis_eligible_count': len(items) if eligible else 0})

# ── Print summary ──
print('='*60)
print('S20M4a RAND-STABILITY AUDIT')
print('='*60)
print('Total summaries: %d' % total_completed)
print('Candidates: %d completed + %d incomplete = %d total' %
      (len(protocol_strict)+len(protocol_usable)+len(unstable), len(incomplete),
       len(protocol_strict)+len(protocol_usable)+len(unstable)+len(incomplete)))
print()
print('PROTOCOL_STRICT (3/3 STRICT, no timeout): %d' % len(protocol_strict))
for cid, m, seeds, mo, ms in protocol_strict:
    print('  %s  phase=%s opens=%s' % (cid, m.get('phase','?'),
          '/'.join(str(s['open']) for s in seeds)))
print()
print('PROTOCOL_USABLE (>=2/3 clean, no RS): %d' % len(protocol_usable))
for cid, m, seeds, nc, mo, ms, ns, nu, nb in sorted(protocol_usable, key=lambda x: -x[3]):
    print('  %s  phase=%s clean=%d/3 opens=%s' % (cid, m.get('phase','?'), nc,
          '/'.join(str(s['open']) for s in seeds)))
print()
print('UNSTABLE: %d' % len(unstable))
for cid, m, seeds, nc, nrs, nb, t, mo in sorted(unstable, key=lambda x: x[4]+x[5]):
    print('  %s  phase=%s opens=%s labels=%s' % (cid, m.get('phase','?'),
          '/'.join(str(s['open']) for s in seeds),
          '/'.join(s['label'][:4] for s in seeds)))
print()
print('INCOMPLETE: %d' % len(incomplete))

vis_candidates = len(protocol_strict) + len(protocol_usable)
print()
print('VIS-eligible: %d candidates (STRICT=%d + USABLE=%d)' %
      (vis_candidates, len(protocol_strict), len(protocol_usable)))
print()
print('Tables written:')
print('  %s/s20m4a_rand_stability_outcome_audit.csv' % T)
print('  %s/s20m4a_parent_stability_summary.csv' % T)
