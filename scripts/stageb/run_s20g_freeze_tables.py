#!/usr/bin/env python3
"""S20G: Freeze paired label tables with QA audit."""
import json, glob, csv, os
from collections import Counter, defaultdict

OUT = '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611'
TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(TABLES, exist_ok=True)

all_rand = {}; all_vis = {}
rand_dirs = [
    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
    '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
]
vis_dirs = rand_dirs + [OUT]

for d in rand_dirs:
    for f in glob.glob(d + '/summary_*random_linf*.json'):
        s = json.load(open(f))
        key = (s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed', s.get('seed', '0'))))
        all_rand[key] = s
for d in vis_dirs:
    for f in glob.glob(d + '/summary_*vis_pgd*.json'):
        s = json.load(open(f))
        key = (s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed', s.get('seed', '0'))))
        all_vis[key] = s

held_out = {('tomato_sauce', '0', 70, 80), ('ketchup', '0', 150, 160)}

# Universe for phase
universe = {}
with open('/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv') as f:
    for r in csv.DictReader(f):
        universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

# Paired label table
paired = []
for key, r in all_rand.items():
    task, sid, ws, we, seed = key
    if (task, sid, ws, we) in held_out: continue
    if key not in all_vis: continue
    v = all_vis[key]

    phase_info = universe.get((task, sid, ws, we), {})
    phase = phase_info.get('phase_id', 'unknown')
    fc = phase_info.get('first_close_step', ''); lift = phase_info.get('lift_step', '')

    r_open = r['decoded_open_count']; r_streak = r['max_open_streak']
    r_done = r['success_done_any']; r_timeout = r.get('timeout', False)
    v_open = v['decoded_open_count']; v_streak = v['max_open_streak']
    v_done = v['success_done_any']; v_timeout = v.get('timeout', False)

    if r_done and not r_timeout and r_open <= 3 and r_streak <= 3:
        label_quality = 'STRICT'
    elif r_done and not r_timeout and r_open <= 5:
        label_quality = 'USABLE'
    else:
        label_quality = 'BORDERLINE'

    if r_timeout or not r_done:
        cls = 'random_sensitive'
    elif v_timeout or not v_done:
        cls = 'task_effect' if v_open > r_open + 3 else 'contact_effect_weak'
    elif v_open > r_open + 2:
        cls = 'cmd_specific'
    elif v_open > r_open:
        cls = 'cmd_borderline'
    else:
        cls = 'no_effect'

    paired.append({
        'task': task, 'state_id': sid, 'window_start': ws, 'window_end': we, 'seed': seed,
        'phase': phase, 'first_close_step': fc, 'lift_step': lift,
        'label_quality': label_quality,
        'rand_open': r_open, 'rand_streak': r_streak, 'rand_done': r_done, 'rand_timeout': r_timeout, 'rand_steps': r['n_steps'],
        'vis_open': v_open, 'vis_streak': v_streak, 'vis_done': v_done, 'vis_timeout': v_timeout, 'vis_steps': v['n_steps'],
        'open_delta': v_open - r_open, 'streak_delta': v_streak - r_streak,
        'classification': cls,
    })

with open(TABLES + '/s20g_v031_paired_label_table.csv', 'w', newline='') as f:
    fields = ['task','state_id','window_start','window_end','seed','phase','first_close_step','lift_step',
              'label_quality','rand_open','rand_streak','rand_done','rand_timeout','rand_steps',
              'vis_open','vis_streak','vis_done','vis_timeout','vis_steps','open_delta','streak_delta','classification']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(paired)
print('Paired labels: %d' % len(paired))
print('Classifications: %s' % dict(Counter(p['classification'] for p in paired)))

# Phase coverage
phase_data = defaultdict(lambda: {'paired':0,'task_effect':0,'cmd_specific':0,'random_sensitive':0})
for p in paired:
    ph = p['phase']
    phase_data[ph]['paired'] += 1
    if p['classification'] == 'task_effect': phase_data[ph]['task_effect'] += 1
    if p['classification'] == 'cmd_specific': phase_data[ph]['cmd_specific'] += 1
    if 'random_sensitive' in p['classification']: phase_data[ph]['random_sensitive'] += 1

with open(TABLES + '/s20g_v031_phase_coverage.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['phase','paired','task_effect','cmd_specific','random_sensitive'])
    for ph in ['approach','grasp_transition','early_transport','transport','preplace','place_or_done']:
        d = phase_data.get(ph, {})
        w.writerow([ph, d.get('paired',0), d.get('task_effect',0), d.get('cmd_specific',0), d.get('random_sensitive',0)])
print('Phase coverage: %s' % {k: v['paired'] for k, v in phase_data.items()})

# Label quality audit
with open(TABLES + '/s20g_v031_label_quality_audit.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['label_quality','count','task_effect','cmd_specific','random_sensitive','no_effect'])
    for q in ['STRICT','USABLE','BORDERLINE']:
        subset = [p for p in paired if p['label_quality'] == q]
        w.writerow([q, len(subset),
                    sum(1 for p in subset if p['classification']=='task_effect'),
                    sum(1 for p in subset if p['classification']=='cmd_specific'),
                    sum(1 for p in subset if 'random_sensitive' in p['classification']),
                    sum(1 for p in subset if p['classification']=='no_effect')])
print('Quality: %s' % dict(Counter(p['label_quality'] for p in paired)))

# VIS effect summary
with open(TABLES + '/s20g_v031_vis_effect_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['classification','count','mean_open_delta','mean_streak_delta'])
    for cls in ['task_effect','cmd_specific','cmd_borderline','no_effect','random_sensitive']:
        subset = [p for p in paired if p['classification'] == cls]
        if subset:
            w.writerow([cls, len(subset),
                        sum(p['open_delta'] for p in subset)/len(subset),
                        sum(p['streak_delta'] for p in subset)/len(subset)])

# Leakage/duplicate audit
leakage_issues = []
for key in all_rand:
    task, sid, ws, we, seed = key
    if (task, sid, ws, we) in held_out:
        leakage_issues.append('HELD_OUT_IN_RAND: %s s%s w%d-%d' % (task, sid, ws, we))
for key in all_vis:
    task, sid, ws, we, seed = key
    if (task, sid, ws, we) in held_out:
        leakage_issues.append('HELD_OUT_IN_VIS: %s s%s w%d-%d' % (task, sid, ws, we))

rand_dupes = [k for k, v in Counter(all_rand.keys()).items() if v > 1]
vis_dupes = [k for k, v in Counter(all_vis.keys()).items() if v > 1]
for k in rand_dupes:
    leakage_issues.append('DUPLICATE_RAND: %s s%s w%d-%d seed=%s' % k)
for k in vis_dupes:
    leakage_issues.append('DUPLICATE_VIS: %s s%s w%d-%d seed=%s' % k)

with open(TABLES + '/s20g_v031_leakage_duplicate_audit.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['issue'])
    for issue in leakage_issues:
        w.writerow([issue])
    if not leakage_issues:
        w.writerow(['CLEAN - no leakage or duplicates found'])

print('Leakage issues: %d' % len(leakage_issues))
for i in leakage_issues:
    print('  %s' % i)

print('\nAll tables written to %s' % TABLES)
