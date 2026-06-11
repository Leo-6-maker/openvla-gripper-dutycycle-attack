#!/usr/bin/env python3
"""S20I audit: duplicate check, coverage, exclusion, refill queues."""
import csv, json, glob, os
from collections import defaultdict, Counter

QUEUES_DIR = '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612/queues'
TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'

seen = defaultdict(list)
all_jobs = []
for q in sorted(glob.glob(QUEUES_DIR + '/s20i_*.csv')):
    with open(q) as f:
        for r in csv.DictReader(f):
            key = (r['task'], r['state_id'], r['window_start'], r['window_end'], r['attack_seed'], r['condition'])
            seen[key].append({'queue': os.path.basename(q), 'job_id': r['job_id'], 'track': r.get('track','?'), 'status': r.get('status','?')})
            all_jobs.append(r)

dupes = {k: v for k, v in seen.items() if len(v) > 1}
print('=== DUPLICATE AUDIT ===')
print('Total unique keys: %d, Duplicate keys: %d' % (len(seen), len(dupes)))
for k, v in list(dupes.items())[:20]:
    print('  %s s%s w%s-%s seed=%s %s' % (k[0], k[1], k[2], k[3], k[4], k[5]))
    for x in v:
        print('    %s job=%s track=%s' % (x['queue'], x['job_id'], x['track']))

with open(TABLES + '/s20i_queue_duplicate_audit.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task','state_id','window_start','window_end','attack_seed','condition','count','queues','job_ids'])
    for k, v in dupes.items():
        w.writerow([k[0], k[1], k[2], k[3], k[4], k[5], len(v),
                    ';'.join(x['queue'] for x in v), ';'.join(x['job_id'] for x in v)])

# Track C coverage
track_c = [j for j in all_jobs if j.get('track') == 'C' and j['condition'] == 'random_linf']
task_counts = Counter(j['task'] for j in track_c)
print('\n=== TRACK C COVERAGE ===')
print('Tasks: %d, candidates: %d' % (len(task_counts), len(track_c)))
for t, n in task_counts.most_common():
    print('  %s: %d' % (t, n))

with open(TABLES + '/s20i_trackC_task_phase_coverage.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task','phase','n'])
    for t in sorted(task_counts):
        phases = Counter(j.get('tier','?') for j in track_c if j['task'] == t)
        for p, n in sorted(phases.items()):
            w.writerow([t, p, n])

# Known-parent exclusion
held_out = {('tomato_sauce', '0', '70', '80'), ('ketchup', '0', '150', '160')}
excluded_found = [j for j in all_jobs if (j['task'], j['state_id'], j['window_start'], j['window_end']) in held_out]
print('\n=== HELD-OUT AUDIT ===')
if excluded_found:
    print('FOUND (need train_exclude=true):')
    for j in excluded_found:
        print('  %s s%s w%s-%s job=%s track=%s' % (j['task'], j['state_id'], j['window_start'], j['window_end'], j['job_id'], j.get('track','?')))
else:
    print('CLEAN')

# Build refill queues with parent-level exclusion
reserved_keys = set(seen.keys())  # seed+condition level
reserved_parents = set()  # parent level (task, state, ws, we) — exclude from fresh broad
for k in seen:
    reserved_parents.add((k[0], k[1], int(k[2]), int(k[3])))

for d in ['/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
          '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
          '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
          '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612']:
    for f in glob.glob(d + '/summary_*.json'):
        s = json.load(open(f))
        reserved_keys.add((s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed','0')), s['condition']))
        reserved_parents.add((s['task'], str(s['state_id']), int(s['window_start']), int(s['window_end'])))

universe = {}
with open('/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv') as f:
    for r in csv.DictReader(f):
        universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

priority_phases = ['grasp_transition', 'early_transport', 'transport', 'preplace']
priority_tasks = ['tomato_sauce', 'ketchup', 'milk', 'orange_juice', 'bbq_sauce', 'salad_dressing', 'cream_cheese', 'butter', 'alphabet_soup', 'chocolate_pudding']

# Round-robin: per phase, per task, pick fresh parent NOT in any prior queue/summary
refill_cands = []
seen_cands = set()
for phase in priority_phases:
    for task in priority_tasks:
        for (t, sid, ws, we), u in universe.items():
            if t != task: continue
            if (t, sid, ws, we) in held_out: continue
            if u.get('phase_id', '?') != phase: continue
            cid = '%s_s%s_w%d_%d' % (t, sid, ws, we)
            if cid in seen_cands: continue
            # Parent-level exclusion: skip if this parent already queued/run
            if (t, str(sid), ws, we) in reserved_parents: continue
            refill_cands.append((t, str(sid), ws, we, phase))
            seen_cands.add(cid)

print('Refill candidates (fresh parents only): %d' % len(refill_cands))
phase_counts = Counter(c[4] for c in refill_cands)
task_counts = Counter(c[0] for c in refill_cands)
print('  Tasks: %d, Phases: %s' % (len(task_counts), dict(phase_counts)))

# Use seed 85 for refill
refill_jobs = []
jid = 220000
for task, sid, ws, we, phase in refill_cands[:30]:
    cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)
    jid += 1; refill_jobs.append({'job_id':str(jid),'task':task,'state_id':sid,'window_start':str(ws),'window_end':str(we),'condition':'random_linf','attack_seed':'85','random_control_seed':'85','seed':'0','candidate_id':cid,'tier':'C_refill_'+phase,'track':'C_refill','status':'pending'})
    jid += 1; refill_jobs.append({'job_id':str(jid),'task':task,'state_id':sid,'window_start':str(ws),'window_end':str(we),'condition':'vis_pgd','attack_seed':'85','random_control_seed':'','seed':'0','candidate_id':cid,'tier':'C_refill_'+phase,'track':'C_refill','status':'pending'})

# Write refill coverage
with open(TABLES + '/s20i_refill_task_phase_coverage.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task','phase','n'])
    for t in sorted(task_counts):
        for p in priority_phases:
            n = sum(1 for c in refill_cands if c[0] == t and c[4] == p)
            if n > 0: w.writerow([t, p, n])

# Re-run duplicate audit including refill
all_refill_keys = [(j['task'], j['state_id'], j['window_start'], j['window_end'], j['attack_seed'], j['condition']) for j in refill_jobs]
refill_dupes_internal = [k for k, v in Counter(all_refill_keys).items() if v > 1]
refill_dupes_vs_main = [k for k in all_refill_keys if k in seen]
print('Refill internal dupes: %d, vs main queues: %d' % (len(refill_dupes_internal), len(refill_dupes_vs_main)))
if refill_dupes_internal:
    print('  INTERNAL DUPES: %s' % refill_dupes_internal)
if refill_dupes_vs_main:
    print('  VS MAIN DUPES: %s' % refill_dupes_vs_main[:5])

# Split across 3 refill queues
refill_qs = {'gpu10': [], 'gpu26': [], 'gpu45': []}
gpus = ['gpu10', 'gpu26', 'gpu45']
pairs = [(refill_jobs[i], refill_jobs[i+1]) for i in range(0, len(refill_jobs), 2)]
for i, (rj, vj) in enumerate(pairs):
    refill_qs[gpus[i % 3]].append(rj)
    refill_qs[gpus[i % 3]].append(vj)

for gpu, jobs in refill_qs.items():
    if not jobs: continue
    qpath = QUEUES_DIR + '/s20i_%s_refill_001.csv' % gpu
    with open(qpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(jobs)
    print('%s refill: %d jobs (%d candidates)' % (gpu, len(jobs), len(jobs)//2))

print('\nRefill total: %d jobs, %d unique candidates' % (len(refill_jobs), len(refill_jobs)//2))
print('Audit done.')
