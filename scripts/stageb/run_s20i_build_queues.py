#!/usr/bin/env python3
"""S20I-DataMax: Build Track A/B/C queues for 9h run."""
import csv, json, glob, os

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
OUT = '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612'
os.makedirs(OUT + '/queues', exist_ok=True)

# Load universe
universe = {}
with open('/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv') as f:
    for r in csv.DictReader(f):
        universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

# Existing keys
existing_keys = set()
for d in [
    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
    '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
    '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
    '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612',
]:
    for f in glob.glob(d + '/summary_*.json'):
        s = json.load(open(f))
        existing_keys.add((s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed', '0'))))

held_out = {('tomato_sauce', '0', 70, 80), ('ketchup', '0', 150, 160)}

jid = 210000

# Track A: tomato_sauce s0 neighborhood
track_a = []
for ws in [50, 60, 70, 80, 90, 100]:
    we = ws + 10
    cid = 'tomato_sauce_s0_w%d_%d' % (ws, we)
    for seed in ['83', '84']:
        jid += 1; track_a.append({'job_id': str(jid), 'task': 'tomato_sauce', 'state_id': '0', 'window_start': str(ws), 'window_end': str(we), 'condition': 'random_linf', 'attack_seed': seed, 'random_control_seed': seed, 'seed': '0', 'candidate_id': cid, 'tier': 'A_neighborhood', 'track': 'A', 'status': 'pending'})
        jid += 1; track_a.append({'job_id': str(jid), 'task': 'tomato_sauce', 'state_id': '0', 'window_start': str(ws), 'window_end': str(we), 'condition': 'vis_pgd', 'attack_seed': seed, 'random_control_seed': '', 'seed': '0', 'candidate_id': cid, 'tier': 'A_neighborhood', 'track': 'A', 'status': 'pending'})

# Track B: tomato_sauce s3, s5 phase-aligned
b_windows = [
    ('tomato_sauce', '3', 60, 70, 'grasp'), ('tomato_sauce', '3', 70, 80, 'early_transport'),
    ('tomato_sauce', '3', 120, 130, 'preplace'), ('tomato_sauce', '5', 60, 70, 'grasp'),
    ('tomato_sauce', '5', 105, 115, 'early_transport'), ('tomato_sauce', '5', 150, 160, 'preplace'),
]
track_b = []
for task, sid, ws, we, phase in b_windows:
    cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)
    for seed in ['83', '84']:
        jid += 1; track_b.append({'job_id': str(jid), 'task': task, 'state_id': sid, 'window_start': str(ws), 'window_end': str(we), 'condition': 'random_linf', 'attack_seed': seed, 'random_control_seed': seed, 'seed': '0', 'candidate_id': cid, 'tier': 'B_cross_%s' % phase, 'track': 'B', 'status': 'pending'})
        jid += 1; track_b.append({'job_id': str(jid), 'task': task, 'state_id': sid, 'window_start': str(ws), 'window_end': str(we), 'condition': 'vis_pgd', 'attack_seed': seed, 'random_control_seed': '', 'seed': '0', 'candidate_id': cid, 'tier': 'B_cross_%s' % phase, 'track': 'B', 'status': 'pending'})

# Track C: broad balanced (30 diverse candidates, seed83)
priority_phases = ['grasp_transition', 'early_transport', 'transport', 'preplace']
priority_tasks = ['tomato_sauce', 'ketchup', 'milk', 'orange_juice', 'bbq_sauce', 'salad_dressing', 'cream_cheese', 'butter', 'alphabet_soup', 'chocolate_pudding']
task_order = {t: i for i, t in enumerate(priority_tasks)}

c_candidates = []
for (task, sid, ws, we), u in universe.items():
    if (task, sid, ws, we) in held_out: continue
    if int(sid) not in (0, 1, 3, 5): continue
    phase = u.get('phase_id', '?')
    if phase not in priority_phases: continue
    if (task, str(sid), ws, we, '83') in existing_keys: continue
    c_candidates.append((task, str(sid), ws, we, phase))

c_candidates.sort(key=lambda x: (task_order.get(x[0], 99), ['grasp_transition','early_transport','transport','preplace'].index(x[4]) if x[4] in ['grasp_transition','early_transport','transport','preplace'] else 99))

track_c = []
for task, sid, ws, we, phase in c_candidates[:30]:
    cid = '%s_s%s_w%d_%d' % (task, sid, ws, we)
    jid += 1; track_c.append({'job_id': str(jid), 'task': task, 'state_id': sid, 'window_start': str(ws), 'window_end': str(we), 'condition': 'random_linf', 'attack_seed': '83', 'random_control_seed': '83', 'seed': '0', 'candidate_id': cid, 'tier': 'C_broad_%s' % phase, 'track': 'C', 'status': 'pending'})
    jid += 1; track_c.append({'job_id': str(jid), 'task': task, 'state_id': sid, 'window_start': str(ws), 'window_end': str(we), 'condition': 'vis_pgd', 'attack_seed': '83', 'random_control_seed': '', 'seed': '0', 'candidate_id': cid, 'tier': 'C_broad_%s' % phase, 'track': 'C', 'status': 'pending'})

# Split across GPUs: A→10, B→26, C round-robin across all 3
gpu_queues = {'gpu10': list(track_a), 'gpu26': list(track_b), 'gpu45': []}
c_rand = [j for j in track_c if j['condition'] == 'random_linf']
c_vis = [j for j in track_c if j['condition'] == 'vis_pgd']
for i, (rj, vj) in enumerate(zip(c_rand, c_vis)):
    gpu = ['gpu10', 'gpu26', 'gpu45'][i % 3]
    gpu_queues[gpu].append(rj)
    gpu_queues[gpu].append(vj)

for gpu, jobs in gpu_queues.items():
    qpath = OUT + '/queues/s20i_%s.csv' % gpu
    fields = list(jobs[0].keys())
    with open(qpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(jobs)
    n_rand = sum(1 for j in jobs if j['condition'] == 'random_linf')
    n_vis = sum(1 for j in jobs if j['condition'] == 'vis_pgd')
    tracks = set(j.get('track', '?') for j in jobs)
    print('%s: %d jobs (%d RAND + %d VIS) — Tracks %s' % (gpu, len(jobs), n_rand, n_vis, ','.join(sorted(tracks))))

total = sum(len(j) for j in gpu_queues.values())
print('Total: %d jobs (max %d candidates)' % (total, total // 2))
print('Output: %s' % OUT)
