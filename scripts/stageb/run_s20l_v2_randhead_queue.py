#!/usr/bin/env python3
"""S20L-v2: Load v0.3.1 randhead model, score fresh S20K candidates, build RAND-only queue.
VIS queue generated separately after RAND audit."""
import csv, json, glob, os, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
OUT = '/data/liuyu/outputs/stageb_s20l_v2_randhead_20260613'
os.makedirs(OUT + '/queues', exist_ok=True)

# ── Load trained randhead (same as run_s20i_train_randhead.py) ──
rand_master = list(csv.DictReader(open(TABLES + '/s20i_rand_label_master.csv')))
train_rows = [r for r in rand_master if r['target_random_sensitive'] in ('0','1')]
y_train = np.array([int(r['target_random_sensitive']) for r in train_rows])

# Build training features: phase-aware clean features (no rand/vis leakage)
universe_train = {}
for upath in [TABLES + '/s20i_v031_non_random_sensitive_candidates.csv',
              TABLES + '/s20i_clean_expansion_candidate_universe.csv']:
    if os.path.exists(upath):
        with open(upath) as f:
            for r in csv.DictReader(f):
                universe_train[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

# Close-transition features
trans_audit = {}
for fpath in glob.glob(TABLES + '/s20g_close_transition_audit.csv'):
    with open(fpath) as f:
        for r in csv.DictReader(f):
            trans_audit[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']), r['seed'])] = r

X_train_feats = []
valid_train_idx = []
for i, r in enumerate(train_rows):
    key = (r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))
    u = universe_train.get(key, {})
    if not u: continue
    phase = u.get('phase', u.get('phase_id', '?'))
    fc = float(u.get('first_close_step', -1) or -1); lift = float(u.get('lift_step', -1) or -1)
    ws = int(r['window_start']); we = int(r['window_end']); wc = (ws+we)/2.0
    dl = float(u.get('done_step', 280) or 280)
    t = trans_audit.get((r['task'], r['state_id'], ws, we, r['attack_seed']), {})
    # Phase-aware clean features (no rand/vis leakage)
    feats = [
        fc if fc>0 else -1, lift if lift>0 else -1,
        ws - fc if fc>0 else 50, ws - lift if lift>0 else 50,
        wc / max(dl, 1),
        float(u.get('clean_open_count', 0)), float(u.get('clean_open_frac', 0)),
        float(u.get('post_grasp_open_count', 0)),
        float(u.get('qpos_mean', 0)), float(u.get('eef_disp', 0)),
        float(t.get('distance_to_transition', 0) or 0),
        float(t.get('pre_open_streak', 0) or 0),
        float(t.get('post_close_streak', 0) or 0),
        int(t.get('transition_overlap_center', 0) or 0),
        float(t.get('close_commitment_score', 0.5) or 0.5),
    ]
    X_train_feats.append(feats); valid_train_idx.append(i)

X_train = np.array(X_train_feats); y_train = y_train[valid_train_idx]
ss = StandardScaler(); X_train_s = ss.fit_transform(X_train)
m_rand = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_rand.fit(X_train_s, y_train)
print('Randhead trained: %d samples, pos=%d' % (len(y_train), int(sum(y_train))))

# ── Score all fresh S20K candidates ──
# Load S20K clean summary
clean_dirs = [
    '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean',
    '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612',
    '/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613',
]
WINDOW_LEN = 10; STRIDE = 5

def detect_phases(rows):
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d
    is_open = [g(r, 'decoded_open_bool') for r in rows]
    fc = None; streak = 0
    for i, o in enumerate(is_open):
        if o == 0:
            if streak == 0: fc = i
            streak += 1
            if streak >= 3: break
        else: streak = 0; fc = None
    if streak < 3: fc = None
    base_obj = float(np.median([g(r, 'obj_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    base_eef = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    lift = None
    for i in range(fc or 0, len(rows)):
        if g(rows[i], 'obj_z') - base_obj >= 0.015 or g(rows[i], 'eef_z') - base_eef >= 0.03:
            lift = i; break
    pp = None; peak_z = None; peak_step = None
    if lift is not None:
        for i in range(lift, len(rows)):
            z = g(rows[i], 'eef_z')
            if peak_z is None or z > peak_z: peak_z = z; peak_step = i
            if peak_step is not None and i > peak_step + 3 and peak_z - z >= 0.005:
                pp = i; break
    done = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary','0') or '0') == 1 or int(r.get('success_done','0') or '0') == 1:
            done = i; break
    if done is None: done = len(rows) - 1
    return {'fc': fc, 'lift': lift, 'pp': pp, 'done': done, 'max_steps': len(rows)}

def phase_id(ws, we, ph):
    wc = (ws + we) / 2.0
    if wc >= ph['done'] - 5: return 'place_or_done'
    if ph['pp'] is not None and wc >= ph['pp']: return 'preplace'
    if ph['lift'] is not None and wc >= ph['lift'] + 5: return 'transport'
    if ph['lift'] is not None and wc >= ph['lift']: return 'early_transport'
    if ph['fc'] is not None and wc >= ph['fc']: return 'grasp_transition'
    return 'approach'

held_out = {('tomato_sauce','0',70,80), ('ketchup','0',150,160)}
scored = []

for d in clean_dirs:
    for summary_f in sorted(glob.glob(d + '/summary_*clean*.json')):
        s = json.load(open(summary_f))
        if not s.get('success_done_any', False): continue
        task = s['task']; sid = str(s['state_id'])
        trace_pat = d + '/trace_%s_s%s_w0_10_s20d_clean_seed0_job*.csv' % (task, sid)
        traces = sorted(glob.glob(trace_pat))
        if not traces: continue
        with open(traces[0]) as f: rows = list(csv.DictReader(f))
        ph = detect_phases(rows)
        for ws in range(5, ph['max_steps'] - WINDOW_LEN, STRIDE):
            we = ws + WINDOW_LEN
            if we > ph['done'] + 5: continue
            if (task, sid, ws, we) in held_out: continue
            phase = phase_id(ws, we, ph)
            wc = (ws+we)/2.0
            window_rows = [r for r in rows if ws <= int(r['step']) < we]
            def g(row, key, d=0.0):
                try: return float(row.get(key,d) or d)
                except: return d
            clean_open = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)
            qpos_vals = [g(r, 'gripper_qpos_before') for r in window_rows]
            eef_vals = [g(r, 'eef_z') for r in window_rows]
            post_grasp = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1 and (ph['fc'] is None or int(r['step']) >= ph['fc']))
            fc = ph['fc']; lift = ph['lift']
            feat = np.array([
                fc if fc else -1, lift if lift else -1,
                ws - (fc or ws+50), ws - (lift or ws+50),
                wc/max(ph['done'],1),
                clean_open, clean_open/WINDOW_LEN, post_grasp,
                np.mean(qpos_vals) if qpos_vals else 0,
                max(eef_vals)-min(eef_vals) if len(eef_vals)>=2 else 0,
                0,0,0,0,0.5])  # transition features unknown for fresh
            feat_s = ss.transform([feat])
            p_rand = float(m_rand.predict_proba(feat_s)[0,1])
            if p_rand <= 0.25: tier = 'eligible_strict'
            elif p_rand <= 0.40: tier = 'eligible_usable'
            else: tier = 'predicted_random_sensitive'
            scored.append({'task':task,'state_id':sid,'ws':ws,'we':we,'phase':phase,
                'p_random_sensitive':round(p_rand,4),'tier':tier,
                'clean_open':clean_open,'source':'s20k' if 's20k' in d else 'legacy'})

tier_counts = Counter(c['tier'] for c in scored)
print('Scored: %d candidates' % len(scored))
print('Tiers: %s' % dict(tier_counts))

# Save scored table
with open(TABLES + '/s20l_v2_randhead_scored_fresh_candidates.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','ws','we','phase','p_random_sensitive','tier','clean_open','source'])
    w.writeheader(); w.writerows(scored)

# ── Select RAND-only queue (VIS queue after audit) ──
# Exclude already-run
existing = set()
for d in ['/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
          '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
          '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613']:
    for f in glob.glob(d + '/summary_*.json'):
        s = json.load(open(f))
        existing.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))

pool = [c for c in scored if (c['task'],c['state_id'],c['ws'],c['we']) not in existing]

preferred = ['bbq_sauce','butter','chocolate_pudding','cream_cheese','salad_dressing','orange_juice','alphabet_soup']
limited = ['ketchup','milk','tomato_sauce']

# Sort by tier > preferred > p_rand ascending
pool.sort(key=lambda c: (
    {'eligible_strict':0,'eligible_usable':1,'predicted_random_sensitive':2}.get(c['tier'],9),
    c['task'] not in preferred,
    c['task'] in limited,
    c['p_random_sensitive']))

selected = []; task_n = Counter(); phase_n = Counter()
tier_targets = {'eligible_strict': 15, 'eligible_usable': 8, 'predicted_random_sensitive': 5}

for c in pool:
    if len(selected) >= 28: break
    if task_n[c['task']] >= 8: continue
    if phase_n[c['phase']] >= 28 * 0.35: continue
    if len([s for s in selected if s['tier']==c['tier']]) >= tier_targets.get(c['tier'],99): continue
    selected.append(c)
    task_n[c['task']] += 1; phase_n[c['phase']] += 1

# RAND-only jobs
rand_jobs = []; jid = 265000
for c in selected:
    cid = '%s_s%s_w%d_%d' % (c['task'],c['state_id'],c['ws'],c['we'])
    jid += 1; rand_jobs.append({'job_id':str(jid),'task':c['task'],'state_id':c['state_id'],
        'window_start':str(c['ws']),'window_end':str(c['we']),'condition':'random_linf',
        'attack_seed':'89','random_control_seed':'89','seed':'0',
        'candidate_id':cid,'tier':'L2_'+c['tier'],'track':'S20Lv2','status':'pending'})

# Split across 3 GPUs
queues = {'gpu10':[],'gpu26':[],'gpu45':[]}
gpus = ['gpu10','gpu26','gpu45']
for i, j in enumerate(rand_jobs):
    queues[gpus[i%3]].append(j)

for gpu, gpu_jobs in queues.items():
    qpath = OUT + '/queues/s20l_v2_rand_%s.csv' % gpu
    with open(qpath,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rand_jobs[0].keys())); w.writeheader(); w.writerows(gpu_jobs)
    print('%s: %d RAND jobs' % (gpu, len(gpu_jobs)))

with open(TABLES + '/s20l_v2_fresh_selection_audit.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','ws','we','phase','p_random_sensitive','tier'])
    w.writeheader()
    for c in selected: w.writerow({k: c[k] for k in ['task','state_id','ws','we','phase','p_random_sensitive','tier']})

print()
print('S20L-v2 RAND-only: %d jobs, %d candidates (seed89)' % (len(rand_jobs), len(selected)))
print('Tasks: %s' % dict(task_n))
print('Phases: %s' % dict(phase_n))
print('Tiers: %s' % dict(Counter(c['tier'] for c in selected)))
print('VIS queue will be built after RAND audit')
print('Output: %s' % OUT)
