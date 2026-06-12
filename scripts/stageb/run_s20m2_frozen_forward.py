#!/usr/bin/env python3
"""S20M2: Frozen v0.3.2 forward validation. Train Task+Phase+RF (best v0.3.2 model),
save model, score fresh candidates, build RAND-only queue (seed92).
No VIS, no post-hoc tuning."""
import csv, json, glob, os, pickle, numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
C = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/configs'
M = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/models'
O = '/data/liuyu/outputs/stageb_s20m2_frozen_forward_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(C, exist_ok=True)
os.makedirs(M, exist_ok=True); os.makedirs(O+'/queues', exist_ok=True)

WINDOW = 10; STRIDE = 5; THRESHOLD = 0.40
TRAIN_TASKS = {'ketchup', 'milk', 'tomato_sauce'}
VAL_TASKS   = {'cream_cheese', 'salad_dressing'}
TEST_TASKS  = {'alphabet_soup', 'bbq_sauce', 'butter', 'chocolate_pudding', 'orange_juice'}
HELD_OUT    = {('tomato_sauce','0',70,80), ('ketchup','0',150,160)}

tasks_all = sorted(TRAIN_TASKS | VAL_TASKS | TEST_TASKS)
phases_all = ['approach','grasp_transition','early_transport','transport','preplace','place_or_done']

# ═══════════════════════════════════════════════════════════════
# Phase detectors (must match candidate universe generation)
# ═══════════════════════════════════════════════════════════════
def detect_phases(rows):
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d
    is_open = [g(r, 'decoded_open_bool') for r in rows]
    fc = None; stk = 0
    for i, o in enumerate(is_open):
        if o == 0:
            if stk == 0: fc = i
            stk += 1
            if stk >= 3: break
        else: stk = 0; fc = None
    if stk < 3: fc = None
    bo = float(np.median([g(r, 'obj_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    be = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    lift = None
    for i in range(fc or 0, len(rows)):
        if g(rows[i], 'obj_z')-bo >= 0.015 or g(rows[i], 'eef_z')-be >= 0.03:
            lift = i; break
    pp = None; pz = None; ps = None
    if lift is not None:
        for i in range(lift, len(rows)):
            z = g(rows[i], 'eef_z')
            if pz is None or z > pz: pz = z; ps = i
            if ps is not None and i > ps+3 and pz-z >= 0.005:
                pp = i; break
    done = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary','0') or '0') == 1 or int(r.get('success_done','0') or '0') == 1:
            done = i; break
    if done is None: done = len(rows)-1
    return {'fc': fc, 'lift': lift, 'pp': pp, 'done': done, 'ms': len(rows)}

def phase_id(ws, we, ph):
    wc = (ws+we)/2.0
    if wc >= ph['done']-5: return 'place_or_done'
    if ph['pp'] is not None and wc >= ph['pp']: return 'preplace'
    if ph['lift'] is not None and wc >= ph['lift']+5: return 'transport'
    if ph['lift'] is not None and wc >= ph['lift']: return 'early_transport'
    if ph['fc'] is not None and wc >= ph['fc']: return 'grasp_transition'
    return 'approach'

# ═══════════════════════════════════════════════════════════════
# 1. Load training labels
# ═══════════════════════════════════════════════════════════════
all_rand = {}
dirs = [
    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
    '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
    '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
    '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612',
    '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
    '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
    '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613',
    '/data/liuyu/outputs/stageb_s20l_v2_randonly_20260613',
    '/data/liuyu/outputs/stageb_s20m1_randonly_calibration_20260613',
]
for d in dirs:
    for f in glob.glob(d+'/summary_*random_linf*.json'):
        s = json.load(open(f))
        key = (s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed','0')))
        all_rand[key] = s
    for subd in [d+'/output']:
        if not os.path.exists(subd): continue
        for f in glob.glob(subd+'/summary_*random_linf*.json'):
            s = json.load(open(f))
            key = (s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed','0')))
            all_rand[key] = s

print(f'Loaded {len(all_rand)} RAND labels')

rows = []
for key, s in all_rand.items():
    task, sid, ws, we, seed = key
    o = s['decoded_open_count']; st = s['max_open_streak']
    d = s['success_done_any']; to = s.get('timeout', False)
    if to or not d: label = 'RANDOM_SENSITIVE'; y = 1
    elif o <= 3 and st <= 3: label = 'RAND_STRICT'; y = 0
    elif o <= 5 and st <= 5: label = 'RAND_USABLE'; y = 0
    else: label = 'RAND_BORDERLINE'; y = -1
    if y >= 0:
        rows.append({'task': task, 'state_id': sid, 'ws': ws, 'we': we, 'seed': seed,
                     'rand_open': o, 'rand_streak': st, 'rand_label': label, 'target': y})

# ═══════════════════════════════════════════════════════════════
# 2. Get phase from candidate universe for each training row
# ═══════════════════════════════════════════════════════════════
universe = {}
for upath in [T+'/s20i_v031_non_random_sensitive_candidates.csv',
              T+'/s20i_clean_expansion_candidate_universe.csv']:
    if os.path.exists(upath):
        with open(upath) as f:
            for r in csv.DictReader(f):
                universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

def task_phase_features(r):
    """Task one-hot + Phase one-hot — exactly matching v0.3.2 Task+Phase group."""
    key = (r['task'], r['state_id'], int(r['ws']), int(r['we']))
    u = universe.get(key, {})
    phase = u.get('phase', u.get('phase_id', '?'))
    task_feats = [1 if r['task'] == tk else 0 for tk in tasks_all]
    phase_feats = [1 if phase == p else 0 for p in phases_all]
    return np.array(task_feats + phase_feats)

# Filter to rows with universe info
train_rows = [r for r in rows if r['task'] in TRAIN_TASKS and
              (r['task'], r['state_id'], int(r['ws']), int(r['we'])) in universe]
X_train = np.array([task_phase_features(r) for r in train_rows])
y_train = np.array([r['target'] for r in train_rows])

# ═══════════════════════════════════════════════════════════════
# 3. Train & save frozen model (Task+Phase+RF = v0.3.2 best)
# ═══════════════════════════════════════════════════════════════
ss = StandardScaler(); X_train_s = ss.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_s, y_train)
train_acc = model.score(X_train_s, y_train)
print(f'Frozen model trained: {len(y_train)} samples, pos={int(sum(y_train))}, train_acc={train_acc:.3f}')

with open(M+'/v032_task_phase_rf_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model, 'scaler': ss, 'threshold': THRESHOLD,
        'feature_names': [f'task_{t}' for t in tasks_all] + [f'phase_{p}' for p in phases_all],
        'tasks_all': tasks_all, 'phases_all': phases_all,
        'train_tasks': sorted(TRAIN_TASKS),
    }, f)
print(f'Model saved: {M}/v032_task_phase_rf_model.pkl')

# ═══════════════════════════════════════════════════════════════
# 4. Collect all RAND-tested windows (exclude from forward val)
# ═══════════════════════════════════════════════════════════════
existing = set()
all_rand_dirs = [
    '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
    '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
    '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613',
    '/data/liuyu/outputs/stageb_s20l_v2_randonly_20260613',
    '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612',
    '/data/liuyu/outputs/stageb_s20m1_randonly_calibration_20260613',
    '/data/liuyu/outputs/stageb_s20f_known_parent_lift_20260611',
    '/data/liuyu/outputs/stageb_s20f_v031_repair_20260611',
    '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
    '/data/liuyu/outputs/stageb_s20f_queues_20260611',
]
for d in all_rand_dirs:
    if not os.path.exists(d): continue
    for f in glob.glob(d+'/summary_*.json'):
        try:
            s = json.load(open(f))
            existing.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))
        except: pass
    for subd in [d+'/output']:
        if not os.path.exists(subd): continue
        for f in glob.glob(subd+'/summary_*.json'):
            try:
                s = json.load(open(f))
                existing.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))
            except: pass

print(f'Existing RAND-tested windows: {len(existing)}')

# ═══════════════════════════════════════════════════════════════
# 5. Score fresh candidates (val+test tasks, task+phase features)
# ═══════════════════════════════════════════════════════════════
clean_dirs = [
    '/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613',
    '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612',
    '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean',
]

scored = []
for d in clean_dirs:
    for sf in sorted(glob.glob(d+'/summary_*clean*.json')):
        s = json.load(open(sf))
        if not s.get('success_done_any', False): continue
        task = s['task']; sid = str(s['state_id'])
        if task not in (VAL_TASKS | TEST_TASKS): continue
        tp = d+'/trace_%s_s%s_w0_10_s20d_clean_seed0_job*.csv'%(task, sid)
        tr = sorted(glob.glob(tp))
        if not tr: continue
        with open(tr[0]) as f: rows = list(csv.DictReader(f))
        ph = detect_phases(rows)

        for ws in range(5, ph['ms']-WINDOW, STRIDE):
            we = ws+WINDOW
            if we > ph['done']+5: continue
            if (task, sid, ws, we) in HELD_OUT: continue
            if (task, sid, ws, we) in existing: continue

            phase = phase_id(ws, we, ph)
            window_rows = [r for r in rows if ws <= int(r['step']) < we]
            def g(row, key, d=0.0):
                try: return float(row.get(key, d) or d)
                except: return d
            clean_open = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)

            # Task+Phase features (same as training)
            task_feats = [1 if task == tk else 0 for tk in tasks_all]
            phase_feats = [1 if phase == p else 0 for p in phases_all]
            feat = np.array(task_feats + phase_feats)

            feat_s = ss.transform([feat])
            p_rand = float(model.predict_proba(feat_s)[0, 1])

            if p_rand <= 0.25: tier = 'eligible_strict'
            elif p_rand <= 0.40: tier = 'eligible_usable'
            else: tier = 'predicted_random_sensitive'

            scored.append({
                'task': task, 'state_id': sid, 'ws': ws, 'we': we, 'phase': phase,
                'p_random_sensitive': round(p_rand, 4), 'tier': tier,
                'clean_open': clean_open, 'clean_open_frac': round(clean_open/WINDOW, 2),
            })

print(f'Scored candidates (fresh tasks only): {len(scored)}')
tier_counts = Counter(c['tier'] for c in scored)
print(f'Tiers: {dict(tier_counts)}')
task_counts = Counter(c['task'] for c in scored)
print(f'Tasks: {dict(task_counts)}')

# Save full scored table
with open(T+'/s20m2_frozen_scored_candidates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task', 'state_id', 'ws', 'we', 'phase',
                                      'p_random_sensitive', 'tier', 'clean_open', 'clean_open_frac'])
    w.writeheader(); w.writerows(scored)

# ═══════════════════════════════════════════════════════════════
# 6. Select diverse RAND-only queue
# ═══════════════════════════════════════════════════════════════
TARGET = 27
MAX_PHASE_FRAC = 0.35
TIER_TARGETS = {'eligible_strict': 16, 'eligible_usable': 8, 'predicted_random_sensitive': 3}

# Sort: tier priority > low p_rand
scored.sort(key=lambda c: (
    {'eligible_strict': 0, 'eligible_usable': 1, 'predicted_random_sensitive': 2}.get(c['tier'], 9),
    c['p_random_sensitive']))

selected = []; task_n = Counter(); phase_n = Counter(); adj_n = Counter()

# Dynamic max_per_task: no task > 40% of target
max_per_task = max(4, int(TARGET * 0.40))

for c in scored:
    if len(selected) >= TARGET: break
    if task_n[c['task']] >= max_per_task: continue
    if phase_n[c['phase']] >= TARGET * MAX_PHASE_FRAC: continue
    if len([s for s in selected if s['tier'] == c['tier']]) >= TIER_TARGETS.get(c['tier'], 99): continue
    adj_key = (c['task'], c['state_id'])
    if adj_n[adj_key] >= 2: continue
    selected.append(c)
    task_n[c['task']] += 1; phase_n[c['phase']] += 1; adj_n[adj_key] += 1

# If not enough, relax max_per_task and try again with remaining
if len(selected) < 20:
    print(f'\nOnly {len(selected)} selected, relaxing constraints...')
    max_per_task = max(5, int(TARGET * 0.50))
    for c in scored:
        if len(selected) >= TARGET: break
        if any(s['task']==c['task'] and s['state_id']==c['state_id'] and s['ws']==c['ws'] and s['we']==c['we'] for s in selected): continue
        if task_n[c['task']] >= max_per_task: continue
        if len([s for s in selected if s['tier'] == c['tier']]) >= TIER_TARGETS.get(c['tier'], 99) * 1.5: continue
        adj_key = (c['task'], c['state_id'])
        if adj_n[adj_key] >= 3: continue
        selected.append(c)
        task_n[c['task']] += 1; phase_n[c['phase']] += 1; adj_n[adj_key] += 1

print(f'\nForward validation selected: {len(selected)}')
print(f'Tasks: {dict(task_n)}')
print(f'Phases: {dict(phase_n)}')
print(f'Tiers: {dict(Counter(c["tier"] for c in selected))}')

dominant_frac = max(task_n.values()) / len(selected) if selected else 0
print(f'Dominant task fraction: {dominant_frac:.2f} | task_count: {len(task_n)}')
if dominant_frac > 0.40 or len(task_n) < 4:
    print('WARNING: constraints violated — review selection!')

# ═══════════════════════════════════════════════════════════════
# 7. Build queues (seed92, RAND-only)
# ═══════════════════════════════════════════════════════════════
jobs = []; jid = 290000
for c in selected:
    cid = '%s_s%s_w%d_%d' % (c['task'], c['state_id'], c['ws'], c['we'])
    jid += 1
    jobs.append({
        'job_id': str(jid), 'task': c['task'], 'state_id': c['state_id'],
        'window_start': str(c['ws']), 'window_end': str(c['we']),
        'condition': 'random_linf', 'attack_seed': '92', 'random_control_seed': '92',
        'seed': '0', 'candidate_id': cid,
        'tier': 'M2_'+c['tier'], 'track': 'S20M2', 'status': 'pending',
    })

queues = {'gpu10': [], 'gpu26': [], 'gpu45': []}
gpus = ['gpu10', 'gpu26', 'gpu45']
for i, j in enumerate(jobs):
    queues[gpus[i % 3]].append(j)

for gpu, gj in queues.items():
    qp = O+'/queues/s20m2_frozen_%s.csv' % gpu
    with open(qp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(gj)
    print('%s: %d RAND jobs' % (gpu, len(gj)))

# ═══════════════════════════════════════════════════════════════
# 8. Write manifest & config
# ═══════════════════════════════════════════════════════════════
with open(T+'/s20m2_frozen_forward_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task', 'state_id', 'ws', 'we', 'phase',
                                      'p_random_sensitive', 'tier', 'clean_open', 'clean_open_frac'])
    w.writeheader()
    for c in selected:
        w.writerow({k: c[k] for k in ['task', 'state_id', 'ws', 'we', 'phase',
                                        'p_random_sensitive', 'tier', 'clean_open', 'clean_open_frac']})

with open(C+'/stageb_detector_v032_frozen.yaml', 'w') as f:
    json.dump({
        'version': 'v0.3.2-frozen',
        'model': 'Task+Phase+RF',
        'threshold': THRESHOLD,
        'train_tasks': sorted(TRAIN_TASKS),
        'val_tasks': sorted(VAL_TASKS),
        'test_tasks': sorted(TEST_TASKS),
        'n_train': int(len(y_train)),
        'frozen_forward_seed': 92,
        'frozen_forward_n_jobs': len(jobs),
        'model_path': M+'/v032_task_phase_rf_model.pkl',
    }, f, indent=2)

print(f'\nTotal: {len(jobs)} RAND-only jobs (seed92, no VIS)')
print(f'Manifest: {T}/s20m2_frozen_forward_manifest.csv')
print(f'Config: {C}/stageb_detector_v032_frozen.yaml')
print(f'Model: {M}/v032_task_phase_rf_model.pkl')
print(f'Queues: {O}/queues/')
