#!/usr/bin/env python3
"""S20J: v0.3.1 randhead as conservative non-random-sensitive screener.
Steps 1-3: model comparison, threshold sweep, eligible candidate table, validation queue."""
import csv, json, glob, os, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
OUT = '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613'
os.makedirs(TABLES, exist_ok=True)
os.makedirs(OUT + '/queues', exist_ok=True)

# ── Load RAND master table ──
rand_rows = []
with open(TABLES + '/s20i_rand_label_master.csv') as f:
    rand_rows = list(csv.DictReader(f))

# ── Load universe for features ──
universe = {}
with open('/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv') as f:
    for r in csv.DictReader(f):
        universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

# Load expansion universe
exp_universe = {}
exp_clean_dir = '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612'
for fpath in glob.glob(exp_clean_dir + '/summary_*clean*.json'):
    s = json.load(open(fpath))
    if not s.get('success_done_any', False): continue
    task = s['task']; sid = str(s['state_id'])
    # Need trace for phase detection — use summary-based heuristic
    trace_path = exp_clean_dir + '/trace_%s_s%s_w0_10_s20d_clean_seed0_job*.csv' % (task, sid)
    trace_files = sorted(glob.glob(trace_path))
    if not trace_files: continue
    with open(trace_files[0]) as tf:
        rows = list(csv.DictReader(tf))
    # Simple phase detection
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
    lift = None
    for i in range(fc or 0, len(rows)):
        if g(rows[i], 'obj_z') - base_obj >= 0.015 or g(rows[i], 'eef_z') - base_eef >= 0.03 if 'base_eef' in dir() else False:
            lift = i; break
    base_eef = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
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
    done_step = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary', '0') or '0') == 1 or int(r.get('success_done', '0') or '0') == 1:
            done_step = i; break
    if done_step is None: done_step = len(rows) - 1

    WINDOW_LEN = 10; STRIDE = 5
    for ws in range(5, len(rows) - WINDOW_LEN, STRIDE):
        we = ws + WINDOW_LEN
        if we > done_step + 5: continue
        wc = (ws + we) / 2.0
        if pp is not None and wc >= pp: phase = 'preplace'
        elif lift is not None and wc >= lift + 5: phase = 'transport'
        elif lift is not None and wc >= lift: phase = 'early_transport'
        elif fc is not None and wc >= fc: phase = 'grasp_transition'
        else: phase = 'approach'
        clean_open = sum(1 for r in rows[ws:we] if g(r, 'decoded_open_bool') == 1)
        qpos_vals = [g(r, 'gripper_qpos_before') for r in rows[ws:we]]
        eef_vals = [g(r, 'eef_z') for r in rows[ws:we]]
        exp_universe[(task, sid, ws, we)] = {
            'task': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
            'phase_id': phase, 'rel_timing': wc / max(done_step, 1),
            'clean_open_count': clean_open, 'clean_open_frac': clean_open / WINDOW_LEN,
            'post_grasp_open_count': sum(1 for r in rows[ws:we] if g(r, 'decoded_open_bool') == 1 and (fc is None or int(r['step']) >= fc)),
            'qpos_mean': np.mean(qpos_vals) if qpos_vals else 0,
            'qpos_slope': np.polyfit(range(len(qpos_vals)), qpos_vals, 1)[0] if len(qpos_vals) >= 2 else 0,
            'eef_disp': max(eef_vals) - min(eef_vals) if len(eef_vals) >= 2 else 0,
            'first_close_step': fc, 'lift_step': lift, 'preplace_step': pp, 'done_step': done_step,
        }

print('Main universe: %d, Expansion: %d' % (len(universe), len(exp_universe)))

# ── Feature builder (same as randhead) ──
def build_features(u_entry, task, sid, ws, we):
    u = u_entry
    fc = float(u.get('first_close_step', -1) or -1)
    lift = float(u.get('lift_step', -1) or -1)
    dl = float(u.get('done_step', 280) or 280)
    wc = (ws + we) / 2.0
    return {
        'task': task, 'phase': u.get('phase_id', '?'),
        'clean_open_count': float(u.get('clean_open_count', 0)),
        'clean_open_frac': float(u.get('clean_open_frac', 0)),
        'post_grasp_open': float(u.get('post_grasp_open_count', 0)),
        'qpos_mean': float(u.get('qpos_mean', 0)),
        'qpos_slope': float(u.get('qpos_slope', 0)),
        'eef_disp': float(u.get('eef_disp', 0)),
        'rel_timing': wc / max(dl, 1), 'wc': wc,
        'fc': fc if fc > 0 else -1, 'lift': lift if lift > 0 else -1,
        'ws_minus_fc': ws - fc if fc > 0 else 50,
        'ws_minus_lift': ws - lift if lift > 0 else 50,
        'distance_to_transition': 0, 'pre_open_streak': 0, 'post_close_streak': 0,
        'transition_overlap': 0, 'close_commitment': 0.5,
    }

# ── Train Clean+Transition LR and AllCleanNoTask LR on FULL data ──
# Build training features
trainable = [r for r in rand_rows if r['target_random_sensitive'] >= '0']
feats_train = []
for r in trainable:
    u = universe.get((r['task'], r['state_id'], int(r['window_start']), int(r['window_end'])), {})
    if not u: continue
    feats_train.append(build_features(u, r['task'], r['state_id'], int(r['window_start']), int(r['window_end'])))

y_train = np.array([int(r['target_random_sensitive']) for r in trainable if universe.get((r['task'], r['state_id'], int(r['window_start']), int(r['window_end'])))])
# Filter out rows without universe match
valid_idx = [i for i, r in enumerate(trainable) if universe.get((r['task'], r['state_id'], int(r['window_start']), int(r['window_end'])))]
feats_train = [build_features(universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))], r['task'], r['state_id'], int(r['window_start']), int(r['window_end'])) for r in [trainable[i] for i in valid_idx]]
y_train = np.array([int(trainable[i]['target_random_sensitive']) for i in valid_idx])

print('Training: %d samples, pos=%d' % (len(y_train), int(sum(y_train))))

def clean_transition_features(f):
    return [f['clean_open_count'], f['clean_open_frac'], f['qpos_mean'], f['eef_disp'],
            f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'],
            f['transition_overlap'], f['close_commitment']]

def allclean_features(f):
    return [f['fc'], f['lift'], f['ws_minus_fc'], f['ws_minus_lift'], f['rel_timing'],
            f['clean_open_count'], f['clean_open_frac'], f['post_grasp_open'],
            f['qpos_mean'], f['qpos_slope'], f['eef_disp'],
            f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'],
            f['transition_overlap'], f['close_commitment']]

# Train Clean+Transition LR
X_ct = np.array([clean_transition_features(f) for f in feats_train])
ss_ct = StandardScaler(); X_ct_s = ss_ct.fit_transform(X_ct)
m_ct = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_ct.fit(X_ct_s, y_train)

# Train AllCleanNoTask LR
X_ac = np.array([allclean_features(f) for f in feats_train])
ss_ac = StandardScaler(); X_ac_s = ss_ac.fit_transform(X_ac)
m_ac = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_ac.fit(X_ac_s, y_train)

# ── Threshold sweep on OOF predictions ──
groups = np.array(['%s_%s' % (r['task'], r['state_id']) for i, r in enumerate([trainable[i] for i in valid_idx])])
gkf = GroupKFold(n_splits=min(5, len(set(groups))))
oof_ct = np.zeros(len(y_train)); oof_ac = np.zeros(len(y_train))

for tr, te in gkf.split(X_ct_s, y_train, groups):
    if len(set(y_train[tr])) < 2: continue
    m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m.fit(X_ct_s[tr], y_train[tr]); oof_ct[te] = m.predict_proba(X_ct_s[te])[:, 1]
    m2 = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m2.fit(X_ac_s[tr], y_train[tr]); oof_ac[te] = m2.predict_proba(X_ac_s[te])[:, 1]

# Threshold sweep
with open(TABLES + '/s20i_v031_randhead_threshold_sweep.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model','threshold','eligible_count','eligible_precision','false_clean_rate','abstain_rate','recall'])
    for model_name, oof in [('Clean+Transition_LR', oof_ct), ('AllCleanNoTask_LR', oof_ac)]:
        for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            eligible = oof <= thresh
            n_el = int(sum(eligible))
            if n_el == 0: continue
            el_prec = 1 - np.mean(y_train[eligible])
            false_cl = np.mean(y_train[eligible])
            abstain = 1 - n_el / len(y_train)
            rec = sum((oof <= thresh) & (y_train == 1)) / max(sum(y_train == 1), 1)
            w.writerow([model_name, thresh, n_el, round(el_prec,3), round(false_cl,3), round(abstain,3), round(rec,3)])
            if model_name == 'Clean+Transition_LR':
                print('  %s thresh=%.2f: eligible=%d prec=%.3f false_cl=%.3f abstain=%.3f' % (model_name, thresh, n_el, el_prec, false_cl, abstain))

# ── Score all candidates ──
all_candidates = []
held_out = {('tomato_sauce', '0', 70, 80), ('ketchup', '0', 150, 160)}

# Existing universe
for (task, sid, ws, we), u in universe.items():
    if (task, sid, ws, we) in held_out: continue
    f = build_features(u, task, sid, ws, we)
    p_ct = float(m_ct.predict_proba(ss_ct.transform([clean_transition_features(f)]))[0, 1])
    p_ac = float(m_ac.predict_proba(ss_ac.transform([allclean_features(f)]))[0, 1])
    all_candidates.append({'task': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
        'phase': u.get('phase_id', '?'), 'p_rand_ct_lr': round(p_ct, 4), 'p_rand_ac_lr': round(p_ac, 4),
        'source': 's20f_universe'})

# Expansion universe
for (task, sid, ws, we), u in exp_universe.items():
    if (task, sid, ws, we) in held_out: continue
    f = build_features(u, task, sid, ws, we)
    p_ct = float(m_ct.predict_proba(ss_ct.transform([clean_transition_features(f)]))[0, 1])
    p_ac = float(m_ac.predict_proba(ss_ac.transform([allclean_features(f)]))[0, 1])
    all_candidates.append({'task': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
        'phase': u.get('phase_id', '?'), 'p_rand_ct_lr': round(p_ct, 4), 'p_rand_ac_lr': round(p_ac, 4),
        'source': 's20i_expansion'})

# Classify tiers
for c in all_candidates:
    ct_low = c['p_rand_ct_lr'] <= 0.25
    ac_low = c['p_rand_ac_lr'] <= 0.35
    ct_med = c['p_rand_ct_lr'] <= 0.40
    ac_ok = c['p_rand_ac_lr'] <= 0.50
    both_high = c['p_rand_ct_lr'] >= 0.50 and c['p_rand_ac_lr'] >= 0.50
    if ct_low and ac_low:
        c['eligible_tier'] = 'eligible_strict'
    elif ct_med and ac_ok:
        c['eligible_tier'] = 'eligible_usable'
    elif (ct_low and not ac_low) or (ac_low and not ct_low):
        c['eligible_tier'] = 'disagreement'
    else:
        c['eligible_tier'] = 'predicted_random_sensitive'

tier_counts = Counter(c['eligible_tier'] for c in all_candidates)
print('\nCandidate tiers: %s' % dict(tier_counts))

# Write eligible table
with open(TABLES + '/s20i_v031_non_random_sensitive_candidates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','window_start','window_end','phase','p_rand_ct_lr','p_rand_ac_lr','eligible_tier','source'])
    w.writeheader(); w.writerows(all_candidates)

# ── Build S20J validation queue ──
# 60% eligible_strict, 25% eligible_usable, 10% disagreement, 5% predicted_random_sensitive
strict = [c for c in all_candidates if c['eligible_tier'] == 'eligible_strict']
usable = [c for c in all_candidates if c['eligible_tier'] == 'eligible_usable']
disagree = [c for c in all_candidates if c['eligible_tier'] == 'disagreement']
rand_sens = [c for c in all_candidates if c['eligible_tier'] == 'predicted_random_sensitive']

np.random.seed(42)
n_strict = min(len(strict), 18)
n_usable = min(len(usable), 8)
n_disagree = min(len(disagree), 3)
n_rand = min(len(rand_sens), 2)

queue_cands = (
    list(np.random.choice(strict, n_strict, replace=False) if strict else []) +
    list(np.random.choice(usable, n_usable, replace=False) if usable else []) +
    list(np.random.choice(disagree, n_disagree, replace=False) if disagree else []) +
    list(np.random.choice(rand_sens, n_rand, replace=False) if rand_sens else [])
)

# Build jobs (RAND only for predicted_random_sensitive; RAND+VIS for others)
queue_jobs = []
jid = 240000
for c in queue_cands:
    cid = '%s_s%s_w%d_%d' % (c['task'], c['state_id'], c['window_start'], c['window_end'])
    tier = c['eligible_tier']
    jid += 1; queue_jobs.append({'job_id': str(jid), 'task': c['task'], 'state_id': c['state_id'], 'window_start': str(c['window_start']), 'window_end': str(c['window_end']), 'condition': 'random_linf', 'attack_seed': '86', 'random_control_seed': '86', 'seed': '0', 'candidate_id': cid, 'tier': 'J_'+tier, 'track': 'S20J_randhead', 'status': 'pending'})
    if tier != 'predicted_random_sensitive':
        jid += 1; queue_jobs.append({'job_id': str(jid), 'task': c['task'], 'state_id': c['state_id'], 'window_start': str(c['window_start']), 'window_end': str(c['window_end']), 'condition': 'vis_pgd', 'attack_seed': '86', 'random_control_seed': '', 'seed': '0', 'candidate_id': cid, 'tier': 'J_'+tier, 'track': 'S20J_randhead', 'status': 'pending'})

# Split across 3 GPUs
queues = {'gpu10': [], 'gpu26': [], 'gpu45': []}
gpus = ['gpu10', 'gpu26', 'gpu45']
pairs = [(queue_jobs[i], queue_jobs[i+1]) for i in range(0, len(queue_jobs) - 1, 2) if queue_jobs[i+1]['condition'] == 'vis_pgd']
# Add remaining solo RAND jobs
for i, (rj, vj) in enumerate(pairs):
    gpu = gpus[i % 3]; queues[gpu].append(rj); queues[gpu].append(vj)

for gpu, jobs in queues.items():
    if not jobs: continue
    qpath = OUT + '/queues/s20j_%s.csv' % gpu
    with open(qpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(jobs)
    print('%s: %d jobs' % (gpu, len(jobs)))

with open(TABLES + '/s20j_v031_randhead_screened_queue.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(queue_jobs[0].keys()))
    w.writeheader(); w.writerows(queue_jobs)

print('Queue: %d jobs, %d candidates' % (len(queue_jobs), len(queue_cands)))
print('Tier mix: strict=%d usable=%d disagree=%d rand_sens=%d' % (n_strict, n_usable, n_disagree, n_rand))
print('Done.')
