#!/usr/bin/env python3
"""Selector-level pipeline v0: strategy comparison on stable labels."""
import csv, numpy as np, re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from collections import Counter

STABLE = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/combined_stable_pool_k5_k5b.csv'
LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'

labels = {}
with open(LABELS_72) as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r['state_id'], r.get('seed','0'), r['window_start'], r['window_end'])
        labels[key] = r

stable = {}
with open(STABLE) as f:
    for r in csv.DictReader(f):
        stable[r['parent']] = r

def parse(pk):
    known = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']
    task = sid = ws = we = None
    for tk in known:
        if tk in pk: task = tk; break
    m_s = re.search(r'_s(\d+)', pk)
    if m_s: sid = m_s.group(1)
    m_w = re.search(r'_w(\d+)_(\d+)', pk)
    if m_w: ws = m_w.group(1); we = m_w.group(2)
    return task, sid, ws, we

rows = []
for pk, pr in stable.items():
    if pr['cmd_label'] == 'unstable_or_unknown': continue
    task, sid, ws, we = parse(pk)
    if not all([task, sid, ws, we]): continue
    found = None
    for s in ['0','1','2']:
        if (task, sid, s, ws, we) in labels:
            found = labels[(task, sid, s, ws, we)]; break
    if not found: continue
    def f(field, d=0.0):
        try: return float(found.get(field, d) or d)
        except: return d
    rows.append({
        'parent': pk, 'task': task, 'state': sid, 'seed': found.get('seed','0'),
        'clean_open_count': f('clean_open_count'), 'clean_open_frac': f('clean_open_frac'),
        'raw_gripper_mean': f('raw_gripper_mean'), 'raw_gripper_max': f('raw_gripper_max'),
        'qpos_pre': f('qpos_pre'), 'qpos_mean': f('qpos_mean'),
        'window_start': int(ws), 'window_end': int(we),
        'actual_max_step': int(found.get('actual_max_step', 299) or 299),
        'pV': float(pr['pV_cmd']), 'pR': float(pr['pR_cmd']),
        'pVp': float(pr['pV_phys']), 'pRp': float(pr['pR_phys']),
        'yield_cmd': float(pr['yield_cmd']), 'risk': float(pr['risk_rand']),
        'is_rand': 1 if pr['cmd_label'] == 'stable_rand_sensitive' else 0,
        'is_cmd': 1 if pr['cmd_label'] == 'stable_cmd_specific' else 0,
    })

print('Selector rows: %d' % len(rows))
X_clean = np.column_stack([
    [r['clean_open_count'] for r in rows], [r['clean_open_frac'] for r in rows],
    [r['raw_gripper_mean'] for r in rows], [r['raw_gripper_max'] for r in rows],
    [r['qpos_pre'] for r in rows], [r['qpos_mean'] for r in rows],
])
ws_arr = np.array([r['window_start'] for r in rows])
we_arr = np.array([r['window_end'] for r in rows])
wc_arr = (ws_arr + we_arr) / 2.0
max_arr = np.array([r['actual_max_step'] for r in rows])
rel_timing = wc_arr / np.maximum(max_arr, 1)
X = np.column_stack([X_clean, wc_arr, rel_timing])

tasks = sorted(set(r['task'] for r in rows))
task_oh = np.array([[1 if tk == r['task'] else 0 for tk in tasks] for r in rows])
groups = np.array(['%s_%s_%s' % (r['task'], r['state'], r['seed']) for r in rows])
y_rand = np.array([r['is_rand'] for r in rows])

# Clean rand detector
probas = np.zeros(len(y_rand))
n_splits = min(3, len(set(groups)))
gkf = GroupKFold(n_splits=n_splits)
for train_idx, test_idx in gkf.split(X, y_rand, groups=groups):
    ss = StandardScaler()
    X_tr = ss.fit_transform(X[train_idx])
    X_te = ss.transform(X[test_idx])
    m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m.fit(X_tr, y_rand[train_idx])
    probas[test_idx] = m.predict_proba(X_te)[:, 1]

# TaskOnly rand detector
probas_task = np.zeros(len(y_rand))
for train_idx, test_idx in gkf.split(task_oh, y_rand, groups=groups):
    m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m.fit(task_oh[train_idx], y_rand[train_idx])
    probas_task[test_idx] = m.predict_proba(task_oh[test_idx])[:, 1]

pV = np.array([r['pV'] for r in rows])
pR = np.array([r['pR'] for r in rows])
yield_cmd = np.array([r['yield_cmd'] for r in rows])
risk = np.array([r['risk'] for r in rows])
is_rand = np.array([r['is_rand'] for r in rows])
is_cmd = np.array([r['is_cmd'] for r in rows])
n = len(rows); k = min(8, n)

np.random.seed(0)
rand_order = np.random.permutation(n)

# Strategies
strategies = {
    'Random': ('all', lambda i: True, rand_order),
    'TaskOnly.abstain': ('abstain', lambda i: probas_task[i] <= np.percentile(probas_task, 50), None),
    'Clean.abstain': ('abstain', lambda i: probas[i] <= np.percentile(probas, 50), None),
    'Oracle.abstain': ('abstain', lambda i: not is_rand[i], None),
    'Oracle.abstain+rank': ('abstain_rank', lambda i: not is_rand[i], None),
}

print()
print('=== SELECTOR PIPELINE v0 (K=%d) ===' % k)

for name, (mode, mask_fn, order) in strategies.items():
    mask = np.array([mask_fn(i) for i in range(n)])
    n_avail = sum(mask)

    if mode == 'all':
        valid = [i for i in order if mask[i]][:k]
    elif mode == 'abstain':
        candidates = [(i, yield_cmd[i]) for i in range(n) if mask[i]]
        candidates.sort(key=lambda x: -x[1])
        valid = [i for i, _ in candidates[:k]]
    else:
        candidates = [(i, yield_cmd[i]) for i in range(n) if mask[i]]
        candidates.sort(key=lambda x: -x[1])
        valid = [i for i, _ in candidates[:k]]

    if not valid:
        print('%-22s no windows' % name)
        continue

    s = np.array(valid)
    rand_hit = sum(is_rand[i] for i in s) / len(s)
    cmd_hit = sum(is_cmd[i] for i in s) / len(s)
    mean_pV = np.mean([pV[i] for i in s])
    mean_pR = np.mean([pR[i] for i in s])
    mean_yield = np.mean([yield_cmd[i] for i in s])
    mean_risk = np.mean([risk[i] for i in s])
    tks = Counter(rows[i]['task'] for i in s)

    print('%-22s rand_hit=%.2f cmd_hit=%.2f pV=%.2f pR=%.2f yield=%.2f risk=%.2f | %s' % (
        name, rand_hit, cmd_hit, mean_pV, mean_pR, mean_yield, mean_risk,
        ' '.join('%s:%d' % (t[:4], c) for t, c in tks.most_common(3))))
