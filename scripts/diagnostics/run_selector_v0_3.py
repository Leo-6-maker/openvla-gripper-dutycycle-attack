#!/usr/bin/env python3
"""Selector v0.3: leakage-free on K5+K5b+K5c stable pool v2."""
import csv, numpy as np, re, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from collections import Counter

STABLE = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv'
LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/selector_v0_3_leakage_free.csv'

KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']

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
    task = sid = ws = we = None
    for tk in KNOWN:
        if tk in pk: task = tk; break
    m_s = re.search(r'_s(\d+)', pk)
    if m_s: sid = m_s.group(1)
    m_w = re.search(r'_w(\d+)_(\d+)', pk)
    if m_w: ws = m_w.group(1); we = m_w.group(2)
    return task, sid, ws, we

def get_window_info(pk, pr):
    task, sid, ws, we = parse(pk)
    if not task:
        task = pr.get('task', '?')
    if not ws:
        win = pr.get('window', '')
        parts = win.replace('_env0','').replace('_env1','').replace('_env2','').split('_')
        if len(parts) >= 2:
            ws, we = parts[0], parts[1]
    if not sid:
        win = pr.get('window', '')
        if '_env' in win:
            sid = win.split('_env')[1]
        else:
            sid = '0'
    return task, sid, ws, we

rows = []
skipped = 0
for pk, pr in stable.items():
    if pr['cmd_label'] == 'unstable_or_unknown':
        continue
    task, sid, ws, we = get_window_info(pk, pr)
    if not all([task, ws, we]):
        skipped += 1
        continue

    found = None
    for s in ['0','1','2']:
        if (task, str(sid), s, ws, we) in labels:
            found = labels[(task, str(sid), s, ws, we)]; break
    if not found:
        skipped += 1
        continue

    def f(field, d=0.0):
        try: return float(found.get(field, d) or d)
        except: return d

    rows.append({
        'parent': pk, 'task': task, 'state': str(sid), 'seed': found.get('seed','0'),
        'clean_open_count': f('clean_open_count'), 'clean_open_frac': f('clean_open_frac'),
        'raw_gripper_mean': f('raw_gripper_mean'), 'raw_gripper_max': f('raw_gripper_max'),
        'qpos_pre': f('qpos_pre'), 'qpos_mean': f('qpos_mean'),
        'window_start': int(ws), 'window_end': int(we),
        'actual_max_step': int(found.get('actual_max_step', 299) or 299),
        'pV': float(pr['pV_cmd']), 'pR': float(pr['pR_cmd']),
        'yield_cmd': float(pr['yield_cmd']), 'risk': float(pr['risk_rand']),
        'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
        'is_neg': 1 if pr['cmd_label'] == 'stable_negative' else 0,
    })

print(f'Selector rows: {len(rows)} (skipped: {skipped})')
print(f'  rand={sum(r["is_rand"] for r in rows)} cmd={sum(r["is_cmd"] for r in rows)} neg={sum(r["is_neg"] for r in rows)}')

n = len(rows)
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
y_cmd = np.array([r['is_cmd'] for r in rows])

# OOF
n_splits = min(3, len(set(groups)))
gkf = GroupKFold(n_splits=n_splits)
oof_rand = np.zeros(n)
oof_cmd_clean = np.zeros(n)
oof_cmd_task = np.zeros(n)

for train_idx, test_idx in gkf.split(X, y_rand, groups=groups):
    ss = StandardScaler()
    X_tr = ss.fit_transform(X[train_idx]); X_te = ss.transform(X[test_idx])

    m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m.fit(X_tr, y_rand[train_idx])
    oof_rand[test_idx] = m.predict_proba(X_te)[:,1]

    m2 = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m2.fit(X_tr, y_cmd[train_idx])
    oof_cmd_clean[test_idx] = m2.predict_proba(X_te)[:,1]

    m3 = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    m3.fit(task_oh[train_idx], y_cmd[train_idx])
    oof_cmd_task[test_idx] = m3.predict_proba(task_oh[test_idx])[:,1]

np.random.seed(0)
is_rand = np.array([r['is_rand'] for r in rows])
is_cmd = np.array([r['is_cmd'] for r in rows])
pV = np.array([r['pV'] for r in rows])
yield_cmd = np.array([r['yield_cmd'] for r in rows])

def evaluate(name, abstain_mask, ranking_scores):
    n_avail = sum(abstain_mask)
    k = min(8, n_avail)
    if k == 0: return None
    order = np.argsort(-ranking_scores)
    selected = [i for i in order if abstain_mask[i]][:k]
    if not selected: return None
    s = np.array(selected)
    return {
        'name': name, 'k': k, 'n_avail': n_avail,
        'rand_hit': sum(is_rand[i] for i in s)/len(s),
        'cmd_hit': sum(is_cmd[i] for i in s)/len(s),
        'mean_pV': np.mean([pV[i] for i in s]),
        'mean_pR': np.mean([rows[i]['pR'] for i in s]),
        'mean_yield': np.mean([yield_cmd[i] for i in s]),
        'tasks': Counter(rows[i]['task'] for i in s),
    }

results = []
rand_abstain = oof_rand <= np.percentile(oof_rand, 50)

r = evaluate('Random', np.ones(n, dtype=bool), -np.arange(n)[np.random.permutation(n)])
if r: results.append(r)
r = evaluate('TaskOnly (no abstain)', np.ones(n, dtype=bool), oof_cmd_task)
if r: results.append(r)
r = evaluate('CleanCmd (no abstain)', np.ones(n, dtype=bool), oof_cmd_clean)
if r: results.append(r)
r = evaluate('Abstain(CleanRand)+Random', rand_abstain, -np.arange(n)[np.random.permutation(n)])
if r: results.append(r)
r = evaluate('Abstain(CleanRand)+TaskRank', rand_abstain, oof_cmd_task)
if r: results.append(r)
r = evaluate('Abstain(CleanRand)+CleanCmd', rand_abstain, oof_cmd_clean)
if r: results.append(r)
r = evaluate('Oracle abstain + yield rank', ~is_rand.astype(bool), yield_cmd)
if r: results.append(r)

print()
print('='*100)
print('SELECTOR v0.3 (K5+K5b+K5c, leakage-free OOF)')
print('='*100)
print('%-40s %8s %8s %8s %8s %8s %s' % ('Strategy','rand_hit','cmd_hit','pV','pR','yield','Top tasks'))
print('-'*100)
for r in results:
    tks = ' '.join('%s:%d' % (t[:4], c) for t,c in r['tasks'].most_common(3))
    print('%-40s %8.2f %8.2f %8.2f %8.2f %8.2f %s' % (
        r['name'], r['rand_hit'], r['cmd_hit'], r['mean_pV'], r['mean_pR'], r['mean_yield'], tks))

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['strategy','k','n_avail','rand_hit','cmd_hit','mean_pV','mean_pR','mean_yield'])
    for r in results:
        w.writerow([r['name'],r['k'],r['n_avail'],r['rand_hit'],r['cmd_hit'],r['mean_pV'],r['mean_pR'],r['mean_yield']])
print('\nOutput: %s' % OUT)
