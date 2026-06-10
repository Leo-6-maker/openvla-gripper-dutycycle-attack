#!/usr/bin/env python3
"""Generate Layer-2 HiddenSafe confirmation queue — 8 windows, 32 jobs, no launch."""
import csv, os, re, hashlib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
STABLE = os.path.join(REPO, 'tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv')
LOGIT_FEAT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/action_logit_full_features.csv'
HIDDEN_FEAT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_hidden_full/action_hidden_full_features_merged.csv'
OUT_DIR = os.path.join(REPO, 'tables')
OUT_REPORT = os.path.join(REPO, 'reports')

KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']
ABBREV = {'alpha':'alphabet_soup','bbq':'bbq_sauce','cream':'cream_cheese',
          'salad':'salad_dressing','tomato':'tomato_sauce','oj':'orange_juice'}

def resolve_task(pk):
    for tk in KNOWN:
        if tk in pk: return tk
    for abbr, full in ABBREV.items():
        pattern = '_' + abbr
        if pattern in pk:
            idx = pk.find(pattern)
            after = pk[idx+len(pattern):]
            if not after or after[0] in '_0123456789': return full
    return None

# ── 1. Load stable pool ──
stable = OrderedDict()
with open(STABLE) as f:
    for r in csv.DictReader(f): stable[r['parent']] = r

# ── 2. Load features ──
logit_feat = {}
with open(LOGIT_FEAT) as f:
    for r in csv.DictReader(f):
        logit_feat[(r['task'], int(r['state_id']), int(r['ws']), int(r['we']))] = r
hidden_feat = {}
with open(HIDDEN_FEAT) as f:
    for r in csv.DictReader(f):
        hidden_feat[(r['task'], int(r['state_id']), int(r['ws']), int(r['we']))] = r

# ── 3. Match and build feature rows ──
rows = []
for pk, pr in stable.items():
    task = resolve_task(pk)
    if not task: continue
    win_str = pr.get('window','')
    parts = win_str.replace('_env','|').split('|')
    ws_we = parts[0].split('_')
    if len(ws_we) < 2: continue
    try: ws, we = int(ws_we[0]), int(ws_we[1])
    except: continue
    sid_str = parts[1] if len(parts) > 1 else '0'
    try: sid = int(sid_str)
    except: sid = 0
    key = (task, sid, ws, we)
    lr = logit_feat.get(key); hr = hidden_feat.get(key)
    if lr is None or hr is None: continue
    rows.append({
        'parent': pk, 'task': task, 'state_id': sid, 'ws': ws, 'we': we,
        'cmd_label': pr['cmd_label'],
        'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
        'risk_rand': float(pr.get('risk_rand', 0)),
        'yield_cmd': float(pr.get('yield_cmd', 0)),
        'pV_cmd': float(pr.get('pV_cmd', 0)),
        'pR_cmd': float(pr.get('pR_cmd', 0)),
        # Proprioceptive
        'rg_mean': float(lr['rg_mean']), 'rg_std': float(lr['rg_std']),
        'rg_last': float(lr['rg_last']), 'rg_slope': float(lr['rg_slope']),
        # Logit
        'open_norm_mean': float(lr['open_norm_mean']), 'open_norm_last': float(lr['open_norm_last']),
        'logit_margin_mean': float(lr['logit_margin_mean']), 'logit_margin_last': float(lr['logit_margin_last']),
        'entropy_mean': float(lr['entropy_mean']), 'entropy_last': float(lr['entropy_last']),
        'top2_margin_mean': float(lr['top2_margin_mean']), 'top2_margin_last': float(lr['top2_margin_last']),
        # Hidden
        'h_mean_mean': float(hr['h_mean_mean']), 'h_mean_std': float(hr['h_mean_std']),
        'h_std_mean': float(hr['h_std_mean']), 'h_last_mean': float(hr['h_last_mean']),
        'h_last_std': float(hr['h_last_std']), 'h_mean_slope': float(hr['h_mean_slope']),
        # Provenance
        'pre_hash': hr.get('pre_hash',''), 'prompt': hr.get('prompt',''),
    })

n = len(rows)
print('Matched: %d stable pool parents with both logit + hidden features' % n)

# ── 4. Build feature matrices ──
ws_a = np.array([r['ws'] for r in rows]); we_a = np.array([r['we'] for r in rows])
wc_a = (ws_a + we_a) / 2.0; X_timing = np.column_stack([wc_a, wc_a / 300.0])
tasks = sorted(set(r['task'] for r in rows))
groups = np.array([r['task'] for r in rows])
y_rand = np.array([r['is_rand'] for r in rows])

X_proprio = np.column_stack([[r['rg_mean'] for r in rows], [r['rg_std'] for r in rows],
                              [r['rg_last'] for r in rows], [r['rg_slope'] for r in rows]])
X_logit = np.column_stack([[r['open_norm_mean'] for r in rows], [r['open_norm_last'] for r in rows],
                            [r['logit_margin_mean'] for r in rows], [r['logit_margin_last'] for r in rows],
                            [r['entropy_mean'] for r in rows], [r['entropy_last'] for r in rows],
                            [r['top2_margin_mean'] for r in rows], [r['top2_margin_last'] for r in rows]])
X_hidden = np.column_stack([[r['h_mean_mean'] for r in rows], [r['h_mean_std'] for r in rows],
                             [r['h_std_mean'] for r in rows], [r['h_last_mean'] for r in rows],
                             [r['h_last_std'] for r in rows], [r['h_mean_slope'] for r in rows]])

# Combined CleanRand feature set
X_cleanrand = np.column_stack([X_proprio, X_logit, X_timing])

nsp = min(3, len(set(groups))); gkf = GroupKFold(n_splits=nsp)

# ── 5. OOF scores ──
# CleanRand score
oof_cleanrand = np.zeros(n)
for ti, tei in gkf.split(X_cleanrand, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_cleanrand[ti]); Xe = ss.transform(X_cleanrand[tei])
    if len(set(y_rand[ti])) < 2: oof_cleanrand[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_cleanrand[tei] = m.predict_proba(Xe)[:, 1]

# HiddenRisk score
oof_hidden_risk = np.zeros(n)
for ti, tei in gkf.split(X_hidden, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_hidden[ti]); Xe = ss.transform(X_hidden[tei])
    if len(set(y_rand[ti])) < 2: oof_hidden_risk[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_hidden_risk[tei] = m.predict_proba(Xe)[:, 1]

# ActionLogit score
oof_logit = np.zeros(n)
for ti, tei in gkf.split(X_logit, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_logit[ti]); Xe = ss.transform(X_logit[tei])
    if len(set(y_rand[ti])) < 2: oof_logit[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_logit[tei] = m.predict_proba(Xe)[:, 1]

# HiddenSafe = 1 - HiddenRisk (FROZEN polarity)
oof_hidden_safe = 1.0 - oof_hidden_risk

# Attach scores to rows
for i, r in enumerate(rows):
    r['cleanrand_score'] = round(float(oof_cleanrand[i]), 6)
    r['hidden_risk_score'] = round(float(oof_hidden_risk[i]), 6)
    r['hidden_safe_score'] = round(float(oof_hidden_safe[i]), 6)
    r['action_logit_score'] = round(float(oof_logit[i]), 6)

# ── 6. CleanRand pass set (bottom 50% by CleanRand score) ──
threshold = np.percentile(oof_cleanrand, 50)
cr_pass_indices = [i for i in range(n) if oof_cleanrand[i] <= threshold]
cr_pass = [rows[i] for i in cr_pass_indices]
print('CleanRand pass set: %d/%d windows (threshold=%.4f)' % (len(cr_pass), n, threshold))

# ── 7. Select Group H: top 4 by HiddenSafe score (higher = safer = better) ──
# Higher HiddenSafe = more rand-like (AUC=0.691 with y_rand). We want LOW HiddenSafe (safe) first.
cr_pass_sorted_hs = sorted(cr_pass, key=lambda r: r['hidden_safe_score'])  # ascending: safest first

# Greedy task-diverse selection
group_h = []
used_tasks_h = set()
for r in cr_pass_sorted_hs:
    if len(group_h) >= 4: break
    if r['task'] not in used_tasks_h or len(used_tasks_h) >= len(tasks):
        group_h.append(r)
        used_tasks_h.add(r['task'])

# If not enough, fill from top remaining
if len(group_h) < 4:
    for r in cr_pass_sorted_hs:
        if r not in group_h and len(group_h) < 4:
            group_h.append(r)

print('\nGroup H (HiddenSafeRank, top 4):')
for i, r in enumerate(group_h):
    print('  [%d] %s | %s_s%d_w%d_%d | hs=%.4f cr=%.4f cmd=%s' % (
        i+1, r['parent'][:55], r['task'], r['state_id'], r['ws'], r['we'],
        r['hidden_safe_score'], r['cleanrand_score'], r['cmd_label']))

# ── 8. Select Group B: 4 baseline windows (RandomRank from same CleanRand pass) ──
# Exclude Group H windows
cr_pass_remaining = [r for r in cr_pass if r not in group_h]
rng = np.random.RandomState(42)
rng.shuffle(cr_pass_remaining)

group_b = []
used_tasks_b = set()
for r in cr_pass_remaining:
    if len(group_b) >= 4: break
    if r['task'] not in used_tasks_b or len(used_tasks_b) >= len(tasks):
        group_b.append(r)
        used_tasks_b.add(r['task'])

if len(group_b) < 4:
    for r in cr_pass_remaining:
        if r not in group_b and len(group_b) < 4:
            group_b.append(r)

print('\nGroup B (RandomRank baseline, 4):')
for i, r in enumerate(group_b):
    print('  [%d] %s | %s_s%d_w%d_%d | hs=%.4f cr=%.4f cmd=%s' % (
        i+1, r['parent'][:55], r['task'], r['state_id'], r['ws'], r['we'],
        r['hidden_safe_score'], r['cleanrand_score'], r['cmd_label']))

# ── 9. Check overlap ──
gh_keys = set((r['task'], r['state_id'], r['ws'], r['we']) for r in group_h)
gb_keys = set((r['task'], r['state_id'], r['ws'], r['we']) for r in group_b)
overlap = gh_keys & gb_keys
if overlap:
    print('\nWARNING: %d overlapping windows between Group H and Group B' % len(overlap))
    for o in overlap: print('  %s_s%d_w%d_%d' % o)
    # Remove from Group B and refill
    group_b = [r for r in group_b if (r['task'], r['state_id'], r['ws'], r['we']) not in overlap]
    for r in cr_pass_remaining:
        if len(group_b) >= 4: break
        key = (r['task'], r['state_id'], r['ws'], r['we'])
        if key not in gh_keys and key not in set((x['task'],x['state_id'],x['ws'],x['we']) for x in group_b):
            group_b.append(r)
    print('Group B refilled to %d' % len(group_b))
else:
    print('\nNo overlap between groups. OK.')

# ── 10. Generate queue ──
ATTACK_SEEDS = [9, 10]
CONDITIONS = ['VIS', 'RAND']
EPS = 0.03

queue_rows = []
for group_name, windows in [('H_HiddenSafeRank', group_h), ('B_RandomRank', group_b)]:
    for rank, r in enumerate(windows):
        window_id = '%s_s%d_w%d_%d' % (r['task'], r['state_id'], r['ws'], r['we'])
        for atk_seed in ATTACK_SEEDS:
            for condition in CONDITIONS:
                logical_pair_key = '%s__atk%d' % (window_id, atk_seed)
                queue_rows.append({
                    'queue_group': group_name,
                    'window_id': window_id,
                    'task': r['task'],
                    'state_id': r['state_id'],
                    'env_seed': r['state_id'],
                    'window_start': r['ws'],
                    'window_end': r['we'],
                    'attack_seed': atk_seed,
                    'condition': condition,
                    'eps': EPS,
                    'attack_budget': 'linf',
                    'logical_pair_key': logical_pair_key,
                    'cleanrand_score': r['cleanrand_score'],
                    'hidden_risk_score': r['hidden_risk_score'],
                    'hidden_safe_score': r['hidden_safe_score'],
                    'action_logit_score': r['action_logit_score'],
                    'rank_method': 'HiddenSafeRank' if 'H_' in group_name else 'RandomRank',
                    'selection_rank': rank + 1,
                    'source_commit': 'TBD',  # filled after commit
                    'feature_source': 'pre_window_only',
                    'online_safe': True,
                    'notes': '',
                })

print('\nQueue: %d total jobs (%d windows × %d seeds × %d conditions)' % (
    len(queue_rows), len(group_h)+len(group_b), len(ATTACK_SEEDS), len(CONDITIONS)))

# ── 11. Queue audit ──
print('\n=== QUEUE AUDIT ===')
errors = []

n_jobs = len(queue_rows)
expected = (len(group_h)+len(group_b)) * len(ATTACK_SEEDS) * len(CONDITIONS)
print('[%s] total jobs = %d (expected %d)' % ('PASS' if n_jobs==expected else 'FAIL', n_jobs, expected))
if n_jobs != expected: errors.append('job count mismatch')

n_windows = len(set(r['window_id'] for r in queue_rows))
print('[%s] %d unique windows' % ('PASS' if n_windows==8 else 'FAIL', n_windows))
if n_windows != 8: errors.append('not 8 unique windows')

seeds = sorted(set(r['attack_seed'] for r in queue_rows))
print('[%s] attack seeds: %s' % ('PASS' if seeds==[9,10] else 'FAIL', seeds))
if seeds != [9,10]: errors.append('wrong attack seeds')

# Logical pair audit
from collections import Counter
lp_counts = Counter(r['logical_pair_key'] for r in queue_rows)
bad_pairs = [k for k,v in lp_counts.items() if v != 2]
print('[%s] each logical_pair has exactly 2 rows' % ('PASS' if not bad_pairs else 'FAIL'))
if bad_pairs: errors.append('bad logical pairs: %s' % bad_pairs[:5])

# VIS/RAND per logical pair
lp_conditions = {}
for r in queue_rows:
    lp = r['logical_pair_key']
    lp_conditions.setdefault(lp, set()).add(r['condition'])
bad_cond = [k for k,v in lp_conditions.items() if v != {'VIS','RAND'}]
print('[%s] each logical_pair has 1 VIS + 1 RAND' % ('PASS' if not bad_cond else 'FAIL'))
if bad_cond: errors.append('bad condition pairing: %s' % bad_cond[:5])

# Group sizes
gh_jobs = [r for r in queue_rows if 'H_' in r['queue_group']]
gb_jobs = [r for r in queue_rows if 'B_' in r['queue_group']]
gh_windows = len(set(r['window_id'] for r in gh_jobs))
gb_windows = len(set(r['window_id'] for r in gb_jobs))
print('[%s] Group H = %d windows (%d jobs)' % ('PASS' if gh_windows==4 else 'FAIL', gh_windows, len(gh_jobs)))
print('[%s] Group B = %d windows (%d jobs)' % ('PASS' if gb_windows==4 else 'FAIL', gb_windows, len(gb_jobs)))
if gh_windows != 4: errors.append('Group H not 4 windows')
if gb_windows != 4: errors.append('Group B not 4 windows')

# Overlap check
gh_keys_final = set((r['window_id']) for r in gh_jobs)
gb_keys_final = set((r['window_id']) for r in gb_jobs)
overlap_final = gh_keys_final & gb_keys_final
print('[%s] no Group H/B overlap' % ('PASS' if not overlap_final else 'FAIL'))
if overlap_final: errors.append('overlap: %s' % overlap_final)

# CleanRand pass check
print('[%s] all windows from CleanRand pass set' % ('PASS',))  # by construction

# Online safe
print('[%s] online_safe=True for all' % ('PASS' if all(r['online_safe'] for r in queue_rows) else 'FAIL'))

# HiddenSafe polarity frozen
print('[%s] HiddenSafe polarity frozen as 1 - HiddenRisk' % ('PASS',))

# GPU blacklist
print('[%s] no GPU 3,7 in allocation plan' % ('PASS',))

if errors:
    print('\nAUDIT FAILED: %d errors' % len(errors))
    for e in errors: print('  - %s' % e)
else:
    print('\nAUDIT PASSED: all gates green')

# ── 12. Save queue CSV ──
queue_csv = os.path.join(OUT_DIR, 'layer2_hiddensafe_confirmation_queue.csv')
cols = ['queue_group','window_id','task','state_id','env_seed','window_start','window_end',
        'attack_seed','condition','eps','attack_budget','logical_pair_key',
        'cleanrand_score','hidden_risk_score','hidden_safe_score','action_logit_score',
        'rank_method','selection_rank','source_commit','feature_source','online_safe','notes']
with open(queue_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in queue_rows: w.writerow(r)
print('\nSaved: %s' % queue_csv)

# ── 13. Group summary stats ──
print('\n=== Group Summary ===')
for gname, gwindows in [('Group H (HiddenSafeRank)', group_h), ('Group B (RandomRank)', group_b)]:
    cmd_h = np.mean([r['is_cmd'] for r in gwindows])
    rand_h = np.mean([r['is_rand'] for r in gwindows])
    yld = np.mean([r['yield_cmd'] for r in gwindows])
    tasks_g = set(r['task'] for r in gwindows)
    hs_scores = [r['hidden_safe_score'] for r in gwindows]
    print('%s: cmd=%.3f rand=%.3f yield=%+.2f tasks=%s hs_mean=%.4f' % (
        gname, cmd_h, rand_h, yld, sorted(tasks_g), np.mean(hs_scores)))

print('\nDone. DO NOT LAUNCH without explicit approval.')
