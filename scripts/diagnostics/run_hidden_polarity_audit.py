#!/usr/bin/env python3
"""S7 hidden polarity + fixed-K rank ablation audit. CPU-only."""
import csv, os, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
STABLE = os.path.join(REPO, 'tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv')
LOGIT_FEAT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/action_logit_full_features.csv'
HIDDEN_FEAT = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_hidden_full/action_hidden_full_features_merged.csv'
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_hidden_full'

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
stable = {}
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

# ── 3. Match ──
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
    if lr is None and hr is None: continue
    rows.append({
        'parent': pk, 'task': task, 'ws': ws, 'we': we,
        'cmd_label': pr['cmd_label'],
        'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
        'yield': float(pr.get('yield_cmd', 0)),
        'has_hidden': hr is not None,
        'rg_mean': float(lr['rg_mean']) if lr else 0,
        'rg_std': float(lr['rg_std']) if lr else 0,
        'rg_last': float(lr['rg_last']) if lr else 0,
        'rg_slope': float(lr['rg_slope']) if lr else 0,
        'open_norm_mean': float(lr['open_norm_mean']) if lr else 0,
        'open_norm_last': float(lr['open_norm_last']) if lr else 0,
        'logit_margin_mean': float(lr['logit_margin_mean']) if lr else 0,
        'logit_margin_last': float(lr['logit_margin_last']) if lr else 0,
        'entropy_mean': float(lr['entropy_mean']) if lr else 0,
        'entropy_last': float(lr['entropy_last']) if lr else 0,
        'top2_margin_mean': float(lr['top2_margin_mean']) if lr else 0,
        'top2_margin_last': float(lr['top2_margin_last']) if lr else 0,
        'h_mean_mean': float(hr['h_mean_mean']) if hr else 0,
        'h_mean_std': float(hr['h_mean_std']) if hr else 0,
        'h_std_mean': float(hr['h_std_mean']) if hr else 0,
        'h_last_mean': float(hr['h_last_mean']) if hr else 0,
        'h_last_std': float(hr['h_last_std']) if hr else 0,
        'h_mean_slope': float(hr['h_mean_slope']) if hr else 0,
    })

n = len(rows); n_hidden = sum(1 for r in rows if r['has_hidden'])
print('Total rows: %d, with hidden: %d, missing hidden: %d' % (n, n_hidden, n - n_hidden))

# ── 4. Feature matrices ──
ws_a = np.array([r['ws'] for r in rows]); we_a = np.array([r['we'] for r in rows])
wc_a = (ws_a + we_a) / 2.0; X_timing = np.column_stack([wc_a, wc_a / 300.0])

tasks = sorted(set(r['task'] for r in rows))
task_oh = np.array([[1 if tk == r['task'] else 0 for tk in tasks] for r in rows])
groups = np.array([r['task'] for r in rows])
y_rand = np.array([r['is_rand'] for r in rows])
y_cmd = np.array([r['is_cmd'] for r in rows])

X_proprio = np.column_stack([[r['rg_mean'] for r in rows], [r['rg_std'] for r in rows],
                              [r['rg_last'] for r in rows], [r['rg_slope'] for r in rows]])
X_logit = np.column_stack([[r['open_norm_mean'] for r in rows], [r['open_norm_last'] for r in rows],
                            [r['logit_margin_mean'] for r in rows], [r['logit_margin_last'] for r in rows],
                            [r['entropy_mean'] for r in rows], [r['entropy_last'] for r in rows],
                            [r['top2_margin_mean'] for r in rows], [r['top2_margin_last'] for r in rows]])
X_hidden = np.column_stack([[r['h_mean_mean'] for r in rows], [r['h_mean_std'] for r in rows],
                             [r['h_std_mean'] for r in rows], [r['h_last_mean'] for r in rows],
                             [r['h_last_std'] for r in rows], [r['h_mean_slope'] for r in rows]])

nsp = min(3, len(set(groups))); gkf = GroupKFold(n_splits=nsp)

# ═══════════════════════════════════════════
# AUDIT 2: POLARITY SANITY
# ═══════════════════════════════════════════
print('\n' + '='*60)
print('AUDIT 2: POLARITY SANITY')
print('='*60)

# 2a: Verify labels
fp_idx = [i for i, r in enumerate(rows) if 'k5b_contrast_tomato_far' in r['parent']]
fn_idx = [i for i, r in enumerate(rows) if 'k5b_strict_phys_salad' in r['parent']]
print('\n2a. Label verification:')
for label, indices in [('FP tomato (should be rand=0)', fp_idx), ('FN salad (should be rand=1)', fn_idx)]:
    for i in indices:
        r = rows[i]
        print('  %s: is_rand=%d is_cmd=%d yield=%.2f label=%s parent=%s' % (
            label, r['is_rand'], r['is_cmd'], r['yield'], r['cmd_label'], r['parent'][:60]))

# 2b: Verify predict_proba convention
print('\n2b. predict_proba verification:')
X_ref = np.column_stack([X_proprio, X_logit, X_timing])
oof_ref = np.zeros(n)
for ti, tei in gkf.split(X_ref, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_ref[ti]); Xe = ss.transform(X_ref[tei])
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_ref[tei] = m.predict_proba(Xe)[:, 1]
print('  Classes: %s' % m.classes_)
print('  predict_proba columns: [P(class=0), P(class=1)]')
print('  predict_proba[:,1] = P(rand_sensitive=1)')
print('  y_rand=1 means: rand_sensitive')
print('  y_rand=0 means: not rand_sensitive (cmd_specific, negative, etc.)')
for i in fp_idx:
    print('  FP tomato[%d] score=%.4f (should be LOW if model correct)' % (i, oof_ref[i]))
for i in fn_idx:
    print('  FN salad[%d] score=%.4f (should be HIGH if model correct)' % (i, oof_ref[i]))

# 2c: HiddenRisk vs HiddenSafe
print('\n2c. HiddenRisk (score) vs HiddenSafe (1-score):')
oof_hidden_risk = np.zeros(n)
for ti, tei in gkf.split(X_hidden, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_hidden[ti]); Xe = ss.transform(X_hidden[tei])
    if len(set(y_rand[ti])) < 2: oof_hidden_risk[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_hidden_risk[tei] = m.predict_proba(Xe)[:, 1]

oof_hidden_safe = 1.0 - oof_hidden_risk

for direction, scores in [('HiddenRisk (score)', oof_hidden_risk), ('HiddenSafe (1-score)', oof_hidden_safe)]:
    auc = roc_auc_score(y_rand, scores) if len(set(y_rand)) > 1 else 0
    auc_c = roc_auc_score(y_cmd, scores) if len(set(y_cmd)) > 1 else 0
    fp_s = scores[fp_idx[0]] if fp_idx else 0
    fn_s = scores[fn_idx[0]] if fn_idx else 0
    print('  %-22s AUC=%.3f CmdAUC=%.3f FP_tomato=%.4f FN_salad=%.4f' % (direction, auc, auc_c, fp_s, fn_s))

# ═══════════════════════════════════════════
# AUDIT 3: FIXED-BUDGET RANK ABLATION
# ═══════════════════════════════════════════
print('\n' + '='*60)
print('AUDIT 3: FIXED-BUDGET RANK ABLATION')
print('='*60)

# Build all ranking scores
# CleanRand score (combined proprio + logit)
oof_cleanrand = np.zeros(n)
for ti, tei in gkf.split(X_ref, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_ref[ti]); Xe = ss.transform(X_ref[tei])
    if len(set(y_rand[ti])) < 2: oof_cleanrand[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_cleanrand[tei] = m.predict_proba(Xe)[:, 1]

# Logit score
oof_logit = np.zeros(n)
for ti, tei in gkf.split(X_logit, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_logit[ti]); Xe = ss.transform(X_logit[tei])
    if len(set(y_rand[ti])) < 2: oof_logit[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_logit[tei] = m.predict_proba(Xe)[:, 1]

ranking_methods = {
    'RandomRank': np.random.RandomState(42).rand(n),
    'CleanRandRank': oof_cleanrand,
    'HiddenRiskRank': oof_hidden_risk,
    'HiddenSafeRank': oof_hidden_safe,
    'ActionLogitRank': oof_logit,
}

print('\nSame candidate pool: all %d stable pool parents (CleanRand abstain, then rank)' % n)
print('%-20s' % 'Method', end='')
for K in [7, 10, 15]:
    print('  K=%-2s %6s %6s %7s %5s' % ('K', 'cmd', 'rand', 'yield', 'tasks'), end='')
print()
print('-' * 100)

for method_name, scores in ranking_methods.items():
    print('%-20s' % method_name, end='')
    for K in [7, 10, 15]:
        # CleanRand abstain: keep bottom 50% by CleanRand score
        cr_pass = oof_cleanrand <= np.percentile(oof_cleanrand, 50)
        cr_indices = [i for i, k in enumerate(cr_pass) if k]
        if len(cr_indices) < K:
            print('  K=%-2d (only %d pass)' % (K, len(cr_indices)), end='')
            continue
        # Rank within CleanRand-pass by this method's score (lower = less rand-like = better)
        pool_scores = scores[cr_indices]
        rank_order = np.argsort(pool_scores)  # ascending: low risk first
        selected = [cr_indices[i] for i in rank_order[:K]]
        cmd_h = np.mean([rows[i]['is_cmd'] for i in selected])
        rand_h = np.mean([rows[i]['is_rand'] for i in selected])
        yld = np.mean([rows[i]['yield'] for i in selected])
        n_tasks = len(set(rows[i]['task'] for i in selected))
        print('  K=%-2d %6.4f %6.4f %+7.2f %5d' % (K, cmd_h, rand_h, yld, n_tasks), end='')
    print()

# Oracle
print('%-20s' % 'Oracle', end='')
ideal = sorted([i for i, r in enumerate(rows) if r['is_cmd'] == 1 and r['is_rand'] == 0],
               key=lambda i: rows[i]['yield'], reverse=True)
for K in [7, 10, 15]:
    if len(ideal) < K:
        print('  (only %d ideal)' % len(ideal), end='')
    else:
        sel = ideal[:K]
        print('  K=%-2d %6.4f %6.4f %+7.2f %5d' % (
            K, np.mean([rows[i]['is_cmd'] for i in sel]),
            np.mean([rows[i]['is_rand'] for i in sel]),
            np.mean([rows[i]['yield'] for i in sel]),
            len(set(rows[i]['task'] for i in sel))), end='')
print()

# ── Save ──
out_csv = os.path.join(OUT_DIR, 'action_hidden_polarity_rank_audit.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['method','K','cmd_hit','rand_hit','yield','n_tasks'])
    w.writeheader()
    for method_name, scores in ranking_methods.items():
        cr_pass = oof_cleanrand <= np.percentile(oof_cleanrand, 50)
        cr_indices = [i for i, k in enumerate(cr_pass) if k]
        for K in [7, 10, 15]:
            if len(cr_indices) < K: continue
            pool_scores = scores[cr_indices]
            rank_order = np.argsort(pool_scores)
            selected = [cr_indices[i] for i in rank_order[:K]]
            w.writerow({'method': method_name, 'K': K,
                'cmd_hit': round(np.mean([rows[i]['is_cmd'] for i in selected]), 4),
                'rand_hit': round(np.mean([rows[i]['is_rand'] for i in selected]), 4),
                'yield': round(np.mean([rows[i]['yield'] for i in selected]), 4),
                'n_tasks': len(set(rows[i]['task'] for i in selected))})
    # Oracle
    for K in [7, 10, 15]:
        if len(ideal) < K: continue
        sel = ideal[:K]
        w.writerow({'method': 'Oracle', 'K': K,
            'cmd_hit': round(np.mean([rows[i]['is_cmd'] for i in sel]), 4),
            'rand_hit': round(np.mean([rows[i]['is_rand'] for i in sel]), 4),
            'yield': round(np.mean([rows[i]['yield'] for i in sel]), 4),
            'n_tasks': len(set(rows[i]['task'] for i in sel))})

print('\nSaved: %s' % out_csv)
print('Done.')
