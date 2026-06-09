#!/usr/bin/env python3
"""Action-logit full readout v0 — audit + readout + last-vs-aggregate ablation. CPU-only."""
import csv, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import Counter

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
STABLE = os.path.join(REPO, 'tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv')
FEATURES = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full/action_logit_full_features.csv'
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/action_logit_full'

# ── 1. Load stable pool labels ──
import re
KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']
def parse(pk):
    task=sid=ws=we=None
    for tk in KNOWN:
        if tk in pk: task=tk; break
    m_s=re.search(r'_s(\d+)',pk)
    if m_s: sid=m_s.group(1)
    m_w=re.search(r'_w(\d+)_(\d+)',pk)
    if m_w: ws=m_w.group(1); we=m_w.group(2)
    return task,sid,ws,we

stable = {}
with open(STABLE) as f:
    for r in csv.DictReader(f): stable[r['parent']] = r

# ── 2. Load action-logit features ──
features = {}
with open(FEATURES) as f:
    for r in csv.DictReader(f):
        key = (r['task'], int(r['state_id']), int(r['ws']), int(r['we']))
        features[key] = r

print('Features loaded: %d windows' % len(features))

# ── 3. Coverage audit ──
matched = 0; missing = 0
rows = []
for pk, pr in stable.items():
    # Parse LIBERO task from parent name (NOT from pr['task'] which is category)
    task = None
    for tk in KNOWN:
        if tk in pk: task = tk; break
    if not task: missing += 1; continue
    # Parse ws, we, sid from window field: '{ws}_{we}_env{sid}' or '{ws}_{we}'
    win_str = pr.get('window','')
    parts = win_str.replace('_env','|').split('|')
    ws_we = parts[0].split('_')
    if len(ws_we) >= 2:
        try: ws, we = int(ws_we[0]), int(ws_we[1])
        except: missing += 1; continue
    else: missing += 1; continue
    sid_str = parts[1] if len(parts) > 1 else '0'
    try: sid = int(sid_str)
    except: sid = 0
    if not all([task, isinstance(ws,int), isinstance(we,int)]): missing += 1; continue

    key = (task, sid, ws, we)
    if key in features:
        r = features[key]
        rows.append({
            'parent': pk, 'task': task, 'ws': ws, 'we': we,
            'cmd_label': pr['cmd_label'],
            'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
            'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
            'risk_rand': float(pr.get('risk_rand', 0)),
            'yield': float(pr.get('yield_cmd', 0)),
            'pV': float(pr.get('pV_cmd', 0)), 'pR': float(pr.get('pR_cmd', 0)),
            'open_norm_mean': float(r['open_norm_mean']),
            'open_norm_last': float(r['open_norm_last']),
            'logit_margin_mean': float(r['logit_margin_mean']),
            'logit_margin_last': float(r['logit_margin_last']),
            'entropy_mean': float(r['entropy_mean']),
            'entropy_last': float(r['entropy_last']),
            'top2_margin_mean': float(r['top2_margin_mean']),
            'top2_margin_last': float(r['top2_margin_last']),
            'rg_mean': float(r['rg_mean']),
            'rg_std': float(r['rg_std']),
            'rg_last': float(r['rg_last']),
            'rg_slope': float(r['rg_slope']),
        })
        matched += 1
    else:
        missing += 1

print('Coverage: %d/%d matched, %d missing' % (matched, matched+missing, missing))

if matched < 30:
    print('FAIL: coverage %d < 30 (stable pool label match), stopping' % matched); exit(1)
if matched < 36:
    print('WARN: coverage %d < 36 (some stable pool parents missing features, continuing)' % matched)

n = len(rows)

# ── 4. Feature safety audit (hard-fail) ──
with open(FEATURES) as f:
    feat_rows = list(csv.DictReader(f))
    feat_cols = list(feat_rows[0].keys()) if feat_rows else []
forbidden = ['yield_cmd','pV_cmd','pR_cmd','vis_open','rand_open','qpos_delta','success','failure','win_raw','win_open']
found = [c for c in feat_cols for f in forbidden if f in c.lower()]
if found:
    print('FAIL: forbidden columns in features:', found); exit(1)
for r in feat_rows:
    if r.get('online_safe','') != 'True': print('FAIL: online_safe'); exit(1)
    if r.get('feature_source','') != 'pre_window_only': print('FAIL: feature_source'); exit(1)
    if int(r.get('n_pre',0)) < 1: print('FAIL: n_pre'); exit(1)
print('Feature safety PASS: %d rows, all online_safe=True, pre_window_only' % len(feat_rows))

# ── 5. Readout ──
print('\n=== Action-Logit Readout v0 ===')

# Feature matrices
X_proprio = np.column_stack([
    [r['rg_mean'] for r in rows], [r['rg_std'] for r in rows],
    [r['rg_last'] for r in rows], [r['rg_slope'] for r in rows],
])
X_logit = np.column_stack([
    [r['open_norm_mean'] for r in rows], [r['open_norm_last'] for r in rows],
    [r['logit_margin_mean'] for r in rows], [r['logit_margin_last'] for r in rows],
    [r['entropy_mean'] for r in rows], [r['entropy_last'] for r in rows],
    [r['top2_margin_mean'] for r in rows], [r['top2_margin_last'] for r in rows],
])
X_last_only = np.column_stack([
    [r['open_norm_last'] for r in rows],
    [r['logit_margin_last'] for r in rows],
    [r['entropy_last'] for r in rows],
    [r['top2_margin_last'] for r in rows],
])
X_agg_only = np.column_stack([
    [r['open_norm_mean'] for r in rows],
    [r['logit_margin_mean'] for r in rows],
    [r['entropy_mean'] for r in rows],
    [r['top2_margin_mean'] for r in rows],
])

ws_a = np.array([r['ws'] for r in rows]); we_a = np.array([r['we'] for r in rows])
wc_a = (ws_a + we_a) / 2.0; X_timing = np.column_stack([wc_a, wc_a / 300.0])

tasks = sorted(set(r['task'] for r in rows))
task_oh = np.array([[1 if tk == r['task'] else 0 for tk in tasks] for r in rows])
groups = np.array([r['task'] for r in rows])
y_rand = np.array([r['is_rand'] for r in rows])
y_cmd = np.array([r['is_cmd'] for r in rows])

nsp = min(3, len(set(groups))); gkf = GroupKFold(n_splits=nsp)

configs = {
    'TaskOnly': task_oh,
    'CleanProprio': np.column_stack([X_proprio, X_timing]),
    'ActionLogitOnly': X_logit,
    'CleanProprio+Logit': np.column_stack([X_proprio, X_logit, X_timing]),
    'LogitLastOnly': np.column_stack([X_last_only, X_timing]),
    'LogitAggOnly': np.column_stack([X_agg_only, X_timing]),
    'CleanProprio+Last': np.column_stack([X_proprio, X_last_only, X_timing]),
}

print('%-25s %8s %8s %8s %8s %8s %8s %8s' % ('Model','RandAUC','CmdAUC_r','ShufAUC','FP_score','FN_score','yield','1-AUC'))
print('-' * 95)

for name, X_feat in configs.items():
    oof_rand = np.zeros(n); fold_degenerate = False
    for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        # P1-1: fold single-class fallback
        if len(set(y_rand[ti])) < 2:
            oof_rand[tei] = y_rand[ti].mean()
            fold_degenerate = True; continue
        m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
        m.fit(Xt, y_rand[ti]); oof_rand[tei] = m.predict_proba(Xe)[:, 1]

    auc_r = roc_auc_score(y_rand, oof_rand) if len(set(y_rand)) > 1 else 0
    auc_c_on_rand = roc_auc_score(y_cmd, oof_rand) if len(set(y_cmd)) > 1 else 0  # P1-2: renamed

    # P1-3: label shuffle
    y_rand_shuf = y_rand.copy(); np.random.seed(42); np.random.shuffle(y_rand_shuf)
    oof_shuf = np.zeros(n)
    for ti, tei in gkf.split(X_feat, y_rand_shuf, groups=groups):
        if len(set(y_rand_shuf[ti])) < 2: oof_shuf[tei] = y_rand_shuf[ti].mean(); continue
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
        m.fit(Xt, y_rand_shuf[ti]); oof_shuf[tei] = m.predict_proba(Xe)[:, 1]
    auc_shuf = roc_auc_score(y_rand_shuf, oof_shuf) if len(set(y_rand_shuf)) > 1 else 0

    # FP/FN
    fp_s = fn_s = 0
    for i, r in enumerate(rows):
        if 'k5b_contrast_tomato_far' in r['parent']: fp_s = oof_rand[i]
        if 'k5b_strict_phys_salad' in r['parent']: fn_s = oof_rand[i]

    # Selector-level: CleanRand abstain + random rank
    keep = oof_rand <= np.percentile(oof_rand, 50)
    cmd_hit = np.mean([r['is_cmd'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
    rand_hit = np.mean([r['is_rand'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
    yld = np.mean([r['yield'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0

    print('%-25s %8.3f %8.3f %8.3f %8.4f %8.4f %+8.2f %8.3f n=%d rand=%.2f cmd=%.2f deg=%s' % (
        name, auc_r, auc_c_on_rand, auc_shuf, fp_s, fn_s, yld, 1-auc_r, sum(keep), rand_hit, cmd_hit,
        'Y' if fold_degenerate else 'N'))

# ── 6. Last vs aggregate ablation ──
print('\n=== Ablation: Last-only vs Aggregate-only ===')
print('%-25s %8s %8s %8s %8s' % ('Model','FP_score','FN_score','FP-FN_gap','notes'))
for name in ['LogitLastOnly', 'LogitAggOnly', 'CleanProprio+Last']:
    X_feat = configs[name]
    oof_r = np.zeros(n)
    for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
        m.fit(Xt, y_rand[ti]); oof_r[tei] = m.predict_proba(Xe)[:, 1]
    fp_s = fn_s = 0
    for i, r in enumerate(rows):
        if 'k5b_contrast_tomato_far' in r['parent']: fp_s = oof_r[i]
        if 'k5b_strict_phys_salad' in r['parent']: fn_s = oof_r[i]
    gap = fp_s - fn_s
    note = 'last only best for FP/FN' if abs(gap) > 0.3 else 'weak'
    print('%-25s %8.4f %8.4f %+8.4f %s' % (name, fp_s, fn_s, gap, note))

# ── 7. Save readout results ──
with open(os.path.join(OUT_DIR, 'action_logit_readout_results.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model','Rand_AUC','Cmd_AUC','FP_score','FN_score','cmd_hit','rand_hit','yield','n_pass'])
    for name, X_feat in configs.items():
        oof_r = np.zeros(n)
        for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
            ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
            m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
            m.fit(Xt, y_rand[ti]); oof_r[tei] = m.predict_proba(Xe)[:, 1]
        auc_r = roc_auc_score(y_rand, oof_r) if len(set(y_rand)) > 1 else 0
        fp_s = fn_s = 0
        for i, r in enumerate(rows):
            if 'k5b_contrast_tomato_far' in r['parent']: fp_s = oof_r[i]
            if 'k5b_strict_phys_salad' in r['parent']: fn_s = oof_r[i]
        keep = oof_r <= np.percentile(oof_r, 50)
        cmd_h = np.mean([r['is_cmd'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
        rand_h = np.mean([r['is_rand'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
        yl = np.mean([r['yield'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
        w.writerow([name, round(auc_r,4), round(roc_auc_score(y_cmd, oof_r) if len(set(y_cmd))>1 else 0,4),
                    round(fp_s,4), round(fn_s,4), round(cmd_h,4), round(rand_h,4), round(yl,4), sum(keep)])

print('\nSaved: %s' % os.path.join(OUT_DIR, 'action_logit_readout_results.csv'))
print('Done.')
