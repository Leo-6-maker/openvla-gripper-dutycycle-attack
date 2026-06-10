#!/usr/bin/env python3
"""S7 Action-hidden readout — merge logit+hidden, full model suite. CPU-only."""
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

# ── 2. Load logit features ──
logit_feat = {}
with open(LOGIT_FEAT) as f:
    for r in csv.DictReader(f):
        key = (r['task'], int(r['state_id']), int(r['ws']), int(r['we']))
        logit_feat[key] = r
print('Logit features: %d windows' % len(logit_feat))

# ── 3. Load hidden features ──
hidden_feat = {}
with open(HIDDEN_FEAT) as f:
    for r in csv.DictReader(f):
        key = (r['task'], int(r['state_id']), int(r['ws']), int(r['we']))
        hidden_feat[key] = r
print('Hidden features: %d windows' % len(hidden_feat))

# ── 4. Match stable pool → merged features ──
rows = []
matched_hidden = 0; matched_logit = 0; missing = 0
for pk, pr in stable.items():
    task = resolve_task(pk)
    if not task: missing += 1; continue
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

    key = (task, sid, ws, we)
    lr = logit_feat.get(key)
    hr = hidden_feat.get(key)
    if lr is None and hr is None: missing += 1; continue
    if hr: matched_hidden += 1
    if lr: matched_logit += 1

    rows.append({
        'parent': pk, 'task': task, 'ws': ws, 'we': we,
        'cmd_label': pr['cmd_label'],
        'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
        'risk_rand': float(pr.get('risk_rand', 0)),
        'yield': float(pr.get('yield_cmd', 0)),
        # Proprioceptive (from logit)
        'rg_mean': float(lr['rg_mean']) if lr else 0,
        'rg_std': float(lr['rg_std']) if lr else 0,
        'rg_last': float(lr['rg_last']) if lr else 0,
        'rg_slope': float(lr['rg_slope']) if lr else 0,
        # Action-logit (from logit)
        'open_norm_mean': float(lr['open_norm_mean']) if lr else 0,
        'open_norm_last': float(lr['open_norm_last']) if lr else 0,
        'logit_margin_mean': float(lr['logit_margin_mean']) if lr else 0,
        'logit_margin_last': float(lr['logit_margin_last']) if lr else 0,
        'entropy_mean': float(lr['entropy_mean']) if lr else 0,
        'entropy_last': float(lr['entropy_last']) if lr else 0,
        'top2_margin_mean': float(lr['top2_margin_mean']) if lr else 0,
        'top2_margin_last': float(lr['top2_margin_last']) if lr else 0,
        # Action-hidden (from hidden)
        'h_mean_mean': float(hr['h_mean_mean']) if hr else 0,
        'h_mean_std': float(hr['h_mean_std']) if hr else 0,
        'h_std_mean': float(hr['h_std_mean']) if hr else 0,
        'h_last_mean': float(hr['h_last_mean']) if hr else 0,
        'h_last_std': float(hr['h_last_std']) if hr else 0,
        'h_mean_slope': float(hr['h_mean_slope']) if hr else 0,
    })

n = len(rows)
print('Coverage: logit=%d hidden=%d total_matched=%d missing=%d' % (matched_logit, matched_hidden, n, missing))

if n < 30:
    print('FAIL: coverage %d < 30' % n); exit(1)
if n < 36:
    print('WARN: coverage %d < 36' % n)

# ── 5. Feature safety audit ──
forbidden = ['yield_cmd','pV_cmd','pR_cmd','vis_open','rand_open','qpos_delta','success','failure','win_raw','win_open']
for r in [logit_feat, hidden_feat]:
    for key, row in r.items():
        for fb in forbidden:
            for col in row.keys():
                if fb in col.lower():
                    print('FAIL: forbidden column "%s" in features' % col); exit(1)
print('Feature safety PASS')

# ── 6. Build feature matrices ──
ws_a = np.array([r['ws'] for r in rows]); we_a = np.array([r['we'] for r in rows])
wc_a = (ws_a + we_a) / 2.0; X_timing = np.column_stack([wc_a, wc_a / 300.0])

tasks = sorted(set(r['task'] for r in rows))
task_oh = np.array([[1 if tk == r['task'] else 0 for tk in tasks] for r in rows])
groups = np.array([r['task'] for r in rows])
y_rand = np.array([r['is_rand'] for r in rows])
y_cmd = np.array([r['is_cmd'] for r in rows])

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
X_hidden = np.column_stack([
    [r['h_mean_mean'] for r in rows], [r['h_mean_std'] for r in rows],
    [r['h_std_mean'] for r in rows], [r['h_last_mean'] for r in rows],
    [r['h_last_std'] for r in rows], [r['h_mean_slope'] for r in rows],
])

nsp = min(3, len(set(groups))); gkf = GroupKFold(n_splits=nsp)

configs = {
    'TaskOnly': task_oh,
    'CleanProprio': np.column_stack([X_proprio, X_timing]),
    'ActionLogitOnly': X_logit,
    'ActionHiddenOnly': X_hidden,
    'CleanProprio+Logit': np.column_stack([X_proprio, X_logit, X_timing]),
    'CleanProprio+Hidden': np.column_stack([X_proprio, X_hidden, X_timing]),
    'ActionLogit+Hidden': np.column_stack([X_logit, X_hidden]),
    'CleanProprio+Logit+Hidden': np.column_stack([X_proprio, X_logit, X_hidden, X_timing]),
}

# ── 7. Readout ──
print('\n=== S7 Action-Hidden Readout ===')
print('n=%d  tasks=%d  n_splits=%d' % (n, len(tasks), nsp))
print('%-28s %8s %8s %8s %8s %8s %8s %s' % ('Model','RandAUC','CmdAUC_r','FP_tomato','FN_salad','yield','1-AUC','deg'))
print('-' * 105)

results = []
for name, X_feat in configs.items():
    oof_rand = np.zeros(n); fold_degenerate = False
    for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        if len(set(y_rand[ti])) < 2:
            oof_rand[tei] = y_rand[ti].mean()
            fold_degenerate = True; continue
        m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
        m.fit(Xt, y_rand[ti]); oof_rand[tei] = m.predict_proba(Xe)[:, 1]

    auc_r = roc_auc_score(y_rand, oof_rand) if len(set(y_rand)) > 1 else 0
    auc_c = roc_auc_score(y_cmd, oof_rand) if len(set(y_cmd)) > 1 else 0

    # Shuffle baseline
    y_shuf = y_rand.copy(); np.random.seed(42); np.random.shuffle(y_shuf)
    oof_shuf = np.zeros(n)
    for ti, tei in gkf.split(X_feat, y_shuf, groups=groups):
        if len(set(y_shuf[ti])) < 2: oof_shuf[tei] = y_shuf[ti].mean(); continue
        ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
        m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
        m.fit(Xt, y_shuf[ti]); oof_shuf[tei] = m.predict_proba(Xe)[:, 1]
    auc_shuf = roc_auc_score(y_shuf, oof_shuf) if len(set(y_shuf)) > 1 else 0

    # FP/FN specific parents
    fp_s = fn_s = 0
    for i, r in enumerate(rows):
        if 'k5b_contrast_tomato_far' in r['parent']: fp_s = oof_rand[i]
        if 'k5b_strict_phys_salad' in r['parent']: fn_s = oof_rand[i]

    # CleanRand-like: abstain bottom 50% by rand score, then random rank
    keep = oof_rand <= np.percentile(oof_rand, 50)
    cmd_hit = np.mean([r['is_cmd'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
    rand_hit = np.mean([r['is_rand'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0
    yld = np.mean([r['yield'] for i, r in enumerate(rows) if keep[i]]) if sum(keep) > 0 else 0

    print('%-28s %8.3f %8.3f %8.4f %8.4f %+8.2f %8.3f %s n=%d rand=%.2f cmd=%.2f' % (
        name, auc_r, auc_c, fp_s, fn_s, yld, 1-auc_r,
        'Y' if fold_degenerate else 'N', sum(keep), rand_hit, cmd_hit))

    results.append({
        'model': name, 'RandAUC': round(auc_r,4), 'CmdAUC_on_rand': round(auc_c,4),
        'ShuffleAUC': round(auc_shuf,4), 'FP_tomato': round(fp_s,4), 'FN_salad': round(fn_s,4),
        'yield': round(yld,4), 'cmd_hit': round(cmd_hit,4), 'rand_hit': round(rand_hit,4),
        'n_pass': int(sum(keep)), 'degenerate': 'Y' if fold_degenerate else 'N',
    })

# ── 8. CleanRand + HiddenRank (abstain by rand score, rank by hidden score) ──
print('\n=== Rank Ablation ===')
print('%-28s %8s %8s %8s %8s' % ('Method','cmd_hit','rand_hit','yield','n_pass'))

# CleanRand + RandomRank (baseline)
oof_combined = np.column_stack([X_proprio, X_logit, X_timing])
oof_rand_all = np.zeros(n)
for ti, tei in gkf.split(oof_combined, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(oof_combined[ti]); Xe = ss.transform(oof_combined[tei])
    if len(set(y_rand[ti])) < 2: oof_rand_all[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_rand_all[tei] = m.predict_proba(Xe)[:, 1]

# RandomRank
keep_cr = oof_rand_all <= np.percentile(oof_rand_all, 50)
cmd_cr = np.mean([r['is_cmd'] for i, r in enumerate(rows) if keep_cr[i]]) if sum(keep_cr) > 0 else 0
rand_cr = np.mean([r['is_rand'] for i, r in enumerate(rows) if keep_cr[i]]) if sum(keep_cr) > 0 else 0
yld_cr = np.mean([r['yield'] for i, r in enumerate(rows) if keep_cr[i]]) if sum(keep_cr) > 0 else 0
print('%-28s %8.4f %8.4f %+8.2f %8d' % ('CleanRand+RandomRank', cmd_cr, rand_cr, yld_cr, sum(keep_cr)))

# CleanRand + HiddenRank: same abstain filter, rank by hidden score
oof_hidden_rank = np.zeros(n)
for ti, tei in gkf.split(X_hidden, y_rand, groups=groups):
    ss = StandardScaler(); Xt = ss.fit_transform(X_hidden[ti]); Xe = ss.transform(X_hidden[tei])
    if len(set(y_rand[ti])) < 2: oof_hidden_rank[tei] = y_rand[ti].mean(); continue
    m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
    m.fit(Xt, y_rand[ti]); oof_hidden_rank[tei] = m.predict_proba(Xe)[:, 1]

# HiddenRank within CleanRand-pass
cr_indices = [i for i, k in enumerate(keep_cr) if k]
if len(cr_indices) >= 2:
    hr_scores = oof_hidden_rank[cr_indices]
    hr_rank = np.argsort(hr_scores)  # low score = less rand-like
    top_half = hr_rank[:max(1, len(hr_rank)//2)]
    selected = [cr_indices[i] for i in top_half]
    cmd_hr = np.mean([rows[i]['is_cmd'] for i in selected]) if selected else 0
    rand_hr = np.mean([rows[i]['is_rand'] for i in selected]) if selected else 0
    yld_hr = np.mean([rows[i]['yield'] for i in selected]) if selected else 0
else:
    cmd_hr = rand_hr = yld_hr = 0
print('%-28s %8.4f %8.4f %+8.2f %8d' % ('CleanRand+HiddenRank', cmd_hr, rand_hr, yld_hr, len(selected)))

# Oracle: perfect abstain + perfect ranking
ideal = [i for i, r in enumerate(rows) if r['is_cmd'] == 1 and r['is_rand'] == 0]
cmd_oracle = np.mean([rows[i]['is_cmd'] for i in ideal]) if ideal else 0
rand_oracle = np.mean([rows[i]['is_rand'] for i in ideal]) if ideal else 0
yld_oracle = np.mean([rows[i]['yield'] for i in ideal]) if ideal else 0
print('%-28s %8.4f %8.4f %+8.2f %8d' % ('Oracle', cmd_oracle, rand_oracle, yld_oracle, len(ideal)))

# ── 9. Per-task breakdown ──
print('\n=== Per-Task RandAUC ===')
for task in sorted(set(r['task'] for r in rows)):
    mask = np.array([r['task'] == task for r in rows])
    n_t = sum(mask)
    if n_t < 3: continue
    for name, X_feat in [('CleanProprio', np.column_stack([X_proprio, X_timing])),
                          ('ActionHidden', X_hidden),
                          ('CleanProprio+Hidden', np.column_stack([X_proprio, X_hidden, X_timing]))]:
        oof_t = np.zeros(n)
        for ti, tei in gkf.split(X_feat, y_rand, groups=groups):
            ss = StandardScaler(); Xt = ss.fit_transform(X_feat[ti]); Xe = ss.transform(X_feat[tei])
            if len(set(y_rand[ti])) < 2: oof_t[tei] = y_rand[ti].mean(); continue
            m = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42, C=0.5)
            m.fit(Xt, y_rand[ti]); oof_t[tei] = m.predict_proba(Xe)[:, 1]
        auc_t = roc_auc_score(y_rand[mask], oof_t[mask]) if len(set(y_rand[mask])) > 1 else 0
        fp_t = fn_t = 0
        for i, r in enumerate(rows):
            if r['task'] != task: continue
            if 'k5b_contrast_tomato_far' in r['parent']: fp_t = oof_t[i]
            if 'k5b_strict_phys_salad' in r['parent']: fn_t = oof_t[i]
        print('  %-20s %-22s AUC=%.3f n=%d FP=%.4f FN=%.4f' % (task, name, auc_t, n_t, fp_t, fn_t))

# ── 10. Save ──
out_csv = os.path.join(OUT_DIR, 'action_hidden_readout_results.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','RandAUC','CmdAUC_on_rand','ShuffleAUC','FP_tomato','FN_salad',
                                       'yield','cmd_hit','rand_hit','n_pass','degenerate'])
    w.writeheader()
    for r in results: w.writerow(r)

print('\nSaved: %s' % out_csv)
print('Done.')
