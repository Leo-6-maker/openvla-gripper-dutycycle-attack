#!/usr/bin/env python3
"""Selector v0.2 row-level audit: join quality, per-task distribution,
top-K diversity, fold structure, rand-head vs TaskOnly dominance check.
CPU-only. No training."""

import csv, numpy as np, os, sys, re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import Counter

STABLE = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/combined_stable_pool_k5_k5b.csv'
LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
OUT_CSV = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/selector_v0_2_rows_audit.csv'
OUT_MD = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/reports/STAGEB_RC1A_A7B2F5E_SELECTOR_V0_2_AUDIT.md'

# ---------- Load ----------
labels = {}
with open(LABELS_72) as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r['state_id'], r.get('seed','0'), r['window_start'], r['window_end'])
        labels[key] = r

stable = {}
with open(STABLE) as f:
    for r in csv.DictReader(f):
        stable[r['parent']] = r

KNOWN = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']

def parse(pk):
    task = sid = ws = we = None
    for tk in KNOWN:
        if tk in pk: task = tk; break
    m_s = re.search(r'_s(\d+)', pk)
    if m_s: sid = m_s.group(1)
    m_w = re.search(r'_w(\d+)_(\d+)', pk)
    if m_w: ws = m_w.group(1); we = m_w.group(2)
    return task, sid, ws, we

# ---------- Join ----------
def join_all():
    """Join stable pool to 72-pair. Return (matched_rows, dropped_parents, join_diag)."""
    rows = []
    dropped = []
    join_diag = []  # (parent, cmd_label, task, sid, ws, we, seed, found)
    for pk, pr in stable.items():
        task, sid, ws, we = parse(pk)
        found = None
        if all([task, sid, ws, we]):
            for s in ['0','1','2']:
                if (task, sid, s, ws, we) in labels:
                    found = labels[(task, sid, s, ws, we)]; break

        join_diag.append({
            'parent': pk,
            'cmd_label': pr['cmd_label'],
            'phys_label': pr['phys_label'],
            'task': task or '?',
            'sid': sid or '?',
            'ws': ws or '?',
            'we': we or '?',
            'seed': found.get('seed','?') if found else '?',
            'found': 'yes' if found else 'no',
            'pV_cmd': pr['pV_cmd'],
            'pR_cmd': pr['pR_cmd'],
            'yield_cmd': pr['yield_cmd'],
            'risk_rand': pr['risk_rand'],
        })

        if pr['cmd_label'] == 'unstable_or_unknown':
            dropped.append((pk, 'unstable_or_unknown'))
            continue
        if not found:
            dropped.append((pk, 'no_72pair_match'))
            continue

        def f(field, d=0.0):
            try: return float(found.get(field, d) or d)
            except: return d

        rows.append({
            'parent': pk,
            'task': task, 'state': sid, 'seed': found.get('seed','0'),
            'cmd_label': pr['cmd_label'],
            'phys_label': pr['phys_label'],
            'clean_open_count': f('clean_open_count'),
            'clean_open_frac': f('clean_open_frac'),
            'raw_gripper_mean': f('raw_gripper_mean'),
            'raw_gripper_max': f('raw_gripper_max'),
            'qpos_pre': f('qpos_pre'), 'qpos_mean': f('qpos_mean'),
            'window_start': int(ws), 'window_end': int(we),
            'actual_max_step': int(found.get('actual_max_step', 299) or 299),
            'pV': float(pr['pV_cmd']), 'pR': float(pr['pR_cmd']),
            'yield_cmd': float(pr['yield_cmd']), 'risk': float(pr['risk_rand']),
            'is_rand': 1 if pr['cmd_label'] == 'stable_rand_sensitive' else 0,
            'is_cmd': 1 if pr['cmd_label'] == 'stable_cmd_specific' else 0,
            'is_neg': 1 if pr['cmd_label'] == 'stable_negative' else 0,
            'is_phys': 1 if pr['phys_label'] == 'stable_vis_phys' else 0,
        })
    return rows, dropped, join_diag

rows, dropped, join_diag = join_all()

# ---------- Build feature matrices ----------
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

# ---------- OOF scoring (reproduce v0.2) ----------
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

# ---------- Per-row CSV ----------
os.makedirs(os.path.dirname(OUT_CSV) or '.', exist_ok=True)
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    hdr = ['parent','task','state','seed','cmd_label','phys_label',
           'window_start','window_end','window_center','actual_max_step','rel_timing',
           'clean_open_count','clean_open_frac','raw_gripper_mean','raw_gripper_max',
           'qpos_pre','qpos_mean',
           'pV','pR','yield_cmd','risk',
           'oof_rand','oof_cmd_clean','oof_cmd_task',
           'fold_group']
    w.writerow(hdr)
    for i, r in enumerate(rows):
        w.writerow([
            r['parent'], r['task'], r['state'], r['seed'],
            r['cmd_label'], r['phys_label'],
            r['window_start'], r['window_end'], (r['window_start']+r['window_end'])/2,
            r['actual_max_step'], rel_timing[i],
            r['clean_open_count'], r['clean_open_frac'],
            r['raw_gripper_mean'], r['raw_gripper_max'],
            r['qpos_pre'], r['qpos_mean'],
            r['pV'], r['pR'], r['yield_cmd'], r['risk'],
            round(oof_rand[i],4), round(oof_cmd_clean[i],4), round(oof_cmd_task[i],4),
            groups[i],
        ])

# ---------- Join diagnostic CSV ----------
join_csv = OUT_CSV.replace('selector_v0_2_rows_audit', 'selector_v0_2_join_diag')
with open(join_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['parent','cmd_label','phys_label','task','sid','ws','we','seed','found','pV_cmd','pR_cmd','yield_cmd','risk_rand'])
    for j in join_diag:
        w.writerow([j['parent'],j['cmd_label'],j['phys_label'],j['task'],j['sid'],
                    j['ws'],j['we'],j['seed'],j['found'],j['pV_cmd'],j['pR_cmd'],j['yield_cmd'],j['risk_rand']])

# ---------- Top-K task diversity ----------
rand_abstain = oof_rand <= np.percentile(oof_rand, 50)
n_avail = sum(rand_abstain)
k = min(8, n_avail)

# ranking strategies
def topk(name, mask, scores):
    order = np.argsort(-scores)
    selected = [i for i in order if mask[i]][:k]
    return selected

strategies = {
    'Random': (np.ones(n, dtype=bool), -np.arange(n)[np.random.RandomState(0).permutation(n)]),
    'TaskOnly (no abstain)': (np.ones(n, dtype=bool), oof_cmd_task),
    'CleanCmd (no abstain)': (np.ones(n, dtype=bool), oof_cmd_clean),
    'Abstain(CleanRand)+Random': (rand_abstain, -np.arange(n)[np.random.RandomState(0).permutation(n)]),
    'Abstain(CleanRand)+TaskRank': (rand_abstain, oof_cmd_task),
    'Abstain(CleanRand)+CleanCmd': (rand_abstain, oof_cmd_clean),
}

# ---------- Generate report ----------
report = []
def w(s=''): report.append(s)

w('# Selector v0.2 Row-Level Audit')
w()
w('**Date**: 2026-06-09')
w('**Commit**: a7b2f5e')
w('**Input**: combined_stable_pool_k5_k5b.csv + all_labels_rc1a_14cfabe_72pairs.csv')
w()

w('## 1. Join Summary')
w()
w(f'- Stable pool parents: {len(stable)}')
w(f'- Unstable/unknown (excluded): {sum(1 for d in dropped if d[1]=="unstable_or_unknown")}')
w(f'- No 72-pair match (excluded): {sum(1 for d in dropped if d[1]=="no_72pair_match")}')
w(f'- Joined detector rows: {n}')
w()

# Find which detector rows the original detector v0.1 used (via PARENT_MAP)
detector_v01_parents = {
    'k5_cmd_anchor_milk_s0_w70_80_env0', 'k5_confounded_both_milk_s0_w230_240_env0',
    'k5_rand_command_tomato_sauce_s2_w150_160_env2', 'k5_rand_phys_tomato_sauce_s2_w90_100_env2',
    'k5_hn_surprise_bbq_sauce_s2_w100_110_env2', 'k5_neg_drift_salad_dressing_s2_w120_130_env2',
    'k5_clean_negative_expansion_alphabet_soup_s1_w65_75_env1', 'k5_strict_phys_master_tomato_sauce_s2_w115_125_env2',
    'k5b_contrast_milk_late_milk_s0_w240_250_env0', 'k5b_contrast_milk_early_milk_s0_w75_85_env0',
    'k5b_contrast_milk_mid_milk_s0_w80_90_env0', 'k5b_contrast_tomato_late_tomato_sauce_s2_w155_165_env2',
    'k5b_contrast_tomato_early_tomato_sauce_s2_w95_105_env2', 'k5b_contrast_tomato_far_tomato_sauce_s0_w55_65_env0',
    'k5b_strict_phys_cream_cream_cheese_s2_w50_60_env2', 'k5b_strict_phys_cream2_cream_cheese_s1_w145_155_env1',
    'k5b_strict_phys_tomato_tomato_sauce_s2_w165_175_env2', 'k5b_strict_phys_salad_salad_dressing_s2_w70_80_env2',
    'k5b_rand_alpha_alphabet_soup_s0_w60_70_env0', 'k5b_rand_salad_salad_dressing_s2_w80_90_env2',
    'k5b_rand_salad2_salad_dressing_s1_w50_60_env1', 'k5b_neg_alpha_alphabet_soup_s1_w50_60_env1',
    'k5b_neg_cream_cream_cheese_s0_w85_95_env0', 'k5b_neg_bbq_bbq_sauce_s0_w60_70_env0',
}

# Note: the detector v0.1 script has 24 entries in PARENT_MAP, but server has only 23 unique
# because k5_confounded_both_milk vs k5_confounded_swing_milk naming discrepancy
n_detector_v01 = len(detector_v01_parents)
v01_matched = sum(1 for p in detector_v01_parents if p in stable)
selector_parents = {r['parent'] for r in rows}
only_in_selector = selector_parents - detector_v01_parents
only_in_detector = detector_v01_parents - selector_parents

w('### Detector v0.1 vs Selector v0.2 coverage')
w()
w(f'- Detector v0.1 PARENT_MAP entries: {n_detector_v01}')
w(f'- Detector v0.1 parents found in stable pool: {v01_matched}')
w(f'- Selector v0.2 rows (after unstable + no-72pair filter): {n}')
w(f'- Parents in selector but not detector: {len(only_in_selector)}')
if only_in_selector:
    for p in sorted(only_in_selector):
        w(f'  - {p}')
w(f'- Parents in detector but not selector: {len(only_in_detector)}')
if only_in_detector:
    for p in sorted(only_in_detector):
        w(f'  - {p}  (likely naming mismatch or unstable filter)')
w()

w('## 2. Label Distribution')
w()
w('### CMD labels')
for lbl in ['stable_cmd_specific','stable_rand_sensitive','stable_negative']:
    cnt = sum(1 for r in rows if r['cmd_label'] == lbl)
    w(f'- {lbl}: {cnt}')
w()
w('### PHYS labels')
for lbl in ['stable_vis_phys','stable_rand_phys','stable_no_phys']:
    cnt = sum(1 for r in rows if r['phys_label'] == lbl)
    w(f'- {lbl}: {cnt}')
w()
w('### Per-task')
w()
w('| Task | Total | cmd | rand | neg | phys |')
w('|------|-------|-----|------|-----|------|')
for tk in tasks:
    trows = [r for r in rows if r['task'] == tk]
    w(f'| {tk} | {len(trows)} | {sum(1 for r in trows if r["is_cmd"])} | {sum(1 for r in trows if r["is_rand"])} | {sum(1 for r in trows if r["is_neg"])} | {sum(1 for r in trows if r["is_phys"])} |')
w()

w('## 3. Dropped Parents')
w()
if dropped:
    w('| Parent | Reason |')
    w('|--------|--------|')
    for pk, reason in dropped:
        w(f'| {pk} | {reason} |')
else:
    w('None dropped.')
w()

w('## 4. Top-K Task Diversity (k=8)')
w()
w('| Strategy | rand_hit | cmd_hit | phys_hit | mean_pV | mean_yield | Top tasks |')
w('|----------|----------|---------|----------|---------|------------|-----------|')
for name, (mask, scores) in strategies.items():
    sel = topk(name, mask, scores)
    s = np.array(sel)
    if len(s) == 0:
        w(f'| {name} | - | - | - | - | - | no windows |')
        continue
    rand_hit = sum(y_rand[i] for i in s)/len(s)
    cmd_hit = sum(y_cmd[i] for i in s)/len(s)
    phys_hit = sum(rows[i]['is_phys'] for i in s)/len(s)
    mean_pV = np.mean([rows[i]['pV'] for i in s])
    mean_yield = np.mean([rows[i]['yield_cmd'] for i in s])
    tks = Counter(rows[i]['task'] for i in s)
    tk_str = ' '.join(f'{t[:6]}:{c}' for t,c in tks.most_common(4))
    w(f'| {name} | {rand_hit:.2f} | {cmd_hit:.2f} | {phys_hit:.2f} | {mean_pV:.2f} | {mean_yield:.2f} | {tk_str} |')
w()

w('## 5. Rand-Head Dominance Check')
w()
w('### Is rand head driven by task prior?')
w()

# Check: TaskOnly rand AUROC
task_rand_auroc = roc_auc_score(y_rand, oof_cmd_task) if len(set(y_rand))>1 else 0
clean_rand_auroc = roc_auc_score(y_rand, oof_rand) if len(set(y_rand))>1 else 0
w(f'- TaskOnly rand AUROC: {task_rand_auroc:.3f}')
w(f'- CleanRand OOF AUROC: {clean_rand_auroc:.3f}')
w(f'- Clean > TaskOnly: {clean_rand_auroc > task_rand_auroc + 0.05}')

# Check: per-task rand probability distribution
w()
w('### Rand probability per task')
w()
w('| Task | N | N_rand | Mean oof_rand | Mean oof_cmd_clean | Mean oof_cmd_task |')
w('|------|---|--------|---------------|--------------------|--------------------|')
for tk in tasks:
    mask_tk = np.array([r['task'] == tk for r in rows])
    n_tk = sum(mask_tk)
    n_rand = sum(y_rand[mask_tk])
    if n_tk > 0:
        w(f'| {tk} | {n_tk} | {n_rand} | {np.mean(oof_rand[mask_tk]):.3f} | {np.mean(oof_cmd_clean[mask_tk]):.3f} | {np.mean(oof_cmd_task[mask_tk]):.3f} |')
w()

w('### Top-5 rand scores')
w()
w('| Rank | Parent | Task | cmd_label | oof_rand | oof_cmd_clean | oof_cmd_task |')
w('|------|--------|------|-----------|----------|---------------|--------------|')
rand_order = np.argsort(-oof_rand)
for rank, i in enumerate(rand_order[:5]):
    r = rows[i]
    w(f'| {rank+1} | {r["parent"]} | {r["task"]} | {r["cmd_label"]} | {oof_rand[i]:.4f} | {oof_cmd_clean[i]:.4f} | {oof_cmd_task[i]:.4f} |')
w()

w('## 6. CleanCmd Weakness Diagnosis')
w()
w('### Why is CleanCmd weaker than TaskOnly?')
w()

# Check cmd AUROC
task_cmd_auroc = roc_auc_score(y_cmd, oof_cmd_task) if len(set(y_cmd))>1 else 0
clean_cmd_auroc = roc_auc_score(y_cmd, oof_cmd_clean) if len(set(y_cmd))>1 else 0
w(f'- TaskOnly cmd AUROC: {task_cmd_auroc:.3f}')
w(f'- CleanCmd cmd AUROC: {clean_cmd_auroc:.3f}')
w(f'- TaskOnly > CleanCmd: {task_cmd_auroc > clean_cmd_auroc + 0.02}')

# Check per-task cmd prediction
w()
w('### Per-task cmd separability')
w('| Task | N | N_cmd | N_neg | N_rand | Mean clean_open_frac | Mean raw_gripper |')
w('|------|---|-------|-------|--------|---------------------|------------------|')
for tk in tasks:
    trows = [r for r in rows if r['task'] == tk]
    if not trows: continue
    w(f'| {tk} | {len(trows)} | {sum(1 for r in trows if r["is_cmd"])} | {sum(1 for r in trows if r["is_neg"])} | {sum(1 for r in trows if r["is_rand"])} | {np.mean([r["clean_open_frac"] for r in trows]):.3f} | {np.mean([r["raw_gripper_mean"] for r in trows]):.3f} |')
w()

w('## 7. Fold Structure')
w()
w(f'- n_splits: {n_splits}')
w(f'- n_groups: {len(set(groups))}')
w(f'- Groups: task_state_seed tuples')
# Show fold sizes
for fi, (train_idx, test_idx) in enumerate(gkf.split(X, y_rand, groups=groups)):
    test_groups = set(groups[test_idx])
    train_groups = set(groups[train_idx])
    test_tasks = Counter(groups[i].split('_')[0] for i in test_idx)
    w(f'- Fold {fi}: train={len(train_idx)} test={len(test_idx)} | test groups: {len(test_groups)} | test tasks: {dict(test_tasks)}')
w()

w('## 8. Label-Shuffle Sanity')
w()
np.random.seed(42)
y_rand_shuf = y_rand.copy(); np.random.shuffle(y_rand_shuf)
y_cmd_shuf = y_cmd.copy(); np.random.shuffle(y_cmd_shuf)

for name, y, y_shuf in [('rand/abstain', y_rand, y_rand_shuf), ('cmd_specific', y_cmd, y_cmd_shuf)]:
    oof_shuf = np.zeros(n)
    for train_idx, test_idx in gkf.split(X, y_shuf, groups=groups):
        ss = StandardScaler()
        m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        m.fit(ss.fit_transform(X[train_idx]), y_shuf[train_idx])
        oof_shuf[test_idx] = m.predict_proba(ss.transform(X[test_idx]))[:,1]
    auroc_shuf = roc_auc_score(y_shuf, oof_shuf) if len(set(y_shuf))>1 else 0
    oof_real = oof_rand if name == 'rand/abstain' else oof_cmd_clean
    auroc_real = roc_auc_score(y, oof_real) if len(set(y))>1 else 0
    w(f'- {name}: real AUROC={auroc_real:.3f}, shuffle AUROC={auroc_shuf:.3f}, delta={auroc_real-auroc_shuf:+.3f}')
    w(f'  - Shuffle collapses: {"YES" if auroc_shuf < 0.55 else "NO — possible data leakage"}')
    w()

w('## 9. Gate Checks')
w()
checks = []

# Gate: detector rows explainable
checks.append(('[PASS]' if len(dropped) <= 3 else '[FAIL]', f'Detector rows join explainable: {len(dropped)} dropped ({", ".join(f"{p}({r})" for p,r in dropped)})'))

# Gate: top-K task diversity
best = strategies['Abstain(CleanRand)+TaskRank']
sel = topk('best', best[0], best[1])
sel_tasks = Counter(rows[i]['task'] for i in sel)
max_task_frac = max(sel_tasks.values()) / len(sel) if sel else 1
checks.append(('[PASS]' if max_task_frac <= 0.5 else '[FAIL]', f'Top-K task diversity: max single task = {max_task_frac:.0%} ({dict(sel_tasks)})'))

# Gate: rand head stronger than TaskOnly
checks.append(('[PASS]' if clean_rand_auroc > task_rand_auroc + 0.05 else '[WARN]', f'CleanRand ({clean_rand_auroc:.3f}) > TaskOnly ({task_rand_auroc:.3f}) for rand detection'))

# Gate: CleanCmd weaker than TaskOnly explicitly documented
checks.append(('[PASS]' if task_cmd_auroc > clean_cmd_auroc else '[WARN]', f'CleanCmd ({clean_cmd_auroc:.3f}) < TaskOnly ({task_cmd_auroc:.3f}) confirmed → WIP'))

# Gate: no parent join error
checks.append(('[PASS]' if len(only_in_detector) <= 2 else '[FAIL]', f'Parent join consistency: {len(only_in_detector)} detector-only, {len(only_in_selector)} selector-only'))

for status, msg in checks:
    w(f'{status}: {msg}')

w()
w('## 10. Artifacts')
w()
w(f'- Row audit CSV: `{OUT_CSV}`')
w(f'- Join diagnostic CSV: `{join_csv}`')
w(f'- This report: `{OUT_MD}`')

# Write report
os.makedirs(os.path.dirname(OUT_MD) or '.', exist_ok=True)
with open(OUT_MD, 'w') as f:
    f.write('\n'.join(report))

print('Rows: %d | dropped: %d | fold groups: %d' % (n, len(dropped), len(set(groups))))
print('Rand detection: Clean=%.3f TaskOnly=%.3f' % (clean_rand_auroc, task_rand_auroc))
print('Cmd detection: Clean=%.3f TaskOnly=%.3f' % (clean_cmd_auroc, task_cmd_auroc))
print('\nOutputs:')
print('  CSV:', OUT_CSV)
print('  Report:', OUT_MD)
for status, msg in checks:
    print(' ', status, msg)
