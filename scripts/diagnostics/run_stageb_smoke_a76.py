#!/usr/bin/env python3
"""Stage-B Smoke-A: CPU training on 76 paired labels with label pool audit."""
import csv, json, os, glob, sys
import numpy as np
from collections import Counter, defaultdict

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OUT_DIR = '/data/liuyu/outputs/overnight_stageb_labels_20260607'
FEAT_CSV = os.path.join(REPO, 'tables', 'online_features_object100_teacher_windows.csv')

# ── Load paired summaries ────────────────────────────────────────
summaries = []
for f in glob.glob(OUT_DIR + '/summary_*.json'):
    with open(f) as fh: summaries.append(json.load(fh))

# Pair by window coordinates
windows = defaultdict(dict)
for s in summaries:
    key = (s['task_key'], str(s['state_id']), s['window_start'], s['window_end'])
    windows[key][s['condition']] = s

# Build label rows
rows = []
for key, conds in windows.items():
    task, sid, ws, we = key
    if 'vis_pgd' not in conds or 'random_linf' not in conds: continue
    vs = conds['vis_pgd']; rs = conds['random_linf']
    if vs['infra_status'] != 'ok' or rs['infra_status'] != 'ok': continue

    vis_open = vs['decoded_open_count']; vis_streak = vs['decoded_longest_open_streak']
    rand_open = rs['decoded_open_count']; rand_streak = rs['decoded_longest_open_streak']

    cmd_pos = (vis_open >= 6 or vis_streak >= 6) and not (rand_open >= 6 or rand_streak >= 6)
    rand_conf = (rand_open >= 6 or rand_streak >= 6)
    pending_neg = not cmd_pos and not rand_conf

    rows.append({
        'task_key': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
        'vis_open': vis_open, 'vis_streak': vis_streak,
        'rand_open': rand_open, 'rand_streak': rand_streak,
        'qpos_delta': vs.get('qpos_delta', 0), 'arm_l2': vs.get('mean_arm_l2', 0),
        'cmd_susceptible': int(cmd_pos),
        'random_confounded': int(rand_conf),
        'pending_negative_1r': int(pending_neg),
        'has_qpos_response': int(vs.get('qpos_delta', 0) > 0.01),
    })

print(f'Paired valid: {len(rows)}')
print(f'cmd_susceptible: {sum(r["cmd_susceptible"] for r in rows)}')
print(f'random_confounded: {sum(r["random_confounded"] for r in rows)}')
print(f'pending_negative_1r: {sum(r["pending_negative_1r"] for r in rows)}')

# ── Label Pool Audit ─────────────────────────────────────────────
task_counts = Counter(r['task_key'] for r in rows)
stratum_counts = Counter()  # approximate from window positions

lines_audit = []
lines_audit.append('# Stage-B Label Pool Audit')
lines_audit.append('')
lines_audit.append('| Metric | Count |')
lines_audit.append('|---|---|')
lines_audit.append('| Total summaries | %d |' % len(summaries))
lines_audit.append('| Valid paired | %d |' % len(rows))
vis_only = sum(1 for k, c in windows.items() if 'vis_pgd' in c and 'random_linf' not in c)
rand_only = sum(1 for k, c in windows.items() if 'random_linf' in c and 'vis_pgd' not in c)
lines_audit.append('| VIS-only | %d |' % vis_only)
lines_audit.append('| Random-only | %d |' % rand_only)
lines_audit.append('| cmd_susceptible positive | %d |' % sum(r['cmd_susceptible'] for r in rows))
lines_audit.append('| random_confounded | %d |' % sum(r['random_confounded'] for r in rows))
lines_audit.append('| pending_negative_1r | %d |' % sum(r['pending_negative_1r'] for r in rows))
lines_audit.append('| has_qpos_response | %d |' % sum(r['has_qpos_response'] for r in rows))
lines_audit.append('| Infra fails | %d |' % sum(1 for s in summaries if s['infra_status'] != 'ok'))
lines_audit.append('')
lines_audit.append('## Per-Task Distribution')
lines_audit.append('')
lines_audit.append('| Task | Total | cmd_pos | rand_conf | pend_neg |')
lines_audit.append('|---|---|---|---|---|')
for t, c in task_counts.most_common():
    trows = [r for r in rows if r['task_key'] == t]
    lines_audit.append('| %s | %d | %d | %d | %d |' % (t, c,
        sum(r['cmd_susceptible'] for r in trows),
        sum(r['random_confounded'] for r in trows),
        sum(r['pending_negative_1r'] for r in trows)))
lines_audit.append('')

with open(os.path.join(REPO, 'reports', 'OBJECT100_STAGEB_LABEL_POOL_AUDIT.md'), 'w') as f:
    f.write('\n'.join(lines_audit))
with open(os.path.join(REPO, 'tables', 'object100_stageb_label_pool_audit.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(os.path.join(REPO, 'tables', 'object100_stageb_task_distribution.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['task','total','cmd_pos','rand_conf','pend_neg'])
    for t, c in task_counts.most_common():
        trows = [r for r in rows if r['task_key'] == t]
        w.writerow([t, c, sum(r['cmd_susceptible'] for r in trows),
                    sum(r['random_confounded'] for r in trows),
                    sum(r['pending_negative_1r'] for r in trows)])
print('Label pool audit written')

# Use summary-level features: window position, arm_l2, qpos_delta, task one-hot
# These are always available from the paired summaries
all_tasks = sorted(set(r['task_key'] for r in rows))
task_to_idx = {t: i for i, t in enumerate(all_tasks)}

feature_cols = ['window_len', 'window_start'] + ['task_' + t for t in all_tasks]

X_list = []
for r in rows:
    vec = [
        int(r['window_end']) - int(r['window_start']),
        int(r['window_start']),
    ]
    for t in all_tasks:
        vec.append(1.0 if r['task_key'] == t else 0.0)
    X_list.append(vec)

X = np.array(X_list, dtype=np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y_cmd = np.array([r['cmd_susceptible'] for r in rows])
y_phys = np.array([r['has_qpos_response'] for r in rows])
tasks = np.array([r['task_key'] for r in rows])
matched_rows = rows
unique_tasks = sorted(set(tasks))
print(f'X: {X.shape}, features: {len(feature_cols)}, y_cmd pos: {sum(y_cmd)}, y_phys pos: {sum(y_phys)}')

# ── Train models ─────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def compute_metrics(y_true, y_score):
    n = len(y_true); n_pos = int(sum(y_true))
    if n_pos == 0 or n_pos == n: return {'auroc': float('nan'), 'auprc': float('nan'), 'p_at_10': float('nan')}
    order = np.argsort(y_score)[::-1]; yr = y_true[order]
    pos_s = y_score[y_true==1]; neg_s = y_score[y_true==0]
    auc = sum(1 for p in pos_s for n in neg_s if p>n) + 0.5*sum(1 for p in pos_s for n in neg_s if p==n)
    auc /= (len(pos_s)*len(neg_s))
    tp=0; auprc=0.0; pr=0
    for i in range(n):
        if yr[i]==1: tp+=1
        prec=tp/(i+1); rec=tp/n_pos
        auprc+=prec*(rec-pr); pr=rec
    k=min(10,n); p10=sum(yr[:k])/k
    return {'auroc': round(auc,4), 'auprc': round(auprc,4), 'p_at_10': round(p10,4), 'n_pos': n_pos, 'n': n}

all_metrics = []
for label_name, y_vec in [('cmd_susceptible', y_cmd), ('phys_response', y_phys)]:
    if sum(y_vec) < 3: continue
    for holdout in unique_tasks:
        tr = tasks != holdout; te = tasks == holdout
        if sum(y_vec[te]) < 2 or sum(y_vec[tr]) < 2: continue

        for name, clf in [('LR', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)),
                           ('RF', RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1))]:
            clf.fit(X[tr], y_vec[tr])
            sc = clf.predict_proba(X[te])[:,1]
            m = compute_metrics(y_vec[te], sc)
            m.update({'model': name, 'label': label_name, 'holdout_task': holdout})
            all_metrics.append(m)

        # Task-only baseline
        prev = np.mean(y_vec[tr])
        m = compute_metrics(y_vec[te], np.full(sum(te), prev))
        m.update({'model': 'TaskOnly', 'label': label_name, 'holdout_task': holdout})
        all_metrics.append(m)

        # Label shuffle
        y_shuf = y_vec[te].copy(); np.random.shuffle(y_shuf)
        m = compute_metrics(y_shuf, np.random.rand(sum(te)))
        m.update({'model': 'Shuffle', 'label': label_name, 'holdout_task': holdout})
        all_metrics.append(m)

# ── Aggregate ────────────────────────────────────────────────────
model_summary = defaultdict(lambda: defaultdict(list))
for m in all_metrics:
    if not np.isnan(m['auroc']):
        model_summary[m['label']][m['model']].append(m['auroc'])

# ── Report ───────────────────────────────────────────────────────
lines = []
lines.append('# Stage-B Transfer Scorer — Smoke-A (76 paired)')
lines.append('')
lines.append('**EXPLORATORY ONLY — USES 1R PROVISIONAL LABELS — NOT FINAL DETECTOR**')
lines.append('')
lines.append(f'**Paired rows**: {len(rows)}, **Features matched**: {len(matched_rows)}, **Features**: {len(feature_cols)}')
lines.append('')
lines.append('## Label Distribution')
lines.append('')
lines.append('| Label | Count |')
lines.append('|---|---|')
lines.append('| cmd_susceptible | %d |' % sum(r['cmd_susceptible'] for r in rows))
lines.append('| random_confounded | %d |' % sum(r['random_confounded'] for r in rows))
lines.append('| pending_negative_1r | %d |' % sum(r['pending_negative_1r'] for r in rows))
lines.append('')
lines.append('## Leave-Task-Out AUROC (mean ± std)')
lines.append('')
lines.append('| Label | Model | AUROC |')
lines.append('|---|---|---|')
for label in ['cmd_susceptible', 'phys_response']:
    for model in ['LR', 'RF', 'TaskOnly', 'Shuffle']:
        aucs = model_summary.get(label, {}).get(model, [])
        if aucs:
            lines.append('| %s | %s | %.4f ± %.4f |' % (label, model, np.mean(aucs), np.std(aucs)))
lines.append('')

lines.append('## Gate Check')
lines.append('')
best_auc = max(np.mean(aucs) for model_aucs in model_summary.get('cmd_susceptible', {}).values() for aucs in [model_aucs] if model_aucs) if model_summary.get('cmd_susceptible') else 0
lines.append('- Best cmd_susceptible AUROC: %.4f' % best_auc)
lines.append('- %s' % ('SIGNAL DETECTED' if best_auc > 0.6 else 'WEAK/NO SIGNAL — features insufficient or labels noisy'))
lines.append('')

with open(os.path.join(REPO, 'reports', 'OBJECT100_STAGEB_TRANSFER_SCORER_V0_SMOKE_A_76.md'), 'w') as f:
    f.write('\n'.join(lines))
with open(os.path.join(REPO, 'tables', 'object100_stageb_transfer_scorer_v0_smoke_a76_metrics.csv'), 'w', newline='') as f:
    if all_metrics:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader(); w.writerows(all_metrics)

print('\n'.join(lines))
print('Smoke-A done')
