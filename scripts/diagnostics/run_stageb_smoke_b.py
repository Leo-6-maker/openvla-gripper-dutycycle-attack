#!/usr/bin/env python3
"""Stage-B Smoke-B: richer features + leave-task-out on >=100 paired labels.
EXPLORATORY ONLY — USES 1R PROVISIONAL LABELS — NOT FINAL DETECTOR."""
import csv, json, os, glob, sys
import numpy as np
from collections import Counter, defaultdict

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OUT_DIR = '/data/liuyu/outputs/overnight_stageb_labels_20260607'
FEAT_CSV = os.path.join(REPO, 'tables', 'online_features_object100_teacher_windows.csv')

# ── Load paired labels ───────────────────────────────────────────
summaries = []
for f in glob.glob(OUT_DIR + '/summary_*.json'):
    with open(f) as fh: summaries.append(json.load(fh))

windows = defaultdict(dict)
for s in summaries:
    key = (s['task_key'], str(s['state_id']), s['window_start'], s['window_end'])
    windows[key][s['condition']] = s

paired_rows = []
for key, conds in windows.items():
    if 'vis_pgd' not in conds or 'random_linf' not in conds: continue
    vs = conds['vis_pgd']; rs = conds['random_linf']
    if vs['infra_status'] != 'ok' or rs['infra_status'] != 'ok': continue
    task, sid, ws, we = key
    vis_open = vs['decoded_open_count']; vis_streak = vs['decoded_longest_open_streak']
    rand_open = rs['decoded_open_count']; rand_streak = rs['decoded_longest_open_streak']
    cmd_pos = (vis_open >= 6 or vis_streak >= 6) and not (rand_open >= 6 or rand_streak >= 6)
    rand_conf = (rand_open >= 6 or rand_streak >= 6)
    paired_rows.append({
        'task_key': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
        'vis_open': vis_open, 'vis_streak': vis_streak, 'rand_open': rand_open, 'rand_streak': rand_streak,
        'qpos_delta': vs.get('qpos_delta', 0), 'arm_l2': vs.get('mean_arm_l2', 0),
        'cmd_susceptible': int(cmd_pos), 'random_confounded': int(rand_conf),
    })

print('Paired valid: %d, cmd_pos: %d, rand_conf: %d' % (
    len(paired_rows), sum(r['cmd_susceptible'] for r in paired_rows),
    sum(r['random_confounded'] for r in paired_rows)))

if len(paired_rows) < 100:
    print('NOT ENOUGH paired rows (<100). Exiting.')
    sys.exit(0)

# ── Load online features ─────────────────────────────────────────
# Load all online features from opportunity dataset
opp_feats = defaultdict(dict)
for csv_path in [os.path.join(REPO, 'tables', 'online_features_object100_teacher_windows.csv'),
                  os.path.join(REPO, 'tables', 'online_features_existing31_at_vis_windows.csv')]:
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                k = (r.get('task_key',''), r.get('state_id',''),
                     int(r.get('window_start',0) or 0), int(r.get('window_end',0) or 0))
                opp_feats[k] = r

# Try to match paired rows to online features
FEAT_EXCLUDE = {'row_id','episode_key','task_key','state_id','seed','window_start','window_end',
                'stratum','train_use','exclude_reason','mechanism_type','teacher_window_original',
                'opportunity_label','window_start_frac','window_len_frac','n_window_frames','n_total_steps',
                'feature_source','features_available','window_id','candidate_id',
                'label_status','taxonomy','vis_open_count','label_physical_response','qpos_label','phys_resp',
                'recommended_use','provenance'}

feature_cols = []
X_rows = []
y_cmd = []; matched_rows = []
all_tasks = sorted(set(r['task_key'] for r in paired_rows))

for r in paired_rows:
    key = (r['task_key'], r['state_id'], r['window_start'], r['window_end'])
    feats = opp_feats.get(key)
    if feats is None:
        # Loose match by task+state (use first feature row for that episode)
        for fk, fv in opp_feats.items():
            if fk[0] == r['task_key'] and fk[1] == r['state_id']:
                feats = fv; break
    if feats is None: continue

    if not feature_cols:
        feature_cols = sorted([c for c in feats.keys() if c not in FEAT_EXCLUDE])

    vec = []
    for c in feature_cols:
        try: vec.append(float(feats.get(c, 0)))
        except: vec.append(0.0)
    # Add task one-hot
    for t in all_tasks:
        vec.append(1.0 if r['task_key'] == t else 0.0)

    X_rows.append(vec)
    y_cmd.append(r['cmd_susceptible'])
    matched_rows.append(r)

if not feature_cols:
    print('No online features matched. Falling back to summary-level features.')
    feature_cols = ['window_len', 'window_start', 'vis_open', 'vis_streak', 'rand_open', 'rand_streak', 'qpos_delta', 'arm_l2']
    X_rows = []
    for r in paired_rows:
        vec = [int(r['window_end'])-int(r['window_start']), int(r['window_start']),
               r['vis_open'], r['vis_streak'], r['rand_open'], r['rand_streak'],
               r['qpos_delta'], r['arm_l2']]
        for t in all_tasks: vec.append(1.0 if r['task_key']==t else 0.0)
        X_rows.append(vec)
    matched_rows = paired_rows
    feature_cols = feature_cols + ['task_'+t for t in all_tasks]

X = np.array(X_rows, dtype=np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y_cmd = np.array(y_cmd)
tasks = np.array([r['task_key'] for r in matched_rows])
unique_tasks = sorted(set(tasks))
print('X: %s, features: %d, y pos: %d' % (X.shape, len(feature_cols), sum(y_cmd)))

if sum(y_cmd) < 5:
    print('Too few positives for meaningful model.')
    sys.exit(0)

# ── Train ────────────────────────────────────────────────────────
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
for label_name, y_vec in [('cmd_susceptible', y_cmd)]:
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
        # TaskOnly
        prev = np.mean(y_vec[tr])
        m = compute_metrics(y_vec[te], np.full(sum(te), prev))
        m.update({'model': 'TaskOnly', 'label': label_name, 'holdout_task': holdout})
        all_metrics.append(m)
        # Shuffle
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
lines.append('# Stage-B Transfer Scorer — Smoke-B (%d paired)' % len(paired_rows))
lines.append('')
lines.append('**EXPLORATORY ONLY — USES 1R PROVISIONAL LABELS — NOT FINAL DETECTOR**')
lines.append('')
lines.append('**Paired**: %d, **Features**: %d, **Matched**: %d' % (len(paired_rows), len(feature_cols), len(matched_rows)))
lines.append('**qpos**: FIXED for future runs / BLOCKED for existing 194 traces')
lines.append('**physical_response**: NOT TRAINED (qpos not reliable for existing data)')
lines.append('')
lines.append('## Label Distribution')
lines.append('')
lines.append('| Label | Count |')
lines.append('|---|---|')
lines.append('| cmd_susceptible | %d |' % sum(r['cmd_susceptible'] for r in paired_rows))
lines.append('| random_confounded | %d |' % sum(r['random_confounded'] for r in paired_rows))
lines.append('')
lines.append('## Leave-Task-Out AUROC')
lines.append('')
lines.append('| Label | Model | AUROC (mean±std) |')
lines.append('|---|---|---|')
for label in ['cmd_susceptible']:
    for model in ['LR', 'RF', 'TaskOnly', 'Shuffle']:
        aucs = model_summary.get(label, {}).get(model, [])
        if aucs:
            lines.append('| %s | %s | %.4f ± %.4f |' % (label, model, np.mean(aucs), np.std(aucs)))
lines.append('')

lines.append('## Per-Task AUROC')
lines.append('')
lines.append('| Task | LR | RF | TaskOnly | Shuffle |')
lines.append('|---|---|---|---|---|')
for task in unique_tasks:
    lr = next((m['auroc'] for m in all_metrics if m['model']=='LR' and m['holdout_task']==task and m['label']=='cmd_susceptible'), float('nan'))
    rf = next((m['auroc'] for m in all_metrics if m['model']=='RF' and m['holdout_task']==task and m['label']=='cmd_susceptible'), float('nan'))
    to = next((m['auroc'] for m in all_metrics if m['model']=='TaskOnly' and m['holdout_task']==task and m['label']=='cmd_susceptible'), float('nan'))
    sh = next((m['auroc'] for m in all_metrics if m['model']=='Shuffle' and m['holdout_task']==task and m['label']=='cmd_susceptible'), float('nan'))
    lines.append('| %s | %s | %s | %s | %s |' % (task, lr, rf, to, sh))
lines.append('')

with open(os.path.join(REPO, 'reports', 'OBJECT100_STAGEB_TRANSFER_SCORER_V0_SMOKE_B_FULL.md'), 'w') as f:
    f.write('\n'.join(lines))
with open(os.path.join(REPO, 'tables', 'object100_stageb_transfer_scorer_v0_smoke_b_metrics.csv'), 'w', newline='') as f:
    if all_metrics:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader(); w.writerows(all_metrics)

print('\n'.join(lines))
