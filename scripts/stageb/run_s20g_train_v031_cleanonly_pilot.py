#!/usr/bin/env python3
"""S20G v0.3.1 CLEAN-ONLY detector: no VIS features, GroupKFold by task+state.
Corrective: previous pilot used vis_open/vis_streak/open_delta → post-hoc audit, not deployable."""
import csv, json, os, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'

# Load paired labels + transition audit
paired = []
with open(TABLES + '/s20g_v031_paired_label_table.csv') as f:
    paired = list(csv.DictReader(f))

trans_audit = {}
with open(TABLES + '/s20g_close_transition_audit.csv') as f:
    for r in csv.DictReader(f):
        key = (r['task'], r['state_id'], r['window_start'], r['window_end'], r['seed'])
        trans_audit[key] = r

# Load candidate universe (unseen windows)
universe = []
with open('/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv') as f:
    universe = list(csv.DictReader(f))

held_out_windows = {('tomato_sauce', '0', 70, 80), ('ketchup', '0', 150, 160)}

# ── Build features ──
def build_clean_features(p_list, mode='clean_only'):
    """
    mode='clean_only': no attack features
    mode='post_rand_triage': adds rand_open, rand_streak, rand_done, rand_timeout
    """
    features = []
    for p in p_list:
        key = (p['task'], p['state_id'], p['window_start'], p['window_end'], p['seed'])
        t = trans_audit.get(key, {})

        ws = float(p['window_start']); we = float(p['window_end'])
        wc = (ws + we) / 2.0
        fc = float(p.get('first_close_step', -1) or -1)
        lift = float(p.get('lift_step', -1) or -1)

        feats = [
            # Phase features
            fc if fc > 0 else -1,
            lift if lift > 0 else -1,
            ws - fc if fc > 0 else 50,  # distance from first close
            ws - lift if lift > 0 else 50,  # distance from lift
            wc / max(float(p.get('rand_steps', 280)), 1),  # rel_timing
            # Close-transition features
            float(t.get('distance_to_transition', 0) or 0),
            float(t.get('pre_open_streak', 0) or 0),
            float(t.get('post_close_streak', 0) or 0),
            int(t.get('transition_overlap_center', 0) or 0),
            float(t.get('close_commitment_score', 0.5) or 0.5),
        ]

        if mode == 'post_rand_triage':
            feats += [
                float(p['rand_open']),
                float(p['rand_streak']),
                1.0 if p['rand_timeout'] == 'True' else 0.0,
                float(p['rand_open']) / max(float(p['rand_streak']), 1),
            ]

        features.append(feats)
    return np.array(features)

# ── Targets ──
# From paired set (all RAND-pass windows that got VIS)
y_cmd = np.array([1 if p['classification'] in ('cmd_specific', 'task_effect', 'contact_effect_weak') else 0 for p in paired])
y_task = np.array([1 if p['classification'] in ('task_effect', 'contact_effect_weak') else 0 for p in paired])

# Groups: task + state_id
groups = np.array(['%s_%s' % (p['task'], p['state_id']) for p in paired])

# ── Evaluate: GroupKFold (strict) + StratifiedKFold (optimistic ref) ──
results = []
for mode in ['clean_only', 'post_rand_triage']:
    X = build_clean_features(paired, mode)
    for model_name, ModelCls in [
        ('LR', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('GB', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        if model_name == 'LR':
            X_use = StandardScaler().fit_transform(X)
        else:
            X_use = X

        for target_name, y in [('cmd_specific', y_cmd), ('task_or_contact', y_task)]:
            n_pos = int(sum(y))
            if n_pos < 3:
                results.append({'mode': mode, 'model': model_name, 'target': target_name,
                               'n': len(y), 'pos': n_pos, 'cv': '', 'auroc_gkf': '', 'bal_gkf': '', 'auroc_skf': ''})
                continue

            # GroupKFold
            try:
                n_splits = max(2, min(3, len(set(groups))))
                gkf = GroupKFold(n_splits=n_splits)
                auroc_gkf = []; bal_gkf = []
                for tr, te in gkf.split(X_use, y, groups):
                    if len(set(y[tr])) < 2: continue
                    m = ModelCls
                    if model_name == 'LR':
                        m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
                    elif model_name == 'RF':
                        m = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                    else:
                        m = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    m.fit(X_use[tr], y[tr])
                    yp = m.predict_proba(X_use[te])[:, 1]
                    auroc_gkf.append(roc_auc_score(y[te], yp))
                    bal_gkf.append(balanced_accuracy_score(y[te], (yp >= 0.5).astype(int)))

                gkf_auroc = round(np.mean(auroc_gkf), 3) if auroc_gkf else 0
                gkf_bal = round(np.mean(bal_gkf), 3) if bal_gkf else 0
            except Exception as e:
                gkf_auroc = 'ERR'; gkf_bal = str(e)[:30]

            # StratifiedKFold (optimistic reference)
            try:
                skf = StratifiedKFold(n_splits=min(3, n_pos), shuffle=True, random_state=42)
                auroc_skf = []
                for tr, te in skf.split(X_use, y):
                    if len(set(y[tr])) < 2: continue
                    m = ModelCls
                    if model_name == 'LR':
                        m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
                    elif model_name == 'RF':
                        m = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                    else:
                        m = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    m.fit(X_use[tr], y[tr])
                    yp = m.predict_proba(X_use[te])[:, 1]
                    auroc_skf.append(roc_auc_score(y[te], yp))
                skf_auroc = round(np.mean(auroc_skf), 3) if auroc_skf else 0
            except Exception:
                skf_auroc = 'ERR'

            results.append({
                'mode': mode, 'model': model_name, 'target': target_name,
                'n': len(y), 'pos': n_pos,
                'auroc_gkf': gkf_auroc, 'bal_gkf': gkf_bal,
                'auroc_skf': skf_auroc,
            })

# Write metrics
with open(TABLES + '/s20g_v031_cleanonly_pilot_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['mode','model','target','n','pos','auroc_gkf','bal_gkf','auroc_skf'])
    w.writeheader(); w.writerows(results)

print('=' * 80)
print('CLEAN-ONLY v0.3.1 DETECTOR — GroupKFold (no VIS leakage)')
print('=' * 80)
print('%-22s %-3s %-18s %4s %4s %8s %8s %8s' % ('Mode', 'Mdl', 'Target', 'N', 'Pos', 'GK_AUROC', 'SK_AUROC', 'GK_Bal'))
print('-' * 80)
for r in results:
    print('%-22s %-3s %-18s %4d %4s %8s %8s %8s' % (
        r['mode'], r['model'], r['target'], r['n'], r['pos'],
        r['auroc_gkf'], r['auroc_skf'], r['bal_gkf']))

# ── Rank unseen candidates ──
# Build features for universe windows using training data statistics
# Train best model on full data, predict on universe

best_row = None
for r in results:
    if r['target'] == 'cmd_specific' and isinstance(r['auroc_gkf'], float) and r['auroc_gkf'] > 0:
        if best_row is None or r['auroc_gkf'] > best_row['auroc_gkf']:
            best_row = r

if best_row:
    print()
    print('Best clean-only model: %s + %s (GK_AUROC=%.3f)' % (best_row['mode'], best_row['model'], best_row['auroc_gkf']))

    mode = best_row['mode']
    X_train = build_clean_features(paired, mode)
    if best_row['model'] == 'LR':
        ss = StandardScaler(); X_train = ss.fit_transform(X_train)

    if best_row['model'] == 'LR':
        m_best = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    elif best_row['model'] == 'RF':
        m_best = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    else:
        m_best = GradientBoostingClassifier(n_estimators=100, random_state=42)

    m_best.fit(X_train, y_cmd)

    # Build features for unseen universe windows
    training_keys = set()
    for p in paired:
        training_keys.add((p['task'], p['state_id'], int(p['window_start']), int(p['window_end'])))

    unseen = []
    for u in universe:
        task = u['task']; sid = u['state_id']
        ws = int(u['window_start']); we = int(u['window_end'])
        if (task, sid, ws, we) in held_out_windows: continue
        if (task, sid, ws, we) in training_keys: continue  # exclude paired 35
        unseen.append(u)

    # Build features for unseen
    unseen_feats = []
    for u in unseen:
        ws = float(u['window_start']); we = float(u['window_end'])
        wc = (ws + we) / 2.0
        fc = float(u.get('first_close_step', -1) or -1)
        lift = float(u.get('lift_step', -1) or -1)
        done_step = float(u.get('done_step', 280) or 280)

        # We don't have transition audit for unseen — use universe features
        # Estimate close-transition features from clean trace stats
        # For unseen: use universe-level defaults
        feats = [
            fc if fc > 0 else -1,
            lift if lift > 0 else -1,
            ws - fc if fc > 0 else 50,
            ws - lift if lift > 0 else 50,
            wc / max(done_step, 1),
            # Close-transition: use heuristics from universe
            0.0,  # distance_to_transition (unknown for unseen)
            float(u.get('clean_open_count', 0)),
            float(u.get('post_grasp_open_count', 0)),
            1 if str(u.get('after_first_close', '')).lower() in ('true', '1') else 0,
            # rand features: unknown for unseen, use 0
            0.0, 0.0, 0.0, 0.0,
        unseen_feats.append(feats)

    X_unseen = np.array(unseen_feats)
    if best_row['model'] == 'LR':
        X_unseen = ss.transform(X_unseen)

    p_cmd_unseen = m_best.predict_proba(X_unseen)[:, 1]

    # Rank
    ranked = []
    for i, u in enumerate(unseen):
        ranked.append({
            'task': u['task'], 'state_id': u['state_id'],
            'window_start': u['window_start'], 'window_end': u['window_end'],
            'phase': u.get('phase_id', '?'),
            'first_close_step': u.get('first_close_step', ''),
            'p_cmd_specific': round(float(p_cmd_unseen[i]), 4),
            'source': 'unseen_candidate_universe',
            'excluded_from_training': True,
        })

    ranked.sort(key=lambda r: -r['p_cmd_specific'])

    with open(TABLES + '/s20g_v031_cleanonly_unseen_candidates_for_s20h.csv', 'w', newline='') as f:
        fields = ['task','state_id','window_start','window_end','phase','first_close_step','p_cmd_specific','source','excluded_from_training']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(ranked)

    print()
    print('Top 10 UNSEEN candidates (leakage-free):')
    for r in ranked[:10]:
        print('  %-14s s%-1s w%-3d-%-3d phase=%-18s p_cmd=%.3f' % (
            r['task'], r['state_id'], r['window_start'], r['window_end'], r['phase'], r['p_cmd_specific']))

# ── Also save labeled positives for multiseed ──
positive_paired = [p for p in paired if p['classification'] in ('cmd_specific', 'task_effect', 'contact_effect_weak')]
with open(TABLES + '/s20g_v031_labeled_positive_candidates_for_multiseed.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(positive_paired[0].keys()), extrasaction='ignore')
    w.writeheader(); w.writerows(positive_paired)

print()
print('Labeled positives for multiseed: %d' % len(positive_paired))
print('Unseen candidates: %d' % len(ranked))
print('All outputs in: %s' % TABLES)
