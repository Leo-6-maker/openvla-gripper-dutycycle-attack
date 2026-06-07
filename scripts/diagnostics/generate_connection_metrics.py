#!/usr/bin/env python3
"""Generate end_to_end_detector_connection_metrics.csv"""
import csv
from collections import Counter

with open('/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/end_to_end_detector_join_table.csv') as f:
    rows = list(csv.DictReader(f))

valid = [r for r in rows if r['vuln_score_available']=='1' and r['label_status'] in ('positive','negative')]
for r in valid:
    r['true_val'] = 1 if r['label_status'] == 'positive' else 0
    r['pred_val'] = int(r['vulnerability_pred']) if r['vulnerability_pred'] in ('0','1') else -1
    r['mech'] = r['mechanism_type']
    r['phase_ok'] = r['phase_available'] == '1'

metrics = []

# Mode A: vuln-only
tp = sum(1 for r in valid if r['true_val']==1 and r['pred_val']==1)
fp = sum(1 for r in valid if r['true_val']==0 and r['pred_val']==1)
tn = sum(1 for r in valid if r['true_val']==0 and r['pred_val']==0)
fn = sum(1 for r in valid if r['true_val']==1 and r['pred_val']==0)
n = len(valid)
pos = tp+fn
neg = tn+fp
metrics.append({
    'mode': 'A_vulnerability_only', 'variant': 'V0_gold_only/LR/D_causal_safe',
    'n_eval': str(n), 'n_pos': str(pos), 'n_neg': str(neg),
    'tp': str(tp), 'fp': str(fp), 'tn': str(tn), 'fn': str(fn),
    'pos_recall': str(round(tp/pos, 4)) if pos else '',
    'neg_recall': str(round(tn/neg, 4)) if neg else '',
    'fpr': str(round(fp/neg, 4)) if neg else '',
    'accuracy': str(round((tp+tn)/n, 4)) if n else '',
    'balanced_acc': str(round(0.5*(tp/pos + tn/neg), 4)) if pos and neg else '',
    'hard_gate': 'no', 'phase_needed': 'no', 'recommended': 'baseline',
    'notes': 'Physical bridge pos recall %.3f; high FPR on no_action_bridge negs' % (tp/pos if pos else 0)
})

# Mode B: Phase-as-feature
metrics.append({
    'mode': 'B_phase_as_feature', 'variant': 'not_evaluated',
    'n_eval': '0', 'n_pos': '0', 'n_neg': '0',
    'tp': '0', 'fp': '0', 'tn': '0', 'fn': '0',
    'pos_recall': '', 'neg_recall': '', 'fpr': '', 'accuracy': '', 'balanced_acc': '',
    'hard_gate': 'no', 'phase_needed': 'yes', 'recommended': 'cannot_evaluate',
    'notes': 'No retraining. Phase features (hazard=0 for all vuln windows) would not help.'
})

# Mode C: Mechanism-aware routing
phys_alarm = sum(1 for r in valid if r['pred_val']==1 and r['mech']=='physical_bridge_positive')
phys_correct = sum(1 for r in valid if r['pred_val']==1 and r['mech']=='physical_bridge_positive' and r['true_val']==1)
policy_escalate = sum(1 for r in valid if r['pred_val']==1 and r['mech']=='negative_unclassified')
manual_review_c = sum(1 for r in valid if r['pred_val']==0)
metrics.append({
    'mode': 'C_mechanism_aware_routing', 'variant': 'V0_gold_only/LR/D_causal_safe',
    'n_eval': str(n), 'n_pos': str(pos), 'n_neg': str(neg),
    'tp': str(tp), 'fp': str(fp), 'tn': str(tn), 'fn': str(fn),
    'pos_recall': str(round(tp/pos, 4)) if pos else '',
    'neg_recall': str(round(tn/neg, 4)) if neg else '',
    'fpr': str(round(fp/neg, 4)) if neg else '',
    'accuracy': str(round((tp+tn)/n, 4)) if n else '',
    'balanced_acc': str(round(0.5*(tp/pos + tn/neg), 4)) if pos and neg else '',
    'hard_gate': 'no', 'phase_needed': 'no', 'recommended': 'RECOMMENDED',
    'notes': '%d physical_alarms(%d/%d correct); %d policy_escalations; %d manual_review; %d suppressed' % (
        phys_alarm, phys_correct, phys_alarm, policy_escalate, manual_review_c, 0)
})

# Mode D: Soft modulation
phase_rows = [r for r in valid if r['phase_ok']]
all_hazard_zero = all(float(r.get('hazard_score_mean', 0) or 0) == 0.0 for r in phase_rows)
metrics.append({
    'mode': 'D_soft_modulation', 'variant': 'phase_factor_0.5_constant',
    'n_eval': str(len(phase_rows)),
    'n_pos': str(sum(1 for r in phase_rows if r['true_val']==1)),
    'n_neg': str(sum(1 for r in phase_rows if r['true_val']==0)),
    'tp': str(tp), 'fp': str(fp), 'tn': str(tn), 'fn': str(fn),
    'pos_recall': '', 'neg_recall': '', 'fpr': '', 'accuracy': '', 'balanced_acc': '',
    'hard_gate': 'no', 'phase_needed': 'yes', 'recommended': 'diagnostic_only',
    'notes': 'All hazard=0 for vuln windows; phase factor constant 0.5. No discrimination.' + (' ALL_ZERO_useless' if all_hazard_zero else ' has_variance')
})

# Mode E: Hard gate
metrics.append({
    'mode': 'E_hard_gate_REJECTED', 'variant': 'phase_gate_AND_vuln',
    'n_eval': '7', 'n_pos': '7', 'n_neg': '0',
    'tp': '0', 'fp': '0', 'tn': '0', 'fn': '7',
    'pos_recall': '0.0', 'neg_recall': '', 'fpr': '', 'accuracy': '0.0', 'balanced_acc': '',
    'hard_gate': 'yes_REJECTED', 'phase_needed': 'yes', 'recommended': 'DO_NOT_USE',
    'notes': 'POC: 0/7 positive recall. Phase gate blocks ALL vulnerability positives.'
})

# Mechanism-stratified
for mech in sorted(set(r['mech'] for r in valid)):
    subset = [r for r in valid if r['mech'] == mech]
    sp = sum(1 for r in subset if r['true_val']==1)
    sn = sum(1 for r in subset if r['true_val']==0)
    sp_hit = sum(1 for r in subset if r['true_val']==1 and r['pred_val']==1)
    sn_hit = sum(1 for r in subset if r['true_val']==0 and r['pred_val']==0)
    metrics.append({
        'mode': 'mechanism_stratified', 'variant': mech,
        'n_eval': str(len(subset)), 'n_pos': str(sp), 'n_neg': str(sn),
        'tp': str(sp_hit), 'fp': str(sn-sn_hit), 'tn': str(sn_hit), 'fn': str(sp-sp_hit),
        'pos_recall': str(round(sp_hit/sp, 4)) if sp else '',
        'neg_recall': str(round(sn_hit/sn, 4)) if sn else '',
        'fpr': str(round((sn-sn_hit)/sn, 4)) if sn else '',
        'accuracy': str(round((sp_hit+sn_hit)/len(subset), 4)) if subset else '',
        'balanced_acc': str(round(0.5*(sp_hit/sp + sn_hit/sn), 4)) if sp and sn else '',
        'hard_gate': 'no', 'phase_needed': 'no', 'recommended': '', 'notes': ''
    })

# Phase coverage
with_phase = sum(1 for r in valid if r['phase_ok'])
metrics.append({
    'mode': 'coverage', 'variant': 'phase_coverage',
    'n_eval': str(len(valid)), 'n_pos': '', 'n_neg': '',
    'tp': str(with_phase), 'fp': str(len(valid) - with_phase), 'tn': '', 'fn': '',
    'pos_recall': 'phase_available_pct',
    'neg_recall': str(round(with_phase/len(valid), 4)) if valid else '',
    'fpr': '', 'accuracy': '', 'balanced_acc': '',
    'hard_gate': 'no', 'phase_needed': 'n/a', 'recommended': '',
    'notes': '%d/%d rows have phase data' % (with_phase, len(valid))
})

# Full pipeline coverage
fj = [r for r in rows if r['join_status'] == 'fully_joined']
metrics.append({
    'mode': 'coverage', 'variant': 'full_pipeline',
    'n_eval': str(len(fj)), 'n_pos': '', 'n_neg': '',
    'tp': str(len(rows)), 'fp': '', 'tn': '', 'fn': '',
    'pos_recall': 'fully_joined_pct',
    'neg_recall': str(round(len(fj)/len(rows), 4)) if rows else '',
    'fpr': '', 'accuracy': '', 'balanced_acc': '',
    'hard_gate': 'no', 'phase_needed': 'yes', 'recommended': '',
    'notes': '%d/%d rows fully joined (phase+vuln+mech)' % (len(fj), len(rows))
})

# Write
out = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/end_to_end_detector_connection_metrics.csv'
cols = ['mode','variant','n_eval','n_pos','n_neg','tp','fp','tn','fn',
        'pos_recall','neg_recall','fpr','accuracy','balanced_acc',
        'hard_gate','phase_needed','recommended','notes']
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(metrics)
print('Wrote %d rows to %s' % (len(metrics), out))
for m in metrics:
    print('  %s/%s: %s' % (m['mode'], m['variant'], m.get('notes','')[:100]))
