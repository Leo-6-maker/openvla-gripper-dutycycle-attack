"""Stage G: C4 calibration — pooled monotonic Platt.

Fits: p = sigmoid(a * raw_logit + b), a > 0
Compares: raw sigmoid vs calibrated (NLL, Brier, ECE)
Audits: unknown/terminal step scores (report-only)
"""
import json, os, sys, hashlib, numpy as np
from collections import defaultdict
from scipy.optimize import minimize

EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
C4_DIR = EVIDENCE + '/c4_raw_ranking_v1'
PRED_PATH = C4_DIR + '/predictions/c4_predictions.json'
C4_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_C4_IDENTITY_MANIFEST_V1.json'
FEAT_ROOT = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOTS = [
    '/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
]
OUT_ROOT = EVIDENCE + '/c4_calibration_v1'
os.makedirs(OUT_ROOT, exist_ok=True)

K10 = 10

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def safe_sigmoid(x):
    """Numerically stable sigmoid for extreme logits."""
    result = np.zeros_like(x, dtype=np.float64)
    pos_mask = x >= 0
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    neg_mask = ~pos_mask
    result[neg_mask] = np.exp(x[neg_mask]) / (1.0 + np.exp(x[neg_mask]))
    return result

# ── 1. Load C4 step-level data with labels ──
print('=== C4 CALIBRATION ===')
print()
print('--- Loading C4 step-level data ---')

c4_ids = json.load(open(C4_MANIFEST_PATH))['identities']
preds = json.load(open(PRED_PATH))

# Extract step-level data with labels
all_logits = []; all_labels = []; all_eids = []; all_steps = []
all_suites = []; all_cc = []
unknown_logits = []; unknown_eids = []; unknown_steps = []
terminal_logits = []; terminal_eids = []; terminal_steps = []

for eid in sorted(preds.keys()):
    p = preds[eid]
    suite, task, state = eid.split('/')

    # Load labels for this episode
    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
    lp = None
    for root in LABEL_ROOTS:
        candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
        if os.path.isfile(candidate): lp = candidate; break
    if not os.path.isfile(fp) or lp is None: continue

    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
    labels.sort(key=lambda r: r['step'])
    T = len(recs); max_t = min(T, T-K10+1)

    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and
                       labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)
    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)

    scores = np.array(p['step_scores_raw_logit'])

    for t in range(T):
        logit = float(scores[t])
        known = k10_k[t]
        feasible = k10_s[t]
        in_max_t = t < max_t

        if known:
            if feasible:
                all_logits.append(logit); all_labels.append(1)
                all_eids.append(eid); all_steps.append(t); all_suites.append(suite)
            else:
                all_logits.append(logit); all_labels.append(0)
                all_eids.append(eid); all_steps.append(t); all_suites.append(suite)
        elif in_max_t:
            # Unknown within valid range
            unknown_logits.append(logit); unknown_eids.append(eid); unknown_steps.append(t)
        elif not in_max_t:
            # Terminal: last K10-1 steps where feasibility can't be determined
            terminal_logits.append(logit); terminal_eids.append(eid); terminal_steps.append(t)

all_logits = np.array(all_logits); all_labels = np.array(all_labels)
unknown_logits = np.array(unknown_logits)
terminal_logits = np.array(terminal_logits)

n_pos = int(all_labels.sum()); n_neg = len(all_labels) - n_pos
print('Known steps: {} (pos={} neg={} prevalence={:.4f})'.format(
    len(all_labels), n_pos, n_neg, n_pos/max(len(all_labels),1)))
print('Unknown steps: {}'.format(len(unknown_logits)))
print('Terminal steps: {}'.format(len(terminal_logits)))

# ── 2. Raw sigmoid baseline ──
print('\n--- Raw sigmoid baseline ---')
raw_prob = sigmoid(all_logits)

def compute_brier(probs, labels):
    return float(((probs - labels) ** 2).mean())

def compute_nll(probs, labels):
    eps = 1e-15
    p = np.clip(probs, eps, 1 - eps)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; total = len(labels)
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if in_bin.sum() == 0: continue
        bin_acc = labels[in_bin].mean()
        bin_conf = probs[in_bin].mean()
        ece += (in_bin.sum() / total) * abs(bin_acc - bin_conf)
    # Handle last bin inclusive
    in_last = probs >= bin_boundaries[-1]
    if in_last.sum() > 0:
        ece += (in_last.sum() / total) * abs(labels[in_last].mean() - probs[in_last].mean())
    return float(ece)

raw_nll = compute_nll(raw_prob, all_labels)
raw_brier = compute_brier(raw_prob, all_labels)
raw_ece = compute_ece(raw_prob, all_labels)
print('Raw NLL:   {:.6f}'.format(raw_nll))
print('Raw Brier: {:.6f}'.format(raw_brier))
print('Raw ECE:   {:.6f}'.format(raw_ece))

# ── 3. Platt calibration ──
print('\n--- Platt calibration ---')

def platt_nll(params, logits, labels):
    a, b = params
    probs = safe_sigmoid(a * logits + b)
    eps = 1e-15
    p = np.clip(probs, eps, 1 - eps)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

# Fit: constrain a > 0, b unrestricted
result = minimize(lambda p: platt_nll(p, all_logits, all_labels),
                  x0=[1.0, 0.0],
                  bounds=[(1e-6, None), (None, None)],
                  method='L-BFGS-B')
a, b = result.x
print('Platt params: a={:.6f} b={:.6f}'.format(a, b))
print('Optimization success: {}'.format(result.success))

cal_prob = safe_sigmoid(a * all_logits + b)
cal_nll = compute_nll(cal_prob, all_labels)
cal_brier = compute_brier(cal_prob, all_labels)
cal_ece = compute_ece(cal_prob, all_labels)
print('Cal NLL:   {:.6f}'.format(cal_nll))
print('Cal Brier: {:.6f}'.format(cal_brier))
print('Cal ECE:   {:.6f}'.format(cal_ece))

# ── 4. Ranking preservation check ──
print('\n--- Ranking preservation ---')
# Monotonic transform with a>0 is mathematically guaranteed to preserve ranking.
# Check for any >0.1% inversions that would indicate implementation bug.
n_check = min(10000, len(all_logits))
indices = np.random.choice(len(all_logits), n_check, replace=False)
# Compare raw sigmoid vs calibrated prob ordering
raw_check = sigmoid(all_logits[indices])
cal_check = safe_sigmoid(a * all_logits[indices] + b)
raw_order = np.argsort(raw_check)
cal_order = np.argsort(cal_check)
# Count strict inversions (where difference > 1e-12 to exclude ties)
tie_tolerance = 1e-12
raw_sorted = raw_check[raw_order]
cal_sorted = cal_check[cal_order]
# Compute Kendall tau-like agreement
agreement = float((raw_order == cal_order).mean())
n_diff = int((raw_order != cal_order).sum())
# Check if differences are just numerical ties
significant_inversions = 0
for i in range(len(raw_order)):
    ri = raw_order[i]; ci = cal_order[i]
    if ri != ci:
        if abs(raw_check[ri] - raw_check[ci]) > tie_tolerance:
            significant_inversions += 1
ranking_preserved = significant_inversions == 0
if not ranking_preserved:
    print('WARNING: {} significant ranking inversions (agreement={:.6f})'.format(
        significant_inversions, agreement))
else:
    print('Ranking preserved: PASS (agreement={:.6f}, {}/{} tie-level diffs)'.format(
        agreement, n_diff - significant_inversions, n_check))

# ── 5. Cross-fit diagnostic ──
print('\n--- Identity-blocked cross-fit diagnostic ---')
unique_eids = sorted(set(all_eids))
n_cv = min(5, len(unique_eids))
np.random.seed(42)
eid_list = np.array(unique_eids)
np.random.shuffle(eid_list)
folds = np.array_split(eid_list, n_cv)

cv_nll_raw = []; cv_nll_cal = []; cv_brier_raw = []; cv_brier_cal = []
cv_a = []; cv_b = []

for fi in range(n_cv):
    val_eids_fold = set(folds[fi])
    train_mask = np.array([eid not in val_eids_fold for eid in all_eids])
    val_mask = ~train_mask

    train_logits = all_logits[train_mask]; train_labels = all_labels[train_mask]
    val_logits = all_logits[val_mask]; val_labels = all_labels[val_mask]

    if len(train_logits) < 10 or len(val_logits) < 10: continue

    r = minimize(lambda p: platt_nll(p, train_logits, train_labels),
                 x0=[1.0, 0.0], bounds=[(1e-6, None), (None, None)], method='L-BFGS-B')
    a_fold, b_fold = r.x
    cv_a.append(a_fold); cv_b.append(b_fold)

    val_cal = safe_sigmoid(a_fold * val_logits + b_fold)
    val_raw = sigmoid(val_logits)

    cv_nll_raw.append(compute_nll(val_raw, val_labels))
    cv_nll_cal.append(compute_nll(val_cal, val_labels))
    cv_brier_raw.append(compute_brier(val_raw, val_labels))
    cv_brier_cal.append(compute_brier(val_cal, val_labels))

print('Cross-fit ({}-fold):'.format(len(cv_a)))
print('  Raw NLL:   {:.6f} +- {:.6f}'.format(np.mean(cv_nll_raw), np.std(cv_nll_raw)))
print('  Cal NLL:   {:.6f} +- {:.6f}'.format(np.mean(cv_nll_cal), np.std(cv_nll_cal)))
print('  Raw Brier: {:.6f} +- {:.6f}'.format(np.mean(cv_brier_raw), np.std(cv_brier_raw)))
print('  Cal Brier: {:.6f} +- {:.6f}'.format(np.mean(cv_brier_cal), np.std(cv_brier_cal)))
print('  Platt a:   {:.6f} +- {:.6f}'.format(np.mean(cv_a), np.std(cv_a)))
print('  Platt b:   {:.6f} +- {:.6f}'.format(np.mean(cv_b), np.std(cv_b)))

cross_fit_stable = all(ai > 0 for ai in cv_a) and np.std(cv_a) / max(abs(np.mean(cv_a)), 1e-8) < 1.0

# ── 6. Unknown/terminal audit (report-only) ──
print('\n--- Unknown/terminal step audit ---')
if len(unknown_logits) > 0:
    unk_prob = safe_sigmoid(a * unknown_logits + b)
    unk_raw = sigmoid(unknown_logits)
    print('Unknown steps (n={}):'.format(len(unknown_logits)))
    print('  Raw sigmoid: mean={:.6f} median={:.6f} >0.5={:.4f}'.format(
        unk_raw.mean(), np.median(unk_raw), (unk_raw > 0.5).mean()))
    print('  Cal prob:    mean={:.6f} median={:.6f} >0.5={:.4f}'.format(
        unk_prob.mean(), np.median(unk_prob), (unk_prob > 0.5).mean()))
else:
    print('Unknown steps: 0')

if len(terminal_logits) > 0:
    term_prob = safe_sigmoid(a * terminal_logits + b)
    term_raw = sigmoid(terminal_logits)
    print('Terminal steps (n={}):'.format(len(terminal_logits)))
    print('  Raw sigmoid: mean={:.6f} median={:.6f} >0.5={:.4f}'.format(
        term_raw.mean(), np.median(term_raw), (term_raw > 0.5).mean()))
    print('  Cal prob:    mean={:.6f} median={:.6f} >0.5={:.4f}'.format(
        term_prob.mean(), np.median(term_prob), (term_prob > 0.5).mean()))
else:
    print('Terminal steps: 0')

# ── 7. Per-suite calibration ──
print('\n--- Per-suite calibration ---')
suite_data = defaultdict(lambda: {'logits': [], 'labels': []})
for i in range(len(all_logits)):
    suite_data[all_suites[i]]['logits'].append(all_logits[i])
    suite_data[all_suites[i]]['labels'].append(all_labels[i])

for suite in sorted(suite_data.keys()):
    d = suite_data[suite]
    sl = np.array(d['logits']); sb = np.array(d['labels'])
    raw_s = sigmoid(sl)
    cal_s = safe_sigmoid(a * sl + b)
    print('  {}: n={} pos={} raw_nll={:.4f} cal_nll={:.4f} raw_ece={:.4f} cal_ece={:.4f}'.format(
        suite, len(sl), int(sb.sum()),
        compute_nll(raw_s, sb), compute_nll(cal_s, sb),
        compute_ece(raw_s, sb), compute_ece(cal_s, sb)))

# ── 8. Score distribution check ──
print('\n--- Score distribution ---')
print('Raw sigmoid: mean={:.6f} median={:.6f} min={:.6f} max={:.6f} std={:.6f}'.format(
    raw_prob.mean(), np.median(raw_prob), raw_prob.min(), raw_prob.max(), raw_prob.std()))
print('Cal prob:    mean={:.6f} median={:.6f} min={:.6f} max={:.6f} std={:.6f}'.format(
    cal_prob.mean(), np.median(cal_prob), cal_prob.min(), cal_prob.max(), cal_prob.std()))

unique_raw = len(set(raw_prob.round(10)))
unique_cal = len(set(cal_prob.round(10)))
print('Unique raw scores: {}  Unique cal scores: {}'.format(unique_raw, unique_cal))

# Saturation check
raw_sat_0 = (raw_prob < 1e-6).mean(); raw_sat_1 = (raw_prob > 1 - 1e-6).mean()
cal_sat_0 = (cal_prob < 1e-6).mean(); cal_sat_1 = (cal_prob > 1 - 1e-6).mean()
print('Raw saturation:  near-0={:.6f} near-1={:.6f}'.format(raw_sat_0, raw_sat_1))
print('Cal saturation:  near-0={:.6f} near-1={:.6f}'.format(cal_sat_0, cal_sat_1))

# ── 9. Decision ──
print('\n' + '='*60)
nll_improves = cal_nll < raw_nll
brier_ok = cal_brier <= raw_brier * 1.01  # Allow 1% tolerance
ece_ok = cal_ece <= raw_ece * 1.01
finite_ok = np.isfinite(cal_prob).all()
nonconstant_ok = cal_prob.std() > 1e-10
no_pathological_sat = cal_sat_0 < 0.95 and cal_sat_1 < 0.95
a_positive = a > 0

conditions = {
    'a_positive': (a_positive, a),
    'finite': (finite_ok, None),
    'nonconstant': (nonconstant_ok, cal_prob.std()),
    'ranking_preserved': (ranking_preserved, None),
    'nll_improves': (nll_improves, '{} -> {}'.format(raw_nll, cal_nll)),
    'brier_ok': (brier_ok, '{} -> {}'.format(raw_brier, cal_brier)),
    'ece_ok': (ece_ok, '{} -> {}'.format(raw_ece, cal_ece)),
    'no_pathological_sat': (no_pathological_sat, 'sat0={:.4f} sat1={:.4f}'.format(cal_sat_0, cal_sat_1)),
    'cross_fit_stable': (cross_fit_stable, 'a={:.4f}+-{:.4f}'.format(np.mean(cv_a), np.std(cv_a)))
}

all_pass = all(v[0] for v in conditions.values())

if all_pass:
    decision = 'PLATT_FROZEN'
    print('DECISION: PLATT_FROZEN')
else:
    decision = 'RAW_IDENTITY'
    print('DECISION: RAW_IDENTITY')

for name, (passed, detail) in conditions.items():
    print('  {}: {} {}'.format(name, 'PASS' if passed else 'FAIL', detail if detail else ''))

print('\nScore transform: {}'.format(decision))

# ── 10. Write outputs ──
print('\n--- Writing outputs ---')

# Acceptance
acceptance = {
    'schema': 'C4_CALIBRATION_ACCEPTANCE_V1',
    'calibration_family': 'pooled_monotonic_Platt',
    'formula': 'p = sigmoid(a * raw_logit + b), a > 0',
    'decision': decision,
    'conditions': {k: {'passed': bool(v[0]), 'detail': str(v[1]) if v[1] is not None else None}
                   for k, v in conditions.items()},
    'platt_a': float(a), 'platt_b': float(b),
    'raw_metrics': {'nll': float(raw_nll), 'brier': float(raw_brier), 'ece': float(raw_ece)},
    'cal_metrics': {'nll': float(cal_nll), 'brier': float(cal_brier), 'ece': float(cal_ece)}
}
with open(os.path.join(OUT_ROOT, 'C4_CALIBRATION_ACCEPTANCE_V1.json'), 'w') as f:
    json.dump(acceptance, f, indent=2)

# Calibrator
if decision == 'PLATT_FROZEN':
    calibrator = {
        'schema': 'C4_CALIBRATOR_V1',
        'type': 'pooled_monotonic_Platt',
        'a': float(a), 'b': float(b),
        'formula': 'calibrated_prob = sigmoid({} * raw_logit + {})'.format(a, b),
        'constraints': 'a > 0, pooled over all C4 known steps',
        'fitted_on': 'C4 known feasible + known infeasible steps'
    }
    with open(os.path.join(OUT_ROOT, 'C4_CALIBRATOR_V1.json'), 'w') as f:
        json.dump(calibrator, f, indent=2)
else:
    raw_id = {
        'schema': 'RAW_IDENTITY_FREEZE_V1',
        'score_transform': 'RAW_IDENTITY_SIGMOID',
        'definition': 'p = sigmoid(raw_logit), no Platt transform',
        'reason': 'Calibration did not meet acceptance conditions'
    }
    with open(os.path.join(OUT_ROOT, 'RAW_IDENTITY_FREEZE_V1.json'), 'w') as f:
        json.dump(raw_id, f, indent=2)

# Unknown/terminal audit
ut_audit = {
    'schema': 'UNKNOWN_TERMINAL_SCORE_AUDIT_V1',
    'unknown_steps': {
        'count': len(unknown_logits),
        'raw_sigmoid_mean': float(sigmoid(unknown_logits).mean()) if len(unknown_logits) > 0 else None,
        'raw_sigmoid_gt_0_5': float((sigmoid(unknown_logits) > 0.5).mean()) if len(unknown_logits) > 0 else None
    },
    'terminal_steps': {
        'count': len(terminal_logits),
        'raw_sigmoid_mean': float(sigmoid(terminal_logits).mean()) if len(terminal_logits) > 0 else None,
        'raw_sigmoid_gt_0_5': float((sigmoid(terminal_logits) > 0.5).mean()) if len(terminal_logits) > 0 else None
    },
    'p4_implication': 'P4 scheduler must treat emit on unknown/terminal steps as INVALID EMIT, not TP'
}
with open(os.path.join(OUT_ROOT, 'UNKNOWN_TERMINAL_SCORE_AUDIT_V1.json'), 'w') as f:
    json.dump(ut_audit, f, indent=2)

# Calibration receipt
receipt = {
    'schema': 'C4_CALIBRATION_RECEIPT_V1',
    'decision': decision,
    'platt_a': float(a), 'platt_b': float(b),
    'raw_metrics': {'nll': raw_nll, 'brier': raw_brier, 'ece': raw_ece},
    'cal_metrics': {'nll': cal_nll, 'brier': cal_brier, 'ece': cal_ece},
    'per_suite': {suite: {'nll_raw': compute_nll(sigmoid(np.array(d['logits'])), np.array(d['labels'])),
                          'nll_cal': compute_nll(safe_sigmoid(a * np.array(d['logits']) + b), np.array(d['labels']))}
                  for suite, d in suite_data.items()},
    'cross_fit': {'n_folds': len(cv_a), 'a_mean': float(np.mean(cv_a)), 'a_std': float(np.std(cv_a)),
                  'raw_nll_mean': float(np.mean(cv_nll_raw)), 'cal_nll_mean': float(np.mean(cv_nll_cal))},
    'unknown_terminal_audit': 'UNKNOWN_TERMINAL_SCORE_AUDIT_V1.json'
}
with open(os.path.join(OUT_ROOT, 'C4_CALIBRATION_RECEIPT_V1.json'), 'w') as f:
    json.dump(receipt, f, indent=2)

# SHA256SUMS
all_files = []
for root, dirs, fns in os.walk(OUT_ROOT):
    for fn in sorted(fns):
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_ROOT)
        if fn == 'SHA256SUMS' or fn.endswith('.sha256'): continue
        all_files.append((rel, sha256_file(fp)))
sums_path = os.path.join(OUT_ROOT, 'SHA256SUMS')
with open(sums_path, 'w') as f:
    for rel, h in sorted(all_files):
        f.write('{}  {}\n'.format(h, rel))
sums_sha = sha256_file(sums_path)
with open(os.path.join(OUT_ROOT, 'SHA256SUMS.sha256'), 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sums_sha))

print('\nOutput: {}'.format(OUT_ROOT))
print('SHA256SUMS: {}'.format(sums_sha[:16]))
print('Decision: {}'.format(decision))
print('DONE.')
