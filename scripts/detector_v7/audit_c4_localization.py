"""C4 post-hoc localization audit. Re-examines frozen C4 prediction bundle.

Does NOT retrain, recalibrate, or access P4/H2.
Checks: step alignment, inside/outside distribution, top-k implementation, argmax autopsy.
"""
import json, os, sys, numpy as np
from collections import defaultdict

EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
FEAT_ROOT = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/clean'
C4_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_C4_IDENTITY_MANIFEST_V1.json'
LABEL_ROOTS = [
    '/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
]
PRED_PATH = EVIDENCE + '/c4_raw_ranking_v1/predictions/c4_predictions.json'
K10 = 10

def load_episode_metadata(eid):
    """Re-load episode metadata for step-level ground truth."""
    suite, task, state = eid.split('/')
    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
    lp = None
    for root in LABEL_ROOTS:
        candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
        if os.path.isfile(candidate): lp = candidate; break
    if not os.path.isfile(fp) or lp is None: return None
    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
    labels.sort(key=lambda r: r['step'])
    T = len(recs); max_t = min(T, T-K10+1)
    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and
                       labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)
    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)
    cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False)
                    for t in range(T)], dtype=bool)
    has_opp = bool(k10_s[:max_t].any())
    return {'eid': eid, 'T': T, 'max_t': max_t, 'k10_s': k10_s, 'k10_k': k10_k,
            'cc': cc, 'has_opp': has_opp, 'suite': suite}

print('=== C4 LOCALIZATION AUDIT ===')
print()

# Load predictions
preds = json.load(open(PRED_PATH))
print('Loaded {} predictions'.format(len(preds)))

# Load C4 manifest
c4_ids = json.load(open(C4_MANIFEST_PATH))['identities']

# Load metadata for each C4 episode
print('Loading episode metadata...')
meta = {}
for eid in c4_ids:
    m = load_episode_metadata(eid)
    if m is not None: meta[eid] = m

opp_eids = [eid for eid, m in meta.items() if m['has_opp']]
print('C4: {} total, {} opp'.format(len(meta), len(opp_eids)))

# ── 1. Step alignment audit ──
print('\n' + '='*60)
print('1. STEP ALIGNMENT AUDIT')
print('='*60)

alignment_issues = []
for eid in sorted(opp_eids)[:5]:  # Detailed check on first 5
    m = meta[eid]; p = preds[eid]
    T_meta = m['T']; T_pred = len(p['step_scores_raw_logit'])
    max_t_meta = m['max_t']

    feasible_steps = np.where(m['k10_s'][:max_t_meta])[0]
    first_feasible = feasible_steps[0] if len(feasible_steps) > 0 else -1
    last_feasible = feasible_steps[-1] if len(feasible_steps) > 0 else -1

    cc_steps = np.where(m['cc'][:max_t_meta])[0]
    n_cc = len(cc_steps)

    scores = np.array(p['step_scores_raw_logit'])
    argmax = int(np.argmax(scores[:max_t_meta])) if max_t_meta > 0 else -1
    max_score = float(scores[argmax]) if argmax >= 0 else float('nan')

    print('\n  {} T={} max_t={} pred_len={} has_opp={}'.format(eid, T_meta, max_t_meta, T_pred, m['has_opp']))
    print('  Feasible steps: {} (first={} last={})'.format(len(feasible_steps), first_feasible, last_feasible))
    print('  Candidate-close steps: {}'.format(n_cc))
    print('  Argmax: step={} score={:.4f}'.format(argmax, max_score))
    print('  Score range: [{:.4f}, {:.4f}]'.format(scores.min(), scores.max()))
    print('  Score mean: {:.4f} median: {:.4f} std: {:.4f}'.format(scores.mean(), float(np.median(scores)), scores.std()))

    # Check for off-by-one
    if T_meta != T_pred:
        alignment_issues.append('{} length mismatch: meta T={} pred T={}'.format(eid, T_meta, T_pred))
        print('  WARNING: Length mismatch!')

print('\n  Summary: {} alignment issues found'.format(len(alignment_issues)))

# ── 2. Score distribution audit ──
print('\n' + '='*60)
print('2. SCORE DISTRIBUTION AUDIT')
print('='*60)

all_inside = []; all_outside = []; all_unknown = []
per_ep_max_inside = []; per_ep_max_outside = []
per_ep_mean_inside = []; per_ep_mean_outside = []

for eid in opp_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']

    # Inside = feasible steps within [0, max_t)
    inside_mask = m['k10_s'][:max_t]
    # Outside = known infeasible steps within [0, max_t)
    outside_mask = m['k10_k'][:max_t] & (~m['k10_s'][:max_t])
    # Unknown = steps with unknown mask = false

    if inside_mask.any():
        all_inside.extend(scores[:max_t][inside_mask].tolist())
        per_ep_max_inside.append(float(scores[:max_t][inside_mask].max()))
        per_ep_mean_inside.append(float(scores[:max_t][inside_mask].mean()))
    if outside_mask.any():
        all_outside.extend(scores[:max_t][outside_mask].tolist())
        per_ep_max_outside.append(float(scores[:max_t][outside_mask].max()))
        per_ep_mean_outside.append(float(scores[:max_t][outside_mask].mean()))

all_inside = np.array(all_inside); all_outside = np.array(all_outside)

print('Inside (feasible) steps: n={}'.format(len(all_inside)))
print('  mean={:.4f}  median={:.4f}  std={:.4f}'.format(
    all_inside.mean(), np.median(all_inside), all_inside.std()))
print('  p10={:.4f}  p25={:.4f}  p75={:.4f}  p90={:.4f}'.format(
    np.percentile(all_inside, 10), np.percentile(all_inside, 25),
    np.percentile(all_inside, 75), np.percentile(all_inside, 90)))
print('  min={:.4f}  max={:.4f}'.format(all_inside.min(), all_inside.max()))

print('Outside (known infeasible) steps: n={}'.format(len(all_outside)))
print('  mean={:.4f}  median={:.4f}  std={:.4f}'.format(
    all_outside.mean(), np.median(all_outside), all_outside.std()))
print('  p10={:.4f}  p25={:.4f}  p75={:.4f}  p90={:.4f}'.format(
    np.percentile(all_outside, 10), np.percentile(all_outside, 25),
    np.percentile(all_outside, 75), np.percentile(all_outside, 90)))
print('  min={:.4f}  max={:.4f}'.format(all_outside.min(), all_outside.max()))

# Per-episode max inside vs max outside
# Align: only compare episodes that have both inside and outside
common_eids = []
for eid in opp_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']
    has_in = m['k10_s'][:max_t].any()
    has_out = (m['k10_k'][:max_t] & (~m['k10_s'][:max_t])).any()
    if has_in and has_out:
        common_eids.append(eid)

per_ep_max_inside = []; per_ep_max_outside = []
for eid in common_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']
    inside_mask = m['k10_s'][:max_t]
    outside_mask = m['k10_k'][:max_t] & (~m['k10_s'][:max_t])
    per_ep_max_inside.append(float(scores[:max_t][inside_mask].max()))
    per_ep_max_outside.append(float(scores[:max_t][outside_mask].max()))
per_ep_max_inside = np.array(per_ep_max_inside)
per_ep_max_outside = np.array(per_ep_max_outside)
max_in_gt_max_out = (per_ep_max_inside > per_ep_max_outside).mean()
print('\nPer-episode max_inside > max_outside: {:.4f} ({}/{})'.format(
    max_in_gt_max_out, int((per_ep_max_inside > per_ep_max_outside).sum()), len(per_ep_max_inside)))
print('  max_inside  mean={:.4f} median={:.4f}'.format(per_ep_max_inside.mean(), np.median(per_ep_max_inside)))
print('  max_outside mean={:.4f} median={:.4f}'.format(per_ep_max_outside.mean(), np.median(per_ep_max_outside)))
print('  margin (max_in - max_out) mean={:.4f} median={:.4f}'.format(
    (per_ep_max_inside - per_ep_max_outside).mean(),
    np.median(per_ep_max_inside - per_ep_max_outside)))

# Also handle eval with sigmoid
print('\n--- Sigmoid-transformed scores ---')
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
all_inside_sig = sigmoid(all_inside)
all_outside_sig = sigmoid(all_outside)
print('Inside (sigmoid):  mean={:.6f}  median={:.6f}  p10={:.6f}  p90={:.6f}'.format(
    all_inside_sig.mean(), np.median(all_inside_sig),
    np.percentile(all_inside_sig, 10), np.percentile(all_inside_sig, 90)))
print('Outside (sigmoid): mean={:.6f}  median={:.6f}  p10={:.6f}  p90={:.6f}'.format(
    all_outside_sig.mean(), np.median(all_outside_sig),
    np.percentile(all_outside_sig, 10), np.percentile(all_outside_sig, 90)))
# Fraction inside steps with sigmoid > 0.5
inside_above_half = (all_inside_sig > 0.5).mean()
outside_above_half = (all_outside_sig > 0.5).mean()
print('Inside steps with sigmoid > 0.5: {:.4f}'.format(inside_above_half))
print('Outside steps with sigmoid > 0.5: {:.4f}'.format(outside_above_half))

# ── 3. Top-K corridor hit — CORRECTED implementation ──
print('\n' + '='*60)
print('3. TOP-K CORRIDOR HIT (CORRECTED)')
print('='*60)

# Original: argmax over all steps in [0, max_t)
# Correct: argmax over candidate_close steps in [0, max_t) — matches episode scoring
# But top-K should be over ALL valid steps, with corridor = [first_feasible, first_feasible + K10)

top1_hit_cc = 0; top3_hit_cc = 0  # Top-K among candidate_close steps
top1_hit_all = 0; top3_hit_all = 0  # Top-K among all steps
total = 0
offsets_all = []; offsets_cc = []
n_outside_above_max_inside = []
rank_of_best_inside = []

for eid in opp_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']

    feasible = np.where(m['k10_s'][:max_t])[0]
    if len(feasible) == 0: continue
    first = feasible[0]
    corridor_end = min(first + K10, max_t)
    total += 1

    # Top-K among ALL steps [0, max_t)
    order_all = np.argsort(scores[:max_t])[::-1]
    top1_all = order_all[0]
    top3_all = order_all[:3]
    if first <= top1_all < corridor_end: top1_hit_all += 1
    if any(first <= t < corridor_end for t in top3_all): top3_hit_all += 1
    offsets_all.append(int(top1_all) - first)

    # Top-K among CANDIDATE_CLOSE steps [0, max_t)
    cc_mask = m['cc'][:max_t]
    if cc_mask.any():
        cc_indices = np.where(cc_mask)[0]
        cc_scores = scores[:max_t][cc_mask]
        order_cc = np.argsort(cc_scores)[::-1]
        top1_cc = cc_indices[order_cc[0]]
        top3_cc = cc_indices[order_cc[:min(3, len(order_cc))]]
        if first <= top1_cc < corridor_end: top1_hit_cc += 1
        if any(first <= t < corridor_end for t in top3_cc): top3_hit_cc += 1
        offsets_cc.append(int(top1_cc) - first)

    # Rank of best inside step among all scores
    inside_scores = scores[:max_t][feasible]
    best_inside_score = inside_scores.max()
    # How many outside (known infeasible) steps have score > best_inside?
    outside_mask = m['k10_k'][:max_t] & (~m['k10_s'][:max_t])
    n_above = int((scores[:max_t][outside_mask] > best_inside_score).sum()) if outside_mask.any() else 0
    n_outside_above_max_inside.append(n_above)
    # Rank: count of all steps with score > best_inside
    rank = int((scores[:max_t] > best_inside_score).sum()) + 1
    rank_of_best_inside.append(rank)

print('Top-1 corridor hit (ALL steps):       {:.4f} ({}/{})'.format(top1_hit_all/max(total,1), top1_hit_all, total))
print('Top-3 corridor hit (ALL steps):       {:.4f} ({}/{})'.format(top3_hit_all/max(total,1), top3_hit_all, total))
print('Top-1 corridor hit (CC steps):        {:.4f} ({}/{})'.format(top1_hit_cc/max(total,1), top1_hit_cc, total))
print('Top-3 corridor hit (CC steps):        {:.4f} ({}/{})'.format(top3_hit_cc/max(total,1), top3_hit_cc, total))

offsets_all = np.array(offsets_all)
offsets_cc = np.array(offsets_cc)
print('\nArgmax offset from first feasible (ALL steps):')
print('  mean={:.1f} median={:.1f} std={:.1f}'.format(offsets_all.mean(), np.median(offsets_all), offsets_all.std()))
print('  min={} max={}'.format(offsets_all.min(), offsets_all.max()))
print('  <0 (before first): {}  =0: {}  >0: {}'.format(
    (offsets_all < 0).sum(), (offsets_all == 0).sum(), (offsets_all > 0).sum()))

if len(offsets_cc) > 0:
    offsets_cc = np.array(offsets_cc)
    print('\nArgmax offset from first feasible (CC steps):')
    print('  mean={:.1f} median={:.1f} std={:.1f}'.format(offsets_cc.mean(), np.median(offsets_cc), offsets_cc.std()))

n_outside_above_max_inside = np.array(n_outside_above_max_inside)
print('\nOutside steps with score > best inside score:')
print('  mean={:.1f} median={:.1f} max={}'.format(
    n_outside_above_max_inside.mean(), np.median(n_outside_above_max_inside),
    n_outside_above_max_inside.max()))
print('  =0: {}  <=5: {}  <=10: {}  >10: {}'.format(
    (n_outside_above_max_inside == 0).sum(),
    (n_outside_above_max_inside <= 5).sum(),
    (n_outside_above_max_inside <= 10).sum(),
    (n_outside_above_max_inside > 10).sum()))

# ── 4. Argmax autopsy ──
print('\n' + '='*60)
print('4. ARGMAX AUTOPSY (ALL steps, per episode)')
print('='*60)

categories = {'before_first': 0, 'in_corridor': 0, 'after_corridor_before_last': 0,
              'after_last': 0, 'no_feasible': 0}
offset_buckets = {'< -20': 0, '-20 to -6': 0, '-5 to -1': 0, '0 (hit)': 0,
                  '1 to 5': 0, '6 to 10': 0, '11 to 20': 0, '21 to 50': 0, '> 50': 0}

for eid in opp_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']
    feasible = np.where(m['k10_s'][:max_t])[0]
    if len(feasible) == 0:
        categories['no_feasible'] += 1
        continue
    first = feasible[0]; last = feasible[-1]
    corridor_end = min(first + K10, max_t)
    argmax = int(np.argmax(scores[:max_t]))

    if argmax < first:
        categories['before_first'] += 1
    elif argmax < corridor_end:
        categories['in_corridor'] += 1
    elif argmax <= last:
        categories['after_corridor_before_last'] += 1
    else:
        categories['after_last'] += 1

    offset = argmax - first
    if offset < -20: offset_buckets['< -20'] += 1
    elif offset < -5: offset_buckets['-20 to -6'] += 1
    elif offset < 0: offset_buckets['-5 to -1'] += 1
    elif offset == 0: offset_buckets['0 (hit)'] += 1
    elif offset <= 5: offset_buckets['1 to 5'] += 1
    elif offset <= 10: offset_buckets['6 to 10'] += 1
    elif offset <= 20: offset_buckets['11 to 20'] += 1
    elif offset <= 50: offset_buckets['21 to 50'] += 1
    else: offset_buckets['> 50'] += 1

total_cat = sum(categories.values())
for cat, cnt in sorted(categories.items()):
    print('  {}: {} ({:.1f}%)'.format(cat, cnt, 100*cnt/max(total_cat,1)))
print('Offset buckets:')
for bucket, cnt in offset_buckets.items():
    print('  {}: {} ({:.1f}%)'.format(bucket, cnt, 100*cnt/max(total_cat,1)))

# ── 5. Bug diagnosis: what were the 158/-6274 numbers? ──
print('\n' + '='*60)
print('5. BUG DIAGNOSIS: Original 158/-6274 numbers')
print('='*60)

# The original script computed:
# inside: float(sc[first:corridor_end].mean()) for each episode
# outside: float(sc[outside_feasible].mean()) for each episode
# where outside_feasible = steps in [0,max_t) with k10_s=false

# Then: np.mean(inside_scores) and np.mean(outside_scores)

# Let me replicate and then also compute correctly
bug_inside = []; bug_outside = []
correct_inside_per_step = []; correct_outside_per_step = []

for eid in opp_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']
    feasible = np.where(m['k10_s'][:max_t])[0]
    if len(feasible) == 0: continue
    first = feasible[0]; corridor_end = min(first + K10, max_t)

    # Bug: inside = mean over CORRIDOR only (not all feasible steps!)
    bug_in = float(scores[first:corridor_end].mean())
    bug_inside.append(bug_in)

    # Bug: outside = mean over steps where k10_s=false (ALL infeasible, not just known)
    outside_feasible = np.array([t for t in range(max_t) if not m['k10_s'][t]])
    if len(outside_feasible) > 0:
        bug_out = float(scores[outside_feasible].mean())
        bug_outside.append(bug_out)

    # Correct: all inside steps (all feasible), all outside steps (known infeasible)
    if m['k10_s'][:max_t].any():
        correct_inside_per_step.extend(scores[:max_t][m['k10_s'][:max_t]].tolist())
    outside_known = m['k10_k'][:max_t] & (~m['k10_s'][:max_t])
    if outside_known.any():
        correct_outside_per_step.extend(scores[:max_t][outside_known].tolist())

bug_inside = np.array(bug_inside); bug_outside = np.array(bug_outside)
correct_inside_per_step = np.array(correct_inside_per_step)
correct_outside_per_step = np.array(correct_outside_per_step)

print('BUG replication (corridor-mean per episode, then mean of means):')
print('  inside  = {:.4f} (n_eps={})'.format(bug_inside.mean(), len(bug_inside)))
print('  outside = {:.4f} (n_eps={})'.format(bug_outside.mean(), len(bug_outside)))
print()
print('BUG: outside is computed over k10_s=false steps, which includes:')
print('  - known infeasible steps (k10_k=true, k10_s=false)')
print('  - unknown steps (k10_k=false) — which are masked out during training!')
print('  - last (K10-1) steps excluded from max_t — these have k10_s=false by construction')
print()
print('CORRECT per-step distribution:')
print('  inside (feasible):  n={}  mean={:.4f}  median={:.4f}'.format(
    len(correct_inside_per_step), correct_inside_per_step.mean(), np.median(correct_inside_per_step)))
print('  outside (known infeasible): n={}  mean={:.4f}  median={:.4f}'.format(
    len(correct_outside_per_step), correct_outside_per_step.mean(), np.median(correct_outside_per_step)))
print('  ratio inside/outside mean: {:.4f}'.format(
    correct_inside_per_step.mean() / max(abs(correct_outside_per_step.mean()), 1e-8)))

# ── 6. Per-suite breakdown ──
print('\n' + '='*60)
print('6. PER-SUITE LOCALIZATION BREAKDOWN')
print('='*60)

suite_data = defaultdict(lambda: {'top1_hit': 0, 'top3_hit': 0, 'total': 0,
                                    'max_in': [], 'max_out': [], 'offsets': []})
for eid in opp_eids:
    m = meta[eid]; p = preds[eid]
    scores = np.array(p['step_scores_raw_logit'])
    max_t = m['max_t']
    feasible = np.where(m['k10_s'][:max_t])[0]
    if len(feasible) == 0: continue
    first = feasible[0]; corridor_end = min(first + K10, max_t)
    suite = m['suite']
    suite_data[suite]['total'] += 1
    argmax = int(np.argmax(scores[:max_t]))
    if first <= argmax < corridor_end: suite_data[suite]['top1_hit'] += 1
    top3 = np.argsort(scores[:max_t])[-3:]
    if any(first <= t < corridor_end for t in top3): suite_data[suite]['top3_hit'] += 1
    suite_data[suite]['offsets'].append(argmax - first)
    if m['k10_s'][:max_t].any():
        suite_data[suite]['max_in'].append(float(scores[:max_t][m['k10_s'][:max_t]].max()))
    outside_known = m['k10_k'][:max_t] & (~m['k10_s'][:max_t])
    if outside_known.any():
        suite_data[suite]['max_out'].append(float(scores[:max_t][outside_known].max()))

for suite in sorted(suite_data.keys()):
    d = suite_data[suite]
    top1 = d['top1_hit'] / max(d['total'], 1)
    top3 = d['top3_hit'] / max(d['total'], 1)
    off = np.array(d['offsets'])
    # Only compare episodes that have both max_in and max_out
    n_common = min(len(d['max_in']), len(d['max_out']))
    if n_common > 0:
        max_in_arr = np.array(d['max_in'][:n_common])
        max_out_arr = np.array(d['max_out'][:n_common])
        max_in_gt = (max_in_arr > max_out_arr).mean()
    else:
        max_in_gt = 0.0
    print('{}: n_opp={} top1={:.3f} top3={:.3f} offset_median={:.0f} max_in>max_out={:.3f}'.format(
        suite, d['total'], top1, top3, np.median(off), max_in_gt))

# ── 7. Summary ──
print('\n' + '='*60)
print('AUDIT SUMMARY')
print('='*60)
print('Bug confirmed: original inside/outside numbers (158/-6274) used:')
print('  1. Corridor-only as "inside" (first to first+K10, not all feasible steps)')
print('  2. ALL k10_s=false steps as "outside" (including unknown-mask steps)')
print('  3. Mean-of-per-episode-means (not per-step distribution)')
print('')
print('Corrected localization:')
print('  Top-1 corridor hit (ALL steps): {:.4f}'.format(top1_hit_all/max(total,1)))
print('  Top-3 corridor hit (ALL steps): {:.4f}'.format(top3_hit_all/max(total,1)))
print('  Median argmax offset: {:.0f} steps'.format(np.median(offsets_all)))
print('  Episodes with max_inside > max_outside: {:.4f}'.format(max_in_gt_max_out))
print('')
print('Audit does NOT modify Student, checkpoints, or predictions.')
print('All computations use frozen C4 prediction bundle.')
