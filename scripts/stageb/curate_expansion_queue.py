#!/usr/bin/env python3
"""Manual curated expansion queue — applying user's constraints post-exploration.

Key rules:
- hard_negative: ≥8 parents, ≥4 tasks, salad max 2, relaxed heuristic (opens<=2, qpos<0.02)
- rand_abstain: tomato_sauce 2, non-tomato 2, butter=0
- sentinel: ≤3, counts toward task cap, tomato sentinel max 1
- Per-task total ≤4 (including all categories + sentinel)
- butter = 0 (no reintroduction of known task bias)
- Total ≈ 27 parents
"""
import csv, os, sys
from collections import Counter

# Known labeled windows (from master table) + windows we select = unique
MASTER = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/master_labels_aggregated_d4a3827.csv'
CANDIDATES = '/data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/targeted_expansion_queue_curated_d4a3827.csv'

# Load labeled keys
labeled = set()
with open(MASTER, 'r') as f:
    for r in csv.DictReader(f):
        labeled.add((
            r['task_key'], r['state_id'], r.get('seed','0'),
            int(r['window_start']), int(r['window_end'])
        ))

# Load pool as dict keyed by (task, state, seed, ws, we)
pool_dict = {}
with open(CANDIDATES, 'r') as f:
    for c in csv.DictReader(f):
        if c.get('trace_version','') != 'corrected_stageb_v1_1': continue
        if c.get('source_snapshot_id','') != 'f9840cb1': continue
        key = (c['task_key'], c['state_id'], c.get('seed','0'),
               int(c['window_start']), int(c['window_end']))
        if key not in labeled:
            pool_dict[key] = c

def get_candidates(task, state_id=None, seed=None, opens_max=999, opens_min=-1,
                   qpos_max=999.0, qpos_min=-1.0, stratum=None):
    """Find pool candidates matching criteria, sorted by score.
    Excludes already-used windows (in used_pool)."""
    results = []
    for key, c in pool_dict.items():
        if key in used_pool:  # <-- FIX: skip already-picked windows
            continue
        tk, sid, sd, ws, we = key
        if tk != task: continue
        if state_id is not None and sid != state_id: continue
        if seed is not None and sd != str(seed): continue
        oc = int(c.get('clean_open_count', 0))
        qp = float(c.get('qpos_abs_sum_pre', 0))
        st = c.get('candidate_stratum', '')
        if oc > opens_max or oc < opens_min: continue
        if qp > qpos_max or qp < qpos_min: continue
        if stratum and st != stratum: continue
        score = 0.0
        if oc == 0: score += 4
        elif oc == 1: score += 2
        score += max(0, 3 - qp * 100)  # lower qpos = higher score
        if st == 'high_opportunity': score += 1
        results.append((score, key, c))
    results.sort(key=lambda x: -x[0])
    return results

# ── Manually curated window picks ──
# Format: (category, task, state_id, seed, window_start, window_end, reason)
curated = []

# Select helpers: picks the top-scoring candidate for (task, state, seed) or nearest
def pick_best(task, opens_max, qpos_max_val, state_id=None, seed=None, n=1, stratum=None):
    cands = get_candidates(task, state_id=state_id, seed=seed,
                           opens_max=opens_max, qpos_max=qpos_max_val, stratum=stratum)
    result = []
    for score, key, c in cands[:n]:
        if key not in used_pool:
            result.append((key, c, score))
    return result

used_pool = set()

def use(key):
    used_pool.add(key)

# ============ cmd_expansion (8) ============
# alphabet_soup: 1 (reduced from 2 — sentinel + rand + hn fill task cap to 4)
cands = get_candidates('alphabet_soup', opens_max=0, qpos_max=999, stratum='high_opportunity')
for s, k, c in cands:
    if k[1] == '1':  # s1 seed=1
        curated.append(('cmd_expansion', c, 'priority non-butter cmd: closed gripper, high opp'))
        use(k); break

# bbq_sauce: 2
cands = get_candidates('bbq_sauce', opens_max=0, qpos_max=999)
for s, k, c in cands:
    if k[1] == '1':  # s1 seed=1
        curated.append(('cmd_expansion', c, 'non-butter cmd: bbq_sauce s1'))
        use(k); break
for s, k, c in cands:
    if k[1] == '2' and k[3] >= 200:  # s2 seed=2, later window
        curated.append(('cmd_expansion', c, 'non-butter cmd: bbq_sauce s2 late window'))
        use(k); break

# cream_cheese: 1
cands = get_candidates('cream_cheese', opens_max=0, qpos_max=999, stratum='high_opportunity')
for s, k, c in cands:
    curated.append(('cmd_expansion', c, 'non-butter cmd: cream_cheese'))
    use(k); break

# salad_dressing: 2 (add 1 more — underrepresented task, has room in cap)
cands = get_candidates('salad_dressing', opens_max=2, qpos_max=999)
n_picked = 0
for s, k, c in cands:
    if n_picked >= 2: break
    if n_picked == 0 and k[1] != '2':  # first pick: non-s2 for diversity
        curated.append(('cmd_expansion', c, 'non-butter cmd: salad_dressing underrepresented'))
        use(k); n_picked += 1
    elif n_picked == 1 and k[1] == '2':  # second pick: s2 for diversity
        curated.append(('cmd_expansion', c, 'non-butter cmd: salad_dressing s2 contrast'))
        use(k); n_picked += 1

# orange_juice: 1 (new)
cands = get_candidates('orange_juice', opens_max=0, qpos_max=999)
for s, k, c in cands:
    curated.append(('cmd_expansion', c, 'non-butter cmd: orange_juice underrepresented'))
    use(k); break

# milk: 1 (new)
cands = get_candidates('milk', opens_max=2, qpos_max=999)
for s, k, c in cands:
    curated.append(('cmd_expansion', c, 'non-butter cmd: milk underrepresented'))
    use(k); break

# ============ phys_enrichment (6) ============
# cream_cheese: 2
cands = get_candidates('cream_cheese', opens_min=-1, opens_max=999, qpos_min=0.01, qpos_max=0.05)
for s, k, c in cands[:2]:
    curated.append(('phys_enrichment', c, 'phys bridge: cream_cheese interaction zone'))
    use(k)

# orange_juice: 1 (reduced from 2 to stay within task cap)
cands = get_candidates('orange_juice', opens_min=-1, opens_max=999, qpos_min=0.01, qpos_max=0.05)
for s, k, c in cands[:1]:
    curated.append(('phys_enrichment', c, 'phys bridge: orange_juice interaction zone'))
    use(k)

# bbq_sauce: 1
cands = get_candidates('bbq_sauce', opens_min=-1, opens_max=999, qpos_min=0.005, qpos_max=0.05)
for s, k, c in cands:
    curated.append(('phys_enrichment', c, 'phys bridge: bbq_sauce underrepresented'))
    use(k); break

# tomato_sauce: 1
cands = get_candidates('tomato_sauce', opens_min=-1, opens_max=999, qpos_min=0.005, qpos_max=0.05)
for s, k, c in cands:
    if k[3] <= 200:  # earlier window
        curated.append(('phys_enrichment', c, 'phys bridge: tomato_sauce under cap'))
        use(k); break

# ============ hard_negative (8) ============
hn_tasks = [
    ('alphabet_soup', 1), ('bbq_sauce', 1), ('cream_cheese', 1),
    ('orange_juice', 1), ('milk', 1), ('tomato_sauce', 1), ('salad_dressing', 2)
]
for task, n_needed in hn_tasks:
    cands = get_candidates(task, opens_max=2, qpos_max=0.02)
    n_picked = 0
    for s, k, c in cands:
        if n_picked >= n_needed: break
        curated.append(('hard_negative', c, 'relaxed HN: %s opens<=2 low qpos' % task))
        use(k); n_picked += 1

# ============ rand_abstain (4) ============
# tomato_sauce: 1 (reduced from 2 to stay within task cap ≤4)
cands = get_candidates('tomato_sauce', opens_min=1, opens_max=4, qpos_min=-1, qpos_max=999,
                       stratum='medium_opportunity')
for s, k, c in cands[:1]:
    curated.append(('rand_abstain', c, 'rand abstain: tomato_sauce'))
    use(k)
# alphabet_soup: 1
cands = get_candidates('alphabet_soup', opens_min=1, opens_max=4, qpos_min=0.005, qpos_max=999)
for s, k, c in cands:
    curated.append(('rand_abstain', c, 'rand abstain: alphabet_soup exploratory'))
    use(k); break

# orange_juice: 1
cands = get_candidates('orange_juice', opens_min=1, opens_max=4, qpos_min=0.005, qpos_max=999)
for s, k, c in cands:
    curated.append(('rand_abstain', c, 'rand abstain: orange_juice exploratory'))
    use(k); break

# ============ sentinel_repeat (3) ============
# Re-using already-labeled stable windows for health check.
# These are NOT in pool_dict (they're labeled), so we add them from the master table.
sentinel_keys = [
    ('milk', '0', '0', 70, 80),            # stable cmd+phys
    ('tomato_sauce', '2', '2', 95, 105),   # stable cmd+phys (only 1 tomato)
    ('alphabet_soup', '0', '0', 60, 70),   # stable cmd+phys
]
# Load these from original candidates (before exclusion)
all_cand_copy = {}
with open(CANDIDATES, 'r') as f:
    for c in csv.DictReader(f):
        key = (c['task_key'], c['state_id'], c.get('seed','0'),
               int(c['window_start']), int(c['window_end']))
        all_cand_copy[key] = c
for tk, sid, sd, ws, we in sentinel_keys:
    key = (tk, sid, sd, ws, we)
    if key in all_cand_copy:
        curated.append(('sentinel_repeat', all_cand_copy[key],
                        'sentinel: %s s%s stable cmd+phys' % (tk, sid)))

# ============ Audit ============
print('Total curated: %d parents (%d GPU jobs)' % (len(curated), len(curated)*2))

cat_counts = Counter(c[0] for c in curated)
print('\nBy category:')
for cat in ['cmd_expansion', 'phys_enrichment', 'hard_negative', 'rand_abstain', 'sentinel_repeat']:
    print('  %s: %d' % (cat, cat_counts.get(cat, 0)))

task_counts = Counter(c[1]['task_key'] for c in curated)
print('\nBy task (including sentinel):')
for tk in sorted(task_counts):
    cats = Counter(c[0] for c in curated if c[1]['task_key'] == tk)
    cat_str = ' '.join('%s=%d' % (cn[:4], n) for cn, n in cats.items())
    print('  %-20s %d  (%s)' % (tk, task_counts[tk], cat_str))

print('\nButter: %d' % task_counts.get('butter', 0))
print('Tasks >4: %s' % [tk for tk, n in task_counts.items() if n > 4])

# ============ Write CSV ============
fieldnames = [
    'category', 'pair_id', 'task_key', 'state_id', 'seed',
    'window_start', 'window_end', 'n_window_steps',
    'actual_max_step', 'candidate_stratum',
    'clean_open_count', 'clean_open_frac',
    'raw_gripper_mean', 'raw_gripper_max',
    'qpos_pre', 'qpos_mean', 'qpos_max', 'qpos_slope',
    'eef_disp', 'selection_reason'
]
rows = []
for cat, c, reason in curated:
    pair_id = '%s_%s_s%s_w%s_%s_seed%s' % (
        cat, c['task_key'], c['state_id'], c['window_start'],
        c['window_end'], c.get('seed', '0'))
    rows.append({
        'category': cat,
        'pair_id': pair_id,
        'task_key': c['task_key'],
        'state_id': c['state_id'],
        'seed': c.get('seed', '0'),
        'window_start': c['window_start'],
        'window_end': c['window_end'],
        'n_window_steps': c.get('n_window_steps', ''),
        'actual_max_step': c.get('actual_max_step', ''),
        'candidate_stratum': c.get('candidate_stratum', ''),
        'clean_open_count': c.get('clean_open_count', ''),
        'clean_open_frac': c.get('clean_open_frac', ''),
        'raw_gripper_mean': c.get('raw_gripper_mean', ''),
        'raw_gripper_max': c.get('raw_gripper_max', ''),
        'qpos_pre': c.get('qpos_abs_sum_pre', ''),
        'qpos_mean': c.get('qpos_abs_sum_window_mean', ''),
        'qpos_max': c.get('qpos_abs_sum_window_max', ''),
        'qpos_slope': c.get('qpos_abs_sum_slope', ''),
        'eef_disp': c.get('eef_displacement', ''),
        'selection_reason': reason,
    })

os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print('\nOutput: %s' % OUT)
print('Rows: %d' % len(rows))
