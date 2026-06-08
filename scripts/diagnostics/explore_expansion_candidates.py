"""Quick exploration: find hard_neg and rand_abstain candidates for manual curation."""
import csv
from collections import defaultdict

CANDIDATES = '/data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv'
MASTER = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/master_labels_aggregated_d4a3827.csv'

# Load already-labeled keys
labeled_keys = set()
with open(MASTER, 'r', newline='') as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r['state_id'], r.get('seed','0'),
               int(r['window_start']), int(r['window_end']))
        labeled_keys.add(key)

# Load pool
pool = []
with open(CANDIDATES, 'r', newline='') as f:
    pool = [c for c in csv.DictReader(f)
            if c.get('trace_version','') == 'corrected_stageb_v1_1'
            and c.get('source_snapshot_id','') == 'f9840cb1'
            and (c['task_key'], c['state_id'], c.get('seed','0'),
                 int(c['window_start']), int(c['window_end'])) not in labeled_keys]

print(f'Pool: {len(pool)} after excluding {len(labeled_keys)} labeled\n')

# ── Hard negative candidates ──
# Relaxed: clean_open_count <= 2, any stratum (prefer idle), qpos_pre < 0.02
# Tasks to prioritize: alphabet_soup, bbq_sauce, cream_cheese, orange_juice, milk, tomato_sauce
HARD_NEG_TASKS = ['alphabet_soup', 'bbq_sauce', 'cream_cheese', 'orange_juice', 'milk', 'tomato_sauce']

print('=== HARD NEGATIVE CANDIDATES (relaxed) ===')
for tk in HARD_NEG_TASKS:
    candidates = []
    for c in pool:
        if c['task_key'] != tk:
            continue
        open_c = int(c.get('clean_open_count', 0))
        qpos_pre = float(c.get('qpos_abs_sum_pre', 0))
        stratum = c.get('candidate_stratum', '')
        if open_c <= 2 and qpos_pre < 0.02:
            score = 0.0
            if open_c == 0: score += 4
            elif open_c == 1: score += 2
            else: score += 1
            if qpos_pre < 0.005: score += 3
            elif qpos_pre < 0.01: score += 2
            else: score += 1
            if stratum == 'hard_negative_or_idle': score += 2
            # Prefer same-episode diversity (different state_id/seed)
            candidates.append((score, c['state_id'], c.get('seed','0'),
                               c['window_start'], c['window_end'],
                               open_c, round(qpos_pre, 6), stratum))
    candidates.sort(key=lambda x: -x[0])
    print(f'\n{tk} ({len(candidates)} candidates):')
    for s, sid, seed, ws, we, oc, qp, st in candidates[:8]:
        print(f'  score={s} s{sid} seed={seed} [{ws},{we}] opens={oc} qpos_pre={qp} {st}')
    if not candidates:
        print('  NONE')

# ── Rand abstain candidates (non-tomato, non-butter exploratory) ──
print('\n\n=== RAND ABSTAIN EXPLORATORY (non-tomato, non-butter) ===')
RAND_TASKS = ['alphabet_soup', 'bbq_sauce', 'cream_cheese', 'orange_juice', 'milk', 'salad_dressing']
for tk in RAND_TASKS:
    candidates = []
    for c in pool:
        if c['task_key'] != tk:
            continue
        open_c = int(c.get('clean_open_count', 0))
        qpos_pre = float(c.get('qpos_abs_sum_pre', 0))
        stratum = c.get('candidate_stratum', '')
        if 1 <= open_c <= 4 and stratum in ('medium_opportunity', 'high_opportunity'):
            score = 0.0
            if 1 <= open_c <= 2: score += 3
            else: score += 1.5
            if qpos_pre > 0.005: score += 2
            if stratum == 'high_opportunity': score += 1
            candidates.append((score, c['state_id'], c.get('seed','0'),
                               c['window_start'], c['window_end'],
                               open_c, round(qpos_pre, 6), stratum))
    candidates.sort(key=lambda x: -x[0])
    print(f'\n{tk} ({len(candidates)} candidates):')
    for s, sid, seed, ws, we, oc, qp, st in candidates[:5]:
        print(f'  score={s} s{sid} seed={seed} [{ws},{we}] opens={oc} qpos_pre={qp} {st}')
    if not candidates:
        print('  NONE')

# ── Sentinel check ──
print('\n\n=== SENTINEL CANDIDATES (non-tomato stable cmd+phys) ===')
for r in csv.DictReader(open(MASTER, 'r', newline='')):
    if r['task_key'] == 'butter':
        continue
    if r.get('silver_status','') == 'stable_cmd+phys':
        print(f'  {r["task_key"]} s{r["state_id"]} [{r["window_start"]},{r["window_end"]}] '
              f'qpos_pre={r.get("qpos_pre","?")} seed={r.get("seed","?")}')
