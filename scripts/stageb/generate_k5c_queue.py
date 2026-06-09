#!/usr/bin/env python3
"""Generate K5c targeted expansion queue: 16 parents × K=5 × VIS/RAND = 160 jobs.

Categories:
  A: rand-sensitive expansion (5 parents)
  B: same-task contrast for cmd de-biasing (6 parents)
  C: strict phys enrichment (4 parents)
  D: sentinel health check (1 parent)

All parents use fixed env_seed, attack_seed 0..4.
Output: tables/stageb_v1_1_k5c_queue_rc1a_<commit>.csv
"""

import csv, os, sys, re

# ── Candidate definitions ──
# Format: (parent_key, task, state_id, env_seed, window_start, window_end, category, rationale)
CANDIDATES = [
    # ── A: Rand-sensitive expansion (5) ──
    ("k5c_rand_butter1", "butter", 0, 0, 90, 100,
     "rand_sensitive", "Old RCMD label, butter has 0 windows, cf=0.36"),
    ("k5c_rand_butter2", "butter", 0, 0, 95, 105,
     "rand_sensitive", "Old RCMD+VPHYS label, cf=0.73"),
    ("k5c_rand_cream", "cream_cheese", 2, 2, 65, 75,
     "rand_sensitive", "Heuristic: high cf=0.82 in cream (0 rand in pool)"),
    ("k5c_rand_oj", "orange_juice", 1, 1, 50, 60,
     "rand_sensitive", "Heuristic: high cf=0.73, orange_juice has 0 windows"),
    ("k5c_rand_alpha", "alphabet_soup", 0, 0, 55, 65,
     "rand_sensitive", "Heuristic: mid cf=0.46, alpha only has 1 rand"),

    # ── B: Same-task contrast for cmd (6) ──
    ("k5c_cmd_tomato_early", "tomato_sauce", 0, 0, 50, 60,
     "same_task_contrast", "Old CMD, tomato same-task contrast (tomato has 7 windows)"),
    ("k5c_cmd_milk_neg", "milk", 0, 0, 235, 245,
     "same_task_contrast", "Old NEG, milk same-task negative contrast"),
    ("k5c_cmd_cream", "cream_cheese", 1, 1, 85, 95,
     "same_task_contrast", "Old CMD, cream same-task contrast"),
    ("k5c_cmd_salad_neg", "salad_dressing", 1, 1, 55, 65,
     "same_task_contrast", "Old NEG, salad same-task negative contrast"),
    ("k5c_cmd_alpha", "alphabet_soup", 0, 0, 65, 75,
     "same_task_contrast", "Old CMD+VPHYS, alpha has 0 cmd"),
    ("k5c_cmd_butter", "butter", 0, 0, 80, 90,
     "same_task_contrast", "Old CMD, butter needs cmd"),

    # ── C: Strict phys enrichment (4) ──
    ("k5c_phys_butter", "butter", 0, 0, 135, 145,
     "strict_phys", "Old CMD+VPHYS, NOT abstain, cf=0.73 — best phys candidate"),
    ("k5c_phys_tomato", "tomato_sauce", 2, 2, 85, 95,
     "strict_phys", "Old CMD+VPHYS+ABSTAIN, tomato phys context"),
    ("k5c_phys_salad", "salad_dressing", 2, 2, 75, 85,
     "strict_phys", "Heuristic: salad phys context, salad has 1 phys"),
    ("k5c_phys_bbq", "bbq_sauce", 0, 0, 70, 80,
     "strict_phys", "Heuristic: bbq phys context, bbq has 0 phys"),

    # ── D: Sentinel (1) ──
    ("k5c_sentinel_milk_gold", "milk", 0, 0, 70, 80,
     "sentinel", "Re-run K5 GOLD anchor (pV=1.0, pR=0.0) for runner health check"),
]

# ── Validations ──
KNOWN_TASKS = ['alphabet_soup','bbq_sauce','butter','cream_cheese','milk','orange_juice','salad_dressing','tomato_sauce']
assert len(CANDIDATES) == 16, f"Expected 16 parents, got {len(CANDIDATES)}"

# Check unique parent keys
keys = [c[0] for c in CANDIDATES]
assert len(set(keys)) == len(keys), f"Duplicate parent keys: {len(keys)} vs {len(set(keys))}"

# Check no edge windows (ws>=40, we <= 300)
for c in CANDIDATES:
    pk, task, sid, env_seed, ws, we, cat, rationale = c
    assert ws >= 40, f"{pk}: ws={ws} < 40 (edge)"
    assert we - ws >= 8, f"{pk}: window too narrow ({we-ws})"
    assert we - ws <= 30, f"{pk}: window too wide ({we-ws})"

# Check tasks valid
for c in CANDIDATES:
    assert c[1] in KNOWN_TASKS, f"Unknown task: {c[1]}"

# ── Generate queue rows ──
OUT_DIR = f"/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_{'S6COMMIT'}"
CSV_PATH = f"tables/stageb_v1_1_k5c_queue_rc1a_S6COMMIT.csv"

rows = []
job_id = 520000  # New job_id base, unique from K5(500xxx) and K5b(510xxx)

for parent_idx, c in enumerate(CANDIDATES):
    pk, task, sid, env_seed, ws, we, cat, rationale = c
    pair_id = parent_idx  # 0..15
    for attack_seed in range(5):  # 0..4
        for condition in ['vis_pgd', 'random_linf']:
            rows.append({
                'parent_key': pk,
                'pair_id': pair_id,
                'job_id': job_id,
                'task': task,
                'state_id': sid,
                'env_seed': env_seed,
                'attack_seed': attack_seed,
                'window_start': ws,
                'window_end': we,
                'condition': condition,
                'category': cat,
                'rationale': rationale,
                'output_dir': OUT_DIR,
            })
            job_id += 1

print(f"Total jobs: {len(rows)}")
print(f"Job ID range: {520000}–{job_id-1}")
print(f"Pairs: {len(CANDIDATES)}, K=5 per pair, VIS+RAND per K")

# Per-category counts
from collections import Counter
cat_counts = Counter(r['category'] for r in rows)
for cat, cnt in sorted(cat_counts.items()):
    print(f"  {cat}: {cnt//10} parents, {cnt} jobs")

# Task distribution
task_counts = Counter(r['task'] for r in rows)
print(f"\nPer-task jobs:")
for tk in sorted(task_counts):
    print(f"  {tk}: {task_counts[tk]//10} parents, {task_counts[tk]} jobs")

# ── Write CSV ──
os.makedirs('tables', exist_ok=True)
with open(CSV_PATH, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['parent_key','pair_id','job_id','task','state_id','env_seed','attack_seed',
                'window_start','window_end','condition','category','rationale','output_dir'])
    for r in rows:
        w.writerow([r['parent_key'], r['pair_id'], r['job_id'], r['task'], r['state_id'],
                    r['env_seed'], r['attack_seed'], r['window_start'], r['window_end'],
                    r['condition'], r['category'], r['rationale'], r['output_dir']])

print(f"\nQueue CSV: {CSV_PATH}")
print("Ready for smoke.")
