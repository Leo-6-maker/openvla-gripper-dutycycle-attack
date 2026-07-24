#!/usr/bin/env python3
"""Audit canonical SC5 dataset: mechanism classification, task distribution, SC5 validity."""
import csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path

EP_CSV = sys.argv[1] if len(sys.argv) > 1 else 'tables/v2_sc5_canonical_episodes.csv'
ROWS_CSV = sys.argv[2] if len(sys.argv) > 2 else 'tables/v2_sc5_canonical_dataset.csv'

# Load episodes
eps = []
with open(EP_CSV) as f:
    for r in csv.DictReader(f):
        eps.append(r)

print(f"Total episodes: {len(eps)}")
print(f"Train: {len([e for e in eps if e['is_held_out'] != 'True'])}")
print(f"Held-out: {len([e for e in eps if e['is_held_out'] == 'True'])}")

print("\n=== Non-pick_and_place SC5-valid episodes ===")
for e in eps:
    if e['sc5_valid'] == 'True' and e['mechanism'] != 'pick_and_place':
        task = e['task'][:55]
        src = e['source_milestone'][:35]
        print(f"  {e['mechanism']:12s} | {task:55s} | anchor={e['sc5_anchor']:5s} | cor=[{e['corridor_start']},{e['corridor_end']}] | {src}")

print("\n=== 'other' mechanism tasks ===")
other_tasks = Counter()
for e in eps:
    if e['mechanism'] == 'other':
        other_tasks[e['task']] += 1
for task, cnt in other_tasks.most_common():
    print(f"  [{cnt}] {task}")

print("\n=== Task counts by mechanism ===")
mech_tasks = defaultdict(Counter)
for e in eps:
    mech_tasks[e['mechanism']][e['task']] += 1
for mech in ['pick_and_place', 'drawer', 'stove', 'other']:
    items = mech_tasks[mech]
    sc5_count = len([e for e in eps if e['mechanism'] == mech and e['sc5_valid'] == 'True'])
    print(f"\n{mech}: {len(items)} unique tasks, {sum(items.values())} episodes ({sc5_count} SC5-valid)")
    for task, cnt in items.most_common(15):
        ep_sc5 = [e for e in eps if e['task'] == task]
        sc5_n = len([e for e in ep_sc5 if e['sc5_valid'] == 'True'])
        print(f"  [{cnt}] sc5={sc5_n}/{cnt} {task[:65]}")

print("\n=== SC5 validity by mechanism ===")
for mech in ['pick_and_place', 'drawer', 'stove', 'other']:
    total = len([e for e in eps if e['mechanism'] == mech])
    valid = len([e for e in eps if e['mechanism'] == mech and e['sc5_valid'] == 'True'])
    print(f"  {mech}: {valid}/{total} ({100*valid/max(total,1):.0f}%)")

print("\n=== Per-milestone SC5 validity ===")
milestone_stats = defaultdict(lambda: {'total': 0, 'sc5': 0, 'pp': 0, 'pp_sc5': 0})
for e in eps:
    m = e['source_milestone'][:50]
    milestone_stats[m]['total'] += 1
    if e['sc5_valid'] == 'True':
        milestone_stats[m]['sc5'] += 1
    if e['mechanism'] == 'pick_and_place':
        milestone_stats[m]['pp'] += 1
        if e['sc5_valid'] == 'True':
            milestone_stats[m]['pp_sc5'] += 1
for m, s in sorted(milestone_stats.items(), key=lambda x: -x[1]['total']):
    print(f"  {m}: {s['total']} total, {s['sc5']} SC5, {s['pp']} PP, {s['pp_sc5']} PP+SC5")

# Dataset stats
print("\n=== Dataset row stats ===")
with open(ROWS_CSV) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total_rows = len(rows)
phase_counts = Counter(r['teacher_phase'] for r in rows)
corridor_active = len([r for r in rows if r.get('teacher_sc5_corridor_active') == '1'])
sc5_ready = len([r for r in rows if r.get('teacher_sc5_ready') == '1'])

print(f"Total rows: {total_rows}")
print(f"Phase distribution: {dict(phase_counts.most_common())}")
print(f"Corridor-active rows: {corridor_active}")
print(f"SC5-ready rows: {sc5_ready}")

# Held-out breakdown
held_rows = [r for r in rows if r.get('is_held_out') == 'True']
held_phases = Counter(r['teacher_phase'] for r in held_rows)
print(f"\nHeld-out rows: {len(held_rows)}")
print(f"Held-out phases: {dict(held_phases.most_common())}")

# Per-mechanism phase distribution
print("\n=== Phase distribution by mechanism ===")
for mech in ['pick_and_place', 'drawer', 'stove', 'other']:
    mech_rows = [r for r in rows if r.get('mechanism') == mech]
    phases = Counter(r['teacher_phase'] for r in mech_rows)
    print(f"{mech} ({len(mech_rows)} rows): {dict(phases.most_common(5))}")
