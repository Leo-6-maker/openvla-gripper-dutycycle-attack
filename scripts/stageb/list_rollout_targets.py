#!/usr/bin/env python3
"""List SC5-valid episodes for rollout targeting."""
import csv, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'tables/v2_sc5_canonical_episodes.csv'
rows = list(csv.DictReader(open(path)))
# Butter states: SC5-valid, not held-out
sc5 = [r for r in rows if r['sc5_valid'] == 'True' and r['is_held_out'] != 'True'
       and 'butter' in r.get('task', '').lower()]
print(f'Butter SC5-valid non-held: {len(sc5)}')
for r in sc5:
    print(f"  s{r['state_id']} anchor={r['sc5_anchor']} split={r['split']}")

# Also non-butter SC5-valid train
other = [r for r in rows if r['sc5_valid'] == 'True' and r['is_held_out'] != 'True'
         and 'butter' not in r.get('task', '').lower()
         and r['split'] == 'train']
import random; random.seed(42); random.shuffle(other)
print(f'\nNon-Butter SC5-valid train: {len(other)}')
for r in other[:5]:
    print(f"  {r['task'][:50]} s{r['state_id']} anchor={r['sc5_anchor']}")
