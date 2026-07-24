#!/usr/bin/env python3
"""Survey schema diversity from census inventory for schema adapter design."""
import csv
from collections import Counter, defaultdict

INVENTORY = '/tmp/sc5_source_census_cc356f3_r1/tables/v2_sc5_episode_inventory.csv'

tier_counter = Counter()
schema_n_key_counter = Counter()
schema_keys_samples = defaultdict(list)
schema_key_detail = Counter()

with open(INVENTORY) as f:
    reader = csv.DictReader(f)
    for row in reader:
        tier = row.get('exclusion_reason', 'UNKNOWN')
        tier_counter[tier] += 1

        schema_n_key = row.get('schema_n_key', '0')
        schema_n_key_counter[schema_n_key] += 1

        # Parse individual schema keys
        keys_str = row.get('schema_keys', '')
        if keys_str:
            keys = [k.strip() for k in keys_str.split(',')]
            for k in keys:
                if k:
                    schema_key_detail[k] += 1

print("=== Tier distribution ===")
for tier, cnt in tier_counter.most_common():
    print(f"  {tier}: {cnt}")

print("\n=== Schema n_key distribution ===")
for nk, cnt in schema_n_key_counter.most_common(20):
    print(f"  n_key={nk}: {cnt}")

print("\n=== Schema keys found across ALL episodes (top 40) ===")
for key, cnt in schema_key_detail.most_common(40):
    print(f"  [{cnt}] {key}")

# Now look at one PRIMARY candidate in detail
print("\n=== PRIMARY candidate detail ===")
count = 0
with open(INVENTORY) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if count >= 3:
            break
        tier = row.get('exclusion_reason', '')
        if tier == 'LIBERO_OBJECT_SINGLE_OBJECT_CANDIDATE':
            print(f"\n  episode: {row['episode_id'][:20]}")
            print(f"  task: {row['task']}")
            print(f"  state_id: {row['state_id']}")
            print(f"  suite: {row['suite']}")
            print(f"  success: {row['success']}")
            print(f"  clean_status: {row['clean_status']}")
            print(f"  schema_status: {row['schema_status']}")
            print(f"  schema_n_key: {row['schema_n_key']}")
            keys = row.get('schema_keys', '')[:500]
            print(f"  schema_keys: {keys}")
            count += 1
