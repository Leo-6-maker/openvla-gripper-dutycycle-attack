#!/usr/bin/env python3
"""Analyze raw inventory: mechanisms, suites, schema availability."""
import json, os
from collections import Counter

with open('/data/liuyu/audit_v3/v2_sc5_raw_inventory.json') as f:
    inv = json.load(f)

suites = Counter()
milestones = Counter()
mechanisms = Counter()
schema_sigs = Counter()
missing_fields_summary = Counter()

PICK_PLACE = ['pick_up', 'place_in', 'place_on', 'put_', 'push_']
DRAWER = ['open_the', 'drawer', 'cabinet']
STOVE = ['turn_on_the_stove', 'put_the_moka']

REQUIRED_13 = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

for r in inv['results']:
    if r.get('success') != True:
        continue

    run_dir = r.get('run_dir', '')
    parts = run_dir.replace('\\', '/').split('/')

    if len(parts) >= 1:
        milestones[parts[0]] += 1

    task = r.get('task_name', '')
    is_pp = any(kw in task for kw in PICK_PLACE)
    is_drawer = any(kw in task for kw in DRAWER)
    is_stove = any(kw in task for kw in STOVE)

    if is_pp and not is_drawer:
        mechanism = 'pick_and_place'
    elif is_drawer:
        mechanism = 'drawer'
    elif is_stove:
        mechanism = 'stove'
    else:
        mechanism = 'other'
    mechanisms[mechanism] += 1

    # Suite
    suite = 'unknown'
    for i, p in enumerate(parts):
        if p == 'runs':
            if i + 1 < len(parts):
                suite = parts[i + 1]
            break
    suites[suite] += 1

    # Schema: which required fields are present
    keys = r.get('schema_keys', [])
    present = [f for f in REQUIRED_13 if f in keys]
    missing = [f for f in REQUIRED_13 if f not in keys]
    sig = (len(present), len(missing))
    schema_sigs[sig] += 1
    for m in missing:
        missing_fields_summary[m] += 1

print("=== Milestones ===")
for m, c in milestones.most_common(30):
    print(f"  {m}: {c}")

print("\n=== Suites ===")
for s, c in suites.most_common():
    print(f"  {s}: {c}")

print("\n=== Mechanism ===")
for m, c in mechanisms.most_common():
    print(f"  {m}: {c}")

print("\n=== Schema completeness (13 required fields) ===")
for sig, cnt in sorted(schema_sigs.items(), key=lambda x: -x[1]):
    print(f"  {sig[0]}/13 present, {sig[1]} missing: {cnt} trajectories")

print("\n=== Missing field details ===")
for f, cnt in missing_fields_summary.most_common():
    print(f"  {f}: {cnt}")

print(f"\nTotal success: {sum(mechanisms.values())}")
print(f"SC5-compatible (pick-and-place): {mechanisms.get('pick_and_place', 0)}")
