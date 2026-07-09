#!/usr/bin/env python3
"""NPZ-level label audit with standardized denominators."""
import json, sys, numpy as np
from collections import defaultdict

npz_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_v1.1_emb_stats/c2f_w16_stats_dataset.npz"
print(f"Loading {npz_path} ...")
data = dict(np.load(npz_path, allow_pickle=True))

n_total = len(data["y_primary"])
print(f"Total windows: {n_total}")

# --- Denominator 1: all windows ---
primary_all = int(data["y_primary"].sum())
hazard_all = int(data["y_hazard"].sum())
release_all = int(data["y_release"].sum())

print(f"\n=== Global window-level rates ===")
print(f"primary_all_windows: {primary_all}/{n_total} = {primary_all/n_total*100:.2f}%")
print(f"hazard_all_windows: {hazard_all}/{n_total} = {hazard_all/n_total*100:.2f}%")
print(f"release_all_windows: {release_all}/{n_total} = {release_all/n_total*100:.2f}%")

# --- Per-suite ---
suites = data["suite"]
task_indices = data["task_index"]
splits = data["split"]
y_primary = data["y_primary"]
y_hazard = data["y_hazard"]

suite_set = sorted(set(str(s) for s in suites))
print(f"\n=== Per-suite window rates ===")
print(f"{'Suite':<20} {'windows':>8} {'primary':>8} {'prim%':>8} {'hazard':>8} {'haz%':>8}")
for s in suite_set:
    mask = np.array([str(x) == s for x in suites])
    n = int(mask.sum())
    p = int(y_primary[mask].sum())
    h = int(y_hazard[mask].sum())
    print(f"{s:<20} {n:>8} {p:>8} {p/n*100:>7.2f}% {h:>8} {h/n*100:>7.2f}%")

# --- Per-suite per-task ---
print(f"\n=== Per-suite per-task window rates ===")
print(f"{'Suite':<20} {'Task':>6} {'windows':>8} {'primary':>8} {'prim%':>8}")
# task_indices may be object array from allow_pickle; flatten to int list
task_idx_flat = np.array([int(ti) for ti in task_indices])
for s in suite_set:
    suite_mask = np.array([str(x) == s for x in suites])
    tasks_in_suite = sorted(set(int(t) for t in task_idx_flat[suite_mask]))
    for t in tasks_in_suite:
        mask = suite_mask & (task_idx_flat == t)
        n = int(mask.sum())
        p = int(y_primary[mask].sum())
        if n > 0:
            print(f"{s:<20} {t:>6} {n:>8} {p:>8} {p/n*100:>7.2f}%")

# --- Per-split ---
split_set = sorted(set(str(s) for s in splits))
print(f"\n=== Per-split window rates ===")
print(f"{'Split':<10} {'windows':>8} {'primary':>8} {'prim%':>8}")
for sp in split_set:
    mask = np.array([str(x) == sp for x in splits])
    n = int(mask.sum())
    p = int(y_primary[mask].sum())
    print(f"{sp:<10} {n:>8} {p:>8} {p/n*100:>7.2f}%")

# --- Object task_01 specifically ---
print(f"\n=== Object task_01 detailed ===")
obj_mask = np.array([str(s) == "libero_object" for s in suites])
for t in sorted(set(int(ti) for ti in task_idx_flat[obj_mask])):
    mask = obj_mask & (task_idx_flat == t)
    n = int(mask.sum())
    p = int(y_primary[mask].sum())
    h = int(y_hazard[mask].sum())
    r = int(y_release[mask].sum())
    if n > 0:
        print(f"  task_{t:02d}: windows={n} primary={p} ({p/n*100:.1f}%) hazard={h} ({h/n*100:.1f}%) release={r} ({r/n*100:.1f}%)")

print("\n=== Object suite overall ===")
n_obj = int(obj_mask.sum())
p_obj = int(y_primary[obj_mask].sum())
h_obj = int(y_hazard[obj_mask].sum())
print(f"windows={n_obj} primary={p_obj} ({p_obj/n_obj*100:.1f}%) hazard={h_obj} ({h_obj/n_obj*100:.1f}%)")

print("\ndone")
