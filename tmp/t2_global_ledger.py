#!/usr/bin/env python3
import os, json, csv, math
from collections import defaultdict

BASE = "/mnt/sdc/dty_user/openvla_attack/evidence"
PANELS = {}

# Walk phase7_object for COMPLETE.json
p7 = os.path.join(BASE, "phase7_object")
for sub in sorted(os.listdir(p7)):
    sp = os.path.join(p7, sub)
    if not os.path.isdir(sp):
        continue
    for run_dir in sorted(os.listdir(sp)):
        rp = os.path.join(sp, run_dir)
        if not os.path.isdir(rp):
            continue
        comp = os.path.join(rp, "COMPLETE.json")
        summ = os.path.join(rp, "episode_summary.json")
        if os.path.isfile(comp) and os.path.isfile(summ):
            with open(summ) as f:
                s = json.load(f)
            key_data = {
                "panel": sub, "run_dir": run_dir,
                "task": s.get("task_name", s.get("task", "")),
                "task_idx": s.get("task_idx", ""),
                "state_id": s.get("state_id", ""),
                "condition": s.get("condition", ""),
                "objective_id": s.get("objective_id", ""),
                "arm_lock": s.get("arm_lock", ""),
                "perturbation_seed": s.get("perturbation_seed", ""),
                "eval_seed": s.get("eval_seed", ""),
                "task_success": s.get("task_success", ""),
                "attack_frames": s.get("attack_frames", ""),
                "mlp_triggered": s.get("mlp_triggered", ""),
                "bridge_sha256": (s.get("bridge_sha256", "") or "")[:16],
            }
            PANELS[sub + "/" + run_dir] = key_data

# phase7_table1
pt1 = os.path.join(BASE, "phase7_table1")
for sub in sorted(os.listdir(pt1)):
    sp = os.path.join(pt1, sub)
    if not os.path.isdir(sp):
        continue
    for run_dir in sorted(os.listdir(sp)):
        rp = os.path.join(sp, run_dir)
        if not os.path.isdir(rp):
            continue
        comp = os.path.join(rp, "COMPLETE.json")
        done = os.path.join(rp, ".done")
        summ = os.path.join(rp, "episode_summary.json")
        if (os.path.isfile(comp) or os.path.isfile(done)) and os.path.isfile(summ):
            with open(summ) as f:
                s = json.load(f)
            PANELS["phase7_table1/" + sub + "/" + run_dir] = {
                "panel": "phase7_table1/" + sub, "run_dir": run_dir,
                "condition": s.get("condition", ""),
                "objective_id": s.get("objective_id", ""),
                "task_success": s.get("task_success", ""),
            }

# phase11 tomato
p11 = os.path.join(BASE, "phase11_detector_coverage", "tomato_sweep")
if os.path.isdir(p11):
    for run_dir in sorted(os.listdir(p11)):
        rp = os.path.join(p11, run_dir)
        if not os.path.isdir(rp):
            continue
        comp = os.path.join(rp, "COMPLETE.json")
        summ = os.path.join(rp, "episode_summary.json")
        if os.path.isfile(comp) and os.path.isfile(summ):
            with open(summ) as f:
                s = json.load(f)
            PANELS["tomato_coverage/" + run_dir] = {
                "panel": "tomato_coverage", "run_dir": run_dir,
                "task_success": s.get("task_success", ""),
                "mlp_triggered": s.get("mlp_triggered", ""),
            }

# repeatability_diagnostic_v3
rdv = os.path.join(BASE, "repeatability_diagnostic_v3")
for run_dir in sorted(os.listdir(rdv)):
    rp = os.path.join(rdv, run_dir)
    if not os.path.isdir(rp):
        continue
    comp = os.path.join(rp, "COMPLETE.json")
    summ = os.path.join(rp, "episode_summary.json")
    if os.path.isfile(comp) and os.path.isfile(summ):
        with open(summ) as f:
            s = json.load(f)
        PANELS["repeatability_diag/" + run_dir] = {
            "panel": "repeatability_diag", "run_dir": run_dir,
            "condition": s.get("condition", ""),
            "task_success": s.get("task_success", ""),
        }

# Summary by panel
panel_summary = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0, "no_emit": 0})
for key, r in PANELS.items():
    panel = r.get("panel", "unknown")
    panel_summary[panel]["total"] += 1
    if r.get("task_success") is True:
        panel_summary[panel]["success"] += 1
    elif r.get("task_success") is False:
        panel_summary[panel]["fail"] += 1
    if r.get("mlp_triggered") is False:
        panel_summary[panel]["no_emit"] += 1

print("=== GLOBAL PANEL LEDGER ===")
print("{:40s} {:>5s} {:>5s} {:>5s}".format("panel", "total", "succ", "fail"))
total_all = 0
for panel in sorted(panel_summary.keys()):
    ps = panel_summary[panel]
    total_all += ps["total"]
    print("{:40s} {:>5d} {:>5d} {:>5d}".format(panel, ps["total"], ps["success"], ps["fail"]))
print("{:40s} {:>5d}".format("GRAND TOTAL (COMPLETE.json auditable)", total_all))

# Timing .done count
print()
s7h = os.path.join(BASE, "phase7_object", "supplement_7h")
timing_by_sub = defaultdict(int)
timing_total = 0
for sub in sorted(os.listdir(s7h)):
    sp = os.path.join(s7h, sub)
    if not os.path.isdir(sp):
        continue
    for run_dir in sorted(os.listdir(sp)):
        rp = os.path.join(sp, run_dir)
        if not os.path.isdir(rp):
            continue
        done = os.path.join(rp, ".done")
        if os.path.isfile(done):
            timing_by_sub[sub] += 1
            timing_total += 1

print("=== TIMING (.done only, supplement_7h) ===")
for sub in sorted(timing_by_sub.keys()):
    n = timing_by_sub[sub]
    print("  {:40s}: {:>4d} .done".format(sub, n))
print("  TIMING TOTAL .done: {}".format(timing_total))

# Check OFFLINE_METRICS_PER_RUN.csv for the 183 mystery
print()
csv_path = os.path.join(BASE, "phase7_table1", "OFFLINE_METRICS_PER_RUN.csv")
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print("OFFLINE_METRICS_PER_RUN.csv: {} rows".format(len(rows)))
    conds = defaultdict(int)
    subsets = defaultdict(int)
    for r in rows:
        conds[r.get("condition", "?")] += 1
        subsets[r.get("subset", "?")] += 1
    print("  Conditions:")
    for c, n in sorted(conds.items()):
        print("    {}: {}".format(c, n))
    print("  Subsets:")
    for s, n in sorted(subsets.items()):
        print("    {}: {}".format(s, n))
    # Show first 3 rows
    print("  First 3 rows:")
    for r in rows[:3]:
        print("    cell={} cond={} seed={} succ={} subset={}".format(
            r.get("cell_id","?"), r.get("condition","?"), r.get("seed","?"),
            r.get("task_success","?"), r.get("subset","?")))
