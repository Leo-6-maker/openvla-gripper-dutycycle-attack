#!/usr/bin/env python3
"""T2-v2: Recursive global ledger from episode_summary.json files."""
import os, json, hashlib
from collections import defaultdict
from pathlib import Path

BASE = Path("/mnt/sdc/dty_user/openvla_attack/evidence")

# Find ALL episode_summary.json files
all_summaries = list(BASE.rglob("episode_summary.json"))
print("Total episode_summary.json found: {}".format(len(all_summaries)))

# Build scientific keys
records = []
physical_dirs = set()
sci_keys = defaultdict(list)  # key -> list of physical paths

for sp in all_summaries:
    run_dir = sp.parent
    physical_dir = str(run_dir.relative_to(BASE))
    physical_dirs.add(physical_dir)

    has_complete = (run_dir / "COMPLETE.json").exists()
    has_done = (run_dir / ".done").exists()
    has_telemetry = (run_dir / "step_telemetry.csv").exists()
    has_video = (run_dir / "rollout_raw.mp4").exists()

    try:
        with open(sp) as f:
            s = json.load(f)
    except:
        continue

    # Determine panel from path
    rel = str(run_dir.relative_to(BASE))
    parts = rel.split("/")
    if len(parts) >= 3:
        panel = parts[0] + "/" + parts[1]
    elif len(parts) >= 2:
        panel = parts[0] + "/" + parts[1]
    else:
        panel = parts[0]

    # Build scientific key
    ti = s.get("task_idx", "?")
    si = s.get("state_id", "?")
    ps = s.get("perturbation_seed", "?")
    es = s.get("eval_seed", "?")
    cond = s.get("condition", "?")
    obj = s.get("objective_id", "?")
    al = s.get("arm_lock", "?")
    succ = s.get("task_success", "?")
    af = s.get("attack_frames", "?")
    trig = s.get("mlp_triggered", "?")

    # Check for repeat suffix
    is_repeat = False
    dirname = run_dir.name
    if "_r" in dirname:
        parts_r = dirname.split("_r")
        if len(parts_r) > 1:
            try:
                int(parts_r[-1])
                is_repeat = True
            except:
                pass

    sci_key = "|".join([str(ti), str(si), str(ps), str(cond), str(obj), str(al)])
    sci_keys[sci_key].append(physical_dir)

    records.append({
        "physical_dir": physical_dir,
        "panel": panel,
        "sci_key": sci_key,
        "task_idx": ti,
        "state_id": si,
        "perturbation_seed": ps,
        "eval_seed": es,
        "condition": cond,
        "objective_id": obj,
        "arm_lock": al,
        "task_success": succ,
        "attack_frames": af,
        "mlp_triggered": trig,
        "is_repeat": is_repeat,
        "has_COMPLETE": has_complete,
        "has_done": has_done,
        "has_telemetry": has_telemetry,
        "has_video": has_video,
        "bridge_sha": (s.get("bridge_sha256", "") or "")[:16],
    })

# Counts
n_physical = len(physical_dirs)
n_ep_summaries = len(records)
n_unique_sci_keys = len(sci_keys)
n_duplicate_keys = sum(1 for k, v in sci_keys.items() if len(v) > 1)
n_repeats = sum(1 for r in records if r["is_repeat"])

# By panel (top 2 levels)
panel_counts = defaultdict(lambda: {"total": 0, "formal": 0, "repeat": 0,
                                     "succ": 0, "fail": 0, "no_emit": 0,
                                     "with_COMPLETE": 0, "with_done": 0,
                                     "with_telemetry": 0, "with_video": 0})
for r in records:
    # Simplify panel to top 2 dirs
    parts = r["physical_dir"].split("/")
    if len(parts) >= 2:
        panel = parts[0] + "/" + parts[1]
    else:
        panel = parts[0]
    pc = panel_counts[panel]
    pc["total"] += 1
    if r["is_repeat"]:
        pc["repeat"] += 1
    else:
        pc["formal"] += 1
    if r["task_success"] is True: pc["succ"] += 1
    elif r["task_success"] is False: pc["fail"] += 1
    if r["mlp_triggered"] is False: pc["no_emit"] += 1
    if r["has_COMPLETE"]: pc["with_COMPLETE"] += 1
    if r["has_done"]: pc["with_done"] += 1
    if r["has_telemetry"]: pc["with_telemetry"] += 1
    if r["has_video"]: pc["with_video"] += 1

print()
print("=== PHYSICAL vs SCIENTIFIC COUNTS ===")
print("Physical directories with episode_summary.json: {}".format(n_physical))
print("Total episode_summary.json files: {}".format(n_ep_summaries))
print("Unique scientific keys: {}".format(n_unique_sci_keys))
print("Keys with >1 physical copy (duplicates/repeats): {}".format(n_duplicate_keys))
print("Explicit repeat runs (_rN suffix): {}".format(n_repeats))

print()
print("=== BY PANEL (top 2 dirs) ===")
print("{:45s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
    "panel", "total", "formal", "repeat", "succ", "fail", "no_emit"))
grand = 0
for panel in sorted(panel_counts.keys()):
    pc = panel_counts[panel]
    grand += pc["total"]
    print("{:45s} {:>5d} {:>5d} {:>5d} {:>5d} {:>5d} {:>5d}".format(
        panel, pc["total"], pc["formal"], pc["repeat"], pc["succ"], pc["fail"], pc["no_emit"]))
print("{:45s} {:>5d}".format("GRAND TOTAL (ep_summary auditable)", grand))

# Show duplicate keys
print()
print("=== DUPLICATE SCIENTIFIC KEYS ===")
dup_count = 0
for key in sorted(sci_keys.keys()):
    paths = sci_keys[key]
    if len(paths) > 1:
        dup_count += 1
        if dup_count <= 10:
            print("  KEY: {}".format(key))
            for p in paths:
                print("    -> {}".format(p))

# Show panels with different bridge versions
print()
print("=== BRIDGE SHA DISTRIBUTION (by panel) ===")
bridge_by_panel = defaultdict(lambda: defaultdict(int))
for r in records:
    parts = r["physical_dir"].split("/")
    panel = parts[0] + "/" + parts[1] if len(parts) >= 2 else parts[0]
    bridge_by_panel[panel][r["bridge_sha"]] += 1

for panel in sorted(bridge_by_panel.keys()):
    shas = bridge_by_panel[panel]
    if len(shas) > 1:
        print("  {}: MULTIPLE bridge SHAs!".format(panel))
        for sha, cnt in shas.items():
            print("    {}: {}".format(sha, cnt))
    else:
        sha = list(shas.keys())[0]
        cnt = list(shas.values())[0]
        print("  {}: {} ({} runs)".format(panel, sha, cnt))

# Show timing (supplement_7h) details
print()
print("=== TIMING SUPPLEMENT_7H BREAKDOWN ===")
timing_records = [r for r in records if "supplement_7h" in r["physical_dir"]]
print("Timing runs with episode_summary.json: {}".format(len(timing_records)))
for r in timing_records[:5]:
    print("  {}: cond={} obj={} succ={} af={}".format(
        r["physical_dir"], r["condition"], r["objective_id"],
        r["task_success"], r["attack_frames"]))
if len(timing_records) > 5:
    print("  ... and {} more".format(len(timing_records) - 5))
