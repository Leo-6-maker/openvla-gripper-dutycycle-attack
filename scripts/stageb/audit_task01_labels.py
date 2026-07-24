#!/usr/bin/env python3
"""Audit v1.1 Object labels per-task to verify status report accuracy."""
import json, subprocess, collections

result = subprocess.run(
    ["find", "/mnt/sdc/dty_user/openvla_attack_evidence/c2f/clean2000_v1.1_caveat",
     "-name", "step_records.jsonl"],
    capture_output=True, text=True)
paths = [p for p in result.stdout.strip().split("\n") if p]
print(f"Total step_records files: {len(paths)}")

task_stats = collections.defaultdict(lambda: {"sc": 0, "primary": 0, "episodes": set()})
for p in paths:
    parts = p.split("/")
    task_part = None
    for part in parts:
        if part.startswith("task_"):
            task_part = part
            break
    if not task_part:
        continue
    # Extract episode key
    try:
        idx = parts.index("libero_object")
        ep_parts = parts[idx+1:]
        # find state_xxx
        state_idx = None
        for i, p2 in enumerate(ep_parts):
            if p2.startswith("state_"):
                state_idx = i
                break
        if state_idx is not None:
            ep_key = "/".join(ep_parts[:state_idx+1])
        else:
            ep_key = "/".join(ep_parts)
    except ValueError:
        ep_key = p
    with open(p) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("teacher_phase") == "stable_carry":
                task_stats[task_part]["sc"] += 1
                task_stats[task_part]["episodes"].add(ep_key)
                if r.get("teacher_primary_attackable"):
                    task_stats[task_part]["primary"] += 1

print()
print("Per-task breakdown:")
for tk in sorted(task_stats.keys()):
    s = task_stats[tk]
    rate = s["primary"] / max(s["sc"], 1) * 100
    print("  {}: sc={} primary={} rate={:.1f}% episodes={}".format(
        tk, s["sc"], s["primary"], rate, len(s["episodes"])))

# Sum overall
total_sc = sum(s["sc"] for s in task_stats.values())
total_primary = sum(s["primary"] for s in task_stats.values())
print()
print("Overall Object: sc={} primary={} rate={:.1f}%".format(
    total_sc, total_primary,
    total_primary / max(total_sc, 1) * 100))
