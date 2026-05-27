#!/usr/bin/env python3
"""Phase 0-2: Collect and compare official vs v4 Object 100 manifests."""
import csv, json, os, sys
from pathlib import Path
from collections import defaultdict

OUT = Path("/data/liuyu/outputs/milestone_r2_official_v4_object_alignment_20260526")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "reports").mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. Per-episode data
# ═══════════════════════════════════════════════════════════

# --- Official eval ---
# W45: tasks 0-4, W26: tasks 5-9
# Task order from LIBERO: [0..9] = alphabet_soup, cream_cheese, salad_dressing, bbq_sauce, ketchup,
#                                          tomato_sauce, butter, milk, chocolate_pudding, orange_juice
LIBERO_OBJECT_ORDER = {
    0: "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    2: "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    4: "pick_up_the_ketchup_and_place_it_in_the_basket",
    5: "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    6: "pick_up_the_butter_and_place_it_in_the_basket",
    7: "pick_up_the_milk_and_place_it_in_the_basket",
    8: "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
}

# Official task SR (from logs):
official_task_sr = {
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": (8, 10),
    "pick_up_the_cream_cheese_and_place_it_in_the_basket": (8, 10),
    "pick_up_the_salad_dressing_and_place_it_in_the_basket": (9, 10),
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": (5, 10),
    "pick_up_the_ketchup_and_place_it_in_the_basket": (10, 10),
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket": (9, 10),
    "pick_up_the_butter_and_place_it_in_the_basket": (8, 10),
    "pick_up_the_milk_and_place_it_in_the_basket": (8, 10),
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket": (7, 10),
    "pick_up_the_orange_juice_and_place_it_in_the_basket": (8, 10),
}

# Official per-episode from the surviving manifest (W26 only — tasks 5-9)
official_manifest_path = "/data/liuyu/outputs/milestone_r1_official_eval_20260526/tables/object_official_script_100_manifest.csv"
official_per_episode = {}  # (task_name, state_id) -> row
try:
    with open(official_manifest_path) as f:
        for row in csv.DictReader(f):
            key = (row["task_name"], int(row["state_id"]))
            official_per_episode[key] = row
    print(f"Official manifest: {len(official_per_episode)} episodes from CSV")
except Exception as e:
    print(f"Official manifest read error: {e}")

# --- v4 runner ---
# Collect all summary.csv files from object_full_10x10
v4_full10x10 = Path("/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10")
v4_per_episode = {}
for summary_file in sorted(v4_full10x10.glob("obj_*/summary.csv")):
    run_dir = summary_file.parent.name
    try:
        with open(summary_file) as f:
            row = next(csv.DictReader(f))
    except Exception:
        continue
    # Parse run_dir name: obj_<task>_s<state>
    # e.g. obj_cream_cheese_s0, obj_bbq_sauce_s1
    parts = run_dir.rsplit("_s", 1)
    if len(parts) != 2:
        continue
    task_key = parts[0].replace("obj_", "").replace("_", " ")
    state_id = int(parts[1])
    # Map short task name to full
    task_name = None
    for tname in LIBERO_OBJECT_ORDER.values():
        if task_key in tname.replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", "").replace("_", " "):
            task_name = tname
            break
    if task_name is None:
        task_name = f"unknown_{task_key}"
    fr = float(row.get("FR_attack", 1.0))
    success = fr < 0.5  # FR_attack=0.0 means success, 1.0 means failure
    v4_per_episode[(task_name, state_id)] = {
        "run_id": row.get("run_id", ""),
        "success": success,
        "failure_phase": "no_grasp" if fr > 0.5 else "success_libero",
        "seed": int(row.get("seed", 0)),
    }

print(f"v4 runner: {len(v4_per_episode)} episodes collected")

# Compute v4 task-level SR
v4_task_sr = defaultdict(lambda: [0, 0])
for (task_name, state_id), ep in v4_per_episode.items():
    v4_task_sr[task_name][0] += 1 if ep["success"] else 0
    v4_task_sr[task_name][1] += 1

# ═══════════════════════════════════════════════════════════
# 2. Build per-episode manifest
# ═══════════════════════════════════════════════════════════

def build_manifest():
    """Reconstruct best-effort per-episode manifest combining log + CSV data."""
    rows = []
    for task_id in range(10):
        task_name = LIBERO_OBJECT_ORDER[task_id]
        off_success, off_total = official_task_sr[task_name]
        v4_success, v4_total = v4_task_sr.get(task_name, (0, 0))
        for state_id in range(10):
            off_ep = official_per_episode.get((task_name, state_id))
            v4_ep = v4_per_episode.get((task_name, state_id))

            row = {
                "task_name": task_name,
                "task_id": task_id,
                "state_id": state_id,
                "official_success": "",
                "official_failure_phase": "",
                "official_source_confidence": "low",
                "v4_success": "",
                "v4_failure_phase": "",
                "v4_source_confidence": "low",
                "match": "",
            }

            if off_ep:
                row["official_success"] = str(off_ep.get("success", "")).lower() == "true"
                row["official_failure_phase"] = off_ep.get("failure_phase", "")
                row["official_source_confidence"] = "high"

            if v4_ep:
                row["v4_success"] = v4_ep["success"]
                row["v4_failure_phase"] = v4_ep.get("failure_phase", "")

            # Determine match
            off_s = row["official_success"]
            v4_s = row["v4_success"]
            if off_s == "" and v4_s == "":
                row["match"] = "unknown"
            elif off_s == "":
                row["match"] = "v4_only_known"
            elif v4_s == "":
                row["match"] = "official_only_known"
            elif off_s == v4_s:
                row["match"] = "matched"
            else:
                row["match"] = "mismatch"

            rows.append(row)

    return rows

manifest_rows = build_manifest()

# Write reconstructed manifests
off_fields = ["task_name", "task_id", "state_id", "runner_type", "success",
              "failure_phase", "num_steps", "runtime_status", "worker_id", "gpu_group",
              "python_executable", "checkpoint", "unnorm_key", "prompt_format",
              "seed", "center_crop", "attention_backend", "source_confidence", "notes"]
off_rows = []
for task_id in range(10):
    task_name = LIBERO_OBJECT_ORDER[task_id]
    for state_id in range(10):
        ep = official_per_episode.get((task_name, state_id))
        off_rows.append({
            "task_name": task_name, "task_id": task_id, "state_id": state_id,
            "runner_type": "official_corrected",
            "success": ep.get("success", "") if ep else "",
            "failure_phase": ep.get("failure_phase", "") if ep else "",
            "num_steps": ep.get("num_steps", "") if ep else "",
            "runtime_status": "",
            "worker_id": "W45" if task_id < 5 else "W26",
            "gpu_group": "4,5" if task_id < 5 else "2,6",
            "python_executable": "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python",
            "checkpoint": "openvla-7b-finetuned-libero-object",
            "unnorm_key": "libero_object",
            "prompt_format": "In: What action should the robot take to {task}?\\nOut:",
            "seed": "0",
            "center_crop": "True",
            "attention_backend": "eager",
            "source_confidence": "high" if ep else "low",
            "notes": "from W26 CSV" if ep else "W45 CSV overwritten, inferred from task SR",
        })

with open(OUT / "tables" / "object_official_corrected_100_manifest_reconstructed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=off_fields)
    w.writeheader()
    w.writerows(off_rows)

v4_fields = ["task_name", "task_id", "state_id", "runner_type", "success",
             "failure_phase", "seed", "run_id",
             "python_executable", "checkpoint", "unnorm_key",
             "seed_val", "center_crop", "attention_backend", "source_confidence", "notes"]
v4_rows_out = []
for (task_name, state_id), ep in sorted(v4_per_episode.items()):
    task_id = [k for k, v in LIBERO_OBJECT_ORDER.items() if v == task_name][0]
    v4_rows_out.append({
        "task_name": task_name, "task_id": task_id, "state_id": state_id,
        "runner_type": "v4_runner",
        "success": ep["success"],
        "failure_phase": ep.get("failure_phase", ""),
        "seed": ep.get("seed", ""),
        "run_id": ep.get("run_id", ""),
        "python_executable": "conda openvla_official_libero_20260525 (v4 runner)",
        "checkpoint": "openvla-7b-finetuned-libero-object",
        "unnorm_key": "libero_object",
        "seed_val": "varied (v4 runner state-based seeding)",
        "center_crop": "True",
        "attention_backend": "eager",
        "source_confidence": "high",
        "notes": "from object_full_10x10 summary.csv",
    })

with open(OUT / "tables" / "object_v4_100_manifest_reconstructed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=v4_fields)
    w.writeheader()
    w.writerows(v4_rows_out)

# ═══════════════════════════════════════════════════════════
# 3. Diff table
# ═══════════════════════════════════════════════════════════

diff_fields = ["task_name", "task_id", "state_id", "official_success", "v4_success",
               "official_failure_phase", "v4_failure_phase", "diff_type",
               "official_source", "v4_source"]
diff_rows = []
for row in manifest_rows:
    off = row["official_success"]
    v4 = row["v4_success"]
    if off == "" or v4 == "":
        diff_type = "unknown"
    elif off and v4:
        diff_type = "both_success"
    elif not off and not v4:
        diff_type = "both_failure"
    elif off and not v4:
        diff_type = "official_success_v4_failure"
    else:
        diff_type = "v4_success_official_failure"
    diff_rows.append({
        "task_name": row["task_name"], "task_id": row["task_id"],
        "state_id": row["state_id"],
        "official_success": off, "v4_success": v4,
        "official_failure_phase": row["official_failure_phase"],
        "v4_failure_phase": row["v4_failure_phase"],
        "diff_type": diff_type,
        "official_source": row["official_source_confidence"],
        "v4_source": row["v4_source_confidence"],
    })

with open(OUT / "tables" / "object_official_vs_v4_per_episode_diff.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=diff_fields)
    w.writeheader()
    w.writerows(diff_rows)

# ═══════════════════════════════════════════════════════════
# 4. Mismatch summary
# ═══════════════════════════════════════════════════════════

mismatch_fields = ["task_name", "official_successes", "official_total", "official_sr",
                   "v4_successes", "v4_total", "v4_sr", "gap", "official_only_success",
                   "v4_only_success", "matched_success", "known_episodes", "priority"]
mismatch_rows = []
total_off_only = 0
total_v4_only = 0
total_matched_success = 0
total_matched_failure = 0
total_known = 0

for task_id in range(10):
    task_name = LIBERO_OBJECT_ORDER[task_id]
    off_s, off_t = official_task_sr[task_name]
    v4_s, v4_t = v4_task_sr.get(task_name, (0, 10))
    gap = off_s - v4_s

    off_only = 0
    v4_only = 0
    matched_s = 0
    matched_f = 0
    known = 0

    for state_id in range(10):
        off_ep = official_per_episode.get((task_name, state_id))
        v4_ep = v4_per_episode.get((task_name, state_id))
        if off_ep and v4_ep:
            known += 1
            total_known += 1
            off_ok = str(off_ep.get("success", "")).lower() == "true"
            v4_ok = v4_ep["success"]
            if off_ok and v4_ok:
                matched_s += 1
                total_matched_success += 1
            elif not off_ok and not v4_ok:
                matched_f += 1
                total_matched_failure += 1
            elif off_ok and not v4_ok:
                off_only += 1
                total_off_only += 1
            else:
                v4_only += 1
                total_v4_only += 1

    mismatch_rows.append({
        "task_name": task_name,
        "official_successes": off_s, "official_total": off_t,
        "official_sr": f"{off_s/off_t:.2f}",
        "v4_successes": v4_s, "v4_total": v4_t,
        "v4_sr": f"{v4_s/v4_t:.2f}",
        "gap": f"+{gap}" if gap > 0 else str(gap),
        "official_only_success": off_only,
        "v4_only_success": v4_only,
        "matched_success": matched_s,
        "known_episodes": known,
        "priority": "HIGH" if gap >= 2 else "MEDIUM" if gap >= 1 else "LOW",
    })

with open(OUT / "tables" / "object_official_vs_v4_task_gap_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=mismatch_fields)
    w.writeheader()
    w.writerows(mismatch_rows)

# ═══════════════════════════════════════════════════════════
# 5. Summary report
# ═══════════════════════════════════════════════════════════

lines = [
    "# Official vs v4 Object Gap Summary",
    "",
    "## Overall",
    "",
    f"- Official corrected: 80/100 = 0.800",
    f"- v4 runner: 71/100 = 0.710",
    f"- Gap: +9 (official higher)",
    "",
    "## Task-Level Gap",
    "",
    "| Task | Official | v4 | Gap |",
    "|------|----------|----|-----|",
]
for row in mismatch_rows:
    lines.append(f"| {row['task_name']} | {row['official_sr']} | {row['v4_sr']} | {row['gap']} |")

lines.extend([
    "",
    "## Per-Episode Analysis (W26 tasks only — 50 episodes with complete data)",
    "",
    f"- Known pairs (both manifests): {total_known}",
    f"- Official-only successes: {total_off_only}",
    f"- v4-only successes: {total_v4_only}",
    f"- Matched successes: {total_matched_success}",
    f"- Matched failures: {total_matched_failure}",
    "",
    "**Note:** W45 tasks (alphabet_soup, cream_cheese, salad_dressing, bbq_sauce, ketchup) have official per-episode data missing due to CSV overwrite.",
    "Per-episode mismatch comes from ONLY W26's 50 episodes.",
    "Task-level comparison is complete for all 100 episodes.",
    "",
    "## Key Gap Contributors",
    "",
    "The +9 official-over-v4 gap comes primarily from:",
])

for row in sorted(mismatch_rows, key=lambda r: -int(r['gap'].lstrip('+').lstrip('-') or 0)):
    gap_val = int(row['gap'].lstrip('+').lstrip('-') or 0)
    if gap_val > 0:
        lines.append(f"- **{row['task_name']}**: official {row['official_sr']} vs v4 {row['v4_sr']} (gap={row['gap']})")

lines.extend([
    "",
    "## Preflight Package Versions",
    "",
    "```",
    "python=3.10.13",
    "mujoco=3.8.1",
    "robosuite=1.4.0",
    "numpy=1.26.4",
    "torch=2.2.0+cu121",
    "transformers=4.40.1",
    "tokenizers=0.19.1",
    "PIL=12.2.0",
    "```",
    "",
    "## Preflight Configuration",
    "",
    "- prompt_format: `In: What action should the robot take to {task}?\\nOut:` (CORRECT)",
    "- checkpoint: openvla-7b-finetuned-libero-object",
    "- unnorm_key: libero_object",
    "- seed: 0 (official), varied (v4)",
    "- center_crop: True (both)",
    "- attention: eager (both)",
    "- GPU: 4,5 / 2,6 (official); varied (v4)",
    "- GPU0/GPU7/GPU37: EXCLUDED from all runs",
])

with open(OUT / "reports" / "OFFICIAL_V4_GAP_SUMMARY.md", "w") as f:
    f.write("\n".join(lines))

# ═══════════════════════════════════════════════════════════
# 6. Print summary
# ═══════════════════════════════════════════════════════════
print("\n=== TASK GAP SUMMARY ===")
for row in sorted(mismatch_rows, key=lambda r: -int(r['gap'].lstrip('+').lstrip('-') or 0)):
    print(f"  {row['task_name']:60s} off={row['official_sr']} v4={row['v4_sr']} gap={row['gap']}")
print(f"\nTotal known per-episode pairs: {total_known}/100")
print(f"Official-only successes: {total_off_only}")
print(f"v4-only successes: {total_v4_only}")
print(f"Matched successes: {total_matched_success}")
print(f"Matched failures: {total_matched_failure}")
print(f"\nAll outputs → {OUT}")
