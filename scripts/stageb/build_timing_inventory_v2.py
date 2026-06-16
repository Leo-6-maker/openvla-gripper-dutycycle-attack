#!/usr/bin/env python3
"""Phase 1: Build full train/val timing inventory and pre-register 12 parents."""
import csv, json, os, random, sys

LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
EVAL = "/data/liuyu/outputs/d5_training/d5_evaluation_readout.csv"
MANIFEST = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
OUT_DIR = "/data/liuyu/outputs"

labels = list(csv.DictReader(open(LABELS)))
eval_rows = list(csv.DictReader(open(EVAL)))
emit_map = {(r["task"], int(r["state_id"])): r for r in eval_rows}
manifest = list(csv.DictReader(open(MANIFEST)))
acc_map = {}
for r in manifest:
    if r.get("status") == "BOUND":
        acc_map[(r["task"], int(r["state_id"]))] = r

# Filter: train or val only, labeled, clean_success=1
eligible = []
for r in labels:
    if r["status"] != "VALID_LABELED":
        continue
    key = (r["task"], int(r["state_id"]))
    acc = acc_map.get(key, {})
    sp = acc.get("split", "?")
    if sp not in ("train", "val"):
        continue
    if acc.get("success", "0") != "1":
        continue
    e = emit_map.get(key, {})
    emit = int(e.get("emit_step", -1))
    if emit < 0:
        continue
    ws = int(r["ws"])
    anchor = int(r["anchor"])
    we = int(r["we"])

    if emit == anchor:
        cls = "exact"
    elif emit < ws:
        cls = "early"
    elif emit < anchor:
        cls = "in_window_pre_anchor"
    elif emit < we:
        cls = "in_window_post_anchor"
    else:
        cls = "late"

    eligible.append({
        "task": r["task"], "state_id": int(r["state_id"]), "split": sp,
        "teacher_anchor": anchor, "teacher_ws": ws, "teacher_we": we,
        "d5_emit": emit, "d5_score": float(e.get("emit_score", 0)),
        "emit_class": cls,
        "trace_dir": acc.get("accepted_root", "") + "/" + acc.get("accepted_episode_dir", ""),
        "n_candidates": int(e.get("n_candidates", 0)),
    })

print("Total eligible (train+val, labeled, success=1, emit>=0): " + str(len(eligible)))
for cls in ["exact", "in_window_pre_anchor", "in_window_post_anchor", "early", "late"]:
    n = sum(1 for e in eligible if e["emit_class"] == cls)
    print("  " + cls + ": " + str(n))

# Miss candidates
miss_candidates = []
for r in labels:
    if r["status"] != "VALID_LABELED":
        continue
    key = (r["task"], int(r["state_id"]))
    acc = acc_map.get(key, {})
    sp = acc.get("split", "?")
    if sp not in ("train", "val"):
        continue
    if acc.get("success", "0") != "1":
        continue
    e = emit_map.get(key, {})
    if int(e.get("emit_step", -1)) >= 0:
        continue
    miss_candidates.append({
        "task": r["task"], "state_id": int(r["state_id"]), "split": sp,
        "teacher_anchor": int(r["anchor"]),
    })
print("Miss candidates: " + str(len(miss_candidates)))


def select(candidates, n_needed, label, exclude_tasks=None):
    if exclude_tasks is None:
        exclude_tasks = set()
    task_count = {}
    selected = []
    random.seed(12345)
    # Prioritize uncovered tasks
    uncovered = [c for c in candidates if c["task"] not in exclude_tasks]
    covered = [c for c in candidates if c["task"] in exclude_tasks]
    random.shuffle(uncovered)
    random.shuffle(covered)
    for c in uncovered + covered:
        t = c["task"]
        if t not in task_count:
            task_count[t] = 0
        if task_count[t] >= 2:
            continue
        selected.append(c)
        task_count[t] += 1
        if len(selected) >= n_needed:
            break
    return selected


# Select in order: exact first (to establish task coverage baseline)
exact = select([e for e in eligible if e["emit_class"] == "exact"], 4, "exact")
covered_tasks = set(p["task"] for p in exact)
# Early: prioritize uncovered tasks for diversity
early_list = select([e for e in eligible if e["emit_class"] == "early"], 5, "early", covered_tasks)
covered_tasks.update(p["task"] for p in early_list)
# Late: prioritize uncovered
late_list = select([e for e in eligible if e["emit_class"] == "late"], 2, "late", covered_tasks)
covered_tasks.update(p["task"] for p in late_list)
# Miss: prioritize uncovered
miss_list = select(miss_candidates, 2, "miss", covered_tasks)
covered_tasks.update(p["task"] for p in miss_list)

# If still < 8 tasks, add more exact from uncovered
if len(covered_tasks) < 8:
    extra = select([e for e in eligible if e["emit_class"] == "exact"], 2, "exact", covered_tasks)
    exact.extend(extra)
    covered_tasks.update(p["task"] for p in extra)

# Late quota: only 1 available in train+val
if len(late_list) < 2:
    print("WARNING: Only " + str(len(late_list)) + " late parents available in train+val. PARTIAL_QUOTA.")

all_selected = exact + early_list + late_list + miss_list
task_set = set(p["task"] for p in all_selected)
print("\nSelected: " + str(len(all_selected)) + " parents, " + str(len(task_set)) + " tasks")
for p in all_selected:
    cls = p.get("emit_class", "miss")
    anchor = p.get("teacher_anchor", "?")
    emit = p.get("d5_emit", "miss")
    print("  " + p["task"] + "_s" + str(p["state_id"]) + ": " + p["split"] + " " + cls + " anchor=" + str(anchor) + " emit=" + str(emit))

# Write inventory
out = os.path.join(OUT_DIR, "l12_timing_candidate_inventory_v2.csv")
with open(out, "w", newline="") as f:
    fields = ["task", "state_id", "split", "teacher_anchor", "teacher_ws", "teacher_we",
              "d5_emit", "d5_score", "emit_class", "trace_dir", "n_candidates"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(eligible)
print("\nInventory: " + out + " (" + str(len(eligible)) + " rows)")

# Pre-registration
out2 = os.path.join(OUT_DIR, "l12_timing_panel_v2_prereg.csv")
with open(out2, "w", newline="") as f:
    fields2 = ["task", "state_id", "split", "emit_class", "teacher_anchor", "d5_emit", "role"]
    w = csv.DictWriter(f, fieldnames=fields2)
    w.writeheader()
    for p in all_selected:
        p["role"] = "primary" if p.get("emit_class", "miss") != "miss" else "diagnostic"
        if "d5_emit" not in p:
            p["d5_emit"] = "miss"
        w.writerow({k: p.get(k, "") for k in fields2})
print("Pre-reg: " + out2)
print("Tasks: " + str(sorted(task_set)))
