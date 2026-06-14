#!/usr/bin/env python3
"""Generate all D1b preregistration artifacts from E4C.2b results."""

import csv, hashlib, os, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(str(ROOT))

status = list(csv.DictReader(open("tables/e4c_audit/l12_e4c2b_trace_status.csv")))
cands = list(csv.DictReader(open("tables/e4c_audit/l12_e4c2b_close_candidates.csv")))
multi = [r for r in status if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"]
print(f"Multi-candidate traces: {len(multi)}")

# Deterministic split plan (frozen at prereg)
SPLIT_PLAN = {
    "alphabet_soup": (12, 3, 3),
    "bbq_sauce": (9, 2, 2),
    "butter": (12, 2, 3),
    "chocolate_pudding": (5, 1, 1),
    "cream_cheese": (10, 3, 2),
    "ketchup": (14, 3, 4),
    "milk": (4, 1, 1),
    "orange_juice": (5, 2, 1),
    "salad_dressing": (7, 1, 2),
    "tomato_sauce": (12, 2, 2),
}
t_train = sum(v[0] for v in SPLIT_PLAN.values())
t_val = sum(v[1] for v in SPLIT_PLAN.values())
t_test = sum(v[2] for v in SPLIT_PLAN.values())
print(f"Split: train={t_train} val={t_val} test={t_test} sum={t_train+t_val+t_test}")

by_task = defaultdict(list)
for r in multi:
    by_task[r["task_key"]].append(r)

split_map = {}
for task, (n_train, n_val, n_test) in SPLIT_PLAN.items():
    traces = sorted(by_task[task], key=lambda x: int(x["state_id"]))
    for i, tr in enumerate(traces):
        if i < n_train:
            split_map[tr["trace_id"]] = "train"
        elif i < n_train + n_val:
            split_map[tr["trace_id"]] = "val"
        else:
            split_map[tr["trace_id"]] = "test"

out_dir = Path("tables/deepseek_detector")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Split manifest
with open(out_dir / "d1b_split_manifest.csv", "w", newline="") as f:
    fields = ["trace_id", "task_key", "state_id", "seed", "split",
              "logical_group_id", "n_close_candidates", "n_tp_qualifying", "teacher_p_step"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in sorted(multi, key=lambda x: (x["task_key"], int(x["state_id"]))):
        w.writerow({
            "trace_id": r["trace_id"], "task_key": r["task_key"],
            "state_id": r["state_id"], "seed": r["seed"],
            "split": split_map[r["trace_id"]],
            "logical_group_id": f"{r['task_key']}_s{r['state_id']}_seed{r['seed']}",
            "n_close_candidates": r["n_close_candidates"],
            "n_tp_qualifying": r["n_tp_qualifying_candidates"],
            "teacher_p_step": r["teacher_p_step"],
        })

# 2. Training manifest
inv = list(csv.DictReader(open("tables/e4c_audit/l12_e4c_data_inventory_v2.csv")))
inv_by_file = {r["filename"]: r for r in inv}
with open(out_dir / "d1b_training_manifest.csv", "w", newline="") as f:
    fields = ["trace_id", "task_key", "state_id", "seed", "split",
              "source_path", "source_sha256", "row_count", "remapper_version",
              "grasp_privilege_valid", "n_close_candidates", "n_tp_qualifying_candidates",
              "teacher_p_step", "teacher_r_step", "category"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in sorted(multi, key=lambda x: (x["task_key"], int(x["state_id"]))):
        inv_row = inv_by_file.get(r["trace_id"] + ".csv", {})
        w.writerow({
            "trace_id": r["trace_id"], "task_key": r["task_key"],
            "state_id": r["state_id"], "seed": r["seed"],
            "split": split_map[r["trace_id"]],
            "source_path": inv_row.get("path", ""),
            "source_sha256": inv_row.get("sha256", ""),
            "row_count": r["n_total_rows"],
            "remapper_version": "rc1a_corrected_v2_e1_5",
            "grasp_privilege_valid": r["grasp_privilege_valid"],
            "n_close_candidates": r["n_close_candidates"],
            "n_tp_qualifying_candidates": r["n_tp_qualifying_candidates"],
            "teacher_p_step": r["teacher_p_step"],
            "teacher_r_step": r["teacher_r_step"],
            "category": r["category"],
        })

# 3. Split summary by task
with open(out_dir / "d1b_split_summary.csv", "w", newline="") as f:
    fields = ["task_key", "n_total", "n_train", "n_val", "n_test",
              "train_state_ids", "val_state_ids", "test_state_ids"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for task in sorted(by_task):
        traces = sorted(by_task[task], key=lambda x: int(x["state_id"]))
        train_ids = [r["state_id"] for r in traces if split_map[r["trace_id"]] == "train"]
        val_ids = [r["state_id"] for r in traces if split_map[r["trace_id"]] == "val"]
        test_ids = [r["state_id"] for r in traces if split_map[r["trace_id"]] == "test"]
        w.writerow({
            "task_key": task, "n_total": len(traces),
            "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
            "train_state_ids": ",".join(train_ids),
            "val_state_ids": ",".join(val_ids),
            "test_state_ids": ",".join(test_ids),
        })

# 4. Leakage audit
leakage = []
trace_splits = defaultdict(set)
for r in multi:
    trace_splits[r["trace_id"]].add(split_map[r["trace_id"]])
for tid, splits in trace_splits.items():
    if len(splits) > 1:
        leakage.append({"check": "trace_id_across_split", "detail": f"{tid}: {splits}", "violation": "True"})
group_splits = defaultdict(set)
for r in multi:
    gid = f"{r['task_key']}_s{r['state_id']}_seed{r['seed']}"
    group_splits[gid].add(split_map[r["trace_id"]])
for gid, splits in group_splits.items():
    if len(splits) > 1:
        leakage.append({"check": "logical_group_across_split", "detail": f"{gid}: {splits}", "violation": "True"})
ts_splits = defaultdict(set)
for r in multi:
    ts_splits[(r["task_key"], r["state_id"])].add(split_map[r["trace_id"]])
for (t, s), splits in ts_splits.items():
    if len(splits) > 1:
        leakage.append({"check": "task_state_across_split", "detail": f"{t}_s{s}: {splits}", "violation": "True"})
if not leakage:
    leakage.append({"check": "ALL_CHECKS", "detail": "no leakage detected", "violation": "False"})
with open(out_dir / "d1b_leakage_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["check", "detail", "violation"]); w.writeheader(); w.writerows(leakage)
print(f"Leakage violations: {sum(1 for l in leakage if l['violation'] == 'True')}")

# 5. Feature normalization (from train only)
train_ids = {r["trace_id"] for r in multi if split_map[r["trace_id"]] == "train"}
train_cands = [c for c in cands if c["trace_id"] in train_ids]
print(f"Train candidates for normalization: {len(train_cands)}")

feat_names = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus", "close_onset_qpos_bonus",
    "eef_deceleration_bonus", "qpos_ready_bonus", "eef_speed_now", "eef_speed_prev",
    "eef_deceleration_delta", "close_streak", "raw_crossing", "close_onset",
    "qpos", "time_since_prev_close", "time_since_last_open", "candidate_index",
]
feat_stats = []
for fn in feat_names:
    vals = []; n_missing = 0
    for c in train_cands:
        v = c.get(fn, "")
        if v == "" or v is None: n_missing += 1
        else:
            try: vals.append(float(v))
            except: n_missing += 1
    if vals:
        feat_stats.append({
            "feature": fn, "n_train_values": len(vals), "n_missing": n_missing,
            "mean": round(statistics.mean(vals), 6),
            "stdev": round(statistics.stdev(vals) if len(vals) > 1 else 0, 6),
            "min": round(min(vals), 6), "max": round(max(vals), 6),
            "normalization": "z_score",
        })
    else:
        feat_stats.append({
            "feature": fn, "n_train_values": 0, "n_missing": n_missing,
            "mean": "", "stdev": "", "min": "", "max": "", "normalization": "z_score",
        })
with open(out_dir / "d1b_feature_normalization.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(feat_stats[0].keys())); w.writeheader(); w.writerows(feat_stats)

# Candidate counts per split
for sn in ["train", "val", "test"]:
    ids = {r["trace_id"] for r in multi if split_map[r["trace_id"]] == sn}
    sc = [c for c in cands if c["trace_id"] in ids]
    n_pos = sum(1 for c in sc if c["is_teacher_p"] == "1")
    n_neg = len(sc) - n_pos
    print(f"{sn}: {len(ids)} traces, {len(sc)} candidates ({n_pos} pos, {n_neg} neg)")

# SHAs
for fname in sorted(os.listdir(str(out_dir))):
    if fname.startswith("d1b_") and fname.endswith(".csv"):
        h = hashlib.sha256()
        with open(out_dir / fname, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""): h.update(chunk)
        print(f"SHA {fname}: {h.hexdigest()}")

print("D1b CSVs generated.")
