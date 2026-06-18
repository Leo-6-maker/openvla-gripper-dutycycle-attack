#!/usr/bin/env python3
"""Build SC5 student dataset from clean artifact-rich trajectories (flat dir structure).

Reuses: v2_privileged_teacher, find_sc5_anchor_v2, sc5_streaming_features_v2.
"""
import csv, hashlib, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig, calibrate_thresholds, find_sc5_anchor_v2)
from gripper_attack.sc5_streaming_features_v2 import (
    SC5StreamingFeatureAdapterV2, FEATURE_NAMES)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "tables"
ARTIFACT_DIR = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
DATA_SRC = sys.argv[3] if len(sys.argv) > 3 else \
    "/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/runs/libero_object"

K = 10; GUARD = 5
HELD_OUT_BUTTER = {8, 9, 11}

# --- Collect runs ---
print("Collecting runs...")
all_paths = []; run_entries = []
for d in sorted(os.listdir(DATA_SRC)):
    fp = os.path.join(DATA_SRC, d)
    sf = os.path.join(fp, "step_records.jsonl")
    if os.path.isdir(fp) and os.path.isfile(sf):
        all_paths.append(sf); run_entries.append((d, fp))
print(f"  {len(run_entries)} trajectories")

teacher = V2PrivilegedTeacher(calibrate_thresholds(all_paths))

# --- Build ---
rows = []; episodes = []; stats = defaultdict(int); seen = set()
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(ARTIFACT_DIR, exist_ok=True)

for dirname, full_path in run_entries:
    sf = os.path.join(full_path, "step_records.jsonl")
    mf = os.path.join(full_path, "run_manifest.json")
    stats["total"] += 1

    with open(sf) as f: records = [json.loads(line) for line in f]
    state_id = -1; success = False; task_name = "unknown"
    if os.path.isfile(mf):
        m = json.load(open(mf))
        state_id = m.get("state_id", -1); success = m.get("success", False)
        task_name = m.get("task_name", "unknown")
    if not success: stats["clean_fail"] += 1; continue
    stats["clean_success"] += 1

    # Dedup
    th = hashlib.sha256(dirname.encode()).hexdigest()
    if th in seen: stats["duplicate"] += 1; continue
    seen.add(th)

    # Teacher
    labels = teacher.label_trajectory(records)
    sc5 = find_sc5_anchor_v2(labels, K=K, guard=GUARD)
    stats["privileged"] += 1
    if sc5["valid"]: stats["sc5_valid"] += 1
    else: stats["sc5_invalid_" + sc5.get("reason", "?")] += 1

    is_butter = "butter" in task_name.lower()
    is_held_out = is_butter and state_id in HELD_OUT_BUTTER

    # Feature rows via streaming adapter
    adapter = SC5StreamingFeatureAdapterV2()
    first_step = None
    feat_rows = []
    for r in records:
        if not r.get("teacher_privileged_state_available"): continue
        step_raw = int(r.get("step_idx", r.get("policy_step_idx", 0)))
        if first_step is None: first_step = step_raw
        step = step_raw - first_step  # normalize to start from 0

        raw_grip = float(r.get("gripper_command", 0.5))
        env_grip = -1.0 if raw_grip > 0.5 else 1.0

        try:
            result = adapter.update(
                step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
                gripper_qpos=float(r.get("gripper_qpos", 0)),
                gripper_opening_proxy=float(r.get("gripper_width", r.get("gripper_opening_proxy", 0))),
                eef_x=float(r.get("eef_x", 0)), eef_y=float(r.get("eef_y", 0)),
                eef_z=float(r.get("eef_z", 0)),
                eef_vx=float(r.get("eef_vx", 0)), eef_vy=float(r.get("eef_vy", 0)),
                eef_vz=float(r.get("eef_vz", 0)),
                action_dx=float(r.get("action_dx", 0)), action_dy=float(r.get("action_dy", 0)),
                action_dz=float(r.get("action_dz", 0)),
                action_gripper=float(r.get("action_gripper", r.get("gripper_command", 0))))
        except ValueError: continue
        if not result["valid"]: continue

        tl = labels[step_raw] if step_raw < len(labels) else None
        has_sc5 = sc5["valid"] and sc5["window"] is not None
        in_cor = has_sc5 and sc5["window"][0] <= step_raw <= sc5["window"][1]

        row = dict(result["features"])
        row["step_idx"] = step_raw; row["state_id"] = state_id
        row["task_name"] = task_name; row["is_butter"] = is_butter
        row["is_held_out"] = is_held_out; row["run_id"] = dirname
        row["teacher_phase"] = tl["phase"] if tl else "abstain"
        row["teacher_sc5_corridor_valid"] = int(has_sc5)
        row["teacher_sc5_ready"] = int(in_cor)
        row["teacher_sc5_anchor"] = sc5["anchor"] if has_sc5 else -1
        row["teacher_sc5_window_start"] = sc5["window"][0] if has_sc5 else -1
        row["teacher_sc5_window_end"] = sc5["window"][1] if has_sc5 else -1
        row["teacher_stable_carry_start"] = sc5["stable_carry_start"] if has_sc5 else -1
        row["teacher_full_k10_valid"] = int(has_sc5)
        row["teacher_confidence"] = tl["confidence"] if tl else 0.0
        feat_rows.append(row)

    if feat_rows:
        episodes.append({"run_id": dirname, "task": task_name, "state_id": state_id,
            "is_butter": is_butter, "is_held_out": is_held_out,
            "n_steps": len(feat_rows), "sc5_valid": sc5["valid"],
            "sc5_anchor": sc5["anchor"], "sc5_reason": sc5["reason"]})
        rows.extend(feat_rows)

# --- Write ---
if rows:
    with open(os.path.join(OUTPUT_DIR, "v2_sc5_student_dataset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUTPUT_DIR, "v2_sc5_episode_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0].keys())); w.writeheader(); w.writerows(episodes)

dsm = {"source": DATA_SRC, "K": K, "guard": GUARD, "n_rows": len(rows),
       "n_episodes": len(episodes), "n_features": len(FEATURE_NAMES),
       "feature_names": FEATURE_NAMES, "held_out_butter": list(HELD_OUT_BUTTER),
       "stats": dict(stats)}
with open(os.path.join(ARTIFACT_DIR, "v2_sc5_dataset_manifest.json"), "w") as f:
    json.dump(dsm, f, indent=2, default=str)

print(f"Rows: {len(rows)} Episodes: {len(episodes)}")
for k, v in sorted(stats.items()): print(f"  {k}: {v}")
