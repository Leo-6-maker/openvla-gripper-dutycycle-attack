#!/usr/bin/env python3
"""SC5 dataset v3: fixed action_dy, fail-closed, held-out split, real corridor, content dedup.

Fixes vs v2:
  - action_dy gets correct value (was duplicated from action_dz)
  - Fail-closed: checks all required fields exist before calling adapter
  - Teacher calibrated on TRAIN data only (held-out excluded)
  - Content-based trajectory dedup (not dirname-based)
  - Real valid-start corridor via compute_sc5_valid_start_corridor()
  - Separates attack_window_active, sc5_ready, corridor_active, full_k10_valid
"""
import csv, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.v2_privileged_teacher import (
    V2PrivilegedTeacher, TeacherConfig, calibrate_thresholds,
    find_sc5_anchor_v2, compute_sc5_valid_start_corridor)
from gripper_attack.sc5_streaming_features_v2 import (
    SC5StreamingFeatureAdapterV2, FEATURE_NAMES)

OUT = sys.argv[1] if len(sys.argv) > 1 else "tables"
ART = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
DATA_SRC = sys.argv[3] if len(sys.argv) > 3 else \
    "/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/runs/libero_object"

K, GUARD = 10, 5
HELD_OUT_BUTTER = {8, 9, 11}

# --- Collect runs, split calibration ---
train_paths = []; held_out_paths = []; run_entries = []
for d in sorted(os.listdir(DATA_SRC)):
    fp = os.path.join(DATA_SRC, d); sf = os.path.join(fp, "step_records.jsonl")
    if not (os.path.isdir(fp) and os.path.isfile(sf)): continue
    mf = os.path.join(fp, "run_manifest.json")
    state_id = -1; task_name = "unknown"
    if os.path.isfile(mf):
        m = json.load(open(mf)); state_id = m.get("state_id", -1); task_name = m.get("task_name", "unknown")
    is_butter = "butter" in task_name.lower()
    if is_butter and state_id in HELD_OUT_BUTTER:
        held_out_paths.append(sf)
    else:
        train_paths.append(sf)
    run_entries.append((d, fp, state_id, task_name, is_butter))

print(f"Calibrating on {len(train_paths)} train paths ({len(held_out_paths)} held-out excluded)")
teacher = V2PrivilegedTeacher(calibrate_thresholds(train_paths))

rows = []; episodes = []; stats = defaultdict(int); seen_hashes = set()
os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)

REQUIRED_FIELDS = [
    "gripper_command", "gripper_qpos", "gripper_width",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

for dirname, full_path, state_id, task_name, is_butter in run_entries:
    sf = os.path.join(full_path, "step_records.jsonl")
    mf = os.path.join(full_path, "run_manifest.json")
    stats["total"] += 1

    with open(sf) as f: records = [json.loads(line) for line in f]
    success = json.load(open(mf)).get("success", False) if os.path.isfile(mf) else False
    if not success: stats["clean_fail"] += 1; continue
    stats["clean_success"] += 1

    # Content-based dedup
    content_key = json.dumps([(r.get("step_idx"), round(float(r.get("eef_x", 0)), 4),
        round(float(r.get("eef_y", 0)), 4), round(float(r.get("eef_z", 0)), 4))
        for r in records[:5]], sort_keys=True)
    th = hashlib.sha256(content_key.encode()).hexdigest()
    if th in seen_hashes: stats["duplicate"] += 1; continue
    seen_hashes.add(th)

    is_held_out = is_butter and state_id in HELD_OUT_BUTTER

    # Teacher labels
    labels = teacher.label_trajectory(records)
    sc5 = find_sc5_anchor_v2(labels, K=K, guard=GUARD)
    stats["privileged"] += 1
    if sc5["valid"]: stats["sc5_valid"] += 1
    else: stats["sc5_invalid"] += 1

    # Real corridor
    corridor = None
    if sc5["valid"]:
        corridor = compute_sc5_valid_start_corridor(labels, sc5["anchor"], K=K)

    # Feature rows (fail-closed)
    adapter = SC5StreamingFeatureAdapterV2()
    first_step = None; feat_rows = []
    for r in records:
        if not r.get("teacher_privileged_state_available"): continue
        step_raw = int(r.get("step_idx", r.get("policy_step_idx", 0)))
        if first_step is None: first_step = step_raw
        step = step_raw - first_step

        # Fail-closed: check required fields
        missing = []
        for fld in REQUIRED_FIELDS:
            v = r.get(fld, None)
            if v is None or v == "" or v == "nan": missing.append(fld)
        if missing: continue

        raw_grip = float(r["gripper_command"])
        env_grip = -1.0 if raw_grip > 0.5 else 1.0

        try:
            result = adapter.update(
                step_id=step, raw_gripper=raw_grip, env_gripper=env_grip,
                gripper_qpos=float(r["gripper_qpos"]),
                gripper_opening_proxy=float(r.get("gripper_width", r.get("gripper_opening_proxy", 0))),
                eef_x=float(r["eef_x"]), eef_y=float(r["eef_y"]), eef_z=float(r["eef_z"]),
                eef_vx=float(r.get("eef_vx", 0)), eef_vy=float(r.get("eef_vy", 0)),
                eef_vz=float(r.get("eef_vz", 0)),
                action_dx=float(r.get("action_dx", 0)),
                action_dy=float(r.get("action_dy", 0)),  # FIXED: was action_dz
                action_dz=float(r.get("action_dz", 0)),
                action_gripper=float(r.get("action_gripper", raw_grip)))
        except ValueError: continue
        if not result["valid"]: continue

        tl = labels[step_raw] if step_raw < len(labels) else None
        has_sc5 = sc5["valid"]
        in_attack_window = has_sc5 and sc5["window"][0] <= step_raw <= sc5["window"][1]
        in_corridor = corridor is not None and step_raw in corridor["corridor_active_at_t"]
        k10_valid = (corridor is not None and step_raw < len(corridor["full_k10_valid_at_t"])
                     and corridor["full_k10_valid_at_t"][step_raw])

        row = dict(result["features"])
        row["step_idx"] = step_raw; row["state_id"] = state_id
        row["task_name"] = task_name; row["is_butter"] = is_butter
        row["is_held_out"] = is_held_out; row["run_id"] = dirname
        row["teacher_phase"] = tl["phase"] if tl else "abstain"
        row["teacher_sc5_anchor"] = sc5["anchor"] if has_sc5 else -1
        row["teacher_sc5_attack_window_active"] = int(in_attack_window)
        row["teacher_sc5_ready"] = int(has_sc5 and step_raw == sc5["anchor"])
        row["teacher_sc5_corridor_active"] = int(in_corridor)
        row["teacher_full_k10_valid_at_t"] = int(k10_valid)
        row["teacher_stable_carry_start"] = sc5["stable_carry_start"] if has_sc5 else -1
        row["teacher_confidence"] = tl["confidence"] if tl else 0.0
        feat_rows.append(row)

    if feat_rows:
        episodes.append({"run_id": dirname, "task": task_name, "state_id": state_id,
            "is_butter": is_butter, "is_held_out": is_held_out,
            "n_steps": len(feat_rows), "sc5_valid": sc5["valid"],
            "sc5_anchor": sc5["anchor"],
            "corridor_start": corridor["corridor_start"] if corridor else -1,
            "corridor_end": corridor["corridor_end"] if corridor else -1})
        rows.extend(feat_rows)

# --- Write ---
if rows:
    with open(os.path.join(OUT, "v2_sc5_dataset_v3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "v2_sc5_episode_manifest_v3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        w.writeheader(); w.writerows(episodes)

dsm = {"K": K, "guard": GUARD, "n_rows": len(rows), "n_episodes": len(episodes),
       "held_out_butter": list(HELD_OUT_BUTTER),
       "teacher_calibrated_on": len(train_paths),
       "held_out_excluded_from_cal": len(held_out_paths),
       "stats": dict(stats)}
with open(os.path.join(ART, "v2_sc5_dataset_manifest_v3.json"), "w") as f:
    json.dump(dsm, f, indent=2, default=str)

print(f"Rows: {len(rows)} Episodes: {len(episodes)}")
for k, v in sorted(stats.items()): print(f"  {k}: {v}")
butter_eps = [e for e in episodes if e["is_butter"]]
for e in butter_eps:
    print(f"  butter_s{e['state_id']}: sc5={e['sc5_valid']} anchor={e['sc5_anchor']} cor=[{e['corridor_start']},{e['corridor_end']}] held={e['is_held_out']}")
