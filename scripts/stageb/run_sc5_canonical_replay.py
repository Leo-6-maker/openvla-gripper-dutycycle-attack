#!/usr/bin/env python3
"""SC5 student replay on canonical corpus — per-episode, first-trigger, K10 corridor.

Reuses: train_sc5_v4.SC5MLP model, mature streaming adapter, SC5 trigger state machine.
"""
import csv, json, os, sys, numpy as np, torch
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/sc5_canonical_eng/sc5_mlp_s1.pt"
DATASET_PATH = sys.argv[2] if len(sys.argv) > 2 else "tables/v2_sc5_canonical_dataset.csv"
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "tables"

SC5_FEATURES = [
    "gripper_command","gripper_qpos","gripper_opening_proxy",
    "eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
    "action_dx","action_dy","action_dz","action_gripper",
    "recent_close_streak","recent_open_streak","recent_gripper_flip_count",
    "close_onset","time_since_close","eef_speed",
    "eef_z_delta_since_close","qpos_delta_1","qpos_delta_3",
    "opening_proxy_delta_3","opening_proxy_variance_5","eef_speed_variance_5",
]
# Load shared detector runtime
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES
detector = SC5DetectorRuntime(MODEL_PATH, tau_corridor=0.3, tau_release=0.3, guard=5)
mean = detector.mean; std = detector.std

# Load dataset grouped by episode
episodes = defaultdict(list)
with open(DATASET_PATH) as f:
    for r in csv.DictReader(f):
        eid = r.get("episode_id", r.get("run_id", "?"))
        episodes[eid].append(r)
for eid in episodes:
    episodes[eid].sort(key=lambda r: int(r.get("step_idx", 0)))

K = 10; GUARD = 5
TAU_CORRIDOR = 0.3; TAU_RELEASE = 0.3

results = []
for eid, rows in sorted(episodes.items()):
    teacher_anchor = int(rows[0].get("teacher_sc5_anchor", -1))
    teacher_sc5_valid = rows[0].get("teacher_sc5_anchor", "-1") != "-1"
    is_held = rows[0].get("is_held_out", "False") in ("True", "true", "1")
    split = rows[0].get("split", "?")

    # Shared runtime trigger (same as online detector)
    detector.reset()
    emit_step = -1
    for r in rows:
        feats = {fn: float(r[fn]) for fn in SC5_FEATURES}
        step = int(r["step_idx"])
        decision = detector.update(feats, step)
        if decision["emitted"]:
            emit_step = decision["emit_step"]; break

    # Metrics
    full_k10 = rows[0].get("teacher_full_k10_valid_at_t", "0")
    if isinstance(full_k10, str):
        try:
            k10_map = {int(r["step_idx"]): int(r.get("teacher_full_k10_valid_at_t", 0))
                      for r in rows}
        except: k10_map = {}
    else:
        k10_map = {}

    if emit_step >= 0:
        pred_ws = emit_step; pred_we = emit_step + K - 1
        teacher_ws = teacher_anchor
        teacher_we = teacher_anchor + K - 1 if teacher_anchor >= 0 else -1
        anchor_err = abs(emit_step - teacher_anchor) if teacher_anchor >= 0 else -1
        false_early = emit_step < teacher_anchor if teacher_anchor >= 0 else False
        # Post-release: trigger happened AFTER first release_safe
        first_release = next((int(r["step_idx"]) for r in rows
                             if r.get("teacher_phase","") == "release_safe"), None)
        post_release = (first_release is not None and emit_step > first_release)
        k10_contained = k10_map.get(emit_step, 0) > 0 if k10_map else False
    else:
        pred_ws = -1; pred_we = -1; anchor_err = -1
        false_early = False; post_release = False; k10_contained = False

    results.append({
        "episode_id": eid, "split": split, "is_held_out": is_held,
        "teacher_anchor": teacher_anchor, "teacher_sc5_valid": teacher_sc5_valid,
        "emit_step": emit_step,
        "anchor_error": anchor_err, "false_early": int(false_early),
        "post_release_trigger": int(post_release),
        "k10_contained": int(k10_contained), "triggered": int(emit_step >= 0),
    })

# Summary
n = len(results)
triggered = [r for r in results if r["triggered"]]
sc5_eps = [r for r in results if r["teacher_sc5_valid"]]
not_triggered_in_sc5 = [r for r in sc5_eps if not r["triggered"]]
false_early_eps = [r for r in results if r["false_early"]]
k10_contained_eps = [r for r in triggered if r["k10_contained"]]
post_release_eps = [r for r in results if r["post_release_trigger"]]
held_eps = [r for r in results if r["is_held_out"]]
train_eps = [r for r in results if r["split"] == "train"]
val_eps = [r for r in results if r["split"] == "val"]

# Coverage: proportion of SC5-valid episodes where we trigger
coverage = len([r for r in sc5_eps if r["triggered"]]) / max(len(sc5_eps), 1)
# False-early rate
false_early_rate = len(false_early_eps) / max(len(triggered), 1)
# K10 containment
k10_containment = len(k10_contained_eps) / max(len([r for r in triggered if r["teacher_sc5_valid"]]), 1)
# Post-release rate
post_release_rate = len(post_release_eps) / max(len(triggered), 1)
# Median anchor error
anchor_errors = [r["anchor_error"] for r in triggered if r["anchor_error"] >= 0]
median_err = np.median(anchor_errors) if anchor_errors else -1
# No-corridor abstain
no_corridor_eps = [r for r in results if not r["teacher_sc5_valid"]]
abstain_correct = len([r for r in no_corridor_eps if not r["triggered"]])
no_corridor_abstain = abstain_correct / max(len(no_corridor_eps), 1)

print(f"Episodes: {n} ({len(train_eps)} train, {len(val_eps)} val, {len(held_eps)} held)")
print(f"SC5-valid episodes: {len(sc5_eps)}")
print(f"Triggered: {len(triggered)}")
print(f"Coverage: {coverage:.3f}")
print(f"False-early: {false_early_rate:.3f} ({len(false_early_eps)} episodes)")
print(f"Post-release trigger: {post_release_rate:.3f} ({len(post_release_eps)} episodes)")
print(f"Median anchor error: {median_err:.1f}")
print(f"K10 containment: {k10_containment:.3f} ({len(k10_contained_eps)}/{len([r for r in triggered if r['teacher_sc5_valid']])})")
print(f"No-corridor abstain: {no_corridor_abstain:.3f} ({abstain_correct}/{len(no_corridor_eps)})")

# Held-out breakdown
if held_eps:
    print(f"\nHeld-out:")
    for r in held_eps:
        print(f"  {r['episode_id'][:16]}: trigger={r['triggered']} err={r['anchor_error']} k10={r['k10_contained']}")

# Save CSV
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "v2_sc5_replay_canonical.csv")
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\nSaved: {out_path}")
