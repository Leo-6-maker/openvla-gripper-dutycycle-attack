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
SC5_PHASES = ["approach","grasp_close","stable_grasp","first_lift","stable_carry",
              "pre_place_unsupported","release_safe","recovery_or_regrasp","abstain_unsupported"]

# Load model
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

class SC5MLP(torch.nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Linear(n_feat, hidden), torch.nn.ReLU(),
                                          torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.phase_head = torch.nn.Linear(hidden, len(SC5_PHASES))
        self.corridor_head = torch.nn.Linear(hidden, 1)
        self.release_head = torch.nn.Linear(hidden, 1)
        self.confidence_head = torch.nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        return {"phase_logits": self.phase_head(h), "corridor_logit": self.corridor_head(h),
                "release_logit": self.release_head(h), "confidence_logit": self.confidence_head(h)}

model = SC5MLP(n_feat=len(ckpt["feature_names"]))
model.load_state_dict(ckpt["model_state"])
model.eval()
mean = ckpt["mean"]; std = ckpt["std"]

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

    # Trigger state machine
    state = "MONITORING"; emit_step = -1; emit_anchor = -1
    for r in rows:
        X = np.array([[float(r[fn]) for fn in SC5_FEATURES]], dtype=np.float32)
        X = (X - mean) / (std + 1e-8)
        with torch.no_grad():
            out = model(torch.tensor(X, dtype=torch.float32))
        phase_prob = torch.softmax(out["phase_logits"], dim=1)[0]
        corridor_p = torch.sigmoid(out["corridor_logit"]).item()
        release_p = torch.sigmoid(out["release_logit"]).item()
        pred_phase = SC5_PHASES[phase_prob.argmax().item()]
        step = int(r["step_idx"])

        # State transitions
        if state == "MONITORING":
            if pred_phase == "stable_carry" and corridor_p > TAU_CORRIDOR:
                state = "ARMED"; arm_step = step
        elif state == "ARMED":
            if step >= arm_step + GUARD and corridor_p > TAU_CORRIDOR and release_p < TAU_RELEASE:
                state = "EMITTED"; emit_step = step; emit_anchor = step

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
        pred_ws = emit_anchor; pred_we = emit_anchor + K - 1
        teacher_ws = teacher_anchor
        teacher_we = teacher_anchor + K - 1 if teacher_anchor >= 0 else -1
        anchor_err = abs(emit_anchor - teacher_anchor) if teacher_anchor >= 0 else -1
        false_early = emit_anchor < teacher_anchor if teacher_anchor >= 0 else False
        post_release = any(SC5_PHASES.index(r.get("teacher_phase","abstain_unsupported"))
                          >= SC5_PHASES.index("release_safe")
                          for r in rows if int(r["step_idx"]) >= emit_step)
        k10_contained = k10_map.get(emit_step, 0) > 0 if k10_map else False
    else:
        pred_ws = -1; pred_we = -1; anchor_err = -1
        false_early = False; post_release = False; k10_contained = False

    results.append({
        "episode_id": eid, "split": split, "is_held_out": is_held,
        "teacher_anchor": teacher_anchor, "teacher_sc5_valid": teacher_sc5_valid,
        "emit_step": emit_step, "emit_anchor": emit_anchor,
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
