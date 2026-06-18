#!/usr/bin/env python3
"""Diagnose post-release triggers: offset distribution, corridor/release probs, phase."""
import csv, json, sys, numpy as np, torch
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/sc5_canonical_eng/sc5_mlp_s1.pt"
DATASET_PATH = sys.argv[2] if len(sys.argv) > 2 else "tables/v2_sc5_canonical_dataset.csv"

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
model.load_state_dict(ckpt["model_state"]); model.eval()
mean = ckpt["mean"]; std = ckpt["std"]

episodes = defaultdict(list)
with open(DATASET_PATH) as f:
    for r in csv.DictReader(f):
        episodes[r.get("episode_id", r.get("run_id","?"))].append(r)
for eid in episodes:
    episodes[eid].sort(key=lambda r: int(r.get("step_idx", 0)))

K = 10; GUARD = 5
errors = []  # (offset_from_release, corridor_p, release_p, phase, raw_grip, open_streak)

for eid, rows in sorted(episodes.items()):
    teacher_anchor = int(rows[0].get("teacher_sc5_anchor", -1))
    if teacher_anchor < 0: continue

    # Find first release_safe step
    release_step = None
    for r in rows:
        if r.get("teacher_phase","") == "release_safe":
            release_step = int(r["step_idx"]); break
    if release_step is None: continue

    state = "MONITORING"; emit_step = -1; arm_step = -1
    for r in rows:
        step = int(r["step_idx"])
        X = np.array([[float(r[fn]) for fn in SC5_FEATURES]], dtype=np.float32)
        X = (X - mean) / (std + 1e-8)
        with torch.no_grad():
            out = model(torch.tensor(X, dtype=torch.float32))
        corridor_p = torch.sigmoid(out["corridor_logit"]).item()
        release_p = torch.sigmoid(out["release_logit"]).item()
        phase_prob = torch.softmax(out["phase_logits"], dim=1)[0]
        pred_phase = SC5_PHASES[phase_prob.argmax().item()]
        raw_grip = float(r.get("gripper_command", 0))
        open_streak = int(float(r.get("recent_open_streak", 0)))

        if state == "MONITORING":
            if pred_phase == "stable_carry" and corridor_p > 0.3:
                state = "ARMED"; arm_step = step
        elif state == "ARMED":
            if step >= arm_step + GUARD and corridor_p > 0.3:
                state = "EMITTED"; emit_step = step
                if emit_step > release_step:
                    offset = emit_step - release_step
                    errors.append({
                        "offset": offset,
                        "corridor_p": round(corridor_p, 4),
                        "release_p": round(release_p, 4),
                        "pred_phase": pred_phase,
                        "raw_grip": round(raw_grip, 3),
                        "open_streak": open_streak,
                        "episode": eid[:16], "emit_step": emit_step,
                        "release_step": release_step,
                    })

# Distribution analysis
print(f"Post-release triggers: {len(errors)}")
if errors:
    bins = {"0-2": (0,2), "3-5": (3,5), "6-10": (6,10), "10+": (10,999)}
    for label, (lo, hi) in bins.items():
        in_bin = [e for e in errors if lo <= e["offset"] <= hi]
        if in_bin:
            avg_c = np.mean([e["corridor_p"] for e in in_bin])
            avg_r = np.mean([e["release_p"] for e in in_bin])
            phases = Counter(e["pred_phase"] for e in in_bin)
            avg_os = np.mean([e["open_streak"] for e in in_bin])
            print(f"\n  {label} steps after release ({len(in_bin)} errors):")
            print(f"    avg corridor_p={avg_c:.3f}, avg release_p={avg_r:.3f}")
            print(f"    avg open_streak={avg_os:.1f}")
            print(f"    phases: {dict(phases.most_common(3))}")

    # Overall stats
    all_c = [e["corridor_p"] for e in errors]
    all_r = [e["release_p"] for e in errors]
    print(f"\n  Overall: corridor_p mean={np.mean(all_c):.3f} std={np.std(all_c):.3f}")
    print(f"  Overall: release_p mean={np.mean(all_r):.3f} std={np.std(all_r):.3f}")
    print(f"  release_p < 0.3: {sum(1 for r in all_r if r < 0.3)}/{len(all_r)}")
    print(f"  release_p < 0.5: {sum(1 for r in all_r if r < 0.5)}/{len(all_r)}")
    print(f"  corridor_p > 0.5 AND release_p < 0.3: {sum(1 for e in errors if e['corridor_p'] > 0.5 and e['release_p'] < 0.3)}/{len(all_r)}")
