#!/usr/bin/env python3
"""SC5 student causal replay evaluator — per-episode, first-trigger lock, K10 corridor.

Reuses: ProprioCausalMLP-style inference, streaming feature adapter, SC5 trigger logic.
"""
import csv, json, os, sys, numpy as np, torch
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/sc5_student_v2/sc5_student_mlp.pt"
DATASET_PATH = sys.argv[2] if len(sys.argv) > 2 else "tables/v2_sc5_student_dataset.csv"
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "tables"

# --- Load model ---
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
feature_names = ckpt["feature_names"]
phase_classes = ckpt["phase_classes"]
mean = ckpt["mean"]; std = ckpt["std"]

# Re-create model
from scripts.stageb.train_sc5_student_v2 import SC5ProprioMLP
model = SC5ProprioMLP(n_features=len(feature_names))
model.load_state_dict(ckpt["model_state"])
model.eval()

# --- Load dataset grouped by episode ---
episodes = defaultdict(list)
with open(DATASET_PATH) as f:
    for r in csv.DictReader(f):
        ek = r.get("run_id", "unknown")
        episodes[ek].append(r)

# Sort each episode by step_idx
for ek in episodes:
    episodes[ek].sort(key=lambda r: int(r.get("step_idx", 0)))

K = 10; GUARD = 5
TAU_CORRIDOR = 0.3; TAU_RELEASE = 0.3; TAU_CONF = 0.3

results = []

for ek, rows in sorted(episodes.items()):
    teacher_anchor = int(rows[0].get("teacher_sc5_anchor", -1))
    teacher_sc5_valid = int(rows[0].get("teacher_sc5_corridor_valid", "0"))
    teacher_sc_start = int(rows[0].get("teacher_stable_carry_start", -1))
    is_butter = rows[0].get("is_butter", "False") in ("True", "true", "1")
    is_held_out = rows[0].get("is_held_out", "False") in ("True", "true", "1")

    state = "MONITORING"
    emit_step = -1
    predicted_sc_start = -1
    sc_start_confirmed = 0
    phase_history = []

    for t, row in enumerate(rows):
        # Extract features
        feats = []
        for fn in feature_names:
            try: feats.append(float(row.get(fn, 0)))
            except: feats.append(0.0)
        x = (torch.tensor(feats, dtype=torch.float32) - mean) / std

        with torch.no_grad():
            out = model(x.unsqueeze(0))

        phase_prob = torch.softmax(out["phase_logits"], dim=1)[0]
        phase_idx = phase_prob.argmax().item()
        phase_label = phase_classes[phase_idx]
        corridor_prob = torch.sigmoid(out["corridor_logit"]).item()
        release_prob = torch.sigmoid(out["release_logit"]).item()
        conf_prob = torch.sigmoid(out["confidence_logit"]).item()

        phase_history.append(phase_label)
        step_raw = int(row.get("step_idx", t))

        # SC5 trigger state machine
        stable_carry_prob = phase_prob[phase_classes.index("stable_carry")].item() if "stable_carry" in phase_classes else 0.0
        recovery_prob = phase_prob[phase_classes.index("recovery_or_regrasp")].item() if "recovery_or_regrasp" in phase_classes else 0.0

        if state == "MONITORING":
            if stable_carry_prob > 0.3:
                if predicted_sc_start < 0:
                    predicted_sc_start = step_raw
                sc_start_confirmed += 1
                if sc_start_confirmed >= 3:
                    state = "ARMED"
            else:
                predicted_sc_start = -1
                sc_start_confirmed = 0

        elif state == "ARMED":
            steps_since_sc = step_raw - predicted_sc_start
            if (steps_since_sc >= GUARD and corridor_prob >= TAU_CORRIDOR and
                release_prob < TAU_RELEASE and recovery_prob < 0.3 and
                conf_prob >= TAU_CONF):
                emit_step = step_raw
                state = "EMITTED"

        elif state == "EMITTED":
            state = "LOCKED"  # first-trigger lock

    # Metrics
    anchor_error = emit_step - teacher_anchor if emit_step > 0 and teacher_anchor > 0 else None
    abs_error = abs(anchor_error) if anchor_error is not None else None
    predicted_window_valid = False
    if emit_step > 0 and teacher_anchor > 0:
        # Check if predicted K10 window falls within Teacher SC5 corridor
        teacher_ws = teacher_anchor; teacher_we = teacher_anchor + K - 1
        predicted_ws = emit_step; predicted_we = emit_step + K - 1
        predicted_window_valid = (predicted_ws >= teacher_ws and predicted_we <= teacher_we)

    false_early = emit_step > 0 and teacher_sc_start > 0 and emit_step < teacher_sc_start
    post_release = emit_step > 0 and any(phase_classes[phase_history[min(i, len(phase_history)-1)].index(phase_history[min(i, len(phase_history)-1)])]  if False else False for i in range(len(phase_history)))  # simplified
    # Check if emit occurs during release_safe phase
    emit_during_release = emit_step > 0 and any(r.get("teacher_phase", "") == "release_safe" for r in rows[emit_step:emit_step+K] if int(r.get("step_idx",0)) >= emit_step)
    # Check if emit during recovery
    emit_during_recovery = emit_step > 0 and any(r.get("teacher_phase", "") == "recovery_or_regrasp" for r in rows[emit_step:emit_step+K] if int(r.get("step_idx",0)) >= emit_step)

    abstained = (emit_step < 0)

    results.append({
        "run_id": ek, "is_butter": is_butter, "is_held_out": is_held_out,
        "teacher_sc5_valid": teacher_sc5_valid, "teacher_anchor": teacher_anchor,
        "teacher_sc_start": teacher_sc_start,
        "predicted_sc_start": predicted_sc_start,
        "emit_step": emit_step, "anchor_error": anchor_error,
        "abs_error": abs_error, "predicted_window_valid": predicted_window_valid,
        "false_early": false_early, "emit_during_release": emit_during_release,
        "emit_during_recovery": emit_during_recovery,
        "abstained": abstained, "n_steps": len(rows),
    })

# --- Metrics ---
valid_eps = [r for r in results if r["teacher_sc5_valid"]]
no_corridor_eps = [r for r in results if not r["teacher_sc5_valid"]]
held_out_eps = [r for r in results if r["is_held_out"]]

emitted = [r for r in valid_eps if not r["abstained"]]
coverage = len([r for r in valid_eps if r["emit_step"] > 0]) / max(len(valid_eps), 1)
false_early_rate = sum(1 for r in valid_eps if r["false_early"]) / max(len(valid_eps), 1)
window_ok = sum(1 for r in emitted if r["predicted_window_valid"]) / max(len(emitted), 1)
abstain_rate = sum(1 for r in no_corridor_eps if r["abstained"]) / max(len(no_corridor_eps), 1)
median_abs_error = np.median([r["abs_error"] for r in emitted if r["abs_error"] is not None]) if emitted else float("nan")

print("=== SC5 Student Replay ===")
print(f"Episodes: {len(results)} (valid SC5: {len(valid_eps)}, no-corridor: {len(no_corridor_eps)})")
print(f"Held-out: {len(held_out_eps)}")
print(f"Coverage: {coverage:.3f} ({sum(1 for r in valid_eps if r['emit_step']>0)}/{len(valid_eps)})")
print(f"False-early: {false_early_rate:.3f}")
print(f"Window valid: {window_ok:.3f}")
print(f"Abstain (no-corridor): {abstain_rate:.3f}")
print(f"Median abs error: {median_abs_error:.1f} steps")
print(f"Emit during release: {sum(1 for r in emitted if r['emit_during_release'])}")
print(f"Emit during recovery: {sum(1 for r in emitted if r['emit_during_recovery'])}")

# Butter-specific
butter_valid = [r for r in valid_eps if r["is_butter"]]
butter_held = [r for r in valid_eps if r["is_held_out"]]
print(f"\nButter (all valid): {len(butter_valid)} episodes")
for r in butter_valid:
    print(f"  {r['run_id'][-40:]}: emit={r['emit_step']} teacher={r['teacher_anchor']} error={r['anchor_error']} held={r['is_held_out']}")

# Write
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "v2_sc5_replay_per_episode.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)

gate = {
    "coverage": coverage, "false_early_rate": false_early_rate,
    "window_ok": window_ok, "abstain_rate": abstain_rate,
    "median_abs_error": float(median_abs_error) if not np.isnan(median_abs_error) else None,
    "n_episodes": len(results), "n_valid_sc5": len(valid_eps),
    "n_emitted": len(emitted),
}
with open(os.path.join(OUTPUT_DIR, "v2_sc5_replay_gate.json"), "w") as f:
    json.dump(gate, f, indent=2, default=str)

print(f"\nGate summary: coverage={coverage:.3f} false_early={false_early_rate:.3f} window={window_ok:.3f} abs_error={median_abs_error}")
