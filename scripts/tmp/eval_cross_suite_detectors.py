#!/usr/bin/env python3
"""Cross-suite detector formal evaluation — 7 checkpoints on CLEAN2000 teacher labels.

Metrics: event precision/recall/F1, timing MAE/P90, per-suite breakdown.
Runs detector state machine per episode until emit or episode end (efficient).
"""
import json, os, sys, csv, hashlib, time
import numpy as np

REPO = "/mnt/sdc/dty_user/openvla_attack"
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/scripts")

import torch
from gripper_attack.sc5_detector_runtime import SC5MLP, SC5_FEATURES
from collections import defaultdict

CKPT_ROOT = f"{REPO}/outputs/cross_suite_detector_v1"
FEATURES_CSV = f"{REPO}/evidence/CLEAN2000_CANONICAL_V1/CLEAN2000_FEATURES_25D_VALID_ONLY.csv"
TEACHER_CSV = f"{REPO}/evidence/CLEAN2000_SUPERVISION_AUTH_V1_2/TEACHER_STEP_LABELS.csv"
OUT = f"{REPO}/evidence/CROSS_SUITE_DETECTOR_V1"
os.makedirs(OUT, exist_ok=True)

# ── Load teacher: episode_key -> corridor_start_step (first step where corridor_active=1) ──
print("Loading teacher labels...")
teacher_corridor_start = {}  # episode_key -> step
teacher_no_corridor = set()   # episodes with no corridor
with open(TEACHER_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["episode_key"]
        if ep in teacher_corridor_start:
            continue  # already found start
        if int(row["teacher_sc5_corridor_active"]) == 1:
            teacher_corridor_start[ep] = int(row["step"])
        # If we get past all rows without corridor, it stays unset

# Track episodes without corridor
all_teacher_eps = set()
with open(TEACHER_CSV) as f:
    for row in csv.DictReader(f):
        all_teacher_eps.add(row["episode_key"])

teacher_no_corridor = all_teacher_eps - set(teacher_corridor_start.keys())
print(f"Teacher: {len(all_teacher_eps)} episodes, {len(teacher_corridor_start)} with corridor, {len(teacher_no_corridor)} without")

# ── Load features grouped by episode_key ──
print("Loading features...")
features_by_ep = defaultdict(list)
with open(FEATURES_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ep = row["episode_key"]
        step = int(row["step"])
        feat = {fn: float(row[fn]) for fn in SC5_FEATURES}
        features_by_ep[ep].append((step, feat))

# Sort steps within each episode
for ep in features_by_ep:
    features_by_ep[ep].sort(key=lambda x: x[0])

print(f"Features: {len(features_by_ep)} episodes, {sum(len(v) for v in features_by_ep.values())} total steps")

# ── Detector evaluator ──
class DetEvaluator:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = SC5MLP(n_feat=len(SC5_FEATURES))
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.model.eval()
        self.mean = np.array(ckpt["mean"], dtype=np.float32)
        self.std = np.array(ckpt["std"], dtype=np.float32)
        self.tau_c = ckpt.get("tau_corridor", 0.3)
        self.tau_r = ckpt.get("tau_release", 0.3)
        self.guard = ckpt.get("guard", 5)
        self.split_mode = ckpt.get("split_mode", "unknown")

    def run_episode(self, steps_features):
        """Run state machine until emit or episode end. Returns emit_step or -1."""
        state = "IDLE"; corridor_streak = 0; release_streak = 0
        for step, feat in steps_features:
            X = np.array([[feat[f] for f in SC5_FEATURES]], dtype=np.float32)
            X_norm = (X - self.mean) / (self.std + 1e-8)
            with torch.no_grad():
                out = self.model(torch.tensor(X_norm, dtype=torch.float32))
            cp = float(1.0 / (1.0 + np.exp(-out["corridor_logit"].numpy()[0, 0])))
            rp = float(1.0 / (1.0 + np.exp(-out["release_logit"].numpy()[0, 0])))
            corridor_active = cp > self.tau_c
            release_safe = rp > self.tau_r

            if state == "IDLE":
                if corridor_active:
                    corridor_streak += 1
                    if corridor_streak >= self.guard:
                        return step  # emit at this step
                else:
                    corridor_streak = 0
            elif state == "CORRIDOR":
                if release_safe:
                    release_streak += 1
                    if release_streak >= self.guard:
                        state = "RELEASE"
                else:
                    release_streak = 0
        return -1  # no emission


# ── Evaluate all checkpoints ──
CKPTS = [
    ("pooled_seed1", f"{CKPT_ROOT}/pooled/seed_1/best_model.pt", "pooled"),
    ("pooled_seed2", f"{CKPT_ROOT}/pooled/seed_2/best_model.pt", "pooled"),
    ("pooled_seed3", f"{CKPT_ROOT}/pooled/seed_3/best_model.pt", "pooled"),
    ("loso_OBJECT", f"{CKPT_ROOT}/loso/hold_LIBERO_OBJECT/best_model.pt", "loso_OBJECT"),
    ("loso_SPATIAL", f"{CKPT_ROOT}/loso/hold_LIBERO_SPATIAL/best_model.pt", "loso_SPATIAL"),
    ("loso_GOAL", f"{CKPT_ROOT}/loso/hold_LIBERO_GOAL/best_model.pt", "loso_GOAL"),
    ("loso_L10", f"{CKPT_ROOT}/loso/hold_LIBERO_10/best_model.pt", "loso_L10"),
]

# Suite mapping from episode_key
def get_suite(ep_key):
    suite = ep_key.split("/")[0]
    if "object" in suite: return "OBJECT"
    if "spatial" in suite: return "SPATIAL"
    if "goal" in suite: return "GOAL"
    return "L10"

results = {}

for name, ckpt_path, mode in CKPTS:
    print(f"\n{'='*60}")
    print(f"Evaluating {name} ({mode})...")
    if not os.path.exists(ckpt_path):
        print(f"  SKIP — not found")
        continue

    det = DetEvaluator(ckpt_path)
    print(f"  Loaded: split={det.split_mode}")

    # Per-suite metrics
    suite_m = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "timing_errors": []})
    total_tp, total_fp, total_fn = 0, 0, 0
    all_timing_errors = []

    for ep_key in features_by_ep:
        suite = get_suite(ep_key)
        steps_feat = features_by_ep[ep_key]
        pred_emit = det.run_episode(steps_feat)
        teacher_emit = teacher_corridor_start.get(ep_key, -1)

        if pred_emit >= 0 and teacher_emit >= 0:
            total_tp += 1; suite_m[suite]["tp"] += 1
            err = abs(pred_emit - teacher_emit)
            all_timing_errors.append(err); suite_m[suite]["timing_errors"].append(err)
        elif pred_emit >= 0 and teacher_emit < 0:
            total_fp += 1; suite_m[suite]["fp"] += 1
        elif pred_emit < 0 and teacher_emit >= 0:
            total_fn += 1; suite_m[suite]["fn"] += 1

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    mae = float(np.mean(all_timing_errors)) if all_timing_errors else -1
    p90 = float(np.percentile(all_timing_errors, 90)) if all_timing_errors else -1

    print(f"  Events: TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Global: P={precision:.4f} R={recall:.4f} F1={f1:.4f} MAE={mae:.1f} P90={p90:.1f}")

    suite_summary = {}
    for suite in ["OBJECT", "SPATIAL", "GOAL", "L10"]:
        m = suite_m[suite]
        p = m["tp"] / max(1, m["tp"] + m["fp"])
        r = m["tp"] / max(1, m["tp"] + m["fn"])
        f = 2 * p * r / max(1e-8, p + r)
        smae = float(np.mean(m["timing_errors"])) if m["timing_errors"] else -1
        suite_summary[suite] = {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                                 "precision": round(p, 4), "recall": round(r, 4),
                                 "f1": round(f, 4), "timing_mae": round(smae, 1) if smae >= 0 else None}
        if m["tp"] + m["fp"] + m["fn"] > 0:
            print(f"  Suite {suite}: P={p:.3f} R={r:.3f} F1={f:.3f} MAE={smae:.1f}")

    results[name] = {
        "checkpoint": ckpt_path, "split_mode": det.split_mode,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "timing_mae": round(mae, 1) if mae >= 0 else None,
        "timing_p90": round(p90, 1) if p90 >= 0 else None,
        "suite_metrics": suite_summary,
    }

# ── Summary ──
print(f"\n{'='*70}")
print(f"{'Checkpoint':<20} {'Split':<14} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7} {'MAE':>6} {'P90':>6}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<20} {r['split_mode']:<14} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
          f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} "
          f"{r['timing_mae'] or -1:>6.1f} {r['timing_p90'] or -1:>6.1f}")

# ── Write outputs ──
envelope = {
    "gate": "CROSS_SUITE_DETECTOR_EVALUATION_V1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "features_csv": FEATURES_CSV,
    "teacher_csv": TEACHER_CSV,
    "teacher_total_eps": len(all_teacher_eps),
    "teacher_corridor_eps": len(teacher_corridor_start),
    "teacher_no_corridor_eps": len(teacher_no_corridor),
    "results": results,
}
with open(os.path.join(OUT, "EVALUATION_RESULTS.json"), "w") as f:
    json.dump(envelope, f, indent=2)

with open(os.path.join(OUT, "EVALUATION_TABLE.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["checkpoint", "split_mode", "tp", "fp", "fn", "precision", "recall", "f1", "timing_mae", "timing_p90"])
    for name, r in results.items():
        w.writerow([name, r["split_mode"], r["tp"], r["fp"], r["fn"],
                    r["precision"], r["recall"], r["f1"], r["timing_mae"], r["timing_p90"]])

print(f"\nResults saved to {OUT}/")
print("Done.")
