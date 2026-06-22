#!/usr/bin/env python3
"""M0-R: Offline replay of frozen Object checkpoint on canonical dataset."""
import csv, json, os, sys, hashlib, numpy as np, torch
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES, SC5_PHASES

CKPT = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")
DATASET = os.path.join(REPO, "tables", "v2_sc5_canonical_dataset.csv")
OUT_DIR = os.path.join(REPO, "migration_audit", "object_checkpoint_migration", "m0_close")
os.makedirs(OUT_DIR, exist_ok=True)

# Verify dataset SHA
ds_sha = hashlib.sha256(open(DATASET, "rb").read()).hexdigest()
print(f"Dataset SHA: {ds_sha}")
print(f"Checkpoint expected: f942f4b0856d3449fa4e98f6d6e74ac8d5e8e9af7082373f961f79b0a6930cd9")
assert ds_sha == "f942f4b0856d3449fa4e98f6d6e74ac8d5e8e9af7082373f961f79b0a6930cd9", "DATASET SHA MISMATCH"
print("Dataset SHA: MATCH")

# Load checkpoint with strict detector
detector = SC5DetectorRuntime(CKPT, tau_corridor=0.3, tau_release=0.3, guard=5)
print(f"Detector loaded: cp_sha={detector.checkpoint_sha256[:16]} ds_sha={detector.dataset_sha256[:16]}")

# Load dataset and replay per episode
episodes = {}
with open(DATASET) as f:
    for row in csv.DictReader(f):
        eid = row.get("task_name", "?") + "_s" + row.get("state_id", "?")
        if eid not in episodes:
            episodes[eid] = {"rows": [], "is_held_out": row.get("is_held_out", "False")}
        episodes[eid]["rows"].append(row)

print(f"Episodes: {len(episodes)}")

# Replay metrics
all_steps = []
ep_metrics = []
total_coverage = 0; total_false_early = 0; total_post_rel = 0
total_k10 = 0; total_anchor_err = []; total_no_corr_abstain = 0
n_corridor_eps = 0; n_no_corridor_eps = 0

for eid, ep in sorted(episodes.items()):
    rows = sorted(ep["rows"], key=lambda r: int(r.get("step_idx", 0)))
    # Use frozen C16 teacher annotations
    # C16 corridor detection: anchor >= 0 means corridor episode
    teacher_anchor_any = -1
    for r in rows:
        ta = int(float(r.get("teacher_sc5_anchor", -1)))
        if ta >= 0:
            teacher_anchor_any = ta
            break
    has_corridor = teacher_anchor_any >= 0
    # Get anchor and sc_start from first valid values
    sc_start = -1
    for r in rows:
        ss = int(float(r.get("teacher_stable_carry_start", -1)))
        if ss >= 0:
            sc_start = ss
            break
    teacher_anchor = teacher_anchor_any
    teacher_phases = [r.get("teacher_phase", "?") for r in rows]

    detector.reset()
    emit_step = -1

    for row in rows:
        features_25d = {fn: float(row.get(fn, 0)) for fn in SC5_FEATURES}
        step = int(row.get("step_idx", 0))
        decision = detector.update(features_25d, step)
        all_steps.append({
            "episode": eid, "step": step,
            "state": decision["state"], "emit_step": decision["emit_step"],
            "corridor_p": decision.get("corridor_p", 0),
            "release_p": decision.get("release_p", 0),
            "pred_phase": decision.get("pred_phase", "?"),
            "teacher_phase": teacher_phases[step] if step < len(teacher_phases) else "?",
        })
        if decision.get("emitted"):
            emit_step = step

    if has_corridor:
        n_corridor_eps += 1
        if emit_step >= 0:
            total_coverage += 1
            if teacher_anchor >= 0:
                err = abs(emit_step - teacher_anchor)
                total_anchor_err.append(err)
                if err <= 10:
                    total_k10 += 1
            if sc_start >= 0 and emit_step < sc_start:
                total_false_early += 1
            # Post-release: emit on or after first release_safe step
            release_steps = [int(r.get("step_idx", 0)) for r in rows
                            if r.get("teacher_phase", "") == "release_safe"]
            if release_steps and emit_step >= min(release_steps):
                total_post_rel += 1
    else:
        n_no_corridor_eps += 1
        if emit_step < 0:
            total_no_corr_abstain += 1

    ep_metrics.append({
        "episode": eid, "has_corridor": has_corridor,
        "emit_step": emit_step, "is_held_out": ep["is_held_out"],
    })

# Compute metrics
n_corr = max(1, n_corridor_eps)
coverage = total_coverage / n_corr
false_early = total_false_early / n_corr
post_rel = total_post_rel / n_corr
k10 = total_k10 / n_corr
median_err = np.median(total_anchor_err) if total_anchor_err else 999
no_corr_abstain = total_no_corr_abstain / max(1, n_no_corridor_eps)

print(f"\n=== M0-R Offline Replay Results ===")
print(f"Episodes: {len(episodes)} (corridor={n_corridor_eps} no_corridor={n_no_corridor_eps})")
print(f"Coverage: {coverage:.3f} (C16: 0.873)")
print(f"False-early: {false_early:.3f} (C16: 0.025)")
print(f"Post-release: {post_rel:.3f} (C16: 0.000)")
print(f"K10 containment: {k10:.3f} (C16: 0.974)")
print(f"Median abs error: {median_err:.1f} (C16: 2.7)")
print(f"No-corridor abstain: {no_corr_abstain:.3f} (C16: 0.954)")

# Gate check
gates = {
    "coverage >= 0.80": coverage >= 0.80,
    "false_early <= 0.10": false_early <= 0.10,
    "post_release <= 0.05": post_rel <= 0.05,
    "K10 >= 0.85": k10 >= 0.85,
    "no_corridor_abstain >= 0.90": no_corr_abstain >= 0.90,
    "median_error <= 8": median_err <= 8,
}
all_pass = all(gates.values())
for k, v in gates.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")

print(f"\nM0_OFFLINE_REPLAY: {'PASS' if all_pass else 'FAIL'}")

# Save results
results = {
    "dataset_sha": ds_sha, "checkpoint_sha": detector.checkpoint_sha256,
    "n_episodes": len(episodes), "n_corridor": n_corridor_eps,
    "n_no_corridor": n_no_corridor_eps,
    "metrics": {
        "coverage": round(float(coverage), 4), "false_early": round(float(false_early), 4),
        "post_release": round(float(post_rel), 4), "k10_containment": round(float(k10), 4),
        "median_abs_error": round(float(median_err), 2),
        "no_corridor_abstain": round(float(no_corr_abstain), 4),
    },
    "gates": {k: bool(v) for k, v in gates.items()}, "all_pass": bool(all_pass),
}
json.dump(results, open(os.path.join(OUT_DIR, "offline_replay.json"), "w"), indent=2)

# Save step-level CSV
with open(os.path.join(OUT_DIR, "offline_replay_steps.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_steps[0].keys())
    w.writeheader()
    w.writerows(all_steps)

print(f"\nSaved: {OUT_DIR}")
sys.exit(0 if all_pass else 1)
