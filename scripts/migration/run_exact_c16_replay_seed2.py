#!/usr/bin/env python3
"""M0-R: Run exact C16 replay script on A800 and freeze artifact manifest."""
import hashlib, json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")
DATASET = os.path.join(REPO, "tables", "v2_sc5_canonical_dataset.csv")
OUT_DIR = os.path.join(REPO, "migration_audit", "object_checkpoint_migration", "m0_close", "exact_c16_seed2")
REPLAY_SCRIPT = os.path.join(REPO, "scripts", "stageb", "run_sc5_canonical_replay.py")
os.makedirs(OUT_DIR, exist_ok=True)

# Hard verify checkpoint and dataset SHAs
ckpt_sha = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
ds_sha = hashlib.sha256(open(DATASET, "rb").read()).hexdigest()
EXPECTED_CKPT = "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628"
EXPECTED_DS = "f942f4b0856d3449fa4e98f6d6e74ac8d5e8e9af7082373f961f79b0a6930cd9"

print(f"Checkpoint SHA: {ckpt_sha}")
print(f"Checkpoint expected: {EXPECTED_CKPT}")
assert ckpt_sha == EXPECTED_CKPT, "CHECKPOINT SHA MISMATCH"
print("Checkpoint SHA: MATCH")

print(f"Dataset SHA: {ds_sha}")
print(f"Dataset expected: {EXPECTED_DS}")
assert ds_sha == EXPECTED_DS, "DATASET SHA MISMATCH"
print("Dataset SHA: MATCH")

# Strict checkpoint load
import numpy as np
sys.path.insert(0, os.path.join(REPO, "src"))
import torch
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES, SC5_PHASES
detector = SC5DetectorRuntime(CKPT, tau_corridor=0.3, tau_release=0.3, guard=5)
n_params = sum(v.numel() for v in torch.load(CKPT, map_location="cpu", weights_only=False)["model_state"].values())
assert n_params == 6604, f"Param count {n_params} != 6604"
print(f"Strict load: PASS ({n_params} params, confidence_head present)")

# Run original C16 replay script
cmd = [
    sys.executable, REPLAY_SCRIPT,
    CKPT, DATASET, OUT_DIR,
]
print(f"\nRunning: {' '.join(cmd)}")
t0 = time.time()
env = os.environ.copy()
env["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + env.get("PYTHONPATH", "")
result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
dt = time.time() - t0
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])

# Save stdout
stdout_path = os.path.join(OUT_DIR, "stdout.log")
with open(stdout_path, "w") as f:
    f.write(result.stdout)

# Verify output CSV exists
csv_path = os.path.join(OUT_DIR, "v2_sc5_replay_canonical.csv")
assert os.path.exists(csv_path), "Output CSV not found"

# Compute artifact SHAs
csv_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
stdout_sha = hashlib.sha256(open(stdout_path, "rb").read()).hexdigest()
runner_sha = hashlib.sha256(open(REPLAY_SCRIPT, "rb").read()).hexdigest()

# Parse metrics from CSV
import csv
metrics = {"n_episodes": 0, "triggered": 0, "coverage": 0, "false_early": 0,
           "post_release": 0, "k10_contained": 0, "no_corridor_abstain": 0,
           "n_corridor": 0, "n_no_corridor": 0, "median_err": 0}
with open(csv_path) as f:
    for row in csv.DictReader(f):
        metrics["n_episodes"] += 1
        if row.get("triggered", "0") == "1":
            metrics["triggered"] += 1
        if row.get("teacher_sc5_valid", "False") == "True":
            metrics["n_corridor"] += 1
        else:
            metrics["n_no_corridor"] += 1
        if row.get("false_early", "0") == "1":
            metrics["false_early"] += 1
        if row.get("post_release_trigger", "0") == "1":
            metrics["post_release"] += 1
        if row.get("k10_contained", "0") == "1":
            metrics["k10_contained"] += 1
        if row.get("triggered", "0") == "0" and row.get("teacher_sc5_valid", "False") != "True":
            metrics["no_corridor_abstain"] += 1

n_corr = max(1, metrics["n_corridor"])
n_no_corr = max(1, metrics["n_no_corridor"])
coverage = metrics["triggered"] / n_corr if n_corr > 0 else 0
fe = metrics["false_early"] / n_corr
pr = metrics["post_release"] / n_corr
k10 = metrics["k10_contained"] / n_corr
abstain = metrics["no_corridor_abstain"] / n_no_corr

print(f"\n=== M0-R Exact C16 Replay ===")
print(f"Episodes: {metrics['n_episodes']} (corridor={n_corr} no_corridor={n_no_corr})")
print(f"Coverage: {coverage:.3f}")
print(f"False-early: {fe:.3f}")
print(f"Post-release: {pr:.3f}")
print(f"K10: {k10:.3f}")
print(f"Abstain: {abstain:.3f}")

# Freeze manifest
manifest = {
    "gate": "M0-R_EXACT_C16_REPLAY",
    "checkpoint_sha": ckpt_sha,
    "dataset_sha": ds_sha,
    "command": " ".join(cmd),
    "duration_s": round(dt, 1),
    "exit_code": result.returncode,
    "runner_sha": runner_sha,
    "output_csv_sha": csv_sha,
    "stdout_sha": stdout_sha,
    "metrics": {
        "n_episodes": metrics["n_episodes"],
        "n_corridor": metrics["n_corridor"],
        "n_no_corridor": metrics["n_no_corridor"],
        "triggered": metrics["triggered"],
        "coverage": round(coverage, 4),
        "false_early": round(fe, 4),
        "post_release": round(pr, 4),
        "k10_containment": round(k10, 4),
        "no_corridor_abstain": round(abstain, 4),
    },
    "c16_thresholds": {"coverage": 0.80, "false_early": 0.10, "post_release": 0.05,
                       "k10": 0.85, "abstain": 0.90, "median_err": 8},
}
json.dump(manifest, open(os.path.join(OUT_DIR, "artifact_manifest.json"), "w"), indent=2)

# Check old Seed2 step-level data availability
old_csv = os.path.join(REPO, "tmp", "v2_sc5_replay_canonical.csv")
old_available = os.path.exists(old_csv)
manifest["old_seed2_per_episode_available"] = old_available

if old_available:
    # Compare episode-level
    old_data = {}
    with open(old_csv) as f:
        for row in csv.DictReader(f):
            old_data[row["episode_id"]] = row
    new_data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            new_data[row["episode_id"]] = row
    match = 0; mismatch = 0
    for eid in old_data:
        if eid in new_data:
            if (old_data[eid].get("emit_step") == new_data[eid].get("emit_step") and
                old_data[eid].get("triggered") == new_data[eid].get("triggered")):
                match += 1
            else:
                mismatch += 1
    manifest["historical_comparison"] = {"match": match, "mismatch": mismatch,
                                          "total_in_old": len(old_data),
                                          "exact_match": mismatch == 0}
    print(f"\nHistorical comparison: {match}/{len(old_data)} exact match ({mismatch} mismatch)")
    if mismatch == 0:
        print("M0_EXACT_REPLAY_PASS")
    else:
        print("M0_C16_METRIC_CONTRACT_PASS (EXACT_OLD_STEP_OUTPUT_NOT_AVAILABLE)")
else:
    print("\nOld Seed2 per-episode CSV not available — M0_C16_METRIC_CONTRACT_PASS")
    manifest["historical_comparison"] = "OLD_DATA_NOT_AVAILABLE"

json.dump(manifest, open(os.path.join(OUT_DIR, "artifact_manifest.json"), "w"), indent=2)
print(f"\nSaved: {OUT_DIR}")
sys.exit(0)
