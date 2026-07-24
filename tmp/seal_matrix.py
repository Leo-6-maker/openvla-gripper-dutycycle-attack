#!/usr/bin/env python3
"""Seal current Phase 7 Object matrix before any supplementary runs."""
import csv, hashlib, json, os
from datetime import datetime
from pathlib import Path

BASE = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/attack_benchmark")
OUT = Path("/mnt/sdc/dty_user/openvla_attack/evidence/phase7_object/current_matrix_seal")
OUT.mkdir(parents=True, exist_ok=True)

V2_SHA = "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c"
COMMIT = "ace1876281a9ad6ed68e1229a6e17346356766e9"
BACKEND = "upstream_tf_jpeg"

PARENTS = [
    ("alphabet_soup_s0", 0, 0, "supplementary"),
    ("cream_cheese_s0", 1, 0, "primary"),
    ("salad_dressing_s0", 2, 0, "primary"),
    ("bbq_sauce_s0", 3, 0, "primary"),
    ("ketchup_s0", 4, 0, "primary"),
    ("tomato_sauce_s0", 5, 0, "primary"),
    ("butter_s0", 6, 0, "primary"),
    ("butter_s2", 6, 2, "primary"),
    ("milk_s4", 7, 4, "primary"),
    ("chocolate_pudding_s2", 8, 2, "primary"),
    ("orange_juice_s0", 9, 0, "primary"),
]

SEEDS = [42, 123, 456]
CONDITIONS = ["TRUE_T10", "RAND_T10"]
COND_DIR = {"TRUE_T10": "vis", "RAND_T10": "rand"}

rows = []
issues = []

for cell, task, state, subset in PARENTS:
    for seed in SEEDS:
        for cond in CONDITIONS:
            cd = COND_DIR[cond]
            d = BASE / f"{cell}_{cd}_s{seed}"

            row = {
                "cell_id": cell,
                "task_idx": task,
                "state_id": state,
                "seed": seed,
                "condition": cond,
                "subset": subset,
                "output_dir": str(d),
                "done": os.path.exists(d / ".done"),
                "summary_exists": os.path.exists(d / "episode_summary.json"),
                "telemetry_exists": os.path.exists(d / "step_telemetry.csv"),
            }

            if row["summary_exists"]:
                s = json.load(open(d / "episode_summary.json"))
                row["mlp_emit_step"] = s.get("mlp_emit_step", "MISSING")
                row["task_success"] = s.get("task_success", "MISSING")
                row["attack_frames"] = s.get("attack_frames", "MISSING")
                row["token_open_duty"] = s.get("token_open_duty", "MISSING")
                row["env_open_duty"] = s.get("env_open_duty", "MISSING")
                row["checkpoint_sha16"] = s.get("checkpoint_sha256", "")[:16]
                row["backend"] = s.get("preprocess_backend_resolved", "")
                row["invalid_feature_steps"] = s.get("invalid_feature_steps", "MISSING")
                row["manual_anchor"] = s.get("manual_anchor_used", "MISSING")
                row["privileged_input"] = s.get("privileged_detector_input_used", "MISSING")

                # Audit checks
                ckpt = s.get("checkpoint_sha256", "")
                if ckpt != V2_SHA:
                    issues.append(f"{cell}_{cond}_s{seed}: SHA mismatch")
                if s.get("preprocess_backend_resolved", "") != BACKEND:
                    issues.append(f"{cell}_{cond}_s{seed}: backend mismatch")
                if s.get("invalid_feature_steps", 0) != 0:
                    issues.append(f"{cell}_{cond}_s{seed}: {s.get('invalid_feature_steps')} invalid steps")
                if s.get("manual_anchor_used", False):
                    issues.append(f"{cell}_{cond}_s{seed}: manual_anchor=True")
                if s.get("privileged_detector_input_used", False):
                    issues.append(f"{cell}_{cond}_s{seed}: privileged_input=True")
                # Attack frames check
                emit = s.get("mlp_emit_step", -1)
                if emit is None or emit == "":
                    emit = -1
                try:
                    emit = int(emit)
                except (ValueError, TypeError):
                    emit = -1
                expected_atk = 10 if emit >= 0 else 0
                actual_atk = s.get("attack_frames", -1)
                if actual_atk is None:
                    actual_atk = -1
                if actual_atk != expected_atk:
                    issues.append(f"{cell}_{cond}_s{seed}: attack_frames={actual_atk} expected={expected_atk}")

            rows.append(row)

vis_done = sum(1 for r in rows if r["condition"] == "TRUE_T10" and r["done"])
rand_done = sum(1 for r in rows if r["condition"] == "RAND_T10" and r["done"])
vis_planned = sum(1 for r in rows if r["condition"] == "TRUE_T10")
rand_planned = sum(1 for r in rows if r["condition"] == "RAND_T10")

seal_sha = hashlib.sha256(
    json.dumps([{k: str(v) for k, v in r.items()} for r in rows], sort_keys=True).encode()
).hexdigest()

seal = {
    "gate": "PHASE7_OBJECT_CURRENT_MATRIX_SEAL",
    "timestamp": datetime.now().isoformat(),
    "execution_commit": COMMIT,
    "v2_checkpoint_sha": V2_SHA,
    "backend": BACKEND,
    "vis_planned": vis_planned,
    "vis_done": vis_done,
    "vis_complete": vis_done == vis_planned,
    "rand_planned": rand_planned,
    "rand_done": rand_done,
    "rand_complete": rand_done == rand_planned,
    "total_runs_sealed": len(rows),
    "total_done": sum(1 for r in rows if r["done"]),
    "issues_found": len(issues),
    "issues": issues,
    "seal_sha256": seal_sha,
    "claim_boundary": (
        "matched independent rollouts with audited pre-trigger feature parity; "
        "NOT exact-prefix counterfactual branches"
    ),
    "denominator_note": (
        "11 Object cells x 3 perturbation seeds = 33 per condition; "
        "2 no-emit cells (cream_cheese_s0, chocolate_pudding_s2)"
    ),
}

with open(OUT / "PHASE7_CURRENT_MATRIX_SEAL.json", "w") as f:
    json.dump(seal, f, indent=2, default=str)

with open(OUT / "PHASE7_CURRENT_MATRIX_RUNS.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"VIS: {vis_done}/{vis_planned} done (complete={vis_done==vis_planned})")
print(f"RAND: {rand_done}/{rand_planned} done (complete={rand_done==rand_planned})")
print(f"Issues: {len(issues)}")
for i in issues:
    print(f"  ISSUE: {i}")
print(f"Seal SHA: {seal_sha[:16]}")
print(f"Output: {OUT}")
