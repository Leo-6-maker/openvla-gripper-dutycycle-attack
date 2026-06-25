#!/usr/bin/env python3
"""P0 NC Teacher relabel: classify 31 NC-assumed CLEAN shadow cells as A/B/C/D/U.
Uses available telemetry (obj_z, eef_obj_dist, task_success, 25D features).
"""
import csv, json, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

NC_CLEAN_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase6c_nc_clean_shadow"
NC_ATTACK_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase6c_nc_controls"
OUT_DIR = REPO / "evidence/phase6_gpu"


def load_telemetry(path):
    """Load step_telemetry.csv, return list of dicts sorted by step."""
    tel_path = Path(path) / "step_telemetry.csv"
    if not tel_path.exists():
        return None
    rows = list(csv.DictReader(open(tel_path)))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def load_summary(path):
    """Load episode_summary.json."""
    sum_path = Path(path) / "episode_summary.json"
    if not sum_path.exists():
        return {}
    with open(sum_path) as f:
        return json.load(f)


def detect_lift(rows, z_field="obj_z", threshold=0.015, sustain=2):
    """Detect if object was lifted. Returns dict with lifted, lift_start, max_delta."""
    result = {"lifted": False, "lift_start": -1, "max_delta": 0.0}
    z0 = None
    for r in rows:
        try:
            z0 = float(r.get(z_field, "nan"))
            if not np.isnan(z0):
                break
        except (ValueError, TypeError):
            continue
    if z0 is None:
        return result

    max_z = z0
    lift_count = 0
    lift_start = -1
    for r in rows:
        try:
            z = float(r.get(z_field, "nan"))
        except (ValueError, TypeError):
            continue
        if np.isnan(z):
            continue
        max_z = max(max_z, z)
        if z - z0 > threshold:
            lift_count += 1
            if lift_start < 0:
                lift_start = int(r.get("step", -1))
        else:
            lift_count = 0
            lift_start = -1

    result["lifted"] = lift_count >= sustain
    result["lift_start"] = lift_start
    result["max_delta"] = float(max_z - z0)
    return result


def detect_grasp(rows, dist_field="eef_obj_dist", dist_max=0.12, sustain=3):
    """Detect if grasp was established (EEF close to object)."""
    grasp_count = 0
    for r in rows:
        try:
            d = float(r.get(dist_field, "nan"))
        except (ValueError, TypeError):
            continue
        if np.isnan(d):
            continue
        if d <= dist_max:
            grasp_count += 1
        else:
            grasp_count = 0
    return grasp_count >= sustain


def compute_teacher_label(clean_rows, summary):
    """Compute teacher corridor validity from CLEAN telemetry.

    Uses available privileged signals (not detector output):
    - task_success: did the policy succeed?
    - obj_z lift: was the object lifted?
    - eef_obj_dist: was grasp established?
    - feat_valid: were 25D features valid?

    Returns dict with corridor_valid, anchor estimate, etc.
    """
    if clean_rows is None:
        return {"corridor_valid": False, "category": "U", "reason": "no_telemetry"}

    success = summary.get("task_success", summary.get("task_success_official", False))
    lift_info = detect_lift(clean_rows)
    lifted = lift_info["lifted"]
    lift_step = lift_info["lift_start"]
    lift_delta = lift_info["max_delta"]
    grasped = detect_grasp(clean_rows)

    # Check 25D feature validity
    feat_valid_steps = sum(1 for r in clean_rows if r.get("feat_valid") == "True")
    total_steps = len(clean_rows)

    # Corridor estimate from behavior (not detector):
    # corridor exists if: (lifted OR task_success) AND grasped
    corridor_valid = (lifted or success) and grasped

    # Estimate anchor: step where object lift stabilizes
    anchor_estimate = lift_step if lift_step >= 0 else -1

    # Classification
    emit_step = summary.get("mlp_emit", summary.get("mlp_emit_step", -1))
    if emit_step is None or emit_step == "":
        emit_step = -1
    emit_step = int(emit_step)
    v2_emitted = emit_step >= 0

    if not corridor_valid and not success:
        if v2_emitted:
            cat = "C"  # genuine NC false trigger
        else:
            cat = "D"  # correct abstain
    elif corridor_valid or success:
        if v2_emitted:
            cat = "A"  # true positive
        else:
            cat = "B"  # TV miss
    else:
        cat = "U"  # unresolved

    return {
        "corridor_valid": corridor_valid,
        "task_success": success,
        "lifted": lifted,
        "lift_delta": round(lift_delta, 4),
        "grasped": grasped,
        "feat_valid_steps": feat_valid_steps,
        "total_steps": total_steps,
        "anchor_estimate": anchor_estimate,
        "v2_emit_step": emit_step,
        "v2_emitted": v2_emitted,
        "category": cat,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect all NC CLEAN shadow cells (only those with completed telemetry)
    nc_base = Path(NC_CLEAN_DIR)
    cells = sorted([d.name for d in nc_base.iterdir()
                    if d.is_dir() and (d / "step_telemetry.csv").exists()])

    results = []
    for cell in cells:
        clean_path = nc_base / cell
        rows = load_telemetry(clean_path)
        summary = load_summary(clean_path)
        label = compute_teacher_label(rows, summary)
        label["cell_id"] = cell

        # Parse task/state from cell name
        parts = cell.split("_")
        if len(parts) >= 4:
            label["task"] = int(parts[2][1:]) if parts[2].startswith("t") else -1
            label["state"] = int(parts[3][1:]) if parts[3].startswith("s") else -1

        # Check if TRUE_T10 attacked version exists
        attack_path = Path(NC_ATTACK_DIR) / cell
        label["has_attacked"] = (attack_path / "step_telemetry.csv").exists()

        results.append(label)
        cat = label["category"]
        print(f"  {cell}: {cat} (lifted={label['lifted']} grasped={label['grasped']} "
              f"succ={label['task_success']} emit={label['v2_emit_step']})")

    # Summary
    cats = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r["cell_id"])

    print(f"\n=== RELABEL SUMMARY ===")
    for cat in ["A", "B", "C", "D", "U"]:
        print(f"  {cat}: {len(cats[cat])} cells {cats[cat][:5]}")

    # Save CSV
    csv_path = OUT_DIR / "NC_OFFICIAL_TEACHER_RELABEL.csv"
    fields = ["cell_id", "task", "state", "category", "corridor_valid",
              "task_success", "lifted", "lift_delta", "grasped",
              "v2_emitted", "v2_emit_step", "anchor_estimate",
              "feat_valid_steps", "total_steps", "has_attacked"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")

    # Save JSON
    json_path = OUT_DIR / "NC_OFFICIAL_TEACHER_RELABEL.json"
    summary_json = {
        "gate": "NC_OFFICIAL_TEACHER_RELABEL",
        "total_cells": len(results),
        "classification": {cat: len(cats[cat]) for cat in ["A", "B", "C", "D", "U"]},
        "genuine_nc_false_triggers": len(cats["C"]),
        "true_positives": len(cats["A"]),
        "tv_misses": len(cats["B"]),
        "correct_abstains": len(cats["D"]),
        "unresolved": len(cats["U"]),
        "per_cell": {r["cell_id"]: r for r in results},
    }
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # Print final verdict
    n_c = len(cats["C"])
    n_a = len(cats["A"])
    print(f"\n=== VERDICT ===")
    print(f"  Genuine NC false triggers (C): {n_c}")
    print(f"  True positives / TV correct (A): {n_a}")
    if n_c > 0:
        print(f"  ALERT: {n_c} genuine NC false triggers detected!")
    if len(cats["B"]) > 0:
        print(f"  TV misses (B): {len(cats['B'])}")


if __name__ == "__main__":
    main()
