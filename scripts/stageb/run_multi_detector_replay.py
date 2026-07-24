#!/usr/bin/env python3
"""P0 multi-detector replay: replay V1/M1/M1-OS/M2 on same CLEAN shadow 25D features."""
import csv, json, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime, SC5_FEATURES

TAU_C, TAU_R, GUARD = 0.3, 0.3, 5

DETECTORS = {
    "V1": "/mnt/sdc/dty_user/openvla_attack/artifacts/detector/sc5_mlp_s2.pt",
    "M1": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_ablation_primary_seed42/sc5_mlp_v2.pt",
    "M1_OS": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_ablation_oversampled_seed42/sc5_mlp_v2.pt",
    "M2": "/mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt",
}

NC_CLEAN_DIR = "/mnt/sdc/dty_user/openvla_attack/evidence/m1c/phase6c_nc_clean_shadow"
RELABEL_CSV = str(REPO / "evidence/phase6_gpu/NC_OFFICIAL_TEACHER_RELABEL.csv")


def replay_detector(rt, rows):
    """Replay detector on step_telemetry rows. Returns emit/arm/prob info."""
    rt.reset()
    arm_step = -1; emit_step = -1
    max_cp = 0.0; max_cp_step = -1
    cp_streak = 0; max_streak = 0
    phase_at_max_cp = "?"
    phase_at_emit = "?"
    rp_at_emit = 0.0

    for r in rows:
        feats = {}
        ok = True
        for fn in SC5_FEATURES:
            # Features stored with f_ prefix in telemetry
            val = r.get(f"f_{fn}", r.get(fn, ""))
            if val in ("", "nan", "NaN", None):
                ok = False; break
            try:
                feats[fn] = float(val)
            except (ValueError, TypeError):
                ok = False; break
        if not ok:
            continue

        x = np.array([feats[fn] for fn in SC5_FEATURES], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            continue

        step = int(r.get("step", 0))
        dec = rt.update({fn: float(x[i]) for i, fn in enumerate(SC5_FEATURES)}, step)

        cp = dec.get("corridor_p", 0)
        if cp is not None and not np.isnan(cp) and cp > max_cp:
            max_cp = cp; max_cp_step = step
            phase_at_max_cp = dec.get("pred_phase", "?")

        if cp is not None and not np.isnan(cp):
            if cp > TAU_C:
                cp_streak += 1
                max_streak = max(max_streak, cp_streak)
            else:
                cp_streak = 0

        if rt.state == "ARMED" and arm_step < 0:
            arm_step = step
        if dec.get("emitted"):
            emit_step = step
            phase_at_emit = dec.get("pred_phase", "?")
            rp_at_emit = dec.get("release_p", 0)
            break

    return {
        "armed": arm_step >= 0, "emitted": rt.emitted,
        "arm_step": arm_step, "emit_step": emit_step,
        "max_corridor_p": round(max_cp, 6), "max_corridor_step": max_cp_step,
        "max_corridor_streak": max_streak,
        "phase_at_max_cp": phase_at_max_cp,
        "phase_at_emit": phase_at_emit,
        "release_p_at_emit": round(rp_at_emit, 6) if rp_at_emit else 0,
    }


def main():
    # Load relabel data
    relabel = {}
    if os.path.exists(RELABEL_CSV):
        for r in csv.DictReader(open(RELABEL_CSV)):
            relabel[r["cell_id"]] = r

    # Load runtimes
    print("Loading detectors...")
    runtimes = {}
    for name, path in DETECTORS.items():
        if os.path.exists(path):
            runtimes[name] = SC5DetectorRuntime(path, tau_corridor=TAU_C, tau_release=TAU_R, guard=GUARD)
            print(f"  {name}: {runtimes[name].checkpoint_sha256[:16]}")
        else:
            print(f"  {name}: MISSING ({path})")

    # Replay all cells
    nc_base = Path(NC_CLEAN_DIR)
    cells = sorted([d.name for d in nc_base.iterdir()
                    if d.is_dir() and (d / "step_telemetry.csv").exists()])

    print(f"\nReplaying {len(cells)} cells with {len(runtimes)} detectors...")
    results = []

    for cell in cells:
        tel_path = nc_base / cell / "step_telemetry.csv"
        rows = list(csv.DictReader(open(tel_path)))
        rows.sort(key=lambda r: int(r.get("step", 0)))

        rl = relabel.get(cell, {})
        row = {
            "cell_id": cell,
            "teacher_category": rl.get("category", "?"),
            "corridor_valid": rl.get("corridor_valid", "?"),
            "task_success": rl.get("task_success", "?"),
        }

        for name, rt in runtimes.items():
            res = replay_detector(rt, rows)
            row[f"{name}_emitted"] = res["emitted"]
            row[f"{name}_emit_step"] = res["emit_step"]
            row[f"{name}_armed"] = res["armed"]
            row[f"{name}_arm_step"] = res["arm_step"]
            row[f"{name}_max_cp"] = res["max_corridor_p"]
            row[f"{name}_max_streak"] = res["max_corridor_streak"]
            row[f"{name}_phase_emit"] = res["phase_at_emit"]
            row[f"{name}_rp_emit"] = res["release_p_at_emit"]

        results.append(row)

    # Save matrix CSV
    csv_path = REPO / "evidence/phase6_gpu/NC_SAME_TRAJECTORY_DETECTOR_MATRIX.csv"
    fields = ["cell_id", "teacher_category", "corridor_valid", "task_success"]
    for name in DETECTORS:
        fields += [f"{name}_emitted", f"{name}_emit_step", f"{name}_armed",
                   f"{name}_max_cp", f"{name}_max_streak", f"{name}_phase_emit", f"{name}_rp_emit"]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")

    # Analysis
    print(f"\n=== MULTI-DETECTOR COMPARISON ===")
    # Per-category analysis
    for cat_name, cat_label in [("C", "Genuine NC false triggers"), ("D", "Correct abstains"),
                                  ("A", "True positives"), ("B", "TV misses")]:
        cat_cells = [r for r in results if r["teacher_category"] == cat_name]
        if not cat_cells:
            continue
        print(f"\n{cat_label} ({len(cat_cells)} cells):")
        for name in DETECTORS:
            emits = sum(1 for r in cat_cells if r.get(f"{name}_emitted"))
            print(f"  {name}: {emits}/{len(cat_cells)} emits")

    # Paired comparison: M1 vs M2 on all cells
    print(f"\n=== M1 vs M2 PAIRED ===")
    m1_m2 = {"both_emit": 0, "both_abstain": 0, "m1_only": 0, "m2_only": 0}
    for r in results:
        m1e = r.get("M1_emitted", False)
        m2e = r.get("M2_emitted", False)
        if m1e and m2e: m1_m2["both_emit"] += 1
        elif not m1e and not m2e: m1_m2["both_abstain"] += 1
        elif m1e and not m2e: m1_m2["m1_only"] += 1
        elif not m1e and m2e: m1_m2["m2_only"] += 1
    for k, v in m1_m2.items():
        print(f"  {k}: {v}")

    # Save JSON
    json_path = REPO / "evidence/phase6_gpu/NC_SAME_TRAJECTORY_DETECTOR_MATRIX.json"
    summary = {
        "gate": "NC_SAME_TRAJECTORY_DETECTOR_MATRIX",
        "total_cells": len(results),
        "detectors": list(DETECTORS.keys()),
        "paired_m1_vs_m2": m1_m2,
        "per_cell": {r["cell_id"]: r for r in results},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
