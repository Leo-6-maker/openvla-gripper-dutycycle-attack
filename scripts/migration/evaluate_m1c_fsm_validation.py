#!/usr/bin/env python3
"""P5: R1/R2 independent validation evaluation.

Replays frozen R1 and 72 R2 configurations on validation telemetry.
Uses pre-computed teacher labels for ground truth.
Outputs per-config metrics, lexicographic selection, and CI.
"""
import os, sys, json, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

R1_FREEZE_PATH = REPO / "migration_audit/m1c/R1_RUNTIME_BASELINE_FROZEN_V2.json"
R2_GRID_PATH = REPO / "migration_audit/m1c/r2_validation_grid.json"
CORPUS_ROOT = REPO / "evidence/m1c/object_clean_corpus"
LABELS_PATH = REPO / "evidence/m1c/object_teacher_labels_v1/per_cell_teacher_labels.csv"
CKPT_PATH = REPO / "artifacts/detector/sc5_mlp_s2.pt"

SIX_GATES = {
    "coverage": 0.80, "false_early": 0.10, "post_release": 0.05,
    "k10_containment": 0.85, "median_anchor_error": 8, "no_corridor_abstain": 0.90,
    "feature_valid_rate": 0.99, "silent_stalls": 0,
}


def sha256(p):
    if not p.exists(): return "MISSING"
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()


def wilson_ci(numerator, denominator, z=1.96):
    if denominator == 0: return (0, 0)
    p = numerator / denominator
    d = z * np.sqrt(p * (1 - p) / denominator + z**2 / (4 * denominator**2))
    lo = (p + z**2 / (2 * denominator) - d) / (1 + z**2 / denominator)
    hi = (p + z**2 / (2 * denominator) + d) / (1 + z**2 / denominator)
    return max(0, lo), min(1, hi)


CKPT = None  # set by main

def evaluate_fsm(detector_cls, config, val_cells, teacher_labels):
    """Replay FSM on validation telemetry and compute all metrics."""
    n_tv = 0; n_nc = 0; triggered = []
    silent_stalls = 0; total_disarms = 0

    for cell in val_cells:
        key = (cell["task"], cell["state"])
        tl = teacher_labels.get(key, {})
        tv = tl.get("teacher_valid", False)
        anchor = tl.get("teacher_anchor", -1)
        if tv: n_tv += 1
        else: n_nc += 1

        d = detector_cls(str(CKPT), **config)
        tel = cell["path"] / "step_telemetry.csv"
        if not tel.exists(): continue
        rows = list(csv.DictReader(open(tel)))

        emit_step = None
        first_arm = -1
        n_disarm = 0
        for step, r in enumerate(rows):
            cp_s = r.get("corridor_p", ""); rp_s = r.get("release_p", "")
            pp = r.get("pred_phase", ""); fv = r.get("feat_valid", "") == "True"
            cp = float(cp_s) if cp_s and cp_s != "" else float("nan")
            rp = float(rp_s) if rp_s and rp_s != "" else float("nan")
            dec = d.update_from_scores(cp, rp, pp, step, feat_valid=fv)
            if d.state == "ARMED" and first_arm < 0: first_arm = step
            if dec.get("disarm_count", 0) > n_disarm: n_disarm = dec["disarm_count"]
            if d.emitted and emit_step is None: emit_step = step
        if d.state == "ARMED" and not d.emitted: silent_stalls += 1
        total_disarms += n_disarm

        if emit_step is not None:
            age = abs(emit_step - anchor) if anchor >= 0 else None
            triggered.append({
                "tv": tv, "anchor": anchor, "emit": emit_step, "age": age,
                "k10": (anchor <= emit_step < anchor + 10) if (tv and anchor >= 0) else None,
                "post_release": (emit_step >= anchor + 10) if (tv and anchor >= 0) else None,
                "false_early": (emit_step < anchor) if anchor >= 0 else None,
            })
        total_disarms += n_disarm

    # Episode-level aggregation — each cell contributes at most one event
    tv_triggered = [t for t in triggered if t["tv"]]
    nc_triggered = [t for t in triggered if not t["tv"]]
    n_tv_trig = len(tv_triggered)
    n_nc_trig = len(nc_triggered)
    n_all_trig = len(triggered)

    coverage = n_tv_trig / n_tv if n_tv > 0 else 0
    k10_ok = sum(1 for t in tv_triggered if t.get("k10") is True)
    k10_rate = k10_ok / len(tv_triggered) if tv_triggered else 0
    ages = [t["age"] for t in tv_triggered if t.get("age") is not None]
    median_age = float(np.median(ages)) if ages else -1
    fe_count = sum(1 for t in triggered if t.get("false_early") is True)
    fe_rate = fe_count / n_all_trig if n_all_trig else 0
    pr_count = sum(1 for t in triggered if t.get("post_release") is True)
    pr_rate = pr_count / n_all_trig if n_all_trig else 0
    nc_abstained = n_nc - n_nc_trig
    nc_rate = nc_abstained / n_nc if n_nc > 0 else 0

    # Assertions
    assert 0.0 <= coverage <= 1.0, f"coverage={coverage} out of [0,1]"
    assert 0.0 <= nc_rate <= 1.0, f"nc_rate={nc_rate} out of [0,1]"
    assert 0.0 <= k10_rate <= 1.0, f"k10={k10_rate} out of [0,1]"
    # Feature valid
    fv_ok = sum(1 for c in val_cells if c.get("fv_ok", True))
    fv_rate = fv_ok / len(val_cells) if val_cells else 0

    gates = {
        "coverage": coverage, "false_early": fe_rate, "post_release": pr_rate,
        "k10_containment": k10_rate, "median_anchor_error": median_age,
        "no_corridor_abstain": nc_rate, "feature_valid_rate": fv_rate,
        "silent_stalls": silent_stalls,
    }
    passes = {
        k: (v >= SIX_GATES[k] if k != "silent_stalls" else v == 0)
        for k, v in gates.items() if k in SIX_GATES
    }
    all_pass = all(passes.values())

    ci = {}
    for k in ["coverage", "false_early", "post_release", "k10_containment", "no_corridor_abstain", "feature_valid_rate"]:
        if k == "coverage": n, d = n_tv_trig, n_tv
        elif k == "k10_containment": n, d = k10_ok, len(tv_triggered) if tv_triggered else 0
        elif k == "false_early": n, d = fe_count, n_all_trig
        elif k == "post_release": n, d = pr_count, n_all_trig
        elif k == "no_corridor_abstain": n, d = nc_abstained, n_nc
        elif k == "feature_valid_rate": n, d = fv_ok, len(val_cells)
        else: continue
        lo, hi = wilson_ci(n, d)
        ci[k] = {"lo": round(lo, 4), "hi": round(hi, 4), "n": n, "d": d}

    return {
        "gates": gates, "all_pass": all_pass, "passes": passes, "ci": ci,
        "n_tv": n_tv, "n_nc": n_nc, "n_trig": n_all_trig,
        "silent_stalls": silent_stalls, "total_disarms": total_disarms,
    }


def main():
    ap = argparse.ArgumentParser(description="P5 R1/R2 Validation Evaluation")
    ap.add_argument("--ckpt", default=str(CKPT_PATH), help="Detector checkpoint path")
    ap.add_argument("--validation-root", default=str(CORPUS_ROOT / "validation"))
    ap.add_argument("--teacher-labels", default=str(LABELS_PATH))
    ap.add_argument("--r1-freeze", default=str(R1_FREEZE_PATH))
    ap.add_argument("--r2-grid", default=str(R2_GRID_PATH))
    ap.add_argument("--output-root", default=str(REPO / "evidence/m1c/validation_fsm_eval"))
    args = ap.parse_args()

    global CKPT
    CKPT = Path(args.ckpt)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load teacher labels
    teacher_labels = {}
    for r in csv.DictReader(open(args.teacher_labels)):
        pool = r["pool"]
        if pool != "validation": continue
        teacher_labels[(int(r["task"]), int(r["state"]))] = {
            "teacher_valid": r["teacher_valid"] == "True",
            "teacher_anchor": int(r["teacher_anchor"]),
            "no_corridor": r["no_corridor"] == "True",
            "hard_negative_category": r.get("hard_negative_category", ""),
        }

    # Load validation cells
    val_dir = Path(args.validation_root)
    val_cells = []
    for cell_dir in sorted(val_dir.iterdir()):
        if not cell_dir.is_dir(): continue
        try:
            parts = cell_dir.name.split("_")
            task = int(parts[0].replace("task", ""))
            state = int(parts[1].replace("state", ""))
        except (ValueError, IndexError): continue
        val_cells.append({"task": task, "state": state, "path": cell_dir})

    print(f"P5: {len(val_cells)} validation cells, {len(teacher_labels)} teacher labels")

    from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R

    # R1 evaluation
    r1_config = json.load(open(args.r1_freeze))["parameters"]
    print(f"\nR1: {r1_config}")
    r1_result = evaluate_fsm(SC5DetectorRuntimeV1R, r1_config, val_cells, teacher_labels)
    print(f"  TV={r1_result['n_tv']} NC={r1_result['n_nc']} Trig={r1_result['n_trig']}")
    print(f"  Coverage={r1_result['gates']['coverage']:.4f}  "
          f"K10={r1_result['gates']['k10_containment']:.4f}  "
          f"FE={r1_result['gates']['false_early']:.4f}  "
          f"PR={r1_result['gates']['post_release']:.4f}  "
          f"AnchorErr={r1_result['gates']['median_anchor_error']:.1f}  "
          f"Abstain={r1_result['gates']['no_corridor_abstain']:.4f}")
    print(f"  All pass: {r1_result['all_pass']}")
    r1_result["config"] = r1_config

    # R2 grid evaluation
    r2_grid = json.load(open(args.r2_grid))["configs"]
    r2_results = []
    for i, cfg in enumerate(r2_grid):
        fsm_cfg = {k: v for k, v in cfg.items() if k not in ("id",)}
        result = evaluate_fsm(SC5DetectorRuntimeV1R, fsm_cfg, val_cells, teacher_labels)
        result["id"] = cfg["id"]
        result["config"] = fsm_cfg
        r2_results.append(result)
        tag = "PASS" if result["all_pass"] else "FAIL"
        print(f"  R2[{i+1}/{len(r2_grid)}] id={cfg['id']} {tag}  "
              f"abstain={result['gates']['no_corridor_abstain']:.4f}  "
              f"cov={result['gates']['coverage']:.4f}")

    # Selection: filter, maximize abstain, tie-break
    survivors = [r for r in r2_results if r["all_pass"]]
    if survivors:
        survivors.sort(key=lambda r: (
            r["gates"]["no_corridor_abstain"],
            r["gates"]["k10_containment"],
            -r["gates"]["median_anchor_error"],
            -r["gates"]["false_early"],
            r["total_disarms"],
            r["config"].get("n_candidate", 99),
            r["config"].get("tau_on", 0) - r["config"].get("tau_off", 0),
        ), reverse=True)
        selected = survivors[0]
        print(f"\nSelected R2: id={selected['id']} config={selected['config']}")
        print(f"  abstain={selected['gates']['no_corridor_abstain']:.4f}")
    else:
        selected = None
        print("\nNo R2 config passed all gates")

    # Output
    with open(out_root / "r1_metrics.json", "w") as f:
        json.dump(r1_result, f, indent=2, default=str)
    with open(out_root / "r2_grid_metrics.json", "w") as f:
        json.dump(r2_results, f, indent=2, default=str)

    # CSV
    r2_rows = []
    for r in r2_results:
        row = {"id": r["id"], "all_pass": r["all_pass"]}
        for k, v in r["gates"].items():
            row[k] = v
        for k, v in r["config"].items():
            row[f"cfg_{k}"] = v
        r2_rows.append(row)
    with open(out_root / "r2_grid_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=r2_rows[0].keys())
        w.writeheader()
        w.writerows(r2_rows)

    # CI
    ci_rows = [{"label": "R1", **{f"{k}_lo": v["lo"] for k, v in r1_result["ci"].items()},
                **{f"{k}_hi": v["hi"] for k, v in r1_result["ci"].items()}}]
    if selected:
        ci_rows.append({"label": "R2_selected", **{f"{k}_lo": v["lo"] for k, v in selected["ci"].items()},
                        **{f"{k}_hi": v["hi"] for k, v in selected["ci"].items()}})
    with open(out_root / "confidence_intervals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ci_rows[0].keys())
        w.writeheader()
        w.writerows(ci_rows)

    # Per-cell
    per_cell = []
    for cell in val_cells:
        tl = teacher_labels.get((cell["task"], cell["state"]), {})
        per_cell.append({"task": cell["task"], "state": cell["state"],
                         "teacher_valid": tl.get("teacher_valid",""),
                         "teacher_anchor": tl.get("teacher_anchor",""),
                         "no_corridor": tl.get("no_corridor",""),
                         "hard_negative_category": tl.get("hard_negative_category","")})
    with open(out_root / "per_cell_teacher_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_cell[0].keys())
        w.writeheader()
        w.writerows(per_cell)

    # Manifest
    manifest = {
        "gate": "P5_VALIDATION",
        "r1_config": r1_config,
        "r1_all_pass": r1_result["all_pass"],
        "r2_grid_size": len(r2_grid),
        "r2_survivors": len(survivors),
        "selected_config": selected["config"] if selected else None,
        "selected_id": selected["id"] if selected else None,
        "decision": "R2_FORMAL_VALIDATION_PASS" if selected else ("R1_FORMAL_VALIDATION_PASS" if r1_result["all_pass"] else "RUNTIME_ONLY_REPAIR_INSUFFICIENT"),
        "teacher_labels_sha": sha256(Path(args.teacher_labels)),
        "r1_freeze_sha": sha256(Path(args.r1_freeze)),
        "r2_grid_sha": sha256(Path(args.r2_grid)),
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if selected:
        print(f"\n  R2 SELECTED: id={selected['id']}")
    elif r1_result["all_pass"]:
        print(f"\n  R1 PASSES ALL GATES")
    else:
        print(f"\n  RUNTIME_ONLY_REPAIR_INSUFFICIENT → SC5_V2_TRAINING = GO")
    print(f"  Output: {out_root}")


if __name__ == "__main__":
    main()
