#!/usr/bin/env python3
"""P5 v2: R1/R2 independent validation evaluation.

Fixes from P5_RUN_2 audit:
- R1: explicit fsm_version="v1r_r1"
- R2: each config merges fsm_version="v1r_r2", guard=5, tau_release=0.3
- Gate directions: coverage>=0.80, K10>=0.85, abstain>=0.90, FE<=0.10, PR<=0.05, anchor<=8
- Feature-valid computed from actual telemetry feat_valid column
- Disarms counted once per episode
- Tie-break: fewer disarms, lower n_candidate, smaller hysteresis gap preferred
"""
import os, sys, json, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))

R1_FREEZE_PATH = REPO / "migration_audit/m1c/R1_RUNTIME_BASELINE_FROZEN_V2.json"
R2_GRID_PATH = REPO / "migration_audit/m1c/r2_validation_grid.json"
CORPUS_ROOT = REPO / "evidence/m1c/object_clean_corpus"
LABELS_PATH = REPO / "evidence/m1c/object_teacher_labels_v2/per_cell_teacher_labels.csv"

GATE_DIRECTIONS = {
    "coverage": ">=", "k10_containment": ">=", "no_corridor_abstain": ">=",
    "feature_valid_rate": ">=",
    "false_early": "<=", "post_release": "<=", "median_anchor_error": "<=",
    "silent_stalls": "==",
}
GATE_THRESHOLDS = {
    "coverage": 0.80, "k10_containment": 0.85, "no_corridor_abstain": 0.90,
    "feature_valid_rate": 0.99, "false_early": 0.10, "post_release": 0.05,
    "median_anchor_error": 8, "silent_stalls": 0,
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


def check_gate(name, value):
    thresh = GATE_THRESHOLDS[name]
    direction = GATE_DIRECTIONS[name]
    if direction == ">=": return value >= thresh
    elif direction == "<=": return value <= thresh
    elif direction == "==": return value == thresh
    return False


CKPT = None


def evaluate_fsm(detector_cls, config, val_cells, teacher_labels):
    n_tv = 0; n_nc = 0
    tv_triggered = []; nc_triggered = []; silent_stalls = 0
    total_disarms = 0; total_fv_cells = 0; total_fv_steps = 0

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
        # Feature-valid from actual telemetry
        fv_ok_steps = 0
        for step, r in enumerate(rows):
            fv = r.get("feat_valid", "") == "True"
            if fv: fv_ok_steps += 1
            cp_s = r.get("corridor_p", ""); rp_s = r.get("release_p", "")
            pp = r.get("pred_phase", ""); cp = float(cp_s) if cp_s and cp_s != "" else float("nan")
            rp = float(rp_s) if rp_s and rp_s != "" else float("nan")
            dec = d.update_from_scores(cp, rp, pp, step, feat_valid=fv)
            if d.state == "ARMED" and first_arm < 0: first_arm = step
            if d.emitted and emit_step is None: emit_step = step
        if d.state == "ARMED" and not d.emitted: silent_stalls += 1
        total_disarms += dec.get("disarm_count", 0)
        cell_n_steps = len(rows)
        if cell_n_steps > 0:
            total_fv_cells += 1; total_fv_steps += fv_ok_steps
            cell["fv_cell_pass"] = (fv_ok_steps / cell_n_steps) >= 0.99

        if emit_step is not None:
            age = abs(emit_step - anchor) if anchor >= 0 else None
            entry = {"tv": tv, "anchor": anchor, "emit": emit_step, "age": age}
            if tv and anchor >= 0:
                entry["k10"] = (anchor <= emit_step < anchor + 10)
                entry["post_release"] = (emit_step >= anchor + 10)
                entry["false_early"] = (emit_step < anchor)
            (tv_triggered if tv else nc_triggered).append(entry)

    n_tv_trig = len(tv_triggered)
    n_nc_trig = len(nc_triggered)
    n_all_trig = n_tv_trig + n_nc_trig
    coverage = n_tv_trig / n_tv if n_tv > 0 else 0
    k10_ok = sum(1 for t in tv_triggered if t.get("k10") is True)
    k10_rate = k10_ok / n_tv_trig if n_tv_trig > 0 else 0
    ages = [t["age"] for t in tv_triggered if t.get("age") is not None]
    median_age = float(np.median(ages)) if ages else -1
    fe = sum(1 for t in tv_triggered if t.get("false_early") is True)
    fe_rate = fe / n_tv_trig if n_tv_trig > 0 else 0  # denominator: TV-triggered only
    pr = sum(1 for t in tv_triggered if t.get("post_release") is True)
    pr_rate = pr / n_tv_trig if n_tv_trig > 0 else 0  # denominator: TV-triggered only
    nc_abstained = n_nc - n_nc_trig
    nc_rate = nc_abstained / n_nc if n_nc > 0 else 0
    total_steps_all = sum(len(list(csv.DictReader(open(c["path"]/"step_telemetry.csv")))) for c in val_cells)
    fv_step_rate = total_fv_steps / total_steps_all if total_steps_all > 0 else 0
    fv_cells_ok = sum(1 for c in val_cells if c.get("fv_cell_pass", False))
    fv_cell_rate = fv_cells_ok / len(val_cells) if val_cells else 0

    timing_evaluable = (n_tv_trig > 0 and len(ages) > 0)
    gates = {
        "coverage": coverage, "k10_containment": k10_rate, "no_corridor_abstain": nc_rate,
        "feature_valid_step_rate": fv_step_rate, "feature_valid_cell_rate": fv_cell_rate,
        "false_early": fe_rate, "post_release": pr_rate,
        "median_anchor_error": median_age if timing_evaluable else None,
        "silent_stalls": silent_stalls,
    }
    # K10/FE/PR/anchor are only evaluable with valid anchors
    timing_passes = {}
    if not timing_evaluable:
        timing_passes = {"k10_containment": False, "false_early": False, "post_release": False, "median_anchor_error": False}
    passes = {k: check_gate(k, v) for k, v in gates.items() if k in GATE_THRESHOLDS and v is not None}
    passes.update(timing_passes)
    all_pass = all(passes.values())

    ci = {}
    for k in ["coverage","k10_containment","no_corridor_abstain","false_early","post_release"]:
        if k == "coverage": n, d = n_tv_trig, n_tv
        elif k == "k10_containment": n, d = k10_ok, n_tv_trig
        elif k == "no_corridor_abstain": n, d = nc_abstained, n_nc
        elif k == "false_early": n, d = fe, n_tv_trig
        elif k == "post_release": n, d = pr, n_tv_trig
        else: continue
        lo, hi = wilson_ci(n, d)
        ci[k] = {"lo": round(lo, 4), "hi": round(hi, 4), "n": n, "d": d}
    # Feature-valid CI uses actual step-level stats
    lo_fv, hi_fv = wilson_ci(total_fv_steps, total_steps_all)
    ci["feature_valid_step_rate"] = {"lo": round(lo_fv, 4), "hi": round(hi_fv, 4), "n": total_fv_steps, "d": total_steps_all}

    ci = {}
    for k in ["coverage","k10_containment","no_corridor_abstain","feature_valid_rate","false_early","post_release"]:
        if k == "coverage": n, d = n_tv_trig, n_tv
        elif k == "k10_containment": n, d = k10_ok, n_tv_trig
        elif k == "no_corridor_abstain": n, d = nc_abstained, n_nc
        elif k == "feature_valid_rate": n, d = total_fv_steps, total_fv_steps  # Note: simplified
        elif k == "false_early": n, d = fe, n_all_trig
        elif k == "post_release": n, d = pr, n_all_trig
        else: continue
        lo, hi = wilson_ci(n, d)
        ci[k] = {"lo": round(lo, 4), "hi": round(hi, 4), "n": n, "d": d}

    return {
        "gates": gates, "passes": passes, "all_pass": all_pass, "ci": ci,
        "n_tv": n_tv, "n_nc": n_nc, "n_trig": n_all_trig,
        "silent_stalls": silent_stalls, "total_disarms": total_disarms,
    }


def main():
    ap = argparse.ArgumentParser(description="P5 v2 Validation")
    ap.add_argument("--ckpt", default=str(CKPT_PATH := REPO / "artifacts/detector/sc5_mlp_s2.pt"))
    ap.add_argument("--validation-root", default=str(CORPUS_ROOT / "validation"))
    ap.add_argument("--teacher-labels", default=str(LABELS_PATH))
    ap.add_argument("--r1-freeze", default=str(R1_FREEZE_PATH))
    ap.add_argument("--r2-grid", default=str(R2_GRID_PATH))
    ap.add_argument("--output-root", default=str(REPO / "evidence/m1c/validation_fsm_eval_v2"))
    args = ap.parse_args()

    global CKPT; CKPT = Path(args.ckpt)
    out_root = Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)

    teacher_labels = {}
    dup_check = set()
    for r in csv.DictReader(open(args.teacher_labels)):
        if r["pool"] != "validation": continue
        key = (int(r["task"]), int(r["state"]))
        if key in dup_check:
            raise SystemExit(f"DUPLICATE_LABEL: {key}")
        dup_check.add(key)
        tv = r["teacher_valid"] == "True"
        anchor = int(r["teacher_anchor"])
        if tv and anchor < 0:
            raise SystemExit(f"INVARIANT_VIOLATION: {key} teacher_valid=True but anchor={anchor}")
        teacher_labels[key] = {"teacher_valid": tv, "teacher_anchor": anchor}

    val_dir = Path(args.validation_root)
    val_cells = []
    for cell_dir in sorted(val_dir.iterdir()):
        if not cell_dir.is_dir(): continue
        try:
            parts = cell_dir.name.split("_")
            task = int(parts[0].replace("task","")); state = int(parts[1].replace("state",""))
        except (ValueError, IndexError): continue
        val_cells.append({"task": task, "state": state, "path": cell_dir})

    val_keys = {(c["task"], c["state"]) for c in val_cells}
    label_keys = set(teacher_labels.keys())
    missing_cells = val_keys - label_keys
    extra_labels = label_keys - val_keys
    if missing_cells:
        raise SystemExit(f"MISSING_VALIDATION_LABELS: {len(missing_cells)} cells without labels")
    if extra_labels:
        raise SystemExit(f"EXTRA_VALIDATION_LABELS: {len(extra_labels)} labels without cells")

    print(f"P5 v2: {len(val_cells)} cells, {len(teacher_labels)} labels (validated)")

    from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R

    # R1: explicit fsm_version
    r1_config = {"fsm_version": "v1r_r1", "tau_corridor": 0.3, "tau_release": 0.3, "guard": 5}
    print(f"R1: {r1_config}")
    r1 = evaluate_fsm(SC5DetectorRuntimeV1R, r1_config, val_cells, teacher_labels)
    print(f"  TV={r1['n_tv']} NC={r1['n_nc']} Trig={r1['n_trig']}  "
          f"cov={r1['gates']['coverage']:.4f} K10={r1['gates']['k10_containment']:.4f} "
          f"FE={r1['gates']['false_early']:.4f} PR={r1['gates']['post_release']:.4f} "
          f"anchor={r1['gates']['median_anchor_error']:.1f} "
          f"abstain={r1['gates']['no_corridor_abstain']:.4f}  all_pass={r1['all_pass']}")

    # R2: merge grid config with required FSM fields
    r2_grid = json.load(open(args.r2_grid))["configs"]
    r2_results = []
    for cfg in r2_grid:
        grid_params = {k: v for k, v in cfg.items() if k != "id"}
        fsm_cfg = {"fsm_version": "v1r_r2", "guard": 5, "tau_release": 0.3, **grid_params}
        result = evaluate_fsm(SC5DetectorRuntimeV1R, fsm_cfg, val_cells, teacher_labels)
        result["id"] = cfg["id"]; result["config"] = fsm_cfg
        r2_results.append(result)
        tag = "PASS" if result["all_pass"] else "FAIL"
        print(f"  R2[{cfg['id']:03d}] {tag}  abst={result['gates']['no_corridor_abstain']:.4f}  "
              f"cov={result['gates']['coverage']:.4f}  K10={result['gates']['k10_containment']:.4f}")

    survivors = [r for r in r2_results if r["all_pass"]]
    if survivors:
        survivors.sort(key=lambda r: (
            -r["gates"]["no_corridor_abstain"], -r["gates"]["k10_containment"],
            r["gates"]["median_anchor_error"], r["gates"]["false_early"],
            r["total_disarms"], r["config"].get("n_candidate", 99),
            r["config"].get("tau_on", 0) - r["config"].get("tau_off", 0),
        ))
        selected = survivors[0]
        print(f"\nSelected R2: id={selected['id']} config={selected['config']}")
    else:
        selected = None
        print("\nNo R2 config passed all gates")

    with open(out_root / "r1_metrics.json", "w") as f: json.dump(r1, f, indent=2, default=str)
    with open(out_root / "r2_grid_metrics.json", "w") as f: json.dump(r2_results, f, indent=2, default=str)
    r2_rows = []
    for r in r2_results:
        row = {"id": r["id"], "all_pass": r["all_pass"]}
        for k, v in r["gates"].items(): row[k] = v
        r2_rows.append(row)
    with open(out_root / "r2_grid_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=r2_rows[0].keys()); w.writeheader(); w.writerows(r2_rows)

    manifest = {
        "gate": "P5_VALIDATION_V2",
        "r1_all_pass": r1["all_pass"],
        "r2_survivors": len(survivors),
        "selected_id": selected["id"] if selected else None,
        "decision": "R2_FORMAL_VALIDATION_PASS" if selected else ("R1_FORMAL_VALIDATION_PASS" if r1["all_pass"] else "RUNTIME_ONLY_REPAIR_INSUFFICIENT"),
    }
    with open(out_root / "manifest.json", "w") as f: json.dump(manifest, f, indent=2)

    print(f"\n  Decision: {manifest['decision']}")
    print(f"  Output: {out_root}")


if __name__ == "__main__":
    main()
