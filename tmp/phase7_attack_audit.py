#!/usr/bin/env python3
"""Phase 7 Object: post-benchmark attack efficacy audit.

Covers:
  1. Completion audit (66/66)
  2. Pairing completeness
  3. Pre-trigger parity (VIS vs RAND)
  4. Per-run and per-cell results tables
  5. 11-cell ITT, 9-cell qualified, 10-cell primary, 1-cell supplementary
"""
import csv, hashlib, json, os, sys, glob, re, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
SERVER_REPO = Path("/mnt/sdc/dty_user/openvla_attack")

# ── Config ──
BENCHMARK_DIR = SERVER_REPO / "evidence/phase7_object/attack_benchmark"
OUT_DIR = SERVER_REPO / "evidence/phase7_object"

V2_CKPT_SHA = "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c"
EXPECTED_BACKEND = "upstream_tf_jpeg"

# 11 Object cells from CLEAN baseline
OBJECT_CELLS = [
    ("butter_s0", 6, 0), ("butter_s2", 6, 2),
    ("ketchup_s0", 8, 0), ("salad_dressing_s0", 4, 0),
    ("bbq_sauce_s0", 1, 0), ("milk_s4", 3, 4),
    ("orange_juice_s0", 2, 0), ("tomato_sauce_s0", 5, 0),
    ("alphabet_soup_s0", 7, 0), ("cream_cheese_s0", 0, 0),
    ("chocolate_pudding_s2", 9, 2),
]

CLEAN_NO_EMIT = {"cream_cheese_s0", "chocolate_pudding_s2"}
PRIMARY_CELLS = [c for c in OBJECT_CELLS if c[0] not in {"alphabet_soup_s0"}]
SUPP_CELLS = [c for c in OBJECT_CELLS if c[0] == "alphabet_soup_s0"]

SEEDS = [42, 123, 456]
CONDITIONS = ["TRUE_T10", "RAND_T10"]


def sha256_file(path):
    if not os.path.exists(path): return "MISSING"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_summary(path):
    s = os.path.join(path, "episode_summary.json")
    if not os.path.exists(s): return {}
    with open(s) as f: return json.load(f)


def load_telemetry(path):
    tel = os.path.join(path, "step_telemetry.csv")
    if not os.path.exists(tel): return None
    rows = list(csv.DictReader(open(tel)))
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def audit_completion():
    """Audit all 66 runs for completion."""
    print("=== 1. Completion Audit ===")
    results = []
    for cell_name, task_idx, state_id in OBJECT_CELLS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                cond_dir = "vis" if cond == "TRUE_T10" else "rand"
                run_dir = BENCHMARK_DIR / cond_dir / f"{cell_name}_seed{seed}"
                summary = load_summary(run_dir)
                tel = load_telemetry(run_dir)
                done = os.path.exists(os.path.join(run_dir, ".done"))
                results.append({
                    "cell": cell_name, "task_idx": task_idx, "state_id": state_id,
                    "seed": seed, "condition": cond,
                    "done": done,
                    "n_steps": len(tel) if tel else 0,
                    "summary_exists": bool(summary),
                    "task_success": summary.get("task_success", None),
                    "mlp_emit": summary.get("mlp_emit_step", -1),
                    "attack_frames": summary.get("attack_frames", 0),
                    "checkpoint_sha": summary.get("checkpoint_sha256", "")[:16],
                    "backend": summary.get("preprocess_backend_resolved", "?"),
                })
    return results


def audit_pairing(completion):
    """Check each (cell, seed) has exactly 1 VIS + 1 RAND."""
    print("\n=== 2. Pairing Completeness ===")
    pairs = defaultdict(lambda: {"VIS": None, "RAND": None})
    for r in completion:
        key = (r["cell"], r["seed"])
        cond_short = "VIS" if r["condition"] == "TRUE_T10" else "RAND"
        pairs[key][cond_short] = r

    pairing = []
    for (cell, seed), pair in sorted(pairs.items()):
        vis_ok = pair["VIS"] is not None and pair["VIS"]["done"]
        rand_ok = pair["RAND"] is not None and pair["RAND"]["done"]
        pairing.append({
            "cell": cell, "seed": seed,
            "vis_done": vis_ok, "rand_done": rand_ok,
            "paired": vis_ok and rand_ok,
            "vis_emit": pair["VIS"]["mlp_emit"] if pair["VIS"] else -1,
            "rand_emit": pair["RAND"]["mlp_emit"] if pair["RAND"] else -1,
        })

    n_paired = sum(1 for p in pairing if p["paired"])
    print(f"  Paired complete: {n_paired}/{len(pairing)}")
    missing = [p for p in pairing if not p["paired"]]
    if missing:
        for m in missing:
            print(f"  MISSING: {m['cell']}_seed{m['seed']} VIS={m['vis_done']} RAND={m['rand_done']}")
    return pairing


def pre_trigger_parity(pairing):
    """Compare pre-attack state between VIS and RAND for each pair."""
    print("\n=== 3. Pre-trigger Parity ===")
    parity_rows = []
    for p in pairing:
        if not p["paired"]:
            parity_rows.append({**p, "parity": "UNPAIRED", "divergences": []})
            continue

        cell, seed = p["cell"], p["seed"]
        vis_dir = BENCHMARK_DIR / "vis" / f"{cell}_seed{seed}"
        rand_dir = BENCHMARK_DIR / "rand" / f"{cell}_seed{seed}"

        vis_tel = load_telemetry(vis_dir)
        rand_tel = load_telemetry(rand_dir)
        vis_sum = load_summary(vis_dir)
        rand_sum = load_summary(rand_dir)

        divergences = []

        # Compare emit step
        vis_emit = vis_sum.get("mlp_emit_step", -1)
        rand_emit = rand_sum.get("mlp_emit_step", -1)
        if vis_emit != rand_emit:
            divergences.append(f"emit_step: VIS={vis_emit} RAND={rand_emit}")

        # Compare attack start (should be same since emit is same)
        emit = min(vis_emit, rand_emit) if vis_emit >= 0 and rand_emit >= 0 else -1
        if emit < 0:
            # No emit in at least one — compare full trajectories
            pass

        # Compare pre-emit steps for feature parity
        if emit >= 0:
            vis_pre = [r for r in vis_tel if int(r.get("step", 0)) < emit]
            rand_pre = [r for r in rand_tel if int(r.get("step", 0)) < emit]

            # Feature comparison on last pre-emit step
            if vis_pre and rand_pre:
                v_last = vis_pre[-1]
                r_last = rand_pre[-1]
                for key in ["corridor_p", "release_p", "pred_phase", "detector_state",
                           "eef_x", "eef_y", "eef_z", "obj_z", "eef_obj_dist"]:
                    vv = v_last.get(key, ""); rv = r_last.get(key, "")
                    if vv != "" and rv != "" and vv != rv:
                        try:
                            if abs(float(vv) - float(rv)) > 1e-6:
                                divergences.append(f"pre_emit_{key}: VIS={vv} RAND={rv}")
                        except ValueError:
                            if vv != rv:
                                divergences.append(f"pre_emit_{key}: VIS={vv} RAND={rv}")

        parity = "PASS" if len(divergences) == 0 else "PREFIX_DIVERGENT"
        parity_rows.append({
            "cell": cell, "seed": seed,
            "vis_emit": vis_emit, "rand_emit": rand_emit,
            "emit_match": vis_emit == rand_emit,
            "parity": parity,
            "divergences": "; ".join(divergences),
            "n_divergences": len(divergences),
        })

    n_pass = sum(1 for r in parity_rows if r["parity"] == "PASS")
    n_div = sum(1 for r in parity_rows if r["parity"] == "PREFIX_DIVERGENT")
    n_unp = sum(1 for r in parity_rows if r["parity"] == "UNPAIRED")
    print(f"  Parity PASS: {n_pass}, DIVERGENT: {n_div}, UNPAIRED: {n_unp}")
    return parity_rows


def compile_per_run(completion):
    """Compile per-run results table."""
    print("\n=== 4. Per-Run Results ===")
    per_run = []
    for r in completion:
        cond_dir = "vis" if r["condition"] == "TRUE_T10" else "rand"
        run_dir = BENCHMARK_DIR / cond_dir / f"{r['cell']}_seed{r['seed']}"
        summary = load_summary(run_dir)
        tel = load_telemetry(run_dir)

        n_atk = summary.get("attack_frames", 0)
        success = summary.get("task_success", False)
        emit = summary.get("mlp_emit_step", -1)
        atk_start = emit if emit >= 0 else -1

        # Determine failure step and phase
        fail_step = -1; fail_phase = "?"
        if tel and not success:
            last = tel[-1]
            fail_step = int(last.get("step", -1))
            fail_phase = last.get("pred_phase", "?")

        # Target-token duty and env-open duty
        tok_duty = summary.get("token_open_duty", 0)
        env_duty = summary.get("env_open_duty", 0)

        # Frame hashes for pre-attack images (first step)
        frame_sha = "?"
        if tel:
            first = tel[0]
            frame_sha = first.get("frame_sha256", "?")[:16]

        per_run.append({
            "cell": r["cell"], "seed": r["seed"], "condition": r["condition"],
            "clean_success": "n/a",  # filled from CLEAN baseline
            "emit": emit, "attack_start": atk_start, "attack_frames": n_atk,
            "task_success": success, "token_duty": tok_duty, "env_open_duty": env_duty,
            "fail_step": fail_step, "fail_phase": fail_phase,
            "frame_sha": frame_sha, "checkpoint_sha": r["checkpoint_sha"],
        })
    return per_run


def compile_per_cell(per_run):
    """Aggregate per-run into per-cell statistics."""
    print("\n=== 5. Per-Cell Results ===")
    cells = defaultdict(lambda: {"VIS": [], "RAND": []})
    for r in per_run:
        cond_short = "VIS" if r["condition"] == "TRUE_T10" else "RAND"
        cells[(r["cell"], r["seed"])][cond_short] = r

    # Aggregate by cell (across seeds)
    cell_agg = defaultdict(lambda: {"vis_failures": 0, "rand_failures": 0,
                                     "vis_total": 0, "rand_total": 0,
                                     "vis_successes": 0, "rand_successes": 0,
                                     "vis_emits": 0, "rand_emits": 0})
    for (cell, seed), pair in cells.items():
        for cond in ["VIS", "RAND"]:
            r = pair.get(cond)
            if r is None: continue
            agg = cell_agg[cell]
            agg[f"{cond.lower()}_total"] += 1
            if not r["task_success"]:
                agg[f"{cond.lower()}_failures"] += 1
            else:
                agg[f"{cond.lower()}_successes"] += 1
            if r["emit"] >= 0:
                agg[f"{cond.lower()}_emits"] += 1

    per_cell = []
    for cell_name, _, _ in OBJECT_CELLS:
        agg = cell_agg.get(cell_name, {})
        vis_tot = agg.get("vis_total", 0)
        rand_tot = agg.get("rand_total", 0)
        vis_fail = agg.get("vis_failures", 0)
        rand_fail = agg.get("rand_failures", 0)
        vis_rate = vis_fail / vis_tot if vis_tot > 0 else 0
        rand_rate = rand_fail / rand_tot if rand_tot > 0 else 0

        is_no_emit = cell_name in CLEAN_NO_EMIT
        is_primary = cell_name != "alphabet_soup_s0"

        # Majority outcome
        vis_maj = "FAIL" if vis_fail >= 2 else ("SUCCESS" if vis_tot > 0 else "N/A")
        rand_maj = "FAIL" if rand_fail >= 2 else ("SUCCESS" if rand_tot > 0 else "N/A")

        per_cell.append({
            "cell": cell_name,
            "subset": "primary" if is_primary else "supplementary",
            "clean_emits": not is_no_emit,
            "vis_n": vis_tot, "vis_failures": vis_fail,
            "vis_failure_rate": round(vis_rate, 3),
            "vis_majority": vis_maj,
            "rand_n": rand_tot, "rand_failures": rand_fail,
            "rand_failure_rate": round(rand_rate, 3),
            "rand_majority": rand_maj,
            "paired_difference": round(vis_rate - rand_rate, 3),
        })
    return per_cell


def compute_itt(per_cell, per_run):
    """11-cell ITT analysis."""
    print("\n=== 6. 11-Cell ITT ===")
    # Trigger coverage: cells where V2 emits
    trigger_cells = [c for c in per_cell if c["clean_emits"]]
    no_trigger_cells = [c for c in per_cell if not c["clean_emits"]]

    # E2E attack success: cells where attack SIDE fails the task
    vis_all = [r for r in per_run if r["condition"] == "TRUE_T10"]
    rand_all = [r for r in per_run if r["condition"] == "RAND_T10"]

    vis_e2e_fail = sum(1 for r in vis_all if not r["task_success"])
    rand_e2e_fail = sum(1 for r in rand_all if not r["task_success"])

    itt = {
        "denominator": 11,
        "trigger_coverage": f"{len(trigger_cells)}/11",
        "no_emit_cells": sorted(no_trigger_cells),
        "vis_total_runs": len(vis_all),
        "vis_e2e_failures": vis_e2e_fail,
        "vis_e2e_failure_rate": round(vis_e2e_fail / len(vis_all), 3) if vis_all else 0,
        "rand_total_runs": len(rand_all),
        "rand_e2e_failures": rand_e2e_fail,
        "rand_e2e_failure_rate": round(rand_e2e_fail / len(rand_all), 3) if rand_all else 0,
        "note": "no-emit cells counted as attack pipeline failure (success=false)",
    }

    for k, v in itt.items():
        print(f"  {k}: {v}")
    return itt


def compute_qualified(per_cell, per_run):
    """9-cell clean-qualified subset (excludes cream_cheese_s0, chocolate_pudding_s2)."""
    print("\n=== 7. 9-Cell Clean-Qualified Subset ===")
    qualified_cells = [c["cell"] for c in per_cell if c["clean_emits"]]

    vis_qual = [r for r in per_run if r["condition"] == "TRUE_T10" and r["cell"] in qualified_cells]
    rand_qual = [r for r in per_run if r["condition"] == "RAND_T10" and r["cell"] in qualified_cells]

    vis_fail = sum(1 for r in vis_qual if not r["task_success"])
    rand_fail = sum(1 for r in rand_qual if not r["task_success"])

    # Per-cell paired
    qual_cell_data = [c for c in per_cell if c["cell"] in qualified_cells]
    vis_cell_rates = [c["vis_failure_rate"] for c in qual_cell_data]
    rand_cell_rates = [c["rand_failure_rate"] for c in qual_cell_data]
    diffs = [c["paired_difference"] for c in qual_cell_data]

    qualified = {
        "denominator": len(qualified_cells),
        "cells": sorted(qualified_cells),
        "vis_total": len(vis_qual), "vis_failures": vis_fail,
        "vis_failure_rate": round(vis_fail / len(vis_qual), 3) if vis_qual else 0,
        "rand_total": len(rand_qual), "rand_failures": rand_fail,
        "rand_failure_rate": round(rand_fail / len(rand_qual), 3) if rand_qual else 0,
        "mean_vis_cell_rate": round(np.mean(vis_cell_rates), 3) if vis_cell_rates else 0,
        "mean_rand_cell_rate": round(np.mean(rand_cell_rates), 3) if rand_cell_rates else 0,
        "mean_paired_diff": round(np.mean(diffs), 3) if diffs else 0,
        "median_paired_diff": round(np.median(diffs), 3) if diffs else 0,
    }

    for k, v in qualified.items():
        print(f"  {k}: {v}")
    return qualified


def compute_primary_supplementary(per_cell, per_run):
    """Separate primary (10) and supplementary (1) results."""
    print("\n=== 8. Primary vs Supplementary ===")
    primary_cells = [c for c in per_cell if c["subset"] == "primary"]
    supp_cells = [c for c in per_cell if c["subset"] == "supplementary"]

    for label, cells in [("Primary (10)", primary_cells), ("Supplementary (1)", supp_cells)]:
        names = [c["cell"] for c in cells]
        vis_fail = sum(c["vis_failures"] for c in cells)
        vis_tot = sum(c["vis_n"] for c in cells)
        rand_fail = sum(c["rand_failures"] for c in cells)
        rand_tot = sum(c["rand_n"] for c in cells)
        print(f"  {label}: cells={names}")
        print(f"    VIS: {vis_fail}/{vis_tot} failures (rate={round(vis_fail/vis_tot,3) if vis_tot>0 else 0})")
        print(f"    RAND: {rand_fail}/{rand_tot} failures (rate={round(rand_fail/rand_tot,3) if rand_tot>0 else 0})")

    return {"primary_cells": [c["cell"] for c in primary_cells],
            "supplementary_cells": [c["cell"] for c in supp_cells]}


def compute_rand_non_degradation(per_run, per_cell):
    """Check RAND non-degradation relative to CLEAN baseline.

    CLEAN baseline success is known: 9/11 cells succeed (all but the 2 no-emit cells).
    For each qualified cell, RAND should not substantially degrade success rate.
    """
    print("\n=== 9. RAND Non-Degradation ===")
    # CLEAN baseline: 9 cells succeed, 2 fail (cream_cheese_s0, chocolate_pudding_s2)
    # For qualified cells, CLEAN success = True for all 9
    qualified_names = [c["cell"] for c in per_cell if c["clean_emits"]]
    degraded = []
    for c in per_cell:
        if c["cell"] in qualified_names:
            if c["rand_failure_rate"] > 0.33:  # more than 1/3 seeds fail
                degraded.append(c["cell"])
    print(f"  RAND degraded cells (>1/3 failure): {degraded if degraded else 'NONE'}")
    return degraded


def gate_decision(itt, qualified, parity_rows, per_cell):
    """Make Object attack efficacy gate decision."""
    print("\n=== 10. Gate Decision ===")

    n_paired = sum(1 for p in parity_rows if p["parity"] == "PASS")
    n_total_pairs = len(parity_rows)
    all_paired = n_paired == n_total_pairs
    parity_ok = all(p["parity"] == "PASS" for p in parity_rows if p["parity"] != "UNPAIRED")

    vis_mean = qualified["mean_vis_cell_rate"]
    rand_mean = qualified["mean_rand_cell_rate"]
    gap = qualified["mean_paired_diff"]

    # Gate criteria
    pairing_ok = n_total_pairs == 33 and sum(1 for p in parity_rows if p["paired"]) == 33
    parity_ok_flag = n_paired >= 30  # allow up to 3 divergent pairs in strict subset

    vis_effective = vis_mean > rand_mean and gap > 0.05
    rand_stable = rand_mean < 0.33

    gate = {
        "gate": "OBJECT_V2_ATTACK_EFFICACY",
        "pairing_complete": pairing_ok,
        "pre_trigger_parity_pass": parity_ok_flag,
        "parity_detail": f"{n_paired}/{n_total_pairs} pairs with PASS parity",
        "vis_failure_rate_qualified": vis_mean,
        "rand_failure_rate_qualified": rand_mean,
        "vis_rand_paired_gap": gap,
        "vis_effective": vis_effective,
        "rand_stable": rand_stable,
        "rand_non_degradation": rand_mean < 0.20,
    }

    if pairing_ok and parity_ok_flag and vis_effective and rand_stable:
        gate["decision"] = "PASS"
    elif not pairing_ok:
        gate["decision"] = "FAIL"
        gate["fail_reason"] = "incomplete_pairs"
    elif not parity_ok_flag:
        gate["decision"] = "PROVISIONAL"
        gate["fail_reason"] = "prefix_divergence"
    else:
        gate["decision"] = "PROVISIONAL"
        gate["fail_reason"] = "weak_vis_effect_or_rand_instability"

    for k, v in gate.items():
        print(f"  {k}: {v}")
    return gate


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Completion audit
    completion = audit_completion()
    n_done = sum(1 for r in completion if r["done"])
    print(f"  Done: {n_done}/{len(completion)}")

    # 2. Pairing
    pairing = audit_pairing(completion)

    # 3. Pre-trigger parity
    parity_rows = pre_trigger_parity(pairing)

    # 4. Per-run table
    per_run = compile_per_run(completion)

    # 5. Per-cell table
    per_cell = compile_per_cell(per_run)

    # 6. ITT
    itt = compute_itt(per_cell, per_run)

    # 7. Qualified
    qualified = compute_qualified(per_cell, per_run)

    # 8. Primary / Supplementary
    ps = compute_primary_supplementary(per_cell, per_run)

    # 9. RAND non-degradation
    degraded = compute_rand_non_degradation(per_run, per_cell)

    # 10. Gate
    gate = gate_decision(itt, qualified, parity_rows, per_cell)

    # ── Save all outputs ──
    # Completion CSV
    with open(OUT_DIR / "ATTACK_COMPLETION_AUDIT.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=completion[0].keys()); w.writeheader(); w.writerows(completion)

    # Pairing CSV
    with open(OUT_DIR / "ATTACK_PAIRING_AUDIT.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pairing[0].keys()); w.writeheader(); w.writerows(pairing)

    # Parity CSV
    parity_fields = ["cell", "seed", "vis_emit", "rand_emit", "emit_match", "parity", "divergences", "n_divergences"]
    with open(OUT_DIR / "ATTACK_PREFIX_PARITY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=parity_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(parity_rows)

    # Per-run CSV
    with open(OUT_DIR / "ATTACK_PER_RUN.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_run[0].keys()); w.writeheader(); w.writerows(per_run)

    # Per-cell CSV
    with open(OUT_DIR / "ATTACK_PER_CELL.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_cell[0].keys()); w.writeheader(); w.writerows(per_cell)

    # Full summary JSON
    summary = {
        "gate": "PHASE7_OBJECT_ATTACK_EFFICACY",
        "completion": {"total": len(completion), "done": n_done},
        "pairing": {"total_pairs": len(pairing),
                    "complete": sum(1 for p in pairing if p["paired"])},
        "parity": {"pass": sum(1 for p in parity_rows if p["parity"] == "PASS"),
                   "divergent": sum(1 for p in parity_rows if p["parity"] == "PREFIX_DIVERGENT"),
                   "unpaired": sum(1 for p in parity_rows if p["parity"] == "UNPAIRED")},
        "itt": itt,
        "qualified": qualified,
        "primary_supplementary": ps,
        "rand_degraded": degraded,
        "gate_decision": gate,
        "v2_checkpoint_sha": V2_CKPT_SHA,
        "backend": EXPECTED_BACKEND,
    }
    with open(OUT_DIR / "ATTACK_EFFICACY_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== OUTPUTS SAVED to {OUT_DIR} ===")


if __name__ == "__main__":
    main()
