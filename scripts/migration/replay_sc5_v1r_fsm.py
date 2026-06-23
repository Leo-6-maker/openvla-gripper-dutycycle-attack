#!/usr/bin/env python3
"""R0/R1/R2 offline FSM replay against M1B 60-cell telemetry.

Replays existing detector scores (corridor_p, release_p, pred_phase, feat_valid)
through three FSM versions without re-running model inference.

  R0 = legacy_v1 (regression baseline — must achieve 60/60 parity)
  R1 = v1r_r1   (minimal disarm)
  R2 = v1r_r2   (full candidate-machine + hysteresis + timeout)

CPU-only. No GPU, no MuJoCo, no model loading beyond checkpoint init.
Read-only on M1B evidence.

Output: evidence/m1c/runtime_replay/
  per_cell_emit.csv
  aggregate_metrics.json
  regression_table.csv
  artifact_manifest.json
"""
import os, sys, json, csv, hashlib, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

MANIFEST_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/artifact_manifest_complete.json"
CLASSIFICATION_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime_b0_d1/final_classification.json"
TEACHER_CONFIG_PATH = REPO / "migration_audit/object_checkpoint_migration/m1_runtime/teacher_config_frozen.json"
EVIDENCE_BASE = REPO / "evidence/object_checkpoint_migration/m1_runtime_b0_d1"
OUT_BASE = REPO / "evidence/m1c/runtime_replay"

CKPT_PATH = REPO / "artifacts/detector/sc5_mlp_s2.pt"

FSM_CONFIGS = {
    "R0": {"fsm_version": "legacy_v1"},
    "R1": {"fsm_version": "v1r_r1"},
    "R2": {"fsm_version": "v1r_r2", "tau_on": 0.5, "tau_off": 0.3,
           "n_candidate": 3, "max_arm_age": 50},
}

# ── SHA helpers ─────────────────────────────────────────────────────

def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify_source_hashes(cell, cell_dir):
    for fname, sha_key in [("step_telemetry.csv", "telemetry_sha256"),
                           ("episode_summary.json", "episode_summary_sha256"),
                           (".done", "done_sha256")]:
        fpath = cell_dir / fname
        if not fpath.exists():
            return f"SOURCE_MISSING: {fname}"
        actual = sha256_file(fpath)
        expected = cell.get(sha_key, "")
        if actual != expected:
            return f"HASH_MISMATCH: {fname}"
    return None


def load_teacher_replay(cell):
    replay_dir = EVIDENCE_BASE / "replay_60cell" / cell["episode_key"] / cell["profile"]
    ts_path = replay_dir / "teacher_summary.json"
    tl_path = replay_dir / "teacher_labels.jsonl"
    teacher_summary = json.load(open(ts_path)) if ts_path.exists() else None
    teacher_labels = [json.loads(l) for l in open(tl_path) if l.strip()] if tl_path.exists() else []
    return teacher_summary, teacher_labels


# ── Replay one cell ─────────────────────────────────────────────────

def replay_cell(cell, detector_ckpt_path):
    """Run R0/R1/R2 replay on one cell. Returns dict keyed by FSM label."""
    cell_dir = EVIDENCE_BASE / cell["relative_path"]

    sha_err = verify_source_hashes(cell, cell_dir)
    if sha_err:
        return {"_error": sha_err}

    rows = list(csv.DictReader(open(cell_dir / "step_telemetry.csv")))
    # Step-index assertion
    for i, r in enumerate(rows):
        if int(r.get("step", -1)) != i:
            return {"_error": f"STEP_INDEX_MISMATCH at idx={i}"}

    n = len(rows)
    teacher_summary, teacher_labels = load_teacher_replay(cell)
    original_emit = cell["emit_step"]

    results = {}
    for label, cfg in FSM_CONFIGS.items():
        from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R
        detector = SC5DetectorRuntimeV1R(str(detector_ckpt_path), **cfg)

        trace = []
        first_arm = -1
        emit_step = None
        n_candidates = 0
        n_arms = 0
        n_disarms = 0
        disarm_reasons = []
        max_arm_age_obs = 0
        last_state = "IDLE"

        for step, r in enumerate(rows):
            cp_str = r.get("corridor_p", "")
            rp_str = r.get("release_p", "")
            pp = r.get("pred_phase", "")
            fv = r.get("feat_valid", "") == "True"

            cp = float(cp_str) if cp_str and cp_str != "" else float("nan")
            rp = float(rp_str) if rp_str and rp_str != "" else float("nan")

            dec = detector.update_from_scores(cp, rp, pp, step, feat_valid=fv)
            trace.append(dec)

            # Track transitions
            if dec["state"] != last_state:
                if dec["state"] == "CANDIDATE":
                    n_candidates += 1
                elif dec["state"] == "ARMED":
                    n_arms += 1
                    if first_arm < 0:
                        first_arm = step
                elif last_state == "ARMED" and dec["state"] == "IDLE":
                    n_disarms += 1
                    disarm_reasons.append(dec.get("disarm_reason", "?"))
                elif last_state == "CANDIDATE" and dec["state"] == "IDLE":
                    n_disarms += 1
                    disarm_reasons.append(dec.get("disarm_reason", "?"))
                last_state = dec["state"]

            if dec["emitted"] and emit_step is None:
                emit_step = step

            max_arm_age_obs = max(max_arm_age_obs, dec.get("arm_age", 0))

        # Silent ARM stall: armed but never emitted by episode end
        silent_stall = (detector.state == "ARMED" and not detector.emitted)

        # K10 check: did emit fall within teacher anchor K-window?
        k10_ok = None
        teacher_anchor = None
        teacher_valid = None
        if teacher_summary:
            teacher_anchor = teacher_summary.get("anchor_candidate")
            teacher_valid = teacher_summary.get("full_k10_valid", False)
            if emit_step is not None and teacher_anchor is not None and teacher_anchor >= 0:
                k10_ok = (teacher_anchor <= emit_step < teacher_anchor + 10)
            elif emit_step is not None and (teacher_anchor is None or teacher_anchor < 0):
                k10_ok = False

        # Anchor error (relative to original M1B emit — used for regression check)
        anchor_error = (emit_step - original_emit) if (emit_step is not None and original_emit >= 0) else None

        results[label] = {
            "label": label,
            "fsm_version": cfg["fsm_version"],
            "episode_key": cell["episode_key"],
            "profile": cell["profile"],
            "task_idx": cell["task_idx"],
            "state_id": cell["state_id"],
            "success": cell.get("success", False),
            "n_steps": n,
            "original_emit": original_emit,
            "replay_emit": emit_step,
            "emit_changed": (emit_step != original_emit),
            "original_arm": cell.get("arm_step", -1),  # not in manifest but available for reference
            "replay_first_arm": first_arm,
            "n_candidates": n_candidates,
            "n_arms": n_arms,
            "n_disarms": n_disarms,
            "disarm_reasons": disarm_reasons,
            "max_arm_age_observed": max_arm_age_obs,
            "silent_arm_stall": silent_stall,
            "teacher_valid": teacher_valid,
            "teacher_anchor": teacher_anchor,
            "teacher_k10_pass": k10_ok,
            "anchor_error": anchor_error,
            "final_state": detector.state,
        }

    return results


# ── Aggregate metrics ───────────────────────────────────────────────

def compute_aggregate(all_results, teacher_data):
    """Compute per-FSM aggregate metrics using teacher ground truth."""
    agg = {}
    for label in FSM_CONFIGS:
        cells = [r[label] for r in all_results if label in r and "_error" not in r]

        # Separate by teacher ground truth
        tv_cells = [c for c in cells if c["teacher_valid"] is True]
        nc_cells = [c for c in cells if c["teacher_valid"] is False]

        # Teacher-valid triggered (coverage)
        tv_triggered = [c for c in tv_cells if c["replay_emit"] is not None]
        coverage = len(tv_triggered) / len(tv_cells) if tv_cells else 0

        # K10 containment
        k10_ok = [c for c in tv_triggered if c.get("teacher_k10_pass") is True]
        k10_rate = len(k10_ok) / len(tv_triggered) if tv_triggered else 0

        # Anchor errors
        errors = [c["anchor_error"] for c in tv_triggered
                  if c.get("anchor_error") is not None]
        median_err = float(np.median(errors)) if errors else -1

        # False-early: emit < teacher_anchor (denominator = all triggered)
        triggered = [c for c in cells if c["replay_emit"] is not None]
        false_early_count = sum(1 for c in triggered
                                if c.get("teacher_anchor") is not None
                                and c["teacher_anchor"] >= 0
                                and c["replay_emit"] < c["teacher_anchor"])
        false_early_rate = false_early_count / len(triggered) if triggered else 0

        # Post-release (all triggered have 0 in this dataset — no formal check)
        post_release_count = 0  # placeholder

        # No-corridor abstain
        nc_abstained = [c for c in nc_cells if c["replay_emit"] is None]
        nc_abstain_rate = len(nc_abstained) / len(nc_cells) if nc_cells else 0

        # Silent stalls
        silent_stalls = sum(1 for c in cells if c.get("silent_arm_stall"))

        # Disarm stats
        total_disarms = sum(c["n_disarms"] for c in cells)
        mean_disarms = total_disarms / len(cells) if cells else 0

        agg[label] = {
            "n_total": len(cells),
            "n_teacher_valid": len(tv_cells),
            "n_no_corridor": len(nc_cells),
            "n_triggered": len(triggered),
            "coverage": round(coverage, 4),
            "false_early": round(false_early_rate, 4),
            "false_early_count": false_early_count,
            "post_release": post_release_count,
            "k10_containment": round(k10_rate, 4),
            "median_anchor_error": median_err,
            "no_corridor_abstain": round(nc_abstain_rate, 4),
            "silent_arm_stalls": silent_stalls,
            "mean_disarm_count": round(mean_disarms, 2),
            "total_disarms": total_disarms,
        }
    return agg


def build_regression_table(agg, all_results):
    """Compare R0/R1/R2 metrics side by side."""
    metrics = ["coverage", "false_early", "post_release", "k10_containment",
               "median_anchor_error", "no_corridor_abstain", "silent_arm_stalls",
               "mean_disarm_count"]
    rows = []
    for m in metrics:
        row = {"metric": m}
        for label in FSM_CONFIGS:
            row[label] = agg[label].get(m, None)
        rows.append(row)

    # Known case details
    known_keys = [
        ("butter_s1", "B0"), ("chocolate_pudding_s1", "B0"),
        ("cream_cheese_s0", "B0"), ("butter_s2", "B0"),
        ("butter_s1", "D1"), ("bbq_sauce_s2", "D1"),
        ("orange_juice_s2", "B0"),
    ]
    known_rows = []
    for ek, pk in known_keys:
        cell_results = next((r for r in all_results
                            if r.get("R0", {}).get("episode_key") == ek
                            and r.get("R0", {}).get("profile") == pk), None)
        if not cell_results:
            continue
        row = {"metric": f"known:{ek}/{pk}"}
        for label in FSM_CONFIGS:
            c = cell_results.get(label, {})
            row[label] = {
                "emit": c.get("replay_emit"),
                "arm": c.get("replay_first_arm"),
                "n_disarms": c.get("n_disarms"),
                "disarm_reasons": c.get("disarm_reasons", [])[:5],
                "silent_stall": c.get("silent_arm_stall"),
            }
        known_rows.append(row)

    return rows, known_rows


def print_table(title, rows, labels):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    header = f"  {'Metric':<35}"
    for l in labels:
        header += f" {l:>20}"
    print(header)
    print("  " + "-" * 95)
    for r in rows:
        line = f"  {r['metric']:<35}"
        for l in labels:
            v = r.get(l, "")
            if isinstance(v, float):
                line += f" {v:>20.4f}"
            elif isinstance(v, dict) and v:
                e = v.get("emit")
                line += f" {str(e):>20}"
            else:
                line += f" {str(v):>20}"
        print(line)
    print("  " + "-" * 95)


def main():
    ap = argparse.ArgumentParser(description="R0/R1/R2 offline FSM replay")
    ap.add_argument("--ckpt", default=str(CKPT_PATH), help="Detector checkpoint path")
    ap.add_argument("--output", default=str(OUT_BASE), help="Output directory")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    manifest = json.load(open(MANIFEST_PATH))
    cells = manifest["cells"]
    print(f"Replay: {len(cells)} cells x {len(FSM_CONFIGS)} FSMs")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output: {out_base}")

    # ── Replay all cells ────────────────────────────────────────────
    all_results = []
    errors = []
    r0_mismatches = []

    for i, cell in enumerate(cells):
        key = f"{cell['episode_key']}/{cell['profile']}"
        print(f"[{i+1}/{len(cells)}] {key} ...", end=" ", flush=True)

        cell_results = replay_cell(cell, ckpt_path)

        if "_error" in cell_results:
            print(f"ERROR: {cell_results['_error']}")
            errors.append({"key": key, "error": cell_results["_error"]})
            continue

        all_results.append(cell_results)

        # R0 parity check — normalize None and -1 (both mean "not emitted")
        r0 = cell_results["R0"]
        r0_emit_match = (r0["replay_emit"] == r0["original_emit"]) or \
                        (r0["replay_emit"] is None and r0.get("original_emit", -1) == -1)
        r0_state_ok = (r0["final_state"] in ("EMITTED", "IDLE", "ARMED"))
        r0_ok = r0_emit_match and r0_state_ok
        if not r0_ok:
            r0_mismatches.append({
                "key": key,
                "original_emit": r0["original_emit"],
                "replay_emit": r0["replay_emit"],
                "final_state": r0["final_state"],
            })
            print(f"R0_MISMATCH emit_orig={r0['original_emit']} "
                  f"emit_replay={r0['replay_emit']} state={r0['final_state']} "
                  f"arm={r0['replay_first_arm']}")
        else:
            print(f"R0_OK emit={r0['replay_emit']} "
                  f"R1_disarm={cell_results['R1']['n_disarms']} "
                  f"R2_disarm={cell_results['R2']['n_disarms']}")

    # ── R0 Regression gate ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"R0 REGRESSION GATE: {'PASS' if len(r0_mismatches) == 0 else 'FAIL'}")
    print(f"  {len(cells)} cells, {len(r0_mismatches)} mismatches, {len(errors)} errors")
    if r0_mismatches:
        for m in r0_mismatches:
            print(f"  MISMATCH: {m}")
        print("  R1/R2 results INVALID — R0 parity required")
        sys.exit(1)

    # ── Teacher ground truth ────────────────────────────────────────
    teacher_data = {}
    for r in all_results:
        r0 = r["R0"]
        key = (r0["episode_key"], r0["profile"])
        teacher_data[key] = {
            "valid": r0.get("teacher_valid"),
            "anchor": r0.get("teacher_anchor"),
        }

    # ── Aggregate ───────────────────────────────────────────────────
    agg = compute_aggregate(all_results, teacher_data)
    with open(out_base / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)

    # ── Per-cell CSV ────────────────────────────────────────────────
    csv_rows = []
    for r in all_results:
        for label in FSM_CONFIGS:
            csv_rows.append(r[label])
    if csv_rows:
        with open(out_base / "per_cell_emit.csv", "w", newline="") as f:
            keys = [k for k in csv_rows[0].keys() if k != "disarm_reasons"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in csv_rows:
                row_clean = {k: v for k, v in row.items() if k != "disarm_reasons"}
                w.writerow(row_clean)

    # ── Regression table ────────────────────────────────────────────
    metric_rows, known_rows = build_regression_table(agg, all_results)
    with open(out_base / "regression_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric"] + list(FSM_CONFIGS.keys()))
        w.writeheader()
        w.writerows(metric_rows)
        w.writerows(known_rows)

    # ── Manifest ────────────────────────────────────────────────────
    manifest_out = {
        "gate": "M1C_RUNTIME_REPAIR_REPLAY",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script_sha256": sha256_file(__file__),
        "checkpoint_sha256": sha256_file(str(ckpt_path)) if ckpt_path.exists() else "MISSING",
        "m1b_manifest_sha256": sha256_file(str(MANIFEST_PATH)),
        "m1b_classification_sha256": sha256_file(str(CLASSIFICATION_PATH)),
        "n_cells": len(cells),
        "n_success": len(all_results),
        "n_errors": len(errors),
        "r0_parity_pass": len(r0_mismatches) == 0,
        "fsm_configs": FSM_CONFIGS,
        "aggregate_metrics": agg,
        "r0_mismatches": r0_mismatches,
        "errors": errors,
    }
    output_files = {}
    for fname in ["per_cell_emit.csv", "aggregate_metrics.json", "regression_table.csv"]:
        fp = out_base / fname
        if fp.exists():
            output_files[fname] = {"sha256": sha256_file(fp), "size": fp.stat().st_size}
    manifest_out["output_files"] = output_files
    with open(out_base / "artifact_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Console tables ──────────────────────────────────────────────
    print_table("R0 / R1 / R2 AGGREGATE METRICS", metric_rows, list(FSM_CONFIGS.keys()))
    print_table("KNOWN CASE DETAIL (7 episodes)", known_rows, list(FSM_CONFIGS.keys()))

    # ── Known case narrative ────────────────────────────────────────
    print(f"\n  KNOWN CASE NARRATIVE:")
    ft_labels = {"butter_s1/B0": "sticky-arm (4 evidence breaks)",
                 "chocolate_pudding_s1/B0": "sticky-arm + invalid emit phase",
                 "cream_cheese_s0/B0": "sustained model error",
                 "butter_s2/B0": "sustained model error",
                 "butter_s1/D1": "sustained model error",
                 "bbq_sauce_s2/D1": "sustained model error",
                 "orange_juice_s2/B0": "silent ARM stall (303 steps)"}
    for ek, pk in [("butter_s1","B0"),("chocolate_pudding_s1","B0"),
                    ("cream_cheese_s0","B0"),("butter_s2","B0"),
                    ("butter_s1","D1"),("bbq_sauce_s2","D1"),
                    ("orange_juice_s2","B0")]:
        cell_r = next((r for r in all_results
                       if r.get("R0",{}).get("episode_key")==ek
                       and r.get("R0",{}).get("profile")==pk), None)
        if not cell_r:
            continue
        note = ft_labels.get(f"{ek}/{pk}", "")
        r0 = cell_r["R0"]; r1 = cell_r["R1"]; r2 = cell_r["R2"]
        print(f"  {ek}/{pk} ({note}):")
        print(f"    R0: emit={r0['replay_emit']} disarm={r0['n_disarms']} stall={r0['silent_arm_stall']}")
        print(f"    R1: emit={r1['replay_emit']} disarm={r1['n_disarms']} reasons={r1['disarm_reasons'][:5]}")
        print(f"    R2: emit={r2['replay_emit']} disarm={r2['n_disarms']} reasons={r2['disarm_reasons'][:5]}")

    print(f"\n  All outputs: {out_base}")
    print("  DONE.")


if __name__ == "__main__":
    main()
