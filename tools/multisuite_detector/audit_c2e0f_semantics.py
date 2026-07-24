#!/usr/bin/env python3
"""C2e0F-Semantics: Verify repaired feature semantics against f_ column ground truth.

Uses the 60 Object files that have valid f_ columns (feat_valid=True for all rows)
as ground truth. Compares C2e0F recomputed derived features against the precomputed
f_ values. Reports max_abs deviation and any semantic inversions.

Also checks raw_gripper close/open encoding direction against SC5 frozen semantics:
  SC5:  raw_gripper > OPEN_THRESHOLD → OPEN,  raw_gripper < OPEN_THRESHOLD → CLOSE
         env_gripper < -0.5 → OPEN,           env_gripper > 0.5 → CLOSE

CPU-only. No GPU. No LIBERO. No OpenVLA. No training.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, pathlib, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path("/mnt/sdc/dty_user/openvla_attack")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/tmp")
from repair_c2e0f_object_base13_recompute import (
    extract_base13_from_raw, compute_derived_features,
    SC5_V2_FEATURES, BASE13, DERIVED_12,
    read_csv_dict, write_csv, write_json, sha256_file,
)

OPEN_THRESHOLD_RAW = 0.5


def parse_action_7d(val):
    if val is None or val.strip() == "":
        return None
    v = val.strip()
    if v.startswith("["):
        try:
            parts = json.loads(v)
            if len(parts) >= 7:
                return tuple(float(x) for x in parts)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    try:
        parts = [float(x.strip()) for x in v.split(",")]
        if len(parts) >= 7:
            return tuple(parts)
    except (ValueError, TypeError):
        pass
    return None


def analyze_gripper_semantics(temporal_path):
    """Analyze raw_gripper and env_gripper distributions in a temporal file."""
    rows = read_csv_dict(temporal_path)
    raw_vals = []
    env_vals = []
    feat_valid_counts = Counter()
    for r in rows:
        rg = r.get("raw_gripper", "").strip()
        eg = r.get("env_gripper", "").strip()
        fv = r.get("feat_valid", "")
        feat_valid_counts[fv] += 1
        if rg:
            try:
                raw_vals.append(float(rg))
            except (ValueError, TypeError):
                pass
        if eg:
            try:
                env_vals.append(float(eg))
            except (ValueError, TypeError):
                pass

    if not raw_vals:
        return {"error": "no raw_gripper values"}

    raw_sorted = sorted(raw_vals)
    env_sorted = sorted(env_vals) if env_vals else []
    n = len(raw_sorted)

    # Determine encoding:
    # If raw_gripper uses raw action space:  >0.5 = OPEN
    # If raw_gripper uses env action space:  <-0.5 = OPEN
    # We can detect by comparing raw_gripper with clean_action_7d[6]
    action_vals = []
    for r in rows:
        ca = r.get("clean_action_7d", "")
        parsed = parse_action_7d(ca)
        if parsed:
            action_vals.append(parsed[6])  # gripper action
    action_sorted = sorted(action_vals) if action_vals else []

    # Compare: if raw_gripper ≈ clean_action_7d[6], it's raw action space
    raw_mean = sum(raw_vals) / n
    action_mean = sum(action_vals) / len(action_vals) if action_vals else None

    raw_action_match = action_mean is not None and abs(raw_mean - action_mean) < 0.1

    # Classify using both conventions
    raw_open_sc5 = sum(1 for v in raw_vals if v > OPEN_THRESHOLD_RAW)
    raw_close_sc5 = sum(1 for v in raw_vals if v < OPEN_THRESHOLD_RAW)
    env_open = sum(1 for v in env_vals if v < -0.5) if env_vals else 0
    env_close = sum(1 for v in env_vals if v > 0.5) if env_vals else 0

    return {
        "n_rows": len(rows),
        "feat_valid_counts": dict(feat_valid_counts),
        "raw_gripper_min": raw_sorted[0],
        "raw_gripper_max": raw_sorted[-1],
        "raw_gripper_median": raw_sorted[n // 2],
        "raw_gripper_mean": raw_mean,
        "raw_gripper_unique_approx": len(set(round(v, 6) for v in raw_vals)),
        "raw_gripper_open_sc5": raw_open_sc5,
        "raw_gripper_close_sc5": raw_close_sc5,
        "env_gripper_min": env_sorted[0] if env_sorted else None,
        "env_gripper_max": env_sorted[-1] if env_sorted else None,
        "env_gripper_open": env_open,
        "env_gripper_close": env_close,
        "clean_action_7d_gripper_mean": action_mean,
        "raw_matches_action_7d_gripper": raw_action_match,
        "space_detected": "raw_action" if raw_action_match else "unknown",
    }


def compare_derived_features(temporal_path, max_rows=400):
    """Compare C2e0F derived features against f_ column ground truth."""
    rows = read_csv_dict(temporal_path)
    n = min(len(rows), max_rows)
    rows = rows[:n]

    # Extract base13 from raw columns
    base13_stream = []
    for row in rows:
        feats, missing = extract_base13_from_raw(row)
        base13_stream.append(feats)

    # Compute C2e0F derived features
    c2e0f_derived = compute_derived_features(base13_stream)

    # Read f_ column ground truth
    f_col_map = {feat: f"f_{feat}" for feat in DERIVED_12}

    diffs = defaultdict(list)
    for i in range(n):
        row = rows[i]
        c2e0f = c2e0f_derived[i]
        for feat in DERIVED_12:
            f_col = f_col_map[feat]
            f_val_str = row.get(f_col, "").strip()
            if not f_val_str or f_val_str == "None":
                continue  # no ground truth for this row

            try:
                f_val = float(f_val_str)
            except (ValueError, TypeError):
                continue

            c2e0f_val = c2e0f.get(feat, None)
            if c2e0f_val is None:
                continue

            diff = abs(c2e0f_val - f_val)
            diffs[feat].append(diff)

    # Aggregate
    comparison = {}
    for feat in DERIVED_12:
        dlist = diffs.get(feat, [])
        if not dlist:
            comparison[feat] = {"n_compared": 0, "max_abs": None, "mean_abs": None, "status": "no_ground_truth"}
        else:
            max_abs = max(dlist)
            mean_abs = sum(dlist) / len(dlist)
            status = "PASS" if max_abs < 0.01 else ("WARN" if max_abs < 0.1 else "FAIL")
            comparison[feat] = {
                "n_compared": len(dlist),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "status": status,
            }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="C2e0F-Semantics: repair semantics audit")
    parser.add_argument("--d4c2e0d-completeness-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load completeness data
    completeness_rows = read_csv_dict(args.d4c2e0d_completeness_csv)
    object_rows = [r for r in completeness_rows if "object" in r.get("raw_temporal_path", "").lower()]
    complete_files = [r for r in object_rows if r.get("complete_before_recompute", "").lower() == "true"]
    incomplete_files = [r for r in object_rows if r.get("complete_before_recompute", "").lower() == "false"]

    print(f"Object files: {len(object_rows)} ({len(complete_files)} complete, {len(incomplete_files)} incomplete)")

    # ==========================================================================
    # Part A: Gripper semantics analysis
    # ==========================================================================
    print("\n=== Part A: Gripper semantics ===")
    import random
    random.seed(42)
    sample_for_semantics = random.sample(complete_files, min(args.sample_size, len(complete_files)))

    semantics_results = []
    for r in sample_for_semantics:
        path = r["raw_temporal_path"]
        sem = analyze_gripper_semantics(path)
        sem["temporal_path"] = path
        semantics_results.append(sem)
        print(f"  {path.split('/')[-3]}: raw=[{sem.get('raw_gripper_min',0):.4f},{sem.get('raw_gripper_max',1):.4f}] "
              f"open_sc5={sem.get('raw_gripper_open_sc5',0)} close_sc5={sem.get('raw_gripper_close_sc5',0)} "
              f"env=[{sem.get('env_gripper_min',0):.4f},{sem.get('env_gripper_max',1):.4f}] "
              f"space={sem.get('space_detected','?')}")

    # Determine dominant encoding
    space_votes = Counter(r.get("space_detected", "?") for r in semantics_results if "error" not in r)
    dominant_space = space_votes.most_common(1)[0][0] if space_votes else "unknown"
    print(f"  Dominant space: {dominant_space} ({dict(space_votes)})")

    # Write semantics summary
    sem_fields = ["temporal_path", "n_rows", "feat_valid_counts",
                   "raw_gripper_min", "raw_gripper_max", "raw_gripper_median", "raw_gripper_mean",
                   "raw_gripper_open_sc5", "raw_gripper_close_sc5",
                   "env_gripper_min", "env_gripper_max", "env_gripper_open", "env_gripper_close",
                   "clean_action_7d_gripper_mean", "raw_matches_action_7d_gripper", "space_detected"]
    write_csv(out / "c2e0f_semantics_gripper_analysis.csv", semantics_results, sem_fields)

    # ==========================================================================
    # Part B: Derived feature comparison against f_ ground truth
    # ==========================================================================
    print("\n=== Part B: Derived feature comparison ===")
    all_comparisons = []
    for r in sample_for_semantics:
        path = r["raw_temporal_path"]
        comp = compare_derived_features(path)
        comp["temporal_path"] = path
        all_comparisons.append(comp)

    # Aggregate
    agg = {}
    for feat in DERIVED_12:
        max_vals = []
        mean_vals = []
        n_total = 0
        fail_count = 0
        for comp in all_comparisons:
            c = comp.get(feat, {})
            if c.get("max_abs") is not None:
                max_vals.append(c["max_abs"])
                mean_vals.append(c["mean_abs"])
                n_total += c.get("n_compared", 0)
            if c.get("status") == "FAIL":
                fail_count += 1

        if max_vals:
            agg[feat] = {
                "n_files_compared": len(max_vals),
                "n_values_compared": n_total,
                "max_max_abs": max(max_vals),
                "mean_mean_abs": sum(mean_vals) / len(mean_vals),
                "files_failing": fail_count,
                "status": "PASS" if max(max_vals) < 0.01 else ("WARN" if max(max_vals) < 0.1 else "FAIL"),
            }
        else:
            agg[feat] = {"status": "no_data", "n_files_compared": 0}

        status = agg[feat]["status"]
        detail = f"max={agg[feat].get('max_max_abs','?'):.6f} mean={agg[feat].get('mean_mean_abs','?'):.6f}"
        print(f"  {feat}: {status} ({detail})")

    # Write comparison CSV
    comp_rows = []
    for feat in DERIVED_12:
        a = agg.get(feat, {})
        comp_rows.append({
            "feature": feat,
            "n_files_compared": a.get("n_files_compared", 0),
            "n_values_compared": a.get("n_values_compared", 0),
            "max_max_abs": a.get("max_max_abs", ""),
            "mean_mean_abs": a.get("mean_mean_abs", ""),
            "files_failing": a.get("files_failing", 0),
            "status": a.get("status", ""),
        })
    write_csv(out / "c2e0f_semantics_derived_comparison.csv", comp_rows,
              ["feature", "n_files_compared", "n_values_compared",
               "max_max_abs", "mean_mean_abs", "files_failing", "status"])

    # ==========================================================================
    # Part C: Check close/open direction specifically
    # ==========================================================================
    print("\n=== Part C: Close/open direction check ===")
    # For a sample complete file, check if f_close_onset matches SC5 convention
    # SC5: raw_gripper > 0.5 → OPEN, so close onset happens when going from open to close:
    #   close_onset = raw_gripper goes from >0.5 to <0.5
    # C2e0F: close_onset = cmd goes from <=0.5 to >0.5
    # These are INVERTED if raw_gripper uses raw action space.

    direction_checks = []
    for r in sample_for_semantics[:5]:
        path = r["raw_temporal_path"]
        rows = read_csv_dict(path)[:100]
        # Compare f_close_onset with both interpretations
        sc5_onset = []  # close = raw < 0.5, onset = transition
        c2e0f_onset = []  # close = raw > 0.5, onset = transition
        prev = None
        for row in rows:
            rg = row.get("raw_gripper", "").strip()
            f_co = row.get("f_close_onset", "").strip()
            try:
                rg_v = float(rg)
                f_co_v = float(f_co) if f_co else 0.0
            except (ValueError, TypeError):
                sc5_onset.append(None)
                c2e0f_onset.append(None)
                prev = None
                continue

            if prev is not None:
                # SC5: raw_gripper > 0.5 = OPEN, so close onset when raw crosses below 0.5
                sc5_on = 1.0 if (prev > OPEN_THRESHOLD_RAW and rg_v < OPEN_THRESHOLD_RAW) else 0.0
                # C2e0F: raw_gripper > 0.5 = CLOSE, so close onset when raw crosses above 0.5
                c2e0f_on = 1.0 if (prev < OPEN_THRESHOLD_RAW and rg_v > OPEN_THRESHOLD_RAW) else 0.0
                sc5_onset.append(sc5_on)
                c2e0f_onset.append(c2e0f_on)

            prev = rg_v

        sc5_match = sum(1 for s, f in zip(sc5_onset, [float(r.get("f_close_onset","0") or "0") for r in rows[1:100]]) if s == f)
        c2e0f_match = sum(1 for s, f in zip(c2e0f_onset, [float(r.get("f_close_onset","0") or "0") for r in rows[1:100]]) if s == f)
        n_compared = len([x for x in sc5_onset if x is not None])

        direction_checks.append({
            "temporal_path": path,
            "n_rows_checked": n_compared,
            "sc5_convention_match": sc5_match,
            "c2e0f_convention_match": c2e0f_match,
            "better_convention": "SC5" if sc5_match > c2e0f_match else ("C2E0F" if c2e0f_match > sc5_match else "TIE"),
        })
        print(f"  {path.split('/')[-3]}: SC5_match={sc5_match}/{n_compared} C2e0F_match={c2e0f_match}/{n_compared} → {direction_checks[-1]['better_convention']}")

    write_csv(out / "c2e0f_semantics_direction_check.csv", direction_checks,
              ["temporal_path", "n_rows_checked", "sc5_convention_match", "c2e0f_convention_match", "better_convention"])

    # ==========================================================================
    # Assessment
    # ==========================================================================
    violations = []
    warnings_list = []

    # Check gripper semantics
    if dominant_space == "raw_action":
        # raw_gripper > 0.5 = OPEN, so C2e0F using >0.5 as CLOSE is WRONG
        violations.append("GRIPPER_CLOSE_OPEN_SEMANTICS_INVERTED: raw_gripper uses raw action space (>0.5=OPEN), "
                          "C2e0F treats >0.5 as CLOSE. close_onset, recent_close_streak, recent_open_streak, "
                          "recent_gripper_flip_count all inverted.")
    elif dominant_space == "unknown":
        warnings_list.append("GRIPPER_SPACE_UNKNOWN: cannot determine if raw_gripper uses raw or env action space")

    # Check derived feature fidelity
    fail_features = [feat for feat, a in agg.items() if a.get("status") == "FAIL"]
    warn_features = [feat for feat, a in agg.items() if a.get("status") == "WARN"]
    if fail_features:
        violations.append(f"DERIVED_FEATURE_DEVIATION_FAIL: {fail_features}")
    if warn_features:
        warnings_list.append(f"DERIVED_FEATURE_DEVIATION_WARN: {warn_features}")

    direction_better = Counter(r["better_convention"] for r in direction_checks).most_common(1)
    if direction_better and direction_better[0][0] == "C2E0F":
        warnings_list.append("C2E0F convention matches f_ ground truth better than SC5 — possible Object-specific encoding")
    elif direction_better and direction_better[0][0] == "SC5":
        violations.append("SC5 convention better matches f_ ground truth, C2e0F has inverted close/open")

    all_ok = len(violations) == 0
    status = "PASS_C2E0F_SEMANTICS" if all_ok else "HOLD_C2E0F_SEMANTICS"

    report = {
        "status": status,
        "git_commit": args.git_commit,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "gate": "C2E0F_SEMANTICS_AUDIT",
        "dominant_gripper_space": dominant_space,
        "space_votes": dict(space_votes),
        "derived_feature_aggregate": agg,
        "direction_check_summary": dict(Counter(r["better_convention"] for r in direction_checks)),
        "violations": violations,
        "warnings": warnings_list,
        "recommendation": ("proceed_c2e0g2" if all_ok else
                          "fix_c2e0f_close_open_inversion_then_re_repair" if "INVERTED" in str(violations) else
                          "investigate_further"),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "OpenVLA_model": "NOT_LOADED",
            "detector_training": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
        },
    }

    if not all_ok:
        report["fix_instructions"] = (
            "In repair_c2e0f_object_base13_recompute.py compute_derived_features():\n"
            "  Change close condition: cmd_vals[i] < OPEN_THRESHOLD_RAW (SC5: raw < thresh = CLOSE)\n"
            "  Change open condition:  cmd_vals[i] > OPEN_THRESHOLD_RAW (SC5: raw > thresh = OPEN)\n"
            "  This flips: close_onset, recent_close_streak, recent_open_streak, recent_gripper_flip_count\n"
            "  Also flip: time_since_close → time_since_open or re-derive from correct close onset\n"
            "  Then re-run C2e0F to regenerate all 351 repaired feature CSVs."
        )

    write_json(out / "c2e0f_semantics_audit_report.json", report)

    # Checksums
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file() and not fn.endswith(".csv"):
            csums[fn] = sha256_file(str(fp))
    for fn in ["c2e0f_semantics_gripper_analysis.csv", "c2e0f_semantics_derived_comparison.csv",
               "c2e0f_semantics_direction_check.csv"]:
        fp = out / fn
        if fp.exists():
            csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)

    print(f"\nC2E0F-Semantics STATUS = {status}")
    print(f"  Dominant space: {dominant_space}")
    print(f"  Violations: {violations}")
    print(f"  Warnings: {warnings_list}")
    print(f"  Output: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
