#!/usr/bin/env python3
"""C2e0G2: Pipeline integration with true C2e0C endpoints and full causal window validation.

Key fixes over C2e0G:
  A. Uses C2e0C endpoint audit for real label-aligned endpoint indices
  B. Validates full causal window [endpoint-W+1, ..., endpoint] for W in [8,16,32]
  C. No silent endpoint clamping — reports as not_materializable if window can't be satisfied
  D. Correct label classification: VALID_PRIMARY and VALID_PRIMARY_CANDIDATE = has_primary

CPU-only. No GPU. No LIBERO. No OpenVLA. No training.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, pathlib, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path("/mnt/sdc/dty_user/openvla_attack")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))
sys.path.insert(0, "/tmp")

from repair_c2e0f_object_base13_recompute import (
    extract_base13_from_raw, compute_derived_features,
    SC5_V2_FEATURES, BASE13, DERIVED_12,
    read_csv_dict, write_csv, write_json, sha256_file,
)

WINDOW_SIZES = [8, 16, 32]
PRIMARY_LABELS = {"VALID_PRIMARY", "VALID_PRIMARY_CANDIDATE"}


def is_finite(v):
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (ValueError, TypeError):
        return False


def check_window_complete(trows, start_idx, end_idx, is_repaired, suite):
    """Check that ALL rows in [start_idx, end_idx] have complete finite 25D features."""
    if start_idx < 0 or end_idx >= len(trows):
        return False, f"window_out_of_bounds:[{start_idx},{end_idx}]_n={len(trows)}", 0

    complete_count = 0
    for i in range(start_idx, end_idx + 1):
        row = trows[i]
        if is_repaired:
            # Read features directly from repaired CSV columns
            all_ok = True
            for feat in SC5_V2_FEATURES:
                v = row.get(feat, "")
                if not v or v == "None" or not is_finite(v):
                    all_ok = False
                    break
            if all_ok:
                complete_count += 1
        elif suite == "libero_object":
            # Object files: use raw column extraction
            feats, missing = extract_base13_from_raw(row)
            if not missing:
                all_ok = True
                for feat in BASE13:
                    if not is_finite(feats.get(feat)):
                        all_ok = False
                        break
                if all_ok:
                    complete_count += 1
        else:
            # Non-Object suites: try v3 compute_features first, then check f_ columns
            all_ok = True
            # Check f_-prefixed columns for all 25D features
            for feat in SC5_V2_FEATURES:
                f_col = f"f_{feat}"
                v = row.get(f_col, row.get(feat, ""))
                if not v or v == "None" or not is_finite(v):
                    all_ok = False
                    break
            if all_ok:
                complete_count += 1

    window_ok = complete_count == (end_idx - start_idx + 1)
    return window_ok, "" if window_ok else f"{complete_count}/{end_idx-start_idx+1}_rows_complete", complete_count


def main():
    parser = argparse.ArgumentParser(description="C2e0G2: Pipeline integration with true endpoints")
    parser.add_argument("--context-dataset", required=True)
    parser.add_argument("--repair-manifest", required=True)
    parser.add_argument("--c2e0c-endpoint-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load data
    context_rows = read_csv_dict(args.context_dataset)
    manifest_rows = read_csv_dict(args.repair_manifest)
    endpoint_rows = read_csv_dict(args.c2e0c_endpoint_csv)

    # Build repair lookup
    repair_map = {}
    for r in manifest_rows:
        if r.get("status") == "REPAIRED":
            repair_map[r["original_temporal_path"]] = r["repaired_feature_path"]

    # C2e0C endpoint CSV is aligned 1:1 with context dataset by row_index
    # Build endpoint lookup by row_index (int)
    endpoint_by_idx = {}
    for r in endpoint_rows:
        try:
            idx = int(r.get("row_index", -1))
            if idx >= 0:
                endpoint_by_idx[idx] = r
        except (ValueError, TypeError):
            pass

    print(f"Context: {len(context_rows)} rows")
    print(f"Repair map: {len(repair_map)} entries")

    # Process
    temporal_cache = {}
    results = []
    suite_window_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "complete": 0}))
    exclusion_candidates = []
    repair_usage = []

    for i, row in enumerate(context_rows):
        suite = row.get("suite", "?")
        group_key = row.get("group_key", "")
        label_status = row.get("teacher_label_status", "")
        tpath = row.get("temporal_path", "")
        split = row.get("split", "")

        # Resolve path
        actual_path = repair_map.get(tpath, tpath)
        used_repair = actual_path != tpath

        if used_repair:
            repair_usage.append({"row_index": i, "suite": suite, "group_key": group_key,
                                 "original_path": tpath, "repaired_path": actual_path})

        # Get endpoint from C2e0C audit (aligned by row_index)
        ep_info = endpoint_by_idx.get(i, {})
        endpoint_str = ep_info.get("endpoint_index", "")
        endpoint = None
        if endpoint_str and endpoint_str.strip() not in ("", "-1", "None"):
            try:
                endpoint = int(endpoint_str)
            except (ValueError, TypeError):
                pass

        # If no endpoint, we cannot validate — mark for exclusion
        if endpoint is None:
            exclusion_candidates.append({
                "row_index": i, "suite": suite, "group_key": group_key,
                "label_status": label_status, "split": split,
                "temporal_path": tpath, "reason": "no_c2e0c_endpoint",
            })
            continue

        # Load temporal data
        if not actual_path:
            exclusion_candidates.append({
                "row_index": i, "suite": suite, "group_key": group_key,
                "label_status": label_status, "split": split,
                "temporal_path": tpath, "reason": "no_temporal_path",
            })
            continue

        try:
            if actual_path not in temporal_cache:
                temporal_cache[actual_path] = read_csv_dict(actual_path)
            trows = temporal_cache[actual_path]
        except Exception as e:
            exclusion_candidates.append({
                "row_index": i, "suite": suite, "group_key": group_key,
                "label_status": label_status, "split": split,
                "temporal_path": tpath, "used_path": actual_path,
                "reason": f"read_error:{str(e)[:100]}",
            })
            continue

        n_trows = len(trows)
        is_repaired = "_all_features_present" in trows[0] if trows else False

        # Check each window size
        window_results = {}
        all_windows_ok = True
        for W in WINDOW_SIZES:
            start_idx = endpoint - W + 1
            if start_idx < 0:
                window_results[f"W{W}"] = "endpoint_too_early"
                all_windows_ok = False
                continue

            ok, detail, _ = check_window_complete(trows, start_idx, endpoint, is_repaired, suite)
            window_results[f"W{W}"] = "PASS" if ok else f"INCOMPLETE:{detail}"
            if not ok:
                all_windows_ok = False

        # Per-suite stats
        for W in WINDOW_SIZES:
            wk = f"W{W}"
            suite_window_stats[suite][W]["total"] += 1
            if window_results.get(wk) == "PASS":
                suite_window_stats[suite][W]["complete"] += 1

        # Classify
        is_primary = label_status in PRIMARY_LABELS
        group_class = "has_primary" if is_primary else "no_primary"

        results.append({
            "row_index": i, "suite": suite, "group_key": group_key,
            "label_status": label_status, "group_class": group_class,
            "split": split, "temporal_path": tpath, "used_repaired": used_repair,
            "endpoint": endpoint, "n_temporal_rows": n_trows,
            "W8": window_results.get("W8", "?"),
            "W16": window_results.get("W16", "?"),
            "W32": window_results.get("W32", "?"),
            "all_windows_ok": all_windows_ok,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(context_rows)}...")

    dt = time.time() - t0

    # =========================================================================
    # Summary
    # =========================================================================
    total_validated = len(results)
    total_all_ok = sum(1 for r in results if r["all_windows_ok"])

    print(f"\n=== C2e0G2 Results ({dt:.1f}s) ===")
    for suite in sorted(suite_window_stats.keys()):
        sw = suite_window_stats[suite]
        parts = []
        for W in WINDOW_SIZES:
            c = sw[W]["complete"]
            t = sw[W]["total"]
            pct = c / t if t > 0 else 0
            parts.append(f"W{W}={c}/{t}({pct:.3f})")
        print(f"  {suite}: {', '.join(parts)}")

    # Overall
    for W in WINDOW_SIZES:
        total = sum(suite_window_stats[s][W]["total"] for s in suite_window_stats)
        complete = sum(suite_window_stats[s][W]["complete"] for s in suite_window_stats)
        pct = complete / total if total > 0 else 0
        print(f"  OVERALL W{W}: {complete}/{total} ({pct:.4f})")

    # Object check
    obj_stats = suite_window_stats.get("libero_object", {})
    obj_w32 = obj_stats.get(32, {"total": 0, "complete": 0})
    obj_rate = obj_w32["complete"] / obj_w32["total"] if obj_w32["total"] > 0 else 0
    print(f"  OBJECT W32: {obj_w32['complete']}/{obj_w32['total']} ({obj_rate:.4f})")

    # =========================================================================
    # Assessment
    # =========================================================================
    violations = []
    warnings_list = []

    # Check per-suite W32 >= 0.95
    suite_pass = {}
    for suite in sorted(suite_window_stats.keys()):
        sw = suite_window_stats[suite].get(32, {"total": 0, "complete": 0})
        rate = sw["complete"] / sw["total"] if sw["total"] > 0 else 0
        suite_pass[suite] = rate >= 0.95
        if not suite_pass[suite]:
            violations.append(f"SUITE_{suite}_W32_BELOW_95: {rate:.4f}")

    overall_w32 = total_all_ok / total_validated if total_validated > 0 else 0
    if overall_w32 < 0.95:
        violations.append(f"OVERALL_W32_BELOW_95: {overall_w32:.4f}")

    # Check no silent clamping
    for r in results:
        for W in WINDOW_SIZES:
            wk = f"W{W}"
            if r[wk] == "endpoint_too_early":
                warnings_list.append(f"ENDPOINT_TOO_EARLY_W{W}: suite={r['suite']} gk={r['group_key']}")
                break

    # Check split leakage
    split_group = defaultdict(set)
    for r in results:
        split_group[r["group_key"]].add(r["split"])
    leakage = sum(1 for g, s in split_group.items() if len(s) > 1)
    if leakage > 0:
        violations.append(f"SPLIT_LEAKAGE: {leakage}")

    # Check exclusion
    n_excluded = len(exclusion_candidates)
    if n_excluded > 0:
        warnings_list.append(f"EXCLUDED_ROWS: {n_excluded} rows excluded (no endpoint, no path, read error)")
        by_reason = Counter(r["reason"] for r in exclusion_candidates)
        for reason, count in by_reason.most_common():
            warnings_list.append(f"  exclusion_reason: {reason}={count}")

    all_ok = len(violations) == 0
    status = "PASS_C2E0G2" if all_ok else "HOLD_C2E0G2"

    # =========================================================================
    # Outputs
    # =========================================================================
    write_csv(out / "c2e0g2_window_completeness_by_row.csv", results,
              ["row_index", "suite", "group_key", "label_status", "group_class",
               "split", "temporal_path", "used_repaired", "endpoint",
               "n_temporal_rows", "W8", "W16", "W32", "all_windows_ok"])

    suite_summary = []
    for suite in sorted(suite_window_stats.keys()):
        row = {"suite": suite}
        for W in WINDOW_SIZES:
            sw = suite_window_stats[suite][W]
            row[f"W{W}_total"] = sw["total"]
            row[f"W{W}_complete"] = sw["complete"]
            row[f"W{W}_rate"] = sw["complete"] / sw["total"] if sw["total"] > 0 else 0
        suite_summary.append(row)

    fields = ["suite"] + [f"W{W}_{k}" for W in WINDOW_SIZES for k in ["total", "complete", "rate"]]
    write_csv(out / "c2e0g2_window_completeness_by_suite.csv", suite_summary, fields)

    if exclusion_candidates:
        write_csv(out / "c2e0g2_exclusion_manifest.csv", exclusion_candidates,
                  ["row_index", "suite", "group_key", "label_status", "split", "temporal_path", "reason"])

    if repair_usage:
        write_csv(out / "c2e0g2_repair_usage_manifest.csv", repair_usage,
                  ["row_index", "suite", "group_key", "original_path", "repaired_path"])

    report = {
        "status": status,
        "git_commit": args.git_commit,
        "created_at_unix": time.time(),
        "runtime_seconds": dt,
        "gate": "C2E0G2_PIPELINE_INTEGRATION_VERIFICATION",
        "total_context_rows": len(context_rows),
        "total_validated": total_validated,
        "total_excluded": n_excluded,
        "total_all_windows_ok": total_all_ok,
        "overall_w32_rate": overall_w32,
        "object_w32_rate": obj_rate,
        "suite_w32_rates": suite_pass,
        "split_leakage_count": leakage,
        "exclusion_by_reason": dict(Counter(r["reason"] for r in exclusion_candidates)),
        "repair_files_used": len(set(r["repaired_path"] for r in repair_usage)),
        "window_sizes": WINDOW_SIZES,
        "violations": violations,
        "warnings": warnings_list,
        "recommendation": ("proceed_to_c2e1" if all_ok else
                          "fix_violations_before_c2e1"),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "OpenVLA_model": "NOT_LOADED",
            "detector_training": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
        },
    }
    write_json(out / "c2e0g2_pipeline_integration_report.json", report)

    # Checksums
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file() and not fn.endswith(".csv"):
            csums[fn] = sha256_file(str(fp))
    for fn in ["c2e0g2_window_completeness_by_suite.csv", "c2e0g2_exclusion_manifest.csv",
               "c2e0g2_repair_usage_manifest.csv"]:
        fp = out / fn
        if fp.exists():
            csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)

    print(f"\nC2E0G2 STATUS = {status}")
    print(f"  Validated: {total_validated}/{len(context_rows)}, All Pass: {total_all_ok}")
    print(f"  Excluded: {n_excluded}, Split Leakage: {leakage}")
    print(f"  Object W32: {obj_rate:.4f}, Overall W32: {overall_w32:.4f}")
    print(f"  Violations: {violations}")
    print(f"  Output: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
