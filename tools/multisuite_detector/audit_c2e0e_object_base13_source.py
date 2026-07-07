#!/usr/bin/env python3
"""C2e0E: Object base13 source audit — diagnose root cause of 351/411 non-materializable rows.

Key finding from D4C2E0B vs D4C2E0D cross-reference:
  D4C2E0B says ALL 411 Object files resolve 25D via f_-prefix aliases (has_all_25d_by_alias=True).
  D4C2E0D says only 60/411 files have complete_before_recompute=True.
  Headers are IDENTICAL between both groups.

Hypothesis: f_-prefix columns in "incomplete" files contain NaN/empty values,
            while "complete" files have valid numeric data.

CPU-only. No GPU. No LIBERO. No OpenVLA. No training.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, pathlib, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))


# ==============================================================================
# Constants
# ==============================================================================
BASE13 = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

SC5_V2_FEATURES = BASE13 + [
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

F_PREFIX_CANDIDATES = [f"f_{f}" for f in SC5_V2_FEATURES]

# Columns in the temporal CSV that ARE NOT feature columns
NON_FEATURE_COLUMNS = {
    "step", "task", "state_id", "perturbation_seed", "eval_seed", "condition",
    "objective_id", "timing_policy", "trigger_source", "teacher_anchor",
    "detector_emit_step", "trigger_step_override", "effective_trigger_step",
    "attack_this", "attack_index",
    "clean_policy_action_7d", "adv_policy_action_7d_before_lock",
    "executed_policy_action_7d_after_lock", "clean_env_action_7d",
    "executed_env_action_7d", "clean_token_ids_7d", "adv_token_ids_7d",
    "target_token", "gripper_qpos_left", "gripper_qpos_right",
    "gripper_qpos_sum", "gripper_width", "raw_gripper", "env_gripper",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "object_x", "object_y", "object_z", "object_eef_distance",
    "target_x", "target_y", "target_z", "object_to_target_distance",
    "gripper_qpos", "clean_forward_ms", "pgd_optimization_ms",
    "adv_decode_ms", "arm_lock_ms", "total_step_ms",
    "attack_count", "adv_token", "prev_delta_used",
    "feat_valid", "feat_error", "detector_state",
    "corridor_p", "release_p", "pred_phase", "model_ms", "qpos_source",
    "raw_action_7d", "raw_gripper", "env_action_7d", "env_gripper",
    "clean_action_7d", "adv_policy_action_7d_before_lock",
}

# f_-prefixed feature columns we expect
F_FEATURE_ALIASES = {
    feat: f"f_{feat}" for feat in SC5_V2_FEATURES
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()


def read_csv_dict(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def value_ok(val) -> bool:
    """Check if a string value is a valid finite number."""
    if val is None or val.strip() == "":
        return False
    try:
        v = float(val)
        return math.isfinite(v)
    except (ValueError, TypeError):
        return False


def analyze_temporal_file(path, alias_map, max_rows=400):
    """Analyze a single temporal file for feature completeness and NaN rates."""
    result = {
        "path": path,
        "exists": os.path.exists(path),
        "readable": False,
        "total_rows": 0,
        "header_columns": [],
        "f_prefixed_columns": [],
        "raw_feature_columns": [],
        "per_feature_nan_rate": {},
        "per_feature_valid_count": {},
        "per_feature_mean": {},
        "per_feature_std": {},
        "any_row_all_finite": False,
        "rows_all_finite": 0,
        "rows_any_nan": 0,
        "read_error": "",
    }
    if not result["exists"]:
        result["read_error"] = "file_not_found"
        return result
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            result["header_columns"] = list(header)
            result["f_prefixed_columns"] = [c for c in header if c.startswith("f_")]
            result["raw_feature_columns"] = [c for c in header if c in SC5_V2_FEATURES]

            # Determine which columns to check for each canonical feature
            # Priority: canonical name, then f_ alias, then any alias from alias_map
            check_columns = {}
            for feat in SC5_V2_FEATURES:
                candidates = []
                if feat in header:
                    candidates.append(feat)
                if f"f_{feat}" in header:
                    candidates.append(f"f_{feat}")
                for c in candidates:
                    if c not in check_columns:
                        check_columns[feat] = c
                        break
                if feat not in check_columns:
                    check_columns[feat] = None  # no column available

            # Accumulate per-column values for stats
            feat_values = defaultdict(list)
            nan_counts = defaultdict(int)
            row_count = 0
            rows_all_finite = 0

            for row in reader:
                if row_count >= max_rows:
                    break
                row_count += 1
                all_finite = True
                for feat, col in check_columns.items():
                    if col is None:
                        nan_counts[feat] += 1
                        all_finite = False
                    else:
                        val = row.get(col, "")
                        if value_ok(val):
                            feat_values[feat].append(float(val))
                        else:
                            nan_counts[feat] += 1
                            all_finite = False
                if all_finite:
                    rows_all_finite += 1

            result["total_rows"] = row_count
            result["rows_all_finite"] = rows_all_finite
            result["rows_any_nan"] = row_count - rows_all_finite
            result["any_row_all_finite"] = rows_all_finite > 0

            for feat in SC5_V2_FEATURES:
                n = row_count
                n_nan = nan_counts[feat]
                rate = n_nan / n if n > 0 else 1.0
                result["per_feature_nan_rate"][feat] = rate
                vals = feat_values[feat]
                result["per_feature_valid_count"][feat] = len(vals)
                if vals:
                    mean = sum(vals) / len(vals)
                    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                    result["per_feature_mean"][feat] = mean
                    result["per_feature_std"][feat] = math.sqrt(variance)
                else:
                    result["per_feature_mean"][feat] = None
                    result["per_feature_std"][feat] = None

            result["readable"] = True
    except Exception as e:
        result["read_error"] = str(e)[:500]
    return result


def main():
    parser = argparse.ArgumentParser(description="C2e0E: Object base13 source audit")
    parser.add_argument("--d4c2e0d-completeness-csv", required=True,
                        help="Path to D4C2E0D object_temporal_completeness_by_artifact.csv")
    parser.add_argument("--d4c2e0b-alias-csv", required=True,
                        help="Path to D4C2E0B temporal_artifact_alias_coverage.csv")
    parser.add_argument("--d4c2e0c-endpoint-csv", required=True,
                        help="Path to D4C2E0C temporal_endpoint_resolution_by_profile.csv")
    parser.add_argument("--d4c2e0b-alias-schema", required=True,
                        help="Path to D4C2E0B temporal_feature_alias_schema.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Number of complete and incomplete files to deep-sample")
    parser.add_argument("--max-rows-per-file", type=int, default=400)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ==========================================================================
    # Load inputs
    # ==========================================================================
    completeness_rows = read_csv_dict(args.d4c2e0d_completeness_csv)
    alias_rows = read_csv_dict(args.d4c2e0b_alias_csv)
    alias_schema = json.loads(open(args.d4c2e0b_alias_schema).read())
    endpoint_rows = read_csv_dict(args.d4c2e0c_endpoint_csv)

    # Build alias lookup by path
    alias_by_path = {}
    for r in alias_rows:
        alias_by_path[r["raw_temporal_path"]] = r

    # Build endpoint lookup by path
    endpoint_by_path = {}
    for r in endpoint_rows:
        endpoint_by_path[r.get("raw_temporal_path", r.get("temporal_path", ""))] = r

    # ==========================================================================
    # Part A: Classify Object files by D4C2E0D completeness
    # ==========================================================================
    object_files = [r for r in completeness_rows if "object" in r.get("raw_temporal_path", "").lower()]

    complete_files = [r for r in object_files if r.get("complete_before_recompute", "").lower() == "true"]
    incomplete_files = [r for r in object_files if r.get("complete_before_recompute", "").lower() == "false"]

    print(f"Object files: {len(object_files)} total = {len(complete_files)} complete + {len(incomplete_files)} incomplete")

    # ==========================================================================
    # Part B: Cross-reference D4C2E0B alias with D4C2E0D completeness
    # ==========================================================================
    cross_ref = []
    for r in object_files:
        path = r["raw_temporal_path"]
        alias = alias_by_path.get(path, {})
        completeness_before = r.get("complete_before_recompute", "").lower() == "true"
        alias_25d = alias.get("has_all_25d_by_alias", "").lower() == "true"
        resolved_count = int(alias.get("resolved_feature_count", 0))
        mismatch = alias_25d and not completeness_before  # alias says OK but derivation says FAIL
        cross_ref.append({
            "temporal_path": path,
            "complete_before_recompute": completeness_before,
            "has_all_25d_by_alias": alias_25d,
            "resolved_feature_count": resolved_count,
            "alias_vs_derivation_mismatch": mismatch,
            "missing_before": r.get("missing_before", ""),
            "missing_after": r.get("missing_after", ""),
        })

    mismatch_count = sum(1 for r in cross_ref if r["alias_vs_derivation_mismatch"])
    alias_pass_count = sum(1 for r in cross_ref if r["has_all_25d_by_alias"])
    print(f"Alias pass (25/25 resolved): {alias_pass_count}/{len(cross_ref)}")
    print(f"Alias-vs-derivation MISMATCH: {mismatch_count}/{len(cross_ref)} (alias=OK but derivation=FAIL)")

    write_csv(out / "c2e0e_alias_vs_derivation_crossref.csv", cross_ref,
              ["temporal_path", "complete_before_recompute", "has_all_25d_by_alias",
               "resolved_feature_count", "alias_vs_derivation_mismatch",
               "missing_before", "missing_after"])

    # ==========================================================================
    # Part C: Path pattern analysis — are complete/incomplete from same wave?
    # ==========================================================================
    def extract_wave_info(path):
        """Extract wave, task, state from path pattern."""
        parts = path.split("/")
        wave = task = state = "unknown"
        for p in parts:
            if p.startswith("wave"):
                wave = p
            if p.startswith("task_"):
                task = p
            if p.startswith("state_"):
                state = p
        return wave, task, state

    pattern_rows = []
    for r in object_files:
        wave, task, state = extract_wave_info(r["raw_temporal_path"])
        pattern_rows.append({
            "temporal_path": r["raw_temporal_path"],
            "wave": wave,
            "task": task,
            "state": state,
            "complete_before_recompute": r.get("complete_before_recompute", "").lower() == "true",
        })

    # Group by task and check consistency
    task_completeness = defaultdict(lambda: {"complete": 0, "incomplete": 0, "states_complete": [], "states_incomplete": []})
    for r in pattern_rows:
        tc = task_completeness[r["task"]]
        if r["complete_before_recompute"]:
            tc["complete"] += 1
            tc["states_complete"].append(r["state"])
        else:
            tc["incomplete"] += 1
            tc["states_incomplete"].append(r["state"])

    mixed_tasks = {t: v for t, v in task_completeness.items()
                   if v["complete"] > 0 and v["incomplete"] > 0}
    all_complete_tasks = {t: v for t, v in task_completeness.items()
                          if v["incomplete"] == 0}
    all_incomplete_tasks = {t: v for t, v in task_completeness.items()
                            if v["complete"] == 0}

    print(f"\nPart C: Path pattern analysis")
    print(f"  All-complete tasks: {len(all_complete_tasks)}")
    print(f"  All-incomplete tasks: {len(all_incomplete_tasks)}")
    print(f"  MIXED tasks (some states complete, some incomplete): {len(mixed_tasks)}")
    for t, v in sorted(mixed_tasks.items()):
        print(f"    {t}: complete={v['complete']} ({v['states_complete']}) incomplete={v['incomplete']} ({v['states_incomplete']})")

    write_csv(out / "c2e0e_task_completeness_pattern.csv", [
        {"task": t, "complete": v["complete"], "incomplete": v["incomplete"],
         "is_mixed": t in mixed_tasks,
         "complete_states": ";".join(v["states_complete"]),
         "incomplete_states": ";".join(v["states_incomplete"])}
        for t, v in sorted(task_completeness.items())
    ], ["task", "complete", "incomplete", "is_mixed", "complete_states", "incomplete_states"])

    # ==========================================================================
    # Part D: Deep-sample data values (NaN rates per feature)
    # ==========================================================================
    # Sample complete and incomplete files
    import random
    random.seed(42)
    sample_complete = random.sample(complete_files, min(args.sample_size, len(complete_files)))
    sample_incomplete = random.sample(incomplete_files, min(args.sample_size, len(incomplete_files)))

    sample_results = []
    for label, file_list in [("complete", sample_complete), ("incomplete", sample_incomplete)]:
        for r in file_list:
            path = r["raw_temporal_path"]
            alias = alias_by_path.get(path, {})
            alias_map_json = alias.get("alias_map_json", "{}")
            try:
                alias_map = json.loads(alias_map_json) if alias_map_json else {}
            except json.JSONDecodeError:
                alias_map = {}
            result = analyze_temporal_file(path, alias_map, args.max_rows_per_file)
            result["label"] = label
            result["task"] = extract_wave_info(path)[1]
            result["state"] = extract_wave_info(path)[2]
            sample_results.append(result)
            print(f"  [{label}] {result['task']}/{result['state']}: rows={result['total_rows']} all_finite={result['rows_all_finite']} any_nan={result['rows_any_nan']}")

    # Aggregate NaN rates by label
    for label in ["complete", "incomplete"]:
        group = [r for r in sample_results if r["label"] == label]
        if not group:
            continue
        print(f"\n  === {label.upper()} aggregate (n={len(group)}) ===")
        for feat in SC5_V2_FEATURES:
            rates = [r["per_feature_nan_rate"].get(feat, 1.0) for r in group]
            avg_rate = sum(rates) / len(rates)
            if avg_rate > 0.01:
                print(f"    {feat}: avg_nan_rate={avg_rate:.4f}")

    # Write sample detail
    sample_detail_rows = []
    for r in sample_results:
        for feat in SC5_V2_FEATURES:
            sample_detail_rows.append({
                "temporal_path": r["path"],
                "label": r["label"],
                "task": r.get("task", ""),
                "state": r.get("state", ""),
                "feature": feat,
                "nan_rate": r["per_feature_nan_rate"].get(feat, 1.0),
                "valid_count": r["per_feature_valid_count"].get(feat, 0),
                "mean": r["per_feature_mean"].get(feat, ""),
                "std": r["per_feature_std"].get(feat, ""),
            })
    write_csv(out / "c2e0e_sample_feature_nan_rates.csv", sample_detail_rows,
              ["temporal_path", "label", "task", "state", "feature", "nan_rate",
               "valid_count", "mean", "std"])

    # Write sample summary
    write_csv(out / "c2e0e_sample_summary.csv", [
        {"path": r["path"], "label": r["label"], "task": r.get("task", ""),
         "state": r.get("state", ""), "total_rows": r["total_rows"],
         "rows_all_finite": r["rows_all_finite"], "rows_any_nan": r["rows_any_nan"],
         "f_prefixed_count": len(r["f_prefixed_columns"]),
         "raw_feature_count": len(r["raw_feature_columns"]),
         "read_error": r["read_error"]}
        for r in sample_results
    ], ["path", "label", "task", "state", "total_rows", "rows_all_finite",
        "rows_any_nan", "f_prefixed_count", "raw_feature_count", "read_error"])

    # ==========================================================================
    # Part E: Check endpoint profiles for Object
    # ==========================================================================
    object_endpoints = [r for r in endpoint_rows if "object" in r.get("suite", "").lower()]
    endpoint_statuses = Counter(r.get("status", r.get("endpoint_status", "unknown")) for r in object_endpoints)
    print(f"\nPart E: Object endpoint profiles: {len(object_endpoints)}")
    print(f"  Status distribution: {dict(endpoint_statuses)}")

    # ==========================================================================
    # Part F: Header comparison between complete and incomplete
    # ==========================================================================
    if sample_complete and sample_incomplete:
        c_header = set(sample_results[0]["header_columns"])
        for r in sample_results:
            if r["label"] == "complete":
                c_header = c_header & set(r["header_columns"])
        i_header = set([r for r in sample_results if r["label"] == "incomplete"][0]["header_columns"])
        for r in sample_results:
            if r["label"] == "incomplete":
                i_header = i_header & set(r["header_columns"])

        common = c_header & i_header
        only_complete = c_header - i_header
        only_incomplete = i_header - c_header
        print(f"\nPart F: Header comparison")
        print(f"  Common columns: {len(common)}")
        print(f"  Only in complete: {len(only_complete)}: {sorted(only_complete)[:20]}")
        print(f"  Only in incomplete: {len(only_incomplete)}: {sorted(only_incomplete)[:20]}")

        header_compare = {
            "common_count": len(common),
            "only_complete": sorted(only_complete),
            "only_incomplete": sorted(only_incomplete),
            "complete_sample_header": sorted(c_header),
            "incomplete_sample_header": sorted(i_header),
        }
    else:
        header_compare = {"error": "not enough samples"}

    # ==========================================================================
    # Part G: Check the D4C2E0C object_endpoint_mismatch_debug.csv
    # ==========================================================================
    debug_path = Path(args.d4c2e0c_endpoint_csv).parent / "object_endpoint_mismatch_debug.csv"
    debug_info = {}
    if debug_path.exists():
        debug_rows = read_csv_dict(str(debug_path))
        debug_info = {
            "debug_row_count": len(debug_rows),
            "debug_columns": list(debug_rows[0].keys()) if debug_rows else [],
            "first_3_rows": debug_rows[:3],
        }
        print(f"\nPart G: Object endpoint mismatch debug: {len(debug_rows)} rows")
    else:
        print(f"\nPart G: No object_endpoint_mismatch_debug.csv found at {debug_path}")

    # ==========================================================================
    # Part H: Check D4C2E0C derivation coverage for Object differences
    # ==========================================================================
    derivation_csv = Path(args.d4c2e0c_endpoint_csv).parent / "temporal_artifact_25d_derivation_coverage.csv"
    derivation_info = {}
    if derivation_csv.exists():
        d_rows = read_csv_dict(str(derivation_csv))
        obj_d_rows = [r for r in d_rows if "object" in r.get("raw_temporal_path", "").lower()]
        obj_complete_d = [r for r in obj_d_rows if r.get("complete_before_recompute", r.get("base13_complete", "")).lower() == "true"]
        obj_incomplete_d = [r for r in obj_d_rows if r.get("complete_before_recompute", r.get("base13_complete", "")).lower() != "true"]
        print(f"\nPart H: Derivation coverage Object: {len(obj_d_rows)} total = {len(obj_complete_d)} complete + {len(obj_incomplete_d)} incomplete")
        derivation_info = {
            "object_total": len(obj_d_rows),
            "object_complete": len(obj_complete_d),
            "object_incomplete": len(obj_incomplete_d),
        }

    # ==========================================================================
    # Assessment and recommendation
    # ==========================================================================
    violations = []
    warnings = []

    if mismatch_count == 0:
        violations.append("NO_ALIAS_DERIVATION_MISMATCH_FOUND: all files consistent between alias and derivation")
    else:
        warnings.append(f"ALIAS_DERIVATION_MISMATCH: {mismatch_count}/{len(cross_ref)} files have alias=OK but derivation=FAIL")

    if len(mixed_tasks) > 0:
        warnings.append(f"MIXED_TASKS: {len(mixed_tasks)} tasks have both complete and incomplete states — suggests data quality issue within same extraction wave, not systematic column absence")

    # Compute NaN stats
    incomplete_nan_feats = set()
    for r in sample_results:
        if r["label"] == "incomplete":
            for feat, rate in r["per_feature_nan_rate"].items():
                if rate > 0.5:
                    incomplete_nan_feats.add(feat)

    complete_nan_feats = set()
    for r in sample_results:
        if r["label"] == "complete":
            for feat, rate in r["per_feature_nan_rate"].items():
                if rate > 0.5:
                    complete_nan_feats.add(feat)

    nan_differential = incomplete_nan_feats - complete_nan_feats

    if nan_differential:
        violations.append(f"INCOMPLETE_ONLY_NAN_FEATURES: {sorted(nan_differential)}")
    elif incomplete_nan_feats and complete_nan_feats and incomplete_nan_feats == complete_nan_feats:
        warnings.append("SAME_NAN_FEATURES_BOTH_GROUPS: both complete and incomplete have similar NaN patterns — check derivation logic")

    all_ok = len(violations) == 0
    status = "PASS_C2E0E_OBJECT_BASE13_SOURCE_AUDITED" if all_ok else "HOLD_C2E0E_OBJECT_BASE13_SOURCE_AUDIT"

    report = {
        "status": status,
        "git_commit": args.git_commit,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "gate": "C2E0E_OBJECT_BASE13_SOURCE_AUDIT",
        "object_total": len(object_files),
        "object_complete_d4c2e0d": len(complete_files),
        "object_incomplete_d4c2e0d": len(incomplete_files),
        "alias_pass_count": alias_pass_count,
        "alias_vs_derivation_mismatch_count": mismatch_count,
        "mixed_task_count": len(mixed_tasks),
        "all_complete_task_count": len(all_complete_tasks),
        "all_incomplete_task_count": len(all_incomplete_tasks),
        "sample_complete_count": len(sample_complete),
        "sample_incomplete_count": len(sample_incomplete),
        "header_compare": header_compare,
        "endpoint_statuses": dict(endpoint_statuses),
        "debug_info": debug_info,
        "derivation_info": derivation_info,
        "incomplete_only_nan_features": sorted(nan_differential),
        "complete_only_nan_features": sorted(complete_nan_feats - incomplete_nan_feats),
        "violations": violations,
        "warnings": warnings,
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "OpenVLA_model": "NOT_LOADED",
            "detector_training": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
        },
    }

    if all_ok:
        report["recommendation"] = "proceed_to_c2e0f_repair"
    elif nan_differential:
        report["recommendation"] = "fix_derivation_pipeline_to_use_f_prefixed_aliases_or_recompute_features_from_raw_columns"
    else:
        report["recommendation"] = "investigate_derivation_pipeline_logic_for_object_specific_bug"

    write_json(out / "c2e0e_object_base13_source_audit_report.json", report)

    # ==========================================================================
    # Checksums
    # ==========================================================================
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file():
            csums[fn] = sha256_file(str(fp))
    write_json(out / "checksum_report.json", csums)
    with open(out / "SHA256SUMS", "w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS", "SHA256SUMS.sha256"):
                f.write(f"{sha}  {fn}\n")
    ss = sha256_file(str(out / "SHA256SUMS"))
    with open(out / "SHA256SUMS.sha256", "w") as f:
        f.write(f"{ss}  SHA256SUMS\n")

    print(f"\nC2E0E STATUS = {status}")
    print(f"  Object: {len(complete_files)}/{len(object_files)} complete")
    print(f"  Alias-derivation mismatch: {mismatch_count}")
    print(f"  Mixed tasks: {len(mixed_tasks)}")
    print(f"  Incomplete-only NaN features: {sorted(nan_differential)}")
    print(f"  Violations: {violations}")
    print(f"  Output: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
