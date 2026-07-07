#!/usr/bin/env python3
"""C2e0F: Object base13 feature recomputation repair.

Root cause (from C2e0E): 351/411 Object temporal files have feat_valid=False
for ~42% of rows, causing f_-prefixed feature columns to contain NaN.
But raw sensor columns (eef_x/y/z, eef_vx/vy/vz, gripper_qpos, gripper_width,
raw_gripper, clean_action_7d) have 0% NaN in ALL files.

Repair strategy:
  - Read raw sensor columns from temporal CSV (ignore f_ prefixed cache)
  - Parse clean_action_7d JSON → action_dx, action_dy, action_dz, action_gripper
  - Use raw_gripper → gripper_command
  - Use gripper_width → gripper_opening_proxy
  - Compute derived 12 features from base13 using the same derivation logic as D4C2E0C
  - Write repaired feature cache file alongside original temporal CSV

CPU-only. No GPU. No LIBERO. No OpenVLA. No training.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, pathlib, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "multisuite_detector"))


# ==============================================================================
# Feature definitions
# ==============================================================================
BASE13 = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

# The 12 derived features computed from base13 stream
DERIVED_12 = [
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

SC5_V2_FEATURES = BASE13 + DERIVED_12


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


def parse_action_7d(val: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse clean_action_7d string to (dx, dy, dz, gripper)."""
    if val is None or val.strip() == "":
        return None
    v = val.strip()
    # Handle JSON list format: [dx, dy, dz, dr, dp, dyaw, gripper]
    if v.startswith("["):
        try:
            parts = json.loads(v)
            if len(parts) >= 7:
                return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[6]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Handle comma-separated format
    try:
        parts = [float(x.strip()) for x in v.split(",")]
        if len(parts) >= 7:
            return (parts[0], parts[1], parts[2], parts[6])
    except (ValueError, TypeError):
        pass
    return None


def safe_float(val: str) -> Optional[float]:
    """Parse string to float, returning None on failure."""
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ==============================================================================
# Base13 extraction from raw columns
# ==============================================================================
def extract_base13_from_raw(row: Dict[str, str]) -> Tuple[Dict[str, float], List[str]]:
    """Extract base13 features from raw sensor columns, bypassing f_ cache.

    Returns (features_dict, missing_features_list).
    """
    features = {}
    missing = []

    # Direct sensor copies
    for feat, col in [("eef_x", "eef_x"), ("eef_y", "eef_y"), ("eef_z", "eef_z"),
                       ("eef_vx", "eef_vx"), ("eef_vy", "eef_vy"), ("eef_vz", "eef_vz"),
                       ("gripper_qpos", "gripper_qpos")]:
        v = safe_float(row.get(col, ""))
        if v is not None:
            features[feat] = v
        else:
            missing.append(feat)

    # gripper_command from raw_gripper
    v = safe_float(row.get("raw_gripper", ""))
    if v is not None:
        features["gripper_command"] = v
    else:
        # Fallback: env_gripper
        v2 = safe_float(row.get("env_gripper", ""))
        if v2 is not None:
            features["gripper_command"] = v2
        else:
            missing.append("gripper_command")

    # gripper_opening_proxy from gripper_width
    v = safe_float(row.get("gripper_width", ""))
    if v is not None:
        features["gripper_opening_proxy"] = v
    else:
        missing.append("gripper_opening_proxy")

    # action_dx/dy/dz/gripper from clean_action_7d
    action_val = row.get("clean_action_7d", "")
    parsed = parse_action_7d(action_val)
    if parsed is not None:
        features["action_dx"] = parsed[0]
        features["action_dy"] = parsed[1]
        features["action_dz"] = parsed[2]
        features["action_gripper"] = parsed[3]
    else:
        # Fallback: executed_env_action_7d
        exec_val = row.get("executed_env_action_7d", "")
        parsed2 = parse_action_7d(exec_val)
        if parsed2 is not None:
            features["action_dx"] = parsed2[0]
            features["action_dy"] = parsed2[1]
            features["action_dz"] = parsed2[2]
            features["action_gripper"] = parsed2[3]
        else:
            missing.extend(["action_dx", "action_dy", "action_dz", "action_gripper"])

    return features, missing


# ==============================================================================
# Derived feature computation (from base13 stream)
# ==============================================================================
def compute_derived_features(base13_stream: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Compute the 12 derived features from base13 temporal stream.

    base13_stream: list of per-row base13 feature dicts, one per timestep.
    Returns: list of per-row derived feature dicts (same length as input).
    """
    n = len(base13_stream)
    derived = [{} for _ in range(n)]

    # Pre-extract arrays
    eef_z_vals = [r.get("eef_z", 0.0) for r in base13_stream]
    eef_speed_vals = []
    eef_vx_vals = [r.get("eef_vx", 0.0) for r in base13_stream]
    eef_vy_vals = [r.get("eef_vy", 0.0) for r in base13_stream]
    eef_vz_vals = [r.get("eef_vz", 0.0) for r in base13_stream]
    opening_vals = [r.get("gripper_opening_proxy", 0.0) for r in base13_stream]
    qpos_vals = [r.get("gripper_qpos", 0.0) for r in base13_stream]
    cmd_vals = [r.get("gripper_command", 0.0) for r in base13_stream]

    # eef_speed
    for i in range(n):
        eef_speed_vals.append(math.sqrt(
            eef_vx_vals[i]**2 + eef_vy_vals[i]**2 + eef_vz_vals[i]**2
        ))

    # close_onset: detect when gripper_command transitions from open to close
    # "close" = command > 0.5 (positive direction = closing)
    close_onset = [0] * n
    for i in range(1, n):
        if cmd_vals[i] > 0.5 and cmd_vals[i-1] <= 0.5:
            close_onset[i] = 1
    # Propagate: first close onset sets the reference
    first_close_idx = None
    for i in range(n):
        if close_onset[i]:
            first_close_idx = i
            break
    if first_close_idx is not None:
        for i in range(n):
            derived[i]["close_onset"] = 1.0 if i == first_close_idx else 0.0
    else:
        for i in range(n):
            derived[i]["close_onset"] = 0.0

    # time_since_close: steps since last close_onset
    last_close = -999999
    for i in range(n):
        if close_onset[i]:
            last_close = i
        derived[i]["time_since_close"] = float(i - last_close)

    # eef_speed
    for i in range(n):
        derived[i]["eef_speed"] = eef_speed_vals[i]

    # eef_z_delta_since_close: eef_z[i] - eef_z[last_close_idx]
    if first_close_idx is not None:
        for i in range(n):
            derived[i]["eef_z_delta_since_close"] = eef_z_vals[i] - eef_z_vals[first_close_idx]
    else:
        for i in range(n):
            derived[i]["eef_z_delta_since_close"] = 0.0

    # qpos_delta_1: qpos[i] - qpos[i-1]
    for i in range(n):
        derived[i]["qpos_delta_1"] = qpos_vals[i] - qpos_vals[i-1] if i > 0 else 0.0

    # qpos_delta_3: qpos[i] - qpos[i-3]
    for i in range(n):
        derived[i]["qpos_delta_3"] = qpos_vals[i] - qpos_vals[i-3] if i >= 3 else (qpos_vals[i] - qpos_vals[0] if i > 0 else 0.0)

    # opening_proxy_delta_3: opening[i] - opening[i-3]
    for i in range(n):
        derived[i]["opening_proxy_delta_3"] = opening_vals[i] - opening_vals[i-3] if i >= 3 else (opening_vals[i] - opening_vals[0] if i > 0 else 0.0)

    # opening_proxy_variance_5: variance of opening over [i-4, i]
    for i in range(n):
        start = max(0, i - 4)
        window = opening_vals[start:i+1]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        derived[i]["opening_proxy_variance_5"] = variance

    # eef_speed_variance_5: variance of eef_speed over [i-4, i]
    for i in range(n):
        start = max(0, i - 4)
        window = eef_speed_vals[start:i+1]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        derived[i]["eef_speed_variance_5"] = variance

    # recent_close_streak: consecutive steps where gripper_command > 0.5
    streak = 0
    for i in range(n):
        if cmd_vals[i] > 0.5:
            streak += 1
        else:
            streak = 0
        derived[i]["recent_close_streak"] = float(streak)

    # recent_open_streak: consecutive steps where gripper_command < -0.5
    streak = 0
    for i in range(n):
        if cmd_vals[i] < -0.5:
            streak += 1
        else:
            streak = 0
        derived[i]["recent_open_streak"] = float(streak)

    # recent_gripper_flip_count: number of direction changes in last 5 steps
    for i in range(n):
        start = max(0, i - 4)
        window = cmd_vals[start:i+1]
        flips = 0
        for j in range(1, len(window)):
            if (window[j] > 0.5 and window[j-1] < -0.5) or (window[j] < -0.5 and window[j-1] > 0.5):
                flips += 1
        derived[i]["recent_gripper_flip_count"] = float(flips)

    return derived


# ==============================================================================
# Main repair function
# ==============================================================================
def repair_temporal_file(temporal_path: str, output_path: str) -> Dict[str, Any]:
    """Repair a single temporal file by recomputing features from raw columns.

    Writes a 25D feature cache CSV to output_path.
    Returns repair status dict.
    """
    result = {
        "temporal_path": temporal_path,
        "output_path": output_path,
        "total_rows": 0,
        "base13_complete_rows": 0,
        "base13_incomplete_rows": 0,
        "derived_complete_rows": 0,
        "missing_features_per_row": [],
        "status": "UNKNOWN",
        "error": "",
    }

    try:
        rows = read_csv_dict(temporal_path)
    except Exception as e:
        result["status"] = "READ_ERROR"
        result["error"] = str(e)[:500]
        return result

    n = len(rows)
    result["total_rows"] = n

    # Step 1: Extract base13 from raw columns for all rows
    base13_stream = []
    complete_count = 0
    for i, row in enumerate(rows):
        feats, missing = extract_base13_from_raw(row)
        base13_stream.append(feats)
        if not missing:
            complete_count += 1
        else:
            result["missing_features_per_row"].append({
                "row_index": i,
                "step": row.get("step", str(i)),
                "missing": missing,
            })

    result["base13_complete_rows"] = complete_count
    result["base13_incomplete_rows"] = n - complete_count

    if complete_count == 0:
        result["status"] = "REPAIR_FAILED_NO_BASE13"
        result["error"] = "No rows have complete base13 from raw columns"
        return result

    # Step 2: Compute derived features
    derived_stream = compute_derived_features(base13_stream)

    # Step 3: Merge base13 + derived into full 25D rows
    output_rows = []
    for i in range(n):
        row_out = {"step": rows[i].get("step", str(i)), "row_index": i}
        # Add base13
        for feat in BASE13:
            row_out[feat] = base13_stream[i].get(feat, None)
        # Add derived
        for feat in DERIVED_12:
            row_out[feat] = derived_stream[i].get(feat, None)
        # Check completeness
        all_present = all(row_out.get(f, None) is not None for f in SC5_V2_FEATURES)
        row_out["_all_features_present"] = all_present
        if all_present:
            result["derived_complete_rows"] += 1
        output_rows.append(row_out)

    # Write repaired cache
    out_fields = ["step", "row_index"] + SC5_V2_FEATURES + ["_all_features_present"]
    write_csv(Path(output_path), output_rows, out_fields)

    result["status"] = "REPAIRED"
    result["repair_rate"] = result["derived_complete_rows"] / n if n > 0 else 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="C2e0F: Object base13 feature recomputation repair")
    parser.add_argument("--d4c2e0d-completeness-csv", required=True,
                        help="Path to D4C2E0D object_temporal_completeness_by_artifact.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repair-dir", default=None,
                        help="Directory to write repaired feature cache files (default: output-root/features/)")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate extract_base13_from_raw on sample files without writing")
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    repair_dir = Path(args.repair_dir) if args.repair_dir else out / "repaired_features"
    repair_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load D4C2E0D completeness data
    completeness_rows = read_csv_dict(args.d4c2e0d_completeness_csv)
    object_rows = [r for r in completeness_rows if "object" in r.get("raw_temporal_path", "").lower()]
    incomplete_files = [r for r in object_rows if r.get("complete_before_recompute", "").lower() == "false"]
    complete_files = [r for r in object_rows if r.get("complete_before_recompute", "").lower() == "true"]

    print(f"Object files: {len(object_rows)} total")
    print(f"  Complete (no repair needed): {len(complete_files)}")
    print(f"  Incomplete (to repair): {len(incomplete_files)}")

    if args.dry_run:
        # Test extract on first 3 incomplete files
        print("\nDRY RUN: testing extract_base13_from_raw on 3 sample files...")
        for r in incomplete_files[:3]:
            path = r["raw_temporal_path"]
            rows = read_csv_dict(path)
            n_valid = 0
            for row in rows:
                feats, missing = extract_base13_from_raw(row)
                if not missing:
                    n_valid += 1
            print(f"  {path.split('/')[-3]}/{path.split('/')[-2]}: {n_valid}/{len(rows)} rows with complete base13 from raw")
        print("DRY RUN complete. No files written.")
        return 0

    # Repair all incomplete files
    repair_results = []
    for i, r in enumerate(incomplete_files):
        temporal_path = r["raw_temporal_path"]
        # Generate output path preserving some structure
        parts = Path(temporal_path).parts
        # Use last 3 meaningful path components
        key_parts = []
        for p in reversed(parts):
            if p in ("step_telemetry.csv", "attempt_1", "jobs"):
                continue
            key_parts.append(p)
            if len(key_parts) >= 2:
                break
        fname = "_".join(reversed(key_parts)) + "_features_25d_repaired.csv"
        output_path = repair_dir / fname

        result = repair_temporal_file(temporal_path, str(output_path))
        repair_results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Repaired {i+1}/{len(incomplete_files)}...")

    # Summary
    succeeded = [r for r in repair_results if r["status"] == "REPAIRED"]
    failed = [r for r in repair_results if r["status"] != "REPAIRED"]
    total_base13_complete = sum(r.get("base13_complete_rows", 0) for r in repair_results)
    total_rows = sum(r.get("total_rows", 0) for r in repair_results)

    print(f"\nRepair summary:")
    print(f"  Succeeded: {len(succeeded)}/{len(repair_results)}")
    print(f"  Failed: {len(failed)}")
    if total_rows > 0:
        print(f"  Base13 complete rows: {total_base13_complete}/{total_rows} ({total_base13_complete/total_rows:.3f})")
        total_derived_complete = sum(r.get("derived_complete_rows", 0) for r in repair_results)
        print(f"  Derived 25D complete rows: {total_derived_complete}/{total_rows} ({total_derived_complete/total_rows:.3f})")

    # Build repair manifest
    manifest = []
    for r in repair_results:
        manifest.append({
            "original_temporal_path": r["temporal_path"],
            "repaired_feature_path": r["output_path"],
            "status": r["status"],
            "total_rows": r["total_rows"],
            "base13_complete_rows": r["base13_complete_rows"],
            "derived_complete_rows": r["derived_complete_rows"],
            "error": r.get("error", ""),
        })
    # Also include complete files (no repair needed)
    for r in complete_files:
        manifest.append({
            "original_temporal_path": r["raw_temporal_path"],
            "repaired_feature_path": "",
            "status": "ALREADY_COMPLETE",
            "total_rows": 0,
            "base13_complete_rows": 0,
            "derived_complete_rows": 0,
            "error": "",
        })

    write_csv(out / "c2e0f_repair_manifest.csv", manifest,
              ["original_temporal_path", "repaired_feature_path", "status",
               "total_rows", "base13_complete_rows", "derived_complete_rows", "error"])

    # Report
    all_ok = len(failed) == 0
    status = "PASS_C2E0F_OBJECT_BASE13_REPAIRED" if all_ok else "HOLD_C2E0F_PARTIAL_REPAIR"

    report = {
        "status": status,
        "git_commit": args.git_commit,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "gate": "C2E0F_OBJECT_BASE13_FEATURE_RECOMPUTATION_REPAIR",
        "object_total": len(object_rows),
        "object_previously_complete": len(complete_files),
        "object_repaired": len(succeeded),
        "object_repair_failed": len(failed),
        "total_rows_repaired": total_rows,
        "total_base13_complete": total_base13_complete,
        "total_derived_complete": sum(r.get("derived_complete_rows", 0) for r in repair_results),
        "repair_method": "extract_base13_from_raw_columns_bypass_f_cache",
        "raw_columns_used": ["eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz",
                              "gripper_qpos","gripper_width","raw_gripper","clean_action_7d"],
        "failed_count": len(failed),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "OpenVLA_model": "NOT_LOADED",
            "detector_training": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
        },
    }

    if not all_ok:
        report["failed_samples"] = [
            {"path": r["temporal_path"], "error": r.get("error", "")}
            for r in failed[:10]
        ]

    write_json(out / "c2e0f_object_base13_repair_report.json", report)

    # Checksums
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file() and fp.suffix != ".csv":  # skip large CSVs
            csums[fn] = sha256_file(str(fp))
    # Add manifest hash
    manifest_path = out / "c2e0f_repair_manifest.csv"
    csums["c2e0f_repair_manifest.csv"] = sha256_file(str(manifest_path))
    write_json(out / "checksum_report.json", csums)

    print(f"\nC2E0F STATUS = {status}")
    print(f"  Repaired: {len(succeeded)} files, {total_base13_complete}/{total_rows} rows")
    print(f"  Failed: {len(failed)} files")
    print(f"  Output: {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
