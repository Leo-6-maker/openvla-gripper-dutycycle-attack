#!/usr/bin/env python3
"""C2e0G: Pipeline integration — verify Object repair with patched feature extraction.

Reads context dataset, uses repaired feature CSVs for Object rows,
normal v3 extraction for other suites. Re-runs completeness check on all 3717 rows.

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
# Also try the codex tools path for v3
sys.path.insert(0, "/mnt/sdc/dty_user/openvla_attack_codex_tools_pr50_f3e6b0/tools/multisuite_detector")

try:
    import probe_clean2000_detector_25d_feature_extraction_v3 as v3
except ImportError:
    v3 = None
    print("WARNING: v3 module not found, will use raw-column extraction for all files")

from gripper_attack.sc5_multisuite_detector_runtime import SC5_V2_FEATURES

BASE13 = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
]

# ==============================================================================
# Raw-column fallback extraction (same logic as C2e0F)
# ==============================================================================
def parse_action_7d(val):
    if val is None or val.strip() == "":
        return None
    v = val.strip()
    if v.startswith("["):
        try:
            parts = json.loads(v)
            if len(parts) >= 7:
                return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[6]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    try:
        parts = [float(x.strip()) for x in v.split(",")]
        if len(parts) >= 7:
            return (parts[0], parts[1], parts[2], parts[6])
    except (ValueError, TypeError):
        pass
    return None


def extract_features_from_raw_row(row):
    """Extract 25D features from raw columns, bypassing f_ cache."""
    values = {}
    missing = []

    # Direct copies
    for f in ["eef_x","eef_y","eef_z","eef_vx","eef_vy","eef_vz","gripper_qpos"]:
        try:
            values[f] = float(row.get(f, ""))
        except (ValueError, TypeError):
            missing.append(f)

    # gripper_command from raw_gripper
    try:
        values["gripper_command"] = float(row.get("raw_gripper", ""))
    except (ValueError, TypeError):
        try:
            values["gripper_command"] = float(row.get("env_gripper", ""))
        except (ValueError, TypeError):
            missing.append("gripper_command")

    # gripper_opening_proxy from gripper_width
    try:
        values["gripper_opening_proxy"] = float(row.get("gripper_width", ""))
    except (ValueError, TypeError):
        missing.append("gripper_opening_proxy")

    # action from clean_action_7d
    parsed = parse_action_7d(row.get("clean_action_7d", ""))
    if parsed:
        values["action_dx"] = parsed[0]
        values["action_dy"] = parsed[1]
        values["action_dz"] = parsed[2]
        values["action_gripper"] = parsed[3]
    else:
        parsed2 = parse_action_7d(row.get("executed_env_action_7d", ""))
        if parsed2:
            values["action_dx"] = parsed2[0]
            values["action_dy"] = parsed2[1]
            values["action_dz"] = parsed2[2]
            values["action_gripper"] = parsed2[3]
        else:
            missing.extend(["action_dx","action_dy","action_dz","action_gripper"])

    return values, missing


def compute_derived_values(base13_stream, idx):
    """Compute derived features at index idx from base13 stream."""
    n = len(base13_stream)
    derived = {}
    b13 = base13_stream[idx]

    # eef_speed
    eef_vx = b13.get("eef_vx", 0)
    eef_vy = b13.get("eef_vy", 0)
    eef_vz = b13.get("eef_vz", 0)
    derived["eef_speed"] = math.sqrt(eef_vx**2 + eef_vy**2 + eef_vz**2)

    # close_onset, time_since_close, eef_z_delta_since_close
    cmd_stream = [r.get("gripper_command", 0) for r in base13_stream]
    eef_z_stream = [r.get("eef_z", 0) for r in base13_stream]

    close_onset = 0
    first_close = None
    for i in range(idx + 1):
        if i > 0 and cmd_stream[i] > 0.5 and cmd_stream[i-1] <= 0.5:
            first_close = i
            break
    # Also check idx itself
    if first_close is None:
        for i in range(n):
            if i > 0 and cmd_stream[i] > 0.5 and cmd_stream[i-1] <= 0.5:
                first_close = i
                break
    derived["close_onset"] = 1.0 if idx == first_close else 0.0
    derived["time_since_close"] = float(idx - first_close) if first_close is not None else float(idx + 999)
    derived["eef_z_delta_since_close"] = eef_z_stream[idx] - eef_z_stream[first_close] if first_close is not None else 0.0

    # qpos_delta
    qpos_stream = [r.get("gripper_qpos", 0) for r in base13_stream]
    derived["qpos_delta_1"] = qpos_stream[idx] - qpos_stream[idx-1] if idx > 0 else 0.0
    derived["qpos_delta_3"] = qpos_stream[idx] - qpos_stream[idx-3] if idx >= 3 else (qpos_stream[idx] - qpos_stream[0] if idx > 0 else 0.0)

    # opening delta/variance
    opening_stream = [r.get("gripper_opening_proxy", 0) for r in base13_stream]
    derived["opening_proxy_delta_3"] = opening_stream[idx] - opening_stream[idx-3] if idx >= 3 else (opening_stream[idx] - opening_stream[0] if idx > 0 else 0.0)
    start = max(0, idx - 4)
    window = opening_stream[start:idx+1]
    mean = sum(window) / len(window)
    derived["opening_proxy_variance_5"] = sum((x-mean)**2 for x in window) / len(window)

    # eef_speed variance
    eef_speed_stream = []
    for i in range(n):
        vx = base13_stream[i].get("eef_vx", 0)
        vy = base13_stream[i].get("eef_vy", 0)
        vz = base13_stream[i].get("eef_vz", 0)
        eef_speed_stream.append(math.sqrt(vx**2 + vy**2 + vz**2))
    start = max(0, idx - 4)
    window = eef_speed_stream[start:idx+1]
    mean = sum(window) / len(window)
    derived["eef_speed_variance_5"] = sum((x-mean)**2 for x in window) / len(window)

    # streaks and flips
    recent_close_streak = 0
    for i in range(idx, -1, -1):
        if cmd_stream[i] > 0.5:
            recent_close_streak += 1
        else:
            break
    derived["recent_close_streak"] = float(recent_close_streak)

    recent_open_streak = 0
    for i in range(idx, -1, -1):
        if cmd_stream[i] < -0.5:
            recent_open_streak += 1
        else:
            break
    derived["recent_open_streak"] = float(recent_open_streak)

    start = max(0, idx - 4)
    window_cmd = cmd_stream[start:idx+1]
    flips = 0
    for j in range(1, len(window_cmd)):
        if (window_cmd[j] > 0.5 and window_cmd[j-1] < -0.5) or (window_cmd[j] < -0.5 and window_cmd[j-1] > 0.5):
            flips += 1
    derived["recent_gripper_flip_count"] = float(flips)

    return derived


# ==============================================================================
# Main
# ==============================================================================
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


def check_row_25d_complete(values, derived):
    """Check if all 25D features are present and finite."""
    for feat in SC5_V2_FEATURES:
        v = values.get(feat) if feat in values else derived.get(feat)
        if v is None or not math.isfinite(float(v)):
            return False, feat
    return True, ""


def extract_features_patched(temporal_path, row_idx, window=32):
    """Patched feature extraction: raw-column fallback for Object files."""
    rows = read_csv_dict(temporal_path)
    n = len(rows)
    if n == 0:
        return {}, ["empty_file"]

    # Check if this is a repaired features CSV vs original step_telemetry
    header = list(rows[0].keys())
    is_repaired = "_all_features_present" in header

    if is_repaired:
        # Read features directly from repaired CSV
        if row_idx >= n:
            row_idx = n - 1
        row = rows[row_idx]
        values = {}
        missing = []
        for feat in SC5_V2_FEATURES:
            v = row.get(feat, "")
            if v and v != "None":
                try:
                    values[feat] = float(v)
                except (ValueError, TypeError):
                    missing.append(feat)
            else:
                missing.append(feat)
        return values, missing

    # Try v3 first
    base13_stream = []
    v3_ok = v3 is not None
    for i, row in enumerate(rows):
        if v3_ok:
            try:
                vals, methods, fields = v3.compute_features(rows, i, "NO_EVENT")
            except Exception:
                vals = {}
        else:
            vals = {}
        # Check finiteness
        base13 = {}
        for feat in BASE13:
            v = vals.get(feat)
            if v is not None and math.isfinite(float(v)):
                base13[feat] = float(v)
                continue
            # Fallback: extract from raw columns
            raw_vals, _ = extract_features_from_raw_row(row)
            if feat in raw_vals:
                base13[feat] = raw_vals[feat]
        base13_stream.append(base13)

    if row_idx >= n:
        row_idx = n - 1
    values = base13_stream[row_idx]
    missing = [f for f in BASE13 if f not in values]
    return values, missing


def main():
    parser = argparse.ArgumentParser(description="C2e0G: Pipeline integration and completeness verification")
    parser.add_argument("--context-dataset", required=True,
                        help="Path to context_detector_dataset_v1b.csv")
    parser.add_argument("--repair-manifest", required=True,
                        help="Path to C2e0F c2e0f_repair_manifest.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load data
    context_rows = read_csv_dict(args.context_dataset)
    manifest_rows = read_csv_dict(args.repair_manifest)

    # Build repair lookup: original_path → repaired_path
    repair_map = {}
    for r in manifest_rows:
        if r.get("status") == "REPAIRED":
            repair_map[r["original_temporal_path"]] = r["repaired_feature_path"]

    print(f"Context dataset: {len(context_rows)} rows")
    print(f"Repair manifest: {len(repair_map)} repaired Object files")

    # Run completeness check on all rows
    results = []
    suites = defaultdict(lambda: {"has_primary": 0, "no_primary": 0, "complete": 0, "incomplete": 0})
    temporal_cache = {}
    temporal_failures = 0

    for i, row in enumerate(context_rows):
        suite = row.get("suite", "unknown")
        tpath = row.get("temporal_path", "")

        # Use repaired path for Object files if available
        actual_path = repair_map.get(tpath, tpath)

        if not actual_path:
            results.append({"row_index": i, "suite": suite, "group_key": row.get("group_key",""),
                "label_status": row.get("teacher_label_status",""),
                "temporal_path": tpath, "used_path": "", "status": "NO_TEMPORAL_PATH",
                "complete": False, "missing": "no_path"})
            continue

        try:
            if actual_path not in temporal_cache:
                temporal_cache[actual_path] = read_csv_dict(actual_path)
            trows = temporal_cache[actual_path]
        except Exception as e:
            temporal_failures += 1
            results.append({"row_index": i, "suite": suite, "group_key": row.get("group_key",""),
                "label_status": row.get("teacher_label_status",""),
                "temporal_path": tpath, "used_path": actual_path, "status": "READ_ERROR",
                "complete": False, "missing": str(e)[:200]})
            continue

        n_rows = len(trows)
        # Use last row as endpoint (clamp to window)
        endpoint = min(n_rows - 1, args.window - 1) if n_rows >= args.window else n_rows - 1
        if endpoint < 0:
            results.append({"row_index": i, "suite": suite, "group_key": row.get("group_key",""),
                "label_status": row.get("teacher_label_status",""),
                "temporal_path": tpath, "used_path": actual_path, "status": "EMPTY_FILE",
                "complete": False, "missing": "empty_file"})
            continue

        values, missing = extract_features_patched(actual_path, endpoint, args.window)
        complete, first_missing = check_row_25d_complete(values, {})

        cls = "has_primary" if row.get("teacher_label_status") == "VALID_PRIMARY" else "no_primary"
        suites[suite][cls] += 1
        if complete:
            suites[suite]["complete"] += 1
        else:
            suites[suite]["incomplete"] += 1

        results.append({
            "row_index": i, "suite": suite, "group_key": row.get("group_key",""),
            "label_status": row.get("teacher_label_status",""),
            "temporal_path": tpath,
            "used_repaired": actual_path != tpath,
            "n_rows": n_rows, "endpoint": endpoint,
            "complete": complete,
            "missing_count": len(missing),
            "first_missing": first_missing,
            "status": "COMPLETE" if complete else "INCOMPLETE",
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(context_rows)}...")

    # Summary by suite
    print(f"\nResults by suite ({args.window}-row window):")
    summary_rows = []
    total_complete = 0
    total_rows = 0
    for suite in sorted(suites.keys()):
        s = suites[suite]
        total_suite = s["complete"] + s["incomplete"]
        rate = s["complete"] / total_suite if total_suite > 0 else 0
        total_complete += s["complete"]
        total_rows += total_suite
        print(f"  {suite}: {s['complete']}/{total_suite} complete ({rate:.4f}), primary={s['has_primary']} no_primary={s['no_primary']}")
        summary_rows.append({
            "suite": suite, "total": total_suite,
            "complete": s["complete"], "incomplete": s["incomplete"],
            "completeness_rate": rate,
            "has_primary_rows": s["has_primary"],
            "no_primary_rows": s["no_primary"],
        })

    overall_rate = total_complete / total_rows if total_rows > 0 else 0
    print(f"  OVERALL: {total_complete}/{total_rows} complete ({overall_rate:.4f})")

    # Check Object specifically
    obj_complete = sum(1 for r in results if r["suite"] == "libero_object" and r["complete"])
    obj_total = sum(1 for r in results if r["suite"] == "libero_object")
    obj_rate = obj_complete / obj_total if obj_total > 0 else 0
    print(f"  OBJECT: {obj_complete}/{obj_total} complete ({obj_rate:.4f})")

    # Write outputs
    write_csv(out / "c2e0g_completeness_by_row.csv", results,
              ["row_index","suite","group_key","label_status","temporal_path","used_repaired",
               "n_rows","endpoint","complete","missing_count","first_missing","status"])

    write_csv(out / "c2e0g_completeness_by_suite.csv", summary_rows,
              ["suite","total","complete","incomplete","completeness_rate","has_primary_rows","no_primary_rows"])

    # Assessment
    obj_ok = obj_rate >= 0.95
    overall_ok = overall_rate >= 0.95
    all_ok = obj_ok and overall_ok

    report = {
        "status": "PASS_C2E0G_PIPELINE_INTEGRATION" if all_ok else "HOLD_C2E0G_INCOMPLETE",
        "git_commit": args.git_commit,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "gate": "C2E0G_PIPELINE_INTEGRATION_VERIFICATION",
        "total_rows": total_rows,
        "total_complete": total_complete,
        "overall_completeness_rate": overall_rate,
        "object_rows": obj_total,
        "object_complete": obj_complete,
        "object_completeness_rate": obj_rate,
        "object_target": 0.95,
        "object_meets_target": obj_ok,
        "repaired_files_used": sum(1 for r in results if r.get("used_repaired")),
        "temporal_failures": temporal_failures,
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "OpenVLA_model": "NOT_LOADED",
            "detector_training": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
        },
    }

    if all_ok:
        report["recommendation"] = "proceed_to_c2e1_temporal_dataset_materialization"
    else:
        report["recommendation"] = "fix_remaining_incomplete_rows_before_c2e1"

    write_json(out / "c2e0g_pipeline_integration_report.json", report)

    # Checksums
    csums = {}
    for fn in sorted(os.listdir(str(out))):
        fp = out / fn
        if fp.is_file() and not fn.endswith(".csv"):
            csums[fn] = sha256_file(str(fp))
    csums["c2e0g_completeness_by_suite.csv"] = sha256_file(str(out / "c2e0g_completeness_by_suite.csv"))
    write_json(out / "checksum_report.json", csums)

    print(f"\nC2E0G STATUS = {report['status']}")
    print(f"  Object: {obj_complete}/{obj_total} ({obj_rate:.4f}) {'PASS' if obj_ok else 'FAIL'}")
    print(f"  Overall: {total_complete}/{total_rows} ({overall_rate:.4f}) {'PASS' if overall_ok else 'FAIL'}")
    print(f"  Repaired files used: {report['repaired_files_used']}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
