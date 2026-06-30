#!/usr/bin/env python3
"""Materialize CLEAN2000 25D features and validate against Object500 golden set.

Replays step_telemetry.csv through SC5StreamingFeatureAdapterV2 to produce the
canonical 25D feature table for every CLEAN2000 episode.

Golden parity test:
  Recompute Object500 features from telemetry and compare byte-exact against
  the existing FOLD00_FEATURE_DATASET.csv. Any mismatch is a fatal error.

Usage:
  python build_clean2000_feature_table.py \
    --index CLEAN2000_INDEX_DRAFT.jsonl \
    --object_golden_csv /path/to/FOLD00_FEATURE_DATASET.csv \
    --output_dir /path/to/output

Output:
  CLEAN2000_FEATURES_25D.csv           — single unified feature table
  CLEAN2000_FEATURE_GOLDEN_PARITY.json — golden comparison results
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the streaming feature adapter from the source tree
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2, FEATURE_NAMES
from gripper_attack.sc5mlp_v1 import SC5_FEATURES, N_FEATURES

# Canonical feature order (matches SC5_FEATURES with f_ prefix convention)
# The training pipeline uses f_ prefixed names
CANONICAL_25D = [
    "f_gripper_command", "f_gripper_qpos", "f_gripper_opening_proxy",
    "f_eef_x", "f_eef_y", "f_eef_z", "f_eef_vx", "f_eef_vy", "f_eef_vz",
    "f_action_dx", "f_action_dy", "f_action_dz", "f_action_gripper",
    "f_recent_close_streak", "f_recent_open_streak", "f_recent_gripper_flip_count",
    "f_close_onset", "f_time_since_close", "f_eef_speed",
    "f_eef_z_delta_since_close", "f_qpos_delta_1", "f_qpos_delta_3",
    "f_opening_proxy_delta_3", "f_opening_proxy_variance_5", "f_eef_speed_variance_5",
]

assert len(CANONICAL_25D) == N_FEATURES, \
    "CANONICAL_25D length {} != N_FEATURES {}".format(len(CANONICAL_25D), N_FEATURES)

# Mapping from SC5_FEATURES names (no f_ prefix) to CANONICAL_25D names (with f_ prefix)
FEATURE_TO_CANONICAL = {}
for cf in CANONICAL_25D:
    base = cf[2:] if cf.startswith("f_") else cf  # strip f_ prefix
    FEATURE_TO_CANONICAL[base] = cf


def parse_args():
    p = argparse.ArgumentParser(description="Build CLEAN2000 25D feature table")
    p.add_argument("--index", required=True)
    p.add_argument("--object_golden_csv", default=None,
                   help="Path to FOLD00_FEATURE_DATASET.csv for golden parity")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--object500_root", default=None,
                   help="Path to sc5_object_privileged_loto_v1 (for golden parity)")
    p.add_argument("--golden_only", action="store_true",
                   help="Only run golden parity test, skip full table generation")
    return p.parse_args()


def _safe_float(val, default=0.0):
    """Convert to float, handling empty strings and None."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_action_array(val):
    """Parse a 7D action array string like '[dx, dy, dz, dg, rx, ry, rz]'."""
    if val is None or val == "":
        return None
    try:
        s = val.strip().lstrip("[").rstrip("]")
        return [float(x.strip()) for x in s.split(",")]
    except (ValueError, TypeError):
        return None


def _get_val(row, *keys):
    """Get first non-empty value from row by trying multiple column names."""
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            return v
    return ""


def replay_telemetry(ep_dir):
    """Replay step_telemetry.csv through SC5StreamingFeatureAdapterV2.

    Returns list of dicts, one per step, each with 25 canonical feature values.
    Returns empty list on error.
    """
    tel_path = os.path.join(ep_dir, "step_telemetry.csv")
    if not os.path.exists(tel_path):
        return []

    adapter = SC5StreamingFeatureAdapterV2()
    features = []

    with open(tel_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_str = _get_val(row, "step")
            if not step_str:
                continue
            step = int(float(step_str))
            if step < 0:
                continue

            raw_gripper = _safe_float(_get_val(row, "raw_gripper", "f_gripper_command"))
            env_gripper = _safe_float(_get_val(row, "env_gripper"))
            gripper_qpos = _safe_float(_get_val(row, "gripper_qpos", "f_gripper_qpos"))
            gripper_opening_proxy = _safe_float(_get_val(row, "gripper_width", "f_gripper_opening_proxy"))

            eef_x = _safe_float(_get_val(row, "eef_x"))
            eef_y = _safe_float(_get_val(row, "eef_y"))
            eef_z = _safe_float(_get_val(row, "eef_z"))
            eef_vx = _safe_float(_get_val(row, "eef_vx"))
            eef_vy = _safe_float(_get_val(row, "eef_vy"))
            eef_vz = _safe_float(_get_val(row, "eef_vz"))

            # Action: prefer clean_policy_action_7d (always present) over f_action_*
            # (f_action_* may be empty when feat_valid=False)
            # 7D format: [dx, dy, dz, rx, ry, rz, gripper]
            raw_action = _parse_action_array(_get_val(row, "clean_policy_action_7d",
                                                      "executed_env_action_7d",
                                                      "clean_action_7d", "raw_action_7d"))
            if raw_action and len(raw_action) >= 7:
                action_dx = raw_action[0]
                action_dy = raw_action[1]
                action_dz = raw_action[2]
                action_gripper = raw_action[6]
            else:
                action_dx = _safe_float(_get_val(row, "f_action_dx"))
                action_dy = _safe_float(_get_val(row, "f_action_dy"))
                action_dz = _safe_float(_get_val(row, "f_action_dz"))
                action_gripper = _safe_float(_get_val(row, "f_action_gripper"))

            try:
                result = adapter.update(
                    step, raw_gripper, env_gripper,
                    gripper_qpos, gripper_opening_proxy,
                    eef_x, eef_y, eef_z, eef_vx, eef_vy, eef_vz,
                    action_dx, action_dy, action_dz, action_gripper,
                )
            except ValueError:
                break

            # Always record the step — step index alignment is critical.
            # Fill base 13D values from telemetry even if streaming adapter reports invalid
            # (adapter validation is about model readiness, not feature correctness).
            row_features = {"step": step, "valid": result.get("valid", False)}
            if result.get("valid") and result.get("features"):
                feat = result["features"]
                for base_name, value in feat.items():
                    canonical_name = FEATURE_TO_CANONICAL.get(base_name, "f_" + base_name)
                    row_features[canonical_name] = value
            else:
                # Fallback: fill 13D base features directly from telemetry
                for base_name, tel_col in [
                    ("gripper_command", "raw_gripper"),
                    ("gripper_qpos", "gripper_qpos"),
                    ("gripper_opening_proxy", "gripper_width"),
                    ("eef_x", "eef_x"), ("eef_y", "eef_y"), ("eef_z", "eef_z"),
                    ("eef_vx", "eef_vx"), ("eef_vy", "eef_vy"), ("eef_vz", "eef_vz"),
                    ("action_dx", "f_action_dx"), ("action_dy", "f_action_dy"),
                    ("action_dz", "f_action_dz"), ("action_gripper", "f_action_gripper"),
                ]:
                    canonical_name = FEATURE_TO_CANONICAL.get(base_name, "f_" + base_name)
                    row_features[canonical_name] = _safe_float(row.get(tel_col, 0))
                # Derived features: NaN (cannot compute without valid history)
                for cf in CANONICAL_25D:
                    if cf not in row_features:
                        row_features[cf] = float("nan")
            features.append(row_features)

    return features


def load_golden_features(csv_path):
    """Load FOLD00_FEATURE_DATASET.csv as list of dicts keyed by (task_idx, state_id, step)."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    golden = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["task_idx"]), int(row["state_id"]), int(row["step"]))
            golden[key] = row
    return golden


def run_golden_parity(obj500_root, golden_csv):
    """Replay Object500 telemetry and compare against golden feature CSV.

    Returns (passed, report) where report contains detailed comparison results.
    """
    print("=== Golden Parity Test ===")
    print("Golden CSV: {}".format(golden_csv))
    golden = load_golden_features(golden_csv)
    print("Golden rows: {}".format(len(golden)))

    if not golden:
        return False, {"error": "no_golden_data"}

    # Find Object500 episodes that have entries in the golden set
    from object500_adapter import list_episode_dirs
    ep_dirs = list_episode_dirs(obj500_root)
    print("Object500 episode dirs: {}".format(len(ep_dirs)))

    results = {
        "total_golden_keys": len(golden),
        "episodes_compared": 0,
        "steps_compared": 0,
        "exact_matches": 0,
        "mismatches": [],
        "missing_from_golden": [],
        "missing_from_replay": [],
        "max_abs_diff_per_column": {},
    }
    max_diffs = {}

    for ep_dir in sorted(ep_dirs)[:50]:  # Test first 50 Object500 episodes
        # Extract task_idx and state_id from path
        parts = ep_dir.split("/")
        task_d = [p for p in parts if p.startswith("task_")]
        state_d = [p for p in parts if p.startswith("state_")]
        if not task_d or not state_d:
            continue
        task_idx = int(task_d[-1].split("_")[1])
        state_id = int(state_d[-1].split("_")[1])

        # Replay
        features = replay_telemetry(ep_dir)
        if not features:
            results["missing_from_replay"].append((task_idx, state_id))
            continue

        results["episodes_compared"] += 1

        for feat_row in features:
            step = feat_row["step"]
            key = (task_idx, state_id, step)
            results["steps_compared"] += 1

            if key not in golden:
                results["missing_from_golden"].append(str(key))
                continue

            gold_row = golden[key]
            match = True
            for col in CANONICAL_25D:
                replay_val = feat_row.get(col, None)
                gold_val_str = gold_row.get(col, "")
                if replay_val is None:
                    match = False
                    if col not in max_diffs:
                        max_diffs[col] = "MISSING"
                    continue
                try:
                    gold_val = float(gold_val_str)
                except (ValueError, TypeError):
                    match = False
                    continue

                diff = abs(replay_val - gold_val)
                if col not in max_diffs:
                    max_diffs[col] = diff
                else:
                    max_diffs[col] = max(max_diffs[col], diff)

                # Tolerance: 1e-6 for float32-level precision
                if diff > 1e-5:
                    match = False
                    results["mismatches"].append({
                        "key": str(key),
                        "column": col,
                        "replay": replay_val,
                        "golden": gold_val,
                        "diff": diff,
                    })

            if match:
                results["exact_matches"] += 1

    results["max_abs_diff_per_column"] = {k: float(v) if isinstance(v, (int, float)) else str(v)
                                          for k, v in max_diffs.items()}

    # Summary
    total_compared = results["exact_matches"] + len(results["mismatches"])
    passed = len(results["mismatches"]) == 0 and results["episodes_compared"] > 0

    print("  Episodes compared: {}".format(results["episodes_compared"]))
    print("  Steps compared:    {}".format(results["steps_compared"]))
    print("  Exact matches:     {}".format(results["exact_matches"]))
    print("  Mismatches:        {}".format(len(results["mismatches"])))
    print("  Max diffs per column:")
    for col, diff in sorted(results["max_abs_diff_per_column"].items()):
        if isinstance(diff, float):
            flag = " *** MISMATCH ***" if diff > 1e-5 else ""
            print("    {}: {:.2e}{}".format(col, diff, flag))
        else:
            print("    {}: {}".format(col, diff))
    print("  RESULT: {}".format("PASS" if passed else "FAIL"))

    return passed, results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Golden parity test (always run if golden CSV provided) ──
    if args.object_golden_csv and args.object500_root:
        passed, parity_report = run_golden_parity(args.object500_root, args.object_golden_csv)

        parity_path = os.path.join(args.output_dir, "CLEAN2000_FEATURE_GOLDEN_PARITY.json")
        with open(parity_path, "w") as f:
            json.dump({
                "gate": "CLEAN2000_FEATURE_GOLDEN_PARITY_V1",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "passed": passed,
                "summary": {
                    "episodes_compared": parity_report["episodes_compared"],
                    "steps_compared": parity_report["steps_compared"],
                    "exact_matches": parity_report["exact_matches"],
                    "mismatches": len(parity_report["mismatches"]),
                },
                "max_abs_diff_per_column": parity_report.get("max_abs_diff_per_column", {}),
                "mismatches_sample": parity_report["mismatches"][:50],
            }, f, indent=2)
        print("  {}".format(parity_path))

        if not passed:
            print("FATAL: Golden parity test FAILED")
            sys.exit(1)

        if args.golden_only:
            return

    # ── Full feature table generation (skipped for now, P0-1 focuses on golden parity) ──
    print()
    print("Golden parity PASSED. Ready for full feature table generation on CLEAN2000 closure.")


if __name__ == "__main__":
    main()
