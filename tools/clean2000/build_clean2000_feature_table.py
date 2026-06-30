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
# Training column names: match frozen SC5_FEATURES (no f_ prefix)
TRAINING_COLUMNS = list(SC5_FEATURES)  # e.g. "gripper_command", "eef_x", ...
assert len(TRAINING_COLUMNS) == N_FEATURES, \
    "TRAINING_COLUMNS length {} != N_FEATURES {}".format(len(TRAINING_COLUMNS), N_FEATURES)

# Golden CSV column names: with f_ prefix (matches FOLD00_FEATURE_DATASET.csv)
GOLDEN_COLUMNS = ["f_" + name for name in TRAINING_COLUMNS]

# Mapping from adapter internal names (base) to golden column names
FEATURE_TO_GOLDEN = {}
for i, name in enumerate(TRAINING_COLUMNS):
    FEATURE_TO_GOLDEN[name] = GOLDEN_COLUMNS[i]


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
    p.add_argument("--formal", action="store_true",
                   help="Require golden parity. Fail on any episode error.")
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
                    canonical_name = FEATURE_TO_GOLDEN.get(base_name, "f_" + base_name)
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
                    canonical_name = FEATURE_TO_GOLDEN.get(base_name, "f_" + base_name)
                    row_features[canonical_name] = _safe_float(row.get(tel_col, 0))
                # Derived features: NaN (cannot compute without valid history)
                for cf in GOLDEN_COLUMNS:
                    if cf not in row_features:
                        row_features[cf] = float("nan")
            features.append(row_features)

    return features


def load_golden_features(csv_path):
    """Load FOLD00_FEATURE_DATASET.csv as list of dicts keyed by (task_idx, state_id, step).
    Fails on duplicate keys."""
    if not csv_path or not os.path.exists(csv_path):
        return {}, 0
    golden = {}
    row_count = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["task_idx"]), int(row["state_id"]), int(row["step"]))
            if key in golden:
                raise ValueError("Duplicate golden key: {}".format(key))
            golden[key] = row
            row_count += 1
    return golden, row_count


def run_golden_parity(obj500_root, golden_csv):
    """Replay Object500 telemetry and compare against golden feature CSV.

    Returns (passed, report) where report contains detailed comparison results.
    """
    print("=== Golden Parity Test ===")
    print("Golden CSV: {}".format(golden_csv))
    golden, golden_row_count = load_golden_features(golden_csv)
    print("Golden rows: {}".format(len(golden)))

    if not golden:
        return False, {"error": "no_golden_data"}

    # Find Object500 episodes and build (task, state) -> ep_dir map
    from object500_adapter import list_episode_dirs
    ep_dirs = list_episode_dirs(obj500_root)
    print("Object500 episode dirs: {}".format(len(ep_dirs)))

    # Reverse selection: use golden CSV (task, state) pairs to select episodes
    golden_pairs = set()
    for (tid, sid, _) in golden:
        golden_pairs.add((tid, sid))
    print("Golden (task, state) pairs: {}".format(len(golden_pairs)))

    episode_map = {}  # (task_id, state_id) -> ep_dir
    for ep_dir in ep_dirs:
        parts = ep_dir.split("/")
        task_d = [p for p in parts if p.startswith("task_")]
        state_d = [p for p in parts if p.startswith("state_")]
        if not task_d or not state_d:
            continue
        tid = int(task_d[-1].split("_")[1])
        sid = int(state_d[-1].split("_")[1])
        key = (tid, sid)
        if key in episode_map:
            # Multiple attempts for same (task, state) — use the first found
            continue
        episode_map[key] = ep_dir

    # Verify all golden pairs have episodes
    missing_episodes = golden_pairs - set(episode_map)
    if missing_episodes:
        print("FATAL: {} golden (task,state) pairs have no episode dir".format(
            len(missing_episodes)))
        for p in sorted(missing_episodes)[:10]:
            print("  {}".format(p))
        return False, {"error": "missing_episodes", "missing": sorted(str(p) for p in missing_episodes)}

    # Select episodes from golden pairs (use all golden pairs for parity)
    selected_pairs = sorted(golden_pairs)
    print("Testing {} episodes (golden-driven selection)".format(len(selected_pairs)))

    results = {
        "total_golden_keys": len(golden),
        "golden_pairs": len(golden_pairs),
        "episodes_compared": 0,
        "steps_compared": 0,
        "exact_matches": 0,
        "mismatches": [],
        "missing_from_golden": [],
        "missing_from_replay": [],
        "max_abs_diff_per_column": {},
    }
    max_diffs = {}

    for (task_idx, state_id) in selected_pairs:
        ep_dir = episode_map[(task_idx, state_id)]

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
            for col in GOLDEN_COLUMNS:
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

    # ── Fail-closed gate ──
    # Replay keys must exactly match golden keys for compared episodes
    replay_key_set = set()
    replay_row_count = 0
    for (tid, sid) in selected_pairs:
        ep_dir = episode_map[(tid, sid)]
        feats = replay_telemetry(ep_dir)
        replay_row_count += len(feats)
        for fr in feats:
            replay_key_set.add((tid, sid, fr["step"]))

    golden_key_set = set()
    for (tid, sid, step), _ in golden.items():
        if (tid, sid) in set(selected_pairs):
            golden_key_set.add((tid, sid, step))

    missing_from_golden = replay_key_set - golden_key_set
    missing_from_replay = golden_key_set - replay_key_set
    results["missing_from_golden"] = sorted(str(k) for k in missing_from_golden)
    results["missing_from_replay"] = sorted(str(k) for k in missing_from_replay)

    nan_count = 0
    non_finite = 0
    for (tid, sid) in selected_pairs:
        ep_dir = episode_map[(tid, sid)]
        feats = replay_telemetry(ep_dir)
        for fr in feats:
            for col in GOLDEN_COLUMNS:
                v = fr.get(col)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    nan_count += 1
                    continue
                if not np.isfinite(fv):
                    if np.isnan(fv):
                        nan_count += 1
                    else:
                        non_finite += 1

    # Check golden values too
    golden_nan = 0
    for key, gr in golden.items():
        for col in GOLDEN_COLUMNS:
            try:
                fv = float(gr.get(col, ""))
            except (TypeError, ValueError):
                golden_nan += 1
                continue
            if not np.isfinite(fv):
                golden_nan += 1

    # Row-level duplicate check (per tested episodes only)
    replay_key_dupes = len(replay_key_set) < replay_row_count
    golden_row_count_tested = sum(1 for (tid, sid, _) in golden if (tid, sid) in set(selected_pairs))
    golden_key_dupes = len(golden_key_set) < golden_row_count_tested

    key_coverage = (
        len(missing_from_golden) == 0 and
        len(missing_from_replay) == 0 and
        len(replay_key_set) > 0
    )
    all_finite = (nan_count == 0 and non_finite == 0 and golden_nan == 0)
    no_duplicate_rows = (not replay_key_dupes and not golden_key_dupes)
    row_count_matches = (replay_row_count == len(replay_key_set) and
                         not golden_key_dupes)
    total_compared = results["exact_matches"] + len(results["mismatches"])
    passed = (
        len(results["mismatches"]) == 0 and
        results["episodes_compared"] > 0 and
        key_coverage and
        all_finite and
        no_duplicate_rows and
        row_count_matches and
        total_compared == len(golden_key_set)
    )

    print("  Episodes compared: {}".format(results["episodes_compared"]))
    print("  Steps compared:    {}".format(results["steps_compared"]))
    print("  Exact matches:     {}".format(results["exact_matches"]))
    print("  Mismatches:        {}".format(len(results["mismatches"])))
    print("  Missing golden:    {}".format(len(missing_from_golden)))
    print("  Missing replay:    {}".format(len(missing_from_replay)))
    print("  NaN values:        {}".format(nan_count))
    print("  Inf values:        {}".format(non_finite))
    print("  Golden NaN:        {}".format(golden_nan))
    print("  Replay row dupes:  {}".format("YES" if replay_key_dupes else "no"))
    print("  Golden row dupes:  {}".format("YES" if golden_key_dupes else "no"))
    print("  Key coverage:      {}".format("PASS" if key_coverage else "FAIL"))
    print("  All finite:        {}".format("PASS" if all_finite else "FAIL"))
    print("  No dupes:          {}".format("PASS" if no_duplicate_rows else "FAIL"))
    print("  Row count match:   {}".format("PASS" if row_count_matches else "FAIL"))
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

    # ── Golden parity test ──
    if args.formal and not (args.object_golden_csv and args.object500_root):
        print("FATAL: --formal requires --object_golden_csv and --object500_root")
        sys.exit(1)

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
    elif args.formal:
        print("FATAL: --formal mode requires golden parity to pass first")
        sys.exit(1)

    # ── Full feature table generation ──
    print()
    print("=== Full 25D Feature Table Generation ===")

    # Load index
    index_rows = []
    with open(args.index) as f:
        for line in f:
            line = line.strip()
            if line:
                index_rows.append(json.loads(line))
    print("Index episodes: {}".format(len(index_rows)))

    index_keys = set(r["episode_key"] for r in index_rows)

    # Generate features for every episode
    all_csv_path = os.path.join(args.output_dir, "CLEAN2000_FEATURES_25D_ALL_STEPS.csv")
    valid_csv_path = os.path.join(args.output_dir, "CLEAN2000_FEATURES_25D_VALID_ONLY.csv")
    report_path = os.path.join(args.output_dir, "CLEAN2000_FEATURE_TABLE_REPORT.json")

    # Training columns: match frozen SC5_FEATURES (no f_ prefix)
    all_output_cols = ["episode_key", "suite", "task_id", "state_id", "step"] + TRAINING_COLUMNS + ["feat_valid"]
    valid_output_cols = ["episode_key", "suite", "task_id", "state_id", "step", "source_step"] + TRAINING_COLUMNS

    total_all_steps = 0
    total_valid_steps = 0
    total_episodes = 0
    failed_episodes = []
    episode_keys_seen = set()
    step_keys_seen = set()
    nan_count = 0
    inf_count = 0
    per_suite_valid_steps = {}
    per_suite_invalid_steps = {}

    with open(all_csv_path, "w", newline="") as f_all, \
         open(valid_csv_path, "w", newline="") as f_valid:

        writer_all = csv.DictWriter(f_all, fieldnames=all_output_cols)
        writer_valid = csv.DictWriter(f_valid, fieldnames=valid_output_cols)
        writer_all.writeheader()
        writer_valid.writeheader()

        for row in index_rows:
            ek = row["episode_key"]
            ep_dir = row["source_root"]
            suite = row["suite"]
            features = replay_telemetry(ep_dir)

            if not features:
                failed_episodes.append({"episode_key": ek, "error": "no_features_generated"})
                if args.formal:
                    print("FATAL: {} — no features generated".format(ek))
                    sys.exit(1)
                continue

            if len(features) != row.get("n_telemetry_rows", -1):
                failed_episodes.append({"episode_key": ek,
                    "error": "step_count_mismatch: features={} telemetry={}".format(
                        len(features), row.get("n_telemetry_rows"))})
                if args.formal:
                    print("FATAL: {} — {}".format(ek, failed_episodes[-1]["error"]))
                    sys.exit(1)
                continue

            valid_step_idx = 0
            for feat_row in features:
                step = feat_row["step"]
                step_key = (ek, step)
                if step_key in step_keys_seen:
                    failed_episodes.append({"episode_key": ek, "error": "duplicate_step_{}".format(step)})
                    if args.formal:
                        print("FATAL: duplicate step {}".format(step_key))
                        sys.exit(1)
                    continue
                step_keys_seen.add(step_key)

                is_valid = feat_row.get("valid", False)

                # ALL_STEPS row
                out_row = {
                    "episode_key": ek, "suite": suite,
                    "task_id": row["task_id"], "state_id": row["state_id"],
                    "step": step,
                    "feat_valid": "true" if is_valid else "false",
                }
                for col in TRAINING_COLUMNS:
                    # Look up value in feat_row using both training name and golden name
                    v = feat_row.get(col) or feat_row.get("f_" + col)
                    try:
                        fv = float(v) if v is not None else None
                    except (TypeError, ValueError):
                        if args.formal:
                            print("FATAL: {} step {} col {} — unparseable value".format(ek, step, col))
                            sys.exit(1)
                        nan_count += 1
                        fv = None
                    if fv is not None:
                        if np.isnan(fv):
                            nan_count += 1
                        elif np.isinf(fv):
                            inf_count += 1
                    out_row[col] = "{:.15e}".format(fv) if fv is not None else ""
                writer_all.writerow(out_row)
                total_all_steps += 1

                # VALID_ONLY row
                if is_valid:
                    valid_row = {
                        "episode_key": ek, "suite": suite,
                        "task_id": row["task_id"], "state_id": row["state_id"],
                        "step": valid_step_idx,
                        "source_step": step,
                    }
                    for col in TRAINING_COLUMNS:
                        valid_row[col] = out_row[col]
                    writer_valid.writerow(valid_row)
                    valid_step_idx += 1
                    total_valid_steps += 1

            # Per-suite stats
            n_valid = valid_step_idx
            n_invalid = len(features) - n_valid
            per_suite_valid_steps[suite] = per_suite_valid_steps.get(suite, 0) + n_valid
            per_suite_invalid_steps[suite] = per_suite_invalid_steps.get(suite, 0) + n_invalid

            episode_keys_seen.add(ek)
            total_episodes += 1

    # Verification
    missing_eps = index_keys - episode_keys_seen
    extra_eps = episode_keys_seen - index_keys

    report = {
        "gate": "CLEAN2000_FEATURE_TABLE_REPORT_V1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_episodes_expected": len(index_rows),
        "total_episodes_generated": total_episodes,
        "total_all_steps": total_all_steps,
        "total_valid_steps": total_valid_steps,
        "per_suite_valid_steps": per_suite_valid_steps,
        "per_suite_invalid_steps": per_suite_invalid_steps,
        "feature_columns": TRAINING_COLUMNS,
        "n_features": N_FEATURES,
        "missing_episodes": sorted(missing_eps),
        "extra_episodes": sorted(extra_eps),
        "failed_episodes": failed_episodes,
        "nan_values": nan_count,
        "inf_values": inf_count,
        "golden_parity_passed": passed if 'passed' in dir() else None,
    }

    all_ok = (len(missing_eps) == 0 and len(extra_eps) == 0 and
              len(failed_episodes) == 0 and nan_count == 0 and inf_count == 0)

    report["all_checks_pass"] = all_ok

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  {}".format(all_csv_path))
    print("  {}".format(valid_csv_path))
    print("  {}".format(report_path))
    print()

    print("=== Feature Table Summary ===")
    print("  Episodes: {}/{}".format(total_episodes, len(index_rows)))
    print("  All steps:   {}".format(total_all_steps))
    print("  Valid steps: {}".format(total_valid_steps))
    print("  Invalid steps: {}".format(total_all_steps - total_valid_steps))
    for suite in sorted(per_suite_valid_steps):
        v = per_suite_valid_steps[suite]
        iv = per_suite_invalid_steps.get(suite, 0)
        print("  {}: valid={}, invalid={}".format(suite, v, iv))
    print("  NaN values: {}".format(nan_count))
    print("  Inf values: {}".format(inf_count))
    print("  Failed episodes: {}".format(len(failed_episodes)))
    print("  Missing episodes: {}".format(len(missing_eps)))
    print("  All checks: {}".format("PASS" if all_ok else "FAIL"))

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
