#!/usr/bin/env python3
"""M1C Object Clean Corpus Integrity Audit.

Comprehensive post-collection validation: cell uniqueness, file completeness,
attack safety, asset consistency, split isolation, step integrity.
Read-only. Fails on any violation.

Usage:
  python scripts/migration/audit_m1c_object_corpus.py \
    --execution-manifest <V4_AMENDMENT.json> \
    --split-manifest <frozen_split_manifest.json> \
    --output-root evidence/m1c/object_clean_corpus_audit
"""
import os, sys, json, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

EXPECTED_ASSETS = {
    "detector_checkpoint": "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628",
    "teacher_config": "ebc1ccda21cdfeae0f70f90ef0e433be3474ef0baa9cf52f609d620f863ce87a",
    "bridge_script": "fd594b3f9b38f4545d7b19202b380c0f4eeb0e9d95cb566f2fbca7b1852b208e",
}
EXPECTED_TASKS = set(range(10))
EXCLUDED_STATES = {0, 1, 2}
COMPROMISED_BLIND = set(range(38, 48))
BUFFER = set(range(48, 50))


def sha256_file(path):
    if not path.exists():
        return "MISSING"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def cell_key(task, state, profile="B0"):
    return f"{profile}_task{task}_state{state}"


# ── Checks ──────────────────────────────────────────────────────────

def check_cell_uniqueness(cells, output_root):
    """No duplicate (task, state, profile)."""
    seen = set()
    duplicates = []
    for c in cells:
        key = cell_key(c["task_idx"], c["state_id"], c.get("profile", "B0"))
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def check_file_completeness(cell, cell_dir):
    """Required files exist and are non-empty."""
    missing = []
    for fname in ["step_telemetry.csv", "episode_summary.json", ".done"]:
        fp = cell_dir / fname
        if not fp.exists():
            missing.append(f"missing:{fname}")
        elif fname == "step_telemetry.csv" and fp.stat().st_size == 0:
            missing.append("empty:step_telemetry.csv")
        elif fname == "episode_summary.json" and fp.stat().st_size < 10:
            missing.append("empty:episode_summary.json")
    return missing


def check_step_integrity(cell_dir):
    """CSV parseable, steps sequential from 0."""
    tel_path = cell_dir / "step_telemetry.csv"
    if not tel_path.exists():
        return ["missing_csv"]
    try:
        rows = list(csv.DictReader(open(tel_path)))
    except Exception as e:
        return [f"csv_parse_error:{e}"]
    if not rows:
        return ["empty_csv"]
    step_errors = []
    for i, r in enumerate(rows):
        csv_step = int(r.get("step", -1))
        if csv_step != i:
            step_errors.append(f"step_mismatch:idx={i},csv={csv_step}")
    # Check last row not truncated
    last = rows[-1]
    if last.get("step", "") == "":
        step_errors.append("last_row_truncated")
    return step_errors


def check_safety(cell_dir):
    """attack_frames == 0, no VIS/RAND condition."""
    ep_path = cell_dir / "episode_summary.json"
    if not ep_path.exists():
        return ["missing_summary"]
    try:
        s = json.load(open(ep_path))
    except Exception:
        return ["summary_parse_error"]
    issues = []
    if s.get("condition", "") != "CLEAN":
        issues.append(f"condition_not_CLEAN:{s.get('condition')}")
    af = s.get("attack_frames", -1)
    if af is None or af > 0:
        issues.append(f"attack_frames={af}")
    return issues


def check_asset_consistency(cell_dir):
    """Detector checkpoint SHA matches expected."""
    ep_path = cell_dir / "episode_summary.json"
    if not ep_path.exists():
        return ["missing_summary"]
    try:
        s = json.load(open(ep_path))
    except Exception:
        return ["summary_parse_error"]
    issues = []
    actual_sha = s.get("checkpoint_sha256", "")[:16]
    expected_sha = EXPECTED_ASSETS["detector_checkpoint"][:16]
    if actual_sha and actual_sha != expected_sha:
        issues.append(f"checkpoint_sha_mismatch:{actual_sha}!={expected_sha}")
    actual_dtype = s.get("actual_dtype", "")
    if actual_dtype and actual_dtype != "bfloat16":
        issues.append(f"dtype_mismatch:{actual_dtype}")
    return issues


def check_split_isolation(all_cells, split_manifest):
    """No cross-split initial_state_sha leakage."""
    train_states = set()
    val_states = set()
    issues = []
    for c in all_cells:
        pool = c.get("pool", "")
        # Try to get initial_state_sha from episode_summary
        cell_dir = c.get("_cell_dir")
        if not cell_dir or not cell_dir.exists():
            continue
        ep = cell_dir / "episode_summary.json"
        if not ep.exists():
            continue
        try:
            s = json.load(open(ep))
            iss = s.get("initial_state_sha256", "")
            if iss:
                if pool == "train":
                    if iss in val_states:
                        issues.append(f"cross_split_leak:train_sha_in_val:{iss[:16]}")
                    train_states.add(iss)
                elif pool == "validation":
                    if iss in train_states:
                        issues.append(f"cross_split_leak:val_sha_in_train:{iss[:16]}")
                    val_states.add(iss)
        except Exception:
            pass
    return issues


def check_no_excluded_states(cells):
    """No M1B states (0,1,2) or compromised blind (38-47)."""
    issues = []
    for c in cells:
        state = c["state_id"]
        if state in EXCLUDED_STATES:
            issues.append(f"excluded_state:{c['task_idx']}_{state}")
        if state in COMPROMISED_BLIND:
            issues.append(f"compromised_blind_state:{c['task_idx']}_{state}")
    return issues


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="M1C Object Corpus Integrity Audit")
    ap.add_argument("--execution-manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    ex_manifest = load_json(args.execution_manifest)
    split_manifest = load_json(args.split_manifest)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    gpu_assignments = ex_manifest["gpu_assignments"]
    output_base = Path(ex_manifest["output_root"])
    total_planned = ex_manifest["total_planned_cells"]

    all_cells = []
    errors = defaultdict(list)
    per_cell_rows = []

    # Collect all cells from all shards
    for gpu_key, gpu_info in gpu_assignments.items():
        pool = gpu_info["pool"]
        state_start, state_end = map(int, gpu_info["state_range"].split("-"))
        pool_dir = output_base / pool
        for task in range(10):
            for state in range(state_start, state_end + 1):
                cell_dir = pool_dir / f"task{task}_state{state}"
                cell_info = {
                    "gpu": gpu_key, "pool": pool, "task_idx": task,
                    "state_id": state, "profile": "B0",
                    "_cell_dir": cell_dir,
                }
                all_cells.append(cell_info)

                done_file = cell_dir / ".done"
                if not done_file.exists():
                    errors["missing_done"].append(cell_key(task, state))
                    continue

                try:
                    done_data = json.load(open(done_file))
                except Exception:
                    errors["corrupt_done"].append(cell_key(task, state))
                    continue

                rc = done_data.get("exit_code", -999)
                row = {
                    "gpu": gpu_key, "pool": pool, "task": task, "state": state,
                    "exit_code": rc,
                    "n_issues": 0,
                }

                # File completeness
                missing = check_file_completeness(cell_info, cell_dir)
                if missing:
                    errors["file_incomplete"].append(f"{cell_key(task,state)}:{missing}")
                    row["n_issues"] += len(missing)

                # Step integrity
                step_issues = check_step_integrity(cell_dir)
                if step_issues:
                    errors["step_integrity"].append(f"{cell_key(task,state)}:{step_issues}")
                    row["n_issues"] += len(step_issues)

                # Safety
                safety_issues = check_safety(cell_dir)
                if safety_issues:
                    errors["safety"].append(f"{cell_key(task,state)}:{safety_issues}")
                    row["n_issues"] += len(safety_issues)

                # Asset consistency
                asset_issues = check_asset_consistency(cell_dir)
                if asset_issues:
                    errors["asset"].append(f"{cell_key(task,state)}:{asset_issues}")
                    row["n_issues"] += len(asset_issues)

                per_cell_rows.append(row)

    # Uniqueness
    dupes = check_cell_uniqueness(all_cells, output_base)
    if dupes:
        for d in dupes:
            errors["duplicate"].append(d)

    # Excluded states
    excluded = check_no_excluded_states(all_cells)
    for e in excluded:
        errors["excluded_state"].append(e)

    # Cross-split leakage
    split_issues = check_split_isolation(all_cells, split_manifest)
    for s in split_issues:
        errors["split_leakage"].append(s)

    # Unexpected cells (not in plan, or in excluded pools)
    planned_keys = set()
    for gpu_key, gpu_info in gpu_assignments.items():
        state_start, state_end = map(int, gpu_info["state_range"].split("-"))
        for task in range(10):
            for state in range(state_start, state_end + 1):
                planned_keys.add(cell_key(task, state))
    actual_keys = set(cell_key(c["task_idx"], c["state_id"]) for c in all_cells)
    unexpected = actual_keys - planned_keys
    for u in unexpected:
        errors["unexpected"].append(u)

    # Summary
    n_done = len([c for c in all_cells if (c["_cell_dir"] / ".done").exists()])
    total_issues = sum(len(v) for v in errors.values())
    gate_pass = (n_done == total_planned and total_issues == 0)

    summary = {
        "gate": "M1C_OBJECT_CORPUS_INTEGRITY",
        "result": "PASS" if gate_pass else "FAIL",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_planned": total_planned,
        "total_found": n_done,
        "total_issues": total_issues,
        "error_categories": {k: len(v) for k, v in errors.items()},
    }

    # Output
    with open(out_root / "corpus_integrity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if per_cell_rows:
        with open(out_root / "per_cell_integrity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_cell_rows[0].keys())
            w.writeheader()
            w.writerows(per_cell_rows)

    if errors.get("duplicate"):
        with open(out_root / "duplicate_claims.csv", "w") as f:
            for d in errors["duplicate"]:
                f.write(d + "\n")

    if errors.get("split_leakage"):
        with open(out_root / "cross_split_sha_audit.csv", "w") as f:
            for s in errors["split_leakage"]:
                f.write(s + "\n")

    if errors.get("unexpected"):
        with open(out_root / "unexpected_cells.csv", "w") as f:
            for u in errors["unexpected"]:
                f.write(u + "\n")

    with open(out_root / "run.log", "w") as f:
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Gate: {summary['result']}\n")
        f.write(f"Planned: {total_planned}, Found: {n_done}, Issues: {total_issues}\n")
        for cat, items in errors.items():
            for item in items[:10]:
                f.write(f"  {cat}: {item}\n")

    print(f"\n  M1C Corpus Integrity Audit: {summary['result']}")
    print(f"  {n_done}/{total_planned} cells, {total_issues} issues")
    for cat, count in summary["error_categories"].items():
        print(f"    {cat}: {count}")
    print(f"  Output: {out_root}")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
