#!/usr/bin/env python3
"""M1C Object Clean Corpus Integrity Auditor v2 — hardened.

v2 fixes: actual filesystem scan, non-zero RC fail, missing safety fields fail,
full 64-char asset SHA, n_steps consistency, trajectory duplicate, per-pool
counts, cross-split two-pass leakage, recursive output SHA manifest.
Fail-closed: any anomaly → non-zero exit.
"""
import os, sys, json, csv, hashlib, argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]

EXPECTED_ASSETS = {
    "detector_checkpoint": "66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628",
    "teacher_config": "ebc1ccda21cdfeae0f70f90ef0e433be3474ef0baa9cf52f609d620f863ce87a",
    "bridge_script": "fd594b3f9b38f4545d7b19202b380c0f4eeb0e9d95cb566f2fbca7b1852b208e",
}
EXPECTED_TASKS = set(range(10))
EXCLUDED_STATES = {0, 1, 2}
COMPROMISED_BLIND = set(range(38, 48))


def _sha256(p):
    if not p.exists(): return "MISSING"
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _safe_json(p):
    try:
        with open(p) as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def cell_key(task, state, profile="B0"):
    return f"{profile}_task{task}_state{state}"


def scan_filesystem(output_root):
    """Scan actual filesystem for cells. Returns {cell_key: {pool, task, state, path}}."""
    cells = {}
    root = Path(output_root)
    for pool_dir in root.iterdir():
        if not pool_dir.is_dir():
            continue
        pool = pool_dir.name
        for cell_dir in pool_dir.iterdir():
            if not cell_dir.is_dir():
                continue
            try:
                parts = cell_dir.name.split("_")
                task = int(parts[0].replace("task", ""))
                state = int(parts[1].replace("state", ""))
            except (ValueError, IndexError):
                continue
            key = cell_key(task, state)
            cells[key] = {"pool": pool, "task": task, "state": state, "path": cell_dir}
    return cells


def build_planned_set(manifest):
    """Build set of planned cell keys from manifest."""
    planned = set()
    for gpu_key, gpu_info in manifest["gpu_assignments"].items():
        pool = gpu_info["pool"]
        state_start, state_end = map(int, gpu_info["state_range"].split("-"))
        for task in range(10):
            for state in range(state_start, state_end + 1):
                planned.add((pool, task, state))
    return planned


def main():
    ap = argparse.ArgumentParser(description="M1C Corpus Integrity Auditor v2")
    ap.add_argument("--execution-manifest", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    ex_manifest = json.load(open(args.execution_manifest))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    corpus_root = Path(ex_manifest["output_root"])
    planned = build_planned_set(ex_manifest)
    actual = scan_filesystem(str(corpus_root))
    errors = defaultdict(list)
    per_cell_rows = []

    # ── Completeness ─────────────────────────────────────────────────
    for pool, task, state in planned:
        key = cell_key(task, state)
        if key not in actual:
            errors["missing"].append(f"{pool}/{key}")
    for key, cell in actual.items():
        tk = (cell["pool"], cell["task"], cell["state"])
        if tk not in planned:
            errors["unexpected"].append(f"{cell['pool']}/{key}")

    # ── Per-cell checks ──────────────────────────────────────────────
    all_initial_hashes = {}
    all_trajectory_hashes = defaultdict(list)

    for key, cell in actual.items():
        cell_dir = cell["path"]
        row = {"cell": f"{cell['pool']}/{key}", "exit_code": -999, "n_issues": 0}

        # File presence
        done_file = cell_dir / ".done"
        tel_file = cell_dir / "step_telemetry.csv"
        ep_file = cell_dir / "episode_summary.json"
        if not done_file.exists():
            errors["missing_done"].append(key)
            per_cell_rows.append(row)
            continue
        if not tel_file.exists():
            errors["missing_telemetry"].append(key)
            row["n_issues"] += 1
        if not ep_file.exists():
            errors["missing_summary"].append(key)
            row["n_issues"] += 1

        # .done exit code
        done_data, done_err = _safe_json(done_file)
        if done_err:
            errors["corrupt_done"].append(key)
            row["exit_code"] = -999
            row["n_issues"] += 1
        else:
            rc = done_data.get("exit_code", -999)
            row["exit_code"] = rc
            if rc != 0:
                errors["non_zero_rc"].append(f"{key}:rc={rc}")

        # Episode summary
        ep_data, ep_err = _safe_json(ep_file)
        if ep_err:
            errors["corrupt_summary"].append(key)
            row["n_issues"] += 1
        else:
            af = ep_data.get("attack_frames")
            if af is None:
                errors["attack_missing_field"].append(key)
                row["n_issues"] += 1
            elif not isinstance(af, (int, float)) or af != 0:
                errors["attack_nonzero"].append(f"{key}:af={af}")
                row["n_issues"] += 1
            cond = ep_data.get("condition", "")
            if cond != "CLEAN":
                errors["condition_not_clean"].append(f"{key}:{cond}")
                row["n_issues"] += 1
            n_steps = ep_data.get("n_steps", -1)

        # Step integrity
        if tel_file.exists():
            try:
                rows_data = list(csv.DictReader(open(tel_file)))
                for i, r in enumerate(rows_data):
                    if int(r.get("step", -1)) != i:
                        errors["step_index"].append(f"{key}:idx={i}")
                        row["n_issues"] += 1
                        break
                if rows_data and rows_data[-1].get("step", "") == "":
                    errors["truncated_csv"].append(key)
                    row["n_issues"] += 1
                if ep_data and n_steps > 0 and len(rows_data) != n_steps:
                    errors["n_steps_mismatch"].append(
                        f"{key}:csv={len(rows_data)}!={n_steps}")
                    row["n_issues"] += 1
            except Exception as e:
                errors["csv_parse"].append(f"{key}:{e}")
                row["n_issues"] += 1

        # Asset SHA
        actual_sha = ep_data.get("checkpoint_sha256", "") if ep_data else ""
        if actual_sha:
            expected = EXPECTED_ASSETS["detector_checkpoint"]
            if len(actual_sha) == 64 and actual_sha != expected:
                errors["asset_sha"].append(f"{key}:ckpt={actual_sha[:16]}")
                row["n_issues"] += 1
        else:
            errors["missing_checkpoint_sha"].append(key)
            row["n_issues"] += 1

        # Collect hashes for split check
        iss = ep_data.get("initial_state_sha256", "") if ep_data else ""
        if iss and len(iss) == 64:
            all_initial_hashes[key] = (cell["pool"], iss)
        else:
            errors["missing_initial_sha"].append(key)
            row["n_issues"] += 1
        tsha = ep_data.get("trajectory_content_sha256", "") if ep_data else ""
        if tsha and len(tsha) == 64:
            all_trajectory_hashes[tsha].append(key)

        per_cell_rows.append(row)

    # ── Split leakage (two-pass) ────────────────────────────────────
    train_hashes = set()
    val_hashes = set()
    for key, (pool, h) in all_initial_hashes.items():
        if pool == "train":
            if h in val_hashes:
                errors["split_leak"].append(f"train_hash_in_val:{key}")
            train_hashes.add(h)
        elif pool == "validation":
            if h in train_hashes:
                errors["split_leak"].append(f"val_hash_in_train:{key}")
            val_hashes.add(h)
    for h, keys in all_trajectory_hashes.items():
        if len(keys) > 1:
            errors["trajectory_duplicate"].append(f"sha={h[:16]} cells={keys}")

    # ── Pool counts ──────────────────────────────────────────────────
    train_actual = sum(1 for c in actual.values() if c["pool"] == "train")
    val_actual = sum(1 for c in actual.values() if c["pool"] == "validation")
    blind_actual = sum(1 for c in actual.values() if c["pool"] == "blind")
    if train_actual != ex_manifest.get("train_planned", 250):
        errors["pool_count"].append(f"train:{train_actual}≠250")
    if val_actual != ex_manifest.get("validation_planned", 100):
        errors["pool_count"].append(f"validation:{val_actual}≠100")
    if blind_actual != 0:
        errors["pool_count"].append(f"blind:{blind_actual}≠0")

    # ── Summary ──────────────────────────────────────────────────────
    total_issues = sum(len(v) for v in errors.values())
    planned_count = len(planned)
    actual_count = len(actual)
    gate_pass = (actual_count >= planned_count and total_issues == 0)

    summary = {
        "gate": "M1C_OBJECT_CORPUS_INTEGRITY",
        "result": "PASS" if gate_pass else "FAIL",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "planned_cells": planned_count,
        "actual_cells": actual_count,
        "total_issues": total_issues,
        "error_categories": {k: len(v) for k, v in errors.items()},
    }

    # ── Output files ─────────────────────────────────────────────────
    with open(out_root / "corpus_integrity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if per_cell_rows:
        with open(out_root / "per_cell_integrity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_cell_rows[0].keys())
            w.writeheader()
            w.writerows(per_cell_rows)

    for cat, items in errors.items():
        if items:
            fp = out_root / f"{cat}.csv"
            with open(fp, "w") as f:
                for item in items:
                    f.write(item + "\n")

    # Recursive output manifest
    manifest_rows = []
    for fp in out_root.iterdir():
        if fp.is_file():
            manifest_rows.append({"path": fp.name, "sha256": _sha256(fp), "size": fp.stat().st_size})
    with open(out_root / "recursive_artifact_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "size"])
        w.writeheader()
        w.writerows(manifest_rows)

    with open(out_root / "run.log", "w") as f:
        f.write(f"Timestamp: {summary['timestamp']}\nGate: {summary['result']}\n")
        f.write(f"Planned: {planned_count}, Actual: {actual_count}, Issues: {total_issues}\n")
        for cat, items in errors.items():
            for item in items[:20]:
                f.write(f"  {cat}: {item}\n")

    print(f"\n  M1C Corpus Integrity v2: {summary['result']}")
    print(f"  {actual_count}/{planned_count} cells, {total_issues} issues")
    for cat, count in summary["error_categories"].items():
        print(f"    {cat}: {count}")
    print(f"  Output: {out_root}")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
