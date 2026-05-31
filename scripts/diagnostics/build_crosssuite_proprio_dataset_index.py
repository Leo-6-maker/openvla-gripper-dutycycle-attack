#!/usr/bin/env python3
"""Build a metadata-only index for future CrossSuite-ProprioNoStep-v2 work."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CSV_FIELDS = [
    "suite",
    "task_id",
    "task_name",
    "state_id",
    "seed",
    "run_id",
    "run_dir",
    "clean_success",
    "mechanism_type",
    "mechanism_eligible",
    "available_features",
    "available_labels",
    "missing_fields",
    "split_candidate",
    "notes",
]
REQUIRED_FEATURES = ("gripper_qpos", "gripper_width", "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz", "action_gripper")
REQUIRED_LABELS = ("teacher_phase", "teacher_hazard", "teacher_release_safe")


def write_schema_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def build_index(input_csv: Path, output_csv: Path) -> None:
    seen = {}
    for row in csv.DictReader(input_csv.open(newline="")):
        key = row.get("episode_key") or "::".join(str(row.get(k, "")) for k in ("suite", "task_id", "state_id", "seed", "run_id"))
        entry = seen.setdefault(key, {
            "suite": row.get("suite", ""),
            "task_id": row.get("task_id", ""),
            "task_name": row.get("task_name", ""),
            "state_id": row.get("state_id", ""),
            "seed": row.get("seed", ""),
            "run_id": row.get("run_id", ""),
            "run_dir": row.get("run_dir", ""),
            "clean_success": row.get("clean_success", ""),
            "mechanism_type": row.get("mechanism_type", ""),
            "mechanism_eligible": row.get("mechanism_eligible", ""),
            "_n_rows": 0,
            "_features": set(),
            "_labels": set(),
        })
        entry["_n_rows"] += 1
        for field in REQUIRED_FEATURES:
            if row.get(field) not in (None, ""):
                entry["_features"].add(field)
        for field in REQUIRED_LABELS:
            if row.get(field) not in (None, ""):
                entry["_labels"].add(field)

    output_rows = []
    for entry in seen.values():
        missing = (set(REQUIRED_FEATURES) - entry["_features"]) | (set(REQUIRED_LABELS) - entry["_labels"])
        has_all = not missing
        has_core_z = {"eef_z", "gripper_qpos", "gripper_width", "action_gripper"}.issubset(entry["_features"]) and set(REQUIRED_LABELS).issubset(entry["_labels"])
        if has_all:
            split_candidate = "yes"
        elif has_core_z:
            split_candidate = "partial_eef_z_only"
        else:
            split_candidate = "no"
        output_rows.append({
            "suite": entry["suite"],
            "task_id": entry["task_id"],
            "task_name": entry["task_name"],
            "state_id": entry["state_id"],
            "seed": entry["seed"],
            "run_id": entry["run_id"],
            "run_dir": entry["run_dir"],
            "clean_success": entry["clean_success"],
            "mechanism_type": entry["mechanism_type"],
            "mechanism_eligible": entry["mechanism_eligible"],
            "available_features": ";".join(sorted(entry["_features"])),
            "available_labels": ";".join(sorted(entry["_labels"])),
            "missing_fields": ";".join(sorted(missing)),
            "split_candidate": split_candidate,
            "notes": "metadata_only_no_training",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dry-run-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()
    output_csv = Path(args.output_csv)
    if args.print_schema:
        print_schema()
    if args.dry_run_schema or args.dry_run or args.print_schema:
        write_schema_csv(output_csv)
        return 0
    if not args.input_csv:
        raise SystemExit("--input_csv is required unless --dry-run-schema is set")
    build_index(Path(args.input_csv), output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
