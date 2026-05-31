#!/usr/bin/env python3
"""Audit raw and causal-relative proprio feature distributions by suite."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


CSV_FIELDS = [
    "suite",
    "feature",
    "transform",
    "n",
    "mean",
    "std",
    "min",
    "max",
    "missing_rate",
    "nan_rate",
    "inf_rate",
    "zero_rate",
    "object_mean_abs_distance",
]


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out


def _stats(values):
    present = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not present:
        return 0, "", "", "", "", 1.0, 0.0, 0.0, 0.0
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / max(len(present), 1)
    missing = sum(v is None for v in values) / max(len(values), 1)
    nan = sum((v is not None and math.isnan(v)) for v in values) / max(len(values), 1)
    inf = sum((v is not None and math.isinf(v)) for v in values) / max(len(values), 1)
    zero = sum(v == 0.0 for v in present) / max(len(present), 1)
    return len(present), mean, math.sqrt(var), min(present), max(present), missing, nan, inf, zero


def write_schema_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def audit(input_csv: Path, output_csv: Path) -> None:
    rows = list(csv.DictReader(input_csv.open(newline="")))
    initial = {}
    for row in rows:
        key = row.get("episode_key") or "::".join(str(row.get(k, "")) for k in ("suite", "task_id", "state_id", "seed", "run_id"))
        if key not in initial:
            initial[key] = {axis: _to_float(row.get(axis)) for axis in ("eef_x", "eef_y", "eef_z")}

    values = defaultdict(list)
    for row in rows:
        suite = row.get("suite", "unknown")
        key = row.get("episode_key") or "::".join(str(row.get(k, "")) for k in ("suite", "task_id", "state_id", "seed", "run_id"))
        for axis in ("eef_x", "eef_y", "eef_z"):
            raw = _to_float(row.get(axis))
            values[(suite, axis, "raw")].append(raw)
            base = initial.get(key, {}).get(axis)
            rel = None if raw is None or base is None else raw - base
            values[(suite, axis, "relative_initial")].append(rel)
        for field in ("eef_vx", "eef_vy", "eef_vz", "gripper_qpos", "gripper_width", "gripper_command", "action_gripper"):
            values[(suite, field, "raw")].append(_to_float(row.get(field)))

    means = {}
    output_rows = []
    for key, vals in sorted(values.items()):
        suite, feature, transform = key
        n, mean, std, min_v, max_v, missing, nan, inf, zero = _stats(vals)
        means[key] = mean if mean != "" else None
        output_rows.append({
            "suite": suite,
            "feature": feature,
            "transform": transform,
            "n": n,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "missing_rate": missing,
            "nan_rate": nan,
            "inf_rate": inf,
            "zero_rate": zero,
            "object_mean_abs_distance": "",
        })

    object_means = {(feature, transform): mean for (suite, feature, transform), mean in means.items() if suite == "libero_object" and mean is not None}
    for row in output_rows:
        obj_mean = object_means.get((row["feature"], row["transform"]))
        mean = row["mean"]
        if obj_mean is not None and mean != "":
            row["object_mean_abs_distance"] = abs(float(mean) - float(obj_mean))

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
    audit(Path(args.input_csv), output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
