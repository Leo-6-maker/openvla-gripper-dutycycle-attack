#!/usr/bin/env python3
"""Validate C6 primary-three-suite source condition outcome rows.

This is a pre-builder validator for the real server-side condition outcomes.
It does not run OpenVLA/LIBERO, train, tune, intervene, or fabricate rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = [
    "parent_id", "episode_key", "suite", "task_id", "condition",
    "clean_success_parent", "condition_success", "contact_quality_failure",
    "contact_quality_success", "nad_g", "delta_open", "qpos_response",
    "width_response", "arm_dev", "latency", "command_open_duty",
    "sustained_open_duty", "exact_prefix_shared",
    "clean_success_parent_denominator",
]
CONDITIONS = {"CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"}
PRIMARY = {"libero_goal", "libero_object", "libero_spatial"}
EXCLUDED = {"libero_10"}
BOOL_TRUE = {"1", "true", "TRUE", "yes", "YES"}
BOOL_FALSE = {"0", "false", "FALSE", "no", "NO", ""}
FLOAT_FIELDS = [
    "nad_g", "delta_open", "qpos_response", "width_response", "arm_dev",
    "latency", "command_open_duty", "sustained_open_duty",
]


class C6SourceValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise C6SourceValidationError(message)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b=""):
            h.update(chunk)
    return h.hexdigest()


def b(value: str, field: str, key: str) -> bool:
    if value in BOOL_TRUE:
        return True
    if value in BOOL_FALSE:
        return False
    fail(f"{key}: {field} must be boolean-like")


def f(value: str, field: str, key: str) -> float:
    if value is None or value == "" or str(value).upper() in {"NA", "N/A", "NOT_APPLICABLE"}:
        return float("nan")
    try:
        out = float(value)
    except ValueError:
        fail(f"{key}: {field} must be numeric")
    if not math.isfinite(out):
        fail(f"{key}: {field} must be finite")
    return out


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            fail("source CSV header mismatch")
        for line_no, row in enumerate(reader, start=2):
            key = f"line {line_no}"
            suite = row["suite"]
            condition = row["condition"]
            if suite in EXCLUDED:
                fail(f"{key}: excluded suite present: {suite}")
            if suite not in PRIMARY:
                fail(f"{key}: non-primary suite present: {suite}")
            if condition not in CONDITIONS:
                fail(f"{key}: unknown condition: {condition}")
            parsed = dict(row)
            for name in ["clean_success_parent", "condition_success", "contact_quality_failure", "contact_quality_success", "exact_prefix_shared", "clean_success_parent_denominator"]:
                parsed[name] = b(row[name], name, key)
            for name in FLOAT_FIELDS:
                parsed[name] = f(row[name], name, key)
            rows.append(parsed)
    if not rows:
        fail("source CSV has no rows")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suites = {row["suite"] for row in rows}
    missing_suites = sorted(PRIMARY - suites)
    if missing_suites:
        fail("missing primary suites: " + ", ".join(missing_suites))
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not row["clean_success_parent"]:
            fail(f"{row['parent_id']}: clean_success_parent must be true")
        if not row["exact_prefix_shared"]:
            fail(f"{row['parent_id']}:{row['condition']}: exact_prefix_shared must be true")
        if not row["clean_success_parent_denominator"]:
            fail(f"{row['parent_id']}:{row['condition']}: clean_success_parent_denominator must be true")
        key = (row["parent_id"], row["condition"])
        if key in seen:
            fail(f"duplicate parent/condition row: {key}")
        seen.add(key)
        by_parent[row["parent_id"]].append(row)
    incomplete = []
    for parent, items in by_parent.items():
        got = {row["condition"] for row in items}
        if got != CONDITIONS:
            incomplete.append({"parent_id": parent, "conditions": sorted(got)})
    if incomplete:
        fail(f"parents missing full condition set: {len(incomplete)}")
    return {
        "status": "PASS",
        "schema_version": "c6_source_condition_outcomes_validation_v1",
        "row_count": len(rows),
        "parent_count": len(by_parent),
        "suites": sorted(suites),
        "conditions": sorted(CONDITIONS),
        "libero_10_positive_denominator": "EXCLUDED",
        "exact_prefix_shared": True,
        "clean_success_parent_denominator": True,
        "label_mutation": "NOT_PERFORMED",
        "detector_training": "NOT_PERFORMED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    try:
        rows = read_rows(args.source_csv)
        report = validate_rows(rows)
        report["source_csv_sha256"] = sha256_file(args.source_csv)
    except (OSError, csv.Error, C6SourceValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
