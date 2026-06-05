#!/usr/bin/env python3
"""Audit Batch4 candidate schema and safety boundaries.

CPU-only. This script validates candidate metadata before any server-side
rollout/VIS scheduling.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict


REQUIRED_COLUMNS = {
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "denominator_plan",
    "expected_role",
    "qpos_verification_status",
    "gripper_qpos_used",
    "gripper_qpos_source_priority",
    "true_closed",
    "phase_proxy_mismatch",
}
QPOS_COLUMNS = {
    "qpos_verification_status",
    "gripper_qpos_used",
    "gripper_qpos_source_priority",
    "true_closed",
    "natural_open",
    "phase_proxy_mismatch",
}
PROXY_BLOCK_TOKENS = ["phase_d", "phase e", "phase_e", "command_proxy", "low_budget", "proxy_label", "silver_proxy"]
GPU_FORBIDDEN_PATTERNS = [
    r"\bgpu\s*3\b",
    r"\bgpu\s*7\b",
    r"\bcuda_visible_devices\s*=\s*[^,\s]*3\b",
    r"\bcuda_visible_devices\s*=\s*[^,\s]*7\b",
    r"\b2\s*,\s*3\b",
    r"\b6\s*,\s*7\b",
]
AUDIT_FIELDS = ["check_id", "severity", "status", "count", "detail"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--output-csv", default="tables/batch4_candidate_schema_audit.csv")
    ap.add_argument("--output-report", default="reports/BATCH4_CANDIDATE_SCHEMA_AUDIT.md")
    ap.add_argument("--min-hard-negatives", type=int, default=6)
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def add(audit, check_id, severity, status, count, detail):
    audit.append(
        {
            "check_id": check_id,
            "severity": severity,
            "status": status,
            "count": count,
            "detail": detail,
        }
    )


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(field).lstrip("\ufeff") for field in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({norm(k).lstrip("\ufeff"): v for k, v in row.items()})
        return fields, rows


def key(row):
    return (norm(row.get("task_key")), norm(row.get("state_id")), norm(row.get("window_start")), norm(row.get("window_end")))


def row_text(row):
    return " ".join(lower(v) for v in row.values())


def has_phase_de_proxy(row):
    text = row_text(row)
    return any(token in text for token in PROXY_BLOCK_TOKENS)


def has_forbidden_gpu_assumption(row):
    text = row_text(row)
    return any(re.search(pattern, text) for pattern in GPU_FORBIDDEN_PATTERNS)


def write_csv(path, audit):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)


def write_report(path, args, audit, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hard_fails = [row for row in audit if row["severity"] == "hard_fail" and row["status"] == "fail"]
    warnings = [row for row in audit if row["severity"] == "warning" and row["status"] != "pass"]
    status = "PASS" if not hard_fails else "FAIL"
    role_counts = Counter(row.get("expected_role", "") for row in rows)
    lines = [
        "# Batch4 Candidate Schema Audit",
        "",
        f"**Status**: {status}",
        f"**Input**: `{args.candidates_csv}`",
        f"**Rows**: {len(rows)}",
        "",
        "This is a CPU-only schema/safety audit. It does not run rollout, VIS, GPU work, watcher jobs, or detector training.",
        "",
        "## Blocking Issues",
        "",
    ]
    if hard_fails:
        lines.extend(f"- `{row['check_id']}`: {row['detail']}" for row in hard_fails)
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- `{row['check_id']}`: {row['detail']}" for row in warnings)
    else:
        lines.append("- None.")
    lines.extend(["", "## Expected Role Counts", ""])
    if role_counts:
        for role, count in sorted(role_counts.items()):
            lines.append(f"- `{role}`: {count}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Safety Checks",
            "",
            "- Candidate rows must not assume GPU3 or GPU7, or disabled pairs 2,3 / 6,7.",
            "- `denominator_plan`, `expected_role`, and qpos verification fields are required.",
            "- Phase D/E proxy labels are not valid Batch4 gold candidates.",
            "- Duplicated task/state/window candidates hard-fail.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    audit = []
    if not os.path.exists(args.candidates_csv):
        add(audit, "input_exists", "hard_fail", "fail", 1, f"candidate CSV not found: {args.candidates_csv}")
        write_csv(args.output_csv, audit)
        write_report(args.output_report, args, audit, [])
        return 1

    fields, rows = read_csv(args.candidates_csv)
    field_set = set(fields)
    add(
        audit,
        "candidate_rows_present",
        "hard_fail",
        "fail" if not rows else "pass",
        len(rows),
        "candidate CSV has no rows" if not rows else f"candidate rows={len(rows)}",
    )

    missing = sorted(REQUIRED_COLUMNS - field_set)
    add(
        audit,
        "required_columns",
        "hard_fail",
        "fail" if missing else "pass",
        len(missing),
        "missing=" + ",".join(missing) if missing else "all required columns present",
    )

    missing_qpos_values = 0
    if QPOS_COLUMNS.issubset(field_set):
        for row in rows:
            if lower(row.get("qpos_verification_status")) == "qpos_verified_true_closed" and norm(row.get("gripper_qpos_used")) == "":
                missing_qpos_values += 1
    add(
        audit,
        "qpos_verification_values",
        "hard_fail",
        "fail" if missing_qpos_values else "pass",
        missing_qpos_values,
        f"qpos_verified rows missing gripper_qpos_used={missing_qpos_values}" if missing_qpos_values else "qpos verification values present",
    )

    duplicate_keys = [k for k, vals in group_by_key(rows).items() if len(vals) > 1]
    add(
        audit,
        "duplicate_task_state_window",
        "hard_fail",
        "fail" if duplicate_keys else "pass",
        len(duplicate_keys),
        f"duplicates={duplicate_keys[:10]}" if duplicate_keys else "no duplicate task/state/window rows",
    )

    gpu_bad = sum(1 for row in rows if has_forbidden_gpu_assumption(row))
    add(
        audit,
        "gpu3_gpu7_assumptions",
        "hard_fail",
        "fail" if gpu_bad else "pass",
        gpu_bad,
        f"rows with forbidden GPU3/GPU7 or disabled pair assumptions={gpu_bad}" if gpu_bad else "no GPU3/GPU7 assumptions found",
    )

    missing_denominator = sum(1 for row in rows if norm(row.get("denominator_plan")) == "")
    add(
        audit,
        "denominator_plan_present",
        "hard_fail",
        "fail" if missing_denominator else "pass",
        missing_denominator,
        f"rows missing denominator_plan={missing_denominator}" if missing_denominator else "all rows have denominator_plan",
    )

    missing_role = sum(1 for row in rows if norm(row.get("expected_role")) == "")
    add(
        audit,
        "expected_role_present",
        "hard_fail",
        "fail" if missing_role else "pass",
        missing_role,
        f"rows missing expected_role={missing_role}" if missing_role else "all rows have expected_role",
    )

    proxy_rows = sum(1 for row in rows if has_phase_de_proxy(row))
    add(
        audit,
        "no_phase_d_e_proxy_labels",
        "hard_fail",
        "fail" if proxy_rows else "pass",
        proxy_rows,
        f"Phase D/E proxy labels found={proxy_rows}" if proxy_rows else "no Phase D/E proxy labels found",
    )

    hard_negatives = [
        row
        for row in rows
        if "hard_negative" in lower(row.get("expected_role")) or lower(row.get("source_error_type")) == "fp"
    ]
    add(
        audit,
        "enough_hard_negatives",
        "hard_fail",
        "fail" if len(hard_negatives) < args.min_hard_negatives else "pass",
        len(hard_negatives),
        f"hard negatives={len(hard_negatives)}, min={args.min_hard_negatives}",
    )

    qpos_true_closed = sum(1 for row in hard_negatives if lower(row.get("qpos_verification_status")) == "qpos_verified_true_closed")
    add(
        audit,
        "hard_negative_qpos_verified_true_closed",
        "warning",
        "warn" if qpos_true_closed < min(args.min_hard_negatives, len(hard_negatives)) else "pass",
        qpos_true_closed,
        f"qpos-verified true_closed hard negatives={qpos_true_closed}",
    )

    write_csv(args.output_csv, audit)
    write_report(args.output_report, args, audit, rows)
    return 1 if any(row["severity"] == "hard_fail" and row["status"] == "fail" for row in audit) else 0


def group_by_key(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
