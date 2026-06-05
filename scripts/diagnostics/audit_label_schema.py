#!/usr/bin/env python3
"""Audit vulnerability-ready label CSV schema and leakage boundaries.

This script is intentionally CPU-only and source-read-only. It writes a compact
audit table and markdown report, then exits non-zero when hard failures are
found.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


ALLOWED_STATUS = {"positive", "negative", "ignore", "manual_review"}
TRAIN_STATUS = {"positive", "negative"}
REQUIRED_FIELDS = {
    "source_batch",
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "label_status",
    "label_vulnerability_ready",
    "denominator_type",
    "provenance_status",
}
BLOCKED_TRAIN_TOKENS = {
    "polluted",
    "random_failed",
    "denominator_failed",
    "infra_failed",
    "xid",
    "oom",
    "missing_trace",
    "provenance_failed",
    "schema_incomplete",
    "manual_review",
    "ambiguous_merge",
}
FORBIDDEN_FEATURE_PATTERNS = [
    r"^claim_usable$",
    r"^done$",
    r"^vis_open$",
    r"^vis_open_count$",
    r"^qpos_after_attack$",
    r"^vis_qpos_opening_delta$",
    r"^denominator_clean$",
    r"^task_failure_positive$",
    r"^physical_bridge_positive$",
    r"random.*outcome",
    r"oracle.*outcome",
    r"manual.*audit.*outcome",
    r"attack.*outcome",
    r"attack.*result",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--output-csv", default="tables/label_schema_audit_v2.csv")
    ap.add_argument("--output-report", default="reports/LABEL_SCHEMA_AUDIT_V2.md")
    return ap.parse_args()


def add(rows, check_id, severity, status, detail, count=0):
    rows.append(
        {
            "check_id": check_id,
            "severity": severity,
            "status": status,
            "count": count,
            "detail": detail,
        }
    )


def norm(value):
    return str(value or "").strip()


def truthy_label(value):
    v = norm(value).lower()
    if v in {"1", "true", "yes", "positive"}:
        return "1"
    if v in {"0", "false", "no", "negative"}:
        return "0"
    return v


def field_matches_forbidden(field):
    low = field.strip().lower()
    return any(re.search(pattern, low) for pattern in FORBIDDEN_FEATURE_PATTERNS)


def row_blocked_reason(row):
    text_fields = [
        "label_status",
        "exclusion_reason",
        "exclusion_or_uncertain_reason",
        "taxonomy",
        "taxonomy_label",
        "denominator_type",
        "denominator_status",
        "provenance_status",
        "provenance_note",
        "candidate_role",
        "notes",
    ]
    blob = " ".join(norm(row.get(f)).lower() for f in text_fields)
    return sorted(token for token in BLOCKED_TRAIN_TOKENS if token in blob)


def row_marked_for_train(row):
    train_fields = ["label_use", "split", "train_split", "is_train", "use_for_training"]
    for field in train_fields:
        value = norm(row.get(field)).lower()
        if value in {"train", "training", "positive", "negative", "1", "true", "yes"}:
            return True
    return False


def write_outputs(args, audit_rows, labels_path, n_rows, n_hard):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["check_id", "severity", "status", "count", "detail"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(audit_rows)

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    failures = [r for r in audit_rows if r["severity"] == "hard_fail" and r["status"] == "fail"]
    warnings = [r for r in audit_rows if r["severity"] == "warning" and r["status"] != "pass"]
    verdict = "PASS" if not failures else "FAIL"
    lines = [
        "# Label Schema Audit V2",
        "",
        f"**Input**: `{labels_path}`",
        f"**Rows**: {n_rows}",
        f"**Verdict**: **{verdict}**",
        "",
        "## Blocking Issues",
        "",
    ]
    if failures:
        for r in failures:
            lines.append(f"- `{r['check_id']}`: {r['detail']}")
    else:
        lines.append("- None.")
    lines += ["", "## Warnings", ""]
    if warnings:
        for r in warnings:
            lines.append(f"- `{r['check_id']}`: {r['detail']}")
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Notes",
        "",
        "- Only `positive` and `negative` rows are train-eligible.",
        "- Outcome/attack fields may exist as labels or audit metadata, but detector training must not use them as inputs.",
        "- This script does not start rollout, VIS, GPU work, or server jobs.",
        "",
    ]
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return n_hard


def main():
    args = parse_args()
    audit = []
    if not os.path.exists(args.labels_csv):
        add(audit, "input_exists", "hard_fail", "fail", f"labels CSV not found: {args.labels_csv}", 0)
        n_hard = write_outputs(args, audit, args.labels_csv, 0, 1)
        return 1 if n_hard else 0

    with open(args.labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    field_set = set(fields)
    missing = sorted(REQUIRED_FIELDS - field_set)
    if "candidate_role" not in field_set and "phase_bin_proxy" not in field_set:
        missing.append("candidate_role_or_phase_bin_proxy")
    add(
        audit,
        "required_fields",
        "hard_fail",
        "fail" if missing else "pass",
        "missing=" + ",".join(missing) if missing else "all required fields present",
        len(missing),
    )

    bad_status = [i + 2 for i, r in enumerate(rows) if norm(r.get("label_status")) not in ALLOWED_STATUS]
    add(
        audit,
        "label_status_values",
        "hard_fail",
        "fail" if bad_status else "pass",
        f"invalid label_status rows={bad_status[:20]}" if bad_status else "all statuses allowed",
        len(bad_status),
    )

    train_bad = []
    train_blocked = []
    non_status_train = []
    for i, r in enumerate(rows):
        status = norm(r.get("label_status"))
        label = truthy_label(r.get("label_vulnerability_ready"))
        if status in TRAIN_STATUS and label not in {"0", "1"}:
            train_bad.append(i + 2)
        if status in TRAIN_STATUS:
            blocked = row_blocked_reason(r)
            if blocked:
                train_blocked.append((i + 2, "|".join(blocked)))
        elif label in {"0", "1"}:
            # Non-train rows can carry label metadata, but should not be train selected.
            pass
        if status not in TRAIN_STATUS and row_marked_for_train(r):
            non_status_train.append(i + 2)
    add(
        audit,
        "train_labels_binary",
        "hard_fail",
        "fail" if train_bad else "pass",
        f"positive/negative rows with non-binary label rows={train_bad[:20]}" if train_bad else "train labels are binary",
        len(train_bad),
    )
    add(
        audit,
        "train_rows_status_source",
        "hard_fail",
        "fail" if non_status_train else "pass",
        f"non-positive/negative rows marked for train={non_status_train[:20]}"
        if non_status_train
        else "train selection comes only from positive/negative rows",
        len(non_status_train),
    )
    add(
        audit,
        "blocked_rows_excluded_from_train",
        "hard_fail",
        "fail" if train_blocked else "pass",
        f"blocked train rows={train_blocked[:20]}" if train_blocked else "blocked taxonomy/provenance rows excluded from train",
        len(train_blocked),
    )

    labels_by_key = defaultdict(set)
    for r in rows:
        key = (norm(r.get("task_key")), norm(r.get("state_id")), norm(r.get("window_start")), norm(r.get("window_end")))
        labels_by_key[key].add((norm(r.get("label_status")), truthy_label(r.get("label_vulnerability_ready"))))
    conflicts = [(k, sorted(v)) for k, v in labels_by_key.items() if len(v) > 1]
    add(
        audit,
        "duplicate_conflict",
        "hard_fail",
        "fail" if conflicts else "pass",
        f"conflicting duplicate labels={conflicts[:10]}" if conflicts else "no conflicting duplicate labels",
        len(conflicts),
    )

    forbidden_cols = [f for f in fields if field_matches_forbidden(f)]
    add(
        audit,
        "forbidden_feature_columns_present",
        "warning",
        "warn" if forbidden_cols else "pass",
        "label-only/outcome columns present; detector must exclude: " + ",".join(forbidden_cols)
        if forbidden_cols
        else "no forbidden feature columns present",
        len(forbidden_cols),
    )

    n_hard = sum(1 for r in audit if r["severity"] == "hard_fail" and r["status"] == "fail")
    write_outputs(args, audit, args.labels_csv, len(rows), n_hard)
    return 1 if n_hard else 0


if __name__ == "__main__":
    sys.exit(main())
