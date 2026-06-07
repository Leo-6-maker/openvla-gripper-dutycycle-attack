#!/usr/bin/env python3
"""Audit adaptive VIS-1R label/status values for pending-negative parsing."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


AUDIT_FIELDS = [
    "source_name", "field", "value", "count", "pending_negative_candidate_count",
    "infra_excluded_count",
]
WATCH_FIELDS = [
    "status", "stage", "label_1r", "label_confidence", "mechanism_status",
    "denominator_status", "outcome", "result", "failure_reason",
    "vis_open_count", "done", "qpos_delta",
]
BAD_TOKENS = ("infra", "infra_failed", "localization_fail", "oom", "xid", "manual_review", "polluted", "precheck_failed")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="tables/adaptive_vis_1r_screening_summary.csv")
    ap.add_argument("--provenance", default="tables/adaptive_vis_1r_provenance.csv")
    ap.add_argument("--output-csv", default="tables/adaptive_1r_label_value_audit.csv")
    ap.add_argument("--output-report", default="reports/ADAPTIVE_1R_LABEL_VALUE_AUDIT.md")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def read_csv(path):
    if not path or not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in reader]
        return [norm(x).lstrip("\ufeff") for x in (reader.fieldnames or [])], rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def bad_token(row):
    text = " ".join(lower(v) for v in row.values())
    return next((tok for tok in BAD_TOKENS if tok in text), "")


def parse_float(value):
    try:
        text = norm(value)
        return None if text == "" else float(text)
    except Exception:
        return None


def parse_bool(value):
    v = lower(value)
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    return None


def is_pending_negative_candidate(row):
    if bad_token(row):
        return False
    label_text = " ".join(lower(row.get(f)) for f in ["label_confidence", "label_1r", "status"])
    if "pending_negative_1r" in label_text or lower(row.get("status")) == "vis1r_pending_negative":
        return True
    completed = lower(row.get("status")) in {"vis1r_done", "completed", "done"} or lower(row.get("stage")) in {"vis1r_done", "completed", "done"} or lower(row.get("result")) in {"completed", "done"}
    if not completed:
        return False
    if "silver_positive_1r" in label_text or "provisional_silver_positive_1r" in label_text:
        return False
    mechanism = lower(row.get("mechanism_status"))
    if mechanism and mechanism not in {"clean", "mechanism_clean"}:
        return False
    open_count = parse_float(row.get("vis_open_count") or row.get("VIS_OPEN") or row.get("vis_OPEN"))
    done = parse_bool(row.get("done"))
    task_failed = parse_bool(row.get("task_failure_positive") or row.get("label_task_failure"))
    if open_count is not None and open_count <= 1 and task_failed is False:
        return True
    if done is False and task_failed is False:
        return True
    return False


def field_counts(source_name, rows):
    output = []
    for field in WATCH_FIELDS:
        counter = Counter(norm(row.get(field)) if norm(row.get(field)) != "" else "<MISSING_OR_EMPTY>" for row in rows)
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:100]:
            subset = [row for row in rows if (norm(row.get(field)) if norm(row.get(field)) != "" else "<MISSING_OR_EMPTY>") == value]
            output.append({
                "source_name": source_name,
                "field": field,
                "value": value,
                "count": str(count),
                "pending_negative_candidate_count": str(sum(1 for row in subset if is_pending_negative_candidate(row))),
                "infra_excluded_count": str(sum(1 for row in subset if bad_token(row))),
            })
    return output


def write_report(path, args, summary_rows, prov_rows, audit_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pending_summary = sum(1 for row in summary_rows if is_pending_negative_candidate(row))
    pending_prov = sum(1 for row in prov_rows if is_pending_negative_candidate(row))
    bad_summary = sum(1 for row in summary_rows if bad_token(row))
    bad_prov = sum(1 for row in prov_rows if bad_token(row))
    lines = [
        "# Adaptive 1R Label Value Audit",
        "",
        f"**Summary path**: `{args.summary}`",
        f"**Summary rows**: {len(summary_rows)}",
        f"**Provenance path**: `{args.provenance}`",
        f"**Provenance rows**: {len(prov_rows)}",
        f"**Pending-negative candidates in summary**: {pending_summary}",
        f"**Pending-negative candidates in provenance**: {pending_prov}",
        f"**Infra/manual/polluted excluded summary rows**: {bad_summary}",
        f"**Infra/manual/polluted excluded provenance rows**: {bad_prov}",
        "",
        "This audit is CPU-only and read-only with respect to the input snapshot.",
        "",
        "## Interpretation",
        "",
    ]
    if pending_summary + pending_prov == 0:
        lines.append("- Current snapshot has no usable pending-negative 1R rows under the hardened parser.")
    else:
        lines.append("- Usable pending-negative 1R rows exist and should be ranked for 3R confirmation, not used as train negatives.")
    if not os.path.exists(args.summary):
        lines.append("- Requested summary path is missing; audit used no summary rows from that path.")
    if not os.path.exists(args.provenance):
        lines.append("- Requested provenance path is missing; audit used no provenance rows from that path.")
    lines.extend(["", "## Value Counts", ""])
    for row in audit_rows[:80]:
        lines.append(f"- `{row['source_name']}.{row['field']}` = `{row['value']}`: {row['count']} rows; pending candidates={row['pending_negative_candidate_count']}; infra/manual/polluted excluded={row['infra_excluded_count']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    _, summary_rows = read_csv(args.summary)
    _, prov_rows = read_csv(args.provenance)
    audit_rows = field_counts("summary", summary_rows) + field_counts("provenance", prov_rows)
    write_csv(args.output_csv, audit_rows)
    write_report(args.output_report, args, summary_rows, prov_rows, audit_rows)
    print(f"audit rows={len(audit_rows)} summary_rows={len(summary_rows)} provenance_rows={len(prov_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
