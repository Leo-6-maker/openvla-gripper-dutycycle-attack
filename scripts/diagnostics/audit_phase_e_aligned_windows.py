#!/usr/bin/env python3
"""Audit Phase E aligned windows before any GPU canary launch."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


AUDIT_FIELDS = ["check_id", "severity", "status", "count", "detail"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-windows", default="tables/phaseE_aligned_windows_v0_server.csv")
    ap.add_argument("--output-csv", default="tables/phaseE_aligned_windows_audit_v0.csv")
    ap.add_argument("--output-report", default="reports/PHASE_E_ALIGNED_WINDOWS_AUDIT_V0.md")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in reader]
    return rows


def truthy(value):
    return lower(value) in {"1", "true", "yes", "recommended"}


def add(audit, check_id, severity, status, count, detail):
    audit.append({"check_id": check_id, "severity": severity, "status": status, "count": count, "detail": detail})


def row_label(row):
    value = lower(row.get("full_vis_label") or row.get("label") or row.get("label_vulnerability_ready"))
    if value in {"1", "true", "positive"}:
        return "positive"
    if value in {"0", "false", "negative"}:
        return "negative"
    return "unknown"


def main():
    args = parse_args()
    audit = []
    if not os.path.exists(args.aligned_windows):
        add(audit, "input_exists", "hard_fail", "fail", 1, f"aligned windows CSV missing: {args.aligned_windows}")
        write_outputs(args, [], audit, "BLOCKED_MISSING_ALIGNED_WINDOWS")
        return 0

    rows = read_csv(args.aligned_windows)
    recommended = [r for r in rows if truthy(r.get("recommended_for_phaseE"))]
    bad_qpos_source = [
        r for r in recommended
        if lower(r.get("qpos_source")) != "mujoco_trace" and lower(r.get("phase_alignment_source")) != "mujoco_trace"
    ]
    add(audit, "recommended_requires_mujoco", "hard_fail", "fail" if bad_qpos_source else "pass", len(bad_qpos_source), "recommended rows must use MuJoCo qpos")
    obs_zero = [r for r in recommended if lower(r.get("qpos_source_warning")) == "obs_qpos_all_zero_untrusted"]
    add(audit, "recommended_not_obs_all_zero", "hard_fail", "fail" if obs_zero else "pass", len(obs_zero), "recommended rows cannot have all-zero obs qpos warning")
    natural = [r for r in recommended if lower(r.get("qpos_phase_class")) == "natural_open"]
    add(audit, "recommended_not_natural_open", "hard_fail", "fail" if natural else "pass", len(natural), "recommended rows cannot be natural_open")
    infra = [r for r in recommended if "infra_failed" in lower(r.get("provenance_status"))]
    add(audit, "recommended_not_infra_failed", "hard_fail", "fail" if infra else "pass", len(infra), "recommended rows cannot have infra_failed provenance")
    polluted = [r for r in recommended if lower(r.get("denominator_status")) == "polluted"]
    add(audit, "recommended_not_polluted", "hard_fail", "fail" if polluted else "pass", len(polluted), "recommended rows cannot have polluted denominator")
    pos = [r for r in recommended if row_label(r) == "positive"]
    neg = [r for r in recommended if row_label(r) == "negative"]
    add(audit, "recommended_positive_negative_presence", "hard_fail", "fail" if not pos or not neg else "pass", len(pos) + len(neg), "need at least one positive and one negative recommended row before canary")
    pending = [r for r in recommended if lower(r.get("denominator_status")) != "clean"]
    add(audit, "recommended_denominator_clean_or_pending", "warning", "warn" if pending else "pass", len(pending), "non-clean denominator rows must remain pending_denominator")

    status = "PASS" if not [a for a in audit if a["severity"] == "hard_fail" and a["status"] == "fail"] else "FAIL"
    write_outputs(args, rows, audit, status)
    return 1 if status == "FAIL" else 0


def write_outputs(args, rows, audit, status):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)

    recommended = [r for r in rows if truthy(r.get("recommended_for_phaseE"))]
    pos = [r for r in recommended if row_label(r) == "positive"]
    neg = [r for r in recommended if row_label(r) == "negative"]
    reasons = Counter(norm(r.get("reason")) for r in rows)
    obs_only = [r for r in rows if lower(r.get("qpos_source")) == "obs_trace"]
    missing_qpos = [r for r in rows if lower(r.get("qpos_phase_class")) == "missing"]
    natural = [r for r in rows if lower(r.get("qpos_phase_class")) == "natural_open"]
    canary_ready = status == "PASS" and bool(pos) and bool(neg)
    lines = [
        "# Phase E Aligned Windows Audit V0",
        "",
        f"**Status**: {status}",
        f"**Input**: `{args.aligned_windows}`",
        f"**Total rows**: {len(rows)}",
        f"**Recommended rows**: {len(recommended)}",
        f"**Recommended positives**: {len(pos)}",
        f"**Recommended negatives**: {len(neg)}",
        f"**Missing qpos rows**: {len(missing_qpos)}",
        f"**Obs-only rows**: {len(obs_only)}",
        f"**Natural-open rows**: {len(natural)}",
        f"**Canary ready**: {'true' if canary_ready else 'false'}",
        f"**Reason**: {'ready' if canary_ready else 'not enough safe recommended positive/negative rows or hard-fail audit'}",
        "",
        "This audit must pass before any Phase E GPU canary. It is CPU-only.",
        "",
        "## Blocked Reason Distribution",
        "",
    ]
    if reasons:
        for reason, count in sorted(reasons.items()):
            lines.append(f"- `{reason or 'blank'}`: {count}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Checks", ""])
    for row in audit:
        lines.append(f"- `{row['check_id']}`: {row['status']} ({row['count']}) {row['detail']}")
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
