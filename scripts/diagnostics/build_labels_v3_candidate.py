#!/usr/bin/env python3
"""Build labels_v3 candidate from labels_v2 plus Batch4 full-VIS gold rows."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--batch4-summary", default="tables/object_phase_response_batch4_vis_summary.csv")
    ap.add_argument("--batch4-candidates", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--output-labels", default="tables/object_phase_response_labels_v3_candidate.csv")
    ap.add_argument("--output-conflicts", default="tables/object_phase_response_labels_v3_conflicts.csv")
    ap.add_argument("--output-readiness", default="reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V3_CANDIDATE.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(v):
    return str(v if v is not None else "").strip()


def lower(v):
    return norm(v).lower()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in reader]


def key(row):
    return (norm(row.get("task_key")), norm(row.get("state_id")), norm(row.get("window_start")), norm(row.get("window_end")))


def is_train(row):
    return lower(row.get("label_status")) in {"positive", "negative"}


def blocked_batch4(row):
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in ["phase_d", "phase_e", "proxy", "silver", "infra_failed", "manual_review", "polluted"])


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not os.path.exists(args.labels_v2) or not os.path.exists(args.batch4_summary):
        write_csv(args.output_labels, ["status"], [{"status": "BLOCKED_MISSING_INPUTS"}])
        write_csv(args.output_conflicts, ["reason"], [])
        write_report(args, [], [], "NOT_READY", ["missing labels_v2 or Batch4 summary"])
        return 0
    fields, labels = read_csv(args.labels_v2)
    _, batch4 = read_csv(args.batch4_summary)
    out = [dict(r) for r in labels]
    conflicts = []
    existing = defaultdict(list)
    for row in out:
        existing[key(row)].append(row)
    for row in batch4:
        clean = lower(row.get("denominator_status")) == "clean"
        ok = lower(row.get("provenance_status")) == "ok"
        status = lower(row.get("label_status"))
        if not clean or not ok or blocked_batch4(row) or status not in {"positive", "negative"}:
            continue
        new = dict(row)
        new.update({
            "source_batch": "batch4",
            "label_status": status,
            "label_vulnerability_ready": "1" if status == "positive" else "0",
            "label_use": "train",
        })
        old = existing.get(key(new), [])
        if old:
            old_labels = {norm(r.get("label_vulnerability_ready")) for r in old}
            if norm(new.get("label_vulnerability_ready")) not in old_labels:
                conflicts.append({"task_key": key(new)[0], "state_id": key(new)[1], "window_start": key(new)[2], "window_end": key(new)[3], "reason": "duplicate_label_conflict"})
                continue
            continue
        out.append(new)
        existing[key(new)].append(new)
    all_fields = sorted(set(fields) | {k for row in out for k in row.keys()})
    write_csv(args.output_labels, all_fields, out)
    write_csv(args.output_conflicts, ["task_key", "state_id", "window_start", "window_end", "reason"], conflicts)
    train = [r for r in out if is_train(r)]
    pos = [r for r in train if lower(r.get("label_status")) == "positive"]
    neg = [r for r in train if lower(r.get("label_status")) == "negative"]
    hard = [r for r in train if "hard_negative" in lower(r.get("expected_role"))]
    controls = [r for r in train if "control" in lower(r.get("expected_role")) or "control" in lower(r.get("candidate_role"))]
    tasks = {norm(r.get("task_key")) for r in train}
    reason = []
    ready = True
    if len(train) < 35:
        ready = False; reason.append("train_rows < 35")
    if conflicts:
        ready = False; reason.append("conflicts present")
    if len(hard) < 8:
        ready = False; reason.append("hard negatives increased by fewer than 8 or unavailable")
    write_report(args, out, conflicts, "READY_FOR_DETECTOR_V3" if ready else "NOT_READY", reason, train, pos, neg, hard, controls, tasks)
    if args.dry_run:
        print(f"DRY RUN: labels_v3 rows={len(out)} train={len(train)} conflicts={len(conflicts)} status={'READY' if ready else 'NOT_READY'}")
    return 1 if conflicts else 0


def write_report(args, rows, conflicts, status, reasons, train=None, pos=None, neg=None, hard=None, controls=None, tasks=None):
    train = train or []; pos = pos or []; neg = neg or []; hard = hard or []; controls = controls or []; tasks = tasks or set()
    manual = [r for r in rows if lower(r.get("label_status")) == "manual_review"]
    infra = [r for r in rows if "infra_failed" in " ".join(lower(v) for v in r.values())]
    lines = [
        "# Object Phase Response Label Readiness V3 Candidate",
        "",
        f"**Training trigger recommendation**: {status}",
        f"**Rows**: {len(rows)}",
        f"**Train rows**: {len(train)}",
        f"**Positive count**: {len(pos)}",
        f"**Negative count**: {len(neg)}",
        f"**Hard negative count**: {len(hard)}",
        f"**Control count**: {len(controls)}",
        f"**Task coverage**: {len(tasks)}",
        f"**Conflict count**: {len(conflicts)}",
        f"**Manual review count**: {len(manual)}",
        f"**Infra failed count**: {len(infra)}",
        f"**Schema status**: {'pass' if not conflicts else 'fail'}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- {r}" for r in reasons] if reasons else ["- None."])
    os.makedirs(os.path.dirname(args.output_readiness) or ".", exist_ok=True)
    with open(args.output_readiness, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
