#!/usr/bin/env python3
"""Build detector-v2.5 incremental dataset from labels_v2 plus VIS-1R audit rows.

This script is a CPU-only CSV wrapper. It does not run VIS, rollout, watcher, or
detector training. Pending VIS-1R negatives are kept for audit/ranking only and
are never emitted as train negatives.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
SILVER_CONF = {"silver_positive_1r", "provisional_silver_positive_1r"}
PENDING_NEG = {"pending_negative_1r"}
BAD_TOKENS = ("infra", "infra_failed", "localization_fail", "oom", "xid", "manual_review", "polluted", "action_confounded", "precheck_failed")
OUT_FIELDS = [
    "task_key", "state_id", "window_start", "window_end",
    "label_status", "label_vulnerability_ready", "label_source",
    "train_use", "train_variant", "sample_weight",
    "source_batch", "candidate_role", "expected_role", "phase_bin_proxy",
    "denominator_status", "provenance_status", "mechanism_status",
    "label_confidence", "label_1r", "inclusion_status", "exclusion_reason",
]
AUDIT_FIELDS = KEY_FIELDS + [
    "row_source", "audit_status", "label_source", "train_use",
    "sample_weight", "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--adaptive-summary", default="tables/adaptive_vis_1r_screening_summary.csv")
    ap.add_argument("--adaptive-provenance", default="tables/adaptive_vis_1r_provenance.csv")
    ap.add_argument("--output-csv", "--output-dataset", dest="output_csv", default="tables/detector_v25_incremental_dataset.csv")
    ap.add_argument("--output-audit", default="tables/detector_v25_incremental_label_audit.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V25_INCREMENTAL_DATASET_AUDIT.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def read_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(x).lstrip("\ufeff") for x in (reader.fieldnames or [])]
        rows = [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in reader]
    return fields, rows


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(norm(row.get(field)) for field in KEY_FIELDS)


def parse_label(row):
    for field in ("label_vulnerability_ready", "label", "full_vis_label", "label_1r"):
        value = lower(row.get(field))
        if value in {"1", "true", "yes", "positive", "silver_positive_1r", "provisional_silver_positive_1r"}:
            return "positive", "1"
        if value in {"0", "false", "no", "negative", "pending_negative_1r"}:
            return "negative", "0"
    status = lower(row.get("label_status"))
    if status in {"positive", "negative"}:
        return status, "1" if status == "positive" else "0"
    return "", ""


def is_silver_positive_1r(value):
    v = lower(value)
    return v in SILVER_CONF or v.startswith("silver_positive_1r") or v.startswith("provisional_silver_positive_1r")


def is_pending_negative_1r(value):
    v = lower(value)
    return v in PENDING_NEG or v.startswith("pending_negative_1r")


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


def inferred_pending_negative_1r(row):
    if bad_token(row):
        return False
    label_text = " ".join(lower(row.get(field)) for field in ["label_confidence", "label_1r", "label_source", "label_status", "status"])
    if "pending_negative_1r" in label_text or lower(row.get("status")) == "vis1r_pending_negative":
        return True
    if "silver_positive_1r" in label_text or "provisional_silver_positive_1r" in label_text:
        return False
    completed = (
        lower(row.get("status")) in {"vis1r_done", "completed", "done"}
        or lower(row.get("stage")) in {"vis1r_done", "completed", "done"}
        or lower(row.get("result")) in {"completed", "done"}
    )
    if not completed:
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


def bad_token(row):
    text = " ".join(lower(v) for v in row.values())
    return next((tok for tok in BAD_TOKENS if tok in text), "")


def ok_if_present(row, field):
    value = lower(row.get(field))
    return value == "" or value in {"clean", "ok"}


def audit_row(row, source, status, reason):
    out = {field: norm(row.get(field)) for field in KEY_FIELDS}
    out.update({
        "row_source": source,
        "audit_status": status,
        "label_source": norm(row.get("label_source")),
        "train_use": norm(row.get("train_use")),
        "sample_weight": norm(row.get("sample_weight")),
        "reason": reason,
    })
    return out


def output_row(row):
    out = {field: norm(row.get(field)) for field in OUT_FIELDS}
    for field, value in row.items():
        if field not in out:
            out[field] = norm(value)
    return out


def build_gold(labels):
    rows = []
    audit = []
    seen = {}
    conflicts = []
    for row in labels:
        status, label = parse_label(row)
        if status not in {"positive", "negative"}:
            audit.append(audit_row(row, "labels_v2", "excluded", "not_positive_or_negative"))
            continue
        k = key(row)
        if k in seen and seen[k] != label:
            conflicts.append(k)
            audit.append(audit_row(row, "labels_v2", "hard_fail", "duplicate_conflicting_gold_label"))
            continue
        if k in seen:
            audit.append(audit_row(row, "labels_v2", "excluded", "duplicate_same_gold_label"))
            continue
        seen[k] = label
        new = output_row(row)
        new.update({
            "label_status": status,
            "label_vulnerability_ready": label,
            "label_source": "gold_v2",
            "train_use": "true",
            "train_variant": "V0_gold,V1_gold_plus_silver",
            "sample_weight": "1.0",
            "inclusion_status": "train_gold_v2",
            "exclusion_reason": "",
        })
        rows.append(new)
        audit.append(audit_row(new, "labels_v2", "included", "gold_v2"))
    return rows, audit, conflicts


def build_1r(summary, provenance, gold_keys):
    prov = {key(r): r for r in provenance}
    rows = []
    audit = []
    for row in summary:
        joined = dict(prov.get(key(row), {}))
        joined.update({k: v for k, v in row.items() if norm(v) != ""})
        k = key(joined)
        if k in gold_keys:
            audit.append(audit_row(joined, "adaptive_1r", "excluded", "gold_v2_not_overwritten"))
            continue
        bad = bad_token(joined)
        if bad:
            new = output_row(joined)
            new.update({"train_use": "false", "sample_weight": "0.0", "inclusion_status": "excluded", "exclusion_reason": bad})
            rows.append(new)
            audit.append(audit_row(new, "adaptive_1r", "excluded", bad))
            continue
        confidence = lower(joined.get("label_confidence")) or lower(joined.get("label_1r"))
        mechanism = lower(joined.get("mechanism_status"))
        if is_silver_positive_1r(confidence):
            if mechanism not in {"", "clean", "mechanism_clean"}:
                audit.append(audit_row(joined, "adaptive_1r", "excluded", "mechanism_not_clean"))
                continue
            if not ok_if_present(joined, "denominator_status") or not ok_if_present(joined, "provenance_status"):
                audit.append(audit_row(joined, "adaptive_1r", "excluded", "denominator_or_provenance_not_clean"))
                continue
            new = output_row(joined)
            new.update({
                "label_status": "positive",
                "label_vulnerability_ready": "1",
                "label_source": "provisional_1r_positive",
                "train_use": "true",
                "train_variant": "V1_gold_plus_silver_only",
                "sample_weight": "0.5",
                "inclusion_status": "train_silver_positive_ablation",
                "exclusion_reason": "",
            })
            rows.append(new)
            audit.append(audit_row(new, "adaptive_1r", "included", "provisional_1r_positive"))
        elif is_pending_negative_1r(confidence) or is_pending_negative_1r(joined.get("label_1r")) or inferred_pending_negative_1r(joined):
            new = output_row(joined)
            new.update({
                "label_status": "ignore",
                "label_vulnerability_ready": "",
                "label_source": "pending_negative_1r",
                "train_use": "false",
                "train_variant": "ranking_only",
                "sample_weight": "0.0",
                "inclusion_status": "ranking_only",
                "exclusion_reason": "pending_negative_1r_never_train",
            })
            rows.append(new)
            audit.append(audit_row(new, "adaptive_1r", "ranking_only", "pending_negative_1r"))
        else:
            audit.append(audit_row(joined, "adaptive_1r", "excluded", "unrecognized_1r_confidence"))
    return rows, audit


def validate(rows, conflicts):
    errors = []
    if conflicts:
        errors.extend(f"duplicate_conflicting_gold_label:{'/'.join(k)}" for k in conflicts)
    for row in rows:
        text = " ".join(lower(v) for v in row.values())
        if lower(row.get("label_source")) == "pending_negative_1r" and lower(row.get("train_use")) == "true":
            errors.append("pending_negative_1r_in_train:" + "/".join(key(row)))
        if "infra_failed" in text and lower(row.get("train_use")) == "true":
            errors.append("infra_failed_in_train:" + "/".join(key(row)))
        if lower(row.get("label_source")) == "provisional_1r_positive" and norm(row.get("sample_weight")) != "0.5":
            errors.append("bad_silver_sample_weight:" + "/".join(key(row)))
    return errors


def write_report(args, rows, audit, errors, notes):
    train = [r for r in rows if lower(r.get("train_use")) == "true"]
    counts = Counter(r.get("label_source") for r in rows)
    status = "HARD_FAIL" if errors else ("BLOCKED_MISSING_INPUTS" if notes else "OK")
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    lines = [
        "# Detector V2.5 Incremental Dataset Audit",
        "",
        f"**Status**: {status}",
        f"**Rows**: {len(rows)}",
        f"**Train rows**: {len(train)}",
        f"**gold_v2 rows**: {counts.get('gold_v2', 0)}",
        f"**provisional_1r_positive rows**: {counts.get('provisional_1r_positive', 0)}",
        f"**pending_negative_1r rows**: {counts.get('pending_negative_1r', 0)}",
        "",
        "CPU-only dataset wrapper. It does not run GPU, VIS, rollout, watcher, or detector training.",
        "",
        "## Policy",
        "",
        "- `labels_v2` rows are the only confirmed training labels.",
        "- 1R positives enter only the silver-positive ablation with `sample_weight=0.5`.",
        "- 1R failures are `pending_negative_1r`, `train_use=false`, and never gold negatives.",
        "- Gold rows are never overwritten by 1R rows.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {n}" for n in notes) if notes else lines.append("- None.")
    if errors:
        lines.extend(["", "## Hard Failures", ""])
        lines.extend(f"- {e}" for e in errors)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    _, labels = read_csv(args.labels_v2)
    if not labels:
        notes.append(f"BLOCKED_MISSING_LABELS_V2: {args.labels_v2}")
    _, summary = read_csv(args.adaptive_summary)
    if not summary:
        notes.append(f"BLOCKED_MISSING_ADAPTIVE_SUMMARY: {args.adaptive_summary}")
    _, provenance = read_csv(args.adaptive_provenance)
    gold_rows, audit, conflicts = build_gold(labels)
    one_r_rows, one_r_audit = build_1r(summary, provenance, {key(r) for r in gold_rows})
    rows = gold_rows + one_r_rows
    audit.extend(one_r_audit)
    errors = validate(rows, conflicts)
    all_fields = sorted(set(OUT_FIELDS) | {k for r in rows for k in r.keys()})
    write_csv(args.output_csv, all_fields, rows)
    write_csv(args.output_audit, AUDIT_FIELDS, audit)
    write_report(args, rows, audit, errors, notes)
    if args.dry_run:
        print(f"DRY RUN: rows={len(rows)} train={sum(1 for r in rows if lower(r.get('train_use')) == 'true')} errors={len(errors)}")
        for note in notes:
            print(note)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
