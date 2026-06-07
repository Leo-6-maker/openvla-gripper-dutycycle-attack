#!/usr/bin/env python3
"""Build detector-v2.6 two-stage dataset."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
FIELDS = KEY_FIELDS + [
    "label_status", "label_vulnerability_ready", "label_source",
    "sample_weight", "train_use", "train_variant", "source_batch",
    "candidate_role", "expected_role", "phase_bin_proxy", "control_type",
    "phase_label", "phase_detector_output", "phase_confidence",
    "phase_is_critical", "qpos_phase_class", "inclusion_status",
    "exclusion_reason",
]
BAD_TOKENS = ("infra", "manual_review", "polluted", "dubious_positive", "xid", "oom", "localization_fail")
CRITICAL_PHASES = ("true_closed", "transitional_pre_open", "near_closed", "pre_lock", "contact", "grasp", "lift")
NONCRITICAL_PHASES = ("far_too_early", "natural_open", "stable_post_lock", "after_done", "no_contact")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--adaptive-summary", default="tables/adaptive_vis_1r_screening_summary.csv")
    ap.add_argument("--adaptive-provenance", default="tables/adaptive_vis_1r_provenance.csv")
    ap.add_argument("--quality-audit", default="")
    ap.add_argument("--clean-control-bank", default="tables/clean_control_negative_bank.csv")
    ap.add_argument("--phase-outputs", default="")
    ap.add_argument("--output-dataset", default="tables/detector_v26_twostage_dataset.csv")
    ap.add_argument("--output-audit", default="tables/detector_v26_twostage_label_audit.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V26_TWOSTAGE_DATASET_AUDIT.md")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def key(row):
    return tuple(norm(row.get(f)) for f in KEY_FIELDS)


def bad(row):
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in BAD_TOKENS)


def parse_label(row):
    for field in ["label_vulnerability_ready", "label", "full_vis_label", "label_1r"]:
        v = lower(row.get(field))
        if v in {"1", "true", "positive", "provisional_silver_positive_1r", "silver_positive_1r"} or v.startswith("provisional_silver_positive_1r"):
            return "positive", "1"
        if v in {"0", "false", "negative"}:
            return "negative", "0"
    s = lower(row.get("label_status"))
    if s in {"positive", "negative"}:
        return s, "1" if s == "positive" else "0"
    return "", ""


def phase_is_critical(row):
    text = " ".join(lower(row.get(f)) for f in ["phase_bin_proxy", "phase_label", "phase_detector_output", "qpos_phase_class", "control_type"])
    if any(tok in text for tok in NONCRITICAL_PHASES):
        return "0"
    if any(tok in text for tok in CRITICAL_PHASES):
        return "1"
    return ""


def base_row(row):
    out = {field: norm(row.get(field)) for field in FIELDS}
    for k, v in row.items():
        if k not in out:
            out[k] = norm(v)
    out["phase_is_critical"] = out.get("phase_is_critical") or phase_is_critical(out)
    return out


def quality_ok(row, qmap):
    q = qmap.get(key(row), {})
    text = " ".join(lower(v) for v in {**row, **q}.values())
    if "dubious" in text or "fail" in text or bad({**row, **q}):
        return False
    return True


def write_report(path, rows, audit, conflicts, notes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = Counter(r.get("label_source") for r in rows)
    train = [r for r in rows if lower(r.get("train_use")).startswith("true")]
    lines = [
        "# Detector V2.6 Two-Stage Dataset Audit",
        "",
        f"**Status**: {'HARD_FAIL' if conflicts else 'OK'}",
        f"**Rows**: {len(rows)}",
        f"**Train rows**: {len(train)}",
        f"**gold_v2 rows**: {counts.get('gold_v2', 0)}",
        f"**eligible_silver_positive_1r rows**: {counts.get('eligible_silver_positive_1r', 0)}",
        f"**clean_control_negative rows**: {counts.get('clean_control_negative', 0)}",
        f"**Conflicts**: {len(conflicts)}",
        "",
        "No pending_negative_1r, infra, manual, polluted, or dubious-positive rows are train labels.",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {n}" for n in notes) if notes else lines.append("- None.")
    if conflicts:
        lines.extend(["", "## Conflicts", ""])
        lines.extend(f"- {c}" for c in conflicts)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    labels = read_csv(args.labels_v2)
    adaptive = read_csv(args.adaptive_summary)
    provenance = {key(r): r for r in read_csv(args.adaptive_provenance)}
    qmap = {key(r): r for r in read_csv(args.quality_audit)}
    controls = read_csv(args.clean_control_bank)
    phase = {key(r): r for r in read_csv(args.phase_outputs)}
    rows, audit, conflicts = [], [], []
    seen = {}
    for src in labels:
        status, label = parse_label(src)
        if status not in {"positive", "negative"}:
            continue
        row = base_row({**src, **phase.get(key(src), {})})
        row.update(label_status=status, label_vulnerability_ready=label, label_source="gold_v2",
                   sample_weight="1.0", train_use="true", train_variant="all",
                   inclusion_status="train_gold_v2", exclusion_reason="")
        if key(row) in seen and seen[key(row)] != label:
            conflicts.append("duplicate_conflicting_gold:" + "/".join(key(row)))
            continue
        seen[key(row)] = label
        rows.append(row)
        audit.append({**{f: row.get(f, "") for f in KEY_FIELDS}, "status": "included", "reason": "gold_v2"})
    gold_keys = set(seen)
    for src0 in adaptive:
        src = dict(provenance.get(key(src0), {}))
        src.update({k: v for k, v in src0.items() if norm(v) != ""})
        if key(src) in gold_keys or bad(src) or not quality_ok(src, qmap):
            continue
        status, label = parse_label(src)
        if label != "1":
            continue
        row = base_row({**src, **phase.get(key(src), {})})
        row.update(label_status="positive", label_vulnerability_ready="1",
                   label_source="eligible_silver_positive_1r", sample_weight="0.5",
                   train_use="true", train_variant="silver_ablation",
                   inclusion_status="train_silver_positive_ablation", exclusion_reason="")
        rows.append(row)
        audit.append({**{f: row.get(f, "") for f in KEY_FIELDS}, "status": "included", "reason": "eligible_silver_positive_1r"})
    for src in controls:
        if key(src) in gold_keys or bad(src):
            if key(src) in gold_keys:
                conflicts.append("clean_control_overlaps_gold:" + "/".join(key(src)))
            continue
        row = base_row({**src, **phase.get(key(src), {})})
        row.update(label_status="negative", label_vulnerability_ready="0",
                   label_source="clean_control_negative", sample_weight=norm(src.get("sample_weight")) or "0.5",
                   train_use="true", train_variant="clean_control_ablation",
                   inclusion_status="train_clean_control_negative", exclusion_reason="")
        rows.append(row)
        audit.append({**{f: row.get(f, "") for f in KEY_FIELDS}, "status": "included", "reason": "clean_control_negative"})
    all_fields = sorted(set(FIELDS) | {k for r in rows for k in r.keys()})
    write_csv(args.output_dataset, all_fields, rows)
    write_csv(args.output_audit, KEY_FIELDS + ["status", "reason"], audit)
    if not controls:
        notes.append("clean_control_negative_bank empty or missing")
    if not qmap:
        notes.append("silver quality audit missing; used conservative row text exclusions only")
    if not phase:
        notes.append("phase detector outputs missing; phase gate variants may be blocked")
    write_report(args.output_report, rows, audit, conflicts, notes)
    print(f"dataset_rows={len(rows)} conflicts={len(conflicts)}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
