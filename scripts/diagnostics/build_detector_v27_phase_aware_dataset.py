#!/usr/bin/env python3
"""Build detector-v2.7 phase-aware exploratory dataset."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
BASE_FIELDS = KEY_FIELDS + [
    "label_status",
    "label_vulnerability_ready",
    "label_source",
    "sample_weight",
    "train_use",
    "train_variant",
    "phase_bin_proxy",
    "qpos_phase_class",
    "control_type",
    "phase_is_critical",
    "predicted_phase",
    "phase_confidence",
    "phase_source",
    "phase_missing_reason",
    "inclusion_status",
    "exclusion_reason",
]
BAD_TOKENS = ("infra", "manual", "polluted", "dubious", "xid", "oom", "localization_fail")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--adaptive-summary", default="tables/adaptive_vis_1r_screening_summary.csv")
    ap.add_argument("--adaptive-provenance", default="tables/object_phase_response_adaptive_candidates.csv")
    ap.add_argument("--quality-audit", default="")
    ap.add_argument("--clean-control-bank-v2", default="tables/clean_control_negative_bank_v2.csv")
    ap.add_argument("--phase-features", default="tables/detector_phase_features_v1.csv")
    ap.add_argument("--silver-weight", default="0.5")
    ap.add_argument("--output-dataset", default="tables/detector_v27_phase_aware_dataset.csv")
    ap.add_argument("--output-audit", default="tables/detector_v27_phase_aware_label_audit.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V27_PHASE_AWARE_DATASET_AUDIT.md")
    return ap.parse_args()


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def lower(value) -> str:
    return norm(value).lower()


def read_csv(path: str) -> list[dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): norm(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(norm(row.get(f)) for f in KEY_FIELDS)


def bad(row: dict[str, str]) -> bool:
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in BAD_TOKENS)


def parse_label(row: dict[str, str]) -> tuple[str, str]:
    status = lower(row.get("label_status"))
    if status in {"positive", "negative"}:
        return status, "1" if status == "positive" else "0"
    vuln = lower(row.get("label_vulnerability_ready"))
    if vuln in {"1", "true", "positive"}:
        return "positive", "1"
    if vuln in {"0", "false", "negative"}:
        return "negative", "0"
    label_1r = lower(row.get("label_1r") or row.get("label_confidence"))
    if label_1r.startswith("provisional_silver_positive_1r") or label_1r.startswith("silver_positive_1r"):
        return "positive", "1"
    return "", ""


def merge_phase(row: dict[str, str], phase_map: dict[tuple[str, str, str, str], dict[str, str]]) -> dict[str, str]:
    out = dict(row)
    phase = phase_map.get(key(row), {})
    for field in ["phase_bin_proxy", "qpos_phase_class", "control_type", "phase_is_critical", "predicted_phase", "phase_confidence", "phase_source", "phase_missing_reason"]:
        if not norm(out.get(field)) and norm(phase.get(field)):
            out[field] = norm(phase.get(field))
    if not norm(out.get("phase_is_critical")):
        out["phase_is_critical"] = "missing"
        out["phase_missing_reason"] = norm(out.get("phase_missing_reason")) or "no_phase_feature_row"
    return out


def quality_ok(row: dict[str, str], quality_map: dict[tuple[str, str, str, str], dict[str, str]]) -> bool:
    q = quality_map.get(key(row), {})
    text = " ".join(lower(v) for v in {**row, **q}.values())
    if "dubious" in text or "pending_negative_1r" in text or bad({**row, **q}):
        return False
    return True


def audit_row(row: dict[str, str], status: str, reason: str) -> dict[str, str]:
    return {**{f: norm(row.get(f)) for f in KEY_FIELDS}, "status": status, "reason": reason}


def main() -> int:
    args = parse_args()
    phase_map = {key(r): r for r in read_csv(args.phase_features)}
    quality_map = {key(r): r for r in read_csv(args.quality_audit)}
    rows: list[dict[str, str]] = []
    audit: list[dict[str, str]] = []
    conflicts: list[str] = []
    gold_labels: dict[tuple[str, str, str, str], str] = {}

    for src in read_csv(args.labels_v2):
        status, label = parse_label(src)
        if status not in {"positive", "negative"}:
            audit.append(audit_row(src, "excluded", "labels_v2_not_train_status"))
            continue
        row = merge_phase(src, phase_map)
        row.update({
            "label_status": status,
            "label_vulnerability_ready": label,
            "label_source": "gold_v2",
            "sample_weight": "1.0",
            "train_use": "true",
            "train_variant": "gold",
            "inclusion_status": "train_gold_v2",
            "exclusion_reason": "",
        })
        k = key(row)
        if k in gold_labels and gold_labels[k] != label:
            conflicts.append("duplicate_conflicting_gold:" + "/".join(k))
            audit.append(audit_row(row, "conflict", "duplicate_conflicting_gold"))
            continue
        gold_labels[k] = label
        rows.append(row)
        audit.append(audit_row(row, "included", "gold_v2"))

    adaptive_map = {key(r): r for r in read_csv(args.adaptive_provenance)}
    for src0 in read_csv(args.adaptive_summary):
        src = dict(adaptive_map.get(key(src0), {}))
        src.update({k: v for k, v in src0.items() if norm(v)})
        if key(src) in gold_labels:
            audit.append(audit_row(src, "excluded", "gold_not_overwritten"))
            continue
        if bad(src) or not quality_ok(src, quality_map):
            audit.append(audit_row(src, "excluded", "bad_or_quality_failed"))
            continue
        status, label = parse_label(src)
        if label != "1":
            audit.append(audit_row(src, "excluded", "not_eligible_silver_positive"))
            continue
        row = merge_phase(src, phase_map)
        row.update({
            "label_status": "positive",
            "label_vulnerability_ready": "1",
            "label_source": "eligible_1r_silver_positive",
            "sample_weight": args.silver_weight,
            "train_use": "true",
            "train_variant": "silver_positive_ablation",
            "inclusion_status": "train_eligible_1r_silver_positive",
            "exclusion_reason": "",
        })
        rows.append(row)
        audit.append(audit_row(row, "included", "eligible_1r_silver_positive"))

    for src in read_csv(args.clean_control_bank_v2):
        if key(src) in gold_labels:
            conflicts.append("clean_control_overlaps_gold:" + "/".join(key(src)))
            audit.append(audit_row(src, "conflict", "clean_control_overlaps_gold"))
            continue
        if bad(src):
            audit.append(audit_row(src, "excluded", "bad_clean_control"))
            continue
        row = merge_phase(src, phase_map)
        row.update({
            "label_status": "negative",
            "label_vulnerability_ready": "0",
            "label_source": "clean_control_negative",
            "sample_weight": norm(src.get("sample_weight")) or "0.5",
            "train_use": "true",
            "train_variant": "clean_control_ablation",
            "inclusion_status": "train_clean_control_negative",
            "exclusion_reason": "",
        })
        rows.append(row)
        audit.append(audit_row(row, "included", "clean_control_negative"))

    all_fields = sorted(set(BASE_FIELDS) | {k for r in rows for k in r.keys()})
    write_csv(args.output_dataset, all_fields, rows)
    write_csv(args.output_audit, KEY_FIELDS + ["status", "reason"], audit)
    counts = Counter(r.get("label_source") for r in rows)
    phase_counts = Counter(r.get("phase_is_critical") or "missing" for r in rows)
    lines = [
        "# Detector V2.7 Phase-Aware Dataset Audit",
        "",
        f"**Status**: {'HARD_FAIL' if conflicts else 'OK'}",
        f"**Rows**: {len(rows)}",
        f"**Train rows**: {sum(1 for r in rows if lower(r.get('train_use')).startswith('true'))}",
        f"**Conflicts**: {len(conflicts)}",
        "",
        "## Label Sources",
        "",
    ]
    for source, count in counts.most_common():
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Phase Coverage", ""])
    for value, count in phase_counts.most_common():
        lines.append(f"- `{value}`: {count}")
    lines.extend(["", "## Policy", ""])
    lines.append("- Gold labels are never overwritten.")
    lines.append("- pending_negative_1r, infra/manual/polluted/dubious rows are excluded from train.")
    lines.append("- Clean controls are ablation negatives, not gold labels.")
    if conflicts:
        lines.extend(["", "## Conflicts", ""])
        lines.extend(f"- {c}" for c in conflicts)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"dataset_v27_rows={len(rows)} conflicts={len(conflicts)}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
