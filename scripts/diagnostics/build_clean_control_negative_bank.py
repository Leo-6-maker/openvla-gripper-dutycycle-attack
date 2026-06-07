#!/usr/bin/env python3
"""Build clean-control negative bank for detector-v2.6 control ablations."""

from __future__ import annotations

import argparse
import csv
import os


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
FIELDS = KEY_FIELDS + [
    "label_status", "label_vulnerability_ready", "label_source",
    "control_type", "control_reason", "sample_weight", "phase_label",
    "phase_detector_output", "qpos_phase_class", "train_use",
    "candidate_id", "expected_role", "phase_bin_proxy",
]
ALLOWED_TYPES = {
    "far_too_early", "stable_post_lock", "natural_open", "no_contact",
    "far_from_object", "after_done", "terminal", "post_lock",
    "post_lock_stable",
}
BAD_TOKENS = ("infra", "manual_review", "polluted", "ambiguous", "xid", "oom", "localization_fail")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--adaptive-candidates", default="tables/object_phase_response_adaptive_candidates.csv")
    ap.add_argument("--phase-outputs", default="")
    ap.add_argument("--output-csv", default="tables/clean_control_negative_bank.csv")
    ap.add_argument("--output-report", default="reports/CLEAN_CONTROL_NEGATIVE_BANK.md")
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


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def key(row):
    return tuple(norm(row.get(field)) for field in KEY_FIELDS)


def bad(row):
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in BAD_TOKENS)


def explicit_control_type(row):
    fields = [
        row.get("control_type"), row.get("candidate_role"), row.get("expected_role"),
        row.get("phase_bin_proxy"), row.get("qpos_phase_class"),
        row.get("source_reason"), row.get("reason_selected"), row.get("control_reason"),
    ]
    text = " ".join(lower(v) for v in fields)
    for control in ALLOWED_TYPES:
        if control in text:
            return control
    if "post_lock" in text and "stable" in text:
        return "post_lock_stable"
    if "far" in text and ("object" in text or "contact" in text):
        return "far_from_object"
    return ""


def write_report(path, args, rows, issues, source_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    status = "HARD_FAIL" if issues else ("OK" if rows else "BLOCKED_MISSING_CLEAN_CONTROL_SOURCE")
    lines = [
        "# Clean-Control Negative Bank",
        "",
        f"**Status**: {status}",
        f"**Source rows scanned**: {source_rows}",
        f"**Control negatives**: {len(rows)}",
        "",
        "Rows are CPU-only diagnostic controls. No GPU, VIS, rollout, watcher, or live output was touched.",
        "",
        "## Policy",
        "",
        "- Clean controls require an explicit non-vulnerable reason.",
        "- Gold positive key overlap is a hard failure.",
        "- Infra/manual/polluted/ambiguous rows are excluded.",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {i}" for i in issues) if issues else lines.append("- None.")
    lines.extend(["", "## Rows", ""])
    if rows:
        lines.extend(["| Task | State | Window | Control type | Reason |",
                      "|---|---:|---|---|---|"])
        for row in rows[:50]:
            lines.append(f"| {row['task_key']} | {row['state_id']} | {row['window_start']}-{row['window_end']} | {row['control_type']} | {row['control_reason']} |")
    else:
        lines.append("- No explicit clean-control negatives available in the scanned sources.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    labels = read_csv(args.labels_v2)
    gold_pos = {key(r) for r in labels if lower(r.get("label_status")) == "positive" or norm(r.get("label_vulnerability_ready")) == "1"}
    candidates = read_csv(args.adaptive_candidates)
    phase = {key(r): r for r in read_csv(args.phase_outputs)}
    rows = []
    issues = []
    seen = set()
    for src in candidates:
        if bad(src):
            continue
        ctype = explicit_control_type(src)
        if not ctype:
            continue
        if key(src) in gold_pos:
            issues.append("clean_control_overlaps_gold_positive:" + "/".join(key(src)))
            continue
        if key(src) in seen:
            continue
        ph = phase.get(key(src), {})
        row = {
            "task_key": norm(src.get("task_key")),
            "state_id": norm(src.get("state_id")),
            "window_start": norm(src.get("window_start")),
            "window_end": norm(src.get("window_end")),
            "label_status": "negative",
            "label_vulnerability_ready": "0",
            "label_source": "clean_control_negative",
            "control_type": ctype,
            "control_reason": norm(src.get("source_reason") or src.get("reason_selected") or src.get("candidate_role") or src.get("phase_bin_proxy") or src.get("qpos_phase_class")),
            "sample_weight": "0.5",
            "phase_label": norm(ph.get("phase_label") or src.get("phase_label")),
            "phase_detector_output": norm(ph.get("predicted_phase") or ph.get("phase_detector_output") or src.get("predicted_phase")),
            "qpos_phase_class": norm(src.get("qpos_phase_class") or ph.get("qpos_phase_class")),
            "train_use": "true_for_control_ablation",
            "candidate_id": norm(src.get("candidate_id")),
            "expected_role": norm(src.get("expected_role")),
            "phase_bin_proxy": norm(src.get("phase_bin_proxy")),
        }
        rows.append(row)
        seen.add(key(src))
    write_csv(args.output_csv, rows)
    write_report(args.output_report, args, rows, issues, len(candidates))
    print(f"clean_control_rows={len(rows)} issues={len(issues)}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
