#!/usr/bin/env python3
"""Generate Batch4 hard-negative candidates from detector-v2 errors.

CPU-only. The generator prioritizes detector false positives, same-task
contrasts, MuJoCo-qpos verified true-closed windows, and explicit controls.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


OUTPUT_FIELDS = [
    "target_id",
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "source_batch",
    "phase_bin_proxy",
    "candidate_role",
    "expected_role",
    "source_error_type",
    "full_vis_label",
    "label_status",
    "denominator_plan",
    "qpos_verification_status",
    "gripper_qpos_used",
    "gripper_qpos_source_priority",
    "true_closed",
    "natural_open",
    "phase_proxy_mismatch",
    "same_task_contrast_group",
    "priority_rank",
    "label_source",
    "reason_selected",
]

PROXY_BLOCK_TOKENS = ["phase_d", "phase e", "phase_e", "command_proxy", "low_budget", "proxy_label", "silver_proxy"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-csv", default="tables/detector_v2_error_analysis.csv")
    ap.add_argument("--qpos-phase-audit-csv", default="tables/labels_v2_mujoco_qpos_phase_audit.csv")
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--output-csv", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--output-report", default="reports/BATCH4_FP_DRIVEN_QPOS_VERIFIED_CANDIDATES.md")
    ap.add_argument("--max-contrasts-per-fp", type=int, default=1)
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


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


def truthy(value):
    return lower(value) in {"1", "true", "yes", "positive"}


def is_proxy_label(row):
    text = " ".join(lower(row.get(field)) for field in row.keys())
    return any(token in text for token in PROXY_BLOCK_TOKENS)


def window_text(row):
    return f"{norm(row.get('window_start'))}-{norm(row.get('window_end'))}"


def build_lookup(rows):
    return {key(row): row for row in rows}


def merge_row(base, *lookups):
    out = dict(base)
    row_key = key(base)
    for lookup in lookups:
        extra = lookup.get(row_key, {})
        for k, v in extra.items():
            if norm(out.get(k)) == "" and norm(v) != "":
                out[k] = v
    return out


def qpos_status(row):
    if lower(row.get("qpos_phase_status")) == "qpos_verified_true_closed" or truthy(row.get("true_closed")):
        return "qpos_verified_true_closed"
    if lower(row.get("qpos_phase_status")) == "natural_open" or truthy(row.get("natural_open")):
        return "natural_open_not_hard_negative"
    if lower(row.get("qpos_phase_status")) == "phase_proxy_mismatch" or truthy(row.get("phase_proxy_mismatch")):
        return "phase_proxy_mismatch_review"
    if lower(row.get("qpos_phase_status")) == "missing_qpos":
        return "missing_qpos"
    return lower(row.get("qpos_phase_status")) or "qpos_unverified"


def candidate_from(row, target_id, expected_role, source_error_type, priority, reason):
    label_value = norm(row.get("label_vulnerability_ready"))
    if label_value == "":
        label_value = norm(row.get("full_vis_label"))
    return {
        "target_id": target_id,
        "task_key": norm(row.get("task_key")),
        "state_id": norm(row.get("state_id")),
        "window_start": norm(row.get("window_start")),
        "window_end": norm(row.get("window_end")),
        "source_batch": norm(row.get("source_batch")),
        "phase_bin_proxy": norm(row.get("phase_bin_proxy")),
        "candidate_role": norm(row.get("candidate_role")),
        "expected_role": expected_role,
        "source_error_type": source_error_type,
        "full_vis_label": label_value,
        "label_status": norm(row.get("label_status")),
        "denominator_plan": "gold_VIS_matched_random_clean_required",
        "qpos_verification_status": qpos_status(row),
        "gripper_qpos_used": norm(row.get("gripper_qpos_used")),
        "gripper_qpos_source_priority": norm(row.get("gripper_qpos_source_priority")),
        "true_closed": norm(row.get("true_closed")),
        "natural_open": norm(row.get("natural_open")),
        "phase_proxy_mismatch": norm(row.get("phase_proxy_mismatch")),
        "same_task_contrast_group": norm(row.get("task_key")),
        "priority_rank": str(priority),
        "label_source": "detector_v2_error_and_labels_v2",
        "reason_selected": reason,
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, status, rows, notes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hard_neg = [r for r in rows if "hard_negative" in lower(r.get("expected_role"))]
    controls = [r for r in rows if "control" in lower(r.get("expected_role"))]
    lines = [
        "# Batch4 FP-Driven Qpos-Verified Candidates",
        "",
        f"**Status**: {status}",
        f"**Error CSV**: `{args.error_csv}`",
        f"**Qpos/phase audit CSV**: `{args.qpos_phase_audit_csv}`",
        f"**Labels CSV**: `{args.labels_csv}`",
        f"**Candidates**: {len(rows)}",
        f"**Hard-negative/control rows**: {len(hard_neg)}",
        f"**Controls**: {len(controls)}",
        "",
        "This generator is CPU-only. It does not run rollout, VIS, GPU work, watcher jobs, or detector training.",
        "",
        "## Notes",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.extend(["", "## Candidate Summary", ""])
    if rows:
        lines.extend(
            [
                "| Rank | Target | Task | State | Window | Expected role | Qpos status | Reason |",
                "|---:|---|---|---:|---|---|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['priority_rank']} | {row['target_id']} | {row['task_key']} | {row['state_id']} | "
                f"{window_text(row)} | {row['expected_role']} | {row['qpos_verification_status']} | {row['reason_selected']} |"
            )
    else:
        lines.append("- No candidates generated.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- These rows are scheduling candidates only.",
            "- Phase D/E proxy labels are excluded and must not be treated as gold labels.",
            "- Candidate readiness still depends on schema audit and DeepSeek server-side generation of full labels_v2 artifacts.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    missing = [path for path in [args.error_csv, args.qpos_phase_audit_csv, args.labels_csv] if not os.path.exists(path)]
    if missing:
        write_csv(args.output_csv, [])
        write_report(args.output_report, args, "BLOCKED_MISSING_INPUTS", [], [f"missing input: {path}" for path in missing])
        return 0

    _, errors = read_csv(args.error_csv)
    _, qpos_rows = read_csv(args.qpos_phase_audit_csv)
    _, label_rows = read_csv(args.labels_csv)
    qpos_by_key = build_lookup(qpos_rows)
    label_by_key = build_lookup(label_rows)

    labels_by_task = defaultdict(list)
    for row in label_rows:
        merged = merge_row(row, qpos_by_key)
        if not is_proxy_label(merged):
            labels_by_task[norm(merged.get("task_key"))].append(merged)

    candidates = []
    seen = set()

    def add(row, expected_role, source_error_type, priority, reason):
        merged = merge_row(row, qpos_by_key, label_by_key)
        if is_proxy_label(merged):
            notes.append(f"excluded Phase D/E proxy label candidate: {key(merged)}")
            return
        if key(merged) in seen:
            return
        target_id = f"batch4_{len(candidates) + 1:03d}"
        candidates.append(candidate_from(merged, target_id, expected_role, source_error_type, priority, reason))
        seen.add(key(merged))

    false_positives = [row for row in errors if lower(row.get("error_type")) == "fp"]
    false_negatives = [row for row in errors if lower(row.get("error_type")) == "fn"]

    for row in false_positives:
        add(row, "hard_negative_fp_control", "FP", 1, "detector_v2_false_positive_priority")
        contrasts = [
            candidate
            for candidate in labels_by_task.get(norm(row.get("task_key")), [])
            if truthy(candidate.get("label_vulnerability_ready")) and key(candidate) != key(row)
        ]
        contrasts.sort(key=lambda r: (norm(r.get("state_id")), norm(r.get("window_start"))))
        for contrast in contrasts[: args.max_contrasts_per_fp]:
            add(contrast, "same_task_positive_contrast", "contrast", 3, "same_task_contrast_for_fp")

    for row in false_negatives:
        add(row, "missed_positive_followup", "FN", 2, "detector_v2_false_negative_followup")

    qpos_closed_negatives = []
    for task_rows in labels_by_task.values():
        for row in task_rows:
            if truthy(row.get("label_vulnerability_ready")):
                continue
            if lower(row.get("qpos_phase_status")) != "qpos_verified_true_closed":
                continue
            if is_proxy_label(row):
                continue
            qpos_closed_negatives.append(row)
    qpos_closed_negatives.sort(key=lambda r: (norm(r.get("task_key")), norm(r.get("state_id")), norm(r.get("window_start"))))
    for row in qpos_closed_negatives:
        add(row, "qpos_verified_hard_negative_control", "qpos_verified_negative", 4, "mujoco_qpos_verified_true_closed_negative")

    candidates.sort(key=lambda r: (int(r["priority_rank"]), r["task_key"], int(float(r["state_id"] or 0)), int(float(r["window_start"] or 0))))
    for i, row in enumerate(candidates, start=1):
        row["priority_rank"] = str(i)
        row["target_id"] = f"batch4_{i:03d}"

    if not false_positives:
        notes.append("no detector-v2 false positives available; hard-negative planning is blocked or incomplete")
    if len(false_positives) != 6:
        notes.append(f"expected 6 detector-v2 FPs, observed {len(false_positives)}")
    if len(false_negatives) != 1:
        notes.append(f"expected 1 detector-v2 FN, observed {len(false_negatives)}")

    status = "READY_FOR_SCHEMA_AUDIT" if candidates and len(false_positives) == 6 else "REVIEW_NEEDED"
    write_csv(args.output_csv, candidates)
    write_report(args.output_report, args, status, candidates, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
