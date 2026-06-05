#!/usr/bin/env python3
"""Audit visual transfer dataset for leakage and online-frame rules."""

import argparse
import csv
import os
import re


FORBIDDEN_PATTERNS = [
    r"^claim_usable$",
    r"^done$",
    r"^vis_open$",
    r"^vis_open_count$",
    r"^qpos_opening_delta$",
    r"^qpos_delta_after_window$",
    r"^denominator_clean$",
    r"^task_failure_positive$",
    r"^physical_bridge_positive$",
    r"random.*outcome",
    r"oracle.*outcome",
    r"manual.*audit.*outcome",
    r"attack.*outcome",
    r"attack.*result",
]
INPUT_PREFIXES = (
    "gripper_", "eef_", "open_streak_pre", "close_streak_pre", "phase_gate_score",
    "hazard_score", "image_", "global_embedding_path", "crop_embedding_path",
    "visual_encoder_name", "visual_feature_dim", "task_key", "candidate_role",
    "phase_bin_proxy", "window_start", "window_end", "trigger_step",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-csv", default="tables/visual_transfer_dataset_v0.csv")
    ap.add_argument("--output-csv", default="tables/visual_transfer_leakage_audit_v0.csv")
    ap.add_argument("--output-report", default="reports/VISUAL_TRANSFER_LEAKAGE_AUDIT_V0.md")
    return ap.parse_args()


def forbidden(field):
    low = field.strip().lower()
    return any(re.search(p, low) for p in FORBIDDEN_PATTERNS)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def audit(rows, fields):
    checks = []
    input_fields = [f for f in fields if f.startswith(INPUT_PREFIXES)]
    bad_inputs = [f for f in input_fields if forbidden(f)]
    checks.append(("forbidden_input_columns", "fail" if bad_inputs else "pass", ",".join(bad_inputs)))
    future_cols = [f for f in fields if re.search(r"trigger_plus|future|plus[0-9]", f.lower())]
    checks.append(("future_frame_columns", "fail" if future_cols else "pass", ",".join(future_cols)))
    label_inputs = [f for f in input_fields if f.startswith("label_")]
    checks.append(("label_input_separation", "fail" if label_inputs else "pass", ",".join(label_inputs)))
    bad_paths = []
    for i, r in enumerate(rows, start=2):
        for f in fields:
            if f.startswith("image_trigger_plus") and r.get(f):
                bad_paths.append(str(i))
    checks.append(("future_frame_values", "fail" if bad_paths else "pass", ",".join(bad_paths[:20])))
    return checks


def write_outputs(args, checks):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check_id", "status", "detail"])
        w.writeheader()
        for check_id, status, detail in checks:
            w.writerow({"check_id": check_id, "status": status, "detail": detail})
    failures = [c for c in checks if c[1] == "fail"]
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    lines = [
        "# Visual Transfer Leakage Audit V0",
        "",
        f"**Verdict**: {'FAIL' if failures else 'PASS'}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c[0]} | {c[1]} | {c[2]} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- Outcome fields may exist only as labels/audit metadata, not model inputs.",
        "- Only trigger and trigger-minus frames are allowed for online-mode visual inputs.",
        "",
    ]
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return 1 if failures else 0


def main():
    args = parse_args()
    rows, fields = read_rows(args.dataset_csv)
    checks = audit(rows, fields)
    rc = write_outputs(args, checks)
    print("Audit: %s" % args.output_csv)
    print("Report: %s" % args.output_report)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
