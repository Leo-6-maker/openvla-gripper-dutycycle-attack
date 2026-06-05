#!/usr/bin/env python3
"""Audit whether existing labels can be connected to trigger-centered RGB paths.

CPU-only metadata/path audit. This script does not open images, extract
embeddings, run rollouts, run VIS, or train models.
"""

import argparse
import csv
import os
from collections import Counter


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v1.csv")
    ap.add_argument("--batch3-summary", default="tables/object_phase_response_batch3_vis_summary.csv")
    ap.add_argument("--trace-root", action="append", dest="trace_roots",
                    help="Trace root to audit. Can be provided multiple times.")
    ap.add_argument("--output-csv", default="tables/visual_data_availability_audit_v0.csv")
    ap.add_argument("--output-report", default="reports/VISUAL_DATA_AVAILABILITY_AUDIT_V0.md")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_rows(labels):
    return [r for r in labels if norm(r.get("label_status")) in {"positive", "negative"}]


def batch3_index(rows):
    idx = {}
    for r in rows:
        key = (norm(r.get("task") or r.get("task_key")), norm(r.get("state_id")), norm(r.get("window_start")), norm(r.get("window_end")))
        idx[key] = r
    return idx


def path_candidates(root, task, state, trigger):
    names = []
    for step in [trigger, max(trigger - 4, 0), max(trigger - 8, 0)]:
        for ext in IMAGE_EXTS:
            names.extend([
                os.path.join(root, task, "state_%s" % state, "rgb", "step_%06d%s" % (step, ext)),
                os.path.join(root, task, "s%s" % state, "rgb", "%06d%s" % (step, ext)),
                os.path.join(root, "frames", task, "state_%s" % state, "step_%06d%s" % (step, ext)),
            ])
    return names


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return ""


def find_manifest_path(roots, row):
    task = norm(row.get("task_key"))
    state = norm(row.get("state_id"))
    candidates = []
    for root in roots:
        candidates.extend([
            os.path.join(root, task, "state_%s" % state, "manifest.json"),
            os.path.join(root, task, "s%s" % state, "manifest.json"),
            os.path.join(root, task, "state_%s" % state, "run_manifest.json"),
            os.path.join(root, task, "s%s" % state, "run_manifest.json"),
            os.path.join(root, "manifests", "%s_s%s.json" % (task, state)),
        ])
    return first_existing(candidates)


def find_frame_paths(roots, row):
    task = norm(row.get("task_key"))
    state = norm(row.get("state_id"))
    trigger = int(float(norm(row.get("window_start")) or 0))
    result = {}
    for label, step in [("trigger", trigger), ("minus4", max(trigger - 4, 0)), ("minus8", max(trigger - 8, 0))]:
        candidates = []
        for root in roots:
            for ext in IMAGE_EXTS:
                candidates.extend([
                    os.path.join(root, task, "state_%s" % state, "rgb", "step_%06d%s" % (step, ext)),
                    os.path.join(root, task, "s%s" % state, "rgb", "%06d%s" % (step, ext)),
                    os.path.join(root, "frames", task, "state_%s" % state, "step_%06d%s" % (step, ext)),
                ])
        result[label] = first_existing(candidates)
    return result


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "sample_id", "source_batch", "task_key", "state_id", "window_start", "window_end",
        "trigger_step", "batch3_summary_found", "localized_trace_found", "manifest_path",
        "trigger_rgb_found", "past_rgb_found", "missing_visual_path",
        "image_trigger_path", "image_trigger_minus4_path", "image_trigger_minus8_path",
        "missing_visual_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_report(path, rows, roots):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    total = len(rows)
    trigger = sum(1 for r in rows if r["trigger_rgb_found"] == "true")
    past = sum(1 for r in rows if r["past_rgb_found"] == "true")
    missing = sum(1 for r in rows if r["missing_visual_reason"])
    by_source = Counter(r["source_batch"] for r in rows)
    by_task = Counter(r["task_key"] for r in rows)
    verdict = "READY_FOR_PATH_LEVEL_PIPELINE" if trigger else "NOT_READY_VISUAL_PATHS_MISSING"
    lines = [
        "# Visual Data Availability Audit V0",
        "",
        "CPU-only path audit. No images were read and no embeddings were extracted.",
        "",
        f"**Total train rows**: {total}",
        f"**Rows with trigger RGB**: {trigger}",
        f"**Rows with past RGB**: {past}",
        f"**Missing path count**: {missing}",
        f"**Visual readiness verdict**: `{verdict}`",
        "",
        "## Trace Roots",
        "",
    ]
    for root in roots:
        lines.append(f"- `{root}`")
    lines += ["", "## By Source", "", "| Source | Count |", "|---|---:|"]
    for k in sorted(by_source):
        lines.append(f"| {k} | {by_source[k]} |")
    lines += ["", "## By Task", "", "| Task | Count |", "|---|---:|"]
    for k in sorted(by_task):
        lines.append(f"| {k} | {by_task[k]} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- Missing visual paths do not invalidate labels; they only block visual-feature extraction.",
        "- This audit does not support any visual detector claim.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    if not args.trace_roots:
        args.trace_roots = [
            "/data/liuyu/outputs/nightly_object_batch3_20260604",
            "/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604",
            "/data/liuyu/outputs/nightly_object_batch3b_20260604",
        ]
    labels = train_rows(read_csv(args.labels_csv))
    b3 = batch3_index(read_csv(args.batch3_summary))
    rows = []
    for r in labels:
        task = norm(r.get("task_key"))
        state = norm(r.get("state_id"))
        ws = norm(r.get("window_start"))
        we = norm(r.get("window_end"))
        sample_id = "%s_%s_s%s_w%s_%s" % (norm(r.get("source_batch")), task, state, ws, we)
        paths = find_frame_paths(args.trace_roots, r)
        manifest = find_manifest_path(args.trace_roots, r)
        past_ok = bool(paths["minus4"] and paths["minus8"])
        missing = []
        if not paths["trigger"]:
            missing.append("missing_trigger_rgb")
        if not past_ok:
            missing.append("missing_past_rgb")
        rows.append({
            "sample_id": sample_id,
            "source_batch": norm(r.get("source_batch")),
            "task_key": task,
            "state_id": state,
            "window_start": ws,
            "window_end": we,
            "trigger_step": ws,
            "batch3_summary_found": str((task, state, ws, we) in b3).lower(),
            "localized_trace_found": str(bool(manifest)).lower(),
            "manifest_path": manifest,
            "trigger_rgb_found": str(bool(paths["trigger"])).lower(),
            "past_rgb_found": str(past_ok).lower(),
            "missing_visual_path": str(bool(missing)).lower(),
            "image_trigger_path": paths["trigger"],
            "image_trigger_minus4_path": paths["minus4"],
            "image_trigger_minus8_path": paths["minus8"],
            "missing_visual_reason": "|".join(missing),
        })
    write_csv(args.output_csv, rows)
    write_report(args.output_report, rows, args.trace_roots)
    print("Wrote %d rows -> %s" % (len(rows), args.output_csv))
    print("Report: %s" % args.output_report)


if __name__ == "__main__":
    main()
