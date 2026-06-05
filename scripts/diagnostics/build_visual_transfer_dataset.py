#!/usr/bin/env python3
"""Build a trigger-centered VisualTransferHead dataset scaffold.

CPU-only. Does not load images, extract embeddings, run VIS, or train models.
"""

import argparse
import csv
import os
from collections import Counter


FIELDS = [
    "sample_id", "source_batch", "task_key", "state_id", "seed", "run_id", "condition",
    "candidate_role", "phase_bin_proxy", "window_start", "window_end", "trigger_step",
    "gripper_qpos_at_trigger", "gripper_width_at_trigger", "gripper_command_at_trigger",
    "eef_speed_mean_pre", "open_streak_pre", "close_streak_pre", "phase_gate_score", "hazard_score",
    "image_trigger_path", "image_trigger_minus4_path", "image_trigger_minus8_path",
    "image_camera_name", "visual_available", "missing_visual_reason",
    "global_embedding_path", "crop_embedding_path", "visual_encoder_name", "visual_feature_dim",
    "label_vulnerability_ready", "label_physical_response", "label_task_failure",
    "label_control_negative", "label_status", "label_source",
    "denominator_status", "provenance_status", "infra_status", "leakage_audit_pass",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v1.csv")
    ap.add_argument("--trace-root", default="/data/liuyu/outputs")
    ap.add_argument("--output-csv", default="tables/visual_transfer_dataset_v0.csv")
    ap.add_argument("--output-report", default="reports/VISUAL_TRANSFER_DATASET_V0_SUMMARY.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def read_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if norm(r.get("label_status")) in {"positive", "negative"}]


def image_path(trace_root, task, state, step):
    candidates = [
        os.path.join(trace_root, task, "state_%s" % state, "rgb", "step_%06d.png" % step),
        os.path.join(trace_root, task, "s%s" % state, "rgb", "%06d.png" % step),
        os.path.join(trace_root, "frames", task, "state_%s" % state, "step_%06d.png" % step),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def build_rows(labels, trace_root):
    rows = []
    for r in labels:
        source = norm(r.get("source_batch"))
        task = norm(r.get("task_key"))
        state = norm(r.get("state_id"))
        ws = int(float(norm(r.get("window_start")) or 0))
        we = int(float(norm(r.get("window_end")) or ws))
        sample_id = "%s_%s_s%s_w%d_%d" % (source, task, state, ws, we)
        p0 = image_path(trace_root, task, state, ws)
        p4 = image_path(trace_root, task, state, max(ws - 4, 0))
        p8 = image_path(trace_root, task, state, max(ws - 8, 0))
        visual_ok = os.path.exists(p0) and os.path.exists(p4) and os.path.exists(p8)
        missing = []
        if not os.path.exists(p0):
            missing.append("missing_trigger")
        if not os.path.exists(p4):
            missing.append("missing_trigger_minus4")
        if not os.path.exists(p8):
            missing.append("missing_trigger_minus8")
        label = norm(r.get("label_vulnerability_ready"))
        label_int = 1 if label == "1" else 0
        status = norm(r.get("label_status"))
        rows.append({
            "sample_id": sample_id,
            "source_batch": source,
            "task_key": task,
            "state_id": state,
            "seed": norm(r.get("seed")),
            "run_id": norm(r.get("run_id")),
            "condition": "label_source",
            "candidate_role": norm(r.get("candidate_role")),
            "phase_bin_proxy": norm(r.get("phase_bin_proxy")),
            "window_start": ws,
            "window_end": we,
            "trigger_step": ws,
            "gripper_qpos_at_trigger": norm(r.get("gripper_qpos_at_trigger")),
            "gripper_width_at_trigger": norm(r.get("gripper_width_at_trigger")),
            "gripper_command_at_trigger": norm(r.get("gripper_command_at_trigger")),
            "eef_speed_mean_pre": norm(r.get("eef_speed_mean_pre")),
            "open_streak_pre": norm(r.get("open_streak_pre")),
            "close_streak_pre": norm(r.get("close_streak_pre")),
            "phase_gate_score": norm(r.get("phase_gate_score")),
            "hazard_score": norm(r.get("hazard_score")),
            "image_trigger_path": p0,
            "image_trigger_minus4_path": p4,
            "image_trigger_minus8_path": p8,
            "image_camera_name": "rgb",
            "visual_available": str(visual_ok).lower(),
            "missing_visual_reason": "|".join(missing),
            "global_embedding_path": "",
            "crop_embedding_path": "",
            "visual_encoder_name": "",
            "visual_feature_dim": "",
            "label_vulnerability_ready": label,
            "label_physical_response": norm(r.get("label_physical_response")),
            "label_task_failure": norm(r.get("label_task_failure")),
            "label_control_negative": str(label_int == 0).lower(),
            "label_status": status,
            "label_source": source,
            "denominator_status": norm(r.get("denominator_status") or r.get("denominator_type")),
            "provenance_status": norm(r.get("provenance_status")),
            "infra_status": norm(r.get("infra_status")),
            "leakage_audit_pass": "unknown",
        })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_report(path, rows, dry_run):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by_source = Counter(r["source_batch"] for r in rows)
    visual = sum(1 for r in rows if r["visual_available"] == "true")
    lines = [
        "# Visual Transfer Dataset V0 Summary",
        "",
        "CPU-only scaffold. No images, embeddings, or visual models were loaded.",
        "",
        f"**Dry run**: {str(dry_run).lower()}",
        f"**Rows**: {len(rows)}",
        f"**Visual available rows**: {visual}",
        "",
        "## By Source",
        "",
        "| Source | Count |",
        "|---|---:|",
    ]
    for k in sorted(by_source):
        lines.append(f"| {k} | {by_source[k]} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- This dataset is path/metadata scaffold only.",
        "- Real frozen embeddings and scientific visual evaluation come later.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    rows = build_rows(read_labels(args.labels_csv), args.trace_root)
    write_csv(args.output_csv, rows)
    write_report(args.output_report, rows, args.dry_run)
    print("Wrote %d rows -> %s" % (len(rows), args.output_csv))
    print("Report: %s" % args.output_report)


if __name__ == "__main__":
    main()
