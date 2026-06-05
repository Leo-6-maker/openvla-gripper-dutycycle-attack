#!/usr/bin/env python3
"""Generate CPU-only window-compression smoke candidates from Batch3 summary."""

import argparse
import csv
import os
import sys


SELECTED = [
    ("cream_cheese", "4", 28, 45, "positive", "Batch3 claim_usable positive; representative new vulnerable task"),
    ("milk", "4", 19, 36, "positive", "Batch3 claim_usable positive; second new vulnerable task"),
    ("ketchup", "1", 21, 38, "positive", "Batch3 claim_usable positive; known ketchup transfer reference"),
    ("salad_dressing", "0", 7, 24, "negative", "Batch3 physical bridge but task-negative control"),
    ("bbq_sauce", "5", 27, 44, "negative", "Batch3 physical bridge but task-negative control"),
]
LENGTHS = [12, 10, 8]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-report", default="reports/OBJECT_WINDOW_COMPRESSION_CANDIDATE_PLAN.md")
    return ap.parse_args()


def centered_window(parent_start, parent_end, length):
    start = int(round(((parent_start + parent_end) - (length - 1)) / 2.0))
    end = start + length - 1
    if start < parent_start:
        start = parent_start
        end = start + length - 1
    if end > parent_end:
        end = parent_end
        start = end - length + 1
    if start < 0:
        raise ValueError("compressed window starts before zero")
    if start < parent_start or end > parent_end:
        raise ValueError("compressed window is not inside parent")
    return start, end


def main():
    args = parse_args()
    if not os.path.exists(args.summary_csv):
        raise SystemExit(f"summary CSV not found: {args.summary_csv}")

    with open(args.summary_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_key = {}
    for r in rows:
        task = r.get("task_key") or r.get("task") or ""
        key = (task, str(r.get("state_id", "")), str(r.get("window_start", "")), str(r.get("window_end", "")))
        by_key[key] = r

    out = []
    missing = []
    seen = set()
    for task, state, ws, we, role, reason in SELECTED:
        row = by_key.get((task, state, str(ws), str(we)))
        if row is None:
            missing.append(f"{task} s{state} [{ws},{we}]")
            continue
        for length in LENGTHS:
            cws, cwe = centered_window(ws, we, length)
            dup_key = (task, state, cws, cwe)
            if dup_key in seen:
                raise SystemExit(f"duplicate compressed window: {dup_key}")
            seen.add(dup_key)
            out.append(
                {
                    "target_id": f"{task}_s{state}_p{ws}_{we}_L{length}_w{cws}_{cwe}",
                    "task_key": task,
                    "state_id": state,
                    "parent_window_start": ws,
                    "parent_window_end": we,
                    "compressed_window_start": cws,
                    "compressed_window_end": cwe,
                    "compression_len": length,
                    "label_role": role,
                    "source_batch": "batch3",
                    "phase_bin_proxy": row.get("phase_bin_proxy") or row.get("phase") or "unknown",
                    "reason_selected": reason,
                }
            )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = [
        "target_id",
        "task_key",
        "state_id",
        "parent_window_start",
        "parent_window_end",
        "compressed_window_start",
        "compressed_window_end",
        "compression_len",
        "label_role",
        "source_batch",
        "phase_bin_proxy",
        "reason_selected",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out)

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    lines = [
        "# Object Window Compression Candidate Plan",
        "",
        f"**Input**: `{args.summary_csv}`",
        f"**Output**: `{args.output_csv}`",
        f"**Candidates**: {len(out)}",
        "",
        "## Selection",
        "",
        "- Positives: cream_cheese s4 [28,45], milk s4 [19,36], ketchup s1 [21,38]",
        "- Negatives: salad_dressing s0 [7,24], bbq_sauce s5 [27,44]",
        "- Compression lengths: L12, L10, L8 centered inside each parent window.",
        "- CPU-only generation; no rollout, VIS, GPU, or server output mutation.",
        "",
        "## Candidate Table",
        "",
        "| target_id | role | parent | compressed |",
        "|---|---|---|---|",
    ]
    for r in out:
        lines.append(
            f"| {r['target_id']} | {r['label_role']} | "
            f"[{r['parent_window_start']},{r['parent_window_end']}] | "
            f"[{r['compressed_window_start']},{r['compressed_window_end']}] |"
        )
    lines += ["", "## Missing Source Rows", ""]
    if missing:
        lines.extend(f"- {m}" for m in missing)
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Use Boundary",
        "",
        "These rows are candidate windows only. They do not establish compressed-window effectiveness until DeepSeek runs matched VIS/random under clean denominator controls.",
        "",
    ]
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(out)} candidates to {args.output_csv}")
    print(f"Report: {args.output_report}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
