#!/usr/bin/env python3
"""Finalize Batch4 full-VIS results once server outputs arrive.

CPU-only. Reads candidates/precheck/VIS CSV artifacts and writes summary,
provenance, and a markdown report. Does not run VIS/rollout/GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from collections import Counter


SUMMARY_FIELDS = [
    "task_key", "state_id", "window_start", "window_end", "source_batch",
    "expected_role", "candidate_role", "phase_bin_proxy", "denominator_status",
    "VIS_OPEN_count", "qpos_opening_delta", "done", "steps", "claim_usable",
    "task_failure", "physical_response_strength", "provenance_status",
    "label_status", "label_vulnerability_ready", "classification", "reason",
]
PROV_FIELDS = ["task_key", "state_id", "window_start", "window_end", "trace_path", "manifest_path", "localized_trace_path", "provenance_status", "reason"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--precheck", default="tables/object_phase_response_batch4_precheck_summary.csv")
    ap.add_argument("--vis-root", default="/data/liuyu/outputs/object_phase_response_batch4_fullVIS_20260605")
    ap.add_argument("--output-summary", default="tables/object_phase_response_batch4_vis_summary.csv")
    ap.add_argument("--output-provenance", default="tables/object_phase_response_batch4_vis_provenance.csv")
    ap.add_argument("--output-report", default="reports/OBJECT_PHASE_RESPONSE_BATCH4_VIS_SUMMARY.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(v):
    return str(v if v is not None else "").strip()


def lower(v):
    return norm(v).lower()


def parse_float(v, default=0.0):
    try:
        text = norm(v)
        return default if text == "" else float(text)
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in reader]


def key(row):
    return (norm(row.get("task_key")), norm(row.get("state_id")), norm(row.get("window_start")), norm(row.get("window_end")))


def first(row, *fields):
    for field in fields:
        if norm(row.get(field)) != "":
            return row.get(field)
    return ""


def blocked_proxy(row):
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in ["phase_d", "phase_e", "command_proxy", "low_budget", "silver", "proxy"])


def discover_vis(root, row):
    if not Path(root).exists():
        return "", "", ""
    task = norm(row.get("task_key"))
    state = norm(row.get("state_id"))
    start = norm(row.get("window_start"))
    end = norm(row.get("window_end"))
    matches = []
    for path in Path(root).rglob("*.csv"):
        low = str(path).lower()
        if task.lower() in low and f"s{state}" in low and (start in low or end in low):
            matches.append(path)
    trace = str(matches[0]) if matches else ""
    manifest = ""
    localized = trace
    if trace:
        for parent in [Path(trace).parent, Path(trace).parent.parent]:
            for name in ["manifest.json", "run_manifest.json"]:
                p = parent / name
                if p.exists():
                    manifest = str(p)
                    break
            if manifest:
                break
    return trace, manifest, localized


def classify(row):
    denom = lower(row.get("denominator_status"))
    prov = lower(row.get("provenance_status"))
    if blocked_proxy(row):
        return "ignore", "", "ignore", "Phase D/E proxy/silver rows excluded"
    if denom != "clean":
        return "polluted", "", "ignore", "denominator_status is not clean"
    if any(tok in prov for tok in ["infra_failed", "xid", "oom", "cuda", "error"]):
        return "infra_failed", "", "ignore", "infra failure is not negative"
    if prov in {"missing_trace", "missing", ""}:
        return "manual_review", "", "manual_review", "missing trace/provenance is not negative"
    usable = lower(row.get("claim_usable"))
    done = lower(row.get("done"))
    qdelta = parse_float(row.get("qpos_opening_delta"), 0.0)
    vis_open = parse_float(row.get("VIS_OPEN_count"), 0.0)
    if usable in {"1", "true", "yes"} or (vis_open > 0 and qdelta > 0.01 and done in {"0", "false"}):
        return "positive", "1", "positive", "full VIS physical/task failure signal"
    if done in {"1", "true"} and qdelta <= 0.003:
        return "negative", "0", "negative", "full VIS did not produce physical failure"
    return "manual_review", "", "manual_review", "insufficient or ambiguous full VIS evidence"


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows, status):
    counts = Counter(r["classification"] for r in rows)
    denom_clean = sum(1 for r in rows if lower(r.get("denominator_status")) == "clean")
    completed = sum(1 for r in rows if lower(r.get("provenance_status")) == "ok")
    lines = [
        "# Object Phase Response Batch4 VIS Summary",
        "",
        f"**Status**: {status}",
        f"**Total candidates**: {len(rows)}",
        f"**Denominator clean count**: {denom_clean}",
        f"**VIS completed count**: {completed}",
        f"**Positives**: {counts.get('positive', 0)}",
        f"**Negatives**: {counts.get('negative', 0)}",
        f"**Controls**: {sum(1 for r in rows if 'control' in lower(r.get('expected_role')))}",
        f"**Infra/manual/polluted**: {counts.get('infra_failed', 0)}/{counts.get('manual_review', 0)}/{counts.get('polluted', 0)}",
        f"**Hard-negative yield**: {sum(1 for r in rows if r['classification'] == 'negative' and 'hard_negative' in lower(r.get('expected_role')))}",
        f"**Likely-positive yield**: {counts.get('positive', 0)}",
        "",
        "This is CPU-only closeout automation. Full VIS only is gold; Phase D/E proxy outputs are excluded.",
        "",
        "## Recommended Next Action",
        "",
        "- If status is blocked, wait for Batch4 full VIS traces and precheck summaries.",
        "- If completed, run labels_v3 candidate builder and schema audit.",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    if not os.path.exists(args.candidates):
        write_csv(args.output_summary, SUMMARY_FIELDS, [])
        write_csv(args.output_provenance, PROV_FIELDS, [])
        write_report(args.output_report, args, [], "BLOCKED_MISSING_BATCH4_CANDIDATES")
        return 0
    candidates = read_csv(args.candidates)
    pre = {key(r): r for r in read_csv(args.precheck)} if os.path.exists(args.precheck) else {}
    status = "OK"
    if not os.path.exists(args.precheck) or not Path(args.vis_root).exists():
        status = "BLOCKED_MISSING_BATCH4_OUTPUTS"
    summary = []
    provenance = []
    for cand in candidates:
        merged = dict(cand)
        merged.update({k: v for k, v in pre.get(key(cand), {}).items() if norm(v) != ""})
        trace, manifest, localized = discover_vis(args.vis_root, cand)
        prov = norm(first(merged, "provenance_status")) or ("ok" if trace else "missing_trace")
        row = {
            "task_key": norm(cand.get("task_key")), "state_id": norm(cand.get("state_id")),
            "window_start": norm(cand.get("window_start")), "window_end": norm(cand.get("window_end")),
            "source_batch": "batch4", "expected_role": norm(cand.get("expected_role")),
            "candidate_role": norm(cand.get("candidate_role")), "phase_bin_proxy": norm(cand.get("phase_bin_proxy")),
            "denominator_status": norm(first(merged, "denominator_status", "denominator_plan")) or "missing",
            "VIS_OPEN_count": norm(first(merged, "VIS_OPEN_count", "vis_open_count")),
            "qpos_opening_delta": norm(first(merged, "qpos_opening_delta", "qpos_delta")),
            "done": norm(merged.get("done")), "steps": norm(merged.get("steps")),
            "claim_usable": norm(merged.get("claim_usable")), "task_failure": norm(merged.get("task_failure")),
            "physical_response_strength": norm(merged.get("physical_response_strength")),
            "provenance_status": prov,
        }
        cls, label, status_label, reason = classify(row)
        row.update({"classification": cls, "label_vulnerability_ready": label, "label_status": status_label, "reason": reason})
        summary.append(row)
        provenance.append({"task_key": row["task_key"], "state_id": row["state_id"], "window_start": row["window_start"], "window_end": row["window_end"], "trace_path": trace, "manifest_path": manifest, "localized_trace_path": localized, "provenance_status": prov, "reason": reason})
    write_csv(args.output_summary, SUMMARY_FIELDS, summary)
    write_csv(args.output_provenance, PROV_FIELDS, provenance)
    write_report(args.output_report, args, summary, status)
    if args.dry_run:
        print(f"DRY RUN: finalized {len(summary)} Batch4 candidates; status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
