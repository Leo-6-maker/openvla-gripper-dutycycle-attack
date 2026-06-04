#!/usr/bin/env python3
"""build_phase_response_labels.py — Build vulnerability_ready labels from VIS outcomes.

CPU-only. Reads Batch1 merged summary + Batch2b VIS summary.
Output: tables/object_phase_response_labels_v0.csv, readiness report.
"""

from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch1-merged", default="tables/object_teacher_delay50_vis_smoke_merged_summary.csv")
    ap.add_argument("--batch2b-deno", default="tables/object_phase_response_batch2b_denominator_summary.csv")
    ap.add_argument("--batch2b-vis", default="tables/object_phase_response_batch2b_vis_summary.csv")
    ap.add_argument("--descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--output-labels", default="tables/object_phase_response_labels_v0.csv")
    ap.add_argument("--output-report", default="reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS.md")
    return ap.parse_args()


def classify_row(r):
    """Classify a VIS outcome row into vulnerability_ready label."""
    denom = str(r.get("denominator_clean","")).lower() in ("true","1","yes")
    vis_open = r.get("vis_OPEN_min","") or r.get("VIS_OPEN","0")
    qpos = float(r.get("vis_qpos_opening_delta_mean", r.get("qpos_opening_delta", 0)) or 0)
    done_false = str(r.get("vis_done_all_false", r.get("VIS_done",""))).lower() == "true"
    claim = str(r.get("claim_usable","")).lower() == "true"
    taxonomy = r.get("taxonomy_label", r.get("taxonomy",""))

    # Infrastructure check
    if "infra" in taxonomy.lower() or "polluted" in taxonomy.lower():
        return "ignore", "infrastructure_or_polluted"
    if not denom:
        return "ignore", "denominator_not_clean"

    # Physical response
    phys_label = "strong" if qpos >= 0.03 else ("weak" if qpos >= 0.01 else "none")

    if phys_label == "strong" and done_false:
        return 1, "claim_usable_positive"
    elif phys_label == "strong" and not done_false:
        return 0, "physical_strong_task_negative"
    elif phys_label == "weak":
        return 0, "weak_physical"
    else:
        return 0, "action_only_physical_none"


def main():
    args = parse_args()

    # Load phase descriptors for phase bin mapping
    phase_map = {}
    if os.path.exists(args.descriptors):
        with open(args.descriptors, newline="") as f:
            for r in csv.DictReader(f):
                key = (r.get("task_key",""), r.get("state_id",""), r.get("window_start",""), r.get("window_end",""))
                phase_map[key] = r.get("phase_bin_proxy","")

    CLOSED_PREGRASP_PHASES = {
        "approach_far_closed_proxy", "approach_near_closed_proxy",
        "pre_lock_closed_proxy", "grasp_formation_pre_lock_proxy",
        "far_closed_proxy", "near_closed_proxy", "pre_grasp_closed_proxy",
    }
    labels = []
    for source, csv_path in [("batch1", args.batch1_merged), ("batch2b", args.batch2b_vis)]:
        if not os.path.exists(csv_path): continue
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                value, reason = classify_row(r)
                task = r.get("task",""); state = r.get("state_id","")
                ws = r.get("window_start",""); we = r.get("window_end","")
                ph = phase_map.get((task, state, ws, we), r.get("phase_bin_proxy",""))
                # Additional labels
                qpos = float(r.get("vis_qpos_opening_delta_mean", r.get("qpos_opening_delta", 0)) or 0)
                vis_open = r.get("vis_OPEN_min","") or r.get("VIS_OPEN","0")
                vis_open_int = int(vis_open.split("/")[0]) if "/" in str(vis_open) else int(vis_open or 0)
                done_false = str(r.get("vis_done_all_false", r.get("VIS_done",""))).lower() == "true"
                phys_label = 1 if qpos >= 0.03 else (0.5 if qpos >= 0.01 else 0)
                labels.append(dict(
                    source=source, task_key=task, state_id=state,
                    window_start=ws, window_end=we, phase_bin_proxy=ph,
                    label_vulnerability_ready=value, label_reason=reason,
                    label_phase_gate_closed_pregrasp=1 if ph in CLOSED_PREGRASP_PHASES else 0,
                    label_physical_response=phys_label,
                    label_task_failure=1 if done_false else 0,
                    label_action_bridge=1 if vis_open_int >= 16 else 0,
                ))

    if not labels:
        print("No labels generated — waiting for Batch2b VIS results")
        return

    # Write labels
    os.makedirs(os.path.dirname(args.output_labels) or ".", exist_ok=True)
    with open(args.output_labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
        w.writeheader(); w.writerows(labels)

    # Counts
    pos = [l for l in labels if l["label_vulnerability_ready"] == 1]
    neg = [l for l in labels if l["label_vulnerability_ready"] == 0]
    ign = [l for l in labels if isinstance(l["label_vulnerability_ready"], str)]

    pos_tasks = set(l["task_key"] for l in pos)
    pos_phases = defaultdict(int)
    for l in pos: pos_phases[l["phase_bin_proxy"]] += 1

    smoke_gate = len(pos) >= 3 and len(neg) >= 2 and len(pos_tasks) >= 2
    paper_gate = len(pos) >= 5 and len(neg) >= 5 and len(pos_tasks) >= 4

    report = f"""# Object Phase-Response Label Readiness v0

**Date**: 2026-06-04

## Label Distribution

| Label | Count |
|-------|-------|
| vulnerable (1) | {len(pos)} |
| not vulnerable (0) | {len(neg)} |
| ignored | {len(ign)} |
| Total | {len(labels)} |

## Positives by Phase Bin

""" + "\n".join(f"| {ph} | {n} |" for ph, n in sorted(pos_phases.items())) + f"""

## Positives by Task

""" + "\n".join(f"| {t} | {sum(1 for l in pos if l['task_key']==t)} |" for t in sorted(pos_tasks)) + f"""

## Negatives

""" + "\n".join(f"| {l['task_key']} | {l['phase_bin_proxy']} | {l['label_reason']} |" for l in neg[:10]) + f"""

## Ignored

""" + "\n".join(f"| {l['task_key']} | {l['label_reason']} |" for l in ign[:10]) + f"""

## Gates

| Gate | Requirement | Status |
|------|------------|--------|
| Smoke training | positives>=3, negatives>=2, tasks>=2 | {'PASS' if smoke_gate else 'FAIL'} |
| Paper-level | positives>=5, negatives>=5, tasks>=4 | {'PASS' if paper_gate else 'FAIL'} |

## Verdict

"""
    if smoke_gate:
        report += "Smoke training gate PASSED. Vulnerability detector smoke training may proceed."
    elif len(pos) > 0:
        report += f"Smoke gate NOT YET passed (pos={len(pos)}, neg={len(neg)}, tasks={len(pos_tasks)}). More VIS results needed."
    else:
        report += "No claim_usable positives yet. Awaiting Batch2b VIS completion."

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Labels: {len(labels)} rows, pos={len(pos)}, neg={len(neg)}, ign={len(ign)}")
    print(f"Smoke gate: {'PASS' if smoke_gate else 'FAIL'}")
    print(f"Report: {args.output_report}")


if __name__ == "__main__":
    main()
