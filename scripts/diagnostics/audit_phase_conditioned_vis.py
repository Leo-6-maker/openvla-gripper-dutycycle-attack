#!/usr/bin/env python3
"""audit_phase_conditioned_vis.py — aggregator for phase-conditioned VIS results."""

from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.gripper_semantics import raw_gripper_is_open

# Claim gate definitions
WINDOW_LEN_TOLERANCE = 2  # Accept OPEN >= window_len - 2 as action_bridge_positive
QPOS_DELTA_THRESH = 0.03
RANDOM_QPOS_THRESH = 0.005


def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("true","1","yes")


def compute_metrics(trace_path):
    with open(trace_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows: return None
    r0 = rows[0]
    wr = [r for r in rows if parse_bool(r.get("in_window","False"))]

    open_cnt = sum(1 for r in wr if raw_gripper_is_open(float(r.get("adv_grip",0.996))))
    clean_open = sum(1 for r in wr if raw_gripper_is_open(float(r.get("clean_grip",0.996)))) if wr and "clean_grip" in wr[0] else 0
    al2 = max(float(r.get("arm_l2",0)) for r in wr) if wr else 0.0
    done = any(parse_bool(r.get("done","False")) for r in rows)
    qpost = [float(r.get("qpos_post_step",0)) for r in wr if r.get("qpos_post_step")]
    qd = max(abs(v-qpost[0]) for v in qpost) if len(qpost)>1 else 0.0
    n = len(wr)

    action_pos = open_cnt >= max(1, n - WINDOW_LEN_TOLERANCE) if n>0 else False
    physical_pos = qd >= QPOS_DELTA_THRESH if n>0 else False
    task_fail = not done if n>0 else False
    clean_ratio = clean_open / max(n,1) if n>0 else 0
    confounded = clean_ratio > 0.5

    return {
        "trace_path": trace_path,
        "task": r0.get("task",""), "seed": r0.get("seed",""), "condition": r0.get("condition",""),
        "window_start": r0.get("window_start",""), "window_end": r0.get("window_end",""),
        "generated_OPEN_count": open_cnt, "generated_OPEN_total": n,
        "clean_natural_OPEN": clean_open, "clean_natural_OPEN_ratio": round(clean_ratio,4),
        "qpos_delta_post": round(qd,6), "armL2_max": round(al2,6),
        "done": done, "timeout": not done and len(rows)>=299,
        "action_bridge_positive": action_pos,
        "physical_bridge_positive": physical_pos,
        "task_failure_positive": task_fail,
        "natural_release_confounded": confounded,
        "failure_phase": "early_grasp_disruption" if (not done and open_cnt>=n-2) else "unknown",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--output-csv", default="tables/phase_conditioned_vis_provenance.csv")
    ap.add_argument("--summary-csv", default="tables/phase_conditioned_vis_summary.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(f"DRY RUN: would scan {args.run_dirs}")
        return

    traces = []
    for d in args.run_dirs:
        for f in Path(d).rglob("*_trace.csv"):
            traces.append(str(f))

    results = []
    for tp in sorted(traces):
        m = compute_metrics(tp)
        if m: results.append(m)

    if not results:
        print("No traces found.")
        return

    # Provenance
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print(f"Wrote {len(results)} rows to {args.output_csv}")

    # Summary: count by (task, condition, window_start, window_end)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        k = (r["task"], r["condition"], r["window_start"], r["window_end"])
        groups[k].append(r)

    summary_fields = ["task","condition","window_start","window_end","n_runs",
        "action_pos","physical_pos","task_fail","confounded","clean_OPEN_mean",
        "generated_OPEN_mean","qpos_delta_mean","armL2_max"]
    summaries = []
    for k, grp in groups.items():
        s = dict(zip(summary_fields, k + (len(grp),)))
        s["action_pos"] = all(r["action_bridge_positive"] for r in grp)
        s["physical_pos"] = all(r["physical_bridge_positive"] for r in grp)
        s["task_fail"] = all(r["task_failure_positive"] for r in grp)
        s["confounded"] = any(r["natural_release_confounded"] for r in grp)
        s["clean_OPEN_mean"] = round(sum(r["clean_natural_OPEN_ratio"] for r in grp)/len(grp),4)
        s["generated_OPEN_mean"] = round(sum(r["generated_OPEN_count"]/max(r["generated_OPEN_total"],1) for r in grp),4)
        s["qpos_delta_mean"] = round(sum(r["qpos_delta_post"] for r in grp)/len(grp),6)
        s["armL2_max"] = round(max(r["armL2_max"] for r in grp),6)
        summaries.append(s)

    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields); w.writeheader(); w.writerows(summaries)
    print(f"Wrote {len(summaries)} groups to {args.summary_csv}")


if __name__ == "__main__":
    main()
