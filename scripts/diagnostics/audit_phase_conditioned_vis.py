#!/usr/bin/env python3
"""audit_phase_conditioned_vis.py — phase-conditioned VIS result audit with claim gates.

Groups by (task, seed, window_source, selector_type, phase, window_start, window_end).
Computes VIS/random/clean metrics per group. Applies strict claim taxonomy.
"""

from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
from collections import defaultdict
try: import numpy as np
except ImportError: np = None

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
_src = str(REPO / "src")
if os.path.isdir(_src): sys.path.insert(0, _src)
try: from gripper_attack.gripper_semantics import raw_gripper_is_open
except ImportError: raw_gripper_is_open = lambda v: float(v) < 0.5

WINDOW_LEN_TOLERANCE = 2
QPOS_OPENING_DELTA_THRESH = 0.03   # directional: qpos_start - qpos_min >= 0.03
RANDOM_QPOS_THRESH = 0.005

# Fields to try for raw gripper, in priority order
GRIP_FIELDS = ("adv_grip", "raw_gripper", "clean_grip", "clean_gripper_action", "adv_gripper_action")


def get_raw_gripper(row):
    """Extract raw gripper value from trace row with fallback."""
    for k in GRIP_FIELDS:
        if k in row and row[k] not in ("", None):
            try: return float(row[k])
            except (ValueError, TypeError): continue
    return None


def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("true","1","yes")


def compute_trace_metrics(trace_path):
    """Per-trace metrics: OPEN count, qpos delta, done, armL2."""
    try:
        with open(trace_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception: return None
    if not rows: return None
    r0 = rows[0]
    wr = [r for r in rows if parse_bool(r.get("in_window","False"))]
    n = len(wr)
    # Use helper for clean/generated OPEN — don't assume adv_grip exists
    open_cnt = sum(1 for r in wr if raw_gripper_is_open(get_raw_gripper(r) or 0.996))
    al2 = max(float(r.get("arm_l2",0)) for r in wr) if wr else 0.0
    done = any(parse_bool(r.get("done","False")) for r in rows)
    qpost = [float(r.get("qpos_post_step",0)) for r in wr if r.get("qpos_post_step") and r["qpos_post_step"] not in ("", None)]
    # Directional opening: OPEN = qpos decreases. physical = start->min drop >= threshold
    qpos_start = qpost[0] if qpost else 0.0
    qpos_min = min(qpost) if qpost else 0.0
    qd_opening = qpos_start - qpos_min if len(qpost) > 1 else 0.0  # directional
    qd_abs = max(abs(v - qpost[0]) for v in qpost) if len(qpost) > 1 else 0.0  # diagnostic only
    invalid = parse_bool(r0.get("attack_invalid","False"))
    # Schema check
    grip_val = get_raw_gripper(r0) if n > 0 else None
    schema_incomplete = (grip_val is None and n > 0) or (len(qpost) == 0 and n > 0)
    return {"trace_path":trace_path,"task":r0.get("task",""),"seed":r0.get("seed",""),
        "condition":r0.get("condition",""),"window_start":r0.get("window_start",""),
        "window_end":r0.get("window_end",""),"window_source":r0.get("window_source","fixed"),
        "selector_type":r0.get("selector_type","unknown"),"phase":r0.get("phase",""),
        "generated_OPEN_count":open_cnt,"generated_OPEN_total":n,
        "qpos_opening_delta":round(qd_opening,6),
        "qpos_abs_delta":round(qd_abs,6),  # diagnostic only, not gated
        "qpos_post_start":round(qpos_start,6),"qpos_post_min":round(qpos_min,6),
        "armL2_max":round(al2,6),
        "done":done,"timeout":not done and len(rows)>=299,
        "attack_invalid":invalid,"valid":not invalid and not schema_incomplete}


def classify_bridge_taxonomy(vis_metrics, random_metrics, clean_metrics):
    """Apply claim gates to a (VIS, random, clean) triplet."""
    result = {
        "action_bridge_positive": False, "physical_bridge_positive": False,
        "task_failure_positive": False, "denominator_clean": False,
        "natural_release_confounded": False, "claim_usable": False,
        "taxonomy_label": "no_action_bridge",
    }
    if vis_metrics is None: return result

    n = vis_metrics.get("generated_OPEN_total",0)
    open_cnt = vis_metrics.get("generated_OPEN_count",0)
    qd_open = vis_metrics.get("qpos_opening_delta",0.0)  # directional: start -> min
    done = vis_metrics.get("done",True)

    result["action_bridge_positive"] = open_cnt >= max(1, n - WINDOW_LEN_TOLERANCE) if n>0 else False
    result["physical_bridge_positive"] = qd_open >= QPOS_OPENING_DELTA_THRESH  # directional gate
    result["task_failure_positive"] = not done

    if random_metrics:
        r_open = random_metrics.get("generated_OPEN_count",-1)
        r_done = random_metrics.get("done",False)
        r_qd = random_metrics.get("qpos_opening_delta", 999.0)
        result["denominator_clean"] = r_done and r_open==0 and r_qd <= RANDOM_QPOS_THRESH

    if clean_metrics:
        c_open = clean_metrics.get("generated_OPEN_count",0)
        c_total = clean_metrics.get("generated_OPEN_total",1)
        result["natural_release_confounded"] = (c_open / max(c_total,1)) > 0.5

    a = result["action_bridge_positive"]
    p = result["physical_bridge_positive"]
    t = result["task_failure_positive"]
    d = result["denominator_clean"]
    c = result["natural_release_confounded"]

    # Build taxonomy label — preserve both confound and denominator info
    labels = []
    if a and p and t and d and not c:
        result["claim_usable"] = True
        labels.append("claim_usable")
    else:
        if not a: labels.append("no_action_bridge")
        elif a and not p:
            labels.append("action_positive_physical_negative")
        elif a and p and not t:
            labels.append("action_positive_physical_positive_task_negative")
        if c: labels.append("natural_release_confounded")
        if not d and a: labels.append("denominator_polluted")
    result["taxonomy_label"] = "+".join(labels) if labels else "unclassified"
    return result


def compute_group_summary(vis_list, random_list, clean_list, window_info):
    """Aggregate metrics across multiple traces in a group."""
    s = dict(window_info)
    s["n_vis"]=len(vis_list); s["n_random"]=len(random_list); s["n_clean"]=len(clean_list)

    if vis_list:
        s["vis_OPEN_min"]=min(r["generated_OPEN_count"] for r in vis_list)
        s["vis_OPEN_max"]=max(r["generated_OPEN_count"] for r in vis_list)
        s["vis_OPEN_mean"]=round(np.mean([r["generated_OPEN_count"] for r in vis_list]),2)
        s["vis_qpos_delta_mean"]=round(np.mean([r["qpos_delta_post"] for r in vis_list]),6)
        s["vis_qpos_delta_min"]=round(min(r["qpos_delta_post"] for r in vis_list),6)
        s["vis_done_all_false"]=all(not r["done"] for r in vis_list)
        s["prefix_armL2_max"]=round(max(r["armL2_max"] for r in vis_list),6)

    if random_list:
        s["random_OPEN_max"]=max(r["generated_OPEN_count"] for r in random_list)
        s["random_done_all_true"]=all(r["done"] for r in random_list)
        s["random_armL2_max"]=round(max(r["armL2_max"] for r in random_list),6)

    if clean_list:
        s["clean_OPEN_mean"]=round(np.mean([r["generated_OPEN_count"]/max(r["generated_OPEN_total"],1) for r in clean_list]),4)

    # Apply taxonomy using first VIS/random/clean
    vis_m = vis_list[0] if vis_list else None
    rand_m = random_list[0] if random_list else None
    clean_m = clean_list[0] if clean_list else None
    tax = classify_bridge_taxonomy(vis_m, rand_m, clean_m)
    s.update(tax)
    return s


def _group_key(r):
    return (r.get("task",""), r.get("seed",""), r.get("window_source","fixed"),
            r.get("selector_type","unknown"), r.get("phase",""),
            r.get("window_start",""), r.get("window_end",""))


def main():
    ap = argparse.ArgumentParser(description="Phase-conditioned VIS audit")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--output-csv", default="tables/phase_conditioned_vis_provenance.csv")
    ap.add_argument("--summary-csv", default="tables/phase_conditioned_vis_summary.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN: audit_phase_conditioned_vis")
        print(f"  Groups by: task, seed, window_source, selector_type, phase, window_start, window_end")
        print(f"  Claim gates: action + physical + task + denominator + no_confound")
        return

    traces = []
    for d in args.run_dirs:
        for f in Path(d).rglob("*_trace.csv"):
            traces.append(str(f))

    all_metrics = []
    for tp in sorted(traces):
        m = compute_trace_metrics(tp)
        if m: all_metrics.append(m)

    print(f"Loaded {len(all_metrics)} traces ({sum(1 for m in all_metrics if m['valid'])} valid, {sum(1 for m in all_metrics if not m['valid'])} invalid)")

    # Group by config, then split by condition
    groups = defaultdict(lambda: {"vis":[], "random":[], "clean":[], "info":{}})
    for m in all_metrics:
        if not m["valid"]: continue
        gk = _group_key(m)
        g = groups[gk]
        g["info"] = {"task":m["task"],"seed":m["seed"],"window_source":m["window_source"],
            "selector_type":m["selector_type"],"phase":m["phase"],
            "window_start":m["window_start"],"window_end":m["window_end"]}
        cond = m["condition"]
        if cond == "vis_pgd": g["vis"].append(m)
        elif cond == "random_linf": g["random"].append(m)
        elif cond == "clean": g["clean"].append(m)

    summaries = []
    for gk, g in sorted(groups.items()):
        s = compute_group_summary(g["vis"], g["random"], g["clean"], g["info"])
        summaries.append(s)

    # Also write per-trace provenance
    prov_fields = ["trace_path","task","seed","condition","window_start","window_end",
        "generated_OPEN_count","generated_OPEN_total","qpos_delta_post","armL2_max",
        "done","timeout","attack_invalid","valid"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=prov_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_metrics)
    print(f"Wrote {len(all_metrics)} provenance rows to {args.output_csv}")

    if summaries:
        os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
        with open(args.summary_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader(); w.writerows(summaries)
        print(f"Wrote {len(summaries)} groups to {args.summary_csv}")
        # Print summary
        for s in summaries:
            label = s.get("taxonomy_label","?")
            claim = "CLAIM_USABLE" if s.get("claim_usable") else ""
            print(f"  {s.get('task')} seed{s.get('seed')} [{s.get('window_start')}-{s.get('window_end')}] {s.get('window_source')}: {label} {claim}")


if __name__ == "__main__":
    main()
