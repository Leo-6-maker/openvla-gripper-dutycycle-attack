#!/usr/bin/env python3
"""audit_phase_conditioned_vis.py — phase-conditioned VIS result audit with claim gates.

Groups by (task, seed, window_source, selector_type, phase, window_start, window_end).
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
QPOS_OPENING_DELTA_THRESH = 0.03
RANDOM_QPOS_THRESH = 0.005

GRIP_FIELDS = ("adv_grip", "raw_gripper", "clean_grip", "clean_gripper_action", "adv_gripper_action")


def get_raw_gripper(row):
    for k in GRIP_FIELDS:
        if k in row and row[k] not in ("", None):
            try: return float(row[k])
            except (ValueError, TypeError): continue
    return None


def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("true","1","yes")


def compute_trace_metrics(trace_path):
    """Per-trace metrics with schema safety."""
    try:
        with open(trace_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception: return None
    if not rows: return None
    r0 = rows[0]
    wr = [r for r in rows if parse_bool(r.get("in_window","False"))]
    n = len(wr)

    # ── OPEN count: EXACT None check, NOT "or 0.996" ──
    open_cnt = 0; missing_grip = 0
    for r in wr:
        g = get_raw_gripper(r)
        if g is None: missing_grip += 1
        elif raw_gripper_is_open(g): open_cnt += 1

    al2 = max(float(r.get("arm_l2",0)) for r in wr) if wr else 0.0
    done = any(parse_bool(r.get("done","False")) for r in rows)
    invalid = parse_bool(r0.get("attack_invalid","False"))

    # Directional qpos opening
    qpost = [float(r["qpos_post_step"]) for r in wr
             if r.get("qpos_post_step") and r["qpos_post_step"] not in ("", None)]
    missing_qpos_count = n - len(qpost)
    qpos_start = qpost[0] if qpost else 0.0
    qpos_min = min(qpost) if qpost else 0.0
    qd_opening = qpos_start - qpos_min if len(qpost) > 1 else 0.0
    qd_abs = max(abs(v - qpost[0]) for v in qpost) if len(qpost) > 1 else 0.0

    # Schema + exclusion: partial missing is also schema_incomplete
    schema_incomplete = (missing_grip > 0 and n > 0) or (missing_qpos_count > 0 and n > 0) or (n == 0)
    exclusion_reason = ""
    if invalid: exclusion_reason = "attack_invalid"
    elif missing_grip > 0 and n > 0 and missing_qpos_count > 0:
        exclusion_reason = f"missing_gripper_{missing_grip}_missing_qpos_{missing_qpos_count}"
    elif missing_grip > 0 and n > 0: exclusion_reason = f"missing_gripper_{missing_grip}"
    elif missing_qpos_count > 0 and n > 0: exclusion_reason = f"missing_qpos_post_step_{missing_qpos_count}"
    elif n == 0: exclusion_reason = "no_in_window_rows"

    return {
        "trace_path": trace_path,
        "task": r0.get("task",""), "seed": r0.get("seed",""),
        "condition": r0.get("condition",""),
        "window_start": r0.get("window_start",""),
        "window_end": r0.get("window_end",""),
        "window_source": r0.get("window_source","fixed"),
        "selector_type": r0.get("selector_type","unknown"),
        "phase": r0.get("phase",""),
        "generated_OPEN_count": open_cnt, "generated_OPEN_total": n,
        "qpos_opening_delta": round(qd_opening,6),
        "qpos_abs_delta": round(qd_abs,6),
        "qpos_post_start": round(qpos_start,6),
        "qpos_post_min": round(qpos_min,6),
        "armL2_max": round(al2,6),
        "done": done, "timeout": not done and len(rows) >= 299,
        "attack_invalid": invalid,
        "missing_gripper_count": missing_grip,
        "missing_qpos_count": missing_qpos_count,
        "schema_incomplete": schema_incomplete,
        "claim_excluded": invalid or schema_incomplete,
        "exclusion_reason": exclusion_reason,
        "valid": not invalid and not schema_incomplete,
    }


def classify_bridge_taxonomy(vis_metrics, random_metrics, clean_metrics):
    result = {
        "action_bridge_positive": False, "physical_bridge_positive": False,
        "task_failure_positive": False, "denominator_clean": False,
        "natural_release_confounded": False, "claim_usable": False,
        "taxonomy_label": "no_action_bridge",
    }
    if vis_metrics is None: return result
    n = vis_metrics.get("generated_OPEN_total",0)
    open_cnt = vis_metrics.get("generated_OPEN_count",0)
    qd_open = vis_metrics.get("qpos_opening_delta",0.0)
    done = vis_metrics.get("done",True)

    result["action_bridge_positive"] = open_cnt >= max(1, n - WINDOW_LEN_TOLERANCE) if n>0 else False
    result["physical_bridge_positive"] = qd_open >= QPOS_OPENING_DELTA_THRESH
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

    labels = []
    if a and p and t and d and not c:
        result["claim_usable"] = True; labels.append("claim_usable")
    else:
        if not a: labels.append("no_action_bridge")
        elif a and not p: labels.append("action_positive_physical_negative")
        elif a and p and not t: labels.append("action_positive_physical_positive_task_negative")
        if c: labels.append("natural_release_confounded")
        if not d and a: labels.append("denominator_polluted")
    result["taxonomy_label"] = "+".join(labels) if labels else "unclassified"
    return result


def compute_group_summary(vis_list, random_list, clean_list, window_info):
    s = dict(window_info)
    s["n_vis"]=len(vis_list); s["n_random"]=len(random_list); s["n_clean"]=len(clean_list)
    s["n_valid"] = sum(1 for r in vis_list+random_list+clean_list if r.get("valid"))

    if vis_list:
        s["vis_OPEN_min"]=min(r["generated_OPEN_count"] for r in vis_list)
        s["vis_OPEN_max"]=max(r["generated_OPEN_count"] for r in vis_list)
        s["vis_OPEN_mean"]=round(np.mean([r["generated_OPEN_count"] for r in vis_list]),2) if np else 0
        s["vis_qpos_opening_delta_mean"]=round(np.mean([r["qpos_opening_delta"] for r in vis_list]),6) if np else 0
        s["vis_qpos_opening_delta_min"]=round(min(r["qpos_opening_delta"] for r in vis_list),6)
        s["vis_qpos_abs_delta_mean"]=round(np.mean([r["qpos_abs_delta"] for r in vis_list]),6) if np else 0
        s["vis_done_all_false"]=all(not r["done"] for r in vis_list)
        s["prefix_armL2_max"]=round(max(r["armL2_max"] for r in vis_list),6)

    if random_list:
        s["random_OPEN_max"]=max(r["generated_OPEN_count"] for r in random_list)
        s["random_done_all_true"]=all(r["done"] for r in random_list)
        s["random_qpos_opening_delta_max"]=round(max(r["qpos_opening_delta"] for r in random_list),6)
        s["random_qpos_abs_delta_max"]=round(max(r["qpos_abs_delta"] for r in random_list),6)
        s["random_armL2_max"]=round(max(r["armL2_max"] for r in random_list),6)

    if clean_list:
        s["clean_OPEN_mean"]=round(np.mean([r["generated_OPEN_count"]/max(r["generated_OPEN_total"],1) for r in clean_list]),4) if np else 0

    # Duplicate detection
    s["duplicate_condition_count"] = 0
    if len(vis_list) > 1: s["duplicate_condition_count"] += 1
    if len(random_list) > 1: s["duplicate_condition_count"] += 1
    if len(clean_list) > 1: s["duplicate_condition_count"] += 1

    # Conservative aggregation: use min OPEN/min qpos for VIS, all-clean for random/denominator
    if vis_list:
        vis_agg = min(vis_list, key=lambda r: r["generated_OPEN_count"])
    else:
        vis_agg = None
    if random_list:
        rand_agg = max(random_list, key=lambda r: r["generated_OPEN_count"])  # worst random
    else:
        rand_agg = None
    if clean_list:
        clean_agg = max(clean_list, key=lambda r: r["generated_OPEN_count"] / max(r["generated_OPEN_total"], 1))
    else:
        clean_agg = None

    tax = classify_bridge_taxonomy(vis_agg, rand_agg, clean_agg)
    if s["duplicate_condition_count"] > 0 and not tax["claim_usable"]:
        tax["taxonomy_label"] = "duplicate_runs_present+" + tax["taxonomy_label"]
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
        return

    traces = []
    for d in args.run_dirs:
        for f in Path(d).rglob("*_trace.csv"):
            traces.append(str(f))

    all_metrics = []
    for tp in sorted(traces):
        m = compute_trace_metrics(tp)
        if m: all_metrics.append(m)

    n_valid = sum(1 for m in all_metrics if m["valid"])
    n_excluded = sum(1 for m in all_metrics if m["claim_excluded"])
    print(f"Loaded {len(all_metrics)} traces ({n_valid} valid, {n_excluded} excluded)")
    for m in all_metrics:
        if m["claim_excluded"]:
            print(f"  EXCLUDED: {m['trace_path']} reason={m['exclusion_reason']}")

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

    # Provenance CSV with correct field names
    prov_fields = ["trace_path","task","seed","condition","window_start","window_end",
        "window_source","selector_type","phase",
        "generated_OPEN_count","generated_OPEN_total",
        "qpos_opening_delta","qpos_abs_delta","qpos_post_start","qpos_post_min",
        "armL2_max","done","timeout",
        "attack_invalid","schema_incomplete","claim_excluded","exclusion_reason"]
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
        for s in summaries:
            label = s.get("taxonomy_label","?")
            claim = "CLAIM_USABLE" if s.get("claim_usable") else ""
            print(f"  {s.get('task')} seed{s.get('seed')} [{s.get('window_start')}-{s.get('window_end')}] {s.get('window_source')}: {label} {claim}")


if __name__ == "__main__":
    main()
