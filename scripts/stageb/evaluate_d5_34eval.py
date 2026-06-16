#!/usr/bin/env python3
"""D5-8: 34-state external evaluation — one-shot, frozen config."""
import csv, json, math, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from evaluate_d5_frozen import load_model, online_detect

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--eval-root", default="/data/liuyu/outputs/d4_34_privileged_replay")
ap.add_argument("--config", default="/data/liuyu/outputs/d5_training/d5_frozen_config.json")
ap.add_argument("--output-dir", default="/data/liuyu/outputs/d5_training")
args = ap.parse_args()

ROOT = args.eval_root
CONFIG = json.load(open(args.config))
OUT = args.output_dir
model, means, stdevs, impute, ckpt = load_model(CONFIG["checkpoint_path"])
tau = CONFIG["tau"]


def sf(v):
    if v is None or v == "":
        return None, False
    try:
        f = float(v)
        return (f, math.isfinite(f))
    except:
        return (None, False)


def teacher_p_anchor(rows):
    streak = 0
    for r in rows:
        env_v, env_ok = sf(r.get("env_gripper"))
        ev, ev_ok = sf(r.get("env_valid"))
        so, so_ok = sf(r.get("semantics_ok"))
        if not env_ok or not ev_ok or not so_ok:
            return -1, -1, -1
        ok = bool(int(ev)) and bool(int(so))
        cc = 1 if (ok and env_v > 0.5) else 0
        co = 1 if (cc and streak == 0) else 0
        streak = streak + 1 if cc else 0
        r["_co"] = co
        do_v, do_ok = sf(r.get("decoded_open"))
        r["_dob"] = int(do_v or 0) > 0

    for t in range(len(rows)):
        r = rows[t]
        if not r.get("_co"):
            continue
        if r.get("_dob"):
            continue
        d_v, d_ok = sf(r.get("eef_to_obj_pre"))
        if not d_ok or d_v > 0.08:
            continue
        z_anc, z_ok = sf(r.get("obj_pre_z"))
        if not z_ok:
            continue
        sustained = 0
        for i in range(1, 16):
            if t + i >= len(rows):
                break
            f = rows[t + i]
            zf, okz = sf(f.get("obj_pre_z"))
            df, okd = sf(f.get("eef_to_obj_pre"))
            if not okz or not okd:
                sustained = 0
                continue
            if (zf - z_anc) >= 0.005 and df <= 0.08:
                sustained += 1
                if sustained >= 2:
                    ws = max(0, t - 2)
                    we = ws + 10
                    return t, ws, we
            else:
                sustained = 0
    return -1, -1, -1


results = []
for d in sorted(os.listdir(ROOT)):
    dp = os.path.join(ROOT, d)
    if not os.path.isdir(dp):
        continue
    m = re.match(r"(.+)_s(\d+)_shadow_attempt1", d)
    if not m:
        continue
    task = m.group(1)
    sid = int(m.group(2))

    stf = os.path.join(dp, "step_trace.csv")
    if not os.path.exists(stf):
        continue
    rows = list(csv.DictReader(open(stf)))
    if not rows:
        continue

    anchor, ws, we = teacher_p_anchor(rows)
    status = "VALID_LABELED" if anchor >= 0 else "VALID_TEACHER_P_ABSTAIN"

    det = online_detect(dp, model, means, stdevs, impute, tau)
    emit = det["emit_step"]

    if emit < 0:
        cls = "miss"
    elif emit < ws:
        cls = "pre_window_early"
    elif emit == anchor:
        cls = "exact_anchor"
    elif emit < anchor:
        cls = "in_window_pre_anchor"
    elif emit < we:
        cls = "in_window_post_anchor"
    else:
        cls = "late"

    in_win = emit >= ws and emit < we if emit >= 0 else False
    offset = emit - anchor if emit >= 0 else None

    results.append({
        "task": task, "state_id": sid, "split": "external_eval",
        "teacher_p_status": status, "teacher_p_anchor": anchor,
        "ws": ws, "we": we,
        "emit_step": emit, "emit_score": det["emit_score"],
        "emit_class": cls, "in_window": int(in_win),
        "offset": offset if offset is not None else "",
        "n_candidates": det["n_candidates"],
        "is_labeled": int(status == "VALID_LABELED"),
    })

# Write
out_csv = os.path.join(OUT, "d5_34eval_readout.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

# Summary
labeled = [r for r in results if r["is_labeled"]]
abstain = [r for r in results if not r["is_labeled"]]
n = len(labeled) if labeled else 1
iw = sum(r["in_window"] for r in labeled)
exact = sum(1 for r in labeled if r["emit_class"] == "exact_anchor")
early = sum(1 for r in labeled if r["emit_class"] == "pre_window_early")
miss = sum(1 for r in labeled if r["emit_class"] == "miss")
emit_n = sum(1 for r in labeled if r["emit_step"] >= 0)

print("34-state External Evaluation (tau={:.3f})".format(tau))
print("Total: {} | Labeled: {} | Abstain: {}".format(
    len(results), len(labeled), len(abstain)))
if labeled:
    print("In-Win:  {}/{} ({:.0f}%)".format(iw, len(labeled), 100 * iw / len(labeled)))
    print("Exact:   {}/{} ({:.0f}%)".format(exact, len(labeled), 100 * exact / len(labeled)))
    print("Early:   {}/{} ({:.0f}%)".format(early, len(labeled), 100 * early / len(labeled)))
    print("Miss:    {}/{} ({:.0f}%)".format(miss, len(labeled), 100 * miss / len(labeled)))
    print("Emit:    {}/{} ({:.0f}%)".format(emit_n, len(labeled), 100 * emit_n / len(labeled)))

# Per-task
from collections import defaultdict
task_m = defaultdict(list)
for r in labeled:
    task_m[r["task"]].append(r)
print("\nPer-task:")
for tk in sorted(task_m):
    tr = task_m[tk]
    print("  {}: {}/{} ({:.0f}%)".format(
        tk, sum(r["in_window"] for r in tr), len(tr),
        100 * sum(r["in_window"] for r in tr) / len(tr)))

print("\nReadout: {}".format(out_csv))
