#!/usr/bin/env python3
"""build_clean_phase_dataset.py — clean-only phase/event-labeled dataset builder.

Input: clean rollout trace CSVs
Output: tables/phase_alignment_clean_rollouts.csv with per-step phase labels
        tables/phase_event_summary.csv with per-rollout event times
"""

from __future__ import annotations
import argparse, csv, os, sys, json
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("ATTACK_REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524"))
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.gripper_semantics import raw_gripper_is_open, CANONICAL_OPEN_SEMANTICS_VERSION

PHASE_6CLASS = {0:"approach",1:"pregrasp",2:"grasp_formation",3:"stable_grasp_or_lift",4:"carry_or_place",5:"release_or_done"}
PHASE_3CLASS = {0:"pre_grasp",1:"grasp_formation",2:"post_grasp"}

# Configurable heuristic thresholds (all can be CLI-overridden)
DEFAULTS = {
    "close_onset_k": 3, "grasp_lock_k": 5, "lift_dz_thresh": 0.001,
    "lift_k": 3, "carry_dist_thresh": 0.005, "release_open_k": 3,
    "qpos_closed_thresh": 0.02, "qpos_std_thresh": 0.0005,
}


def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("true","1","yes")


def load_traces(run_dirs, task_filter=None, seed_filter=None):
    """Recursively find and load clean trace CSVs."""
    traces = []
    for d in run_dirs:
        for f in Path(d).rglob("*_trace.csv"):
            traces.append(str(f))
    loaded = []
    for tp in sorted(traces):
        try:
            with open(tp, newline="") as fh:
                rows = list(csv.DictReader(fh))
            if not rows: continue
            r0 = rows[0]
            cond = r0.get("condition","")
            if cond != "clean": continue
            task = r0.get("task","")
            seed = r0.get("seed","0")
            if task_filter and task not in task_filter: continue
            if seed_filter is not None and int(seed) not in seed_filter: continue
            loaded.append((tp, rows))
        except Exception: continue
    return loaded


def detect_events(steps, cfg=None):
    """Heuristic phase event detection."""
    cfg = cfg or DEFAULTS
    n = len(steps)
    if n < 5: return {}, np.full(n,-1,int), np.full(n,-1,int)

    # Extract arrays
    env_grip = np.array([float(s.get("env_gripper", -1.0)) for s in steps])
    raw_grip = np.array([float(s.get("raw_gripper", s.get("adv_grip", 0.996))) for s in steps])
    qpos = np.array([float(s.get("qpos_post_step", s.get("gripper_qpos", 0.03))) for s in steps])
    eef_z = np.array([float(s.get("eef_z", 0)) for s in steps])
    eef_x = np.array([float(s.get("eef_x", 0)) for s in steps])
    eef_y = np.array([float(s.get("eef_y", 0)) for s in steps])
    done = np.array([parse_bool(s.get("done","False")) for s in steps])

    is_close = env_grip < 0
    is_open_canon = np.array([raw_gripper_is_open(float(s.get("raw_gripper", s.get("adv_grip", 0.996)))) for s in steps])

    # T_gripper_close_onset
    T_close = None
    streak = 0
    for i in range(n):
        streak = streak+1 if is_close[i] else 0
        if streak >= cfg["close_onset_k"]:
            T_close = i - cfg["close_onset_k"] + 1; break

    # T_grasp_formation_start
    T_gform = T_close
    if T_close is not None:
        s0, s1 = max(0,T_close-5), min(n,T_close+10)
        if s1 > s0+1:
            dq = np.diff(qpos[s0:s1])
            for j in range(len(dq)):
                if dq[j] < -0.001: T_gform = s0+j; break

    # T_grasp_lock
    T_lock = None
    if T_close is not None:
        for i in range(T_close+3, min(n, T_close+30) - cfg["grasp_lock_k"]):
            w = qpos[i:i+cfg["grasp_lock_k"]]
            if np.std(w) < cfg["qpos_std_thresh"] and np.mean(w) < cfg["qpos_closed_thresh"]:
                T_lock = i; break

    # T_lift_start
    T_lift = None
    if T_lock is not None:
        s0 = T_lock; s1 = min(n, s0+40)
        if s1 > s0+cfg["lift_k"]:
            dz = np.diff(eef_z[s0:s1])
            st = 0
            for j,d in enumerate(dz):
                st = st+1 if d > cfg["lift_dz_thresh"] else 0
                if st >= cfg["lift_k"]: T_lift = s0+j-cfg["lift_k"]+1; break

    # T_release_start: natural OPEN
    T_rel = None
    s0 = (T_lift or 0)+10; s1 = n
    if s1 > s0+cfg["release_open_k"]:
        st = 0
        for i in range(s0, s1):
            st = st+1 if is_open_canon[i] else 0
            if st >= cfg["release_open_k"]: T_rel = i-cfg["release_open_k"]+1; break

    # T_done
    T_dn = next((i for i in range(n) if done[i]), None)

    events = {"T_gripper_close_onset":T_close,"T_grasp_formation_start":T_gform,
              "T_grasp_lock":T_lock,"T_lift_start":T_lift,
              "T_release_start":T_rel,"T_done":T_dn,"n_steps":n}

    # Per-step labels
    ph6 = np.full(n, -1, int)
    for i in range(n):
        if T_close is not None and i < T_close: ph6[i] = 1
        if T_gform is not None and i >= (T_gform or 0) and (T_lock is None or i < (T_lock or n)): ph6[i] = max(ph6[i],2)
        if T_lock is not None and i >= T_lock and (T_lift is None or i < (T_lift or n)): ph6[i] = 3
        if T_lift is not None and i >= T_lift and (T_rel is None or i < (T_rel or n)): ph6[i] = 4
        if T_rel is not None and i >= T_rel: ph6[i] = 5

    ph3 = np.full(n, -1, int)
    for i in range(n):
        if ph6[i] in (0,1): ph3[i] = 0
        elif ph6[i] == 2: ph3[i] = 1
        elif ph6[i] in (3,4,5): ph3[i] = 2

    return events, ph6, ph3


def main():
    ap = argparse.ArgumentParser(description="Build clean phase-labeled dataset")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--output-csv", default="tables/phase_alignment_clean_rollouts.csv")
    ap.add_argument("--summary-csv", default="tables/phase_event_summary.csv")
    ap.add_argument("--summary-report", default="reports/VIS_PHASE_DATASET_AUDIT.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-privileged-labels", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("Dry run: would load clean traces from", args.run_dirs)
        print("Output:", args.output_csv)
        return

    loaded = load_traces(args.run_dirs, args.tasks, args.seeds)
    print(f"Loaded {len(loaded)} clean traces")

    all_rows = []; summaries = []
    for tp, steps in loaded:
        r0 = steps[0]
        task = r0.get("task","?"); seed = r0.get("seed","?")
        events, ph6, ph3 = detect_events(steps)
        summaries.append({"task":task,"seed":seed,"rollout_id":Path(tp).stem,
            "trace_path":tp,**{k: v if v is not None else "" for k,v in events.items()},
            "label_validity":"heuristic" if events.get("T_grasp_formation_start") is not None else "incomplete"})

        for i,s in enumerate(steps):
            all_rows.append({
                "task":task,"seed":seed,"rollout_id":Path(tp).stem,"trace_path":tp,
                "policy_step":i,
                "raw_gripper":s.get("raw_gripper",s.get("adv_grip","")),
                "env_gripper":s.get("env_gripper",""),
                "qpos":s.get("qpos_post_step",s.get("gripper_qpos","")),
                "eef_x":s.get("eef_x",""),"eef_y":s.get("eef_y",""),"eef_z":s.get("eef_z",""),
                "arm_l2":s.get("arm_l2","0"),"done":s.get("done","False"),
                "phase_label_6class":PHASE_6CLASS.get(int(ph6[i]),"unknown"),
                "phase_label_6class_id":int(ph6[i]),
                "phase_label_3class":PHASE_3CLASS.get(int(ph3[i]),"unknown"),
                "phase_label_3class_id":int(ph3[i]),
                "T_gripper_close_onset":events.get("T_gripper_close_onset",""),
                "T_grasp_formation_start":events.get("T_grasp_formation_start",""),
                "T_grasp_lock":events.get("T_grasp_lock",""),
                "T_lift_start":events.get("T_lift_start",""),
                "T_release_start":events.get("T_release_start",""),
                "T_done":events.get("T_done",""),
                "label_confidence":"medium","label_source":"heuristic",
                "label_validity":"heuristic" if events.get("T_grasp_formation_start") is not None else "incomplete",
            })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    if all_rows:
        with open(args.output_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} rows to {args.output_csv}")
    if summaries:
        with open(args.summary_csv,"w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
        print(f"Wrote {len(summaries)} summaries to {args.summary_csv}")


if __name__ == "__main__":
    main()
