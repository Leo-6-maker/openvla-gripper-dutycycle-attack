#!/usr/bin/env python3
"""D5: Generate Teacher-P labels on D4.4D privileged episode step_trace.csv files.

Reuses the pre-frozen teacher_privileged_critical_close_anchor from phase_detector.py.
"""
import os, sys, csv, json
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from gripper_attack.phase_detector import (
    teacher_privileged_critical_close_anchor,
    teacher_window_proposal,
    teacher_rule_critical_close_anchor,
)

ROOTS = [
    "/data/liuyu/outputs/d5_120_privileged_capture",
    "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
]
OUT = "/data/liuyu/outputs/d5_label_generation"


def compute_close_onset(rows):
    """Add close_onset, clean_close, decoded_open_bool to rows in-place."""
    streak = 0
    for r in rows:
        env = float(r.get("env_gripper", 0) or 0)
        env_valid = int(r.get("env_valid", 1) or 1)
        sem_ok = int(r.get("semantics_ok", 1) or 1)
        ok = bool(env_valid) and bool(sem_ok)
        cc = 1 if (ok and env > 0.5) else 0
        co = 1 if (cc and streak == 0) else 0
        streak = streak + 1 if cc else 0
        r["clean_close"] = cc
        r["close_onset"] = co
        r["decoded_open_bool"] = int(float(r.get("decoded_open", 0) or 0))


def to_teacher_record(r):
    """Map step_trace row to dict expected by teacher_privileged_critical_close_anchor."""
    def f(k, default=0.0):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else default
    return {
        "close_onset": r.get("close_onset", 0),
        "clean_close": r.get("clean_close", 0),
        "decoded_open_bool": r.get("decoded_open_bool", 0),
        "eef_to_obj_distance": f("eef_to_obj_pre"),
        "eef_x": f("eef_pre_x"), "eef_y": f("eef_pre_y"), "eef_z": f("eef_pre_z"),
        "obj_x": f("obj_pre_x"), "obj_y": f("obj_pre_y"), "obj_z": f("obj_pre_z"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)

    episodes = []
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            dp = os.path.join(root, d)
            if not os.path.isdir(dp) or "_shadow_attempt" not in d:
                continue
            mf = os.path.join(dp, "episode_manifest.json")
            scf = os.path.join(dp, "teacher_sidecar.json")
            if not os.path.exists(mf) or not os.path.exists(scf):
                continue
            m = json.load(open(mf))
            sc = json.load(open(scf))
            if m.get("fatal") or m.get("infra_status") != "ok":
                continue
            if sc.get("privileged_valid") != 1:
                continue
            episodes.append((d, dp, m))

    episodes.sort()
    print("Episodes with privileged_valid=1: {}".format(len(episodes)))

    labels = []
    n_labeled = 0
    n_abstain = 0
    n_teacher_r = 0

    for tag, dp, m in episodes:
        task = m["task"]
        sid = m["state_id"]
        trace_path = os.path.join(dp, "step_trace.csv")
        rows = list(csv.DictReader(open(trace_path)))
        if not rows:
            continue

        compute_close_onset(rows)
        records = [to_teacher_record(r) for r in rows]

        # Teacher-P anchor
        p_anchor = teacher_privileged_critical_close_anchor(records)
        p_ws, p_we = teacher_window_proposal(p_anchor) if p_anchor >= 0 else (-1, -1)

        # Teacher-R anchor (deployment-safe baseline)
        r_anchor = teacher_rule_critical_close_anchor(rows)
        if r_anchor >= 0:
            n_teacher_r += 1

        status = "LABELED" if p_anchor >= 0 else "ABSTAIN"
        if p_anchor >= 0:
            n_labeled += 1
        else:
            n_abstain += 1

        labels.append({
            "task": task,
            "state_id": sid,
            "n_steps": m["n_steps"],
            "success": m["success_primary"],
            "teacher_p_anchor": p_anchor,
            "teacher_p_window_start": p_ws,
            "teacher_p_window_end": p_we,
            "teacher_p_status": status,
            "teacher_r_anchor": r_anchor,
            "detector_emit_step": m.get("detector_emit_step", -1),
        })

    # Write labels
    out_path = os.path.join(OUT, "d5_teacher_p_labels.csv")
    fields = [
        "task", "state_id", "n_steps", "success",
        "teacher_p_anchor", "teacher_p_window_start", "teacher_p_window_end",
        "teacher_p_status", "teacher_r_anchor", "detector_emit_step",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(labels)

    # Summary
    print("\n=== Teacher-P Label Summary ===")
    print("Total: {} | Labeled: {} | Abstain: {} | Teacher-R available: {}".format(
        len(labels), n_labeled, n_abstain, n_teacher_r))

    tc = Counter()
    for l in labels:
        tc[(l["task"], l["teacher_p_status"])] += 1
    for tk in sorted(set(l["task"] for l in labels)):
        labeled = tc.get((tk, "LABELED"), 0)
        abstain = tc.get((tk, "ABSTAIN"), 0)
        print("  {}: labeled={} abstain={}".format(tk, labeled, abstain))

    if n_labeled > 0:
        anchors = [l["teacher_p_anchor"] for l in labels if l["teacher_p_anchor"] >= 0]
        print("Anchor range: {} - {} (median {})".format(
            min(anchors), max(anchors), sorted(anchors)[len(anchors) // 2]))

    print("\nOutput: {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
