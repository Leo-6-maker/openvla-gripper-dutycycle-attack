#!/usr/bin/env python3
"""D5-1: Independent Teacher-P label auditor.

Does NOT import teacher_privileged_critical_close_anchor or any producer function.
Independently recomputes: close_onset, EEF-object proximity, sustained lift,
anchor, and window. Compares against producer labels row-by-row.

Hard gates:
  producer/auditor status disagreement = 0
  anchor disagreement = 0
  ws/we disagreement = 0
"""
import argparse, csv, json, math, os, sys
from collections import defaultdict, Counter

# ── Frozen constants (same as phase_detector.py, copied here for independence) ──
EEF_NEAR_M = 0.08
LIFT_DELTA_M = 0.005
LOOKAHEAD = 15
SUSTAINED_FRAMES = 2
PRE_OFFSET = 2
WINDOW_LEN = 10

ROOTS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
}


def safe_float(v):
    if v is None or v == "":
        return None, False
    try:
        f = float(v)
        if not math.isfinite(f):
            return None, False
        return f, True
    except (ValueError, TypeError):
        return None, False


def auditor_compute_close_onset(rows):
    """Independent close_onset computation."""
    streak = 0
    for r in rows:
        env_v, env_ok = safe_float(r.get("env_gripper"))
        ev_v, ev_ok = safe_float(r.get("env_valid"))
        so_v, so_ok = safe_float(r.get("semantics_ok"))
        if not all([env_ok, ev_ok, so_ok]):
            env_v = 0; ev_v = 0; so_v = 0
        ok = bool(int(ev_v)) and bool(int(so_v))
        cc = 1 if (ok and env_v > 0.5) else 0
        co = 1 if (cc and streak == 0) else 0
        streak = streak + 1 if cc else 0
        r["_aud_cc"] = cc
        r["_aud_co"] = co
        do_v, do_ok = safe_float(r.get("decoded_open"))
        r["_aud_dob"] = int(do_v) if (do_ok and do_v > 0.5) else 0


def auditor_teacher_p_anchor(rows):
    """Independent Teacher-P anchor computation."""
    T = len(rows)
    for t in range(T):
        r = rows[t]
        if not r.get("_aud_co"): continue
        if r.get("_aud_dob"): continue

        eef_dist, ok = safe_float(r.get("eef_to_obj_pre"))
        if not ok or eef_dist > EEF_NEAR_M: continue

        # Check EEF/object fields valid
        for fn in ["eef_pre_x", "eef_pre_y", "eef_pre_z",
                   "obj_pre_x", "obj_pre_y", "obj_pre_z"]:
            _, ok = safe_float(r.get(fn))
            if not ok: break
        else:
            # All valid
            obj_z_anchor, ok = safe_float(r.get("obj_pre_z"))
            if not ok: continue

            # Sustained CONSECUTIVE vertical lift
            sustained = 0
            for i in range(1, LOOKAHEAD + 1):
                if t + i >= T: break
                fut = rows[t + i]
                z_fut, ok_z = safe_float(fut.get("obj_pre_z"))
                d_fut, ok_d = safe_float(fut.get("eef_to_obj_pre"))
                if not ok_z or not ok_d:
                    sustained = 0
                    continue
                z_delta = z_fut - obj_z_anchor
                if z_delta >= LIFT_DELTA_M and d_fut <= EEF_NEAR_M:
                    sustained += 1
                    if sustained >= SUSTAINED_FRAMES:
                        ws = max(0, t - PRE_OFFSET)
                        we = ws + WINDOW_LEN
                        return t, ws, we
                else:
                    sustained = 0

    return -1, -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted-manifest", required=True)
    ap.add_argument("--producer-labels", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load accepted manifest
    accepted = {}
    for r in csv.DictReader(open(args.accepted_manifest)):
        if r.get("status") == "BOUND":
            accepted[(r["task"], int(r["state_id"]))] = r

    # Load producer labels
    producer = {}
    for r in csv.DictReader(open(args.producer_labels)):
        producer[(r["task"], int(r["state_id"]))] = r

    # Audit each state
    results = []
    disagreements = 0
    stats = defaultdict(int)

    for key, acc in sorted(accepted.items()):
        task, sid = key
        prod = producer.get(key, {})
        prod_status = prod.get("status", "MISSING")
        prod_anchor = int(prod.get("anchor", -999))
        prod_ws = int(prod.get("ws", -999))
        prod_we = int(prod.get("we", -999))

        rname = acc["accepted_root"]
        edir_name = acc["accepted_episode_dir"]
        rpath = ROOTS.get(rname, "")
        edir = os.path.join(rpath, edir_name) if rpath else ""
        tag = "{}_s{}".format(task, sid)

        if not edir or not os.path.isdir(edir):
            aud_status = "CAPTURE_TERMINAL_INVALID"
            aud_anchor = -1; aud_ws = -1; aud_we = -1
        else:
            stf = os.path.join(edir, "step_trace.csv")
            if not os.path.exists(stf):
                aud_status = "CAPTURE_TERMINAL_INVALID"
                aud_anchor = -1; aud_ws = -1; aud_we = -1
            else:
                rows = list(csv.DictReader(open(stf)))
                if not rows:
                    aud_status = "CAPTURE_TERMINAL_INVALID"
                    aud_anchor = -1; aud_ws = -1; aud_we = -1
                else:
                    # Step sequence check
                    step_ok = True
                    for i, r in enumerate(rows):
                        sv, s_ok = safe_float(r.get("step"))
                        if not s_ok or int(sv) != i:
                            step_ok = False; break
                    if not step_ok:
                        aud_status = "STEP_SEQUENCE_FAIL"
                        aud_anchor = -1; aud_ws = -1; aud_we = -1
                    else:
                        # Privileged fields check
                        priv_ok = True
                        for r in rows:
                            for fn in ["eef_pre_x","eef_pre_y","eef_pre_z",
                                       "obj_pre_x","obj_pre_y","obj_pre_z",
                                       "eef_to_obj_pre"]:
                                _, ok = safe_float(r.get(fn))
                                if not ok: priv_ok = False; break
                            if not priv_ok: break
                        if not priv_ok:
                            aud_status = "PRIVILEGED_INVALID"
                            aud_anchor = -1; aud_ws = -1; aud_we = -1
                        else:
                            auditor_compute_close_onset(rows)
                            aud_anchor, aud_ws, aud_we = auditor_teacher_p_anchor(rows)
                            aud_status = "VALID_LABELED" if aud_anchor >= 0 else "VALID_TEACHER_P_ABSTAIN"

        # Compare
        status_match = aud_status == prod_status
        anchor_match = aud_anchor == prod_anchor
        ws_match = aud_ws == prod_ws
        we_match = aud_we == prod_we
        all_match = status_match and anchor_match and ws_match and we_match

        if not all_match:
            disagreements += 1
            stats["disagreement"] += 1
        stats["total"] += 1

        results.append({
            "task": task, "state_id": sid,
            "prod_status": prod_status, "aud_status": aud_status,
            "prod_anchor": prod_anchor, "aud_anchor": aud_anchor,
            "prod_ws": prod_ws, "aud_ws": aud_ws,
            "prod_we": prod_we, "aud_we": aud_we,
            "status_match": int(status_match),
            "anchor_match": int(anchor_match),
            "ws_match": int(ws_match),
            "we_match": int(we_match),
        })

    # Write
    out = os.path.join(args.output_dir, "d5_teacher_p_independent_audit.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Summary
    n = len(results)
    status_m = sum(r["status_match"] for r in results)
    anchor_m = sum(r["anchor_match"] for r in results)
    ws_m = sum(r["ws_match"] for r in results)
    we_m = sum(r["we_match"] for r in results)

    print("Total audited: {}".format(n))
    print("Status match: {}/{}".format(status_m, n))
    print("Anchor match: {}/{}".format(anchor_m, n))
    print("Window start match: {}/{}".format(ws_m, n))
    print("Window end match: {}/{}".format(we_m, n))
    print("Disagreements: {}".format(disagreements))

    if disagreements > 0:
        print("\nDisagreements:")
        for r in results:
            if not (r["status_match"] and r["anchor_match"] and r["ws_match"] and r["we_match"]):
                print("  {}_s{}: prod=({},{},{},{}) aud=({},{},{},{})".format(
                    r["task"], r["state_id"],
                    r["prod_status"], r["prod_anchor"], r["prod_ws"], r["prod_we"],
                    r["aud_status"], r["aud_anchor"], r["aud_ws"], r["aud_we"]))

    result = "PASS" if disagreements == 0 else "FAIL"
    print("\nAUDITOR: {}".format(result))
    print("Output: {}".format(out))

    return 0 if disagreements == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
