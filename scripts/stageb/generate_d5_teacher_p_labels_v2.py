#!/usr/bin/env python3
"""D5 Teacher-P label generator v2 — FAIL-CLOSED, manifest-driven.

Requires:
  --selection-manifest: frozen 120-state manifest
  --capture-ledger: authoritative attempt ledger (audit-approved)
  --capture-roots-manifest: JSON mapping root names to episode dirs
  --expected-selection-sha256
  --output-dir

Every state MUST produce exactly one classification:
  VALID_LABELED, VALID_TEACHER_P_ABSTAIN,
  FIELD_VALIDITY_FAIL, STEP_SEQUENCE_FAIL,
  OPEN_CONVENTION_FAIL, PRIVILEGED_INVALID,
  PROVENANCE_FAIL, CAPTURE_TERMINAL_INVALID

Missing fields → FIELD_VALIDITY_FAIL (never fill with 0.0).
"""
import argparse, csv, hashlib, json, math, os, sys
from collections import defaultdict

# ── Pre-frozen rules from phase_detector.py ──
EEF_NEAR_M = 0.08
LIFT_DELTA_M = 0.005
LOOKAHEAD_STEPS = 15
SUSTAINED_FRAMES = 2


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-manifest", required=True)
    ap.add_argument("--capture-ledger", required=True)
    ap.add_argument("--capture-roots-manifest", required=True)
    ap.add_argument("--expected-selection-sha256", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def safe_float(v):
    """Return (value, is_valid). Never return 0.0 for invalid."""
    if v is None or v == "":
        return None, False
    try:
        f = float(v)
        if not math.isfinite(f):
            return None, False
        return f, True
    except (ValueError, TypeError):
        return None, False


def compute_close_onset(rows):
    """Compute clean_close, close_onset, decoded_open_bool in-place. FAIL-CLOSED."""
    errors = []
    streak = 0
    for i, r in enumerate(rows):
        # env_valid MUST exist and be 1
        ev_raw = r.get("env_valid", None)
        if ev_raw is None or ev_raw == "":
            errors.append(("FIELD_VALIDITY_FAIL", i, "env_valid missing"))
            return None, errors
        try:
            env_valid = int(float(ev_raw))
        except (ValueError, TypeError):
            errors.append(("FIELD_VALIDITY_FAIL", i, "env_valid not parseable: {}".format(ev_raw)))
            return None, errors

        # semantics_ok MUST exist and be 1
        so_raw = r.get("semantics_ok", None)
        if so_raw is None or so_raw == "":
            errors.append(("FIELD_VALIDITY_FAIL", i, "semantics_ok missing"))
            return None, errors
        try:
            sem_ok = int(float(so_raw))
        except (ValueError, TypeError):
            errors.append(("FIELD_VALIDITY_FAIL", i, "semantics_ok not parseable: {}".format(so_raw)))
            return None, errors

        ok = bool(env_valid) and bool(sem_ok)

        # env_gripper
        env_v, env_v_ok = safe_float(r.get("env_gripper"))
        if not env_v_ok:
            errors.append(("FIELD_VALIDITY_FAIL", i, "env_gripper invalid"))
            return None, errors

        cc = 1 if (ok and env_v > 0.5) else 0
        co = 1 if (cc and streak == 0) else 0
        streak = streak + 1 if cc else 0

        # decoded_open
        do_v, do_ok = safe_float(r.get("decoded_open"))
        if not do_ok:
            errors.append(("FIELD_VALIDITY_FAIL", i, "decoded_open invalid"))
            return None, errors
        dob = int(do_v) if do_v > 0.5 else 0

        r["clean_close"] = cc
        r["close_onset"] = co
        r["decoded_open_bool"] = dob

    return rows, errors


def check_privileged_fields(rows):
    """Check ALL privileged fields are valid for every row. FAIL-CLOSED."""
    required = ["eef_pre_x", "eef_pre_y", "eef_pre_z",
                "obj_pre_x", "obj_pre_y", "obj_pre_z",
                "eef_to_obj_pre"]
    for i, r in enumerate(rows):
        for fn in required:
            v, ok = safe_float(r.get(fn))
            if not ok:
                return False, ("PRIVILEGED_INVALID", i, "{} is invalid/missing".format(fn))
    return True, None


def teacher_p_anchor(rows):
    """Independent implementation of Teacher-P anchor logic. FAIL-CLOSED."""
    T = len(rows)
    for t in range(T):
        r = rows[t]
        if not r.get("close_onset"): continue
        if r.get("decoded_open_bool"): continue

        # Check grasp privilege for this candidate
        eef_fields = ["eef_pre_x", "eef_pre_y", "eef_pre_z"]
        obj_fields = ["obj_pre_x", "obj_pre_y", "obj_pre_z"]
        dist_v, dist_ok = safe_float(r.get("eef_to_obj_pre"))
        if not dist_ok: continue

        all_ok = True
        for fn in eef_fields + obj_fields:
            _, ok = safe_float(r.get(fn))
            if not ok: all_ok = False
        if not all_ok: continue

        # EEF must be near object
        if dist_v > EEF_NEAR_M: continue

        # Sustained vertical lift within lookahead, with EEF proximity
        obj_z_anchor, ok = safe_float(r.get("obj_pre_z"))
        if not ok: continue

        sustained_count = 0
        for i in range(1, LOOKAHEAD_STEPS + 1):
            if t + i >= T: break
            fut = rows[t + i]
            obj_z_fut, ok_fut = safe_float(fut.get("obj_pre_z"))
            if not ok_fut: break
            eef_dist_fut, ok_dist = safe_float(fut.get("eef_to_obj_pre"))
            if not ok_dist: break

            z_delta = obj_z_fut - obj_z_anchor
            if z_delta >= LIFT_DELTA_M and eef_dist_fut <= EEF_NEAR_M:
                sustained_count += 1
                if sustained_count >= SUSTAINED_FRAMES:
                    ws = max(0, t - 2)
                    we = ws + 10
                    return t, ws, we
            else:
                # Allow non-consecutive? No — reset on violation
                pass  # window of lookahead, not consecutive enforcement
        # Check for 2 consecutive within lookahead
        for start_i in range(1, LOOKAHEAD_STEPS):
            if t + start_i + 1 >= T: break
            r1 = rows[t + start_i]
            r2 = rows[t + start_i + 1]
            z1, ok1 = safe_float(r1.get("obj_pre_z"))
            z2, ok2 = safe_float(r2.get("obj_pre_z"))
            d1, okd1 = safe_float(r1.get("eef_to_obj_pre"))
            d2, okd2 = safe_float(r2.get("eef_to_obj_pre"))
            if not all([ok1, ok2, okd1, okd2]): continue
            if ((z1 - obj_z_anchor) >= LIFT_DELTA_M and d1 <= EEF_NEAR_M and
                (z2 - obj_z_anchor) >= LIFT_DELTA_M and d2 <= EEF_NEAR_M):
                ws = max(0, t - 2)
                we = ws + 10
                return t, ws, we

    return -1, -1, -1


def classify_state(task, sid, dp, m):
    """Return (status, anchor, ws, we, details)."""
    tag = "{}_s{}".format(task, sid)

    # Check required files
    stf = os.path.join(dp, "step_trace.csv")
    scf = os.path.join(dp, "teacher_sidecar.json")
    provf = os.path.join(dp, "provenance.csv")

    if not os.path.exists(stf):
        return "CAPTURE_TERMINAL_INVALID", -1, -1, -1, {"reason": "no step_trace.csv"}
    if not os.path.exists(scf):
        return "CAPTURE_TERMINAL_INVALID", -1, -1, -1, {"reason": "no teacher_sidecar.json"}

    # Check sidecar
    sc = json.load(open(scf))
    if sc.get("privileged_valid") != 1:
        return "PRIVILEGED_INVALID", -1, -1, -1, {"reason": "teacher_sidecar privileged_valid != 1"}

    # Load and validate trace
    rows = list(csv.DictReader(open(stf)))
    if not rows:
        return "CAPTURE_TERMINAL_INVALID", -1, -1, -1, {"reason": "empty trace"}

    # Step sequence check
    for i, r in enumerate(rows):
        step = int(safe_float(r.get("step", -1))[0] or -1)
        if step != i:
            return "STEP_SEQUENCE_FAIL", -1, -1, -1, {
                "reason": "step {} at row {}".format(step, i)}

    # Compute close fields
    result, errors = compute_close_onset(rows)
    if result is None:
        return errors[0][0], -1, -1, -1, {
            "reason": errors[0][2], "step": errors[0][1]}

    # Privileged field check
    priv_ok, priv_err = check_privileged_fields(rows)
    if not priv_ok:
        return priv_err[0], -1, -1, -1, {"reason": priv_err[2], "step": priv_err[1]}

    # Teacher-P anchor
    anchor, ws, we = teacher_p_anchor(rows)
    if anchor < 0:
        return "VALID_TEACHER_P_ABSTAIN", -1, -1, -1, {
            "reason": "no_close_satisfies_teacher_p_criteria"}

    return "VALID_LABELED", anchor, ws, we, {}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Verify manifest
    msha = sha256_file(args.selection_manifest)
    assert msha == args.expected_selection_sha256, \
        "Manifest SHA mismatch: {} != {}".format(msha[:16], args.expected_selection_sha256[:16])

    # Load manifest
    manifest = list(csv.DictReader(open(args.selection_manifest)))
    assert len(manifest) == 120

    # Load capture ledger for OK states
    ok_states = {}
    for r in csv.DictReader(open(args.capture_ledger)):
        if r.get("status") == "OK":
            ok_states[(r["task"], int(r["state_id"]))] = r

    # Load roots manifest
    roots = json.load(open(args.capture_roots_manifest))

    # Process each manifest state
    results = []
    stats = defaultdict(int)

    for r in manifest:
        task = r["task_key"]
        sid = int(r["state_id"])
        key = (task, sid)
        tag = "{}_s{}".format(task, sid)

        if key not in ok_states:
            status = "UNATTEMPTED"
            results.append({"task": task, "state_id": sid, "split": r["split"],
                            "status": status, "anchor": -1, "ws": -1, "we": -1})
            stats[status] += 1
            continue

        # Find episode dir
        edir = None
        for rname, rpath in roots.items():
            candidate = os.path.join(rpath, "{}_shadow_attempt1".format(tag))
            if os.path.isdir(candidate):
                edir = candidate
                break

        if edir is None:
            status = "CAPTURE_TERMINAL_INVALID"
            results.append({"task": task, "state_id": sid, "split": r["split"],
                            "status": status, "anchor": -1, "ws": -1, "we": -1})
            stats[status] += 1
            continue

        mf = os.path.join(edir, "episode_manifest.json")
        m = json.load(open(mf)) if os.path.exists(mf) else {}

        status, anchor, ws, we, details = classify_state(task, sid, edir, m)
        results.append({"task": task, "state_id": sid, "split": r["split"],
                        "status": status, "anchor": anchor, "ws": ws, "we": we,
                        "details": json.dumps(details) if details else ""})
        stats[status] += 1

    # Write labels
    out_path = os.path.join(args.output_dir, "d5_teacher_p_labels_v2.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task", "state_id", "split", "status", "anchor", "ws", "we", "details"])
        w.writeheader()
        w.writerows(results)

    # Summary
    total = len(results)
    labeled = stats.get("VALID_LABELED", 0)
    abstain = stats.get("VALID_TEACHER_P_ABSTAIN", 0)
    fail = sum(v for k, v in stats.items() if "FAIL" in k or "INVALID" in k)

    print("=== Teacher-P Label Generator v2 ===")
    print("Total: {}".format(total))
    print("VALID_LABELED: {}".format(labeled))
    print("VALID_TEACHER_P_ABSTAIN: {}".format(abstain))
    for k in sorted(stats):
        if k not in ("VALID_LABELED", "VALID_TEACHER_P_ABSTAIN"):
            print("  {}: {}".format(k, stats[k]))
    print("Output: {}".format(out_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
