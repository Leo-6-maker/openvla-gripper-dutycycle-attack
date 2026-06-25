#!/usr/bin/env python3
"""Phase 6B: Early-emission state audit for SC5-V2 seed42.
For each of the 15 early-emission TV trajectories, extract privileged state
at V2 emit_step vs teacher anchor step. Classify as:
  A = benign early recognition (grasp established, lift begun, stable carry)
  B = premature trigger (still closing, contact unstable, object supported)

Requires: step dataset CSV + raw telemetry CSVs for the 15 episodes.
Run on GPU server with access to telemetry files.
"""
import argparse, csv, json, math, os, sys, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

SC5_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

# The 15 early-emission episodes (from formal evaluator v3 per_episode_results.csv)
EARLY_EMISSION = [
    {"episode_id": "ep_0281", "task": 0, "state": 24, "emit_step": 85, "anchor": 87, "offset": 2},
    {"episode_id": "ep_0283", "task": 0, "state": 26, "emit_step": 107, "anchor": 108, "offset": 1},
    {"episode_id": "ep_0296", "task": 3, "state": 24, "emit_step": 122, "anchor": 124, "offset": 2},
    {"episode_id": "ep_0303", "task": 4, "state": 26, "emit_step": 77, "anchor": 78, "offset": 1},
    {"episode_id": "ep_0306", "task": 5, "state": 24, "emit_step": 105, "anchor": 106, "offset": 1},
    {"episode_id": "ep_0307", "task": 5, "state": 25, "emit_step": 227, "anchor": 230, "offset": 3},
    {"episode_id": "ep_0308", "task": 5, "state": 26, "emit_step": 75, "anchor": 76, "offset": 1},
    {"episode_id": "ep_0311", "task": 6, "state": 24, "emit_step": 108, "anchor": 110, "offset": 2},
    {"episode_id": "ep_0312", "task": 6, "state": 25, "emit_step": 76, "anchor": 77, "offset": 1},
    {"episode_id": "ep_0314", "task": 6, "state": 27, "emit_step": 73, "anchor": 74, "offset": 1},
    {"episode_id": "ep_0316", "task": 7, "state": 24, "emit_step": 87, "anchor": 89, "offset": 2},
    {"episode_id": "ep_0317", "task": 7, "state": 25, "emit_step": 77, "anchor": 78, "offset": 1},
    {"episode_id": "ep_0319", "task": 7, "state": 27, "emit_step": 77, "anchor": 79, "offset": 2},
    {"episode_id": "ep_0322", "task": 8, "state": 25, "emit_step": 75, "anchor": 76, "offset": 1},
    {"episode_id": "ep_0324", "task": 8, "state": 27, "emit_step": 80, "anchor": 81, "offset": 1},
]

# ── Classification thresholds ──
GRASP_VALID_GRIPPER_QPOS_MIN = 0.03      # qpos_sum >= this → fingers closed enough
GRASP_VALID_EEF_OBJ_DIST_MAX = 0.15      # eef_obj_dist <= this → object near gripper
GRASP_VALID_CLOSE_ONSET = True            # close_onset must be True
LIFT_VALID_OBJ_Z_DELTA_MIN = 0.01        # obj_z increase from episode start
CARRY_STABLE_OPENING_VAR_MAX = 0.005     # opening_proxy_variance_5 below this
CARRY_STABLE_EEF_OBJ_DIST_VAR_MAX = 0.02 # eef_obj_dist variance over 5-step window
WINDOW = 5                                # steps for stability window


def load_telemetry(telemetry_dir, episode_id):
    """Load telemetry CSV for an episode. Returns list of dicts sorted by step."""
    # Try to find the telemetry file — could be in different directory structures
    candidates = [
        telemetry_dir / f"{episode_id}" / "step_telemetry.csv",
        telemetry_dir / f"{episode_id}.tmp" / "step_telemetry.csv",
    ]
    # Also search subdirectories
    if telemetry_dir.exists():
        for p in telemetry_dir.rglob(f"*{episode_id}*"):
            if p.is_dir():
                tel = p / "step_telemetry.csv"
                if tel.exists():
                    candidates.append(tel)

    for cand in candidates:
        if cand.exists():
            rows = list(csv.DictReader(open(cand)))
            rows.sort(key=lambda r: int(r.get("step", 0)))
            return rows, str(cand)

    return None, ""


def load_step_dataset_rows(dataset_csv, episode_id):
    """Load all rows for an episode from the step dataset CSV."""
    rows = []
    with open(dataset_csv) as f:
        for r in csv.DictReader(f):
            if r["episode_id"] == episode_id:
                rows.append(r)
    rows.sort(key=lambda r: int(r.get("step_idx", 0)))
    return rows


def get_row_at_step(rows, target_step, step_key="step"):
    """Get the row closest to target_step."""
    best = None
    best_dist = float("inf")
    for r in rows:
        s = int(r.get(step_key, -1))
        dist = abs(s - target_step)
        if dist < best_dist:
            best_dist = dist
            best = r
    return best, best_dist


def compute_window_stats(rows, center_step, window, step_key="step"):
    """Compute mean and variance of key metrics in a window around center_step."""
    vals = defaultdict(list)
    for r in rows:
        s = int(r.get(step_key, -1))
        if center_step - window <= s <= center_step + window:
            for k in r:
                try:
                    vals[k].append(float(r[k]))
                except (ValueError, TypeError):
                    pass

    stats = {}
    for k, vlist in vals.items():
        if len(vlist) >= 2:
            arr = np.array(vlist)
            stats[f"{k}_mean"] = float(np.mean(arr))
            stats[f"{k}_var"] = float(np.var(arr))
            stats[f"{k}_std"] = float(np.std(arr))
    return stats


def audit_trajectory(ep_info, tel_rows, ds_rows):
    """Audit a single trajectory at emit_step and anchor_step.
    Returns classification dict.
    """
    eid = ep_info["episode_id"]
    emit_s = ep_info["emit_step"]
    anchor_s = ep_info["anchor"]
    offset = ep_info["offset"]

    result = {
        "episode_id": eid,
        "task": ep_info["task"],
        "state": ep_info["state"],
        "emit_step": emit_s,
        "anchor_step": anchor_s,
        "offset": offset,
    }

    # Get telemetry rows at both steps
    tel_emit, dist_emit = (None, 999)
    tel_anchor, dist_anchor = (None, 999)
    if tel_rows:
        tel_emit, dist_emit = get_row_at_step(tel_rows, emit_s)
        tel_anchor, dist_anchor = get_row_at_step(tel_rows, anchor_s)

    # Get step dataset rows
    ds_emit, ds_dist_emit = get_row_at_step(ds_rows, emit_s, "step_idx")
    ds_anchor, ds_dist_anchor = get_row_at_step(ds_rows, anchor_s, "step_idx")

    result["tel_available"] = tel_rows is not None
    result["tel_emit_step_match_dist"] = dist_emit
    result["tel_anchor_step_match_dist"] = dist_anchor

    # ── Extract privileged state from telemetry ──
    if tel_emit:
        result["emit_obj_x"] = safe_float(tel_emit, "obj_x")
        result["emit_obj_y"] = safe_float(tel_emit, "obj_y")
        result["emit_obj_z"] = safe_float(tel_emit, "obj_z")
        result["emit_eef_x"] = safe_float(tel_emit, "eef_x")
        result["emit_eef_y"] = safe_float(tel_emit, "eef_y")
        result["emit_eef_z"] = safe_float(tel_emit, "eef_z")
        result["emit_qpos_sum"] = safe_float(tel_emit, "qpos_sum")
        result["emit_eef_obj_dist"] = safe_float(tel_emit, "eef_obj_dist")
        result["emit_raw_gripper"] = safe_float(tel_emit, "raw_gripper")
        result["emit_corridor_p"] = safe_float(tel_emit, "corridor_p")
        result["emit_release_p"] = safe_float(tel_emit, "release_p")

        # Window stats around emit
        emit_win = compute_window_stats(tel_rows, emit_s, WINDOW)
        for k, v in emit_win.items():
            result[f"emit_{k}"] = v

    if tel_anchor:
        result["anchor_obj_x"] = safe_float(tel_anchor, "obj_x")
        result["anchor_obj_y"] = safe_float(tel_anchor, "obj_y")
        result["anchor_obj_z"] = safe_float(tel_anchor, "obj_z")
        result["anchor_eef_x"] = safe_float(tel_anchor, "eef_x")
        result["anchor_eef_y"] = safe_float(tel_anchor, "eef_y")
        result["anchor_eef_z"] = safe_float(tel_anchor, "eef_z")
        result["anchor_qpos_sum"] = safe_float(tel_anchor, "qpos_sum")
        result["anchor_eef_obj_dist"] = safe_float(tel_anchor, "eef_obj_dist")
        result["anchor_raw_gripper"] = safe_float(tel_anchor, "raw_gripper")
        result["anchor_corridor_p"] = safe_float(tel_anchor, "corridor_p")
        result["anchor_release_p"] = safe_float(tel_anchor, "release_p")

    # ── Extract SC5 features at both steps ──
    for fn in SC5_FEATURES:
        if ds_emit:
            result[f"emit_{fn}"] = safe_float(ds_emit, fn)
        if ds_anchor:
            result[f"anchor_{fn}"] = safe_float(ds_anchor, fn)

    # ── Compute deltas between emit and anchor ──
    for metric in ["obj_z", "eef_z", "qpos_sum", "eef_obj_dist"]:
        ev = result.get(f"emit_{metric}")
        av = result.get(f"anchor_{metric}")
        if ev is not None and av is not None and not (math.isnan(ev) or math.isnan(av)):
            result[f"delta_{metric}"] = av - ev

    # ── Classification logic ──
    checks = []

    # Check 1: Grasp established at emit time
    grasp_qpos_ok = True
    if "emit_qpos_sum" in result and result["emit_qpos_sum"] is not None:
        if not math.isnan(result["emit_qpos_sum"]):
            grasp_qpos_ok = result["emit_qpos_sum"] >= GRASP_VALID_GRIPPER_QPOS_MIN

    grasp_dist_ok = True
    if "emit_eef_obj_dist" in result and result["emit_eef_obj_dist"] is not None:
        if not math.isnan(result["emit_eef_obj_dist"]):
            grasp_dist_ok = result["emit_eef_obj_dist"] <= GRASP_VALID_EEF_OBJ_DIST_MAX

    grasp_close_onset_ok = True
    if "emit_close_onset" in result and result["emit_close_onset"] is not None:
        grasp_close_onset_ok = result["emit_close_onset"] > 0.5

    grasp_established = grasp_qpos_ok and grasp_dist_ok and grasp_close_onset_ok
    checks.append({
        "check": "grasp_established",
        "pass": grasp_established,
        "details": f"qpos={grasp_qpos_ok} dist={grasp_dist_ok} close_onset={grasp_close_onset_ok}"
    })

    # Check 2: Object lift begun (obj_z above initial)
    lift_begun = False
    obj_z_initial = None
    if tel_rows and len(tel_rows) > 0:
        try:
            obj_z_initial = float(tel_rows[0].get("obj_z", "nan"))
        except (ValueError, TypeError):
            obj_z_initial = None

    if "emit_obj_z" in result and obj_z_initial is not None:
        ev = result["emit_obj_z"]
        if ev is not None and not math.isnan(ev) and not math.isnan(obj_z_initial):
            lift_begun = (ev - obj_z_initial) >= LIFT_VALID_OBJ_Z_DELTA_MIN
            result["obj_z_initial"] = obj_z_initial
            result["emit_obj_z_delta_from_start"] = ev - obj_z_initial

    checks.append({
        "check": "lift_begun",
        "pass": lift_begun,
        "details": f"obj_z_delta={result.get('emit_obj_z_delta_from_start', 'N/A')}"
    })

    # Check 3: Grasp stability (low opening_proxy variance)
    grasp_stable = True
    emit_op_var = result.get("emit_opening_proxy_variance_5_var")
    if emit_op_var is not None and not math.isnan(emit_op_var):
        grasp_stable = emit_op_var <= CARRY_STABLE_OPENING_VAR_MAX
    elif "emit_opening_proxy_variance_5" in result:
        opv = result["emit_opening_proxy_variance_5"]
        if opv is not None and not math.isnan(opv):
            grasp_stable = opv <= CARRY_STABLE_OPENING_VAR_MAX
    checks.append({
        "check": "grasp_stable",
        "pass": grasp_stable,
        "details": f"opening_var={emit_op_var if emit_op_var else result.get('emit_opening_proxy_variance_5', 'N/A')}"
    })

    # Check 4: EEF-object distance stability
    dist_stable = True
    if "emit_eef_obj_dist_var" in result:
        ev = result["emit_eef_obj_dist_var"]
        if ev is not None and not math.isnan(ev):
            dist_stable = ev <= CARRY_STABLE_EEF_OBJ_DIST_VAR_MAX
    checks.append({
        "check": "eef_obj_dist_stable",
        "pass": dist_stable,
        "details": f"dist_var={result.get('emit_eef_obj_dist_var', 'N/A')}"
    })

    # Check 5: Offset within acceptable range
    offset_ok = offset <= 3
    checks.append({
        "check": "offset_le_3",
        "pass": offset_ok,
        "details": f"offset={offset}"
    })

    # Check 6: Gripper not still closing (gripper_qpos delta small)
    not_still_closing = True
    if "emit_qpos_delta_1" in result:
        qd1 = result["emit_qpos_delta_1"]
        if qd1 is not None and not math.isnan(qd1):
            not_still_closing = abs(qd1) <= 0.01  # small recent change
    checks.append({
        "check": "not_still_closing",
        "pass": not_still_closing,
        "details": f"qpos_delta_1={result.get('emit_qpos_delta_1', 'N/A')}"
    })

    # ── Final classification ──
    n_pass = sum(1 for c in checks if c["pass"])
    n_total = len(checks)

    # Type A (benign): all critical checks pass (grasp + lift + offset)
    critical_checks = ["grasp_established", "lift_begun", "offset_le_3"]
    critical_pass = all(c["pass"] for c in checks if c["check"] in critical_checks)

    if critical_pass and grasp_stable:
        classification = "A_benign_early_recognition"
        confidence = "high" if n_pass >= n_total - 1 else "medium"
    elif critical_pass and not grasp_stable:
        classification = "A_benign_early_recognition"
        confidence = "low"
    elif not critical_pass:
        # Which critical check failed?
        failed = [c["check"] for c in checks if c["check"] in critical_checks and not c["pass"]]
        classification = "B_premature_trigger"
        confidence = "high" if len(failed) >= 2 else "medium"
        result["failed_critical_checks"] = failed
    else:
        classification = "B_premature_trigger"
        confidence = "medium"

    result["classification"] = classification
    result["confidence"] = confidence
    result["checks"] = checks
    result["n_checks_pass"] = n_pass
    result["n_checks_total"] = n_total

    return result


def safe_float(row, key):
    """Safely extract float from row."""
    v = row.get(key, "")
    if v in ("", "nan", "NaN", None):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True, help="Path to SC5_V2_STEP_DATASET.csv")
    ap.add_argument("--telemetry_dir", required=True, help="Directory containing episode telemetry subdirs")
    ap.add_argument("--output_dir", required=True, help="Output directory for audit results")
    ap.add_argument("--episode_ids", nargs="*", help="Specific episodes to audit (default: all 15)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    telemetry_dir = Path(args.telemetry_dir)

    targets = EARLY_EMISSION
    if args.episode_ids:
        targets = [e for e in EARLY_EMISSION if e["episode_id"] in args.episode_ids]

    print(f"Auditing {len(targets)} early-emission trajectories...")
    print(f"Telemetry dir: {telemetry_dir}")
    print(f"Dataset CSV: {args.dataset_csv}")

    results = []
    for ep in targets:
        eid = ep["episode_id"]
        print(f"\n{'='*60}")
        print(f"Episode: {eid}  task={ep['task']}  state={ep['state']}")
        print(f"  emit_step={ep['emit_step']}  anchor={ep['anchor']}  offset={ep['offset']}")

        # Load data
        tel_rows, tel_path = load_telemetry(telemetry_dir, eid)
        if tel_rows:
            print(f"  Telemetry: {tel_path} ({len(tel_rows)} steps)")
        else:
            print(f"  Telemetry: NOT FOUND (searched under {telemetry_dir})")

        ds_rows = load_step_dataset_rows(args.dataset_csv, eid)
        print(f"  Step dataset: {len(ds_rows)} steps")

        # Audit
        result = audit_trajectory(ep, tel_rows, ds_rows)

        # Print classification
        print(f"  Classification: {result['classification']} (confidence={result['confidence']})")
        for c in result["checks"]:
            status = "PASS" if c["pass"] else "FAIL"
            print(f"    [{status}] {c['check']}: {c['details']}")

        results.append(result)

    # ── Save per-episode CSV ──
    csv_path = os.path.join(args.output_dir, "SC5_V2_EARLY_EMISSION_STATE_AUDIT.csv")
    # Collect all field names
    all_fields = ["episode_id", "task", "state", "emit_step", "anchor_step", "offset",
                  "tel_available", "classification", "confidence", "n_checks_pass", "n_checks_total"]
    # Add all metric fields from results
    metric_fields = set()
    for r in results:
        for k in r:
            if k not in all_fields and k != "checks":
                metric_fields.add(k)
    all_fields += sorted(metric_fields)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            # Flatten checks
            row = dict(r)
            row.pop("checks", None)
            w.writerow(row)
    print(f"\nSaved per-episode CSV: {csv_path}")

    # ── Save summary JSON ──
    type_a = [r for r in results if r["classification"].startswith("A")]
    type_b = [r for r in results if r["classification"].startswith("B")]
    tel_avail = sum(1 for r in results if r["tel_available"])

    summary = {
        "gate": "SC5_V2_EARLY_EMISSION_STATE_AUDIT",
        "total_episodes": len(results),
        "telemetry_available": tel_avail,
        "telemetry_missing": len(results) - tel_avail,
        "classification": {
            "A_benign_early_recognition": len(type_a),
            "B_premature_trigger": len(type_b),
        },
        "timing_gate": {
            "max_offset_le_3": all(r["offset"] <= 3 for r in results),
            "zero_grasp_invalid_emits": sum(1 for r in results
                if any(c["check"] == "grasp_established" and not c["pass"] for c in r["checks"])),
            "benign_count_target": ">= 12/15",
            "benign_count_actual": len(type_a),
        },
        "per_episode": [],
    }

    for r in results:
        entry = {
            "episode_id": r["episode_id"],
            "task": r["task"],
            "state": r["state"],
            "offset": r["offset"],
            "classification": r["classification"],
            "confidence": r["confidence"],
            "tel_available": r["tel_available"],
            "failed_checks": [c["check"] for c in r["checks"] if not c["pass"]],
        }
        # Include key metrics
        for k in ["emit_obj_z_delta_from_start", "emit_qpos_sum", "emit_eef_obj_dist",
                   "emit_close_onset", "emit_opening_proxy_variance_5"]:
            if k in r:
                entry[k] = r[k]
        summary["per_episode"].append(entry)

    json_path = os.path.join(args.output_dir, "SC5_V2_EARLY_EMISSION_SUMMARY.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary JSON: {json_path}")

    # ── Print final summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Benign (A): {len(type_a)}/15")
    print(f"  Premature (B): {len(type_b)}/15")
    print(f"  Telemetry available: {tel_avail}/15")
    print(f"  Max offset: {max(r['offset'] for r in results)}")

    if len(type_b) > 0:
        print(f"\n  WARNING: {len(type_b)} premature triggers detected!")
        for r in type_b:
            print(f"    {r['episode_id']}: {r.get('failed_critical_checks', 'N/A')}")

    timing_pass = (
        all(r["offset"] <= 3 for r in results) and
        len(type_b) == 0 and
        len(type_a) >= 12
    )
    print(f"\n  TIMING GATE: {'PASS' if timing_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
