#!/usr/bin/env python3
"""D2 Phase C: Fresh clean trace eligibility pipeline.
Runs the same E4C.2b pipeline on freshly collected traces:
  provenance → field validity → open convention → RC1a remap →
  CLOSE candidate enumeration → per-candidate Teacher-P evidence →
  ambiguity detection → classification.
"""

import argparse, csv, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = "/data/liuyu/l12_e4c2_pipeline"
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "scripts", "stageb"))

import numpy as np
from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.phase_detector import (
    _classify_motion_evidence, _check_grasp_privilege_valid,
    _safe_float, EEF_TO_OBJ_NEAR_THRESHOLD, OBJECT_LIFT_MIN_DELTA,
    OBJECT_LIFT_LOOKAHEAD, SUSTAINED_MOTION_FRAMES,
    MOTION_SUSTAINED_VERTICAL_LIFT,
    check_teacher_p_privilege_capability,
    teacher_privileged_critical_close_anchor,
    teacher_rule_critical_close_anchor,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor, PREDICTION_HORIZON,
)

REQUIRED_HEADER = [
    "obj_x", "obj_y", "obj_z", "eef_x", "eef_y", "eef_z",
    "clean_gripper_env", "decoded_open_bool", "gripper_qpos_before",
]

# Reuse functions from E4C.2b
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "stageb"))
from run_l12_e4c2b_repair import (
    sha256_file, check_field_validity_v3, check_open_convention_v2,
    evaluate_teacher_p_for_candidate, classify_trace_v3,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with fresh trace CSVs")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # Discover fresh traces
    import glob
    trace_files = sorted(glob.glob(os.path.join(args.input_dir, "trace_*.csv")))
    print(f"Fresh traces found: {len(trace_files)}")

    # Parse trace filenames
    import re
    TRACE_RE = re.compile(
        r"trace_(?P<task>.+)_s(?P<state>\d+)_w\d+_\d+_s20d_clean_seed(?P<seed>\d+)_job(?P<job>\d+)\.csv$"
    )

    trace_status = []; trace_candidates = []; failures = Counter()

    for i, fp in enumerate(trace_files):
        fname = os.path.basename(fp)
        m = TRACE_RE.match(fname)
        if not m:
            print(f"  SKIP (no regex match): {fname}")
            continue
        task = m.group("task"); state = int(m.group("state")); seed = int(m.group("seed"))
        tid = f"trace_{task}_s{state}_w0_10_s20d_clean_seed{seed}_job{m.group('job')}"
        print(f"  [{i+1}/{len(trace_files)}] {task}_s{state}")

        prov_ok = field_ok = oc_ok = remap_ok = True

        # Provenance
        if not os.path.isfile(fp):
            prov_ok = False
        else:
            try:
                with open(fp) as f: n_rows = sum(1 for _ in f) - 1
            except: n_rows = -1
            if n_rows <= 0: prov_ok = False

        if not prov_ok:
            failures["PROVENANCE_FAIL"] += 1; continue

        # Field validity
        fv = check_field_validity_v3(fp)
        field_ok = fv["field_validity_pass"]
        if not field_ok:
            failures["FIELD_VALIDITY_FAIL"] += 1; continue

        # Open convention
        oc = check_open_convention_v2(fp)
        oc_ok = oc["open_convention_pass"]
        if not oc_ok:
            failures["OPEN_CONVENTION_FAIL"] += 1; continue

        # RC1a remap
        rows, invariants, field_issues = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
        n_total_rows = len(rows) if rows else 0
        n_inv = len(invariants) if invariants else 0
        remap_ok = n_total_rows > 0 and n_inv == 0
        if not remap_ok:
            failures["RC1A_REMAP_FAIL"] += 1; continue

        # Teacher-P + candidates
        cap = check_teacher_p_privilege_capability(rows)
        grasp_priv = cap["grasp_privilege_valid"]
        legacy_first = teacher_privileged_critical_close_anchor(rows)
        r_anchor = teacher_rule_critical_close_anchor(rows)
        preds = rule_based_close_predictor(rows, horizon=PREDICTION_HORIZON,
                                            teacher_anchor=legacy_first if legacy_first >= 0 else -1)

        all_candidates = [p for p in preds if p.get("is_close_event_candidate")]
        tp_evidence = [evaluate_teacher_p_for_candidate(rows, c["step"]) for c in all_candidates]
        n_tp_qualifying = sum(1 for e in tp_evidence if e["teacher_p_criteria_pass"])

        if n_tp_qualifying == 1:
            unique_tp_step = next(e["candidate_step"] for e in tp_evidence if e["teacher_p_criteria_pass"])
        else:
            unique_tp_step = -1

        category = classify_trace_v3(
            prov_ok, field_ok, oc_ok, remap_ok,
            len(all_candidates), n_tp_qualifying, grasp_priv, "")
        failures[category] += 1

        trace_status.append({
            "trace_id": tid, "task_key": task, "state_id": state, "seed": seed,
            "provenance_pass": True, "field_validity_pass": True,
            "open_convention_pass": True, "remap_pass": True,
            "grasp_privilege_valid": grasp_priv,
            "n_close_candidates": len(all_candidates),
            "n_tp_qualifying_candidates": n_tp_qualifying,
            "teacher_p_step": unique_tp_step,
            "legacy_first_match_step": legacy_first,
            "teacher_r_step": r_anchor,
            "category": category, "failure_detail": "",
        })

        # Detailed candidates for eligible
        if n_tp_qualifying == 1 and len(all_candidates) >= 2:
            emittable_steps = {c["step"] for c in all_candidates if not c.get("abstain")}
            open_steps_from_rows = [
                int(r.get("step", i)) for i, r in enumerate(rows)
                if int(_safe_float(r.get("decoded_open_bool", 0))) == 1
            ]
            tp_ev_by_step = {e["candidate_step"]: e for e in tp_evidence}
            for idx, c in enumerate(all_candidates):
                step = c["step"]; ev = tp_ev_by_step.get(step, {})
                prev_close = all_candidates[idx - 1]["step"] if idx > 0 else None
                last_open = max([s for s in open_steps_from_rows if s < step]) if [s for s in open_steps_from_rows if s < step] else None
                sn = c.get("eef_speed_now", ""); sp = c.get("eef_speed_prev", "")
                decel = round(float(sn) - float(sp), 6) if sn != "" and sp != "" else ""
                trace_candidates.append({
                    "trace_id": tid, "task_key": task, "state_id": state,
                    "candidate_step": step, "candidate_index": idx,
                    "is_teacher_p": int(step == unique_tp_step),
                    "teacher_p_criteria_pass": int(ev.get("teacher_p_criteria_pass", False)),
                    "total_score": round(c.get("score", 0), 4),
                    "raw_crossing_bonus": c.get("raw_crossing_bonus", ""),
                    "close_streak_bonus": c.get("close_streak_bonus", ""),
                    "close_onset_qpos_bonus": c.get("close_onset_qpos_bonus", ""),
                    "eef_deceleration_bonus": c.get("eef_deceleration_bonus", ""),
                    "qpos_ready_bonus": c.get("qpos_ready_bonus", ""),
                    "eef_speed_now": sn, "eef_speed_prev": sp, "eef_deceleration_delta": decel,
                    "close_streak": c.get("close_streak_value", ""),
                    "raw_crossing": int(c.get("raw_open_to_close_crossing", 0)),
                    "close_onset": int(c.get("close_onset", 0)),
                    "qpos": c.get("qpos", ""),
                    "time_since_prev_close": step - prev_close if prev_close else "",
                    "time_since_last_open": step - last_open if last_open else "",
                    "selector_abstain_reason": c.get("abstain", ""),
                    "selector_emittable": int(step in emittable_steps),
                    "eef_to_obj_distance_at_close": ev.get("eef_to_obj_distance_at_close", ""),
                    "grasp_privilege_local_valid": int(ev.get("grasp_privilege_locally_valid", False)),
                    "max_cumulative_vertical_dz": ev.get("max_cumulative_vertical_dz", ""),
                    "max_sustained_vertical_frames": ev.get("max_sustained_vertical_frames", ""),
                    "eef_attachment_consistent": int(ev.get("eef_attachment_consistent", True)),
                    "tp_abstain_reason": ev.get("grasp_local_abstain_reason", ""),
                })

    # Write outputs
    sfields = list(trace_status[0].keys()) if trace_status else []
    if sfields:
        with open(out / "d2_fresh_trace_status.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sfields); w.writeheader(); w.writerows(trace_status)
    if trace_candidates:
        cfields = list(trace_candidates[0].keys())
        with open(out / "d2_fresh_close_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(trace_candidates)

    # Summary
    print(f"\n=== D2 FRESH ELIGIBILITY ===")
    for cat, count in failures.most_common():
        print(f"  {cat}: {count}")
    n_multi = sum(1 for r in trace_status if r["category"] == "ELIGIBLE_MULTI_CANDIDATE")
    n_single = sum(1 for r in trace_status if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE")
    print(f"\nEligible multi: {n_multi}  Single: {n_single}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
