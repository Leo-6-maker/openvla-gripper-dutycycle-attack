#!/usr/bin/env python3
"""D2.2: Fresh clean trace eligibility pipeline (REPAIRED).

Reads frozen canonical inventory. Processes every trace:
  provenance → field validity → open convention → RC1a remap →
  CLOSE candidate enumeration → per-candidate Teacher-P evidence →
  ambiguity detection → classification.

All 98 traces get a row in trace_status (failures retained).
Fail-closed provenance seal at startup.
time_since_* guards use `is not None`.
"""

import argparse, csv, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
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

# ── Frozen expected SHA256 ──
EXPECTED_MANIFEST_SHA = None  # computed at runtime from the passed manifest
EXPECTED_REMAPPER_SHA = "5d9cf327b25da459d399eda0c32527acea635daad5e1509b02a732154503c063"
EXPECTED_PHASE_DETECTOR_SHA = "f9cc7e90f415ee315723eba4738fd8f84f843b02f6dc2c123ae3570dc3a41329"
EXPECTED_SELECTOR_SHA = "81b510ec30716df11a89fbfb45194dceb1cfec09c1a2fd5d357a8a7448b4fa34"

REQUIRED_HEADER = [
    "obj_x", "obj_y", "obj_z", "eef_x", "eef_y", "eef_z",
    "clean_gripper_env", "decoded_open_bool", "gripper_qpos_before",
]

# Reuse core functions from E4C.2b
from run_l12_e4c2b_repair import (
    sha256_file, check_field_validity_v3, check_open_convention_v2,
    evaluate_teacher_p_for_candidate, classify_trace_v3,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="d2_final_trace_inventory.csv (98 rows)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # ── P0-3: Fail-closed provenance seal ──
    print("=== PROVENANCE SEAL ===")
    manifest_sha = sha256_file(args.manifest)
    print(f"  manifest: {manifest_sha[:16]}...")

    # Verify source files
    src_files = {
        "remapper": (os.path.join(PIPELINE_ROOT, "scripts", "stageb", "remap_v4_trace_for_l12.py"), EXPECTED_REMAPPER_SHA),
        "phase_detector": (os.path.join(PIPELINE_ROOT, "src", "gripper_attack", "phase_detector.py"), EXPECTED_PHASE_DETECTOR_SHA),
        "selector": (os.path.join(PIPELINE_ROOT, "src", "gripper_attack", "critical_close_selector.py"), EXPECTED_SELECTOR_SHA),
        "eligibility_runner": (os.path.join(PIPELINE_ROOT, "scripts", "stageb", "run_d2_fresh_eligibility.py"), None),
        "e4c2b_reference": (os.path.join(PIPELINE_ROOT, "scripts", "stageb", "run_l12_e4c2b_repair.py"), None),
    }
    seal_ok = True
    seal_artifacts = {"manifest_sha": manifest_sha}
    for name, (path, expected) in src_files.items():
        actual = sha256_file(path)
        seal_artifacts[f"{name}_sha"] = actual
        status = "OK" if expected is None or actual.lower() == expected.lower() else "MISMATCH"
        if status == "MISMATCH":
            seal_ok = False
        print(f"  {name}: {actual[:16]}... {status}")
    if not seal_ok:
        print("FATAL: provenance seal failed")
        sys.exit(1)
    print("  SEAL PASS\n")

    # ── P0-1: Read manifest (not glob) ──
    manifest_rows = list(csv.DictReader(open(args.manifest)))
    n_manifest = len(manifest_rows)
    print(f"Manifest traces: {n_manifest}")
    assert n_manifest == 98, f"Expected 98, got {n_manifest}"

    trace_status = []
    trace_candidates = []
    failures = Counter()

    for i, mr in enumerate(manifest_rows):
        fp = mr["source_path"]
        task = mr["task_key"]
        state = mr["state_id"]
        seed = mr.get("seed", "0")
        tid = mr["filename"].replace(".csv", "")
        tag = f"{task}_s{state}"

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_manifest}] {tag}")

        prov_ok = field_ok = oc_ok = remap_ok = True
        prov_detail = field_detail = oc_detail = remap_detail = ""

        # Gate 1: Provenance
        if not os.path.isfile(fp):
            prov_ok = False; prov_detail = "file_missing"
        else:
            actual_sha = sha256_file(fp)
            if actual_sha != mr["full_sha256"]:
                prov_ok = False; prov_detail = "sha_mismatch"
            with open(fp, "r") as f:
                n_rows = sum(1 for _ in f) - 1
            if n_rows != int(mr["row_count"]):
                prov_ok = False; prov_detail = "row_count_mismatch"
            if n_rows <= 0:
                prov_ok = False; prov_detail = "empty_file"

        if not prov_ok:
            failures["PROVENANCE_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": seed,
                "provenance_pass": False, "field_validity_pass": "", "open_convention_pass": "",
                "remap_pass": "", "grasp_privilege_valid": "", "n_close_candidates": -1,
                "n_tp_qualifying_candidates": -1, "teacher_p_step": -1,
                "category": "PROVENANCE_FAIL", "failure_detail": prov_detail,
            })
            continue

        # Gate 2: Field validity
        fv = check_field_validity_v3(fp)
        field_ok = fv["field_validity_pass"]
        if not field_ok:
            failures["FIELD_VALIDITY_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": seed,
                "provenance_pass": True, "field_validity_pass": False, "open_convention_pass": "",
                "remap_pass": "", "grasp_privilege_valid": "", "n_close_candidates": -1,
                "n_tp_qualifying_candidates": -1, "teacher_p_step": -1,
                "category": "FIELD_VALIDITY_FAIL", "failure_detail": str(fv["first_invalid_row"]),
            })
            continue

        # Gate 3: Open convention
        oc = check_open_convention_v2(fp)
        oc_ok = oc["open_convention_pass"]
        if not oc_ok:
            failures["OPEN_CONVENTION_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": seed,
                "provenance_pass": True, "field_validity_pass": True, "open_convention_pass": False,
                "remap_pass": "", "grasp_privilege_valid": "", "n_close_candidates": -1,
                "n_tp_qualifying_candidates": -1, "teacher_p_step": -1,
                "category": "OPEN_CONVENTION_FAIL", "failure_detail": str(oc["n_violations"]),
            })
            continue

        # Gate 4: RC1a remap
        rows, invariants, field_issues = remap_v4_to_l12(fp, "/dev/null", raise_on_invariant=False)
        n_total_rows = len(rows) if rows else 0
        n_inv = len(invariants) if invariants else 0
        remap_ok = n_total_rows > 0 and n_inv == 0
        if not remap_ok:
            failures["RC1A_REMAP_FAIL"] += 1
            trace_status.append({
                "trace_id": tid, "task_key": task, "state_id": state, "seed": seed,
                "provenance_pass": True, "field_validity_pass": True, "open_convention_pass": True,
                "remap_pass": False, "grasp_privilege_valid": "", "n_close_candidates": -1,
                "n_tp_qualifying_candidates": -1, "teacher_p_step": -1,
                "category": "RC1A_REMAP_FAIL", "failure_detail": f"rows={n_total_rows} inv={n_inv}",
            })
            continue

        # ── Teacher-P + candidate enumeration ──
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

        category = classify_trace_v3(prov_ok, field_ok, oc_ok, remap_ok,
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

        # ── Detailed candidates (for ALL traces, not just eligible) ──
        emittable_steps = {c["step"] for c in all_candidates if not c.get("abstain")}
        open_steps_from_rows = [
            int(r.get("step", i)) for i, r in enumerate(rows)
            if int(_safe_float(r.get("decoded_open_bool", 0))) == 1
        ]
        tp_ev_by_step = {e["candidate_step"]: e for e in tp_evidence}

        for idx, c in enumerate(all_candidates):
            step = c["step"]
            ev = tp_ev_by_step.get(step, {})
            prev_close = all_candidates[idx - 1]["step"] if idx > 0 else None
            candidates_before = [s for s in open_steps_from_rows if s < step]
            last_open = max(candidates_before) if candidates_before else None

            # P0-4: fix time_since_* — use `is not None`
            time_prev = step - prev_close if prev_close is not None else ""
            time_open = step - last_open if last_open is not None else ""

            sn = c.get("eef_speed_now", "")
            sp = c.get("eef_speed_prev", "")
            decel = ""
            if sn != "" and sp != "":
                try: decel = round(float(sn) - float(sp), 6)
                except: pass

            is_tp = int(step == unique_tp_step) if unique_tp_step >= 0 else 0

            trace_candidates.append({
                "trace_id": tid, "task_key": task, "state_id": state,
                "candidate_step": step, "candidate_index": idx,
                "is_teacher_p": is_tp,
                "teacher_p_criteria_pass": int(ev.get("teacher_p_criteria_pass", False)),
                "distance_to_teacher_p": step - unique_tp_step if unique_tp_step >= 0 else "",
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
                "time_since_prev_close": time_prev,
                "time_since_last_open": time_open,
                "selector_abstain_reason": c.get("abstain", ""),
                "selector_emittable": int(step in emittable_steps),
                "eef_to_obj_distance_at_close": ev.get("eef_to_obj_distance_at_close", ""),
                "grasp_privilege_local_valid": int(ev.get("grasp_privilege_locally_valid", False)),
                "max_cumulative_vertical_dz": ev.get("max_cumulative_vertical_dz", ""),
                "max_sustained_vertical_frames": ev.get("max_sustained_vertical_frames", ""),
                "eef_attachment_consistent": int(ev.get("eef_attachment_consistent", True)),
                "tp_abstain_reason": ev.get("grasp_local_abstain_reason", ""),
            })

    # ── P0-2: Assert denominator = 98 ──
    assert len(trace_status) == n_manifest, \
        f"trace_status={len(trace_status)} != manifest={n_manifest}"

    # ── Write outputs ──
    sfields = list(trace_status[0].keys())
    with open(out / "d2_fresh_trace_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields); w.writeheader(); w.writerows(trace_status)

    if trace_candidates:
        cfields = list(trace_candidates[0].keys())
        with open(out / "d2_fresh_close_candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfields); w.writeheader(); w.writerows(trace_candidates)

    # Task summary
    tasks_seen = set(r["task_key"] for r in trace_status)
    task_rows = []
    for t in sorted(tasks_seen):
        trs = [r for r in trace_status if r["task_key"] == t]
        task_rows.append({
            "task_key": t, "n_total": len(trs),
            "n_tp_qualifying_0": sum(1 for r in trs if int(r["n_tp_qualifying_candidates"]) == 0),
            "n_tp_qualifying_1": sum(1 for r in trs if int(r["n_tp_qualifying_candidates"]) == 1),
            "n_tp_qualifying_gt1": sum(1 for r in trs if int(r["n_tp_qualifying_candidates"]) > 1),
            "n_eligible_multi": sum(1 for r in trs if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"),
            "n_eligible_single": sum(1 for r in trs if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE"),
            "n_tp_unavailable": sum(1 for r in trs if r["category"] == "TEACHER_P_UNAVAILABLE"),
            "n_tp_ambiguous": sum(1 for r in trs if r["category"] == "TEACHER_P_AMBIGUOUS"),
            "n_no_candidate": sum(1 for r in trs if r["category"] == "NO_CLOSE_CANDIDATE"),
        })
    with open(out / "d2_fresh_task_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys())); w.writeheader(); w.writerows(task_rows)

    # Failure taxonomy
    tax_rows = [{"category": cat, "count": count} for cat, count in failures.most_common()]
    with open(out / "d2_fresh_failure_taxonomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "count"]); w.writeheader(); w.writerows(tax_rows)

    # Output hashes
    hash_rows = []
    for fname in sorted(os.listdir(str(out))):
        fpath = out / fname
        if fpath.suffix == ".csv":
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
            hash_rows.append({"file": fname, "sha256": h.hexdigest()})
    with open(out / "d2_fresh_output_hashes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "sha256"]); w.writeheader(); w.writerows(hash_rows)

    # Summary
    print(f"\n=== D2 FRESH ELIGIBILITY ({len(trace_status)}/{n_manifest} traces) ===")
    for cat, count in failures.most_common():
        print(f"  {cat}: {count}")
    n_multi = sum(1 for r in trace_status if r["category"] == "ELIGIBLE_MULTI_CANDIDATE")
    n_single = sum(1 for r in trace_status if r["category"] == "ELIGIBLE_SINGLE_CANDIDATE")
    n_ambig = sum(1 for r in trace_status if r["category"] == "TEACHER_P_AMBIGUOUS")
    tasks_rep = len(set(r["task_key"] for r in trace_status if r["category"] == "ELIGIBLE_MULTI_CANDIDATE"))
    print(f"Eligible multi: {n_multi}  Single: {n_single}  Ambiguous: {n_ambig}")
    print(f"Tasks represented: {tasks_rep}")
    print(f"Candidates: {len(trace_candidates)}")
    print(f"Output: {out}")
    print("FRESH_ELIGIBILITY_COMPLETE")


if __name__ == "__main__":
    main()
