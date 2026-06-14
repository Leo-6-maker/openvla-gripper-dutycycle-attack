#!/usr/bin/env python3
"""E2: Development remap and audit pipeline.

Reads V6 clean observer traces, remaps fields for L12 consumption,
runs offline/online proposals, validates all contracts, and generates
audit manifests.

CPU only. No GPU, no attack, no Layer3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.window_contract import WindowProposal, validate_proposals
from gripper_attack.phase_detector import (
    teacher_rule_phase_labels,
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
    teacher_window_proposal,
    check_teacher_p_privilege_capability,
    _safe_float, _field_is_valid,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    build_clean_proposal,
    _safe_float as _sf, _check_feature_validity,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

# Use the standalone V4 remapper for field-correct remapping
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stageb"))
from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION

SELECTOR_COMMIT = "0f72fda"


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _remap_v6_to_l12(v6_rows: list[dict], source_path: str) -> tuple[list[dict], dict]:
    """Remap V6 clean observer trace to L12-expected format.

    V6 format has: clean_gripper_env, clean_close, close_onset, close_streak,
    obj_x/y/z, eef_x/y/z, decoded_open_bool, gripper_qpos_before/after.

    Missing: clean_gripper_raw (derive from env), eef_to_obj_distance (compute),
    target_obj_x/y/z (NA), obj_to_target_distance (NA).
    """
    prev_clean_close = 0
    close_streak_counter = 0
    _prev_gripper_was_invalid = False
    invariant_issues = []
    field_issues = []
    invalid_gap_resets = 0
    close_after_gap_count = 0

    l12_rows = []
    for t, r in enumerate(v6_rows):
        # ── Gripper semantics ──
        env_raw = r.get("clean_gripper_env", "")
        decoded_raw = r.get("decoded_open_bool", "")

        gripper_semantics_valid = True
        try:
            clean_gripper_env = float(env_raw)
            decoded_open_bool = int(float(decoded_raw))
        except (ValueError, TypeError):
            field_issues.append(f"step {t}: gripper fields invalid")
            gripper_semantics_valid = False
            clean_gripper_env = 0.0
            decoded_open_bool = 1

        # RC1a invariant check
        if gripper_semantics_valid:
            if clean_gripper_env < -0.5 and decoded_open_bool != 1:
                invariant_issues.append(
                    f"step {t}: env={clean_gripper_env} OPEN but decoded={decoded_open_bool}")
            elif clean_gripper_env > 0.5 and decoded_open_bool != 0:
                invariant_issues.append(
                    f"step {t}: env={clean_gripper_env} CLOSE but decoded={decoded_open_bool}")

        # Derive raw proxy from env
        if gripper_semantics_valid:
            if clean_gripper_env < -0.5:
                raw_proxy = 1.0  # OPEN
            elif clean_gripper_env > 0.5:
                raw_proxy = 0.0  # CLOSE
            else:
                raw_proxy = None  # neutral
                gripper_semantics_valid = False
        else:
            raw_proxy = None

        # Close semantics
        close_onset_after_invalid_gap = False
        if raw_proxy is not None:
            clean_close = int(raw_proxy <= 0.5)
            onset = clean_close and not prev_clean_close
            if onset and _prev_gripper_was_invalid:
                close_onset_after_invalid_gap = True
                close_onset_val = 0
                close_streak_counter = 1
                close_after_gap_count += 1
            elif onset:
                close_onset_val = 1
                close_streak_counter += 1
            elif clean_close:
                close_onset_val = 0
                close_streak_counter += 1
            else:
                close_onset_val = 0
                close_streak_counter = 0
            _prev_gripper_was_invalid = False
        else:
            clean_close = None
            close_onset_val = None
            close_streak_counter = 0
            prev_clean_close = 0
            _prev_gripper_was_invalid = True
            invalid_gap_resets += 1

        if clean_close is not None:
            prev_clean_close = clean_close

        # Qpos
        qpos_before = _safe_float(r.get("gripper_qpos_before", "")) if r.get("gripper_qpos_before", "").strip() else None
        qpos_after = _safe_float(r.get("gripper_qpos_after", "")) if r.get("gripper_qpos_after", "").strip() else None

        # EEF
        eef_x = _safe_float(r.get("eef_x", "")) if r.get("eef_x", "").strip() else None
        eef_y = _safe_float(r.get("eef_y", "")) if r.get("eef_y", "").strip() else None
        eef_z = _safe_float(r.get("eef_z", "")) if r.get("eef_z", "").strip() else None
        eef_valid = all(v is not None for v in [eef_x, eef_y, eef_z])

        # Object
        obj_x = _safe_float(r.get("obj_x", "")) if r.get("obj_x", "").strip() else None
        obj_y = _safe_float(r.get("obj_y", "")) if r.get("obj_y", "").strip() else None
        obj_z = _safe_float(r.get("obj_z", "")) if r.get("obj_z", "").strip() else None
        obj_valid = all(v is not None for v in [obj_x, obj_y, obj_z])

        # Distance
        if eef_valid and obj_valid:
            eef_to_obj = ((eef_x-obj_x)**2 + (eef_y-obj_y)**2 + (eef_z-obj_z)**2)**0.5
        else:
            eef_to_obj = None

        row = {
            "step": t,
            "clean_gripper_env": clean_gripper_env if gripper_semantics_valid else "",
            "clean_gripper_raw_proxy": raw_proxy if raw_proxy is not None else "",
            "clean_gripper_raw_is_proxy": 1,
            "clean_gripper_raw_source": "reconstructed_from_env_rc1a_v6",
            "gripper_qpos_before": qpos_before if qpos_before is not None else "",
            "gripper_qpos_after": qpos_after if qpos_after is not None else "",
            "qpos_abs_before": abs(qpos_before) if qpos_before is not None else "",
            "qpos_abs_after": abs(qpos_after) if qpos_after is not None else "",
            "eef_x": eef_x if eef_x is not None else "",
            "eef_y": eef_y if eef_y is not None else "",
            "eef_z": eef_z if eef_z is not None else "",
            "eef_pose_valid": int(eef_valid),
            "eef_to_obj_distance": eef_to_obj if eef_to_obj is not None else "",
            "clean_close": clean_close if clean_close is not None else "",
            "close_onset": close_onset_val if close_onset_val is not None else "",
            "close_onset_after_invalid_gap": int(close_onset_after_invalid_gap),
            "close_streak": close_streak_counter if clean_close is not None else "",
            "decoded_open_bool": decoded_open_bool if gripper_semantics_valid else "",
            "gripper_semantics_valid": int(gripper_semantics_valid),
            "obj_x": obj_x if obj_x is not None else "",
            "obj_y": obj_y if obj_y is not None else "",
            "obj_z": obj_z if obj_z is not None else "",
            "object_pose_valid": int(obj_valid),
            "target_obj_x": "",
            "target_obj_y": "",
            "target_obj_z": "",
            "obj_to_target_distance": "",
            "placement_privilege_valid": 0,
            "success": int(r.get("success_primary", 0) or r.get("success_done", 0)),
            "done": int(r.get("success_done", 0)),
            "source_trace": source_path,
            "remapper_version": REMAPPER_VERSION,
        }
        l12_rows.append(row)

    audit = {
        "invariant_issues": invariant_issues,
        "field_issues": field_issues,
        "invalid_gap_resets": invalid_gap_resets,
        "close_after_gap_count": close_after_gap_count,
    }
    return l12_rows, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True, help="Directory with V6 clean observer CSVs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--commit", default=SELECTOR_COMMIT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Find traces ──
    trace_files = sorted(Path(args.trace_dir).glob("*.csv"))
    if not trace_files:
        print(f"No CSV traces found in {args.trace_dir}")
        return

    # ── Input manifest ──
    input_manifest_rows = []
    for fp in trace_files:
        input_manifest_rows.append({
            "source_trace_path": str(fp),
            "source_trace_sha256": _sha256_file(str(fp)),
            "trace_size": fp.stat().st_size,
            "trace_name": fp.name,
        })

    # ── Process each trace ──
    all_proposals = []
    all_remap_audits = []
    validity_rows = []
    teacher_ref_rows = []

    for fp in trace_files:
        trace_name = fp.name
        trace_sha = _sha256_file(str(fp))
        print(f"\nProcessing: {trace_name}")

        with open(fp, "r", newline="") as f:
            reader = csv.DictReader(f)
            v6_rows = list(reader)

        row_count_in = len(v6_rows)

        # Parse task from filename: trace_<task>_s<state>_...
        task = "unknown"
        state_id = 0
        parts = trace_name.replace(".csv", "").split("_")
        for i, p in enumerate(parts):
            if p.startswith("s") and p[1:].isdigit() and i > 0:
                task = parts[i - 1] if parts[i - 1] not in ("v6", "clean", "observer") else task
                state_id = int(p[1:])
                break
        # Better: find nearest non-keyword before sN
        for i, p in enumerate(parts):
            if p.startswith("s") and len(p) <= 3 and p[1:].isdigit():
                state_id = int(p[1:])
                # Find task name before this
                for j in range(i - 1, -1, -1):
                    cand = parts[j]
                    if cand not in ("v6", "clean", "observer", "seed0", "v2", "phase1", "rep0", "rep1"):
                        task = cand
                        break
                break

        # Remap using RC1a-corrected V4 remapper
        remap_out = str(out / f"remap_{trace_name}")
        l12_rows, inv_issues, field_issues_list = remap_v4_to_l12(
            str(fp), remap_out, raise_on_invariant=False)
        row_count_out = len(l12_rows)
        remap_audit = {
            "invariant_issues": inv_issues,
            "field_issues": field_issues_list,
            "invalid_gap_resets": sum(1 for r in l12_rows if int(r.get("close_onset_after_invalid_gap", 0))),
            "close_after_gap_count": sum(1 for r in l12_rows if int(r.get("close_onset_after_invalid_gap", 0))),
        }

        # Validity stats
        n_total = len(l12_rows)
        n_gripper_valid = sum(1 for r in l12_rows if int(r.get("gripper_semantics_valid", 0)))
        n_eef_valid = sum(1 for r in l12_rows if int(r.get("eef_pose_valid", 0)))
        n_obj_valid = sum(1 for r in l12_rows if int(r.get("object_pose_valid", 0)))
        n_qpos_valid = sum(1 for r in l12_rows if r.get("gripper_qpos_before", "") != "")

        # Privilege capability
        priv_cap = check_teacher_p_privilege_capability(l12_rows)

        # Teacher anchors
        anchor_r = teacher_rule_critical_close_anchor(l12_rows)
        anchor_p = teacher_privileged_critical_close_anchor(l12_rows)
        teacher_p_available = anchor_p >= 0

        # Student proposals
        horizon_anchor = anchor_p if teacher_p_available else -1
        phases = teacher_rule_phase_labels(l12_rows)

        preds_off = rule_based_close_predictor(l12_rows, horizon=PREDICTION_HORIZON, teacher_anchor=horizon_anchor)
        win_off = select_best_window(preds_off, WINDOW_LEN, PRE_OFFSET)
        student_anchor_off = win_off.get("anchor_step", -1)

        preds_on = rule_based_close_predictor(l12_rows, horizon=PREDICTION_HORIZON, teacher_anchor=horizon_anchor)
        win_on = select_online_trigger(preds_on, mode="close_interception")

        # Build proposals
        p_off = build_clean_proposal(
            task_key=task, state_id=state_id, trace_path=str(fp),
            trace_sha256=trace_sha, commit=args.commit, window_info=win_off,
            phase_label=phases[student_anchor_off] if student_anchor_off >= 0 and student_anchor_off < len(phases) else "",
            selection_mode="offline_clean_repeat", is_online=False,
            first_close_horizon=PREDICTION_HORIZON,
        )
        p_on = build_clean_proposal(
            task_key=task, state_id=state_id, trace_path=str(fp),
            trace_sha256=trace_sha, commit=args.commit, window_info=win_on,
            phase_label="", selection_mode="online_streaming", is_online=True,
            first_close_horizon=0,
            prediction_mode=win_on.get("prediction_mode", "observed_close_interception"),
        )

        # Validate
        issues_off, valid_off = validate_proposals([p_off]) if isinstance(p_off, WindowProposal) else ([], False)
        issues_on, valid_on = validate_proposals([p_on]) if isinstance(p_on, WindowProposal) else ([], False)
        if not isinstance(issues_off, tuple):
            issues_off = [issues_off] if isinstance(issues_off, str) else issues_off
            issues_on = [issues_on] if isinstance(issues_on, str) else issues_on
        valid_off = p_off.is_valid() if isinstance(p_off, WindowProposal) else False
        valid_on = p_on.is_valid() if isinstance(p_on, WindowProposal) else False
        issues_off_str = ";".join(p_off.validate()) if isinstance(p_off, WindowProposal) else ""
        issues_on_str = ";".join(p_on.validate()) if isinstance(p_on, WindowProposal) else ""

        # Teacher reference
        student_available_off = student_anchor_off >= 0
        online_trigger_val = win_on.get("trigger_step", -1)
        student_available_on = online_trigger_val >= 0
        anchor_err_p_off = abs(anchor_p - student_anchor_off) if teacher_p_available and student_available_off else None
        anchor_err_r_off = abs(anchor_r - student_anchor_off) if anchor_r >= 0 and student_available_off else None
        anchor_err_p_on = abs(anchor_p - online_trigger_val) if teacher_p_available and student_available_on else None

        all_proposals.append({
            "trace_name": trace_name, "task": task, "state_id": state_id,
            "mode": "offline", "proposal": p_off, "valid": valid_off, "issues": issues_off_str,
            "teacher_p_anchor": anchor_p, "teacher_r_anchor": anchor_r,
            "teacher_p_available": teacher_p_available,
            "student_anchor": student_anchor_off,
            "anchor_error_vs_p": anchor_err_p_off if anchor_err_p_off is not None else "",
            "anchor_error_vs_r": anchor_err_r_off if anchor_err_r_off is not None else "",
            "teacher_reference_unavailable": not teacher_p_available,
        })
        all_proposals.append({
            "trace_name": trace_name, "task": task, "state_id": state_id,
            "mode": "online", "proposal": p_on, "valid": valid_on, "issues": issues_on_str,
            "teacher_p_anchor": anchor_p, "teacher_r_anchor": anchor_r,
            "teacher_p_available": teacher_p_available,
            "student_anchor": online_trigger_val,
            "anchor_error_vs_p": anchor_err_p_on if anchor_err_p_on is not None else "",
            "anchor_error_vs_r": "",
            "teacher_reference_unavailable": not teacher_p_available,
        })

        # Validity row
        validity_rows.append({
            "trace_name": trace_name, "task": task, "state_id": state_id,
            "row_count_in": row_count_in, "row_count_out": row_count_out,
            "invariant_violations": len(remap_audit["invariant_issues"]),
            "field_issues": len(remap_audit["field_issues"]),
            "invalid_gap_resets": remap_audit["invalid_gap_resets"],
            "close_after_gap_count": remap_audit["close_after_gap_count"],
            "gripper_semantics_valid_rate": n_gripper_valid / n_total if n_total else 0,
            "eef_pose_valid_rate": n_eef_valid / n_total if n_total else 0,
            "object_pose_valid_rate": n_obj_valid / n_total if n_total else 0,
            "qpos_valid_rate": n_qpos_valid / n_total if n_total else 0,
            "grasp_privilege_valid": priv_cap["grasp_privilege_valid"],
            "placement_privilege_valid": priv_cap["placement_privilege_valid"],
            "teacher_p_anchor": anchor_p,
            "teacher_p_available": teacher_p_available,
            "teacher_r_anchor": anchor_r,
        })
        all_remap_audits.append(remap_audit)
        teacher_ref_rows.append({
            "trace_name": trace_name, "teacher_p_anchor": anchor_p,
            "teacher_p_available": teacher_p_available,
            "teacher_r_anchor": anchor_r,
        })

    # ── Write outputs ──
    # Input manifest
    with open(out / "l12_e2_input_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(input_manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(input_manifest_rows)

    # Validity summary
    with open(out / "l12_e2_field_validity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(validity_rows[0].keys()))
        w.writeheader()
        w.writerows(validity_rows)

    # Proposals
    prop_fieldnames = [
        "trace_name", "task", "state_id", "mode", "valid", "issues",
        "teacher_p_anchor", "teacher_r_anchor", "teacher_p_available",
        "student_anchor", "anchor_error_vs_p", "anchor_error_vs_r",
        "teacher_reference_unavailable",
    ]
    with open(out / "l12_e2_window_proposals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prop_fieldnames, extrasaction="ignore")
        w.writeheader()
        for p in all_proposals:
            d = {k: p.get(k, "") for k in prop_fieldnames}
            w.writerow(d)

    # Proposal validation
    val_fieldnames = [
        "trace_name", "mode", "proposal_id", "is_valid", "validation_issues",
        "selector_version", "uses_clean_only", "uses_attack_outcome",
        "uses_random_outcome", "uses_privileged_state",
        "features_are_causal", "selection_is_causal", "is_online",
        "selection_mode", "eligible", "abstain_reason",
    ]
    with open(out / "l12_e2_proposal_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=val_fieldnames, extrasaction="ignore")
        w.writeheader()
        for p in all_proposals:
            prop = p["proposal"]
            if isinstance(prop, WindowProposal):
                d = prop.to_dict()
                row = {
                    "trace_name": p["trace_name"], "mode": p["mode"],
                    "proposal_id": d.get("proposal_id", ""),
                    "is_valid": prop.is_valid(),
                    "validation_issues": ";".join(prop.validate()),
                    "selector_version": d.get("selector_version", ""),
                    "uses_clean_only": d.get("uses_clean_only", ""),
                    "uses_attack_outcome": d.get("uses_attack_outcome", ""),
                    "uses_random_outcome": d.get("uses_random_outcome", ""),
                    "uses_privileged_state": d.get("uses_privileged_state", ""),
                    "features_are_causal": d.get("features_are_causal", ""),
                    "selection_is_causal": d.get("selection_is_causal", ""),
                    "is_online": d.get("is_online", ""),
                    "selection_mode": d.get("selection_mode", ""),
                    "eligible": d.get("eligible", ""),
                    "abstain_reason": d.get("abstain_reason", ""),
                }
                w.writerow(row)

    # Teacher reference
    with open(out / "l12_e2_teacher_reference_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(teacher_ref_rows[0].keys()))
        w.writeheader()
        w.writerows(teacher_ref_rows)

    # ── Summary ──
    n_total = len(all_proposals)
    n_valid = sum(1 for p in all_proposals if p["valid"])
    n_p_abstain = sum(1 for p in all_proposals if p["teacher_p_available"] == False)
    n_p_available = n_total - n_p_abstain
    offline_valid = sum(1 for p in all_proposals if p["mode"] == "offline" and p["valid"])
    online_valid = sum(1 for p in all_proposals if p["mode"] == "online" and p["valid"])
    invariant_total = sum(r["invariant_violations"] for r in validity_rows)
    gap_total = sum(r["invalid_gap_resets"] for r in validity_rows)

    print(f"\n=== E2 AUDIT SUMMARY ===")
    print(f"Traces: {len(trace_files)}")
    print(f"Proposals: {n_total} ({n_total//2} offline, {n_total//2} online)")
    print(f"Valid proposals: {n_valid}/{n_total}")
    print(f"  Offline valid: {offline_valid}/{n_total//2}")
    print(f"  Online valid: {online_valid}/{n_total//2}")
    print(f"Teacher-P available: {n_p_available}/{n_total}")
    print(f"Teacher-P abstain: {n_p_abstain}/{n_total}")
    print(f"RC1a invariant violations: {invariant_total}")
    print(f"Invalid gap resets: {gap_total}")
    print(f"\nOutput: {out}")

    # GATE CHECKS
    gates_ok = True
    if invariant_total > 0:
        print(f"GATE FAIL: {invariant_total} RC1a invariant violations")
        gates_ok = False
    if n_valid < n_total:
        print(f"GATE FAIL: {n_total - n_valid} invalid proposals")
        gates_ok = False
    for p in all_proposals:
        prop = p["proposal"]
        if isinstance(prop, WindowProposal):
            d = prop.to_dict()
            if d.get("uses_attack_outcome"):
                print(f"GATE FAIL: attack outcome leakage in {p['trace_name']} {p['mode']}")
                gates_ok = False
            if d.get("uses_random_outcome"):
                print(f"GATE FAIL: random outcome leakage in {p['trace_name']} {p['mode']}")
                gates_ok = False
            if d.get("selector_role") == "student" and d.get("uses_privileged_state"):
                print(f"GATE FAIL: student privileged state in {p['trace_name']} {p['mode']}")
                gates_ok = False

    if gates_ok:
        print("ALL E2 GATES PASSED")
    else:
        print("E2 GATES FAILED — review before proceeding")

    print(f"\nOutput directory: {out}")
    print("E2 COMPLETE — STOP for audit")


if __name__ == "__main__":
    main()
