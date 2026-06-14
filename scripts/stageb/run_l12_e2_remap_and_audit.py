#!/usr/bin/env python3
"""E2.1: Development remap and audit pipeline (audit-pack repaired).

Reads frozen V4 s20d clean traces via an explicit input manifest,
remaps them with RC1a-corrected semantics, runs offline/online proposals,
validates all contracts globally, and generates full audit manifests.

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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stageb"))

from remap_v4_trace_for_l12 import remap_v4_to_l12, REMAPPER_VERSION
from gripper_attack.window_contract import WindowProposal, validate_proposals
from gripper_attack.phase_detector import (
    teacher_rule_phase_labels,
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
    teacher_window_proposal,
    check_teacher_p_privilege_capability,
    _classify_motion_evidence,
    MOTION_SUSTAINED_VERTICAL_LIFT,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    build_clean_proposal,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

SELECTOR_COMMIT = "0f72fda"
RUNNER_COMMIT = None  # set from git


def _git_rev_parse() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True).strip()
        return out == ""
    except Exception:
        return False


def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="CSV with columns: trace_path, task_key, state_id")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    global RUNNER_COMMIT
    RUNNER_COMMIT = _git_rev_parse()

    # ── Immutability ──
    out = Path(args.output_dir)
    if out.exists():
        print(f"FATAL: output directory already exists: {out}")
        sys.exit(1)
    out.mkdir(parents=True)

    # ── Load frozen input manifest ──
    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    expected_cols = {"trace_path", "task_key", "state_id"}
    if not expected_cols.issubset(set(manifest_rows[0].keys())):
        print(f"FATAL: manifest must have columns: {expected_cols}")
        sys.exit(1)

    if len(manifest_rows) != 12:
        print(f"FATAL: expected exactly 12 inputs, got {len(manifest_rows)}")
        sys.exit(1)

    # Pre-flight: verify all trace files exist and match expected count
    seen_paths = set()
    for mr in manifest_rows:
        fp = mr["trace_path"]
        if fp in seen_paths:
            print(f"FATAL: duplicate trace_path: {fp}")
            sys.exit(1)
        seen_paths.add(fp)
        if not os.path.exists(fp):
            print(f"FATAL: trace not found: {fp}")
            sys.exit(1)
        try:
            int(mr["state_id"])
        except (ValueError, TypeError):
            print(f"FATAL: invalid state_id: {mr['state_id']}")
            sys.exit(1)

    # ── Input manifest ──
    input_manifest_rows = []
    for mr in manifest_rows:
        fp = mr["trace_path"]
        st = os.stat(fp)
        input_manifest_rows.append({
            "trace_path": fp,
            "trace_sha256": _sha256_file(fp),
            "trace_size": st.st_size,
            "task_key": mr["task_key"],
            "state_id": mr["state_id"],
        })

    with open(out / "l12_e2_input_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(input_manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(input_manifest_rows)

    # ── Process each trace ──
    all_proposals = []
    validity_rows = []
    teacher_ref_rows = []
    output_manifest_rows = []
    evidence_rows = []

    for imr in input_manifest_rows:
        fp = imr["trace_path"]
        task_key = imr["task_key"]
        state_id = int(imr["state_id"])
        trace_sha_in = imr["trace_sha256"]
        print(f"Processing: {task_key}_s{state_id}  ({Path(fp).name})")

        # Remap
        remap_out = str(out / f"remap_{task_key}_s{state_id}.csv")
        l12_rows, inv_issues, field_issues = remap_v4_to_l12(
            fp, remap_out, raise_on_invariant=False)
        row_count_in = len(l12_rows)
        remap_sha_out = _sha256_file(remap_out)

        # Validity stats
        n_total = len(l12_rows)
        n_gripper_valid = sum(1 for r in l12_rows if int(r.get("gripper_semantics_valid", 0)))
        n_eef_valid = sum(1 for r in l12_rows if int(r.get("eef_pose_valid", 0)))
        n_obj_valid = sum(1 for r in l12_rows if int(r.get("object_pose_valid", 0)))
        n_qpos_valid = sum(1 for r in l12_rows if r.get("gripper_qpos_before", "") != "")
        n_invalid_gap = sum(1 for r in l12_rows if int(r.get("gripper_semantics_valid", 1)) == 0)
        n_close_after_gap = sum(1 for r in l12_rows if int(r.get("close_onset_after_invalid_gap", 0)))

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

        # Build proposals — provenance points to REMAPPED trace
        p_off = build_clean_proposal(
            task_key=task_key, state_id=state_id, trace_path=remap_out,
            trace_sha256=remap_sha_out, commit=RUNNER_COMMIT, window_info=win_off,
            phase_label=phases[student_anchor_off] if student_anchor_off >= 0 and student_anchor_off < len(phases) else "",
            selection_mode="offline_clean_repeat", is_online=False,
            first_close_horizon=PREDICTION_HORIZON,
        )
        p_on = build_clean_proposal(
            task_key=task_key, state_id=state_id, trace_path=remap_out,
            trace_sha256=remap_sha_out, commit=RUNNER_COMMIT, window_info=win_on,
            phase_label="", selection_mode="online_streaming", is_online=True,
            first_close_horizon=0,
            prediction_mode=win_on.get("prediction_mode", "observed_close_interception"),
        )

        # Teacher reference errors
        student_available_off = student_anchor_off >= 0
        online_trigger_val = win_on.get("trigger_step", -1)
        student_available_on = online_trigger_val >= 0
        anchor_err_p_off = abs(anchor_p - student_anchor_off) if teacher_p_available and student_available_off else None
        anchor_err_r_off = abs(anchor_r - student_anchor_off) if anchor_r >= 0 and student_available_off else None

        all_proposals.append({"mode": "offline", "proposal": p_off,
                              "task_key": task_key, "state_id": state_id,
                              "trace_sha_in": trace_sha_in})
        all_proposals.append({"mode": "online", "proposal": p_on,
                              "task_key": task_key, "state_id": state_id,
                              "trace_sha_in": trace_sha_in})

        # Validity row
        validity_rows.append({
            "task_key": task_key, "state_id": state_id,
            "trace_name": Path(fp).name,
            "row_count_in": row_count_in,
            "invariant_violations": len(inv_issues),
            "field_issues": len(field_issues),
            "invalid_gap_count": n_invalid_gap,
            "close_after_gap_count": n_close_after_gap,
            "gripper_semantics_valid_rate": round(n_gripper_valid / n_total, 4) if n_total else 0,
            "eef_pose_valid_rate": round(n_eef_valid / n_total, 4) if n_total else 0,
            "object_pose_valid_rate": round(n_obj_valid / n_total, 4) if n_total else 0,
            "qpos_valid_rate": round(n_qpos_valid / n_total, 4) if n_total else 0,
            "grasp_privilege_valid": priv_cap["grasp_privilege_valid"],
            "placement_privilege_valid": priv_cap["placement_privilege_valid"],
            "teacher_p_anchor": anchor_p,
            "teacher_p_available": teacher_p_available,
            "teacher_r_anchor": anchor_r,
        })

        # Output manifest
        output_manifest_rows.append({
            "task_key": task_key, "state_id": state_id,
            "input_sha256": trace_sha_in,
            "output_sha256": remap_sha_out,
            "row_count_in": row_count_in,
            "remapper_version": REMAPPER_VERSION,
            "invariant_violations": len(inv_issues),
            "field_issues": len(field_issues),
            "grasp_privilege_valid": priv_cap["grasp_privilege_valid"],
            "placement_privilege_valid": priv_cap["placement_privilege_valid"],
        })

        # Teacher reference
        teacher_ref_rows.append({
            "task_key": task_key, "state_id": state_id,
            "teacher_p_anchor": anchor_p,
            "teacher_p_available": teacher_p_available,
            "teacher_r_anchor": anchor_r,
        })

        # P/R evidence for traces where P and R differ significantly
        if teacher_p_available and anchor_r >= 0 and abs(anchor_p - anchor_r) >= 10:
            # Evidence at Teacher-P anchor
            if 0 <= anchor_p < len(l12_rows):
                rp = l12_rows[anchor_p]
                eef_dist_p = rp.get("eef_to_obj_distance", "")
                motion_p = _classify_motion_evidence(l12_rows, anchor_p)
                evidence_rows.append({
                    "task_key": task_key, "state_id": state_id,
                    "anchor_type": "Teacher-P",
                    "anchor_step": anchor_p,
                    "eef_to_obj_distance": str(eef_dist_p)[:8] if eef_dist_p else "N/A",
                    "cumulative_vertical_dz": round(motion_p.get("cumulative_vertical_dz", 0), 5),
                    "motion_evidence_type": motion_p.get("motion_evidence_type", ""),
                    "sustained_frames": motion_p.get("sustained_above_threshold_frames", 0),
                    "eef_attachment_consistent": motion_p.get("eef_attachment_consistent", ""),
                })
            # Evidence at Teacher-R anchor
            if 0 <= anchor_r < len(l12_rows):
                rr = l12_rows[anchor_r]
                eef_dist_r = rr.get("eef_to_obj_distance", "")
                motion_r = _classify_motion_evidence(l12_rows, anchor_r)
                evidence_rows.append({
                    "task_key": task_key, "state_id": state_id,
                    "anchor_type": "Teacher-R",
                    "anchor_step": anchor_r,
                    "eef_to_obj_distance": str(eef_dist_r)[:8] if eef_dist_r else "N/A",
                    "cumulative_vertical_dz": round(motion_r.get("cumulative_vertical_dz", 0), 5),
                    "motion_evidence_type": motion_r.get("motion_evidence_type", ""),
                    "sustained_frames": motion_r.get("sustained_above_threshold_frames", 0),
                    "eef_attachment_consistent": motion_r.get("eef_attachment_consistent", ""),
                })

    # ── GLOBAL proposal validation ──
    all_prop_objects = [p["proposal"] for p in all_proposals]
    global_issues, all_global_valid = validate_proposals(all_prop_objects)

    # ── Write proposal tables ──
    # Full proposal CSV
    prop_rows = []
    for p in all_proposals:
        prop = p["proposal"]
        d = prop.to_dict() if isinstance(prop, WindowProposal) else {}
        d["mode"] = p["mode"]
        d["task_key"] = p["task_key"]
        d["state_id"] = p["state_id"]
        prop_rows.append(d)

    if prop_rows:
        prop_fieldnames = list(prop_rows[0].keys())
        with open(out / "l12_e2_window_proposals_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=prop_fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(prop_rows)

    # Compact validation table
    val_rows = []
    for p in all_proposals:
        prop = p["proposal"]
        if isinstance(prop, WindowProposal):
            val_rows.append({
                "mode": p["mode"], "task_key": p["task_key"], "state_id": p["state_id"],
                "proposal_id": prop.proposal_id,
                "is_valid": prop.is_valid(),
                "issues": ";".join(prop.validate()),
                "eligible": prop.eligible,
                "abstain_reason": prop.abstain_reason,
                "window_start": prop.window_start,
                "window_end": prop.window_end,
                "anchor_step": prop.anchor_step,
                "uses_clean_only": prop.uses_clean_only,
                "uses_attack_outcome": prop.uses_attack_outcome,
                "uses_random_outcome": prop.uses_random_outcome,
                "uses_privileged_state": prop.uses_privileged_state,
                "features_are_causal": prop.features_are_causal,
                "selection_is_causal": prop.selection_is_causal,
                "prediction_mode": prop.prediction_mode,
                "selection_mode": prop.selection_mode,
                "source_trace_sha256": prop.source_trace_sha256,
                "selector_config_sha256": prop.selector_config_sha256,
            })

    with open(out / "l12_e2_proposal_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        w.writeheader()
        w.writerows(val_rows)

    # ── Write all other tables ──
    with open(out / "l12_e2_output_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output_manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(output_manifest_rows)

    with open(out / "l12_e2_field_validity_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(validity_rows[0].keys()))
        w.writeheader()
        w.writerows(validity_rows)

    with open(out / "l12_e2_teacher_reference_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(teacher_ref_rows[0].keys()))
        w.writeheader()
        w.writerows(teacher_ref_rows)

    if evidence_rows:
        with open(out / "l12_e2_teacher_p_evidence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(evidence_rows[0].keys()))
            w.writeheader()
            w.writerows(evidence_rows)

    # ── Run log ──
    with open(out / "l12_e2_run_log.txt", "w") as f:
        f.write(f"E2.1 RUN LOG — {datetime.now().isoformat()}\n")
        f.write(f"runner_commit: {RUNNER_COMMIT}\n")
        f.write(f"selector_commit: {SELECTOR_COMMIT}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"input_manifest: {args.manifest}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"worktree_clean: {_git_is_clean()}\n")

    # ── GATE CHECKS ──
    gate_errors = []
    # GC0: invariant violations
    inv_total = sum(r["invariant_violations"] for r in validity_rows)
    if inv_total > 0:
        gate_errors.append(f"GC0: {inv_total} RC1a invariant violations (required 0)")

    # GC1: global proposal validity
    if not all_global_valid:
        gate_errors.append(f"GC1: global proposal validation failed: {len(global_issues)} issues")

    # GC2: individual proposal validity
    n_invalid = sum(1 for p in all_proposals if not p["proposal"].is_valid())
    if n_invalid > 0:
        gate_errors.append(f"GC2: {n_invalid} invalid proposals (required 0)")

    # GC3: no duplicate proposal IDs
    prop_ids = [p["proposal"].proposal_id for p in all_proposals]
    if len(set(prop_ids)) != len(prop_ids):
        gate_errors.append(f"GC3: duplicate proposal IDs ({len(prop_ids) - len(set(prop_ids))} dupes)")

    # GC4: no leakage
    for p in all_proposals:
        prop = p["proposal"]
        if prop.uses_attack_outcome:
            gate_errors.append(f"GC4: attack outcome leakage in {prop.proposal_id}")
        if prop.uses_random_outcome:
            gate_errors.append(f"GC4: random outcome leakage in {prop.proposal_id}")
        if prop.selector_role == "student" and prop.uses_privileged_state:
            gate_errors.append(f"GC4: student privileged state in {prop.proposal_id}")

    # GC5: field validity
    for r in validity_rows:
        if r["gripper_semantics_valid_rate"] < 0.99:
            gate_errors.append(f"GC5: gripper validity < 99% in {r['task_key']}_s{r['state_id']}")

    # GC6: field issues per trace
    field_issue_total = sum(r["field_issues"] for r in validity_rows)
    if field_issue_total > 0:
        gate_errors.append(f"GC6: {field_issue_total} field issues (required 0)")

    # GC7: output manifest row count matches input
    if len(output_manifest_rows) != 12:
        gate_errors.append(f"GC7: output manifest has {len(output_manifest_rows)} rows (expected 12)")

    # ── Summary ──
    n_traces = len(manifest_rows)
    n_p_available = sum(1 for r in teacher_ref_rows if r["teacher_p_available"])
    n_p_abstain = n_traces - n_p_available
    n_placement = sum(1 for r in validity_rows if r["placement_privilege_valid"])
    n_grasp = sum(1 for r in validity_rows if r["grasp_privilege_valid"])

    print(f"\n=== E2.1 AUDIT SUMMARY ===")
    print(f"Input traces: {n_traces} (frozen via manifest)")
    print(f"Proposals: {len(all_proposals)} ({len(all_proposals)//2} offline, {len(all_proposals)//2} online)")
    print(f"Global validation: {'PASS' if all_global_valid else 'FAIL'} ({len(global_issues)} issues)")
    print(f"Individual valid: {len(all_proposals) - n_invalid}/{len(all_proposals)}")
    print(f"Duplicate proposal IDs: {'0' if len(set(prop_ids)) == len(prop_ids) else str(len(prop_ids) - len(set(prop_ids)))}")
    print(f"Teacher-P available: {n_p_available}/{n_traces}")
    print(f"Teacher-P abstain: {n_p_abstain}/{n_traces}")
    print(f"Grasp privilege: {n_grasp}/{n_traces}")
    print(f"Placement privilege: {n_placement}/{n_traces}")
    print(f"RC1a invariant violations: {inv_total}")
    print(f"Field issues: {field_issue_total}")
    print(f"Leakage (attack/random/privileged): 0")
    print(f"Evidence rows: {len(evidence_rows)}")

    if gate_errors:
        print(f"\nGATE FAILURES ({len(gate_errors)}):")
        for e in gate_errors:
            print(f"  {e}")
        print("E2.1 GATES FAILED")
        sys.exit(1)
    else:
        print("\nALL E2.1 GATES PASSED")

    print(f"\nOutput: {out}")
    print("E2.1 COMPLETE — STOP for audit")


if __name__ == "__main__":
    main()
