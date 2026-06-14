#!/usr/bin/env python3
"""E2.2: Hash-frozen development remap and audit pipeline.

Reads V4 s20d clean traces via hash-pinned input manifest,
remaps with RC1a-corrected semantics, runs offline/online proposals,
validates all contracts globally, and generates full audit manifests.

Hard gates: SHA match, row-count match, remapper version, invariants,
proposal validity, leakage, field issues.

CPU only. No GPU, no attack, no Layer3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    build_clean_proposal,
    WINDOW_LEN, PRE_OFFSET, PREDICTION_HORIZON,
)

SELECTOR_COMMIT = "0f72fda"


def _git_rev_parse() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=str(REPO_ROOT), text=True).strip()
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


def _count_csv_rows(path: str) -> int:
    with open(path, "r", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tracked-tables-dir", default=None,
                    help="Copy audit tables here for Git tracking")
    args = ap.parse_args()

    RUNNER_COMMIT = _git_rev_parse()
    RUNNER_SHORT = RUNNER_COMMIT[:7] if len(RUNNER_COMMIT) >= 7 else RUNNER_COMMIT
    START_TIME = datetime.now(timezone.utc)

    # ── Preflight ──
    errors = []

    if not _git_is_clean():
        errors.append("G1: tracked worktree is dirty")

    out = Path(args.output_dir)
    if out.exists():
        errors.append(f"FATAL: output directory exists: {out}")
    else:
        out.mkdir(parents=True)

    # Load manifest
    with open(args.manifest, "r", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    expected_cols = {"trace_path", "task_key", "state_id",
                     "expected_sha256", "expected_row_count"}
    if not expected_cols.issubset(set(manifest_rows[0].keys())):
        errors.append(f"Manifest must have columns: {expected_cols}")

    if len(manifest_rows) != 12:
        errors.append(f"G2: expected 12 manifest rows, got {len(manifest_rows)}")

    # Pre-flight: verify all inputs
    for mr in manifest_rows:
        fp = mr["trace_path"]
        if not os.path.exists(fp):
            errors.append(f"G3: trace not found: {fp}")
            continue
        actual_sha = _sha256_file(fp)
        expected_sha = mr["expected_sha256"]
        if actual_sha != expected_sha:
            errors.append(
                f"G3: SHA mismatch for {mr['task_key']}_s{mr['state_id']}: "
                f"expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
        actual_rows = _count_csv_rows(fp)
        expected_rows = int(mr["expected_row_count"])
        if actual_rows != expected_rows:
            errors.append(
                f"G4: row count mismatch for {mr['task_key']}_s{mr['state_id']}: "
                f"expected {expected_rows}, got {actual_rows}")

    if errors:
        for e in errors:
            print(e)
        print(f"\nPREFLIGHT FAILED ({len(errors)} errors)")
        sys.exit(1)

    print(f"=== E2.2 START ===\n")
    print(f"runner_commit: {RUNNER_COMMIT}")
    print(f"selector_commit: {SELECTOR_COMMIT}")
    print(f"remapper_version: {REMAPPER_VERSION}")
    print(f"output_dir: {out}")
    print()

    # ── Process traces ──
    all_proposals = []
    validity_rows = []
    teacher_ref_rows = []
    output_manifest_rows = []
    evidence_rows = []
    gate_errors = []

    for mr in manifest_rows:
        fp = mr["trace_path"]
        task_key = mr["task_key"]
        state_id = int(mr["state_id"])
        expected_row_count = int(mr["expected_row_count"])
        trace_sha_in = mr["expected_sha256"]

        print(f"  {task_key}_s{state_id} ...", end=" ", flush=True)

        # Remap
        remap_out = str(out / f"remap_{task_key}_s{state_id}.csv")
        try:
            l12_rows, inv_issues, field_issues = remap_v4_to_l12(
                fp, remap_out, raise_on_invariant=False)
        except Exception as e:
            gate_errors.append(f"Remap failed for {task_key}_s{state_id}: {e}")
            print("FAIL (remap)")
            continue

        output_row_count = len(l12_rows)
        remap_sha_out = _sha256_file(remap_out)

        # G5: row count match
        if output_row_count != expected_row_count:
            gate_errors.append(
                f"G5: row count mismatch {task_key}_s{state_id}: "
                f"input={expected_row_count} output={output_row_count}")

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

        # G6: remapper version
        actual_remapper = l12_rows[0].get("remapper_version", "") if l12_rows else ""
        if actual_remapper != REMAPPER_VERSION:
            gate_errors.append(
                f"G6: remapper version mismatch for {task_key}_s{state_id}: "
                f"expected {REMAPPER_VERSION}, got {actual_remapper}")

        # G7: invariant violations
        if inv_issues:
            gate_errors.append(
                f"G7: {len(inv_issues)} RC1a invariant violations in {task_key}_s{state_id}")

        # G8: field issues
        if field_issues:
            gate_errors.append(
                f"G8: {len(field_issues)} field issues in {task_key}_s{state_id}")

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

        all_proposals.append({"mode": "offline", "proposal": p_off,
                              "task_key": task_key, "state_id": state_id})
        all_proposals.append({"mode": "online", "proposal": p_on,
                              "task_key": task_key, "state_id": state_id})

        # Validity row
        validity_rows.append({
            "task_key": task_key, "state_id": state_id,
            "input_row_count": expected_row_count,
            "output_row_count": output_row_count,
            "row_count_match": output_row_count == expected_row_count,
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
            "input_row_count": expected_row_count,
            "output_row_count": output_row_count,
            "remapper_version": actual_remapper,
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

        # P/R evidence for divergent cases
        if teacher_p_available and anchor_r >= 0 and abs(anchor_p - anchor_r) >= 10:
            for label, astep in [("Teacher-P", anchor_p), ("Teacher-R", anchor_r)]:
                if 0 <= astep < len(l12_rows):
                    rp = l12_rows[astep]
                    motion = _classify_motion_evidence(l12_rows, astep)
                    evidence_rows.append({
                        "task_key": task_key, "state_id": state_id,
                        "anchor_type": label,
                        "anchor_step": astep,
                        "eef_to_obj_distance": str(rp.get("eef_to_obj_distance", ""))[:8],
                        "cumulative_vertical_dz": round(motion.get("cumulative_vertical_dz", 0), 5),
                        "cumulative_horizontal_dxy": round(motion.get("cumulative_horizontal_dxy", 0), 5),
                        "motion_evidence_type": motion.get("motion_evidence_type", ""),
                        "sustained_above_threshold_frames": motion.get("sustained_above_threshold_frames", 0),
                        "eef_attachment_consistent": motion.get("eef_attachment_consistent", ""),
                        "clean_close": rp.get("clean_close", ""),
                        "close_onset": rp.get("close_onset", ""),
                        "decoded_open_bool": rp.get("decoded_open_bool", ""),
                        "gripper_qpos_before": str(rp.get("gripper_qpos_before", ""))[:8],
                    })
        print("OK")

    # ── Global proposal validation ──
    all_prop_objects = [p["proposal"] for p in all_proposals]
    global_issues, all_global_valid = validate_proposals(all_prop_objects)
    if not all_global_valid:
        gate_errors.append(f"G9: global proposal validation failed: {len(global_issues)} issues")

    # Individual validation
    n_invalid = sum(1 for p in all_proposals if not p["proposal"].is_valid())
    if n_invalid > 0:
        gate_errors.append(f"G10: {n_invalid} invalid proposals")

    # Duplicate IDs
    prop_ids = [p["proposal"].proposal_id for p in all_proposals]
    n_dupes = len(prop_ids) - len(set(prop_ids))
    if n_dupes > 0:
        gate_errors.append(f"G11: {n_dupes} duplicate proposal IDs")

    # Leakage
    for p in all_proposals:
        prop = p["proposal"]
        if prop.uses_attack_outcome:
            gate_errors.append(f"G12: attack leakage in {prop.proposal_id}")
        if prop.uses_random_outcome:
            gate_errors.append(f"G12: random leakage in {prop.proposal_id}")
        if prop.selector_role == "student" and prop.uses_privileged_state:
            gate_errors.append(f"G12: student privileged state in {prop.proposal_id}")

    # Output manifest
    if len(output_manifest_rows) != 12:
        gate_errors.append(f"G13: output manifest has {len(output_manifest_rows)} rows (expected 12)")

    # ── Write tables ──
    def _write_csv(rows, name):
        if not rows:
            return
        path = out / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path

    # Full proposals
    if all_proposals:
        prop_rows = []
        for p in all_proposals:
            d = p["proposal"].to_dict()
            d["mode"] = p["mode"]
            d["task_key"] = p["task_key"]
            d["state_id"] = p["state_id"]
            prop_rows.append(d)
        _write_csv(prop_rows, "l12_e2_window_proposals_full.csv")

        # Compact validation
        val_rows = []
        for p in all_proposals:
            prop = p["proposal"]
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
                "prediction_mode": prop.prediction_mode,
                "selection_mode": prop.selection_mode,
                "features_are_causal": prop.features_are_causal,
                "selection_is_causal": prop.selection_is_causal,
                "uses_attack_outcome": prop.uses_attack_outcome,
                "uses_random_outcome": prop.uses_random_outcome,
                "uses_privileged_state": prop.uses_privileged_state,
                "selector_config_sha256": prop.selector_config_sha256,
                "source_trace_sha256": prop.source_trace_sha256,
            })
        _write_csv(val_rows, "l12_e2_proposal_validation.csv")

    _write_csv(validity_rows, "l12_e2_field_validity_summary.csv")
    _write_csv(output_manifest_rows, "l12_e2_output_manifest.csv")
    _write_csv(teacher_ref_rows, "l12_e2_teacher_reference_summary.csv")
    if evidence_rows:
        _write_csv(evidence_rows, "l12_e2_teacher_p_evidence.csv")

    # ── Run log ──
    END_TIME = datetime.now(timezone.utc)
    log_path = out / "l12_e2_run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"E2.2 RUN LOG\n")
        f.write(f"start: {START_TIME.isoformat()}\n")
        f.write(f"end: {END_TIME.isoformat()}\n")
        f.write(f"runner_commit: {RUNNER_COMMIT}\n")
        f.write(f"selector_commit: {SELECTOR_COMMIT}\n")
        f.write(f"remapper_version: {REMAPPER_VERSION}\n")
        f.write(f"remapper_file: scripts/stageb/remap_v4_trace_for_l12.py\n")
        f.write(f"input_manifest: {args.manifest}\n")
        f.write(f"output_dir: {args.output_dir}\n")
        f.write(f"worktree_clean: {_git_is_clean()}\n")
        f.write(f"python: {sys.version}\n")
        f.write(f"platform: {sys.platform}\n")
        f.write(f"input_trace_count: {len(manifest_rows)}\n")
        f.write(f"proposal_count: {len(all_proposals)}\n")
        f.write(f"teacher_p_available: {sum(1 for r in teacher_ref_rows if r['teacher_p_available'])}/{len(teacher_ref_rows)}\n")
        f.write(f"grasp_privilege: {sum(1 for r in validity_rows if r['grasp_privilege_valid'])}/{len(validity_rows)}\n")
        n_p_abstain = sum(1 for r in teacher_ref_rows if not r['teacher_p_available'])
        f.write(f"teacher_p_abstain: {n_p_abstain}/{len(teacher_ref_rows)}\n")
        f.write(f"global_issues: {len(global_issues)}\n")
        f.write(f"individual_invalid: {n_invalid}\n")
        f.write(f"duplicate_ids: {n_dupes}\n")
        if gate_errors:
            f.write(f"gate_errors: {len(gate_errors)}\n")
            for ge in gate_errors:
                f.write(f"  {ge}\n")
        else:
            f.write("ALL GATES PASSED\n")

    # ── Summary ──
    n_traces = len(manifest_rows)
    n_p_available = sum(1 for r in teacher_ref_rows if r["teacher_p_available"])

    print(f"\n=== E2.2 AUDIT SUMMARY ===")
    print(f"Traces: {n_traces} (hash-frozen)")
    print(f"Proposals: {len(all_proposals)}")
    print(f"Global validation: {'PASS' if all_global_valid else 'FAIL'}")
    print(f"Valid/Total: {len(all_proposals) - n_invalid}/{len(all_proposals)}")
    print(f"Duplicate IDs: {n_dupes}")
    print(f"Teacher-P available: {n_p_available}/{n_traces}")
    print(f"Grasp privilege: {sum(1 for r in validity_rows if r['grasp_privilege_valid'])}/{n_traces}")
    print(f"Placement privilege: {sum(1 for r in validity_rows if r['placement_privilege_valid'])}/{n_traces}")
    print(f"RC1a invariant violations: {sum(r['invariant_violations'] for r in validity_rows)}")
    print(f"Row count mismatches: {sum(1 for r in validity_rows if not r['row_count_match'])}")
    print(f"Evidence rows: {len(evidence_rows)}")

    if gate_errors:
        print(f"\nGATE FAILURES ({len(gate_errors)}):")
        for e in gate_errors:
            print(f"  {e}")
        print("E2.2 GATES FAILED")
        sys.exit(1)
    else:
        print("\nALL E2.2 GATES PASSED")

    # ── Copy to tracked tables dir ──
    if args.tracked_tables_dir:
        tracked = Path(args.tracked_tables_dir)
        tracked.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.csv"):
            shutil.copy2(f, tracked / f.name)
        shutil.copy2(log_path, tracked / "l12_e2_run_log.txt")
        print(f"\nTracked artifacts copied to: {tracked}")
    else:
        print("\nWARNING: --tracked-tables-dir not set; artifacts not in Git-tracked path")

    print(f"\nOutput: {out}")
    print("E2.2 COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
