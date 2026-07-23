#!/usr/bin/env python3
"""Authorize H Student prediction inference (P0-7, P0-8).

Separate gate from heldout-L3 evaluation. Authorizes EXACTLY ONE H prediction run.
Uses atomic claim root for true single-use (P0-8).
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from factorized_phase_c_integrity import (
    FROZEN_SPLITS, sha256_file, load_strict_json, verify_bundle_seal,
    seal_output_dir, extract_manifest_identities, consume_sealed_receipt,
    claim_atomic_root,
)

SELF_SHA = None


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    # Sealed receipt roots (P0-6)
    ap.add_argument("--phase-b-validation-root", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-validation-root", type=Path, required=True)
    # Identity manifests
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    # Checkpoint
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    # H resources (must exist — Teacher and runtime bundles)
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-runtime-bundle-root", type=Path, required=True)
    # Output roots
    ap.add_argument("--authorized-h-prediction-output-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]
    expected_set = set(expected)
    if len(expected) != 12 or len(expected_set) != 12 or expected_set != FROZEN_SPLITS:
        raise SystemExit("SPLIT_ENFORCEMENT: requires exactly 12 splits")

    # P0-6: Consume all sealed receipt roots
    phase_b, _ = consume_sealed_receipt(args.phase_b_validation_root,
        "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "heldout_l3_data_ready", True, "PHASE_B")
    consume_sealed_receipt(args.cp_prediction_validation_root,
        "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1", "cp_predictions_ready", True, "CP_VAL")
    consume_sealed_receipt(args.calibrator_freeze_validation_root,
        "FACTORIZED_CALIBRATOR_FREEZE_VALIDATION_V1", "status", "PASS", "CAL_FREEZE_VAL")
    consume_sealed_receipt(args.scheduler_freeze_validation_root,
        "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1", "status", "PASS", "SCHED_FREEZE_VAL")

    # Load freeze contracts
    cf_root = args.calibrator_freeze_root.resolve()
    sf_root = args.scheduler_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    verify_bundle_seal(sf_root, "SCHED_FREEZE")
    cf = load_strict_json(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CAL_FREEZE")
    sf = load_strict_json(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json", "SCHED_FREEZE")

    # Verify flags
    if cf.get("attack_authorized") is not False:
        raise SystemExit("CAL_FREEZE_ATTACK_AUTHORIZED")
    if sf.get("attack_authorized") is not False:
        raise SystemExit("SCHED_FREEZE_ATTACK_AUTHORIZED")
    if sf.get("status") != "COMPLETE":
        raise SystemExit("SCHED_FREEZE_NOT_COMPLETE")
    ws = sf.get("worst_split_false_start")
    if ws is None or ws > 0.10:
        raise SystemExit(f"SCHED_FREEZE_WORST_FALSE_START: {ws}")

    # Verify H resources
    h_teacher_root = args.heldout_teacher_bundle_root.resolve()
    h_rt_root = args.heldout_runtime_bundle_root.resolve()
    h_teacher_seal = verify_bundle_seal(h_teacher_root, "H_TEACHER")
    h_rt_seal = verify_bundle_seal(h_rt_root, "H_RUNTIME")

    # Load manifests
    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CAL_MANIFEST")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")

    # P0-8: Verify H prediction output root does not pre-exist
    h_pred_out = args.authorized_h_prediction_output_root.resolve()
    if h_pred_out.exists():
        raise SystemExit(f"H_PRED_OUTPUT_EXISTS: {h_pred_out}")
    # P0-8: Atomic claim root
    claim_root = h_pred_out.with_name(f"{h_pred_out.name}_CLAIM")

    # Verify identity closure C∩P∩H = ∅
    errors: list[str] = []
    for sk in expected:
        c_ids = extract_manifest_identities(cal_manifest, "calibrator_fit", sk)
        p_ids = extract_manifest_identities(pol_manifest, "policy_selection", sk)
        h_ids = extract_manifest_identities(held_manifest, "heldout_l3", sk)
        if c_ids & h_ids:
            errors.append(f"C_H_OVERLAP: {sk}")
        if p_ids & h_ids:
            errors.append(f"P_H_OVERLAP: {sk}")
        if c_ids & p_ids:
            errors.append(f"C_P_OVERLAP: {sk}")

    all_pass = len(errors) == 0

    # Build authorization receipt BEFORE writing claim (claim is a separate output)
    receipt = {
        "schema": "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1",
        "authorization_code_sha256": SELF_SHA,
        "status": "AUTHORIZED" if all_pass else "HOLD",
        "heldout_prediction_inference_authorized": all_pass,
        "authorization_scope": "EXACTLY_ONE_RUN",
        "authorized_h_manifest_sha256": sha256_file(args.heldout_l3_manifest),
        "authorized_checkpoint_manifest_root": str(args.checkpoint_manifest_root.resolve()),
        "authorized_calibrator_freeze_sha256": sha256_file(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"),
        "authorized_scheduler_freeze_sha256": sha256_file(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"),
        "authorized_h_teacher_bundle_sha256": h_teacher_seal,
        "authorized_h_runtime_bundle_sha256": h_rt_seal,
        "authorized_h_prediction_output_root": str(h_pred_out),
        "authorized_claim_root": str(claim_root),
        "attack_authorized": False,
        "canary_authorized": False,
        "errors": errors,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging)
    os.replace(staging, out_root)

    # P0-8: Atomic claim
    if all_pass:
        claim_atomic_root(claim_root, sha256_file(
            out_root / "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1.json"), "H_PRED")

    print(f"H Prediction Authorization: {'AUTHORIZED' if all_pass else 'HOLD'}")
    for e in errors:
        print(f"  ERROR: {e}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
