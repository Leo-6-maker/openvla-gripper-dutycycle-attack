#!/usr/bin/env python3
"""Authorize heldout-L3 evaluation (P0-7).

Separate gate from H prediction authorization. This authorizes the L3 evaluation
AFTER H predictions are validated. H prediction output root and evaluation
output root are separate.
"""
from __future__ import annotations

import argparse, json, os, sys, time, uuid
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
    ap.add_argument("--heldout-prediction-authorization-root", type=Path, required=True)
    ap.add_argument("--heldout-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-validation-root", type=Path, required=True)
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--authorized-l3-evaluation-output-root", type=Path, required=True)
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
        raise SystemExit("SPLIT_ENFORCEMENT")

    # P0-6: Consume all sealed receipt roots
    h_pred_auth, _ = consume_sealed_receipt(args.heldout_prediction_authorization_root,
        "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1",
        "heldout_prediction_inference_authorized", True, "H_PRED_AUTH")
    consume_sealed_receipt(args.heldout_prediction_validation_root,
        "FACTORIZED_HELDOUT_PREDICTION_VALIDATION_RECEIPT_V1",
        "h_predictions_ready", True, "H_PRED_VAL")
    consume_sealed_receipt(args.calibrator_freeze_validation_root,
        "FACTORIZED_CALIBRATOR_FREEZE_VALIDATION_V1", "status", "PASS", "CAL_FREEZE_VAL")
    consume_sealed_receipt(args.scheduler_freeze_validation_root,
        "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1", "status", "PASS", "SCHED_FREEZE_VAL")

    cf_root = args.calibrator_freeze_root.resolve()
    sf_root = args.scheduler_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    verify_bundle_seal(sf_root, "SCHED_FREEZE")
    cf = load_strict_json(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CAL_FREEZE")
    sf = load_strict_json(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json", "SCHED_FREEZE")

    if cf.get("attack_authorized") is not False:
        raise SystemExit("CAL_FREEZE_ATTACK_AUTHORIZED")
    if sf.get("attack_authorized") is not False:
        raise SystemExit("SCHED_FREEZE_ATTACK_AUTHORIZED")
    if sf.get("status") != "COMPLETE":
        raise SystemExit("SCHED_FREEZE_NOT_COMPLETE")
    ws = sf.get("worst_split_false_start")
    if ws is None or ws > 0.10:
        raise SystemExit(f"SCHED_FREEZE_WORST_FALSE_START: {ws}")

    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CAL_MANIFEST")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")

    errors: list[str] = []
    for sk in expected:
        c_ids = extract_manifest_identities(cal_manifest, "calibrator_fit", sk)
        p_ids = extract_manifest_identities(pol_manifest, "policy_selection", sk)
        h_ids = extract_manifest_identities(held_manifest, "heldout_l3", sk)
        if c_ids & h_ids: errors.append(f"C_H_OVERLAP: {sk}")
        if p_ids & h_ids: errors.append(f"P_H_OVERLAP: {sk}")

    eval_out = args.authorized_l3_evaluation_output_root.resolve()
    if eval_out.exists():
        errors.append(f"EVAL_OUTPUT_EXISTS: {eval_out}")

    all_pass = len(errors) == 0

    receipt = {
        "schema": "FACTORIZED_HELDOUT_L3_EVALUATION_AUTHORIZATION_RECEIPT_V1",
        "authorization_code_sha256": SELF_SHA,
        "status": "AUTHORIZED" if all_pass else "HOLD",
        "heldout_l3_evaluation_authorized": all_pass,
        "authorized_h_manifest_sha256": sha256_file(args.heldout_l3_manifest),
        "authorized_calibrator_freeze_sha256": sha256_file(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"),
        "authorized_scheduler_freeze_sha256": sha256_file(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"),
        "authorized_l3_evaluation_output_root": str(eval_out),
        "attack_authorized": False,
        "errors": errors,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "FACTORIZED_HELDOUT_L3_EVALUATION_AUTHORIZATION_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"Heldout-L3 Evaluation Authorization: {'AUTHORIZED' if all_pass else 'HOLD'}")
    for e in errors: print(f"  ERROR: {e}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
