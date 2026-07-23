#!/usr/bin/env python3
"""Validate H Student prediction bundle (P0-7, P0-9).

Strict validation before L3 replay: exact join with Teacher and runtime,
no silent skip of missing rows, full seal verification.
"""
from __future__ import annotations

import argparse, json, os, sys, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from factorized_phase_c_integrity import (
    FROZEN_SPLITS, HEADS, sha256_file, load_strict_json, load_strict_jsonl,
    verify_bundle_seal, seal_output_dir, extract_manifest_identities,
    verify_identity_closure, verify_step_closure, exact_three_way_join,
    validate_prediction_schema, validate_numeric_constraints,
    consume_sealed_receipt,
)

SELF_SHA = None


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-teacher-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-runtime-bundle-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--heldout-prediction-authorization-root", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    ap.add_argument("--runtime-source-root", type=Path, default=None)
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

    # P0-6: Consume authorization receipt as sealed root
    auth, _ = consume_sealed_receipt(args.heldout_prediction_authorization_root,
        "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1",
        "heldout_prediction_inference_authorized", True, "H_PRED_AUTH")

    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")

    h_pred_root = args.heldout_prediction_bundle_root.resolve()
    h_teacher_root = args.heldout_teacher_bundle_root.resolve()
    h_rt_root = args.heldout_runtime_bundle_root.resolve()
    h_pred_seal = verify_bundle_seal(h_pred_root, "H_PRED")
    h_teacher_seal = verify_bundle_seal(h_teacher_root, "H_TEACHER")
    h_rt_seal = verify_bundle_seal(h_rt_root, "H_RUNTIME")

    # Verify manifest binding
    if auth.get("authorized_h_manifest_sha256") != sha256_file(args.heldout_l3_manifest):
        raise SystemExit("H_MANIFEST_BINDING_MISMATCH")

    feature_sha = sha256_file(args.feature_order_contract)
    norm_sha = sha256_file(args.normalization_contract)

    all_errors: list[str] = []
    per_split: dict[str, Any] = {}
    h_pred_ready = True

    for sk in expected:
        h_ids = extract_manifest_identities(held_manifest, "heldout_l3", sk)

        pred_rows = load_strict_jsonl(h_pred_root / sk / "predictions.jsonl", f"H_PRED_{sk}")
        teacher_rows = load_strict_jsonl(h_teacher_root / sk / "factorized_teacher_v1.jsonl", f"H_TEACHER_{sk}")
        rt_rows = load_strict_jsonl(h_rt_root / sk / "runtime_scheduler_inputs.jsonl", f"H_RUNTIME_{sk}")

        pred_ids = {r["canonical_parent_key"] for r in pred_rows}
        try:
            verify_identity_closure(pred_ids, h_ids, "HELDOUT", sk)
            verify_step_closure(pred_rows, f"H_PRED_{sk}")
            validate_prediction_schema(pred_rows, f"H_PRED_{sk}")
            validate_numeric_constraints(pred_rows, f"H_PRED_{sk}")

            # P0-9: Exact 3-way join — NO silent "continue"
            _, _, _ = exact_three_way_join(pred_rows, teacher_rows, rt_rows, f"H_{sk}")

            # Verify checkpoint binding
            from factorized_phase_c_integrity import verify_checkpoint_from_manifest
            binding = pred_rows[0]
            verify_checkpoint_from_manifest(args.checkpoint_manifest_root, sk,
                                            binding.get("checkpoint_sha256", ""), f"H_PRED_{sk}")
            if binding.get("feature_order_sha256", "") != feature_sha:
                raise SystemExit(f"H_PRED_{sk}_FEATURE_MISMATCH")
            if binding.get("normalization_sha256", "") != norm_sha:
                raise SystemExit(f"H_PRED_{sk}_NORM_MISMATCH")

            per_split[sk] = {"pass": True, "h_identities": len(h_ids)}
        except SystemExit as e:
            all_errors.append(f"{sk}: {e}")
            per_split[sk] = {"pass": False, "error": str(e)}
            h_pred_ready = False

    receipt = {
        "schema": "FACTORIZED_HELDOUT_PREDICTION_VALIDATION_RECEIPT_V1",
        "validator_code_sha256": SELF_SHA,
        "status": "COMPLETE", "h_predictions_ready": h_pred_ready,
        "h_prediction_seal_sha256": h_pred_seal,
        "h_teacher_seal_sha256": h_teacher_seal,
        "h_runtime_seal_sha256": h_rt_seal,
        "n_errors": len(all_errors), "per_split": per_split,
    }
    if all_errors:
        receipt["errors"] = all_errors

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "FACTORIZED_HELDOUT_PREDICTION_VALIDATION_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"H Prediction Validation: ready={h_pred_ready} errors={len(all_errors)}")
    return 0 if h_pred_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
