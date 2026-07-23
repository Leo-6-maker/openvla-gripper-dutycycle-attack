#!/usr/bin/env python3
"""Freeze Factorized calibrators (P0-6: sealed receipt consumption).

Reads only calibrator-fit identities (C). Consumes sealed receipt roots.
Uses shared integrity module for all strict loading.
"""
from __future__ import annotations

import argparse, json, os, sys, time, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from factorized_phase_c_integrity import (
    FROZEN_SPLITS, HEADS, sha256_file, load_strict_json, load_strict_jsonl,
    verify_bundle_seal, seal_output_dir, extract_manifest_identities,
    verify_identity_closure, exact_three_way_join, consume_sealed_receipt,
)
from fit_factorized_calibrators import (
    fit_raw, fit_intercept, fit_platt, sigmoid,
)

SELF_SHA = None


def fit_all_methods(records: list[dict[str, Any]], head: str) -> list[dict[str, Any]]:
    return [fit_raw(records, head), fit_intercept(records, head), fit_platt(records, head)]


def select_method(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in results if r.get("method_valid")]
    if not candidates:
        return results[0]
    def priority(r: dict[str, Any]) -> tuple[int, int, int]:
        m, np, nn = r.get("method", ""), r.get("n_fit_pos", 0), r.get("n_fit_neg", 0)
        if m == "PLATT" and np >= 10 and nn >= 10: return 0, np + nn, 0
        if m == "INTERCEPT_ONLY" and np >= 5 and nn >= 5: return 1, np + nn, 0
        if m == "RAW": return 2, np + nn, 0
        return 3, 0, 0
    candidates.sort(key=priority)
    return candidates[0]


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--calibration-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--calibration-teacher-bundle-root", type=Path, required=True)
    # P0-6: sealed receipt roots
    ap.add_argument("--phase-b-validation-root", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--checkpoint-training-ledger", type=Path, required=True)
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]

    # P0-6: Consume sealed receipts
    phase_b, _ = consume_sealed_receipt(args.phase_b_validation_root,
        "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "cp_inference_authorized", True, "PHASE_B")
    consume_sealed_receipt(args.cp_prediction_validation_root,
        "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1", "cp_predictions_ready", True, "CP_VAL")

    fit_manifest = load_strict_json(args.calibrator_fit_manifest, "FIT_MANIFEST")
    training_ledger = load_strict_json(args.checkpoint_training_ledger, "TRAINING_LEDGER")
    feature_sha = sha256_file(args.feature_order_contract)
    norm_sha = sha256_file(args.normalization_contract)

    c_pred_root = args.calibration_prediction_bundle_root.resolve()
    c_teacher_root = args.calibration_teacher_bundle_root.resolve()
    c_pred_seal = verify_bundle_seal(c_pred_root, "C_PRED")
    c_teacher_seal = verify_bundle_seal(c_teacher_root, "C_TEACHER")

    per_split_calibrators: dict[str, dict[str, Any]] = {}
    fit_metrics: dict[str, Any] = {}
    identity_receipt: dict[str, dict[str, Any]] = {}
    all_valid = True
    hold_reasons: list[str] = []

    for sk in expected:
        c_ids = extract_manifest_identities(fit_manifest, "calibrator_fit", sk)
        t_ids = extract_manifest_identities(training_ledger, "checkpoint_training", sk)
        if c_ids & t_ids:
            raise SystemExit(f"CAL_TRAIN_OVERLAP: {sk}")

        pred_rows = load_strict_jsonl(c_pred_root / sk / "predictions.jsonl", f"C_PRED_{sk}")
        teacher_rows = load_strict_jsonl(c_teacher_root / sk / "factorized_teacher_v1.jsonl", f"C_TEACHER_{sk}")

        pred_ids = {r["canonical_parent_key"] for r in pred_rows}
        teacher_ids = {r["canonical_parent_key"] for r in teacher_rows}
        verify_identity_closure(pred_ids, c_ids, "CALIBRATION", sk)
        verify_identity_closure(teacher_ids, c_ids, "CAL_TEACHER", sk)

        pred_by_key, teacher_by_key, _ = exact_three_way_join(pred_rows, teacher_rows, None, f"CAL_{sk}")

        cal_records: list[dict[str, Any]] = []
        for (ep, step), p_row in pred_by_key.items():
            t_row = teacher_by_key[(ep, step)]
            record = {"episode": ep, "step": step}
            for head in HEADS:
                record[f"{head}_logit"] = p_row[f"{head}_logit"]
                record[f"{head}_probability"] = p_row[f"{head}_probability"]
                record[f"{head}_known_mask"] = t_row.get(
                    f"{'grasp_established' if head == 'grasp' else 'manipulation_active' if head == 'manipulation' else 'release_or_instability'}_known_mask", False)
                record[f"{head}_target"] = t_row.get(
                    f"{'grasp_established' if head == 'grasp' else 'manipulation_active' if head == 'manipulation' else 'release_or_instability'}", False)
            cal_records.append(record)

        split_result: dict[str, Any] = {}
        for head in HEADS:
            all_results = fit_all_methods(cal_records, head)
            selected = select_method(all_results)
            for r in all_results:
                r["checkpoint_sha256"] = pred_rows[0].get("checkpoint_sha256", "")
                r["split"] = sk
            selected["all_candidates"] = all_results
            if not selected.get("method_valid"):
                all_valid = False
                hold_reasons.append(f"{sk}/{head}: {selected.get('method_status')}")
            split_result[head] = selected

        per_split_calibrators[sk] = split_result
        fit_metrics[sk] = {}
        for head in HEADS:
            sel = split_result[head]
            fit_metrics[sk][head] = {
                "method": sel["method"], "method_valid": sel.get("method_valid", False),
                "n_fit_pos": sel.get("n_fit_pos", 0), "n_fit_neg": sel.get("n_fit_neg", 0),
                "a": sel.get("a", 1.0), "b": sel.get("b", 0.0),
                "method_status": sel.get("method_status", "HOLD"),
            }

    freeze_contract = {
        "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
        "status": "COMPLETE" if all_valid else "HOLD_INSUFFICIENT_DATA",
        "all_heads_frozen": all_valid,
        "freeze_bindings": {
            "phase_b_validation_seal_sha256": sha256_file(args.phase_b_validation_root / "SHA256SUMS"),
            "cp_prediction_validation_seal_sha256": sha256_file(args.cp_prediction_validation_root / "SHA256SUMS"),
            "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
            "calibration_prediction_bundle_sha256": c_pred_seal,
            "calibration_teacher_bundle_sha256": c_teacher_seal,
            "feature_order_sha256": feature_sha,
            "normalization_sha256": norm_sha,
            "freeze_code_sha256": SELF_SHA,
        },
        "selection_rule": "PLATT(n>=10)→INTERCEPT(n>=5)→RAW; deterministic tie-break by n_fit_pos+n_fit_neg",
        "per_split": {},
        "attack_authorized": False, "heldout_l3_authorized": False,
        "full_fit_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for sk in expected:
        freeze_contract["per_split"][sk] = {}
        for head in HEADS:
            sel = per_split_calibrators[sk][head]
            freeze_contract["per_split"][sk][head] = {
                "method": sel["method"], "a": sel.get("a", 1.0), "b": sel.get("b", 0.0),
                "method_valid": sel.get("method_valid", False),
                "n_fit_pos": sel.get("n_fit_pos", 0), "n_fit_neg": sel.get("n_fit_neg", 0),
            }

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, data in [
        ("FACTORIZED_CALIBRATOR_FREEZE_V1.json", freeze_contract),
        ("FACTORIZED_CALIBRATOR_FIT_METRICS_V1.json", fit_metrics),
        ("FACTORIZED_CALIBRATOR_IDENTITY_RECEIPT_V1.json", identity_receipt),
    ]:
        (staging / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"Calibrator Freeze: {out_root} all_valid={all_valid}")
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
