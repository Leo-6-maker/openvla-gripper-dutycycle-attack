#!/usr/bin/env python3
"""Validate C/P Student prediction bundles (P0-4, P0-5, P0-6).

FAIL-CLOSED. Authoritative mode requires all 5 identity roots.
Verifies actual checkpoint files, runtime source SHAs, sealed receipts.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from factorized_phase_c_integrity import (
    FROZEN_SPLITS, HEADS, sha256_file, is_64char_hex, is_40char_hex,
    load_strict_json, load_strict_jsonl, verify_bundle_seal, seal_output_dir,
    extract_manifest_identities, verify_identity_closure, verify_step_closure,
    validate_prediction_schema, validate_numeric_constraints,
    validate_binding_uniformity, validate_cross_role_disjointness,
    consume_sealed_receipt, verify_receipt_binding,
    verify_checkpoint_from_manifest, verify_runtime_source_files,
)

SELF_SHA = None


# ── Backward-compatible exports (used by tests) ────────────────────────

def validate_cp_physical_separation(
    c_root: Path, p_root: Path, c_ids: set[str], p_ids: set[str],
) -> None:
    c_resolved = c_root.resolve(); p_resolved = p_root.resolve()
    if c_resolved == p_resolved:
        raise SystemExit("CP_SAME_DIR")
    c_seal = sha256_file(c_root / "SHA256SUMS")
    p_seal = sha256_file(p_root / "SHA256SUMS")
    if c_seal == p_seal:
        raise SystemExit("CP_SAME_SEAL")
    if c_ids & p_ids:
        raise SystemExit(f"CP_IDENTITY_OVERLAP: n={len(c_ids & p_ids)}")


def validate_checkpoint_binding(
    rows: list[dict[str, Any]], checkpoint_manifest_root: Path, split_key: str, label: str,
) -> str:
    from factorized_phase_c_integrity import verify_checkpoint_from_manifest
    row_sha = next((r.get("checkpoint_sha256", "").lower() for r in rows), "").lower()
    result = verify_checkpoint_from_manifest(checkpoint_manifest_root, split_key, row_sha, label,
                                              require_actual_file=True)
    return result["declared_sha256"]


def validate_phase_b_receipt(receipt_path: Path, authoritative: bool) -> dict[str, Any]:
    receipt = load_strict_json(receipt_path, "PHASE_B_RECEIPT")
    if receipt.get("schema") != "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2":
        raise SystemExit("PHASE_B_RECEIPT_SCHEMA_INVALID")
    if authoritative:
        for field, expect in [("cp_inference_authorized", True), ("phase_b_data_integrity", "PASS"),
                               ("phase_b_scientific_coverage", "PASS"), ("k10_contract_parity", "PASS")]:
            if receipt.get(field) != expect:
                raise SystemExit(f"PHASE_B_{field.upper()}: expected {expect!r} got {receipt.get(field)!r}")
    return receipt


def validate_inference_not_run(output_root: Path) -> None:
    if output_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {output_root}")


def validate_sha_format(value: str, label: str) -> None:
    if not is_64char_hex(value):
        raise SystemExit(f"{label}_SHA_INVALID: {value[:40]}")


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    # Sealed receipt roots (P0-6)
    ap.add_argument("--phase-b-validation-root", type=Path, required=True)
    # Bundle roots
    ap.add_argument("--calibration-prediction-bundle-root", type=Path, required=True)
    ap.add_argument("--policy-prediction-bundle-root", type=Path, required=True)
    # All 5 identity manifests (P0-4: H and A are REQUIRED in authoritative mode)
    ap.add_argument("--calibrator-fit-manifest", type=Path, required=True)
    ap.add_argument("--policy-selection-manifest", type=Path, required=True)
    ap.add_argument("--checkpoint-training-ledger", type=Path, required=True)
    ap.add_argument("--heldout-l3-manifest", type=Path, required=True)
    ap.add_argument("--attack-eval-manifest", type=Path, required=True)
    # Checkpoint verification (P0-5)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    # Contracts
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    # Runtime source root (P0-5)
    ap.add_argument("--runtime-source-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    ap.add_argument("--require-cp-ready", action="store_true")
    ap.add_argument("--expected-splits", type=str,
                    default="o0_i0,o0_i1,o0_i2,o1_i0,o1_i1,o1_i2,o2_i0,o2_i1,o2_i2,o3_i0,o3_i1,o3_i2")
    args = ap.parse_args()

    authoritative = args.mode == "authoritative"
    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    expected = [s.strip() for s in args.expected_splits.split(",")]
    expected_set = set(expected)
    if authoritative and (len(expected) != 12 or len(expected_set) != 12 or expected_set != FROZEN_SPLITS):
        raise SystemExit("SPLIT_ENFORCEMENT: authoritative mode requires exactly 12 unique frozen splits")

    # P0-6: Consume Phase B as sealed root
    phase_b, phase_b_seal = consume_sealed_receipt(
        args.phase_b_validation_root, "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2",
        "cp_inference_authorized", True, "PHASE_B",
    )
    if authoritative:
        for field, expect in [("phase_b_data_integrity", "PASS"), ("phase_b_scientific_coverage", "PASS"),
                               ("k10_contract_parity", "PASS")]:
            if phase_b.get(field) != expect:
                raise SystemExit(f"PHASE_B_{field.upper()}: expected {expect!r}, got {phase_b.get(field)!r}")
        if not phase_b.get("calibration_coverage_pass"):
            raise SystemExit("PHASE_B: calibration_coverage_pass=false")
        if not phase_b.get("policy_coverage_pass"):
            raise SystemExit("PHASE_B: policy_coverage_pass=false")

    # P0-4: All 5 manifests required in authoritative mode
    if authoritative and (not args.heldout_l3_manifest or not args.attack_eval_manifest):
        raise SystemExit("AUTHORITATIVE: heldout-l3-manifest and attack-eval-manifest are required")

    cal_manifest = load_strict_json(args.calibrator_fit_manifest, "CAL_MANIFEST")
    pol_manifest = load_strict_json(args.policy_selection_manifest, "POL_MANIFEST")
    training_ledger = load_strict_json(args.checkpoint_training_ledger, "TRAINING_LEDGER")
    held_manifest = load_strict_json(args.heldout_l3_manifest, "HELD_MANIFEST")
    atk_manifest = load_strict_json(args.attack_eval_manifest, "ATK_MANIFEST")

    c_root = args.calibration_prediction_bundle_root.resolve()
    p_root = args.policy_prediction_bundle_root.resolve()

    # P0-6: Verify bundle seals
    c_seal = verify_bundle_seal(c_root, "C_PREDICTION")
    p_seal = verify_bundle_seal(p_root, "P_PREDICTION")
    if c_root == p_root:
        raise SystemExit("CP_SAME_DIR")
    if c_seal == p_seal:
        raise SystemExit("CP_SAME_SEAL")

    # P0-5: Verify actual runtime source SHAs
    runtime_src_root = args.runtime_source_root.resolve() if args.runtime_source_root else ROOT
    rt_sources = verify_runtime_source_files(runtime_src_root)
    runtime_adapter_sha = rt_sources["runtime_adapter_source_sha256"]

    feature_sha = sha256_file(args.feature_order_contract)
    norm_sha = sha256_file(args.normalization_contract)

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    all_errors: list[str] = []
    per_split: dict[str, dict[str, Any]] = {}
    id_rows: list[list[Any]] = []
    step_rows: list[list[Any]] = []
    binding_rows: list[list[Any]] = []
    cp_ready = True

    for sk in expected:
        split_errors: list[str] = []

        c_manifest_ids = extract_manifest_identities(cal_manifest, "calibrator_fit", sk)
        p_manifest_ids = extract_manifest_identities(pol_manifest, "policy_selection", sk)
        t_ids_dict = {sk: extract_manifest_identities(training_ledger, "checkpoint_training", sk)}
        h_ids_dict = {sk: extract_manifest_identities(held_manifest, "heldout_l3", sk)}
        a_ids_dict = {sk: extract_manifest_identities(atk_manifest, "attack_eval", sk)}

        c_rows = load_strict_jsonl(c_root / sk / "predictions.jsonl", f"C_PRED_{sk}")
        p_rows = load_strict_jsonl(p_root / sk / "predictions.jsonl", f"P_PRED_{sk}")

        c_pred_ids = {r["canonical_parent_key"] for r in c_rows}
        p_pred_ids = {r["canonical_parent_key"] for r in p_rows}

        try:
            validate_prediction_schema(c_rows, f"C_PRED_{sk}")
            validate_prediction_schema(p_rows, f"P_PRED_{sk}")
            validate_numeric_constraints(c_rows, f"C_PRED_{sk}")
            validate_numeric_constraints(p_rows, f"P_PRED_{sk}")
            verify_step_closure(c_rows, f"C_PRED_{sk}")
            verify_step_closure(p_rows, f"P_PRED_{sk}")

            c_binding = validate_binding_uniformity(c_rows, f"C_PRED_{sk}")
            p_binding = validate_binding_uniformity(p_rows, f"P_PRED_{sk}")

            verify_identity_closure(c_pred_ids, c_manifest_ids, "CALIBRATION", sk)
            verify_identity_closure(p_pred_ids, p_manifest_ids, "POLICY", sk)
            validate_cross_role_disjointness(c_pred_ids, p_pred_ids, t_ids_dict, h_ids_dict, a_ids_dict, sk)

            # P0-5: Verify checkpoint binding against actual files
            verify_checkpoint_from_manifest(args.checkpoint_manifest_root, sk,
                                            c_binding["checkpoint_sha256"], f"C_PRED_{sk}")
            verify_checkpoint_from_manifest(args.checkpoint_manifest_root, sk,
                                            p_binding["checkpoint_sha256"], f"P_PRED_{sk}")

            # P0-5: Verify runtime source SHA matches actual file
            declared_rt = c_binding.get("runtime_source_sha256", "")
            if declared_rt != runtime_adapter_sha:
                raise SystemExit(f"C_PRED_{sk}_RUNTIME_SOURCE_MISMATCH: declared={declared_rt[:16]} actual={runtime_adapter_sha[:16]}")
            declared_rt_p = p_binding.get("runtime_source_sha256", "")
            if declared_rt_p != runtime_adapter_sha:
                raise SystemExit(f"P_PRED_{sk}_RUNTIME_SOURCE_MISMATCH: declared={declared_rt_p[:16]} actual={runtime_adapter_sha[:16]}")

            if c_binding["feature_order_sha256"] != feature_sha:
                raise SystemExit(f"C_PRED_{sk}_FEATURE_MISMATCH")
            if p_binding["feature_order_sha256"] != feature_sha:
                raise SystemExit(f"P_PRED_{sk}_FEATURE_MISMATCH")
            if c_binding["normalization_sha256"] != norm_sha:
                raise SystemExit(f"C_PRED_{sk}_NORM_MISMATCH")
            if p_binding["normalization_sha256"] != norm_sha:
                raise SystemExit(f"P_PRED_{sk}_NORM_MISMATCH")

            for label, rows in [("C", c_rows), ("P", p_rows)]:
                by_ep: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for r in rows:
                    by_ep[r["canonical_parent_key"]].append(r)
                for ep_id, ep_rows in by_ep.items():
                    source_shas = {r.get("source_artifact_recursive_sha256") for r in ep_rows}
                    if len(source_shas) != 1:
                        raise SystemExit(f"{label}_PRED_{sk}_SOURCE_SHA_MULTIPLE: {ep_id}")

            id_rows.append([sk, "C", len(c_pred_ids), len(c_manifest_ids), "PASS" if c_pred_ids == c_manifest_ids else "FAIL"])
            id_rows.append([sk, "P", len(p_pred_ids), len(p_manifest_ids), "PASS" if p_pred_ids == p_manifest_ids else "FAIL"])
            step_rows.append([sk, "C", len(c_rows), "PASS"])
            step_rows.append([sk, "P", len(p_rows), "PASS"])
            binding_rows.append([sk, "C", c_binding["checkpoint_sha256"][:16], c_binding["feature_order_sha256"][:16], "PASS"])
            binding_rows.append([sk, "P", p_binding["checkpoint_sha256"][:16], p_binding["feature_order_sha256"][:16], "PASS"])

        except SystemExit as e:
            split_errors.append(str(e))
            cp_ready = False

        per_split[sk] = {"errors": split_errors, "c_identities": len(c_pred_ids),
                         "p_identities": len(p_pred_ids), "pass": len(split_errors) == 0}
        all_errors.extend(split_errors)

    receipt = {
        "schema": "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1",
        "validator_code_sha256": SELF_SHA,
        "status": "COMPLETE", "cp_predictions_ready": cp_ready, "mode": args.mode,
        "phase_b_validation_seal_sha256": phase_b_seal,
        "phase_b_receipt_sha256": sha256_file(
            next(p for p in args.phase_b_validation_root.iterdir() if p.suffix == ".json" and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
        ) if args.phase_b_validation_root.is_dir() else "",
        "calibration_prediction_seal_sha256": c_seal,
        "policy_prediction_seal_sha256": p_seal,
        "calibrator_fit_manifest_sha256": sha256_file(args.calibrator_fit_manifest),
        "policy_selection_manifest_sha256": sha256_file(args.policy_selection_manifest),
        "checkpoint_training_ledger_sha256": sha256_file(args.checkpoint_training_ledger),
        "heldout_l3_manifest_sha256": sha256_file(args.heldout_l3_manifest),
        "attack_eval_manifest_sha256": sha256_file(args.attack_eval_manifest),
        "feature_order_contract_sha256": feature_sha,
        "normalization_contract_sha256": norm_sha,
        "runtime_adapter_source_sha256": runtime_adapter_sha,
        "n_errors": len(all_errors), "n_splits": len(expected), "per_split": per_split,
    }
    if all_errors:
        receipt["errors"] = all_errors[:50]

    with open(staging / "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    # Write CSVs
    def _csv(path, headers, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(headers)
            for row in rows: w.writerow(row)

    _csv(staging / "CP_PREDICTION_IDENTITY_CLOSURE.csv",
         ["split", "role", "prediction_identity_count", "manifest_identity_count", "closure_status"], id_rows)
    _csv(staging / "CP_PREDICTION_STEP_CLOSURE.csv",
         ["split", "role", "total_rows", "step_closure_status"], step_rows)
    _csv(staging / "CP_PREDICTION_BINDING_AUDIT.csv",
         ["split", "role", "checkpoint_sha16", "feature_order_sha16", "binding_status"], binding_rows)

    seal_output_dir(staging)
    os.replace(staging, out_root)

    print(f"CP Prediction Validation: ready={cp_ready} errors={len(all_errors)}")
    return 0 if cp_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
