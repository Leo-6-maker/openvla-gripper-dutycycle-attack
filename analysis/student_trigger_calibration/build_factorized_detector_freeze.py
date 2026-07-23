#!/usr/bin/env python3
"""Build FACTORIZED_DETECTOR_FREEZE_V1 (P0-6: sealed roots, all artifacts bound)."""
from __future__ import annotations
import argparse, json, os, sys, time, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
from factorized_phase_c_integrity import (
    sha256_file, load_strict_json, verify_bundle_seal, seal_output_dir,
    consume_sealed_receipt, verify_runtime_source_files,
)
SELF_SHA = None

def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-b-validation-root", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-validation-root", type=Path, required=True)
    ap.add_argument("--heldout-prediction-authorization-root", type=Path, required=True)
    ap.add_argument("--heldout-prediction-validation-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-evaluation-authorization-root", type=Path, required=True)
    ap.add_argument("--heldout-l3-run-root", type=Path, default=None)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--runtime-source-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # P0-6: Consume all sealed receipt roots
    consume_sealed_receipt(args.phase_b_validation_root, "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2", "cp_inference_authorized", True, "PHASE_B")
    consume_sealed_receipt(args.cp_prediction_validation_root, "DEEPSEEK_CP_PREDICTION_VALIDATION_RECEIPT_V1", "cp_predictions_ready", True, "CP_VAL")
    consume_sealed_receipt(args.calibrator_freeze_validation_root, "FACTORIZED_CALIBRATOR_FREEZE_VALIDATION_V1", "status", "PASS", "CAL_FREEZE_VAL")
    consume_sealed_receipt(args.scheduler_freeze_validation_root, "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1", "status", "PASS", "SCHED_FREEZE_VAL")
    consume_sealed_receipt(args.heldout_prediction_authorization_root, "FACTORIZED_HELDOUT_PREDICTION_AUTHORIZATION_RECEIPT_V1", "heldout_prediction_inference_authorized", True, "H_PRED_AUTH")
    consume_sealed_receipt(args.heldout_prediction_validation_root, "FACTORIZED_HELDOUT_PREDICTION_VALIDATION_RECEIPT_V1", "h_predictions_ready", True, "H_PRED_VAL")

    cf_root = args.calibrator_freeze_root.resolve()
    sf_root = args.scheduler_freeze_root.resolve()
    verify_bundle_seal(cf_root, "CAL_FREEZE")
    verify_bundle_seal(sf_root, "SCHED_FREEZE")
    cf = load_strict_json(cf_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CAL_FREEZE")
    sf = load_strict_json(sf_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json", "SCHED_FREEZE")

    runtime_src_root = args.runtime_source_root.resolve() if args.runtime_source_root else ROOT
    rt_sources = verify_runtime_source_files(runtime_src_root)
    structure_sha = sha256_file(args.structure_config)

    h_l3_run_sha = ""
    h_l3_gate = {}
    if args.heldout_l3_run_root:
        h_l3_run = consume_sealed_receipt(args.heldout_l3_run_root, "HELDOUT_L3_RUN_RECEIPT_V1", "run_status", "COMPLETE", "H_L3_RUN")
        h_l3_run_sha = sha256_file(args.heldout_l3_run_root / "SHA256SUMS")
        h_l3_gate = {"worst_split_false_start_rate": h_l3_run[0].get("worst_split_false_start_rate"),
                     "gate_pass": h_l3_run[0].get("gate_pass", False) if isinstance(h_l3_run, tuple) else False}

    errors: list[str] = []
    if cf.get("all_heads_frozen") is not True: errors.append("CAL_NOT_FROZEN")
    if sf.get("status") != "COMPLETE": errors.append("SCHED_NOT_COMPLETE")

    def _seal_of(root): return sha256_file(root / "SHA256SUMS")

    detector_freeze = {
        "schema": "FACTORIZED_DETECTOR_FREEZE_V1",
        "status": "COMPLETE" if not errors else "HOLD",
        "errors": errors,
        "bindings": {
            "phase_b_validation_seal": _seal_of(args.phase_b_validation_root),
            "cp_prediction_validation_seal": _seal_of(args.cp_prediction_validation_root),
            "calibrator_freeze_seal": _seal_of(cf_root),
            "calibrator_freeze_validation_seal": _seal_of(args.calibrator_freeze_validation_root),
            "scheduler_freeze_seal": _seal_of(sf_root),
            "scheduler_freeze_validation_seal": _seal_of(args.scheduler_freeze_validation_root),
            "heldout_prediction_authorization_seal": _seal_of(args.heldout_prediction_authorization_root),
            "heldout_prediction_validation_seal": _seal_of(args.heldout_prediction_validation_root),
            "heldout_l3_evaluation_authorization_seal": _seal_of(args.heldout_l3_evaluation_authorization_root),
            "feature_order_sha256": sha256_file(args.feature_order_contract),
            "normalization_sha256": sha256_file(args.normalization_contract),
            "structural_config_sha256": structure_sha,
            "scheduler_source_sha256": rt_sources["scheduler_source_sha256"],
            "runtime_adapter_source_sha256": rt_sources["runtime_adapter_source_sha256"],
            "freeze_builder_code_sha256": SELF_SHA,
        },
        "selected_thresholds": sf.get("selected_thresholds", {}),
        "calibrator_methods": {sk: {h: cf["per_split"][sk][h]["method"] for h in ("grasp", "manipulation", "release")} for sk in cf.get("per_split", {})},
        "heldout_l3_gate": h_l3_gate,
        "attack_authorized": False,
        "canary_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if h_l3_run_sha: detector_freeze["bindings"]["heldout_l3_run_seal"] = h_l3_run_sha

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging"); staging.mkdir(parents=True)
    (staging / "FACTORIZED_DETECTOR_FREEZE_V1.json").write_text(json.dumps(detector_freeze, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging); os.replace(staging, out_root)
    print(f"Detector Freeze: {detector_freeze['status']}")
    return 0 if not errors else 1

if __name__ == "__main__": raise SystemExit(main())
