#!/usr/bin/env python3
"""Validate FACTORIZED_DETECTOR_FREEZE_V1 (P0-6: sealed root, attack_authorized always false)."""
from __future__ import annotations
import argparse, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
from factorized_phase_c_integrity import (
    sha256_file, is_64char_hex, load_strict_json, verify_bundle_seal, seal_output_dir,
)
SELF_SHA = None
REQUIRED_BINDINGS = (
    "phase_b_validation_seal", "cp_prediction_validation_seal",
    "calibrator_freeze_seal", "calibrator_freeze_validation_seal",
    "scheduler_freeze_seal", "scheduler_freeze_validation_seal",
    "heldout_prediction_authorization_seal", "heldout_prediction_validation_seal",
    "heldout_l3_evaluation_authorization_seal",
    "feature_order_sha256", "normalization_sha256", "structural_config_sha256",
    "scheduler_source_sha256", "runtime_adapter_source_sha256", "freeze_builder_code_sha256",
)

def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-contract-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["authoritative", "diagnostic"], default="diagnostic")
    args = ap.parse_args()
    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    freeze_root = args.freeze_contract_root.resolve()
    seal = verify_bundle_seal(freeze_root, "FREEZE")
    contract = load_strict_json(freeze_root / "FACTORIZED_DETECTOR_FREEZE_V1.json", "CONTRACT")
    errors: list[str] = []

    if contract.get("schema") != "FACTORIZED_DETECTOR_FREEZE_V1": errors.append("SCHEMA_INVALID")
    if contract.get("attack_authorized") is not False: errors.append("ATTACK_AUTHORIZED")
    if contract.get("canary_authorized") is not False: errors.append("CANARY_AUTHORIZED")

    bindings = contract.get("bindings", {})
    for key in REQUIRED_BINDINGS:
        if not is_64char_hex(bindings.get(key, "")): errors.append(f"BINDING_INVALID: {key}")

    thresholds = contract.get("selected_thresholds", {})
    for head in ("grasp", "manipulation", "release"):
        if not isinstance(thresholds.get(head), (int, float)): errors.append(f"THRESHOLD_MISSING: {head}")

    h_gate = contract.get("heldout_l3_gate", {})
    if h_gate and h_gate.get("gate_pass") is not True: errors.append("HELDOUT_GATE_NOT_PASS")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging"); staging.mkdir(parents=True)
    v = {"schema": "FACTORIZED_DETECTOR_FREEZE_VALIDATION_V1", "validator_code_sha256": SELF_SHA,
         "status": "PASS" if not errors else "HOLD", "errors": errors,
         "freeze_contract_seal_sha256": seal, "mode": args.mode}
    (staging / "FACTORIZED_DETECTOR_FREEZE_VALIDATION_V1.json").write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging); os.replace(staging, out_root)
    print(f"Detector Freeze Validation: {v['status']}")
    for e in errors: print(f"  ERROR: {e}")
    return 0 if not errors else 1

if __name__ == "__main__": raise SystemExit(main())
