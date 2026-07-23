#!/usr/bin/env python3
"""Validate FACTORIZED_CALIBRATOR_FREEZE_V1 (P0-6: sealed root with cross-binding)."""
from __future__ import annotations
import argparse, json, math, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
from factorized_phase_c_integrity import (
    HEADS, sha256_file, is_64char_hex, load_strict_json, verify_bundle_seal, seal_output_dir,
)

SELF_SHA = None

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
    contract = load_strict_json(freeze_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json", "CONTRACT")

    errors: list[str] = []
    if contract.get("schema") != "FACTORIZED_CALIBRATOR_FREEZE_V1": errors.append("SCHEMA_INVALID")
    if contract.get("attack_authorized") is not False: errors.append("ATTACK_AUTHORIZED")
    if contract.get("heldout_l3_authorized") is not False: errors.append("HELDOUT_L3_AUTHORIZED")

    bindings = contract.get("freeze_bindings", {})
    for key in ("phase_b_validation_seal_sha256", "cp_prediction_validation_seal_sha256",
                "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256",
                "calibration_teacher_bundle_sha256", "feature_order_sha256",
                "normalization_sha256", "freeze_code_sha256"):
        if not is_64char_hex(bindings.get(key, "")): errors.append(f"BINDING_INVALID: {key}")

    per_split = contract.get("per_split", {})
    frozen_count = 0
    for sk in sorted(per_split):
        for head in HEADS:
            hd = per_split[sk].get(head, {})
            if not isinstance(hd, dict): errors.append(f"HEAD_MISSING: {sk}/{head}"); continue
            if hd.get("method") not in ("RAW", "INTERCEPT_ONLY", "PLATT"): errors.append(f"METHOD_INVALID: {sk}/{head}")
            if hd.get("method_valid"):
                if hd.get("n_fit_pos", 0) == 0: errors.append(f"NO_POS: {sk}/{head}")
                elif hd.get("n_fit_neg", 0) == 0: errors.append(f"NO_NEG: {sk}/{head}")
                else: frozen_count += 1
            a_val, b_val = hd.get("a"), hd.get("b")
            if isinstance(a_val, bool) or not isinstance(a_val, (int, float)) or not math.isfinite(float(a_val)): errors.append(f"PARAM_A: {sk}/{head}")
            if isinstance(b_val, bool) or not isinstance(b_val, (int, float)) or not math.isfinite(float(b_val)): errors.append(f"PARAM_B: {sk}/{head}")

    if frozen_count == 0: errors.append("NO_HEADS_FROZEN")

    status = "PASS" if not errors else "HOLD"
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging"); staging.mkdir(parents=True)
    validation = {"schema": "FACTORIZED_CALIBRATOR_FREEZE_VALIDATION_V1", "validator_code_sha256": SELF_SHA,
                  "status": status, "errors": errors, "frozen_heads_count": frozen_count,
                  "freeze_contract_seal_sha256": seal, "mode": args.mode}
    (staging / "FACTORIZED_CALIBRATOR_FREEZE_VALIDATION_V1.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging); os.replace(staging, out_root)
    print(f"Cal Freeze Validation: {status} frozen={frozen_count}")
    return 0 if not errors else 1

if __name__ == "__main__": raise SystemExit(main())
