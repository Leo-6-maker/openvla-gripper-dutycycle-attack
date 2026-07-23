#!/usr/bin/env python3
"""Validate FACTORIZED_SCHEDULER_FREEZE_V1 (P0-6: sealed root, cross-binding, recall>0)."""
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
REQUIRED_BINDINGS = (
    "calibrator_freeze_sha256", "calibrator_fit_manifest_sha256",
    "policy_selection_manifest_sha256", "policy_prediction_bundle_sha256",
    "policy_teacher_bundle_sha256", "policy_runtime_bundle_sha256",
    "runtime_adapter_source_sha256", "scheduler_source_sha256",
    "structural_config_sha256", "freeze_code_sha256",
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
    contract = load_strict_json(freeze_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json", "CONTRACT")
    errors: list[str] = []

    if contract.get("schema") != "FACTORIZED_SCHEDULER_FREEZE_V1": errors.append("SCHEMA_INVALID")
    if contract.get("attack_authorized") is not False: errors.append("ATTACK_AUTHORIZED")
    if contract.get("heldout_l3_authorized") is not False: errors.append("HELDOUT_L3_AUTHORIZED")

    status = contract.get("status", "")
    bindings = contract.get("bindings", {})
    for key in REQUIRED_BINDINGS:
        if not is_64char_hex(bindings.get(key, "")): errors.append(f"BINDING_INVALID: {key}")

    if status == "HOLD_NO_FEASIBLE_THRESHOLD":
        errors.append("NO_FEASIBLE_THRESHOLD")
    elif status == "COMPLETE":
        for head in HEADS:
            t = contract.get("selected_thresholds", {}).get(head)
            if not isinstance(t, (int, float)) or not 0.0 <= float(t) <= 1.0: errors.append(f"THRESHOLD_INVALID: {head}")
        ws = contract.get("worst_split_false_start")
        if ws is None or ws > 0.10: errors.append(f"WORST_FALSE_START: {ws}")
        recall = contract.get("selected_metrics", {}).get("valid_opportunity_recall", 0)
        if recall is None or recall <= 0: errors.append(f"RECALL_ZERO: {recall}")
        if len(contract.get("per_split", {})) != 12: errors.append("SPLIT_COUNT")
    else:
        errors.append(f"STATUS_UNKNOWN: {status}")

    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging"); staging.mkdir(parents=True)
    v = {"schema": "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1", "validator_code_sha256": SELF_SHA,
         "status": "PASS" if not errors else "HOLD", "errors": errors,
         "freeze_contract_seal_sha256": seal, "mode": args.mode}
    (staging / "FACTORIZED_SCHEDULER_FREEZE_VALIDATION_V1.json").write_text(json.dumps(v, indent=2, sort_keys=True) + "\n")
    seal_output_dir(staging); os.replace(staging, out_root)
    print(f"Sched Freeze Validation: {v['status']}")
    for e in errors: print(f"  ERROR: {e}")
    return 0 if not errors else 1

if __name__ == "__main__": raise SystemExit(main())
