#!/usr/bin/env python3
"""Build the final FACTORIZED_DETECTOR_FREEZE_V1.json contract.

Binds all Phase B and Phase C artifacts into a single frozen detector
configuration. Default: attack_authorized=false, canary_authorized=false.
Only a separate Codex runtime parity/canary validator can change these.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
SELF_SHA = None


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): d.update(chunk)
    return d.hexdigest()


def is_64char_hex(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    dups: list[str] = []
    def hook(pairs):
        seen = set(); result = {}
        for k, v in pairs:
            if k in seen: dups.append(k)
            seen.add(k)
            result[k] = v
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except json.JSONDecodeError as e:
        raise SystemExit(f"{label}_JSON_PARSE: {path} {e}")
    if dups:
        raise SystemExit(f"{label}_DUP_KEYS: {path}")
    return value


def _seal_output(output_root: Path, files: dict[str, str]) -> str:
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, content in files.items():
        (staging / name).write_text(content, encoding="utf-8")
    data = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in data))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    os.replace(staging, output_root)
    return seal


def main() -> int:
    global SELF_SHA
    SELF_SHA = sha256_file(Path(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-b-receipt", type=Path, required=True)
    ap.add_argument("--cp-prediction-validation-receipt", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-root", type=Path, required=True)
    ap.add_argument("--calibrator-freeze-validation-receipt", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-root", type=Path, required=True)
    ap.add_argument("--scheduler-freeze-validation-receipt", type=Path, required=True)
    ap.add_argument("--heldout-authorization-receipt", type=Path, required=True)
    ap.add_argument("--heldout-l3-run-receipt", type=Path, required=True)
    ap.add_argument("--checkpoint-manifest-root", type=Path, required=True)
    ap.add_argument("--feature-order-contract", type=Path, required=True)
    ap.add_argument("--normalization-contract", type=Path, required=True)
    ap.add_argument("--structure-config", type=Path,
                    default=ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS: {out_root}")

    # Load all artifacts
    phase_b = load_strict_json(args.phase_b_receipt, "PHASE_B")
    cp_val = load_strict_json(args.cp_prediction_validation_receipt, "CP_VAL")
    cf_path = args.calibrator_freeze_root / "FACTORIZED_CALIBRATOR_FREEZE_V1.json"
    sf_path = args.scheduler_freeze_root / "FACTORIZED_SCHEDULER_FREEZE_V1.json"
    cf = load_strict_json(cf_path, "CAL_FREEZE")
    sf = load_strict_json(sf_path, "SCHED_FREEZE")
    cf_val = load_strict_json(args.calibrator_freeze_validation_receipt, "CAL_FREEZE_VAL")
    sf_val = load_strict_json(args.scheduler_freeze_validation_receipt, "SCHED_FREEZE_VAL")
    auth = load_strict_json(args.heldout_authorization_receipt, "H_AUTH")
    h_run = load_strict_json(args.heldout_l3_run_receipt, "H_RUN")
    structure = load_strict_json(args.structure_config, "STRUCTURE")
    feature_order = load_strict_json(args.feature_order_contract, "FEATURE")
    normalization = load_strict_json(args.normalization_contract, "NORM")

    scheduler_path = ROOT / "src/gripper_attack/factorized_scheduler.py"
    adapter_path = ROOT / "src/gripper_attack/factorized_scheduler_adapter.py"

    # Validate prereqs
    errors: list[str] = []
    if not phase_b.get("cp_inference_authorized"):
        errors.append("PHASE_B_CP_NOT_AUTHORIZED")
    if not cp_val.get("cp_predictions_ready"):
        errors.append("CP_PREDICTIONS_NOT_READY")
    if not cf.get("all_heads_frozen"):
        errors.append("CAL_NOT_ALL_FROZEN")
    if sf.get("status") != "COMPLETE":
        errors.append("SCHED_NOT_COMPLETE")
    if not auth.get("heldout_l3_inference_authorized"):
        errors.append("H_NOT_AUTHORIZED")
    if h_run.get("run_status") != "COMPLETE":
        errors.append("H_RUN_NOT_COMPLETE")

    # Build freeze
    detector_freeze = {
        "schema": "FACTORIZED_DETECTOR_FREEZE_V1",
        "status": "COMPLETE" if not errors else "HOLD",
        "errors": errors,
        "bindings": {
            "phase_b_receipt_sha256": sha256_file(args.phase_b_receipt),
            "cp_prediction_validation_receipt_sha256": sha256_file(args.cp_prediction_validation_receipt),
            "calibrator_freeze_sha256": sha256_file(cf_path),
            "calibrator_freeze_validation_sha256": sha256_file(args.calibrator_freeze_validation_receipt),
            "scheduler_freeze_sha256": sha256_file(sf_path),
            "scheduler_freeze_validation_sha256": sha256_file(args.scheduler_freeze_validation_receipt),
            "heldout_authorization_receipt_sha256": sha256_file(args.heldout_authorization_receipt),
            "heldout_l3_run_receipt_sha256": sha256_file(args.heldout_l3_run_receipt),
            "feature_order_sha256": sha256_file(args.feature_order_contract),
            "normalization_sha256": sha256_file(args.normalization_contract),
            "structural_config_sha256": sha256_file(args.structure_config),
            "scheduler_source_sha256": sha256_file(scheduler_path),
            "runtime_adapter_source_sha256": sha256_file(adapter_path),
            "freeze_builder_code_sha256": SELF_SHA,
        },
        "selected_thresholds": sf.get("selected_thresholds", {}),
        "calibrator_methods": {
            sk: {head: cf["per_split"][sk][head]["method"] for head in ("grasp", "manipulation", "release")}
            for sk in cf.get("per_split", {})
        },
        "heldout_l3_gate": {
            "worst_split_false_start_rate": h_run.get("worst_split_false_start_rate"),
            "gate_pass": h_run.get("gate_pass", False),
        },
        "attack_authorized": False,
        "canary_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    files = {
        "FACTORIZED_DETECTOR_FREEZE_V1.json": json.dumps(detector_freeze, indent=2, sort_keys=True) + "\n",
    }
    _seal_output(out_root, files)
    print(f"Detector Freeze: {out_root} status={detector_freeze['status']}")
    print(f"  attack_authorized: {detector_freeze['attack_authorized']}")
    print(f"  canary_authorized: {detector_freeze['canary_authorized']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
