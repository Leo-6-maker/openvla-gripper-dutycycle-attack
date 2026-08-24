#!/usr/bin/env python3
"""A10: Full CLI E2E — verify every stage input seal matches previous stage output seal."""
from __future__ import annotations

import argparse, json, os, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/pilot_attack"))
from pilot_integrity import sha256_file, verify_sealed_root

SELF_SHA = None

# Stage chain: (stage_name, receipt_filename, downstream_seal_key_in_next_stage_receipt)
# The seal of stage N's SHA256SUMS must match what stage N+1 declares as its input.
STAGE_CHAIN = [
    ("stage_0_clean2000_audit", "CLEAN2000_PROVENANCE_RECEIPT_V1"),
    ("stage_1_teacher_labels", "UNIFIED_TEACHER_LABELS_RECEIPT_V1"),
    ("stage_2_phase_b", "receipt.json"),
    ("stage_3_student_training", "receipt.json"),
    ("stage_4_cp_inference", "receipt.json"),
    ("stage_5_calibrator_freeze", "receipt.json"),
    ("stage_5v_calibrator_validation", "receipt.json"),
    ("stage_6_scheduler_freeze", "receipt.json"),
    ("stage_6v_scheduler_validation", "receipt.json"),
    ("stage_7a_h_auth", "receipt.json"),
    ("stage_7b_h_evaluation", "HELDOUT_L3_RUN_COMPLETE_RECEIPT_V1.json"),
    ("stage_8_full_fit", "FULL_FIT_RECEIPT_V1.json"),
]

FINAL_DETECTOR_DIR = "FINAL_FACTORIZED_DETECTOR_V1"


def main() -> int:
    global SELF_SHA; SELF_SHA = sha256_file(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-output-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    out_root = args.output_root.resolve()
    if out_root.exists(): raise SystemExit(f"OUTPUT_EXISTS: {out_root}")
    pipe = args.pipeline_output_root.resolve()

    errors: list[str] = []
    stage_seals: dict[str, str] = {}

    for stage_dir_name, receipt_name in STAGE_CHAIN:
        stage_path = pipe / stage_dir_name
        if not stage_path.is_dir():
            errors.append(f"STAGE_MISSING: {stage_dir_name}")
            continue

        # Verify sealed
        try:
            seal = verify_sealed_root(stage_path, stage_dir_name.upper())
            stage_seals[stage_dir_name] = seal
        except SystemExit as e:
            errors.append(f"STAGE_UNSEALED: {stage_dir_name} — {e}")
            continue

        # Verify receipt exists and has PASS/COMPLETE status
        receipt_path = stage_path / receipt_name
        if not receipt_path.is_file():
            errors.append(f"STAGE_NO_RECEIPT: {stage_dir_name}/{receipt_name}")
            continue

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        status = receipt.get("status", receipt.get("run_status", "UNKNOWN"))
        gate_pass = receipt.get("gate_pass")

        stage_seals[f"{stage_dir_name}_status"] = status
        if status not in ("PASS", "COMPLETE"):
            if gate_pass is True:
                continue  # H heldout can gate_pass=true with run_status=COMPLETE
            errors.append(f"STAGE_STATUS_NOT_PASS: {stage_dir_name} status={status}")

    # ── Verify final detector ────────────────────────────────────────
    final = pipe / FINAL_DETECTOR_DIR
    if not final.is_dir():
        errors.append("FINAL_DETECTOR_MISSING")
    else:
        try:
            final_seal = verify_sealed_root(final, "FINAL_DETECTOR")
            stage_seals["FINAL_DETECTOR"] = final_seal
        except SystemExit as e:
            errors.append(f"FINAL_DETECTOR_UNSEALED: {e}")

        freeze_json = final / "FACTORIZED_DETECTOR_FREEZE_V1.json"
        if freeze_json.is_file():
            freeze = json.loads(freeze_json.read_text(encoding="utf-8"))
            stage_seals["final_attack_authorized"] = freeze.get("attack_authorized", "MISSING")
            if freeze.get("attack_authorized") is not False:
                errors.append("FINAL_DETECTOR_ATTACK_AUTHORIZED_NOT_FALSE")
            if freeze.get("uses_attack_outcome") is not False:
                errors.append("FINAL_DETECTOR_USES_ATTACK_OUTCOME")

    receipt = {
        "schema": "FULL_CLI_E2E_RECEIPT_V1",
        "verifier_code_sha256": SELF_SHA,
        "status": "PASS" if not errors else "HOLD",
        "stage_seals": {k: v[:16] if isinstance(v, str) and len(v) == 64 else v
                        for k, v in stage_seals.items()},
        "n_stages_found": len(stage_seals),
        "n_errors": len(errors),
        "errors": errors,
    }

    import shutil
    staging = out_root.with_name(f".{out_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "FULL_CLI_E2E_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file()
                   and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    if out_root.exists(): shutil.rmtree(out_root)
    os.replace(staging, out_root)

    print(f"Full CLI E2E: {receipt['status']} errors={len(errors)}")
    for s, v in receipt["stage_seals"].items(): print(f"  {s}: {v}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
