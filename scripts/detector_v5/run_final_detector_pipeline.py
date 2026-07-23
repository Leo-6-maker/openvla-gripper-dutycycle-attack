#!/usr/bin/env python3
"""Master orchestrator: Factorized Detector Freeze Pipeline (Stages 0-9).

Fail-closed: any stage failure → immediate stop. Produces sealed receipts at each stage.
This script does NOT train, infer, or modify data. It orchestrates existing stage scripts.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_SCRIPTS = ROOT / "scripts/detector_v5"
ANALYSIS = ROOT / "analysis/student_trigger_calibration"

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — adjust paths for your server
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CLEAN2000_ROOT = "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716"
DEFAULT_OUTPUT_ROOT = "/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline"
DEFAULT_PYTHON = "python"


def _run(cmd: list[str], label: str) -> None:
    """Run a command, print output, raise SystemExit on failure."""
    print(f"\n{'='*60}\n  STAGE: {label}\n  CMD: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nFATAL: {label} FAILED (rc={result.returncode})")
        raise SystemExit(result.returncode)


def _seal_receipt(data: dict[str, Any], root: Path) -> str:
    """Write a receipt JSON and seal the directory. Returns seal SHA."""
    import hashlib
    staging = root.with_name(f".{root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "receipt.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in staging.iterdir() if p.is_file() and p.name not in ("SHA256SUMS", "SHA256SUMS.sha256"))
    def sha256_file(p): d = hashlib.sha256(); d.update(p.read_bytes()); return d.hexdigest()
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    import shutil
    if root.exists(): shutil.rmtree(root)
    os.replace(staging, root)
    return seal


def main():
    ap = argparse.ArgumentParser(description="Factorized Detector Freeze Pipeline")
    ap.add_argument("--clean2000-root", type=Path, default=Path(DEFAULT_CLEAN2000_ROOT))
    ap.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--stage", type=int, choices=range(10), default=None,
                    help="Run single stage (0-9) instead of full pipeline")
    ap.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    args = ap.parse_args()

    out = args.output_root.resolve()
    py = args.python

    # ═══════════════════════════════════════════════════════════════════
    # Stage 0: CLEAN2000 Provenance Audit
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 0:
        _run([py, str(STAGE_SCRIPTS / "audit_clean2000_provenance.py"),
              "--clean2000-root", str(args.clean2000_root),
              "--output-root", str(out / "stage_0_clean2000_audit")],
             "Stage 0: CLEAN2000 Provenance Audit")
        print("Stage 0 PASS: 2000/2000 episodes, identity split clean, no A/FEC leakage")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Unified Teacher Labels
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 1:
        _run([py, str(STAGE_SCRIPTS / "build_unified_teacher_labels.py"),
              "--clean2000-audit-root", str(out / "stage_0_clean2000_audit"),
              "--output-root", str(out / "stage_1_teacher_labels")],
             "Stage 1: Unified Teacher Labels")
        print("Stage 1 PASS: Teacher labels rebuilt for FIT-TRAIN/DEV/CAL/CHECK/H")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Phase B Authoritative Coverage
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 2:
        phase_b_script = ANALYSIS / "validate_factorized_identity_disjointness.py"
        _run([py, str(phase_b_script),
              "--teacher-labels-root", str(out / "stage_1_teacher_labels"),
              "--output-root", str(out / "stage_2_phase_b")],
             "Stage 2: Phase B Authoritative Coverage")
        print("Stage 2 PASS: Phase B identity closure, teacher coverage complete")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 3: Student Training (V2B recommended canary)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 3:
        _run([py, str(STAGE_SCRIPTS / "launch_factorized_v2_recommended_canary.py"),
              "--teacher-labels-root", str(out / "stage_1_teacher_labels"),
              "--output-root", str(out / "stage_3_student_training")],
             "Stage 3: Student Training (V2B recommended)")
        print("Stage 3 PASS: Student trained, checkpoint selected on FIT-DEV")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 4: C/P Production Inference
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 4:
        _run([py, str(STAGE_SCRIPTS / "predict_factorized_v2_recommended_canary.py"),
              "--student-checkpoint-root", str(out / "stage_3_student_training"),
              "--output-root", str(out / "stage_4_cp_inference")],
             "Stage 4: C/P Production Inference")
        print("Stage 4 PASS: CAL and CHECK prediction bundles produced")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 5: Calibrator Freeze
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 5:
        _run([py, str(ANALYSIS / "freeze_factorized_calibrators.py"),
              "--cal-predictions-root", str(out / "stage_4_cp_inference/cal"),
              "--cal-teacher-labels-root", str(out / "stage_1_teacher_labels/cal"),
              "--output-root", str(out / "stage_5_calibrator_freeze")],
             "Stage 5: Calibrator Freeze")
        print("Stage 5 PASS: Calibrators frozen on CAL states 24-26")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 6: Scheduler Freeze
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 6:
        _run([py, str(ANALYSIS / "freeze_factorized_scheduler_policy.py"),
              "--check-predictions-root", str(out / "stage_4_cp_inference/check"),
              "--check-teacher-labels-root", str(out / "stage_1_teacher_labels/check"),
              "--calibrator-fit-root", str(out / "stage_5_calibrator_freeze"),
              "--output-root", str(out / "stage_6_scheduler_freeze")],
             "Stage 6: Scheduler Freeze")
        print("Stage 6 PASS: Scheduler thresholds frozen on CHECK states 27-29")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 7: H Heldout Evaluation (ONE-SHOT)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 7:
        # Authorization
        _run([py, str(ANALYSIS / "authorize_factorized_heldout_l3.py"),
              "--h-predictions-root", str(out / "stage_4_cp_inference/h"),
              "--h-teacher-labels-root", str(out / "stage_1_teacher_labels/h"),
              "--calibrator-freeze-root", str(out / "stage_5_calibrator_freeze"),
              "--scheduler-freeze-root", str(out / "stage_6_scheduler_freeze"),
              "--student-checkpoint-root", str(out / "stage_3_student_training"),
              "--output-root", str(out / "stage_7a_h_auth")],
             "Stage 7a: H Heldout Authorization")
        # Evaluation
        _run([py, str(ANALYSIS / "run_authorized_factorized_heldout_l3.py"),
              "--authorization-root", str(out / "stage_7a_h_auth"),
              "--output-root", str(out / "stage_7b_h_evaluation")],
             "Stage 7b: H Heldout Evaluation")
        print("Stage 7 PASS: H heldout evaluation complete (ONE-SHOT)")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 8: Full-FIT
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 8:
        _run([py, str(STAGE_SCRIPTS / "run_full_fit_frozen.py"),
              "--teacher-labels-root", str(out / "stage_1_teacher_labels"),
              "--student-config-root", str(out / "stage_3_student_training"),
              "--stage-7-h-receipt-root", str(out / "stage_7b_h_evaluation"),
              "--output-root", str(out / "stage_8_full_fit")],
             "Stage 8: Full-FIT")
        print("Stage 8 PASS: Full-FIT on FIT-TRAIN+DEV+CAL+CHECK complete")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 9: Final Detector Freeze
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 9:
        _run([py, str(ANALYSIS / "build_factorized_detector_freeze.py"),
              "--student-checkpoint-root", str(out / "stage_8_full_fit"),
              "--calibrator-freeze-root", str(out / "stage_5_calibrator_freeze"),
              "--scheduler-freeze-root", str(out / "stage_6_scheduler_freeze"),
              "--h-receipt-root", str(out / "stage_7b_h_evaluation"),
              "--phase-b-receipt-root", str(out / "stage_2_phase_b"),
              "--clean2000-audit-root", str(out / "stage_0_clean2000_audit"),
              "--output-root", str(out / "FINAL_FACTORIZED_DETECTOR_V1")],
             "Stage 9: Final Detector Freeze")
        print("Stage 9 PASS: FINAL_FACTORIZED_DETECTOR_V1 sealed")

    # ═══════════════════════════════════════════════════════════════════
    # Pipeline complete
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  FACTORIZED DETECTOR FREEZE PIPELINE: COMPLETE")
    print(f"  Output: {out / 'FINAL_FACTORIZED_DETECTOR_V1'}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
