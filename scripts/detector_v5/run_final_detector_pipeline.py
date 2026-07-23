#!/usr/bin/env python3
"""Master orchestrator: Factorized Detector Freeze Pipeline (Stages 0-9).

Fail-closed: any stage failure → immediate halt.
Uses exact CLI argument names from the actual Phase C / V2 scripts.
All paths come from a pipeline config JSON file (no fabricated paths).
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_SCRIPTS = ROOT / "scripts/detector_v5"
ANALYSIS = ROOT / "analysis/student_trigger_calibration"
SRC = ROOT / "src"

# ═══════════════════════════════════════════════════════════════════════════
# Pipeline config template — fill in actual server paths before execution.
# All keys under "paths" must be absolute paths on the target server.
# ═══════════════════════════════════════════════════════════════════════════
CONFIG_TEMPLATE: dict[str, Any] = {
    "paths": {
        "clean2000_root": "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716",
        "output_root": "/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline",
        "python": "python",
        # ── Intermediate artifacts (populated by pipeline stages) ──────
        "stage_0_audit_root": None,              # → stage_0_clean2000_audit
        "stage_1_teacher_labels_root": None,      # → stage_1_teacher_labels
        "stage_2_phase_b_root": None,             # → stage_2_phase_b
        "stage_3_student_training_root": None,    # → stage_3_student_training
        "stage_4_cp_inference_root": None,        # → stage_4_cp_inference
        "stage_5_calibrator_freeze_root": None,   # → stage_5_calibrator_freeze
        "stage_6_scheduler_freeze_root": None,    # → stage_6_scheduler_freeze
        "stage_7a_h_auth_root": None,             # → stage_7a_h_auth
        "stage_7b_h_eval_root": None,             # → stage_7b_h_evaluation
        "stage_8_full_fit_root": None,            # → stage_8_full_fit
        "stage_9_final_detector_root": None,      # → FINAL_FACTORIZED_DETECTOR_V1
        # ── Pre-existing sealed artifacts (must exist before pipeline) ─
        "identity_source_discovery": None,         # identity split manifest
        "checkpoint_training_ledger": None,        # checkpoint training ledger
        "checkpoint_manifest_root": None,          # student checkpoint manifest
        "feature_order_contract": None,            # 25D feature schema
        "normalization_contract": None,            # normalization artifact
        "inner_cv_splits_root": None,              # inner-CV split definitions
        "calibrator_fit_manifest": None,           # calibrator fit manifest
        "policy_selection_manifest": None,         # scheduler policy selection manifest
        "calibrator_freeze_validation_root": None, # Stage 5 validation output
        "scheduler_freeze_validation_root": None,  # Stage 6 validation output
        "cp_prediction_validation_root": None,     # C/P prediction validation receipt
        "heldout_prediction_auth_root": None,      # H prediction authorization
        "heldout_prediction_validation_root": None,# H prediction validation receipt
        "heldout_prediction_bundle_root": None,    # H prediction bundle
        "heldout_teacher_bundle_root": None,       # H teacher labels
        "heldout_runtime_bundle_root": None,       # H runtime bundle
        "heldout_l3_manifest": None,               # H L3 evaluation manifest
        "heldout_l3_eval_auth_root": None,         # H L3 evaluation authorization
        "heldout_l3_run_root": None,               # H L3 run output (optional)
    }
}


def _cfg(paths: dict, key: str) -> str:
    v = paths.get(key)
    if v is None:
        raise SystemExit(f"CONFIG_MISSING: paths.{key} is None — fill in pipeline config")
    return str(v)


def _run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}\n  STAGE: {label}\n  CMD: {' '.join(cmd)}\n{'='*60}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFATAL: {label} FAILED (rc={result.returncode})")
        raise SystemExit(result.returncode)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Factorized Detector Freeze Pipeline")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON config file with pipeline paths (uses template if omitted)")
    ap.add_argument("--stage", type=int, choices=range(10), default=None,
                    help="Run single stage (0-9) instead of full pipeline")
    ap.add_argument("--clean2000-root", type=str, default=None,
                    help="Override clean2000_root from config")
    ap.add_argument("--output-root", type=str, default=None,
                    help="Override output_root from config")
    args = ap.parse_args()

    # Load config
    if args.config and args.config.is_file():
        config = json.loads(args.config.read_text())
    else:
        config = json.loads(json.dumps(CONFIG_TEMPLATE))
    paths: dict[str, Any] = config.get("paths", {})
    py = paths.get("python", "python")

    # Override from CLI
    if args.clean2000_root:
        paths["clean2000_root"] = args.clean2000_root
    if args.output_root:
        paths["output_root"] = args.output_root

    out_base = Path(_cfg(paths, "output_root"))

    # ── Resolve stage output paths ────────────────────────────────────
    S = {
        "s0": out_base / "stage_0_clean2000_audit",
        "s1": out_base / "stage_1_teacher_labels",
        "s2": out_base / "stage_2_phase_b",
        "s3": out_base / "stage_3_student_training",
        "s4": out_base / "stage_4_cp_inference",
        "s5": out_base / "stage_5_calibrator_freeze",
        "s6": out_base / "stage_6_scheduler_freeze",
        "s7a": out_base / "stage_7a_h_auth",
        "s7b": out_base / "stage_7b_h_evaluation",
        "s8": out_base / "stage_8_full_fit",
        "s9": out_base / "FINAL_FACTORIZED_DETECTOR_V1",
    }

    # ═══════════════════════════════════════════════════════════════════
    # Stage 0: CLEAN2000 Provenance Audit
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 0:
        _run([py, str(STAGE_SCRIPTS / "audit_clean2000_provenance.py"),
              "--clean2000-root", _cfg(paths, "clean2000_root"),
              "--output-root", str(S["s0"])],
             "Stage 0: CLEAN2000 Provenance Audit")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Unified Teacher Labels
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 1:
        _run([py, str(STAGE_SCRIPTS / "build_unified_teacher_labels.py"),
              "--clean2000-audit-root", str(S["s0"]),
              "--output-root", str(S["s1"])],
             "Stage 1: Unified Teacher Labels")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Phase B Authoritative Coverage
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 2:
        _run([py, str(ANALYSIS / "validate_factorized_identity_disjointness.py"),
              "--identity-source-discovery", _cfg(paths, "identity_source_discovery"),
              "--checkpoint-training-ledger", _cfg(paths, "checkpoint_training_ledger"),
              "--output-root", str(S["s2"])],
             "Stage 2: Phase B Authoritative Coverage")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 3: Student Training (V2B recommended canary)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 3:
        _run([py, str(STAGE_SCRIPTS / "launch_factorized_v2_recommended_canary.py"),
              "--output-base", str(S["s3"])],
             "Stage 3: Student Training (V2B recommended)")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 4: C/P Production Inference
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 4:
        _run([py, str(STAGE_SCRIPTS / "predict_factorized_v2_recommended_canary.py"),
              "--checkpoint-dir", str(S["s3"] / "checkpoint"),
              "--inner-cv-splits-root", _cfg(paths, "inner_cv_splits_root"),
              "--output-root", str(S["s4"])],
             "Stage 4: C/P Production Inference")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 5: Calibrator Freeze
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 5:
        _run([py, str(ANALYSIS / "freeze_factorized_calibrators.py"),
              "--calibrator-fit-manifest", _cfg(paths, "calibrator_fit_manifest"),
              "--calibration-prediction-bundle-root", str(S["s4"] / "cal"),
              "--calibration-teacher-bundle-root", str(S["s1"] / "cal"),
              "--phase-b-validation-root", str(S["s2"]),
              "--cp-prediction-validation-root", _cfg(paths, "cp_prediction_validation_root"),
              "--checkpoint-manifest-root", _cfg(paths, "checkpoint_manifest_root"),
              "--checkpoint-training-ledger", _cfg(paths, "checkpoint_training_ledger"),
              "--feature-order-contract", _cfg(paths, "feature_order_contract"),
              "--normalization-contract", _cfg(paths, "normalization_contract"),
              "--output-root", str(S["s5"])],
             "Stage 5: Calibrator Freeze")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 6: Scheduler Freeze
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 6:
        _run([py, str(ANALYSIS / "freeze_factorized_scheduler_policy.py"),
              "--policy-selection-manifest", _cfg(paths, "policy_selection_manifest"),
              "--policy-prediction-bundle-root", str(S["s4"] / "check"),
              "--policy-teacher-bundle-root", str(S["s1"] / "check"),
              "--policy-runtime-bundle-root", str(S["s4"] / "check_runtime"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--phase-b-validation-root", str(S["s2"]),
              "--cp-prediction-validation-root", _cfg(paths, "cp_prediction_validation_root"),
              "--calibrator-freeze-validation-root", _cfg(paths, "calibrator_freeze_validation_root"),
              "--calibrator-fit-manifest", _cfg(paths, "calibrator_fit_manifest"),
              "--checkpoint-manifest-root", _cfg(paths, "checkpoint_manifest_root"),
              "--output-root", str(S["s6"])],
             "Stage 6: Scheduler Freeze")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 7a: H Heldout Authorization
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 7:
        _run([py, str(ANALYSIS / "authorize_factorized_heldout_l3.py"),
              "--heldout-prediction-authorization-root", _cfg(paths, "heldout_prediction_auth_root"),
              "--heldout-prediction-validation-root", _cfg(paths, "heldout_prediction_validation_root"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--calibrator-freeze-validation-root", _cfg(paths, "calibrator_freeze_validation_root"),
              "--scheduler-freeze-root", str(S["s6"]),
              "--scheduler-freeze-validation-root", _cfg(paths, "scheduler_freeze_validation_root"),
              "--calibrator-fit-manifest", _cfg(paths, "calibrator_fit_manifest"),
              "--policy-selection-manifest", _cfg(paths, "policy_selection_manifest"),
              "--heldout-l3-manifest", _cfg(paths, "heldout_l3_manifest"),
              "--checkpoint-manifest-root", _cfg(paths, "checkpoint_manifest_root"),
              "--authorized-l3-evaluation-output-root", str(S["s7a"]),
              "--output-root", str(S["s7a"])],
             "Stage 7a: H Heldout Authorization")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 7b: H Heldout Evaluation
    # ═══════════════════════════════════════════════════════════════════
        _run([py, str(ANALYSIS / "run_authorized_factorized_heldout_l3.py"),
              "--heldout-l3-evaluation-authorization-root", str(S["s7a"]),
              "--heldout-prediction-authorization-root", _cfg(paths, "heldout_prediction_auth_root"),
              "--heldout-prediction-validation-root", _cfg(paths, "heldout_prediction_validation_root"),
              "--heldout-prediction-bundle-root", _cfg(paths, "heldout_prediction_bundle_root"),
              "--heldout-teacher-bundle-root", _cfg(paths, "heldout_teacher_bundle_root"),
              "--heldout-runtime-bundle-root", _cfg(paths, "heldout_runtime_bundle_root"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--scheduler-freeze-root", str(S["s6"]),
              "--heldout-l3-manifest", _cfg(paths, "heldout_l3_manifest"),
              "--calibrator-fit-manifest", _cfg(paths, "calibrator_fit_manifest"),
              "--policy-selection-manifest", _cfg(paths, "policy_selection_manifest"),
              "--output-root", str(S["s7b"]),
              "--claim-root", str(S["s7b"] / "claim")],
             "Stage 7b: H Heldout Evaluation")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 8: Full-FIT
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 8:
        _run([py, str(STAGE_SCRIPTS / "run_full_fit_frozen.py"),
              "--teacher-labels-root", str(S["s1"]),
              "--student-config-root", str(S["s3"]),
              "--stage-7-h-receipt-root", str(S["s7b"]),
              "--output-root", str(S["s8"])],
             "Stage 8: Full-FIT")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 9: Final Detector Freeze
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is None or args.stage == 9:
        cmd9: list[str] = [
            py, str(ANALYSIS / "build_factorized_detector_freeze.py"),
            "--phase-b-validation-root", str(S["s2"]),
            "--cp-prediction-validation-root", _cfg(paths, "cp_prediction_validation_root"),
            "--calibrator-freeze-root", str(S["s5"]),
            "--calibrator-freeze-validation-root", _cfg(paths, "calibrator_freeze_validation_root"),
            "--scheduler-freeze-root", str(S["s6"]),
            "--scheduler-freeze-validation-root", _cfg(paths, "scheduler_freeze_validation_root"),
            "--heldout-prediction-authorization-root", _cfg(paths, "heldout_prediction_auth_root"),
            "--heldout-prediction-validation-root", _cfg(paths, "heldout_prediction_validation_root"),
            "--heldout-l3-evaluation-authorization-root", str(S["s7a"]),
            "--checkpoint-manifest-root", _cfg(paths, "checkpoint_manifest_root"),
            "--feature-order-contract", _cfg(paths, "feature_order_contract"),
            "--normalization-contract", _cfg(paths, "normalization_contract"),
            "--output-root", str(S["s9"]),
        ]
        heldout_l3_run = paths.get("heldout_l3_run_root")
        if heldout_l3_run:
            cmd9.extend(["--heldout-l3-run-root", str(heldout_l3_run)])
        _run(cmd9, "Stage 9: Final Detector Freeze")

    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  FACTORIZED DETECTOR FREEZE PIPELINE: COMPLETE")
    print(f"  Final detector: {S['s9']}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
