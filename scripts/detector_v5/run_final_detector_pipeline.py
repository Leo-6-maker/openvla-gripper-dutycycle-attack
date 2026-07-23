#!/usr/bin/env python3
"""Master orchestrator: Factorized Detector Freeze Pipeline (Stages 0-9).

Self-derives all intermediate artifact paths from stage outputs.
Only requires truly external inputs in the pipeline config.
Fail-closed: any stage or its validator fails → immediate halt.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_SCRIPTS = ROOT / "scripts/detector_v5"
ANALYSIS = ROOT / "analysis/student_trigger_calibration"
PILOT = ROOT / "analysis/pilot_attack"

# ═══════════════════════════════════════════════════════════════════════════
# Pipeline config — only truly external inputs. All intermediate paths
# are derived automatically from the output root.
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "clean2000_root": "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716",
    "output_root": "/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline",
    "python": "python",
    "gpu": 0,
    "identity_source_discovery": None,      # identity split manifest root (sealed)
    "checkpoint_training_ledger": None,      # checkpoint training ledger root (sealed)
    "inner_cv_splits_root": None,            # inner-CV split definitions root (sealed)
    "feature_order_contract": None,          # 25D feature schema file
    "normalization_contract": None,          # normalization artifact file
    "calibrator_fit_manifest": None,         # calibrator fit manifest root (sealed)
    "policy_selection_manifest": None,       # scheduler policy selection manifest root (sealed)
    "heldout_l3_manifest": None,             # H L3 evaluation manifest root (sealed)
    "runtime_wrapper_path": None,            # exact runtime wrapper path
    "runtime_bundle_builder": None,          # path to runtime bundle builder script
}


def _require(paths: dict, key: str) -> str:
    v = paths.get(key)
    if v is None:
        raise SystemExit(f"CONFIG_MISSING: '{key}' — must be provided in pipeline config")
    return str(v)


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}\n  {label}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: {label} FAILED (rc={r.returncode})")


def main():
    ap = argparse.ArgumentParser(description="Factorized Detector Freeze Pipeline")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON config with external paths (uses defaults if omitted)")
    ap.add_argument("--stage", type=int, choices=range(10), default=None)
    ap.add_argument("--clean2000-root", type=str, default=None)
    ap.add_argument("--output-root", type=str, default=None)
    ap.add_argument("--python", type=str, default=None)
    args = ap.parse_args()

    paths: dict[str, Any] = dict(DEFAULT_CONFIG)
    if args.config and args.config.is_file():
        paths.update(json.loads(args.config.read_text()).get("paths", {}))
    if args.clean2000_root: paths["clean2000_root"] = args.clean2000_root
    if args.output_root: paths["output_root"] = args.output_root
    if args.python: paths["python"] = args.python

    py = paths["python"]
    out = Path(_require(paths, "output_root"))
    clean2000 = _require(paths, "clean2000_root")

    # ── Stage output directories (auto-derived) ──────────────────────
    S = {
        "s0": out / "stage_0_clean2000_audit",
        "s1": out / "stage_1_teacher_labels",
        "s2": out / "stage_2_phase_b",
        "s3": out / "stage_3_student_training",
        "s4": out / "stage_4_cp_inference",
        "s5": out / "stage_5_calibrator_freeze",
        "s5v": out / "stage_5v_calibrator_validation",
        "s6": out / "stage_6_scheduler_freeze",
        "s6v": out / "stage_6v_scheduler_validation",
        "s7a": out / "stage_7a_h_auth",
        "s7b": out / "stage_7b_h_evaluation",
        "s8": out / "stage_8_full_fit",
        "s9": out / "FINAL_FACTORIZED_DETECTOR_V1",
    }

    # ═══════════════════════════════════════════════════════════════════
    # Stage 0: CLEAN2000 Provenance Audit (uses pilot_integrity pattern)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 0):
        _run([py, str(STAGE_SCRIPTS / "audit_clean2000_provenance.py"),
              "--clean2000-root", clean2000,
              "--output-root", str(S["s0"])], "Stage 0: CLEAN2000 Audit")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Unified Teacher Labels
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 1):
        _run([py, str(STAGE_SCRIPTS / "build_unified_teacher_labels.py"),
              "--clean2000-root", clean2000,
              "--clean2000-audit-root", str(S["s0"]),
              "--output-root", str(S["s1"])], "Stage 1: Teacher Labels")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Phase B Authoritative Coverage
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 2):
        _run([py, str(ANALYSIS / "validate_factorized_identity_disjointness.py"),
              "--identity-source-discovery", _require(paths, "identity_source_discovery"),
              "--checkpoint-training-ledger", _require(paths, "checkpoint_training_ledger"),
              "--teacher-labels-root", str(S["s1"]),
              "--output-root", str(S["s2"])], "Stage 2: Phase B Coverage")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 3: Student Training (V2B formal, NOT engineering sidecar)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 3):
        inner_cv = _require(paths, "inner_cv_splits_root")
        auth_root = paths.get("reference_authorization_root") or str(S["s2"])
        S["s3"].mkdir(parents=True)
        # Train 12 folds (4 outer × 3 inner) — single-GPU sequential
        for outer_fold in range(4):
            for inner_fold in range(3):
                for seed in [42]:
                    fold_dir = S["s3"] / f"V2B_W32_H64_D0.1_WD1e-4_o{outer_fold}_i{inner_fold}_s{seed}"
                    if fold_dir.is_dir():
                        continue  # resume
                    _run([py, str(STAGE_SCRIPTS / "train_factorized_v2_inner_cv.py"),
                          "--candidate", "V2B",
                          "--outer-fold", str(outer_fold),
                          "--inner-fold", str(inner_fold),
                          "--seed", str(seed),
                          "--gpu", str(paths.get("gpu", 0)),
                          "--receptive-field", "32",
                          "--hidden-dim", "64",
                          "--dropout", "0.1",
                          "--weight-decay", "1e-4",
                          "--epochs", "30",
                          "--output-root", str(fold_dir),
                          "--inner-cv-splits-root", inner_cv,
                          "--authorization-root", auth_root],
                         f"Stage 3: Train fold {outer_fold}.{inner_fold}")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 4: C/P Production Inference
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 4):
        _run([py, str(STAGE_SCRIPTS / "predict_factorized_v2_recommended_canary.py"),
              "--checkpoint-dir", str(S["s3"]),
              "--inner-cv-splits-root", _require(paths, "inner_cv_splits_root"),
              "--output-root", str(S["s4"])], "Stage 4: C/P Inference")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 5: Calibrator Freeze + Validation
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 5):
        _run([py, str(ANALYSIS / "freeze_factorized_calibrators.py"),
              "--calibrator-fit-manifest", _require(paths, "calibrator_fit_manifest"),
              "--calibration-prediction-bundle-root", str(S["s4"] / "cal"),
              "--calibration-teacher-bundle-root", str(S["s1"] / "cal"),
              "--phase-b-validation-root", str(S["s2"]),
              "--cp-prediction-validation-root", str(S["s4"]),  # auto-validated in stage
              "--checkpoint-manifest-root", str(S["s3"]),
              "--checkpoint-training-ledger", _require(paths, "checkpoint_training_ledger"),
              "--feature-order-contract", _require(paths, "feature_order_contract"),
              "--normalization-contract", _require(paths, "normalization_contract"),
              "--output-root", str(S["s5"])], "Stage 5: Calibrator Freeze")
        _run([py, str(ANALYSIS / "validate_factorized_calibrator_freeze.py"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--output-root", str(S["s5v"])], "Stage 5v: Calibrator Validation")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 6: Scheduler Freeze + Validation
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 6):
        _run([py, str(ANALYSIS / "freeze_factorized_scheduler_policy.py"),
              "--policy-selection-manifest", _require(paths, "policy_selection_manifest"),
              "--policy-prediction-bundle-root", str(S["s4"] / "check"),
              "--policy-teacher-bundle-root", str(S["s1"] / "check"),
              "--policy-runtime-bundle-root", str(S["s4"] / "check_runtime"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--phase-b-validation-root", str(S["s2"]),
              "--cp-prediction-validation-root", str(S["s4"]),
              "--calibrator-freeze-validation-root", str(S["s5v"]),
              "--calibrator-fit-manifest", _require(paths, "calibrator_fit_manifest"),
              "--checkpoint-manifest-root", str(S["s3"]),
              "--output-root", str(S["s6"])], "Stage 6: Scheduler Freeze")
        _run([py, str(ANALYSIS / "validate_factorized_scheduler_freeze.py"),
              "--scheduler-freeze-root", str(S["s6"]),
              "--output-root", str(S["s6v"])], "Stage 6v: Scheduler Validation")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 7a: H Heldout Authorization
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 7):
        _run([py, str(ANALYSIS / "authorize_factorized_heldout_l3.py"),
              "--heldout-prediction-authorization-root", str(S["s4"] / "h_auth"),
              "--heldout-prediction-validation-root", str(S["s4"] / "h_val"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--calibrator-freeze-validation-root", str(S["s5v"]),
              "--scheduler-freeze-root", str(S["s6"]),
              "--scheduler-freeze-validation-root", str(S["s6v"]),
              "--calibrator-fit-manifest", _require(paths, "calibrator_fit_manifest"),
              "--policy-selection-manifest", _require(paths, "policy_selection_manifest"),
              "--heldout-l3-manifest", _require(paths, "heldout_l3_manifest"),
              "--checkpoint-manifest-root", str(S["s3"]),
              "--authorized-l3-evaluation-output-root", str(S["s7a"]),
              "--output-root", str(S["s7a"])], "Stage 7a: H Authorization")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 7b: H Heldout Evaluation (ONE-SHOT)
    # ═══════════════════════════════════════════════════════════════════
        _run([py, str(ANALYSIS / "run_authorized_factorized_heldout_l3.py"),
              "--heldout-l3-evaluation-authorization-root", str(S["s7a"]),
              "--heldout-prediction-authorization-root", str(S["s4"] / "h_auth"),
              "--heldout-prediction-validation-root", str(S["s4"] / "h_val"),
              "--heldout-prediction-bundle-root", str(S["s4"] / "h_predictions"),
              "--heldout-teacher-bundle-root", str(S["s1"] / "h"),
              "--heldout-runtime-bundle-root", str(S["s4"] / "h_runtime"),
              "--calibrator-freeze-root", str(S["s5"]),
              "--scheduler-freeze-root", str(S["s6"]),
              "--heldout-l3-manifest", _require(paths, "heldout_l3_manifest"),
              "--calibrator-fit-manifest", _require(paths, "calibrator_fit_manifest"),
              "--policy-selection-manifest", _require(paths, "policy_selection_manifest"),
              "--output-root", str(S["s7b"]),
              "--claim-root", str(S["s7b"] / "claim")], "Stage 7b: H Evaluation")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 8: Full-FIT (requires H gate_pass=true)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 8):
        _run([py, str(STAGE_SCRIPTS / "run_full_fit_frozen.py"),
              "--teacher-labels-root", str(S["s1"]),
              "--student-config-root", str(S["s3"]),
              "--stage-7-h-receipt-root", str(S["s7b"]),
              "--output-root", str(S["s8"])], "Stage 8: Full-FIT")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 9: Final Detector Freeze (Full-FIT and H mandatory)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 9):
        _run([py, str(ANALYSIS / "build_factorized_detector_freeze.py"),
              "--phase-b-validation-root", str(S["s2"]),
              "--cp-prediction-validation-root", str(S["s4"]),
              "--calibrator-freeze-root", str(S["s5"]),
              "--calibrator-freeze-validation-root", str(S["s5v"]),
              "--scheduler-freeze-root", str(S["s6"]),
              "--scheduler-freeze-validation-root", str(S["s6v"]),
              "--heldout-prediction-authorization-root", str(S["s4"] / "h_auth"),
              "--heldout-prediction-validation-root", str(S["s4"] / "h_val"),
              "--heldout-l3-evaluation-authorization-root", str(S["s7a"]),
              "--heldout-l3-run-root", str(S["s7b"]),
              "--checkpoint-manifest-root", str(S["s8"]),  # Full-FIT checkpoint
              "--feature-order-contract", _require(paths, "feature_order_contract"),
              "--normalization-contract", _require(paths, "normalization_contract"),
              "--output-root", str(S["s9"])], "Stage 9: Final Detector Freeze")

    print(f"\n{'='*60}\n  PIPELINE COMPLETE\n  Detector: {S['s9']}\n{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
