#!/usr/bin/env python3
"""Factorized Detector Freeze Pipeline — Stages 0-9.

Stages 0-1: Automated (self-contained).
Stages 2-9: Require pre-existing frozen server artifacts. This script validates
prerequisites and prints the exact commands to run. Codex must execute each
stage on the A800 server with the correct artifact paths.

Config file (--config) provides server-specific paths for Stages 2+.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_SCRIPTS = ROOT / "scripts/detector_v5"
ANALYSIS = ROOT / "analysis/student_trigger_calibration"


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}\n  {label}\n  {' '.join(cmd)}\n{'='*60}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: {label} FAILED (rc={r.returncode})")


def main():
    ap = argparse.ArgumentParser(description="Factorized Detector Freeze Pipeline")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stage", type=int, choices=range(10), default=None)
    ap.add_argument("--clean2000-root", type=str, required=True)
    ap.add_argument("--output-root", type=str, required=True)
    ap.add_argument("--python", type=str, default="python")
    args = ap.parse_args()

    py = args.python
    out = Path(args.output_root)
    clean2000 = args.clean2000_root

    config: dict[str, Any] = {}
    if args.config and args.config.is_file():
        config = json.loads(args.config.read_text())

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
    # Stage 0: CLEAN2000 Provenance Audit (SELF-CONTAINED)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 0):
        _run([py, str(STAGE_SCRIPTS / "audit_clean2000_provenance.py"),
              "--clean2000-root", clean2000,
              "--output-root", str(S["s0"])], "Stage 0: CLEAN2000 Audit")
        print("Stage 0 PASS: registry CSV + provenance receipt produced")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 1: Unified Teacher Labels (needs frozen server artifacts)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 1):
        required = ["registry_root", "decoder_root", "physics_audit_root",
                     "protocol", "k10_root", "expected_k10_schema"]
        missing = [k for k in required if k not in config]
        if missing:
            raise SystemExit(f"CONFIG_MISSING for Stage 1: {missing}\n"
                             f"These are pre-existing frozen server artifacts.")
        _run([py, str(STAGE_SCRIPTS / "build_unified_teacher_labels.py"),
              "--clean2000-audit-root", str(S["s0"]),
              "--registry-root", config["registry_root"],
              "--decoder-root", config["decoder_root"],
              "--physics-audit-root", config["physics_audit_root"],
              "--protocol", config["protocol"],
              "--k10-root", config["k10_root"],
              "--expected-k10-schema", config["expected_k10_schema"],
              "--output-root", str(S["s1"])],
              "Stage 1: Teacher Labels (all 5 splits)")
        print("Stage 1 PASS: Teacher labels for FIT_TRAIN/DEV/CAL/CHECK/H")

    # ═══════════════════════════════════════════════════════════════════
    # Stage 2: Phase B Authoritative Coverage (needs server manifests)
    # ═══════════════════════════════════════════════════════════════════
    if args.stage in (None, 2):
        required_2 = ["identity_source_discovery", "checkpoint_training_ledger"]
        missing_2 = [k for k in required_2 if k not in config]
        if missing_2:
            raise SystemExit(f"CONFIG_MISSING for Stage 2: {missing_2}")
        cmd2 = [py, str(ANALYSIS / "validate_factorized_identity_disjointness.py"),
                "--identity-source-discovery", config["identity_source_discovery"],
                "--checkpoint-training-ledger", config["checkpoint_training_ledger"],
                "--mode", "authoritative",
                "--output-root", str(S["s2"])]
        # Optional but recommended for authoritative mode:
        for opt in ["calibrator_fit_manifest", "policy_selection_manifest",
                     "heldout_l3_manifest", "calibration_teacher_bundle_root",
                     "policy_teacher_bundle_root", "heldout_teacher_bundle_root",
                     "teacher_contract_file"]:
            if opt in config:
                cmd2.extend([f"--{opt.replace('_', '-')}", config[opt]])
        if "require_cp_authorization" in config and config["require_cp_authorization"]:
            cmd2.append("--require-cp-authorization")
        _run(cmd2, "Stage 2: Phase B Authoritative Coverage")
        print("Stage 2 PASS: Phase B identity closure verified")

    # ═══════════════════════════════════════════════════════════════════
    # Stages 3-9: Require frozen server artifacts beyond script scope
    # ═══════════════════════════════════════════════════════════════════
    if args.stage is not None and args.stage >= 3:
        print(f"\nStages 3-9 require server-side GPU execution with frozen artifacts.")
        print(f"Refer to FINAL_DETECTOR_COMPLETION_REPORT.md for stage details.")
        print(f"Key scripts involved:")
        for stage_script in ["train_factorized_v2_inner_cv.py",
                              "freeze_factorized_calibrators.py",
                              "freeze_factorized_scheduler_policy.py",
                              "run_authorized_factorized_heldout_l3.py",
                              "run_full_fit_frozen.py",
                              "build_factorized_detector_freeze.py"]:
            print(f"  {ANALYSIS / stage_script}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
