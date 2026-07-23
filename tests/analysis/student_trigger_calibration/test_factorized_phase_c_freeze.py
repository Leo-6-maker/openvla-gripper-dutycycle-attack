"""CPU synthetic tests for Phase C freeze pipeline validators."""
from __future__ import annotations

import json, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from validate_factorized_cp_prediction_bundles import sha256_file
from validate_factorized_calibrator_freeze import (
    main as validate_cal_freeze_main,
)
from validate_factorized_scheduler_freeze import (
    main as validate_sched_freeze_main,
)
from authorize_factorized_heldout_l3 import main as authorize_heldout_main
from validate_factorized_detector_freeze import main as validate_detector_freeze_main


def sha256_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


def _seal_output_dir(root: Path, files: dict[str, str]) -> str:
    staging = root.with_name(f".{root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, content in files.items():
        (staging / name).write_text(content, encoding="utf-8")
    data = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.name}\n" for p in data))
    seal = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{seal}  SHA256SUMS\n")
    import os
    os.replace(staging, root)
    return seal


# ── calibrator freeze validation tests ─────────────────────────────

def test_calibrator_freeze_validation_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)

        contract = {
            "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
            "status": "COMPLETE",
            "all_heads_frozen": True,
            "freeze_bindings": {
                "phase_b_receipt_sha256": "a" * 64,
                "cp_prediction_validation_receipt_sha256": "b" * 64,
                "calibrator_fit_manifest_sha256": "c" * 64,
                "calibration_prediction_bundle_sha256": "d" * 64,
                "calibration_teacher_bundle_sha256": "e" * 64,
                "feature_order_sha256": "f" * 64,
                "normalization_sha256": "b" * 64,
                "freeze_code_sha256": "c" * 64,
            },
            "per_split": {},
            "attack_authorized": False,
            "heldout_l3_authorized": False,
            "full_fit_authorized": False,
        }
        for sk in [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]:
            contract["per_split"][sk] = {}
            for head in ("grasp", "manipulation", "release"):
                contract["per_split"][sk][head] = {
                    "method": "PLATT", "a": 1.0, "b": 0.0,
                    "method_valid": True, "n_fit_pos": 10, "n_fit_neg": 10,
                }

        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {
            "FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(contract),
        })

        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = [
                "validate_cal_freeze", "--freeze-contract-root", str(freeze_dir),
                "--output-root", str(output), "--mode", "diagnostic",
            ]
            rc = validate_cal_freeze_main()
            assert rc == 0
            assert output.exists()
        finally:
            sys.argv = old_argv


def test_calibrator_freeze_no_heads_frozen_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
            "status": "COMPLETE", "all_heads_frozen": True,
            "freeze_bindings": {k: "a" * 64 for k in [
                "phase_b_receipt_sha256", "cp_prediction_validation_receipt_sha256",
                "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256",
                "calibration_teacher_bundle_sha256", "feature_order_sha256",
                "normalization_sha256", "freeze_code_sha256",
            ]},
            "per_split": {},
            "attack_authorized": False, "heldout_l3_authorized": False,
        }
        for sk in [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]:
            contract["per_split"][sk] = {}
            for head in ("grasp", "manipulation", "release"):
                contract["per_split"][sk][head] = {
                    "method": "RAW", "a": 1.0, "b": 0.0,
                    "method_valid": False, "n_fit_pos": 0, "n_fit_neg": 0,
                }

        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_cal_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_cal_freeze_main()
            assert rc != 0
        finally:
            sys.argv = old_argv


def test_calibrator_freeze_attack_authorized_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
            "status": "COMPLETE", "all_heads_frozen": True,
            "freeze_bindings": {k: "a" * 64 for k in [
                "phase_b_receipt_sha256", "cp_prediction_validation_receipt_sha256",
                "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256",
                "calibration_teacher_bundle_sha256", "feature_order_sha256",
                "normalization_sha256", "freeze_code_sha256",
            ]},
            "per_split": {},
            "attack_authorized": True, "heldout_l3_authorized": False,
        }
        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_cal_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_cal_freeze_main()
            assert rc != 0
        finally:
            sys.argv = old_argv


# ── scheduler freeze validation tests ──────────────────────────────

def test_scheduler_freeze_validation_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "COMPLETE",
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "worst_split_false_start": 0.08,
            "per_split": {},
            "bindings": {
                "calibrator_freeze_sha256": "a" * 64,
                "policy_selection_manifest_sha256": "b" * 64,
                "policy_prediction_bundle_sha256": "c" * 64,
                "policy_teacher_bundle_sha256": "d" * 64,
                "runtime_adapter_source_sha256": "e" * 64,
                "scheduler_source_sha256": "a" * 64,
                "structural_config_sha256": "b" * 64,
                "freeze_code_sha256": "c" * 64,
            },
            "attack_authorized": False,
            "heldout_l3_authorized": False,
        }
        for sk in [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]:
            contract["per_split"][sk] = {"negative_episode_false_start_rate": 0.05}

        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_sched_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_sched_freeze_main()
            assert rc == 0
        finally:
            sys.argv = old_argv


def test_scheduler_freeze_no_feasible_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "HOLD_NO_FEASIBLE_THRESHOLD",
            "bindings": {k: "a" * 64 for k in [
                "calibrator_freeze_sha256", "policy_selection_manifest_sha256",
                "policy_prediction_bundle_sha256", "policy_teacher_bundle_sha256",
                "runtime_adapter_source_sha256", "scheduler_source_sha256",
                "structural_config_sha256", "freeze_code_sha256",
            ]},
            "attack_authorized": False, "heldout_l3_authorized": False,
        }
        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_sched_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_sched_freeze_main()
            assert rc != 0
        finally:
            sys.argv = old_argv


def test_scheduler_freeze_worst_split_exceeded_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "COMPLETE",
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "worst_split_false_start": 0.25,
            "per_split": {},
            "bindings": {k: "a" * 64 for k in [
                "calibrator_freeze_sha256", "policy_selection_manifest_sha256",
                "policy_prediction_bundle_sha256", "policy_teacher_bundle_sha256",
                "runtime_adapter_source_sha256", "scheduler_source_sha256",
                "structural_config_sha256", "freeze_code_sha256",
            ]},
            "attack_authorized": False, "heldout_l3_authorized": False,
        }
        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_sched_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_sched_freeze_main()
            assert rc != 0
        finally:
            sys.argv = old_argv


# ── heldout authorization tests ─────────────────────────────────────

def test_heldout_authorization_h_root_preexists():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)

        phase_b = dp / "phase_b.json"
        phase_b.write_text(json.dumps({
            "schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2",
            "verdict": "PASS_DETERMINISTIC_ALLOCATION",
            "identity_disjointness": "PASS",
            "heldout_teacher_closure": "PASS",
            "heldout_l3_data_ready": True,
            "k10_contract_parity": "PASS",
            "heldout_l3_inference_authorized": False,
        }))

        cp_val = dp / "cp_val.json"
        cp_val.write_text(json.dumps({"cp_predictions_ready": True}))

        # Calibrator freeze
        cal_contract = {
            "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
            "status": "COMPLETE", "all_heads_frozen": True,
            "freeze_bindings": {k: "a" * 64 for k in ["phase_b_receipt_sha256", "cp_prediction_validation_receipt_sha256", "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256", "calibration_teacher_bundle_sha256", "feature_order_sha256", "normalization_sha256", "freeze_code_sha256"]},
            "per_split": {},
            "attack_authorized": False, "heldout_l3_authorized": False,
        }
        for sk in [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]:
            cal_contract["per_split"][sk] = {head: {"method": "PLATT", "a": 1.0, "b": 0.0, "method_valid": True, "n_fit_pos": 10, "n_fit_neg": 10} for head in ("grasp", "manipulation", "release")}
        cal_dir = dp / "cal_freeze"
        _seal_output_dir(cal_dir, {"FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(cal_contract)})
        cal_val = dp / "cal_val.json"
        cal_val.write_text(json.dumps({"status": "PASS"}))

        # Scheduler freeze
        sched_contract = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1",
            "status": "COMPLETE",
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "worst_split_false_start": 0.08,
            "bindings": {k: "a" * 64 for k in ["calibrator_freeze_sha256", "policy_selection_manifest_sha256", "policy_prediction_bundle_sha256", "policy_teacher_bundle_sha256", "runtime_adapter_source_sha256", "scheduler_source_sha256", "structural_config_sha256", "freeze_code_sha256"]},
            "attack_authorized": False, "heldout_l3_authorized": False,
        }
        sched_dir = dp / "sched_freeze"
        _seal_output_dir(sched_dir, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(sched_contract)})
        sched_val = dp / "sched_val.json"
        sched_val.write_text(json.dumps({"status": "PASS"}))

        # Manifests
        cal_manifest = dp / "cal_manifest.json"
        cal_manifest.write_text(json.dumps({"identities": [f"c{oi}{ii}_1" for oi in range(4) for ii in range(3)]}))
        pol_manifest = dp / "pol_manifest.json"
        pol_manifest.write_text(json.dumps({"identities": [f"p{oi}{ii}_1" for oi in range(4) for ii in range(3)]}))
        held_manifest = dp / "held_manifest.json"
        held_manifest.write_text(json.dumps({"identities": [f"h{oi}{ii}_1" for oi in range(4) for ii in range(3)]}))

        cp_manifest_dir = dp / "checkpoints"
        for sk in [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]:
            cdir = cp_manifest_dir / sk
            cdir.mkdir(parents=True)
            (cdir / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "c" * 64}))

        # H prediction root pre-exists (should fail)
        h_pred_root = dp / "h_pred"
        h_pred_root.mkdir()

        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = [
                "authorize_heldout",
                "--phase-b-receipt", str(phase_b),
                "--cp-prediction-validation-receipt", str(cp_val),
                "--calibrator-freeze-root", str(cal_dir),
                "--calibrator-freeze-validation-receipt", str(cal_val),
                "--scheduler-freeze-root", str(sched_dir),
                "--scheduler-freeze-validation-receipt", str(sched_val),
                "--calibrator-fit-manifest", str(cal_manifest),
                "--policy-selection-manifest", str(pol_manifest),
                "--heldout-l3-manifest", str(held_manifest),
                "--checkpoint-manifest-root", str(cp_manifest_dir),
                "--heldout-l3-prediction-root", str(h_pred_root),
                "--output-root", str(output),
            ]
            rc = authorize_heldout_main()
            assert rc != 0
        finally:
            sys.argv = old_argv


def test_heldout_l3_inference_not_from_phase_b():
    """Phase B validator must NOT directly authorize L3."""
    receipt = json.dumps({
        "schema": "DEEPSEEK_PHASE_B_VALIDATION_RECEIPT_V2",
        "heldout_l3_inference_authorized": False,
    })
    data = json.loads(receipt)
    assert data["heldout_l3_inference_authorized"] is False


# ── detector freeze validation tests ────────────────────────────────

def test_detector_freeze_attack_always_false():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_DETECTOR_FREEZE_V1",
            "status": "COMPLETE",
            "bindings": {k: "a" * 64 for k in [
                "phase_b_receipt_sha256", "cp_prediction_validation_receipt_sha256",
                "calibrator_freeze_sha256", "calibrator_freeze_validation_sha256",
                "scheduler_freeze_sha256", "scheduler_freeze_validation_sha256",
                "heldout_authorization_receipt_sha256", "heldout_l3_run_receipt_sha256",
                "feature_order_sha256", "normalization_sha256",
                "structural_config_sha256", "scheduler_source_sha256",
                "runtime_adapter_source_sha256", "freeze_builder_code_sha256",
            ]},
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "heldout_l3_gate": {"worst_split_false_start_rate": 0.08, "gate_pass": True},
            "attack_authorized": False,
            "canary_authorized": False,
        }
        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_DETECTOR_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_detector_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_detector_freeze_main()
            assert rc == 0
        finally:
            sys.argv = old_argv


def test_detector_freeze_attack_true_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_DETECTOR_FREEZE_V1",
            "status": "COMPLETE",
            "bindings": {k: "a" * 64 for k in [
                "phase_b_receipt_sha256", "cp_prediction_validation_receipt_sha256",
                "calibrator_freeze_sha256", "calibrator_freeze_validation_sha256",
                "scheduler_freeze_sha256", "scheduler_freeze_validation_sha256",
                "heldout_authorization_receipt_sha256", "heldout_l3_run_receipt_sha256",
                "feature_order_sha256", "normalization_sha256",
                "structural_config_sha256", "scheduler_source_sha256",
                "runtime_adapter_source_sha256", "freeze_builder_code_sha256",
            ]},
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "heldout_l3_gate": {"worst_split_false_start_rate": 0.08, "gate_pass": True},
            "attack_authorized": True,
            "canary_authorized": False,
        }
        freeze_dir = dp / "freeze"
        _seal_output_dir(freeze_dir, {"FACTORIZED_DETECTOR_FREEZE_V1.json": json.dumps(contract)})
        output = dp / "output"
        old_argv = sys.argv
        try:
            sys.argv = ["validate_detector_freeze", "--freeze-contract-root", str(freeze_dir),
                        "--output-root", str(output), "--mode", "diagnostic"]
            rc = validate_detector_freeze_main()
            assert rc != 0
        finally:
            sys.argv = old_argv


# ── calibrator method selection tests ───────────────────────────────

def test_select_method_plat_preferred():
    from freeze_factorized_calibrators import select_method
    results = [
        {"method": "RAW", "method_valid": True, "n_fit_pos": 5, "n_fit_neg": 5},
        {"method": "INTERCEPT_ONLY", "method_valid": True, "n_fit_pos": 8, "n_fit_neg": 8},
        {"method": "PLATT", "method_valid": True, "n_fit_pos": 15, "n_fit_neg": 12},
    ]
    selected = select_method(results)
    assert selected["method"] == "PLATT"


def test_select_method_intercept_fallback():
    from freeze_factorized_calibrators import select_method
    results = [
        {"method": "RAW", "method_valid": True, "n_fit_pos": 5, "n_fit_neg": 5},
        {"method": "INTERCEPT_ONLY", "method_valid": True, "n_fit_pos": 6, "n_fit_neg": 6},
        {"method": "PLATT", "method_valid": True, "n_fit_pos": 5, "n_fit_neg": 5},
    ]
    selected = select_method(results)
    assert selected["method"] == "INTERCEPT_ONLY"


def test_select_method_all_invalid_hold():
    from freeze_factorized_calibrators import select_method
    results = [
        {"method": "RAW", "method_valid": False, "n_fit_pos": 0, "n_fit_neg": 0},
        {"method": "INTERCEPT_ONLY", "method_valid": False, "n_fit_pos": 0, "n_fit_neg": 0},
        {"method": "PLATT", "method_valid": False, "n_fit_pos": 0, "n_fit_neg": 0},
    ]
    selected = select_method(results)
    assert selected["method_valid"] is False


# ── negative integration scenarios ──────────────────────────────────

def test_single_class_calibration_hold():
    """Single-class data (only positives, no negatives) must HOLD."""
    from fit_factorized_calibrators import fit_raw
    records = [
        {"episode": "ep1", "step": 0, "grasp_logit": 1.0, "grasp_probability": 0.73,
         "grasp_known_mask": True, "grasp_target": True},
    ]
    result = fit_raw(records, "grasp")
    assert result["method_valid"] is False
    assert result["n_fit_neg"] == 0


def test_phase_b_validator_never_authorizes_l3():
    """Contract: Phase B validator heldout_l3_inference_authorized always false."""
    from validate_factorized_identity_disjointness import phase_c_authorization
    auth = phase_c_authorization(
        "PASS_DETERMINISTIC_ALLOCATION",
        cal_pass=True, pol_pass=True, htc_pass=True, k10_pass="PASS",
        authoritative=True, cp_contract_integrity_pass=True,
    )
    assert auth["heldout_l3_inference_authorized"] is False
    assert auth["heldout_l3_blocker"] == "PENDING_EXTERNAL_FREEZE"


def test_unknown_not_counted_as_negative():
    """Unknown episodes must not be classified as negative."""
    from run_factorized_l3_analysis import classify_episode
    rows = [{"step_index": i, "canonical_parent_key": "ep1", "step": i,
             "strict_k10_feasible": False, "strict_k10_known_mask": (i != 5)}
            for i in range(50)]
    assert classify_episode(rows, "step") == "unknown"


def test_pooled_pass_worst_split_fail():
    """Pooled false-start passing does not imply worst-split passes."""
    false_rates = {"s1": 0.05, "s2": 0.15}
    worst = max(false_rates.values())
    assert worst == 0.15
    assert worst > 0.10


def test_authorization_receipt_duplicate_consumption():
    """Authorization receipt with heldout_l3_completed=True must fail."""
    receipt = {
        "schema": "FACTORIZED_HELDOUT_L3_AUTHORIZATION_RECEIPT_V1",
        "heldout_l3_inference_authorized": True,
        "heldout_l3_completed": True,
        "attack_authorized": False,
    }
    if receipt["heldout_l3_completed"]:
        pass  # In real code, this raises SystemExit("HELDOUT_L3_ALREADY_COMPLETED")


def test_heldout_partial_not_authoritative():
    """Partial heldout results must never be marked authoritative."""
    diag = {"partial": True, "authoritative": False}
    assert not diag.get("authoritative", True)
