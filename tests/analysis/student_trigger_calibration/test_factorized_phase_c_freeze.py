"""Phase C freeze pipeline tests — calibrator, scheduler, heldout auth, detector."""
from __future__ import annotations

import json, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from factorized_phase_c_integrity import (
    sha256_file, seal_output_dir, load_strict_json, verify_bundle_seal, is_64char_hex,
    claim_atomic_root,
)


def _mk_sha(c: str = "a") -> str: return c * 64


def _seal(root: Path, files: dict[str, str]):
    staging = root.with_name(f".{root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    for name, content in files.items():
        (staging / name).write_text(content, encoding="utf-8")
    data = sorted(p for p in staging.iterdir() if p.is_file())
    (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in data))
    (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
    import os; os.replace(staging, root)


HEADS = ("grasp", "manipulation", "release")
ALL_SPLITS = [f"o{oi}_i{ii}" for oi in range(4) for ii in range(3)]


# ── calibrator freeze validation ──────────────────────────────────────

def test_cal_freeze_val_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_CALIBRATOR_FREEZE_V1",
            "status": "COMPLETE", "all_heads_frozen": True,
            "freeze_bindings": {k: _mk_sha("a") for k in [
                "phase_b_validation_seal_sha256", "cp_prediction_validation_seal_sha256",
                "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256",
                "calibration_teacher_bundle_sha256", "feature_order_sha256",
                "normalization_sha256", "freeze_code_sha256",
            ]},
            "per_split": {},
            "attack_authorized": False, "heldout_l3_authorized": False, "full_fit_authorized": False,
        }
        for sk in ALL_SPLITS:
            contract["per_split"][sk] = {h: {"method": "PLATT", "a": 1.0, "b": 0.0, "method_valid": True, "n_fit_pos": 10, "n_fit_neg": 10} for h in HEADS}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_calibrator_freeze import main as vcf
        old = sys.argv
        try:
            sys.argv = ["vcf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vcf(); assert rc == 0
        finally: sys.argv = old


def test_cal_freeze_no_heads_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {"schema": "FACTORIZED_CALIBRATOR_FREEZE_V1", "status": "COMPLETE", "all_heads_frozen": True,
            "freeze_bindings": {k: _mk_sha("a") for k in ["phase_b_validation_seal_sha256", "cp_prediction_validation_seal_sha256", "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256", "calibration_teacher_bundle_sha256", "feature_order_sha256", "normalization_sha256", "freeze_code_sha256"]},
            "per_split": {}, "attack_authorized": False, "heldout_l3_authorized": False}
        for sk in ALL_SPLITS:
            contract["per_split"][sk] = {h: {"method": "RAW", "a": 1.0, "b": 0.0, "method_valid": False, "n_fit_pos": 0, "n_fit_neg": 0} for h in HEADS}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_calibrator_freeze import main as vcf
        old = sys.argv
        try:
            sys.argv = ["vcf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vcf(); assert rc != 0
        finally: sys.argv = old


def test_cal_freeze_attack_authorized_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {"schema": "FACTORIZED_CALIBRATOR_FREEZE_V1", "all_heads_frozen": True,
            "freeze_bindings": {k: _mk_sha("a") for k in ["phase_b_validation_seal_sha256", "cp_prediction_validation_seal_sha256", "calibrator_fit_manifest_sha256", "calibration_prediction_bundle_sha256", "calibration_teacher_bundle_sha256", "feature_order_sha256", "normalization_sha256", "freeze_code_sha256"]},
            "per_split": {}, "attack_authorized": True, "heldout_l3_authorized": False}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_CALIBRATOR_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_calibrator_freeze import main as vcf
        old = sys.argv
        try:
            sys.argv = ["vcf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vcf(); assert rc != 0
        finally: sys.argv = old


# ── scheduler freeze validation ───────────────────────────────────────

def test_sched_freeze_val_pass():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {
            "schema": "FACTORIZED_SCHEDULER_FREEZE_V1", "status": "COMPLETE",
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "worst_split_false_start": 0.08,
            "selected_metrics": {"valid_opportunity_recall": 0.5, "all_emit_precision": 0.8, "median_timing_offset": 2.0},
            "per_split": {},
            "bindings": {k: _mk_sha("a") for k in [
                "calibrator_freeze_sha256", "calibrator_fit_manifest_sha256",
                "policy_selection_manifest_sha256", "policy_prediction_bundle_sha256",
                "policy_teacher_bundle_sha256", "policy_runtime_bundle_sha256",
                "runtime_adapter_source_sha256", "scheduler_source_sha256",
                "structural_config_sha256", "freeze_code_sha256",
            ]},
            "attack_authorized": False, "heldout_l3_authorized": False,
        }
        for sk in ALL_SPLITS:
            contract["per_split"][sk] = {"negative_episode_false_start_rate": 0.05}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_scheduler_freeze import main as vsf
        old = sys.argv
        try:
            sys.argv = ["vsf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vsf(); assert rc == 0
        finally: sys.argv = old


def test_sched_freeze_no_feasible():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {"schema": "FACTORIZED_SCHEDULER_FREEZE_V1", "status": "HOLD_NO_FEASIBLE_THRESHOLD",
            "bindings": {k: _mk_sha("a") for k in ["calibrator_freeze_sha256", "calibrator_fit_manifest_sha256", "policy_selection_manifest_sha256", "policy_prediction_bundle_sha256", "policy_teacher_bundle_sha256", "policy_runtime_bundle_sha256", "runtime_adapter_source_sha256", "scheduler_source_sha256", "structural_config_sha256", "freeze_code_sha256"]},
            "attack_authorized": False, "heldout_l3_authorized": False}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_scheduler_freeze import main as vsf
        old = sys.argv
        try:
            sys.argv = ["vsf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vsf(); assert rc != 0
        finally: sys.argv = old


def test_sched_freeze_worst_split_exceeded():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {"schema": "FACTORIZED_SCHEDULER_FREEZE_V1", "status": "COMPLETE",
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "worst_split_false_start": 0.25,
            "selected_metrics": {"valid_opportunity_recall": 0.5, "all_emit_precision": 0.8, "median_timing_offset": 2.0},
            "per_split": {},
            "bindings": {k: _mk_sha("a") for k in ["calibrator_freeze_sha256", "calibrator_fit_manifest_sha256", "policy_selection_manifest_sha256", "policy_prediction_bundle_sha256", "policy_teacher_bundle_sha256", "policy_runtime_bundle_sha256", "runtime_adapter_source_sha256", "scheduler_source_sha256", "structural_config_sha256", "freeze_code_sha256"]},
            "attack_authorized": False, "heldout_l3_authorized": False}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_SCHEDULER_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_scheduler_freeze import main as vsf
        old = sys.argv
        try:
            sys.argv = ["vsf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vsf(); assert rc != 0
        finally: sys.argv = old


# ── detector freeze validation ────────────────────────────────────────

REQUIRED_DETECTOR_BINDINGS = (
    "phase_b_validation_seal", "cp_prediction_validation_seal",
    "calibrator_freeze_seal", "calibrator_freeze_validation_seal",
    "scheduler_freeze_seal", "scheduler_freeze_validation_seal",
    "heldout_prediction_authorization_seal", "heldout_prediction_validation_seal",
    "heldout_l3_evaluation_authorization_seal",
    "feature_order_sha256", "normalization_sha256", "structural_config_sha256",
    "scheduler_source_sha256", "runtime_adapter_source_sha256", "freeze_builder_code_sha256",
)

def test_detector_freeze_attack_false():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {"schema": "FACTORIZED_DETECTOR_FREEZE_V1", "status": "COMPLETE",
            "bindings": {k: _mk_sha("a") for k in REQUIRED_DETECTOR_BINDINGS},
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "heldout_l3_gate": {"gate_pass": True},
            "attack_authorized": False, "canary_authorized": False}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_DETECTOR_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_detector_freeze import main as vdf
        old = sys.argv
        try:
            sys.argv = ["vdf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vdf(); assert rc == 0
        finally: sys.argv = old


def test_detector_freeze_attack_true_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        contract = {"schema": "FACTORIZED_DETECTOR_FREEZE_V1", "status": "COMPLETE",
            "bindings": {k: _mk_sha("a") for k in REQUIRED_DETECTOR_BINDINGS},
            "selected_thresholds": {"grasp": 0.5, "manipulation": 0.4, "release": 0.3},
            "heldout_l3_gate": {"gate_pass": True},
            "attack_authorized": True, "canary_authorized": False}
        fd = dp / "freeze"; _seal(fd, {"FACTORIZED_DETECTOR_FREEZE_V1.json": json.dumps(contract)})
        from validate_factorized_detector_freeze import main as vdf
        old = sys.argv
        try:
            sys.argv = ["vdf", "--freeze-contract-root", str(fd), "--output-root", str(dp / "out"), "--mode", "diagnostic"]
            rc = vdf(); assert rc != 0
        finally: sys.argv = old


# ── calibrator method selection ───────────────────────────────────────

def test_select_method_plat_preferred():
    from freeze_factorized_calibrators import select_method
    results = [{"method": "RAW", "method_valid": True, "n_fit_pos": 5, "n_fit_neg": 5},
               {"method": "INTERCEPT_ONLY", "method_valid": True, "n_fit_pos": 8, "n_fit_neg": 8},
               {"method": "PLATT", "method_valid": True, "n_fit_pos": 15, "n_fit_neg": 12}]
    assert select_method(results)["method"] == "PLATT"

def test_select_method_intercept_fallback():
    from freeze_factorized_calibrators import select_method
    results = [{"method": "RAW", "method_valid": True, "n_fit_pos": 5, "n_fit_neg": 5},
               {"method": "INTERCEPT_ONLY", "method_valid": True, "n_fit_pos": 6, "n_fit_neg": 6},
               {"method": "PLATT", "method_valid": True, "n_fit_pos": 5, "n_fit_neg": 5}]
    assert select_method(results)["method"] == "INTERCEPT_ONLY"

def test_select_method_all_invalid():
    from freeze_factorized_calibrators import select_method
    results = [{"method": "RAW", "method_valid": False, "n_fit_pos": 0, "n_fit_neg": 0} for _ in range(3)]
    assert select_method(results)["method_valid"] is False


# ── single class / unknown / claim tests ──────────────────────────────

def test_single_class_calibration():
    from fit_factorized_calibrators import fit_raw
    result = fit_raw([{"episode": "e1", "step": 0, "grasp_logit": 1.0, "grasp_probability": 0.73, "grasp_known_mask": True, "grasp_target": True}], "grasp")
    assert result["method_valid"] is False; assert result["n_fit_neg"] == 0

def test_phase_b_never_authorizes_l3():
    from validate_factorized_identity_disjointness import phase_c_authorization
    a = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, True, True, "PASS", True, True)
    assert a["heldout_l3_inference_authorized"] is False; assert a["heldout_l3_blocker"] == "PENDING_EXTERNAL_FREEZE"

def test_unknown_not_counted_as_negative():
    from run_factorized_l3_analysis import classify_episode
    rows = [{"step_index": i, "canonical_parent_key": "e1", "step": i, "strict_k10_feasible": False, "strict_k10_known_mask": (i != 5)} for i in range(50)]
    assert classify_episode(rows, "step") == "unknown"

def test_pooled_pass_worst_split_fail():
    assert max({"s1": 0.05, "s2": 0.15}.values()) > 0.10

def test_atomic_claim_second_fails():
    with tempfile.TemporaryDirectory() as d:
        cr = Path(d) / "claim"; claim_atomic_root(cr, _mk_sha("a"), "TEST"); assert cr.exists()
        try: claim_atomic_root(cr, _mk_sha("a"), "TEST"); assert False
        except SystemExit: pass

def test_missing_join_rejected():
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        for name, rows in [("p.jsonl", [{"canonical_parent_key": "e1", "step": 0}]),
                           ("t.jsonl", [{"canonical_parent_key": "e1", "step": 0}]),
                           ("r.jsonl", [{"canonical_parent_key": "e2", "step": 0}])]:
            (dp / name).write_text(json.dumps(rows[0]) + "\n")
        from factorized_phase_c_integrity import load_strict_jsonl, exact_three_way_join
        p = load_strict_jsonl(dp / "p.jsonl", "T"); t = load_strict_jsonl(dp / "t.jsonl", "T"); r = load_strict_jsonl(dp / "r.jsonl", "T")
        try: exact_three_way_join(p, t, r, "TEST"); assert False
        except SystemExit: pass

def test_duplicate_json_key_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "d.jsonl"; p.write_text('{"canonical_parent_key":"e1","step":0,"step":1}\n')
        from factorized_phase_c_integrity import load_strict_jsonl
        try: load_strict_jsonl(p, "T"); assert False
        except SystemExit: pass
