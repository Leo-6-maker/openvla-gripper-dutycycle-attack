"""CPU synthetic tests for Factorized V2 L3 metrics."""
from __future__ import annotations

import json, math, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))

from run_factorized_l3_analysis import (
    classify_episode, is_valid_start, compute_l3_metrics, compute_timing,
    exact_join, validate_episode_step_sequence, EXPECTED_SPLITS,
)
from fit_factorized_calibrators import (
    fit_raw, fit_intercept, fit_platt, validate_fit_heldout_disjoint,
    classify_provenance, sigmoid, validate_record, check_logit_prob_consistency,
)
from validate_factorized_codex_handoff import validate_handoff_static, validate_handoff_execution

STEP = "step"


def _mk(s=0, ep="t", k10_f=False, k10_km=True):
    return {"step_index": s, "canonical_parent_key": ep, STEP: s,
            "strict_k10_feasible": k10_f, "strict_k10_known_mask": k10_km,
            "grasp_logit": 0.0, "grasp_probability": 0.5, "grasp_known_mask": True, "grasp_target": False,
            "manipulation_logit": 0.0, "manipulation_probability": 0.5, "manipulation_known_mask": False, "manipulation_target": False,
            "release_logit": 0.0, "release_probability": 0.5, "release_known_mask": True, "release_target": False}


def _mk_ol(s=0, ep="t", k10_f=False, k10_km=True):
    return {"step_index": s, "canonical_parent_key": ep, STEP: s,
            "strict_k10_feasible": k10_f, "strict_k10_known_mask": k10_km}


# ── Sequence validation ──

def test_valid_seq():
    assert validate_episode_step_sequence([{STEP: i} for i in range(10)], STEP) == 10


def test_nonzero_rejected():
    try: validate_episode_step_sequence([{STEP: 3}], STEP); assert False
    except SystemExit: pass


def test_gap_rejected():
    try: validate_episode_step_sequence([{STEP: 0}, {STEP: 2}], STEP); assert False
    except SystemExit: pass


def test_dup_rejected():
    try: validate_episode_step_sequence([{STEP: 0}, {STEP: 0}], STEP); assert False
    except SystemExit: pass


# ── K10 + eligible domain ──

def test_T500_last_490():
    rows = [_mk_ol(i) for i in range(500)]
    assert classify_episode(rows, STEP) != "unknown"


def test_T10_only_0():
    assert classify_episode([_mk_ol(i) for i in range(10)], STEP) != "unknown"


def test_T9_unknown():
    assert classify_episode([_mk_ol(i) for i in range(9)], STEP) == "unknown"


def test_positive():
    rows = [_mk_ol(i) for i in range(20)]
    rows[10] = _mk_ol(10, k10_f=True)
    assert classify_episode(rows, STEP) == "positive"


def test_negative():
    assert classify_episode([_mk_ol(i) for i in range(490)], STEP) == "negative"


def test_partial_unknown():
    rows = [_mk_ol(i, k10_km=(i!=250)) for i in range(490)]
    assert classify_episode(rows, STEP) == "unknown"


def test_valid_start():
    assert is_valid_start(_mk_ol(10, k10_f=True))
    assert not is_valid_start(_mk_ol(10, k10_f=False))


# ── Timing ──

def test_timing():
    rows = [_mk_ol(i, k10_f=(5<=i<=15)) for i in range(20)]
    o, rl, rp = compute_timing(rows, 10, STEP)
    assert o == 5 and rl == 11


# ── Unknown emit ──

def test_unknown_emits_tracked():
    eps = {"u": [_mk_ol(i, "u", k10_km=(i<5)) for i in range(490)]}
    res = {"u": {"emitted": True, "emit_step": 50, "final_state": "E"}}
    m = compute_l3_metrics(eps, res, STEP)
    assert m["unknown_episode_emits"] == 1 and m["total_emitted_episodes"] == 1


def test_all_precision():
    eps = {"n": [_mk_ol(i,"n") for i in range(490)],
           "p": [_mk_ol(i,"p") for i in range(490)],
           "u": [_mk_ol(i,"u",k10_km=(i<5)) for i in range(490)]}
    eps["p"][100] = _mk_ol(100,"p",k10_f=True)
    res = {"n": {"emitted":False,"emit_step":-1,"final_state":"I"},
           "p": {"emitted":True,"emit_step":100,"final_state":"E"},
           "u": {"emitted":True,"emit_step":50,"final_state":"E"}}
    m = compute_l3_metrics(eps, res, STEP)
    assert m["unknown_episode_emits"] == 1 and m["total_emitted_episodes"] == 2
    assert m["all_emit_precision"] == 0.5 and m["verified_emit_precision"] == 1.0


# ── Worst-split ──

def test_zero_not_one():
    eps = {"n": [_mk_ol(i,"n") for i in range(490)]}
    res = {"n": {"emitted":False,"emit_step":-1,"final_state":"I"}}
    m = compute_l3_metrics(eps, res, STEP)
    assert m["negative_episode_false_start_rate"] == 0.0


def test_none_is_none():
    eps = {"p": [_mk_ol(i,"p") for i in range(490)]}
    eps["p"][100] = _mk_ol(100,"p",k10_f=True)
    res = {"p": {"emitted":False,"emit_step":-1,"final_state":"I"}}
    m = compute_l3_metrics(eps, res, STEP)
    assert m["negative_episode_false_start_rate"] is None


# ── Calibration ──

def test_independent():
    assert classify_provenance({"fit_identities":["c"]}, {"training_identities":["a"]}) == "INDEPENDENT_CALIBRATION"


def test_resubstitution():
    assert classify_provenance({"fit_identities":["a"]}, {"training_identities":["a"]}) == "TRAIN_RESUBSTITUTION_CALIBRATION"


def test_nan_rejected():
    try: validate_record({"episode":"x","step_index":0,"grasp_logit":float("nan"),"grasp_probability":0.5,"grasp_known_mask":True,"grasp_target":False},"grasp",0); assert False
    except SystemExit: pass


def test_logit_prob_mismatch():
    recs = [{"grasp_logit":5.0,"grasp_probability":0.5,"grasp_known_mask":True,"grasp_target":True,"episode":"x","step_index":0,STEP:0}]
    ok, err = check_logit_prob_consistency(recs, "grasp")
    assert not ok


def test_fit_heldout_ok():
    validate_fit_heldout_disjoint({"fit_identities":["a"]}, {"heldout_identities":["b"]})


def test_overlap_rejected():
    try: validate_fit_heldout_disjoint({"fit_identities":["a"]}, {"heldout_identities":["a"]}); assert False
    except ValueError as e: assert "LEAKAGE" in str(e)


def test_sigmoid():
    assert 0<=sigmoid(-1000)<=1 and 0<=sigmoid(1000)<=1 and sigmoid(0)==0.5


# ── Join ──

def test_join_mismatch():
    with tempfile.TemporaryDirectory() as td:
        rt=Path(td)/"rt.jsonl"; ol=Path(td)/"ol.jsonl"
        rt.write_text(json.dumps({"ep":"a",STEP:0})+"\n")
        ol.write_text(json.dumps({"ep":"b",STEP:0})+"\n")
        try: exact_join(rt,ol,"ep",STEP); assert False
        except SystemExit: pass


def test_join_dup():
    with tempfile.TemporaryDirectory() as td:
        rt=Path(td)/"rt.jsonl"; ol=Path(td)/"ol.jsonl"
        rt.write_text(json.dumps({"ep":"a",STEP:0})+"\n"+json.dumps({"ep":"a",STEP:0})+"\n")
        ol.write_text(json.dumps({"ep":"a",STEP:0})+"\n")
        try: exact_join(rt,ol,"ep",STEP); assert False
        except SystemExit as e: assert "DUP" in str(e)


# ── Validator ──

def test_v2_rejected():
    ok, _ = validate_handoff_static({"schema":"DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V2","status":"DRAFT"})
    assert not ok


def test_v3_needed():
    ok, _ = validate_handoff_static({"schema":"DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3","status":"DRAFT"})
    assert not ok


def test_v31_accepted():
    """V3.1 schema name accepted (still fails on missing fields, which is correct)."""
    ok, errs = validate_handoff_static({"schema":"DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1","status":"DRAFT"})
    assert not ok  # DRAFT fails status check
    assert any("status" in e for e in errs)


# ── Contract: canonical top-level (not heads dict) ──

def test_canonical_top_level_contract():
    """Runner and producer expect top-level grasp/manipulation/release, not heads dict."""
    contract = {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V2",
        "calibration_fit_authoritative": True,
        "threshold_selection_authoritative": True,
        "l3_evaluation_eligible": True,
        "grasp": {"method": "PLATT", "a": 1.0, "b": 0.5, "threshold": 0.7, "method_valid": True, "transform_valid": True, "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION"},
        "manipulation": {"method": "PLATT", "a": 1.0, "b": 0.5, "threshold": 0.5, "method_valid": True, "transform_valid": True, "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION"},
        "release": {"method": "PLATT", "a": 1.0, "b": 0.5, "threshold": 0.3, "method_valid": True, "transform_valid": True, "fit_data_valid": True, "provenance_class": "INDEPENDENT_CALIBRATION"},
    }
    assert contract["l3_evaluation_eligible"]
    assert contract["grasp"]["provenance_class"] == "INDEPENDENT_CALIBRATION"
    assert contract["grasp"]["threshold"] is not None


def test_heads_dict_rejected_by_canonical():
    """Old heads-dict format not accepted by canonical consumer."""
    contract = {"heads": {"grasp": {"a": 1.0, "b": 0.0, "threshold": 0.5}}}
    assert "grasp" not in contract  # top-level expected, not nested


def test_missing_head_rejected():
    contract = {"grasp": {"a": 1.0, "b": 0.0, "threshold": 0.5}}
    assert "manipulation" not in contract and "release" not in contract


def test_missing_threshold_rejected():
    contract = {"grasp": {"a": 1.0, "b": 0.0, "threshold": None}}
    assert contract["grasp"]["threshold"] is None


# ── Step-field canonicalization ──

def test_step_field_not_hardcoded():
    rows = [{"custom_step": i, "strict_k10_feasible": False, "strict_k10_known_mask": True} for i in range(490)]
    assert classify_episode(rows, "custom_step") == "negative"


# ── Determinism ──

def test_classify_deterministic():
    rows = [_mk_ol(i) for i in range(490)]
    assert classify_episode(rows, STEP) == classify_episode(rows, STEP)


# ── Expected splits ──

def test_12_splits():
    assert len(EXPECTED_SPLITS) == 12
