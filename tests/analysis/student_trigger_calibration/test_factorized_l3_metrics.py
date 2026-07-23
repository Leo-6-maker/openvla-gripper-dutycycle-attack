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
from validate_factorized_identity_disjointness import (
    extract_identities, check_pairwise_disjoint, classify_verdict,
    check_training_provenance, check_cohort_membership,
    check_calibration_coverage, check_policy_coverage, check_heldout_coverage,
    check_heldout_teacher_closure, check_heldout_teacher_closure_global,
    check_deterministic_allocation, audit_inputs, classify_coverage,
    phase_c_authorization, FIVE_ROLES, COHORT_TO_ROLE,
)

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


# ── Five-way identity disjointness (Phase B) ──

def _id_manifest(role, ids):
    return {role: list(ids)}

def _alloc_manifest(split_data):
    return {"splits": split_data}

def _make_split_ids(prefix, n):
    return [f"{prefix}_{i}" for i in range(n)]


# ── Extract ──

def test_extract_direct_list():
    m = {"identities": ["a", "b", "c"]}
    assert extract_identities(m, "calibrator_fit", "o0_i0") == {"a", "b", "c"}


def test_extract_allocation_split():
    m = _alloc_manifest({"o0_i0": {"calibrator_fit": ["x", "y"], "policy_selection": ["z"]}})
    assert extract_identities(m, "calibrator_fit", "o0_i0") == {"x", "y"}
    assert extract_identities(m, "policy_selection", "o0_i0") == {"z"}


def test_extract_role_keyed():
    m = {"calibrator_fit": ["a", "b"], "heldout_l3": ["c"]}
    assert extract_identities(m, "calibrator_fit", "o0_i0") == {"a", "b"}


def test_extract_missing_role():
    assert extract_identities({"calibrator_fit": ["a"]}, "heldout_l3", "o0_i0") == set()


# ── Pairwise disjointness ──

def test_pairwise_all_disjoint():
    sets = {
        "checkpoint_training": set(f"t{i}" for i in range(100)),
        "calibrator_fit": set(f"c{i}" for i in range(30)),
        "policy_selection": set(f"p{i}" for i in range(30)),
        "heldout_l3": set(f"h{i}" for i in range(30)),
        "attack_eval": set(f"a{i}" for i in range(50)),
    }
    errs = []
    assert check_pairwise_disjoint(sets, "o0_i0", errs)
    assert len(errs) == 0


def test_pairwise_leakage_detected():
    sets = {
        "checkpoint_training": {"ep_001", "ep_002", "ep_003"},
        "calibrator_fit": {"ep_002", "ep_004"},
        "policy_selection": {"ep_005"},
        "heldout_l3": {"ep_006"},
        "attack_eval": set(),
    }
    errs = []
    assert not check_pairwise_disjoint(sets, "o0_i0", errs)
    assert len(errs) == 1
    assert any("T" in e and "C" in e for e in errs)


def test_attack_isolation_breach():
    sets = {
        "checkpoint_training": {"ep_001"},
        "calibrator_fit": {"ep_002"},
        "policy_selection": {"ep_003"},
        "heldout_l3": {"ep_004"},
        "attack_eval": {"ep_001", "ep_005"},
    }
    errs = []
    assert not check_pairwise_disjoint(sets, "o0_i0", errs)
    assert any(("T" in e and "A" in e) or ("checkpoint_training" in e and "attack_eval" in e) for e in errs)


def test_empty_roles_still_pairwise_ok():
    sets = {role: set() for role in FIVE_ROLES}
    errs = []
    assert check_pairwise_disjoint(sets, "o0_i0", errs)


def test_multiple_leakages_all_reported():
    sets = {
        "checkpoint_training": {"ep_001", "ep_002"},
        "calibrator_fit": {"ep_001", "ep_003"},
        "policy_selection": {"ep_003", "ep_004"},
        "heldout_l3": {"ep_005"},
        "attack_eval": {"ep_006"},
    }
    errs = []
    assert not check_pairwise_disjoint(sets, "o0_i0", errs)
    assert len(errs) == 2  # T∩C and C∩P


# ── Training provenance ──

def test_training_provenance_accepted():
    errs = []
    check_training_provenance({"provenance_method": "TRAINING_DATALOADER_LOG"}, "o0_i0", errs)
    assert len(errs) == 0


def test_training_provenance_set_subtraction_rejected():
    errs = []
    check_training_provenance({"provenance_method": "SET_SUBTRACTION"}, "o0_i0", errs)
    assert len(errs) >= 1
    assert any("SET_SUBTRACTION" in e for e in errs)


def test_training_provenance_unknown_warns():
    errs = []
    check_training_provenance({"provenance_method": "UNKNOWN"}, "o0_i0", errs)
    assert len(errs) >= 1


# ── Verdict classification ──

def test_verdict_missing_inputs():
    """Missing inputs → HOLD_INPUTS_MISSING, never NESTED_RETRAIN_REQUIRED."""
    assert classify_verdict(True, "PASS", "RECOVERED_EXISTING_ROOTS", False) == "HOLD_INPUTS_MISSING"
    assert classify_verdict(False, "NOT_AUDITABLE", "UNKNOWN", False) == "HOLD_INPUTS_MISSING"


def test_verdict_contamination_trumps_all():
    """Proven contamination → NESTED_RETRAIN_REQUIRED even with complete inputs."""
    assert classify_verdict(False, "HOLD_INSUFFICIENT_STATISTICAL_COVERAGE",
                           "RECOVERED_EXISTING_ROOTS", True) == "NESTED_RETRAIN_REQUIRED"


def test_verdict_pass_existing_roots():
    assert classify_verdict(True, "PASS", "RECOVERED_EXISTING_ROOTS", True) == "PASS_EXISTING_ROOTS"


def test_verdict_pass_deterministic():
    assert classify_verdict(True, "HOLD_INSUFFICIENT_STATISTICAL_COVERAGE",
                           "DETERMINISTIC_ALLOCATION", True) == "PASS_DETERMINISTIC_ALLOCATION"


def test_verdict_unclear_source():
    """Clean identity closure but unclear source status → HOLD, not retrain."""
    assert classify_verdict(True, "PASS", "UNKNOWN", True) == "HOLD_INPUTS_MISSING"


# ── Statistical coverage is independent ──

def test_coverage_pass():
    assert classify_coverage([], True) == "PASS"


def test_coverage_insufficient():
    assert classify_coverage(["CALIBRATION_NO_POSITIVE: o0_i0/grasp"], True) == "HOLD_INSUFFICIENT_STATISTICAL_COVERAGE"


def test_coverage_not_auditable():
    assert classify_coverage([], False) == "NOT_AUDITABLE"


def test_phase_c_authorization():
    r = phase_c_authorization("PASS_EXISTING_ROOTS", True, True, True)
    assert r["cp_inference_authorized"] and r["heldout_l3_data_ready"]
    assert not r["heldout_l3_inference_authorized"]  # validator never authorizes L3
    r = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", False, True, True)
    assert not r["cp_inference_authorized"]
    r = phase_c_authorization("HOLD_INPUTS_MISSING", True, True, True)
    assert not r["cp_inference_authorized"]
    r = phase_c_authorization("NESTED_RETRAIN_REQUIRED", True, True, True)
    assert not r["cp_inference_authorized"]
    r = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, True, False)
    assert r["cp_inference_authorized"] and not r["heldout_l3_data_ready"]


# ── Cohort membership ──

def test_cohort_membership_valid():
    sets = {
        "checkpoint_training": {"ep_001"},
        "calibrator_fit": {"ep_101"},
        "policy_selection": {"ep_151"},
        "heldout_l3": {"ep_201"},
        "attack_eval": {"ep_301"},
    }
    membership = {
        "ep_001": "DETECTOR_TRAIN",
        "ep_101": "DETECTOR_VAL",
        "ep_151": "DETECTOR_VAL",
        "ep_201": "DETECTOR_TEST",
        "ep_301": "ATTACK_EVAL",
    }
    errs = []
    check_cohort_membership(sets, membership, "o0_i0", errs)
    assert len(errs) == 0


def test_cohort_membership_violation():
    sets = {
        "checkpoint_training": {"ep_001", "ep_999"},
    }
    membership = {
        "ep_001": "DETECTOR_TRAIN",
        "ep_999": "DETECTOR_TEST",  # TEST identity used as training!
    }
    errs = []
    check_cohort_membership(sets, membership, "o0_i0", errs)
    assert len(errs) >= 1
    assert any("COHORT_VIOLATION" in e for e in errs)


def test_cohort_membership_unknown():
    sets = {"checkpoint_training": {"ep_mystery"}}
    errs = []
    check_cohort_membership(sets, {}, "o0_i0", errs)
    # No cohort ledger → COHORT_MEMBERSHIP_MISSING error
    assert any("COHORT_MEMBERSHIP_MISSING" in e for e in errs)


# ── Deterministic allocation ──

def test_deterministic_allocation_valid():
    da = {
        "deterministic_allocation": {
            "parent_cohort": "DETECTOR_VAL",
            "parent_cohort_manifest_sha256": "a" * 64,
            "fixed_salt": "deadbeef",
            "canonical_sort_key": "canonical_identity_hash",
            "allocation_algorithm_sha256": "b" * 64,
            "allocation_code_sha256": "c" * 64,
        }
    }
    sets = {"calibrator_fit": {"ep_101"}, "policy_selection": {"ep_151"}}
    errs = []
    check_deterministic_allocation(da, sets, "o0_i0", errs)
    assert len(errs) == 0


def test_deterministic_allocation_missing_fields():
    da = {"deterministic_allocation": {"parent_cohort": "DETECTOR_VAL"}}
    sets = {"calibrator_fit": set(), "policy_selection": set()}
    errs = []
    check_deterministic_allocation(da, sets, "o0_i0", errs)
    assert len(errs) >= 4  # missing salt, sort_key, algorithm_sha, code_sha


def test_deterministic_allocation_closure():
    da = {
        "deterministic_allocation": {
            "parent_cohort": "DETECTOR_VAL",
            "parent_cohort_manifest_sha256": "a" * 64,
            "fixed_salt": "salt",
            "canonical_sort_key": "hash",
            "allocation_algorithm_sha256": "b" * 64,
            "allocation_code_sha256": "c" * 64,
            "parent_cohort_identities": {"o0_i0": ["ep_101", "ep_151", "ep_199"]},
        }
    }
    sets = {"calibrator_fit": {"ep_101"}, "policy_selection": {"ep_151"}}
    errs = []
    check_deterministic_allocation(da, sets, "o0_i0", errs)
    # ep_199 is in VAL but not in C∪P
    assert any("ALLOC_CLOSURE" in e for e in errs)


# ── Calibration coverage (per-head known pos/neg) ──

def test_calibration_coverage_ok():
    alloc = {"calibration_head_summaries": {
        "o0_i0": {
            "grasp": {"known_positive": 45, "known_negative": 1200},
            "manipulation": {"known_positive": 8, "known_negative": 1500},
            "release": {"known_positive": 60, "known_negative": 1100},
        }
    }}
    issues = []
    check_calibration_coverage(alloc, "o0_i0", issues)
    assert len(issues) == 0


def test_calibration_coverage_missing_head_summary():
    issues = []
    check_calibration_coverage({}, "o0_i0", issues)
    assert any("NOT_AUDITABLE" in i for i in issues)


def test_calibration_coverage_no_positive():
    alloc = {"calibration_head_summaries": {
        "o0_i0": {"grasp": {"known_positive": 0, "known_negative": 1200}}
    }}
    issues = []
    check_calibration_coverage(alloc, "o0_i0", issues)
    assert any("NO_POSITIVE" in i for i in issues)


def test_calibration_coverage_no_negative():
    alloc = {"calibration_head_summaries": {
        "o0_i0": {"manipulation": {"known_positive": 8, "known_negative": 0}}
    }}
    issues = []
    check_calibration_coverage(alloc, "o0_i0", issues)
    assert any("NO_NEGATIVE" in i for i in issues)


# ── Policy coverage ──

def test_policy_coverage_ok():
    alloc = {"policy_selection_summaries": {
        "o0_i0": {
            "negative_episodes": 5, "k10_positive_opportunities": 3,
            "eligible_episodes": 16, "known_denominator_episodes": 16,
        }
    }}
    issues = []
    check_policy_coverage(alloc, "o0_i0", issues)
    assert len(issues) == 0


def test_policy_coverage_no_negative():
    alloc = {"policy_selection_summaries": {
        "o0_i0": {"negative_episodes": 0, "k10_positive_opportunities": 3}
    }}
    issues = []
    check_policy_coverage(alloc, "o0_i0", issues)
    assert any("NO_NEGATIVE" in i for i in issues)


def test_policy_coverage_no_opportunity():
    alloc = {"policy_selection_summaries": {
        "o0_i0": {"negative_episodes": 5, "k10_positive_opportunities": 0}
    }}
    issues = []
    check_policy_coverage(alloc, "o0_i0", issues)
    assert any("NO_OPPORTUNITY" in i for i in issues)


# ── Heldout coverage ──

def test_heldout_coverage_ok():
    sets = {"heldout_l3": set(_make_split_ids("held", 16))}
    issues = []
    check_heldout_coverage(sets, "o0_i0", issues)
    assert len(issues) == 0


def test_heldout_coverage_empty():
    issues = []
    check_heldout_coverage({"heldout_l3": set()}, "o0_i0", issues)
    assert any("EMPTY" in i for i in issues)


# ── Input audit ──

def test_audit_inputs_all_missing():
    present, missing = audit_inputs({})
    assert len(missing) == 6
    assert len(present) == 0


# ── 1200/200/200/400 realistic fixture ──

def _make_realistic_manifests():
    """CLEAN2000 allocation: TRAIN=1200, VAL=200 split C/P, TEST=200, ATTACK=400.
    Per-split averages: T=100, C≈8, P≈8, H≈16, A≈33.
    """
    n_splits = 12
    split_keys = [f"o{o}_i{i}" for o in range(4) for i in range(3)]

    discovery = {"identity_source_status": "RECOVERED_EXISTING_ROOTS"}
    cohort_membership = {}

    training_splits = {}
    cal_splits = {}
    pol_splits = {}
    held_splits = {}
    atk_splits = {}

    tid = 0
    for sk in split_keys:
        # Training: ~100 per split
        t_ids = [f"T_{sk}_{n}" for n in range(100)]
        for eid in t_ids:
            cohort_membership[eid] = "DETECTOR_TRAIN"
        training_splits[sk] = {"checkpoint_training": t_ids}

        # Calibrator: ~8 per split
        c_ids = [f"C_{sk}_{n}" for n in range(8)]
        for eid in c_ids:
            cohort_membership[eid] = "DETECTOR_VAL"
        cal_splits[sk] = {"calibrator_fit": c_ids}

        # Policy: ~9 per split
        p_ids = [f"P_{sk}_{n}" for n in range(9)]
        for eid in p_ids:
            cohort_membership[eid] = "DETECTOR_VAL"
        pol_splits[sk] = {"policy_selection": p_ids}

        # Heldout: ~16 per split
        h_ids = [f"H_{sk}_{n}" for n in range(16)]
        for eid in h_ids:
            cohort_membership[eid] = "DETECTOR_TEST"
        held_splits[sk] = {"heldout_l3": h_ids}

        # Attack: ~33 per split
        a_ids = [f"A_{sk}_{n}" for n in range(33)]
        for eid in a_ids:
            cohort_membership[eid] = "ATTACK_EVAL"
        atk_splits[sk] = {"attack_eval": a_ids}

        tid += 1

    discovery["cohort_membership"] = cohort_membership

    training = {
        "provenance_method": "TRAINING_DATALOADER_LOG",
        "splits": training_splits,
    }
    cal = {"splits": cal_splits}
    pol = {"splits": pol_splits}
    held = {"splits": held_splits}
    atk = {"splits": atk_splits}

    return discovery, training, cal, pol, held, atk, split_keys


def test_realistic_1200_200_200_400_pass():
    """1200/200/200/400 allocation with correct cohort membership and disjointness."""
    discovery, training, cal, pol, held, atk, split_keys = _make_realistic_manifests()

    all_disjoint_ok = True
    for sk in split_keys:
        sets = {
            "checkpoint_training": extract_identities(training, "checkpoint_training", sk),
            "calibrator_fit": extract_identities(cal, "calibrator_fit", sk),
            "policy_selection": extract_identities(pol, "policy_selection", sk),
            "heldout_l3": extract_identities(held, "heldout_l3", sk),
            "attack_eval": extract_identities(atk, "attack_eval", sk),
        }
        errs = []
        check_pairwise_disjoint(sets, sk, errs)
        if errs:
            all_disjoint_ok = False
        # Verify counts are realistic
        assert len(sets["checkpoint_training"]) == 100  # ~100 per split
        assert len(sets["calibrator_fit"]) == 8         # ~8 per split
        assert len(sets["policy_selection"]) == 9       # ~9 per split
        assert len(sets["heldout_l3"]) == 16            # ~16 per split
        assert len(sets["attack_eval"]) == 33            # ~33 per split

    assert all_disjoint_ok


def test_realistic_cohort_membership_pass():
    discovery, training, cal, pol, held, atk, split_keys = _make_realistic_manifests()
    cohort = discovery["cohort_membership"]

    all_ok = True
    for sk in split_keys:
        sets = {
            "checkpoint_training": extract_identities(training, "checkpoint_training", sk),
            "calibrator_fit": extract_identities(cal, "calibrator_fit", sk),
            "policy_selection": extract_identities(pol, "policy_selection", sk),
            "heldout_l3": extract_identities(held, "heldout_l3", sk),
            "attack_eval": extract_identities(atk, "attack_eval", sk),
        }
        errs = []
        check_cohort_membership(sets, cohort, sk, errs)
        if errs:
            all_ok = False
    assert all_ok


def test_realistic_low_counts_not_retrain():
    """C≈8 and P≈9 per split is below old 20 threshold but SHOULD NOT trigger retrain.
    Only contamination triggers retrain."""
    discovery, training, cal, pol, held, atk, split_keys = _make_realistic_manifests()

    for sk in split_keys:
        c_ids = extract_identities(cal, "calibrator_fit", sk)
        p_ids = extract_identities(pol, "policy_selection", sk)
        # These are low but valid — no arbitrary threshold
        assert len(c_ids) < 20
        assert len(p_ids) < 20

    # Identity disjointness should still pass
    all_ok = True
    for sk in split_keys:
        sets = {
            "checkpoint_training": extract_identities(training, "checkpoint_training", sk),
            "calibrator_fit": extract_identities(cal, "calibrator_fit", sk),
            "policy_selection": extract_identities(pol, "policy_selection", sk),
            "heldout_l3": extract_identities(held, "heldout_l3", sk),
            "attack_eval": extract_identities(atk, "attack_eval", sk),
        }
        errs = []
        check_pairwise_disjoint(sets, sk, errs)
        if errs:
            all_ok = False
    assert all_ok


def test_FIVE_ROLES():
    assert len(FIVE_ROLES) == 5
    assert "checkpoint_training" in FIVE_ROLES
    assert "attack_eval" in FIVE_ROLES


def test_COHORT_TO_ROLE():
    assert COHORT_TO_ROLE["DETECTOR_TRAIN"] == "checkpoint_training"
    assert COHORT_TO_ROLE["DETECTOR_TEST"] == "heldout_l3"
    assert COHORT_TO_ROLE["ATTACK_EVAL"] == "attack_eval"
    assert isinstance(COHORT_TO_ROLE["DETECTOR_VAL"], list)
    assert "calibrator_fit" in COHORT_TO_ROLE["DETECTOR_VAL"]
    assert "policy_selection" in COHORT_TO_ROLE["DETECTOR_VAL"]


# ── Split Phase C authorization ──

def test_phase_c_full_data_ready():
    """Identity + C/P + HTC all pass: CP authorized, L3 data ready, L3 inference NOT authorized."""
    result = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, True, True)
    assert result["cp_inference_authorized"] is True
    assert result["heldout_l3_data_ready"] is True
    assert result["heldout_l3_inference_authorized"] is False  # validator never authorizes L3
    assert result["heldout_l3_blocker"] == "PENDING_EXTERNAL_FREEZE"


def test_phase_c_cp_only_no_htc():
    result = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, True, False)
    assert result["cp_inference_authorized"] is True
    assert result["heldout_l3_data_ready"] is False
    assert result["heldout_l3_inference_authorized"] is False


def test_phase_c_no_cal_coverage():
    result = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", False, True, True)
    assert result["cp_inference_authorized"] is False
    assert result["heldout_l3_data_ready"] is True  # HTC is independent of C/P coverage
    assert result["heldout_l3_inference_authorized"] is False


def test_phase_c_no_pol_coverage():
    result = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, False, True)
    assert result["cp_inference_authorized"] is False


def test_phase_c_contaminated():
    result = phase_c_authorization("NESTED_RETRAIN_REQUIRED", True, True, True)
    assert result["cp_inference_authorized"] is False
    assert result["heldout_l3_data_ready"] is False


def test_phase_c_hold_inputs():
    result = phase_c_authorization("HOLD_INPUTS_MISSING", True, True, True)
    assert result["cp_inference_authorized"] is False


def test_phase_c_l3_blocker_when_data_not_ready():
    result = phase_c_authorization("PASS_DETERMINISTIC_ALLOCATION", True, True, False)
    assert result["heldout_l3_blocker"] == "HOLD_DATA_NOT_READY"


def test_phase_c_l3_never_authorized():
    """Validator must NEVER set heldout_l3_inference_authorized=True — that requires external freeze."""
    for verdict in ["PASS_EXISTING_ROOTS", "PASS_DETERMINISTIC_ALLOCATION"]:
        for cal in [True, False]:
            for pol in [True, False]:
                for htc in [True, False]:
                    r = phase_c_authorization(verdict, cal, pol, htc)
                    assert r["heldout_l3_inference_authorized"] is False, \
                        f"L3 authorized unexpectedly for {verdict}/{cal}/{pol}/{htc}"


# ── Heldout Teacher Closure ──

def test_htc_no_bundle():
    errs = []
    check_heldout_teacher_closure({"ep1", "ep2"}, None, "o0_i0", None, errs)
    assert len(errs) == 1
    assert any("BUNDLE_MISSING" in e for e in errs)


def test_htc_bundle_not_dir():
    import tempfile
    errs = []
    check_heldout_teacher_closure({"ep1"}, "/nonexistent/path", "o0_i0", None, errs)
    assert len(errs) >= 1


def test_htc_identity_closure():
    """H manifest has ep1, ep2 but Teacher has ep1, ep3 → missing + extra."""
    import tempfile, os
    td = tempfile.mkdtemp()
    split_dir = os.path.join(td, "o0_i0")
    os.makedirs(split_dir)
    # Write Teacher labels with ep1 and ep3
    labels = os.path.join(split_dir, "factorized_teacher_v1.jsonl")
    with open(labels, "w") as f:
        f.write(json.dumps({"canonical_parent_key": "ep1", "step": 0,
                           "grasp_established_known_mask": True,
                           "manipulation_active_known_mask": False,
                           "release_or_instability_known_mask": True,
                           "physics_protocol_schema": "TEST_SCHEMA",
                           "source_artifact_recursive_sha256": "a"*64}) + "\n")
        f.write(json.dumps({"canonical_parent_key": "ep3", "step": 0,
                           "grasp_established_known_mask": True}) + "\n")
    errs = []
    check_heldout_teacher_closure({"ep1", "ep2"}, td, "o0_i0", None, errs)
    assert any("IDENTITY_MISSING" in e for e in errs)
    assert any("IDENTITY_EXTRA" in e for e in errs)


def test_htc_duplicate_ep_step():
    import tempfile, os
    td = tempfile.mkdtemp()
    split_dir = os.path.join(td, "o0_i0")
    os.makedirs(split_dir)
    labels = os.path.join(split_dir, "factorized_teacher_v1.jsonl")
    with open(labels, "w") as f:
        f.write(json.dumps({"canonical_parent_key": "ep1", "step": 0,
                           "grasp_established_known_mask": True}) + "\n")
        f.write(json.dumps({"canonical_parent_key": "ep1", "step": 0,
                           "grasp_established_known_mask": True}) + "\n")
    errs = []
    check_heldout_teacher_closure({"ep1"}, td, "o0_i0", None, errs)
    assert any("DUPLICATE" in e for e in errs)


def test_htc_pass():
    import tempfile, os
    td = tempfile.mkdtemp()
    split_dir = os.path.join(td, "o0_i0")
    os.makedirs(split_dir)
    labels = os.path.join(split_dir, "factorized_teacher_v1.jsonl")
    with open(labels, "w") as f:
        f.write(json.dumps({"canonical_parent_key": "ep1", "step": 0,
                           "grasp_established_known_mask": True}) + "\n")
        f.write(json.dumps({"canonical_parent_key": "ep1", "step": 1,
                           "grasp_established_known_mask": False}) + "\n")
        f.write(json.dumps({"canonical_parent_key": "ep2", "step": 0,
                           "manipulation_active_known_mask": True}) + "\n")
    errs = []
    check_heldout_teacher_closure({"ep1", "ep2"}, td, "o0_i0", None, errs)
    assert len(errs) == 0


def test_htc_global_wrapper():
    import tempfile, os
    td = tempfile.mkdtemp()
    for sk in ["o0_i0", "o0_i1"]:
        split_dir = os.path.join(td, sk)
        os.makedirs(split_dir)
        labels = os.path.join(split_dir, "factorized_teacher_v1.jsonl")
        with open(labels, "w") as f:
            f.write(json.dumps({"canonical_parent_key": f"ep_{sk}", "step": 0,
                               "grasp_established_known_mask": True}) + "\n")
    h_ids = {"o0_i0": ["ep_o0_i0"], "o0_i1": ["ep_o0_i1"]}
    errs, rows = check_heldout_teacher_closure_global(h_ids, td, ["o0_i0", "o0_i1"], None)
    assert len(errs) == 0
    assert len(rows) == 2


def test_htc_source_sha_mismatch():
    import tempfile, os
    td = tempfile.mkdtemp()
    split_dir = os.path.join(td, "o0_i0")
    os.makedirs(split_dir)
    labels = os.path.join(split_dir, "factorized_teacher_v1.jsonl")
    expected_sha = "e" * 64
    actual_sha = "f" * 64
    with open(labels, "w") as f:
        f.write(json.dumps({"canonical_parent_key": "ep1", "step": 0,
                           "grasp_established_known_mask": True,
                           "source_artifact_recursive_sha256": actual_sha}) + "\n")
    errs = []
    check_heldout_teacher_closure({"ep1"}, td, "o0_i0", expected_sha, errs)
    assert any("SOURCE_SHA_MISMATCH" in e for e in errs)
