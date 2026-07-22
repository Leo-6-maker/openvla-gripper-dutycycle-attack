"""CPU tests for Factorized V2 calibration, selection and L3 replay."""
from __future__ import annotations

import json
import copy
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from fit_factorized_calibrators import (  # noqa: E402
    check_logit_prob_consistency,
    classify_provenance,
    fit_intercept,
    fit_platt,
    fit_raw,
    sigmoid,
    validate_fit_heldout_disjoint,
    validate_record,
)
from gripper_attack.factorized_scheduler_adapter import (  # noqa: E402
    FactorizedV2SchedulerAdapter,
)
from produce_factorized_calibration_threshold_contract import (  # noqa: E402
    validate_against_schema,
)
from run_factorized_l3_analysis import (  # noqa: E402
    EXPECTED_SPLITS,
    classify_episode,
    compute_l3_metrics,
    compute_timing,
    exact_join,
    is_valid_start,
    validate_episode_step_sequence,
)
from select_factorized_scheduler_thresholds import (  # noqa: E402
    evaluate_candidate,
    select_thresholds,
)
from validate_factorized_codex_handoff import (  # noqa: E402
    canonical_handoff_sha,
    canonical_reference_sha,
    validate_handoff_execution,
    validate_handoff_static,
)
from load_factorized_handoff import load_handoff_file  # noqa: E402

STEP = "step"
SHA = "a" * 64
COMMIT = "b" * 40


def _calibration_row(
    step: int = 0,
    episode: str = "t",
    *,
    grasp_logit: float = 0.0,
    manipulation_logit: float = 0.0,
    release_logit: float = 0.0,
):
    def probability(value):
        return sigmoid(value)

    return {
        "episode": episode,
        "step": step,
        "grasp_logit": grasp_logit,
        "grasp_probability": probability(grasp_logit),
        "grasp_known_mask": True,
        "grasp_target": False,
        "manipulation_logit": manipulation_logit,
        "manipulation_probability": probability(manipulation_logit),
        "manipulation_known_mask": True,
        "manipulation_target": False,
        "release_logit": release_logit,
        "release_probability": probability(release_logit),
        "release_known_mask": True,
        "release_target": False,
    }


def _offline_row(
    step: int = 0,
    episode: str = "t",
    *,
    feasible: bool = False,
    known: bool = True,
):
    return {
        "episode": episode,
        "step": step,
        "strict_k10_feasible": feasible,
        "strict_k10_known_mask": known,
    }


def _runtime_row(
    step: int,
    episode: str,
    *,
    grasp_logit: float,
    manipulation_logit: float,
    release_logit: float = -4.0,
):
    return {
        "episode": episode,
        "step": step,
        "candidate_close": True,
        "action_known": True,
        "student_valid": True,
        "route_supported": True,
        "checkpoint_sha256": SHA,
        "source_commit": COMMIT,
        "feature_order_sha256": SHA,
        "split": "o0_i0",
        "scheduler_source_sha256": SHA,
        "structural_config_sha256": SHA,
        "grasp_logit": grasp_logit,
        "manipulation_logit": manipulation_logit,
        "release_logit": release_logit,
    }


def _fit_contract():
    calibrators = []
    for head in ("grasp", "manipulation", "release"):
        calibrators.append(
            {
                "head": head,
                "method": "RAW",
                "a": 1.0,
                "b": 0.0,
                "n_fit_pos": 10,
                "n_fit_neg": 10,
                "method_valid": True,
            }
        )
    return {
        "schema": "FACTORIZED_V2_CALIBRATION_CONTRACT_V2",
        "split": "o0_i0",
        "checkpoint_sha256": SHA,
        "student_source_commit": COMMIT,
        "provenance": "INDEPENDENT_CALIBRATION",
        "authoritative": True,
        "all_heads_valid": True,
        "calibrators": calibrators,
        "fit_manifest_sha256": SHA,
    }


def _structure():
    return {
        "schema": "FACTORIZED_V2_SCHEDULER_STRUCTURE_V1",
        "candidate_dwell": 1,
        "candidate_dwell_counts_before_grasp": False,
        "persistence_window": 1,
        "persistence_required": 1,
        "warmup_steps": 0,
        "invalid_step_policy": "reset",
        "attack_enabled": False,
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    }


def _valid_v3_contract():
    head = {
        "method": "RAW",
        "a": 1.0,
        "b": 0.0,
        "threshold": 0.5,
        "transform": "probability=sigmoid(a*raw_logit+b)",
        "method_valid": True,
        "transform_valid": True,
        "fit_data_valid": True,
        "provenance_class": "INDEPENDENT_CALIBRATION",
        "fit_manifest_sha256": SHA,
        "policy_selection_manifest_sha256": SHA,
    }
    return {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3",
        "status": "AUTHORITATIVE",
        "split": "o0_i0",
        "checkpoint_sha256": SHA,
        "scheduler_source_sha256": SHA,
        "structural_config_sha256": SHA,
        "student_source_commit": COMMIT,
        "feature_order_sha256": SHA,
        "calibration_fit_authoritative": True,
        "threshold_selection_authoritative": True,
        "l3_evaluation_eligible": True,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "grasp": dict(head),
        "manipulation": dict(head),
        "release": dict(head),
    }


# Sequence and K10 classification -------------------------------------------------

def test_valid_sequence():
    assert validate_episode_step_sequence([{STEP: index} for index in range(10)], STEP) == 10


@pytest.mark.parametrize(
    "rows",
    (
        [{STEP: 3}],
        [{STEP: 0}, {STEP: 2}],
        [{STEP: 0}, {STEP: 0}],
        [{STEP: True}],
    ),
)
def test_invalid_sequence_rejected(rows):
    with pytest.raises(SystemExit):
        validate_episode_step_sequence(rows, STEP)


def test_t9_unknown_and_t10_known():
    assert classify_episode([_offline_row(index) for index in range(9)], STEP) == "unknown"
    assert classify_episode([_offline_row(index) for index in range(10)], STEP) == "negative"


def test_positive_negative_unknown_classification():
    positive = [_offline_row(index) for index in range(20)]
    positive[5] = _offline_row(5, feasible=True)
    assert classify_episode(positive, STEP) == "positive"
    assert classify_episode([_offline_row(index) for index in range(20)], STEP) == "negative"
    partial = [_offline_row(index, known=index != 5) for index in range(20)]
    assert classify_episode(partial, STEP) == "unknown"


def test_valid_start_and_timing():
    rows = [_offline_row(index, feasible=5 <= index <= 15) for index in range(20)]
    assert is_valid_start(rows[10])
    assert not is_valid_start(rows[0])
    offset, length, relative = compute_timing(rows, 10, STEP)
    assert offset == 5
    assert length == 11
    assert relative == 0.5


# Metrics -----------------------------------------------------------------------

def test_metric_denominators_are_unambiguous():
    episodes = {
        "negative": [_offline_row(index, "negative") for index in range(20)],
        "positive": [_offline_row(index, "positive") for index in range(20)],
        "unknown": [
            _offline_row(index, "unknown", known=index < 5)
            for index in range(20)
        ],
    }
    episodes["positive"][5] = _offline_row(5, "positive", feasible=True)
    results = {
        "negative": {"emitted": False, "emit_step": -1, "final_state": "IDLE"},
        "positive": {"emitted": True, "emit_step": 5, "final_state": "DONE"},
        "unknown": {"emitted": True, "emit_step": 5, "final_state": "DONE"},
    }
    metrics = compute_l3_metrics(episodes, results, STEP)
    assert metrics["total_emitted_all"] == 2
    assert metrics["total_emitted_verified"] == 1
    assert metrics["all_emit_precision"] == 0.5
    assert metrics["verified_emit_precision"] == 1.0
    assert "total_emitted_episodes" not in metrics
    positive_row = next(
        row for row in metrics["per_episode"] if row["episode_key"] == "positive"
    )
    assert positive_row["timing_offset"] == 0


def test_false_start_zero_and_undefined():
    negative = {"n": [_offline_row(index, "n") for index in range(20)]}
    metrics = compute_l3_metrics(
        negative,
        {"n": {"emitted": False, "emit_step": -1, "final_state": "IDLE"}},
        STEP,
    )
    assert metrics["negative_episode_false_start_rate"] == 0.0

    positive = {"p": [_offline_row(index, "p") for index in range(20)]}
    positive["p"][5] = _offline_row(5, "p", feasible=True)
    metrics = compute_l3_metrics(
        positive,
        {"p": {"emitted": False, "emit_step": -1, "final_state": "IDLE"}},
        STEP,
    )
    assert metrics["negative_episode_false_start_rate"] is None


# Calibration -------------------------------------------------------------------

def test_calibration_record_strict_types():
    valid = _calibration_row()
    assert validate_record(valid, "grasp", 0)[0] == 0.0

    for field, value in (
        ("step", True),
        ("grasp_logit", True),
        ("grasp_probability", True),
        ("episode", ""),
    ):
        invalid = dict(valid)
        invalid[field] = value
        with pytest.raises(SystemExit):
            validate_record(invalid, "grasp", 0)


def test_logit_probability_binding():
    records = [_calibration_row(grasp_logit=5.0)]
    assert check_logit_prob_consistency(records, "grasp")[0]
    records[0]["grasp_probability"] = 0.5
    assert not check_logit_prob_consistency(records, "grasp")[0]


def test_calibrator_methods():
    records = []
    for index in range(12):
        row = _calibration_row(
            step=index,
            grasp_logit=2.0 if index < 6 else -2.0,
        )
        row["grasp_target"] = index < 6
        records.append(row)
    assert fit_raw(records, "grasp")["method_valid"]
    assert fit_intercept(records, "grasp")["method_valid"]
    assert fit_platt(records, "grasp")["method_valid"]


def test_identity_provenance_and_disjointness():
    assert (
        classify_provenance(
            {"fit_identities": ["fit"]},
            {"training_identities": ["train"], "checkpoint_sha256": SHA},
        )
        == "INDEPENDENT_CALIBRATION"
    )
    assert (
        classify_provenance(
            {"fit_identities": ["same"]},
            {"training_identities": ["same"], "checkpoint_sha256": SHA},
        )
        == "TRAIN_RESUBSTITUTION_CALIBRATION"
    )
    validate_fit_heldout_disjoint(
        {"fit_identities": ["fit"]},
        {"heldout_identities": ["heldout"]},
    )
    with pytest.raises(ValueError):
        validate_fit_heldout_disjoint(
            {"fit_identities": ["same"]},
            {"heldout_identities": ["same"]},
        )


def test_sigmoid_bounds():
    assert 0.0 <= sigmoid(-1000.0) <= 1.0
    assert 0.0 <= sigmoid(1000.0) <= 1.0
    assert sigmoid(0.0) == 0.5


# Join --------------------------------------------------------------------------

def test_join_mismatch_and_duplicate_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        runtime = root / "runtime.jsonl"
        offline = root / "offline.jsonl"
        runtime.write_text(json.dumps({"episode": "a", STEP: 0}) + "\n")
        offline.write_text(json.dumps({"episode": "b", STEP: 0}) + "\n")
        with pytest.raises(SystemExit):
            exact_join(runtime, offline, "episode", STEP)

        runtime.write_text(
            json.dumps({"episode": "a", STEP: 0})
            + "\n"
            + json.dumps({"episode": "a", STEP: 0})
            + "\n"
        )
        offline.write_text(json.dumps({"episode": "a", STEP: 0}) + "\n")
        with pytest.raises(SystemExit):
            exact_join(runtime, offline, "episode", STEP)


# Contract and adapter ----------------------------------------------------------

def test_v3_schema_exact_validation():
    contract = _valid_v3_contract()
    validate_against_schema(contract)
    invalid = dict(contract)
    invalid["extra"] = True
    with pytest.raises(SystemExit):
        validate_against_schema(invalid)


def test_real_codex_adapter_fixture_parity():
    fixture = json.loads(
        (ROOT / "tests/fixtures/factorized_scheduler_api_v3_1_trace.json").read_text()
    )
    adapter = FactorizedV2SchedulerAdapter(
        fixture["structure"],
        fixture["calibration"],
    )
    result = adapter.run_episode(fixture["runtime_rows"])
    assert result["first_emit_step"] == fixture["expected"]["first_emit_step"]
    assert [
        row["step"] for row in result["per_step_trace"] if row["emit"]
    ] == fixture["expected"]["emit_steps"]
    assert result["diagnostic_only"] is fixture["expected"]["diagnostic_only"]


def test_authoritative_contract_requires_l3_flags():
    contract = _valid_v3_contract()
    adapter = FactorizedV2SchedulerAdapter(
        _structure(),
        contract,
        require_l3_eligible=True,
    )
    assert adapter.l3_evaluation_eligible
    contract["l3_evaluation_eligible"] = False
    with pytest.raises(Exception):
        FactorizedV2SchedulerAdapter(
            _structure(),
            contract,
            require_l3_eligible=True,
        )


# Threshold selector ------------------------------------------------------------

def test_joint_threshold_selector_uses_real_adapter():
    positive_runtime = [
        _runtime_row(
            index,
            "positive",
            grasp_logit=3.0,
            manipulation_logit=3.0,
        )
        for index in range(10)
    ]
    negative_runtime = [
        _runtime_row(
            index,
            "negative",
            grasp_logit=0.0,
            manipulation_logit=0.0,
        )
        for index in range(10)
    ]
    positive_offline = [
        _offline_row(index, "positive", feasible=index == 0)
        for index in range(10)
    ]
    negative_offline = [_offline_row(index, "negative") for index in range(10)]
    payload = {
        "split": "o0_i0",
        "runtime_episodes": {
            "positive": positive_runtime,
            "negative": negative_runtime,
        },
        "evaluation_episodes": {
            "positive": positive_offline,
            "negative": negative_offline,
        },
        "calibration_contract": _fit_contract(),
        "fit_manifest_sha256": SHA,
        "policy_manifest_sha256": SHA,
        "binding": {
            "checkpoint_sha256": SHA,
            "source_commit": COMMIT,
            "feature_order_sha256": SHA,
            "split": "o0_i0",
            "scheduler_source_sha256": SHA,
            "structural_config_sha256": SHA,
        },
        "structure": _structure(),
    }

    bad = evaluate_candidate(
        [payload],
        {"grasp": 0.4, "manipulation": 0.4, "release": 0.9},
    )
    assert bad["worst_split_negative_false_start_rate"] == 1.0

    best, results = select_thresholds(
        [payload],
        grasp_grid=(0.4, 0.9),
        manipulation_grid=(0.4, 0.9),
        release_grid=(0.9,),
        max_false_start=0.0,
    )
    assert len(results) == 4
    assert best is not None
    assert best["thresholds"]["grasp"] == 0.9
    assert best["thresholds"]["manipulation"] == 0.9
    assert best["aggregate"]["valid_opportunity_recall"] == 1.0


# Handoff validator -------------------------------------------------------------

def test_old_handoffs_rejected():
    ok, _ = validate_handoff_static(
        {"schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V2", "status": "DRAFT"}
    )
    assert not ok
    ok, _ = validate_handoff_static(
        {"schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3", "status": "DRAFT"}
    )
    assert not ok


def test_v31_incomplete_fails_closed():
    ok, errors = validate_handoff_static(
        {
            "schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1",
            "status": "DRAFT",
        }
    )
    assert not ok
    assert errors


def test_v32_is_strict_superset_and_loader_compatible(tmp_path: Path):
    base = json.loads((ROOT / "reports/DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1.json").read_text())
    value = copy.deepcopy(base)
    value["schema"] = "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_2"
    value["interface_revision"] = "V3.2"
    value["production_input_audit"] = {
        "summary": {
            "path": "reports/FACTORIZED_V2_PRODUCTION_INPUT_CHAIN_BLOCKER.json",
            "sha256": canonical_reference_sha(ROOT / "reports/FACTORIZED_V2_PRODUCTION_INPUT_CHAIN_BLOCKER.json"),
        },
        "verdict": True,
    }
    value["identity_audit"] = {
        "summary": {
            "path": "reports/FACTORIZED_CALIBRATION_IDENTITY_FEASIBILITY_AUDIT_V2.json",
            "sha256": canonical_reference_sha(ROOT / "reports/FACTORIZED_CALIBRATION_IDENTITY_FEASIBILITY_AUDIT_V2.json"),
        },
        "verdict": "GROUP_CROSS_FITTED_OOF_FEASIBLE",
    }
    value["production_bundle"] = {
        "root_path": "reports",
        "manifest": {
            "path": "reports/FACTORIZED_V2_PRODUCTION_INPUT_CHAIN_BLOCKER.json",
            "sha256": canonical_reference_sha(ROOT / "reports/FACTORIZED_V2_PRODUCTION_INPUT_CHAIN_BLOCKER.json"),
        },
        "seal": {
            "path": "reports/FACTORIZED_V2_PRODUCTION_INPUT_CHAIN_BLOCKER.json",
            "sha256": canonical_reference_sha(ROOT / "reports/FACTORIZED_V2_PRODUCTION_INPUT_CHAIN_BLOCKER.json"),
        },
        "split_keys": sorted(EXPECTED_SPLITS),
    }
    value["handoff_blob_sha256"] = canonical_handoff_sha(value)
    ok, errors = validate_handoff_static(value)
    assert ok, errors
    blocked = copy.deepcopy(value)
    blocked["identity_audit"]["verdict"] = "BLOCKED_ROOTS_NOT_MOUNTED"
    blocked["handoff_blob_sha256"] = canonical_handoff_sha(blocked)
    ok, errors = validate_handoff_static(blocked)
    assert not ok
    assert "FAIL: V3.2 identity_audit verdict" in errors
    tampered = copy.deepcopy(value)
    tampered["branch"] = "tampered"
    ok, errors = validate_handoff_static(tampered)
    assert not ok
    assert "FAIL: handoff_blob_sha256" in errors
    path = tmp_path / "v3_2.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_handoff_file(path, ROOT)
    assert loaded["schema"] == "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_2"
    ok, errors = validate_handoff_execution(
        {
            "schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1",
            "status": "DRAFT",
        }
    )
    assert not ok
    assert errors


def test_expected_split_count():
    assert len(EXPECTED_SPLITS) == 12
