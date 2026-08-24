from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gripper_attack.factorized_scheduler import (
    FactorizedSchedulerConfig,
    FactorizedSchedulerError,
    FactorizedState,
    FactorizedV2OneShotScheduler,
)


def _config(**changes):
    structure = json.loads(Path("configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json").read_text())
    calibration = json.loads(Path("tests/fixtures/factorized_calibration_fixture.json").read_text())
    structure.update(changes)
    return FactorizedSchedulerConfig.from_mapping(structure, calibration)


def _step(step=0, **changes):
    value = {
        "step": step,
        "candidate_close": True,
        "action_known": True,
        "student_valid": True,
        "route_supported": True,
        "grasp_probability": 0.9,
        "manipulation_probability": 0.9,
        "release_probability": 0.0,
    }
    value.update(changes)
    return value


def _emit_trace(scheduler):
    return [scheduler.step(_step(step=i)) for i in range(10)]


def test_external_calibration_is_required_and_thresholds_are_not_structural():
    structure = json.loads(Path("configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json").read_text())
    with pytest.raises(FactorizedSchedulerError, match="CALIBRATION_CONTRACT_REQUIRED"):
        FactorizedSchedulerConfig.from_mapping(structure)
    assert "grasp_threshold" not in structure
    with pytest.raises(FactorizedSchedulerError):
        _config(attack_enabled=True)


def test_runtime_gates_reset_and_boundary_abstains():
    scheduler = FactorizedV2OneShotScheduler(_config())
    assert scheduler.step(_step(candidate_close=False))["reason"] == "NO_CLOSE_INTENT"
    assert scheduler.step(_step(action_known=False))["reason"] == "ACTION_UNKNOWN"
    assert scheduler.step(_step(route_supported=False))["reason"] == "UNSUPPORTED_ROUTE"
    assert scheduler.step(_step(student_valid=False))["reason"] == "STUDENT_INVALID"


def test_missing_or_nonfinite_runtime_probability_fails_closed():
    scheduler = FactorizedV2OneShotScheduler(_config())
    with pytest.raises(FactorizedSchedulerError):
        scheduler.step(_step(grasp_probability=None))
    with pytest.raises(FactorizedSchedulerError):
        scheduler.step(_step(release_probability=float("nan")))


def test_teacher_masks_and_legacy_heads_are_rejected():
    scheduler = FactorizedV2OneShotScheduler(_config())
    for name in ("event_id", "teacher_phase", "strict_k10_feasible", "grasp_known", "known_mask", "utility_probability", "regrasp_probability"):
        row = _step()
        row[name] = 1
        with pytest.raises(FactorizedSchedulerError):
            scheduler.step(row)


def test_candidate_dwell_semantics_are_explicit():
    scheduler = FactorizedV2OneShotScheduler(_config())
    trace = scheduler.step(_step(grasp_probability=0.1))
    assert trace["dwell"] == 1
    assert trace["candidate_dwell_counts_before_grasp"] is True
    assert trace["emit"] is False


def test_release_veto_one_shot_and_no_action_mutation():
    scheduler = FactorizedV2OneShotScheduler(_config())
    assert scheduler.step(_step(release_probability=0.9))["reason"] == "RELEASE_VETO"
    traces = _emit_trace(scheduler)
    assert sum(int(item["emit"]) for item in traces) == 1
    assert scheduler.state == FactorizedState.DONE
    assert scheduler.step(_step(step=10))["reason"] == "ONE_SHOT_LATCHED"


def test_persistence_requires_three_of_five():
    scheduler = FactorizedV2OneShotScheduler(_config())
    last = None
    for i in range(9):
        last = scheduler.step(_step(step=i, manipulation_probability=0.1 if i % 2 == 0 else 0.9))
    assert last is not None and last["emit"] is False


def test_episode_reset_and_deterministic_trace():
    a = FactorizedV2OneShotScheduler(_config())
    b = FactorizedV2OneShotScheduler(_config())
    trace_a = _emit_trace(a)
    trace_b = _emit_trace(b)
    assert trace_a == trace_b
    a.reset()
    assert a.state == FactorizedState.IDLE
    assert a.emitted is False and a.dwell == 0 and a.manipulation_history == []


def test_input_mapping_is_not_mutated():
    scheduler = FactorizedV2OneShotScheduler(_config())
    row = _step()
    original = copy.deepcopy(row)
    scheduler.step(row)
    assert row == original
