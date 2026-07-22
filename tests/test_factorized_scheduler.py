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
    value = json.loads(Path("configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json").read_text())
    value.update(changes)
    return FactorizedSchedulerConfig.from_mapping(value)


def _step(step=0, **changes):
    value = {
        "step": step,
        "candidate_close": True,
        "action_known": True,
        "student_valid": True,
        "route_supported": True,
        "grasp_probability": 0.9,
        "grasp_known": True,
        "manipulation_probability": 0.9,
        "manipulation_known": True,
        "release_probability": 0.0,
        "release_known": True,
    }
    value.update(changes)
    return value


def _emit_trace(scheduler):
    return [scheduler.step(_step(step=i)) for i in range(10)]


def test_config_is_required_and_attack_disabled():
    assert _config().attack_enabled is False
    with pytest.raises(FactorizedSchedulerError):
        FactorizedSchedulerConfig.from_mapping({})
    with pytest.raises(FactorizedSchedulerError):
        _config(attack_enabled=True)


def test_close_intent_insufficient_and_boundary_abstain():
    scheduler = FactorizedV2OneShotScheduler(_config())
    assert scheduler.step(_step(candidate_close=False))["reason"] == "NO_CLOSE_INTENT"
    assert scheduler.step(_step(action_known=False))["reason"] == "NO_CLOSE_INTENT"


def test_route_and_student_valid_gates():
    scheduler = FactorizedV2OneShotScheduler(_config())
    assert scheduler.step(_step(route_supported=False))["reason"] == "UNSUPPORTED_ROUTE"
    assert scheduler.step(_step(student_valid=False))["reason"] == "STUDENT_INVALID"


def test_grasp_precedes_manipulation():
    scheduler = FactorizedV2OneShotScheduler(_config())
    assert scheduler.step(_step(grasp_known=False))["reason"] == "GRASP_UNKNOWN"
    assert scheduler.state == FactorizedState.GRASP_EVIDENCE
    assert scheduler.step(_step(grasp_probability=0.1))["reason"] == "GRASP_BELOW_THRESHOLD"
    assert scheduler.step(_step(manipulation_known=False))["reason"] == "MANIPULATION_UNKNOWN"


def test_release_veto_and_unknown_policy():
    scheduler = FactorizedV2OneShotScheduler(_config())
    assert scheduler.step(_step(release_probability=0.9))["reason"] == "RELEASE_VETO"
    scheduler = FactorizedV2OneShotScheduler(_config(unknown_policy="pause"))
    assert scheduler.step(_step(release_known=False))["reason"] == "RELEASE_UNKNOWN_PAUSE"
    scheduler = FactorizedV2OneShotScheduler(_config(unknown_policy="reset"))
    assert scheduler.step(_step(release_known=False))["state_after"] == "IDLE"


def test_dwell_persistence_and_one_shot():
    scheduler = FactorizedV2OneShotScheduler(_config())
    traces = _emit_trace(scheduler)
    assert sum(int(item["emit"]) for item in traces) == 1
    assert traces[-1]["emit"] is True
    assert scheduler.state == FactorizedState.DONE
    assert scheduler.step(_step(step=10))["reason"] == "ONE_SHOT_LATCHED"


def test_persistence_requires_three_of_five():
    scheduler = FactorizedV2OneShotScheduler(_config())
    for i in range(9):
        item = _step(step=i, manipulation_probability=0.1 if i % 2 == 0 else 0.9)
        trace = scheduler.step(item)
    assert trace["emit"] is False


def test_episode_reset_clears_state():
    scheduler = FactorizedV2OneShotScheduler(_config())
    _emit_trace(scheduler)
    scheduler.reset()
    assert scheduler.state == FactorizedState.IDLE
    assert scheduler.emitted is False
    assert scheduler.dwell == 0
    assert scheduler.manipulation_history == []


def test_warmup_and_deterministic_trace():
    config = _config(warmup_steps=2)
    a = FactorizedV2OneShotScheduler(config)
    b = FactorizedV2OneShotScheduler(config)
    trace_a = [a.step(_step(step=i)) for i in range(10)]
    trace_b = [b.step(_step(step=i)) for i in range(10)]
    assert trace_a == trace_b
    assert trace_a[0]["reason"] == "WARMUP"
    assert trace_a[1]["reason"] == "WARMUP"


def test_teacher_event_future_and_action_fields_rejected():
    scheduler = FactorizedV2OneShotScheduler(_config())
    for name in ("event_id", "teacher_phase", "future_score", "attack_outcome", "object_state"):
        row = _step()
        row[name] = 1
        with pytest.raises(FactorizedSchedulerError):
            scheduler.step(row)
    row = _step()
    row["executed_action"] = [0.0] * 7
    with pytest.raises(FactorizedSchedulerError):
        scheduler.step(row)


def test_unknown_heads_are_not_negative_defaults():
    scheduler = FactorizedV2OneShotScheduler(_config())
    trace = scheduler.step(_step(grasp_known=False, manipulation_probability=0.0))
    assert trace["emit"] is False
    assert trace["reason"] == "GRASP_UNKNOWN"


def test_input_mapping_is_not_mutated():
    scheduler = FactorizedV2OneShotScheduler(_config())
    row = _step()
    original = copy.deepcopy(row)
    scheduler.step(row)
    assert row == original
