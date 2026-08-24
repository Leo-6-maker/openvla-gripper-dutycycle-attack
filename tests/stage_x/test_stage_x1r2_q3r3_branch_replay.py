from __future__ import annotations

import json
from pathlib import Path

import pytest

from gripper_attack.stage_x_q3r3_branch_replay import (
    BranchContractError,
    BranchReplay,
    ProtectedCounters,
    ReferenceClean,
    compare_branch_state,
)


def reference() -> ReferenceClean:
    return ReferenceClean.from_record(
        {
            "status": "PASS_REFERENCE_CLEAN",
            "clean_success": True,
            "initial_state": {"seed": 7},
            "dummy_wait_steps": 10,
            "policy_horizon": 20,
            "first_emit_step": 3,
            "t5": 5,
            "h_phys": 10,
            "student_calls": 1,
            "env_actions": [[0.1], [0.2], [0.3], [0.4]],
            "observation_bytes": [b"o0", b"o1", b"o2", b"branch", b"o4"],
        }
    )


def test_reference_timing_and_prefix_replay_have_no_prebranch_model_path():
    branch = BranchReplay(reference(), "TRUE_PGD_T5")
    events: list[tuple[int, tuple[float, ...]]] = []
    assert branch.replay_prefix(lambda step, action: events.append((step, action))) == 3
    assert events == [(0, (0.1,)), (1, (0.2,)), (2, (0.3,))]
    assert branch.common_first_observation == b"branch"
    assert branch.prebranch_actions == ((0.1,), (0.2,), (0.3,))


def test_common_observation_and_attack_boundary_are_fail_closed():
    branch = BranchReplay(reference(), "CLEAN")
    branch.validate_first_decision(3, b"branch")
    with pytest.raises(BranchContractError, match="COMMON_FIRST_OBSERVATION"):
        branch.validate_first_decision(3, b"rerendered")
    with pytest.raises(BranchContractError, match="ATTACKED_STEP_BEFORE_BRANCH"):
        branch.authorize_attacked_step(2, True)
    with pytest.raises(BranchContractError, match="STRUCTURAL_GATES"):
        branch.authorize_attacked_step(3, False)
    branch.authorize_attacked_step(3, True)


def test_branch_state_contract_uses_frozen_tolerance_and_required_fields():
    state = {"model_identity": "m", "suite_task_state_identity": "s", "seed_and_dummy_wait": [7, 10], "wrapper_step_index": 3, "qpos": [1.0], "qvel": [2.0], "act": [3.0], "ctrl": [4.0], "time": 5.0, "mocap_state": [6.0], "task_object_state": {"x": 7.0}, "controller_state": {"y": [8.0]}}
    close = {**state, "qpos": [1.0 + 1e-13]}
    far = {**state, "qpos": [1.0 + 1e-9]}
    assert compare_branch_state(state, close)["equal"] is True
    assert compare_branch_state(state, far)["equal"] is False
    assert compare_branch_state(state, {"model_identity": "m"})["missing"]


def test_protected_counters_start_and_remain_zero():
    counters = ProtectedCounters()
    counters.assert_zero()
    assert counters.as_dict() == {"model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0}


def test_runner_config_and_sources_are_cpu_mock_only():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs/STAGE_X_X1R2_Q3R3_BRANCH_RUNNER_V1.json").read_text(encoding="utf-8"))
    assert config["scientific_authority"] is False
    assert config["requirements"]["no_tolerance_widening"] is True
    for path in (root / "src/gripper_attack/stage_x_q3r3_branch_replay.py", root / "scripts/stage_x/audit_stage_x1r2_q3r3_branch_replay.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "import torch" not in source
        assert "transformers" not in source
        assert "env.step(" not in source
