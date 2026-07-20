"""Tests for R10.4D/R10.4E passive deployment runtime.

Gate E-R2.5C: Production-level CPU tests that import run_passive_episode directly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gripper_attack.r10_4_runtime import FEATURE_NAMES, FEATURE_ORDER_SHA256
from gripper_attack.r10_4d_passive import (
    R10_4DContractError,
    RoutedGraspDetector,
    SUPPORTED_PARENT,
    parse_route,
    run_passive_episode,
    safe_json_value,
    validate_authorization_receipt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fake dependencies
# ═══════════════════════════════════════════════════════════════════════════════

class FakeModel:
    def site_name2id(self, name):
        assert name == "gripper0_grip_site"
        return 0

    def geom_id2name(self, geom_id):
        return f"geom_{geom_id}"


class FakeData:
    def __init__(self):
        self.site_xpos = np.array([[0.5, 0.0, 0.8]], dtype=np.float32)
        self.ncon = 0
        self.contact = []


class FakeEnv:
    """Configurable fake env for testing. Only counts policy steps after dummy wait."""

    def __init__(self, policy_steps_before_done=3):
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData())
        self.policy_steps_before_done = policy_steps_before_done
        self.total_step_calls = 0
        self.policy_step_calls = 0
        self.actions = []
        self._check_success_val = True
        self._check_success_raises = None
        self.observation = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float32),
            "object-state": np.zeros(4, dtype=np.float32),
        }

    def set_check_success(self, value=None, raises=None):
        self._check_success_val = value
        self._check_success_raises = raises

    def set_init_state(self, state):
        self.initial_state = state
        return self.observation

    def step(self, action):
        self.total_step_calls += 1
        self.actions.append(list(action))
        done = False
        if self.total_step_calls > 10:
            self.policy_step_calls += 1
            done = self.policy_step_calls >= self.policy_steps_before_done
        info = {"mock_step": self.total_step_calls, "policy_step": self.policy_step_calls if self.total_step_calls > 10 else 0}
        return self.observation, 0.0, done, info

    def check_success(self):
        if self._check_success_raises is not None:
            raise self._check_success_raises
        return self._check_success_val


class FakeAdapter:
    def __init__(self, generation_passes=1):
        self.generation_passes = generation_passes
        self.calls = 0

    def predict_action(self, *, image_np, task_label, capture=False):
        assert image_np.dtype == np.uint8
        assert capture is True
        self.calls += 1
        raw = np.zeros(7, dtype=np.float32)
        raw[-1] = 1.0
        metadata = {}
        if self.generation_passes is not None:
            metadata["generation_passes_per_step"] = self.generation_passes
        return raw, metadata

    def postprocess(self, action):
        env = np.asarray(action, dtype=np.float32).copy()
        env[-1] = -1.0
        return env


class FakeDetector:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def step(self, features, route):
        values = np.asarray(features, dtype=np.float32)
        assert values.shape == (25,)
        assert np.isfinite(values).all()
        self.calls.append((values.copy(), route))
        return -10.0, 1.0 / (1.0 + np.exp(10.0))


def image_getter(observation):
    return observation["agentview_image"]


def receipt(**overrides):
    value = {
        "schema": "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_AUTH_V1",
        "scope": "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE",
        "passive_only": True,
        "model_load_authorized": True,
        "detector_execution_authorized": True,
        "action_mutation_authorized": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "command_open_authorized": False,
        "visual_attack_authorized": False,
        "random_attack_authorized": False,
        "episodes_authorized": 1,
        "selected_parent": SUPPORTED_PARENT,
        "source_commit": "a" * 40,
        "detector_checkpoint_sha256": "b" * 64,
        "bundle_sha256s_sha256": "c" * 64,
        "model_tree_sha256": "d" * 64,
        "r4c_classification": "CONTACT_DYNAMICS_REPLAY_DIVERGENCE",
        "feature_order_sha256": FEATURE_ORDER_SHA256,
    }
    value.update(overrides)
    return value


def _run(**overrides):
    """Minimal helper for run_passive_episode with defaults.
    Tests can override any kwarg; the helper only supplies defaults for missing keys.
    """
    defaults = {
        "env": FakeEnv(policy_steps_before_done=3),
        "initial_state": {"state": 20},
        "task_language": "put both objects in the basket",
        "identity": SUPPORTED_PARENT,
        "openvla_adapter": FakeAdapter(),
        "detector": FakeDetector(),
        "image_getter": image_getter,
        "max_steps": 20,
    }
    kwargs = {**defaults, **overrides}
    kwargs.setdefault("authorized_parents", frozenset({kwargs["identity"]}))
    return run_passive_episode(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Original R10.4D tests (updated for E-R2.5B)
# ═══════════════════════════════════════════════════════════════════════════════

def test_dual_head_model_parameter_count_is_frozen():
    model = RoutedGraspDetector()
    assert sum(parameter.numel() for parameter in model.parameters()) == 46658


def test_supported_parent_and_fail_closed_routing():
    assert parse_route(SUPPORTED_PARENT) == "multi_object_transfer"
    assert parse_route("libero_object/task_00/state_20") == "unsupported_abstain"
    assert parse_route("libero_10/task_99/state_20") == "unsupported_abstain"


def test_passive_episode_runs_real_step_count_without_recorded_actions():
    env = FakeEnv(policy_steps_before_done=3)
    result = _run(env=env)
    assert result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}
    assert result["n_steps"] == 3
    assert result["action_mutation"] is False
    for row in result["step_records"]:
        assert row["generation_passes_per_step"] == 1
        assert row["action_max_abs_error"] == 0.0
        assert row["clean_env_action_7d"] == row["executed_action_7d"]
        assert len(row["features_25d"]) == len(FEATURE_NAMES) == 25
        assert "info" in row


@pytest.mark.parametrize("generation_passes", [None, 0, 2, True])
def test_generation_contract_fails_closed(generation_passes):
    env = FakeEnv(policy_steps_before_done=3)
    with pytest.raises(R10_4DContractError, match="PASSIVE_GENERATION_COUNT"):
        _run(env=env, max_steps=3, openvla_adapter=FakeAdapter(generation_passes=generation_passes))
    assert env.total_step_calls == 10
    assert env.policy_step_calls == 0


def test_unsupported_parent_never_reaches_model_or_env():
    env = FakeEnv()
    adapter = FakeAdapter()
    with pytest.raises(R10_4DContractError, match="PASSIVE_PARENT_ROUTE_FAIL"):
        run_passive_episode(
            env=env,
            initial_state={},
            task_language="task",
            identity="libero_object/task_00/state_20",
            openvla_adapter=adapter,
            detector=FakeDetector(),
            image_getter=image_getter,
            max_steps=3,
            authorized_parents=frozenset({"libero_object/task_00/state_20"}),
        )
    assert adapter.calls == 0
    assert env.total_step_calls == 0


def test_privileged_sidecar_is_collected_after_detector_and_never_input():
    env = FakeEnv(policy_steps_before_done=1)
    detector = FakeDetector()
    result = run_passive_episode(
        env=env,
        initial_state={},
        task_language="task",
        identity=SUPPORTED_PARENT,
        openvla_adapter=FakeAdapter(),
        detector=detector,
        image_getter=image_getter,
        max_steps=3,
        authorized_parents=frozenset({SUPPORTED_PARENT}),
        privileged_observer=lambda _env, _obs, step: {"step": step, "secret": 123},
    )
    assert result["privileged_runtime_input"] is False
    assert result["privileged_records"] == [{"step": 0, "secret": 123, "detector_input": False}]
    assert len(detector.calls) == 1


def test_authorization_receipt_is_exact_and_attack_fields_fail_closed():
    validate_authorization_receipt(
        receipt(),
        expected_head="a" * 40,
        expected_parent=SUPPORTED_PARENT,
        expected_checkpoint_sha256="b" * 64,
        expected_bundle_sha256s="c" * 64,
        expected_model_tree_sha256="d" * 64,
    )
    with pytest.raises(R10_4DContractError):
        validate_authorization_receipt(
            receipt(formal_attack_authorized=True),
            expected_head="a" * 40,
            expected_parent=SUPPORTED_PARENT,
            expected_checkpoint_sha256="b" * 64,
            expected_bundle_sha256s="c" * 64,
            expected_model_tree_sha256="d" * 64,
        )


def test_unauthorized_parent_rejected():
    env = FakeEnv(policy_steps_before_done=3)
    with pytest.raises(R10_4DContractError, match="PASSIVE_PARENT_NOT_AUTHORIZED"):
        run_passive_episode(
            env=env,
            initial_state={},
            task_language="task",
            identity="libero_10/task_01/state_20",
            openvla_adapter=FakeAdapter(),
            detector=FakeDetector(),
            image_getter=image_getter,
            max_steps=3,
            authorized_parents=frozenset({SUPPORTED_PARENT}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Gate E-R2.5C: Termination classification tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_termination_success():
    """early done + check_success True → SUCCESS_TERMINATION"""
    env = FakeEnv(policy_steps_before_done=5)
    env.set_check_success(True)
    result = _run(env=env, max_steps=50)
    assert result["termination_reason"] == "SUCCESS_TERMINATION"
    assert result["task_success"] is True
    assert result["done"] is True
    assert result["status"] != "FAIL_TERMINATION"


def test_termination_early_done_without_success():
    """early done + check_success False → EARLY_DONE_WITHOUT_SUCCESS (hard failure)"""
    env = FakeEnv(policy_steps_before_done=5)
    env.set_check_success(False)
    result = _run(env=env, max_steps=50)
    assert result["termination_reason"] == "EARLY_DONE_WITHOUT_SUCCESS"
    assert result["task_success"] is False
    assert result["done"] is True
    assert result["status"] == "FAIL_TERMINATION"
    assert "HARD_FAILURE:EARLY_DONE_WITHOUT_SUCCESS" in result["violations"]


def test_termination_horizon():
    """done at horizon, not success → HORIZON_TERMINATION"""
    env = FakeEnv(policy_steps_before_done=50)
    env.set_check_success(False)
    result = _run(env=env, max_steps=50)
    assert result["termination_reason"] == "HORIZON_TERMINATION"
    assert result["task_success"] is False
    assert result["done"] is True
    assert result["status"] == "PASS_RUNTIME_NO_EMIT"


def test_termination_full_loop_task_failure():
    """full policy steps without done, check_success False → FULL_LOOP_TASK_FAILURE"""
    env = FakeEnv(policy_steps_before_done=999)  # never done
    env.set_check_success(False)
    result = _run(env=env, max_steps=10)
    assert result["termination_reason"] == "FULL_LOOP_TASK_FAILURE"
    assert result["task_success"] is False
    assert result["done"] is False
    assert result["n_steps"] == 10
    assert result["status"] == "PASS_RUNTIME_NO_EMIT"


def test_termination_no_steps():
    """max_steps < 1 is rejected at validation (before classification)."""
    with pytest.raises(R10_4DContractError, match="PASSIVE_MAX_STEPS_INVALID"):
        _run(max_steps=0)


def test_check_success_raises():
    """check_success() raises → CHECK_SUCCESS_FAILURE (fail-closed, P0-9)"""
    env = FakeEnv(policy_steps_before_done=5)
    env.set_check_success(raises=RuntimeError("sim exploded"))
    result = _run(env=env, max_steps=50)
    assert result["env_check_success"] is None
    assert result["check_success_error"] is not None
    assert "RuntimeError" in result["check_success_error"]
    assert result["termination_reason"] == "CHECK_SUCCESS_FAILURE"
    assert result["status"] == "FAIL_TERMINATION"


def test_done_never_equals_success():
    """done=True without check_success=True must NOT be task_success"""
    env = FakeEnv(policy_steps_before_done=5)
    env.set_check_success(False)
    result = _run(env=env, max_steps=50)
    assert result["done"] is True
    assert result["task_success"] is False
    assert result["termination_reason"] != "SUCCESS_TERMINATION"


def test_termination_reason_always_present():
    """Every result must have termination_reason."""
    result = _run()
    assert "termination_reason" in result
    assert result["termination_reason"] in {
        "SUCCESS_TERMINATION", "HORIZON_TERMINATION",
        "FULL_LOOP_TASK_FAILURE", "EARLY_DONE_WITHOUT_SUCCESS",
        "NO_STEPS", "UNCLASSIFIED",
    }


def test_horizon_fields_recorded():
    """configured_horizon and observed_horizon must be recorded."""
    env = FakeEnv(policy_steps_before_done=3)
    result = _run(env=env, max_steps=20)
    assert result["configured_horizon"] == 20
    assert result["observed_horizon"] == 3


def test_simulator_timestep_recorded():
    """simulator_timestep must be a float (even if -1.0 for fake env)."""
    result = _run()
    assert isinstance(result["simulator_timestep"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Gate E-R2.5C: safe_json_value / info serialization tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_safe_json_value_numpy_scalar():
    assert safe_json_value(np.int64(42)) == 42
    assert isinstance(safe_json_value(np.int64(42)), int)
    assert safe_json_value(np.float32(3.14)) == pytest.approx(3.14)
    assert isinstance(safe_json_value(np.float32(3.14)), float)
    assert safe_json_value(np.float64(np.nan)) == "nan"
    assert safe_json_value(np.float64(np.inf)) == "inf"


def test_safe_json_value_numpy_array():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = safe_json_value(arr)
    assert result == [1.0, 2.0, 3.0]
    assert json.dumps(result)


def test_safe_json_value_nested_unserializable():
    obj = {
        "scalar": np.float32(1.5),
        "array": np.array([1, 2, 3]),
        "tuple": (1, 2),
        "set": {3, 4},
        "nested": {"deep": np.int64(99)},
    }
    result = safe_json_value(obj)
    dumped = json.dumps(result, sort_keys=True)
    assert json.loads(dumped) is not None


def test_safe_json_value_path_and_bytes():
    from pathlib import Path as _Path
    test_path = _Path("/tmp/test")
    assert safe_json_value(test_path) == str(test_path)
    assert safe_json_value(b"hello") == "hello"


def test_info_in_step_records_is_serializable():
    """info field in step_records must survive json.dumps."""
    result = _run()
    for row in result["step_records"]:
        info = row["info"]
        # Must be a dict
        assert isinstance(info, dict)
        # Must be JSON-serializable
        dumped = json.dumps(info, sort_keys=True, default=str)
        assert json.loads(dumped) is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Gate E-R2.5C: Invariant tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_action_strict_zero_error():
    """Every step must have action_max_abs_error == 0.0."""
    result = _run()
    for row in result["step_records"]:
        assert row["action_max_abs_error"] == 0.0


def test_generation_integer_one():
    """Every step must have generation_passes_per_step == 1 (int)."""
    result = _run()
    for row in result["step_records"]:
        assert isinstance(row["generation_passes_per_step"], int)
        assert row["generation_passes_per_step"] == 1


def test_detector_feature_fsm_reset_per_episode():
    """Detector, feature adapter, and FSM must reset each episode."""
    detector = FakeDetector()
    result1 = _run(detector=detector)
    assert len(detector.calls) > 0
    result2 = _run(detector=detector)
    # detector.reset() is called at top of run_passive_episode
    assert len(detector.calls) == result2["n_steps"]


def test_schema_upgraded_to_v1():
    """Result must use R10_4E schema."""
    result = _run()
    assert result["schema"] == "R10_4E_SINGLE_EPISODE_PASSIVE_RESULT_V1"


def test_task00_requires_explicit_authorization():
    """Backward-compat: without authorized_parents, only SUPPORTED_PARENT works."""
    result = run_passive_episode(
        env=FakeEnv(policy_steps_before_done=1),
        initial_state={},
        task_language="test",
        identity=SUPPORTED_PARENT,
        openvla_adapter=FakeAdapter(),
        detector=FakeDetector(),
        image_getter=image_getter,
        max_steps=3,
    )
    assert result["identity"] == SUPPORTED_PARENT
    # But task_01 should be rejected without explicit authorized_parents
    with pytest.raises(R10_4DContractError, match="PASSIVE_PARENT_NOT_AUTHORIZED"):
        run_passive_episode(
            env=FakeEnv(),
            initial_state={},
            task_language="test",
            identity="libero_10/task_01/state_20",
            openvla_adapter=FakeAdapter(),
            detector=FakeDetector(),
            image_getter=image_getter,
            max_steps=3,
        )
