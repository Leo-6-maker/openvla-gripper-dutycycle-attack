"""CPU tests for R10.4E production episode and termination contracts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gripper_attack.r10_4d_passive import (
    R10_4DContractError,
    _classify_termination,
    run_passive_episode,
    safe_json_value,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "r10_4"))


class FakeModel:
    def site_name2id(self, _name):
        return 0

    def geom_id2name(self, geom_id):
        return f"geom_{geom_id}"


class FakeData:
    def __init__(self):
        self.site_xpos = np.array([[0.5, 0.0, 0.8]], dtype=np.float32)
        self.ncon = 0
        self.contact = []
        self.time = 0.0


class FakeEnv:
    def __init__(self, policy_steps: int = 3):
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData())
        self.policy_steps = policy_steps
        self.total_calls = 0
        self.policy_calls = 0
        self.actions: list[list[float]] = []
        self._success = True
        self._raises: Exception | None = None
        self.obs = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float32),
            "object-state": np.zeros(4, dtype=np.float32),
        }

    def set_check_success(self, value=None, raises=None):
        self._success = value
        self._raises = raises

    def set_init_state(self, _state):
        return self.obs

    def step(self, action):
        self.total_calls += 1
        self.actions.append(list(action))
        done = False
        if self.total_calls > 10:
            self.policy_calls += 1
            done = self.policy_calls >= self.policy_steps
        return self.obs, 0.0, done, {"step": self.total_calls}

    def check_success(self):
        if self._raises is not None:
            raise self._raises
        return self._success

    def close(self):
        return None


class FakeAdapter:
    def __init__(self, generation_passes=1):
        self.generation_passes = generation_passes
        self.calls = 0

    def predict_action(self, *, image_np, task_label, capture=False):
        assert image_np.dtype == np.uint8
        assert capture is True
        self.calls += 1
        action = np.zeros(7, dtype=np.float32)
        action[-1] = 1.0
        metadata = {}
        if self.generation_passes is not None:
            metadata["generation_passes_per_step"] = self.generation_passes
        return action, metadata

    def postprocess(self, action):
        result = np.asarray(action, dtype=np.float32).copy()
        result[-1] = -1.0
        return result


class FakeDetector:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def step(self, features, route):
        values = np.asarray(features, dtype=np.float32)
        self.calls.append((values.copy(), route))
        return -10.0, 1.0 / (1.0 + np.exp(10.0))


def image_getter(observation):
    return observation["agentview_image"]


def run_episode(**overrides):
    defaults = {
        "env": FakeEnv(policy_steps=3),
        "initial_state": {},
        "task_language": "test",
        "identity": "libero_10/task_01/state_20",
        "openvla_adapter": FakeAdapter(),
        "detector": FakeDetector(),
        "image_getter": image_getter,
        "max_steps": 20,
        "authorized_parents": frozenset({"libero_10/task_01/state_20"}),
    }
    defaults.update(overrides)
    return run_passive_episode(**defaults)


def test_success_termination() -> None:
    env = FakeEnv(policy_steps=5)
    env.set_check_success(True)
    result = run_episode(env=env, max_steps=50)
    assert result["termination_reason"] == "SUCCESS_TERMINATION"
    assert result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}


def test_early_done_without_success_is_hard_failure() -> None:
    env = FakeEnv(policy_steps=3)
    env.set_check_success(False)
    result = run_episode(env=env, max_steps=50)
    assert result["termination_reason"] == "EARLY_DONE_WITHOUT_SUCCESS"
    assert result["status"] == "FAIL_TERMINATION"


def test_horizon_termination_is_runtime_valid() -> None:
    env = FakeEnv(policy_steps=20)
    env.set_check_success(False)
    result = run_episode(env=env, max_steps=20)
    assert result["termination_reason"] == "HORIZON_TERMINATION"
    assert result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}


def test_full_loop_task_failure_is_runtime_valid() -> None:
    env = FakeEnv(policy_steps=999)
    env.set_check_success(False)
    result = run_episode(env=env, max_steps=10)
    assert result["termination_reason"] == "FULL_LOOP_TASK_FAILURE"
    assert result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}


def test_check_success_exception_is_fail_closed() -> None:
    env = FakeEnv(policy_steps=20)
    env.set_check_success(raises=RuntimeError("boom"))
    result = run_episode(env=env, max_steps=20)
    assert result["termination_reason"] == "CHECK_SUCCESS_FAILURE"
    assert result["status"] == "FAIL_TERMINATION"


def test_classifier_no_steps_is_hard_failure() -> None:
    result = _classify_termination([], 10, FakeEnv(), [])
    assert result["termination_reason"] == "NO_STEPS"
    assert result["is_hard_failure"] is True


def test_authorized_parent_is_exact() -> None:
    result = run_episode()
    assert result["identity"] == "libero_10/task_01/state_20"
    with pytest.raises(R10_4DContractError, match="PASSIVE_PARENT_NOT_AUTHORIZED"):
        run_episode(
            identity="libero_10/task_02/state_20",
            authorized_parents=frozenset({"libero_10/task_01/state_20"}),
        )


def test_generation_and_action_contracts() -> None:
    result = run_episode()
    for row in result["step_records"]:
        assert type(row["generation_passes_per_step"]) is int
        assert row["generation_passes_per_step"] == 1
        assert row["action_max_abs_error"] == 0.0
        assert row["clean_env_action_7d"] == row["executed_action_7d"]
        assert len(row["features_25d"]) == 25


@pytest.mark.parametrize("generation_passes", [None, 0, 2, True])
def test_invalid_generation_count_fails_closed(generation_passes) -> None:
    with pytest.raises(R10_4DContractError, match="PASSIVE_GENERATION_COUNT"):
        run_episode(openvla_adapter=FakeAdapter(generation_passes=generation_passes))


def test_privileged_sidecar_is_not_runtime_input() -> None:
    result = run_episode(
        privileged_observer=lambda _env, _obs, step: {"step": step, "secret": 123}
    )
    assert result["privileged_runtime_input"] is False
    assert all(row["detector_input"] is False for row in result["privileged_records"])


def test_safe_json_value_nested_types() -> None:
    value = {
        "scalar": np.float32(1.5),
        "array": np.array([1, 2, 3]),
        "tuple": (1, 2),
        "set": {3, 4},
        "path": Path("/tmp/example"),
        "bytes": b"hello",
    }
    normalized = safe_json_value(value)
    assert json.loads(json.dumps(normalized, sort_keys=True)) is not None


def test_new_auditor_rejects_unsealed_empty_root() -> None:
    from audit_r10_4e_sealed_roots import audit_fresh_task01

    with tempfile.TemporaryDirectory() as directory:
        report = audit_fresh_task01(
            Path(directory),
            expected_head="a" * 40,
            expected_comment_id=1,
            expected_receipt_sha="b" * 64,
        )
        assert report["valid"] is False
