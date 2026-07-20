from __future__ import annotations

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
    validate_authorization_receipt,
)


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
    def __init__(self, policy_steps_before_done=3):
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData())
        self.policy_steps_before_done = policy_steps_before_done
        self.total_step_calls = 0
        self.policy_step_calls = 0
        self.actions = []
        self.observation = {
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float32),
            "object-state": np.zeros(4, dtype=np.float32),
        }

    def set_init_state(self, state):
        self.initial_state = state
        return self.observation

    def step(self, action):
        self.total_step_calls += 1
        self.actions.append(list(action))
        # The first ten calls are the frozen dummy wait.
        done = False
        if self.total_step_calls > 10:
            self.policy_step_calls += 1
            done = self.policy_step_calls >= self.policy_steps_before_done
        return self.observation, 0.0, done, {}

    def check_success(self):
        return self.policy_step_calls >= self.policy_steps_before_done


class FakeAdapter:
    def __init__(self, generation_passes=1):
        self.generation_passes = generation_passes
        self.calls = 0

    def predict_action(self, *, image_np, task_label, capture=False):
        assert image_np.dtype == np.uint8
        assert capture is True
        self.calls += 1
        raw = np.zeros(7, dtype=np.float32)
        raw[-1] = 1.0  # official raw OPEN
        metadata = {}
        if self.generation_passes is not None:
            metadata["generation_passes_per_step"] = self.generation_passes
        return raw, metadata

    def postprocess(self, action):
        env = np.asarray(action, dtype=np.float32).copy()
        env[-1] = -1.0  # official LIBERO OPEN
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


def test_dual_head_model_parameter_count_is_frozen():
    model = RoutedGraspDetector()
    assert sum(parameter.numel() for parameter in model.parameters()) == 46658


def test_supported_parent_and_fail_closed_routing():
    assert parse_route(SUPPORTED_PARENT) == "multi_object_transfer"
    assert parse_route("libero_object/task_00/state_20") == "unsupported_abstain"
    assert parse_route("libero_10/task_99/state_20") == "unsupported_abstain"


def test_passive_episode_runs_real_step_count_without_recorded_actions():
    env = FakeEnv(policy_steps_before_done=3)
    adapter = FakeAdapter(generation_passes=1)
    detector = FakeDetector()
    result = run_passive_episode(
        env=env,
        initial_state={"state": 20},
        task_language="put both objects in the basket",
        identity=SUPPORTED_PARENT,
        openvla_adapter=adapter,
        detector=detector,
        image_getter=image_getter,
        max_steps=20,
    )
    assert result["status"] == "PASS_RUNTIME_NO_EMIT"
    assert result["n_steps"] == 3
    assert adapter.calls == 3
    assert len(detector.calls) == 3
    assert env.total_step_calls == 13
    assert result["action_mutation"] is False
    for row in result["step_records"]:
        assert row["generation_passes_per_step"] == 1
        assert row["action_max_abs_error"] == 0.0
        assert row["clean_env_action_7d"] == row["executed_action_7d"]
        assert len(row["features_25d"]) == len(FEATURE_NAMES) == 25


@pytest.mark.parametrize("generation_passes", [None, 0, 2, True])
def test_generation_contract_fails_closed(generation_passes):
    env = FakeEnv(policy_steps_before_done=3)
    with pytest.raises(R10_4DContractError, match="PASSIVE_GENERATION_COUNT"):
        run_passive_episode(
            env=env,
            initial_state={},
            task_language="task",
            identity=SUPPORTED_PARENT,
            openvla_adapter=FakeAdapter(generation_passes=generation_passes),
            detector=FakeDetector(),
            image_getter=image_getter,
            max_steps=3,
        )
    # Only the ten immutable dummy-wait actions may have reached the env.
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
