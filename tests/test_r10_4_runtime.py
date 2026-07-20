from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
from gripper_attack.r10_4_runtime import (
    ACTION_DIM,
    FEATURE_NAMES,
    FEATURE_ORDER_SHA256,
    R10_4ContractError,
    load_model_after_receipt,
    run_common_passive_loop,
    validate_runtime_receipt,
)


class FakeEnv:
    def __init__(self):
        self.calls = []
        self.observation = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

    def set_init_state(self, state):
        self.calls.append(("set_init_state", state))
        return self.observation

    def step(self, action):
        self.calls.append(("step", list(action)))
        return self.observation, 0.0, False, {}


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def predict_action(self, *, image_np, task_label, capture=False):
        self.calls.append((image_np, task_label, capture))
        return np.zeros(ACTION_DIM, dtype=np.float32), {"generation_passes_per_step": 1}

    def postprocess(self, action):
        return np.asarray(action, dtype=np.float32)


class FakeFeatures:
    def reset(self):
        self.reset_count = getattr(self, "reset_count", 0) + 1

    def update_from_observation(self, observation, action):
        return {"valid": True, "features": {name: 0.0 for name in FEATURE_NAMES}}


def receipt(**overrides):
    value = {
        "schema": "R10_4_RUNTIME_AUTHORIZATION_RECEIPT_V1",
        "scope": "R10_4_R4_RUNTIME_INTEGRATION",
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "feature_order_sha256": FEATURE_ORDER_SHA256,
        "model_load_authorized": False,
    }
    value.update(overrides)
    return value


def test_official_adapter_signature_is_not_reimplemented():
    signature = inspect.signature(OfficialOpenVLAActionAdapter.predict_action)
    assert list(signature.parameters)[:3] == ["self", "image_np", "task_label"]
    assert "capture" in signature.parameters


def test_injected_common_control_flow_reaches_episode_loop_without_model_load():
    env = FakeEnv()
    adapter = FakeAdapter()
    rows = run_common_passive_loop(
        env=env,
        initial_state={"state": 1},
        task_language="task",
        adapter=adapter,
        feature_adapter=FakeFeatures(),
        image_getter=lambda observation: observation["agentview_image"],
        actions=[np.zeros(ACTION_DIM, dtype=np.float32)],
    )
    assert len(rows) == 1
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0].dtype == np.uint8
    assert adapter.calls[0][2] is True
    assert [call[0] for call in env.calls].count("set_init_state") == 1
    assert [call[0] for call in env.calls].count("step") == 11


def test_invalid_receipt_blocks_loader_before_invocation():
    called = []
    with pytest.raises(R10_4ContractError):
        load_model_after_receipt(receipt(model_load_authorized=False), lambda: called.append(True))
    assert called == []


def test_fake_receipt_is_not_formal_or_attack_authorization():
    validate_runtime_receipt(receipt(), require_model_load=False)
    with pytest.raises(R10_4ContractError):
        validate_runtime_receipt(receipt(formal_attack_authorized=True), require_model_load=False)


def test_r10_runtime_has_no_external_image_normalization():
    path = Path(__file__).parents[1] / "src" / "gripper_attack" / "r10_4_runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            assert not (isinstance(node.right, ast.Constant) and node.right.value == 255)


def test_feature_contract_hash_is_frozen():
    from gripper_attack.r10_4_runtime import feature_order_sha256

    assert feature_order_sha256() == FEATURE_ORDER_SHA256
