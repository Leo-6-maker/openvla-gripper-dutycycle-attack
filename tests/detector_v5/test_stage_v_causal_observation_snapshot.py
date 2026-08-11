from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gripper_attack.stage_v_causal_observation_snapshot import (
    CausalSnapshotError,
    assert_exact,
    assert_primary_observation_exact,
    capture_runtime_state,
    capture_simulator_state,
    load_snapshot,
    matched_action,
    reference_action_window,
    write_snapshot,
)


def test_snapshot_roundtrip_preserves_raw_arrays(tmp_path: Path) -> None:
    payload = {
        "raw_observation": {"agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)},
        "policy_input": {
            "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
            "pixel_values": np.asarray([[[[0.25]]]], dtype=np.float32),
        },
        "reference_actions": [{"env_action": np.asarray([0.0] * 7, dtype=np.float32)}],
    }
    manifest = write_snapshot(tmp_path, payload, binding={"parent": "p", "probe": "Q00"})
    loaded = load_snapshot(tmp_path)
    assert manifest["schema"] == "STAGE_V_CAUSAL_PROBE_SNAPSHOT_V2"
    assert_exact(loaded["payload"], payload, label="roundtrip")
    assert loaded["manifest"]["fresh_render_equality_gate_used"] is False


def test_snapshot_tamper_fails_closed(tmp_path: Path) -> None:
    write_snapshot(tmp_path, {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}, binding={"probe": "Q00"})
    descriptor = json.loads((tmp_path / "CAUSAL_PROBE_SNAPSHOT_V2.json").read_text(encoding="utf-8"))["arrays"][0]
    target = tmp_path / descriptor["binary_path"]
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(CausalSnapshotError, match="ARRAY_BYTES_SHA_MISMATCH"):
        load_snapshot(tmp_path)


def test_primary_observation_digest_binding_fails_closed() -> None:
    payload = {
        "raw_observation": {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)},
        "canonical_policy_rgb_224": np.zeros((2, 2, 3), dtype=np.uint8),
        "processed_image": np.asarray([1.0], dtype=np.float32),
        "input_ids": np.asarray([[1, 2]], dtype=np.int64),
        "pixel_values": np.asarray([[[[0.25]]]], dtype=np.float32),
        "attention_mask": None,
        "prompt": "pick",
        "decode_config": {"do_sample": False},
    }
    payload.update(assert_primary_observation_exact({**payload, **{}}))
    assert_primary_observation_exact(payload)
    payload["prompt"] = "tampered"
    with pytest.raises(CausalSnapshotError, match="policy_input_sha256"):
        assert_primary_observation_exact(payload)


def test_runtime_capture_records_controller_wrapper_observable_and_rng_state() -> None:
    class Buffer:
        def __init__(self) -> None:
            self.buf = np.asarray([[1.0, 2.0]], dtype=np.float32)
            self.index = 1

    class Interpolator:
        start = np.asarray([0.0])
        goal = np.asarray([1.0])
        step = 2
        total_steps = 3

    class Controller:
        goal_pos = np.asarray([1.0, 2.0, 3.0])
        goal_ori = np.eye(3)
        interpolator_pos = Interpolator()

    class Robot:
        controller = Controller()
        recent_actions = Buffer()

    class Observable:
        _time_since_last_sample = 0.1
        _current_delay = 0.0
        _current_observed_value = np.asarray([1.0])
        _sampled = True

    class Inner:
        timestep = 4
        cur_time = 0.2
        done = False
        horizon = 20
        robots = [Robot()]
        _observables = {"camera": Observable()}

    class Env:
        env = Inner()

    state = capture_runtime_state(Env())
    assert state["environment"]["timestep"] == 4
    assert state["robots"][0]["controller"]["interpolator_pos"]["step"] == 2
    assert state["observables"]["camera"]["_sampled"] is True
    assert "python_random" in state["rng"]


def test_reference_window_and_surgical_open_action() -> None:
    rows = [{"step": i, "raw_action": [0.1] * 6 + [0.2], "env_action": [0.1] * 6 + [1.0]} for i in range(4)]
    window = reference_action_window(rows, start_step=1, length=2)
    assert [row["step"] for row in window] == [1, 2]
    open_action = matched_action(window[0], forced_open=True)
    assert open_action["arm_delta_linf"] == 0.0
    assert open_action["env_action"][-1] == -1.0


def test_simulator_capture_keeps_registered_and_data_state() -> None:
    class Data:
        qpos = np.asarray([1.0, 2.0], dtype=np.float64)
        qvel = np.asarray([3.0, 4.0], dtype=np.float64)

    class Sim:
        data = Data()

    class Env:
        sim = Sim()

        def get_sim_state(self):
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

    state = capture_simulator_state(Env())
    assert state["schema"] == "STAGE_V_FULL_SIM_STATE_DIAGNOSTIC_V2"
    assert state["data"]["qpos"].tolist() == [1.0, 2.0]
    assert state["registered_flat_state"].tolist() == [1.0, 2.0, 3.0]
