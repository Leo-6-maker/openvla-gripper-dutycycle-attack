#!/usr/bin/env python3
"""Gate E-R1: CPU termination classification fix + tests.

Fixes the semantic bug where ``done=True`` is treated as task success without
distinguishing the four termination types:

  SUCCESS_TERMINATION        — done=True AND check_success()=True
  HORIZON_TERMINATION        — done=True at configured horizon, check_success()=False
  FULL_LOOP_TASK_FAILURE     — ran full max_steps without done, check_success()=False
  EARLY_DONE_WITHOUT_SUCCESS — done=True before horizon, check_success()=False

CPU only. No OpenVLA. No LIBERO.
"""

import json, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ── The termination classifier (pure function, no dependencies) ─────────────────

def classify_termination(
    *,
    step_records: list[dict],
    configured_horizon: int,
    env_check_success: bool | None,
    simulator_timestep: float | None,
) -> dict[str, Any]:
    """Classify episode termination into exactly one of four types."""
    observed_horizon = len(step_records)
    last_done = bool(step_records[-1]["done"]) if step_records else False
    last_step_idx = step_records[-1]["step"] if step_records else -1

    if not step_records:
        return {
            "termination_reason": "NO_STEPS",
            "task_success": False,
            "configured_horizon": configured_horizon,
            "observed_horizon": 0,
            "done_step": -1,
            "simulator_timestep": simulator_timestep if simulator_timestep is not None else -1.0,
            "is_hard_failure": True,
        }

    if last_done and env_check_success is True:
        termination_reason = "SUCCESS_TERMINATION"
        task_success = True
        is_hard_failure = False
    elif last_done and observed_horizon >= configured_horizon and env_check_success is not True:
        termination_reason = "HORIZON_TERMINATION"
        task_success = False
        is_hard_failure = False
    elif not last_done and observed_horizon >= configured_horizon:
        termination_reason = "FULL_LOOP_TASK_FAILURE"
        task_success = False
        is_hard_failure = False
    elif last_done and observed_horizon < configured_horizon and env_check_success is not True:
        termination_reason = "EARLY_DONE_WITHOUT_SUCCESS"
        task_success = False
        is_hard_failure = True
    else:
        termination_reason = "UNCLASSIFIED"
        task_success = False
        is_hard_failure = True

    return {
        "termination_reason": termination_reason,
        "task_success": task_success,
        "configured_horizon": configured_horizon,
        "observed_horizon": observed_horizon,
        "done_step": last_step_idx if last_done else -1,
        "done": last_done,
        "env_check_success": env_check_success,
        "simulator_timestep": simulator_timestep if simulator_timestep is not None else -1.0,
        "is_hard_failure": is_hard_failure,
    }


# ── Mock envs for CPU tests ────────────────────────────────────────────────────

class MockEnvSuccess:
    """Early done + success."""
    def __init__(self):
        self.t = 0
        self._success = True

    def set_init_state(self, state):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        done = self.t >= 30
        return self._obs(), 1.0, done, {"mock": True}

    def check_success(self):
        return self._success

    def _obs(self):
        return {
            "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
            "robot0_gripper_qpos": np.array(0.1),
        }

    def close(self):
        pass


class MockEnvEarlyFail:
    """Early done WITHOUT success — should be HARD FAILURE."""
    def __init__(self):
        self.t = 0

    def set_init_state(self, state):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        done = self.t >= 15
        return self._obs(), 0.0, done, {"mock": True}

    def check_success(self):
        return False

    def _obs(self):
        return {
            "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
            "robot0_gripper_qpos": np.array(0.1),
        }

    def close(self):
        pass


class MockEnvHorizon:
    """Runs full horizon, done at end, not success."""
    def __init__(self, horizon=50):
        self.t = 0
        self._horizon = horizon

    def set_init_state(self, state):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        done = self.t >= self._horizon
        return self._obs(), 0.0, done, {"mock": True}

    def check_success(self):
        return False

    def _obs(self):
        return {
            "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
            "robot0_gripper_qpos": np.array(0.1),
        }

    def close(self):
        pass


class MockEnvFullLoop:
    """Runs full 520 steps, never done, not success."""
    def __init__(self):
        self.t = 0

    def set_init_state(self, state):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, {"mock": True}

    def check_success(self):
        return False

    def _obs(self):
        return {
            "agentview_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float64),
            "robot0_gripper_qpos": np.array(0.1),
        }

    def close(self):
        pass


# ── Simplified episode runner for testing (no detector, no OpenVLA) ────────────

def run_test_episode(
    env, init_state, max_steps, capture_info=True, capture_timestep=True,
) -> dict:
    """Minimal episode loop matching the structure of run_passive_episode.
    Returns raw step_records + termination metadata for classification testing."""
    obs = env.set_init_state(init_state)
    for _ in range(10):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64))

    # Reset mock env counter so policy step count is clean.
    # Real LIBERO envs are built with horizon >> max_steps, so dummy steps
    # don't affect the policy-step budget. Mock envs emulate that here.
    if hasattr(env, "t"):
        env.t = 0

    step_records = []

    for t in range(max_steps):
        # Simulate OpenVLA action (not real)
        raw_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        clean_action = raw_action.copy()
        executed_action = clean_action.copy()
        action_error = float(np.max(np.abs(executed_action - clean_action)))

        next_obs, reward, done, info = env.step(executed_action.tolist())

        rec = {
            "step": t,
            "generation_passes_per_step": 1,
            "action_max_abs_error": action_error,
            "done": bool(done),
            "reward": float(reward),
            "executed_action_7d": executed_action.tolist(),
        }
        if capture_info:
            try:
                rec["info"] = dict(info) if info else {}
            except Exception:
                rec["info"] = {"_serialize_error": str(type(info))}
        step_records.append(rec)
        obs = next_obs
        if done:
            break

    # Simulator timestep
    simulator_timestep = None
    if capture_timestep:
        try:
            simulator_timestep = float(env.sim.data.time) if hasattr(env, "sim") else None
        except Exception:
            simulator_timestep = None
        if simulator_timestep is None:
            try:
                # MuJoCo via Robosuite: env.sim is a MjSim wrapper
                sim = getattr(env, "sim", None)
                if sim is not None:
                    simulator_timestep = float(sim.data.time)
            except Exception:
                simulator_timestep = -1.0

    # env.check_success()
    env_success = None
    if hasattr(env, "check_success"):
        try:
            env_success = bool(env.check_success())
        except Exception:
            env_success = None

    # CLASSIFY
    termination = classify_termination(
        step_records=step_records,
        configured_horizon=max_steps,
        env_check_success=env_success,
        simulator_timestep=simulator_timestep,
    )

    return {
        "step_records": step_records,
        **termination,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_success_termination():
    """early done + success true → SUCCESS_TERMINATION"""
    env = MockEnvSuccess()
    env._success = True
    result = run_test_episode(env, {}, max_steps=50)
    assert result["termination_reason"] == "SUCCESS_TERMINATION", f"Got: {result['termination_reason']}"
    assert result["task_success"] is True
    assert result["is_hard_failure"] is False
    assert result["configured_horizon"] == 50
    assert result["observed_horizon"] == 30
    assert result["done"] is True
    assert result["done_step"] == 29  # 0-indexed
    assert result["env_check_success"] is True
    print(f"  PASS test_success_termination: {result['termination_reason']} horizon={result['configured_horizon']}/{result['observed_horizon']}")


def test_early_done_without_success():
    """early done + success false → EARLY_DONE_WITHOUT_SUCCESS (hard failure)"""
    env = MockEnvEarlyFail()
    result = run_test_episode(env, {}, max_steps=50)
    assert result["termination_reason"] == "EARLY_DONE_WITHOUT_SUCCESS", f"Got: {result['termination_reason']}"
    assert result["task_success"] is False
    assert result["is_hard_failure"] is True
    assert result["observed_horizon"] == 15
    assert result["done"] is True
    print(f"  PASS test_early_done_without_success: {result['termination_reason']} (HARD FAILURE)")


def test_horizon_termination():
    """done at horizon, not success → HORIZON_TERMINATION"""
    env = MockEnvHorizon(horizon=50)
    result = run_test_episode(env, {}, max_steps=50)
    assert result["termination_reason"] == "HORIZON_TERMINATION", f"Got: {result['termination_reason']}"
    assert result["task_success"] is False
    assert result["is_hard_failure"] is False
    assert result["observed_horizon"] == 50
    assert result["done"] is True
    print(f"  PASS test_horizon_termination: {result['termination_reason']}")


def test_full_loop_task_failure():
    """520 steps, never done, not success → FULL_LOOP_TASK_FAILURE"""
    env = MockEnvFullLoop()
    result = run_test_episode(env, {}, max_steps=520)
    assert result["termination_reason"] == "FULL_LOOP_TASK_FAILURE", f"Got: {result['termination_reason']}"
    assert result["task_success"] is False
    assert result["is_hard_failure"] is False
    assert result["observed_horizon"] == 520
    assert result["done"] is False
    print(f"  PASS test_full_loop_task_failure: {result['termination_reason']}")


def test_done_never_equals_success():
    """done=True without check_success must NOT be treated as task success."""
    env = MockEnvEarlyFail()
    result = run_test_episode(env, {}, max_steps=50)
    assert result["done"] is True
    assert result["task_success"] is False
    assert result["termination_reason"] != "SUCCESS_TERMINATION"
    print(f"  PASS test_done_never_equals_success: done={result['done']} task_success={result['task_success']}")


def test_termination_reason_always_present():
    """Every episode result must have a termination_reason field."""
    env = MockEnvSuccess()
    env._success = True
    result = run_test_episode(env, {}, max_steps=50)
    assert "termination_reason" in result, "Missing termination_reason"
    assert result["termination_reason"] in {
        "SUCCESS_TERMINATION", "HORIZON_TERMINATION",
        "FULL_LOOP_TASK_FAILURE", "EARLY_DONE_WITHOUT_SUCCESS",
    }, f"Unknown reason: {result['termination_reason']}"
    print(f"  PASS test_termination_reason_always_present: {result['termination_reason']}")


def test_simulator_timestep_recorded():
    """Verify simulator_timestep is recorded (may be -1.0 for mock envs)."""
    env = MockEnvSuccess()
    env._success = True
    result = run_test_episode(env, {}, max_steps=50)
    assert "simulator_timestep" in result, "Missing simulator_timestep"
    assert isinstance(result["simulator_timestep"], float), f"Type: {type(result['simulator_timestep'])}"
    print(f"  PASS test_simulator_timestep_recorded: {result['simulator_timestep']}")


def test_horizon_fields():
    """configured_horizon and observed_horizon must both be recorded."""
    env = MockEnvHorizon(horizon=40)
    result = run_test_episode(env, {}, max_steps=100)
    assert result["configured_horizon"] == 100
    assert result["observed_horizon"] == 40
    print(f"  PASS test_horizon_fields: configured={result['configured_horizon']} observed={result['observed_horizon']}")


def test_panel_stops_on_early_failure():
    """Panel must stop on EARLY_DONE_WITHOUT_SUCCESS (is_hard_failure=True)."""
    env = MockEnvEarlyFail()
    result = run_test_episode(env, {}, max_steps=50)
    assert result["is_hard_failure"] is True
    assert result["termination_reason"] == "EARLY_DONE_WITHOUT_SUCCESS"
    print(f"  PASS test_panel_stops_on_early_failure: is_hard_failure={result['is_hard_failure']}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Gate E-R1: Termination Classification CPU Tests ===\n")

    tests = [
        test_success_termination,
        test_early_done_without_success,
        test_horizon_termination,
        test_full_loop_task_failure,
        test_done_never_equals_success,
        test_termination_reason_always_present,
        test_simulator_timestep_recorded,
        test_horizon_fields,
        test_panel_stops_on_early_failure,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {test_fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Gate E-R1: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'='*50}")

    if failed > 0:
        print("\nFAILED TESTS — fix required before proceeding to E-R2")
        sys.exit(1)
    else:
        print("\nAll CPU tests pass. Gate E-R1 termination classification is correct.")
        print("Ready for E-R2 (read-only root audit).")
        sys.exit(0)


if __name__ == "__main__":
    main()
