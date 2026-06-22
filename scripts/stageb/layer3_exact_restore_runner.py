#!/usr/bin/env python3
"""Clean-only exact-restore runner scaffolding for Layer3.

This file is deliberately attack-free.  It provides the state capture/restore
and five-step replay comparison primitives that the later GPU runner will use
before VIS/RAND are allowed.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:  # torch is optional for CPU unit tests that do not touch CUDA.
    import torch
except Exception:  # pragma: no cover - exercised only in minimal environments.
    torch = None  # type: ignore[assignment]

from scripts.stageb.layer3_exact_branching_contract import (
    BranchRunRecord,
    Layer3BranchingContractError,
    PrefixBranchSnapshot,
    arm_preservation_telemetry,
    require_sha256,
    sha256_jsonable,
    validate_branch_records,
)

SNAPSHOT_BOUNDARY = "PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T"
FLOAT_TOLERANCE = 1e-7
RESTORE_STEPS = 5


class ExactRestoreError(RuntimeError):
    """Raised when clean restore cannot be trusted."""


class RestoreEnv(Protocol):
    def step(self, action: Sequence[float]) -> tuple[Any, float, bool, Mapping[str, Any]]:
        ...


class Policy(Protocol):
    def act(self, obs: Any) -> tuple[Sequence[float], Sequence[int]]:
        ...


@dataclass(frozen=True)
class Layer3ParentDependencyManifest:
    """Dependency manifest required before a parent may enter restore smoke."""

    suite: str
    task_idx: int
    state_id: int
    eval_seed: int
    parent_key: str
    openvla_model_sha256: str
    unnorm_key: str
    layer2_dataset_sha256: str
    detector_checkpoint_sha256: str
    tau_corridor: float
    tau_release: float
    libero_version: str
    mujoco_version: str
    task_instruction_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "openvla_model_sha256",
            "layer2_dataset_sha256",
            "detector_checkpoint_sha256",
            "task_instruction_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if not self.unnorm_key:
            raise ExactRestoreError("unnorm_key is required")
        if not self.parent_key:
            raise ExactRestoreError("parent_key is required")

    @property
    def manifest_sha256(self) -> str:
        return sha256_jsonable(asdict(self))


@dataclass
class StepObservation:
    step: int
    observation_sha256: str
    proprio_sha256: str
    action_sha256: str
    token_sha256: str
    reward: float
    done: bool
    success: bool
    qpos_sha256: str
    qvel_sha256: str
    eef_pose_sha256: str
    gripper_width: float
    detector_state_sha256: str
    feature_history_sha256: str


def _json_clone(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def hash_array(value: Any) -> str:
    arr = np.asarray(value)
    payload = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "bytes": arr.tobytes().hex(),
    }
    return sha256_jsonable(payload)


def hash_jsonable(value: Any) -> str:
    return sha256_jsonable(_json_clone(value))


def capture_python_rng_state() -> dict[str, Any]:
    return {"python_random": repr(random.getstate())}


def capture_numpy_rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def capture_torch_rng_state() -> dict[str, Any]:
    if torch is None:
        return {"torch_available": False}
    out: dict[str, Any] = {
        "torch_available": True,
        "cpu_rng_sha256": hash_array(torch.get_rng_state().cpu().numpy()),
    }
    if torch.cuda.is_available():
        out["cuda_rng_sha256"] = [hash_array(x.cpu().numpy()) for x in torch.cuda.get_rng_state_all()]
    else:
        out["cuda_rng_sha256"] = []
    return out


def capture_policy_rng_state(policy: Any | None = None) -> dict[str, Any]:
    state = {
        "python": capture_python_rng_state(),
        "numpy": capture_numpy_rng_state(),
        "torch": capture_torch_rng_state(),
    }
    if policy is not None and hasattr(policy, "rng_state"):
        state["policy"] = _json_clone(policy.rng_state())
    return state


def capture_mujoco_state(env: Any) -> dict[str, Any]:
    sim = env.sim
    data = sim.data
    state = {
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "time": float(getattr(data, "time", 0.0)),
    }
    for name in ("act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
        if hasattr(data, name):
            state[name] = np.asarray(getattr(data, name)).copy()
    return state


def restore_mujoco_state(env: Any, state: Mapping[str, Any]) -> None:
    data = env.sim.data
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
        if name in state and hasattr(data, name):
            target = getattr(data, name)
            target[...] = np.asarray(state[name], dtype=target.dtype)
    if "time" in state and hasattr(data, "time"):
        data.time = float(state["time"])
    if hasattr(env.sim, "forward"):
        env.sim.forward()


def capture_env_internal_state(env: Any) -> dict[str, Any]:
    if hasattr(env, "get_internal_state"):
        return _json_clone(env.get_internal_state())
    return {}


def restore_env_internal_state(env: Any, state: Mapping[str, Any]) -> None:
    if hasattr(env, "set_internal_state"):
        env.set_internal_state(copy.deepcopy(dict(state)))


def capture_student_state(student: Any) -> dict[str, Any]:
    if hasattr(student, "snapshot_state"):
        return _json_clone(student.snapshot_state())
    return _json_clone(getattr(student, "__dict__", {}))


def restore_student_state(student: Any, state: Mapping[str, Any]) -> None:
    if hasattr(student, "restore_state"):
        student.restore_state(copy.deepcopy(dict(state)))
    else:
        student.__dict__.clear()
        student.__dict__.update(copy.deepcopy(dict(state)))


@dataclass
class ExactRestoreSnapshotPayload:
    prefix: PrefixBranchSnapshot
    parent_manifest: Layer3ParentDependencyManifest
    mujoco_state: dict[str, Any]
    env_internal_state: dict[str, Any]
    policy_rng_state: dict[str, Any]
    student_state: dict[str, Any]
    feature_history: Any
    observation: Any
    clean_action_t: Sequence[float]
    clean_tokens_t: Sequence[int]

    @property
    def payload_sha256(self) -> str:
        return sha256_jsonable(
            {
                "prefix": asdict(self.prefix),
                "parent_manifest_sha256": self.parent_manifest.manifest_sha256,
                "mujoco_state_sha256": hash_jsonable(self.mujoco_state),
                "env_internal_state_sha256": hash_jsonable(self.env_internal_state),
                "policy_rng_state_sha256": hash_jsonable(self.policy_rng_state),
                "student_state_sha256": hash_jsonable(self.student_state),
                "feature_history_sha256": hash_jsonable(self.feature_history),
                "observation_sha256": hash_jsonable(self.observation),
                "clean_action_t": list(self.clean_action_t),
                "clean_tokens_t": list(self.clean_tokens_t),
            }
        )


def build_prefix_snapshot(
    *,
    parent: Layer3ParentDependencyManifest,
    emit_step: int,
    observation: Any,
    mujoco_state: Mapping[str, Any],
    policy_rng_state: Mapping[str, Any],
    student_state: Mapping[str, Any],
    feature_history: Any,
    source_episode_relpath: str,
) -> PrefixBranchSnapshot:
    return PrefixBranchSnapshot(
        suite=parent.suite,
        task_idx=parent.task_idx,
        state_id=parent.state_id,
        eval_seed=parent.eval_seed,
        emit_step=int(emit_step),
        observation_sha256=hash_jsonable(observation),
        sim_state_sha256=hash_jsonable(mujoco_state),
        policy_rng_sha256=hash_jsonable(policy_rng_state),
        detector_state_sha256=hash_jsonable(student_state),
        feature_history_sha256=hash_jsonable(feature_history),
        source_episode_relpath=source_episode_relpath,
    )


def restore_snapshot(env: Any, student: Any, snapshot: ExactRestoreSnapshotPayload) -> None:
    restore_mujoco_state(env, snapshot.mujoco_state)
    restore_env_internal_state(env, snapshot.env_internal_state)
    restore_student_state(student, snapshot.student_state)


def observe_step(
    *,
    step: int,
    obs: Any,
    action: Sequence[float],
    tokens: Sequence[int],
    reward: float,
    done: bool,
    success: bool,
    env: Any,
    student: Any,
    feature_history: Any,
) -> StepObservation:
    return StepObservation(
        step=int(step),
        observation_sha256=hash_jsonable(obs),
        proprio_sha256=hash_jsonable(obs.get("proprio", {}) if isinstance(obs, Mapping) else {}),
        action_sha256=hash_jsonable(list(action)),
        token_sha256=hash_jsonable(list(tokens)),
        reward=float(reward),
        done=bool(done),
        success=bool(success),
        qpos_sha256=hash_array(getattr(env.sim.data, "qpos", [])),
        qvel_sha256=hash_array(getattr(env.sim.data, "qvel", [])),
        eef_pose_sha256=hash_jsonable(obs.get("eef_pose", []) if isinstance(obs, Mapping) else []),
        gripper_width=float(obs.get("gripper_width", 0.0) if isinstance(obs, Mapping) else 0.0),
        detector_state_sha256=hash_jsonable(capture_student_state(student)),
        feature_history_sha256=hash_jsonable(feature_history),
    )


def compare_step_sequences(
    reference: Sequence[StepObservation],
    replay: Sequence[StepObservation],
    *,
    float_tolerance: float = FLOAT_TOLERANCE,
) -> list[str]:
    problems: list[str] = []
    if len(reference) != len(replay):
        problems.append(f"length_mismatch:{len(reference)}!={len(replay)}")
        return problems
    exact_fields = [
        "observation_sha256",
        "proprio_sha256",
        "action_sha256",
        "token_sha256",
        "done",
        "success",
        "qpos_sha256",
        "qvel_sha256",
        "eef_pose_sha256",
        "detector_state_sha256",
        "feature_history_sha256",
    ]
    for idx, (a, b) in enumerate(zip(reference, replay)):
        for field in exact_fields:
            if getattr(a, field) != getattr(b, field):
                problems.append(f"step{idx}:{field}_mismatch")
        if abs(a.reward - b.reward) > float_tolerance:
            problems.append(f"step{idx}:reward_mismatch")
        if abs(a.gripper_width - b.gripper_width) > float_tolerance:
            problems.append(f"step{idx}:gripper_width_mismatch")
    return problems


def rollout_clean_steps(
    *,
    env: RestoreEnv,
    student: Any,
    policy: Policy,
    initial_obs: Any,
    start_step: int,
    count: int = RESTORE_STEPS,
) -> list[StepObservation]:
    obs = copy.deepcopy(initial_obs)
    rows: list[StepObservation] = []
    feature_history = getattr(student, "feature_history", [])
    for offset in range(count):
        action, tokens = policy.act(obs)
        next_obs, reward, done, info = env.step(action)
        success = bool(info.get("success", False)) if isinstance(info, Mapping) else False
        rows.append(
            observe_step(
                step=start_step + offset,
                obs=next_obs,
                action=action,
                tokens=tokens,
                reward=float(reward),
                done=bool(done),
                success=success,
                env=env,
                student=student,
                feature_history=feature_history,
            )
        )
        obs = copy.deepcopy(next_obs)
        if done:
            break
    return rows


def validate_clean_restore_pair(
    *,
    snapshot: ExactRestoreSnapshotPayload,
    branch_records: Sequence[BranchRunRecord | Mapping[str, Any]],
    reference: Sequence[StepObservation],
    replay_a: Sequence[StepObservation],
    replay_b: Sequence[StepObservation],
) -> dict[str, Any]:
    branch_summary = validate_branch_records(snapshot.prefix, branch_records, required_conditions=("CLEAN_REPLAY",))
    ref_a = compare_step_sequences(reference, replay_a)
    a_b = compare_step_sequences(replay_a, replay_b)
    problems = [f"reference_vs_replay:{p}" for p in ref_a] + [f"replay_a_vs_replay_b:{p}" for p in a_b]
    if problems:
        raise ExactRestoreError(";".join(problems))
    return {
        "clean_restore_pass": True,
        "restore_steps": len(reference),
        "prefix_snapshot_sha256": snapshot.prefix.snapshot_sha256,
        "snapshot_payload_sha256": snapshot.payload_sha256,
        "branch_summary": branch_summary,
        "reference_vs_replay_mismatch_count": 0,
        "replay_a_vs_replay_b_mismatch_count": 0,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mock", action="store_true", help="Run the CPU mock restore smoke; no LIBERO/OpenVLA/GPU.")
    return ap.parse_args()


def run_mock(output_dir: Path) -> None:
    case = build_mock_restore_case()
    result = validate_clean_restore_pair(
        snapshot=case["snapshot"],
        branch_records=case["branch_records"],
        reference=case["reference"],
        replay_a=case["replay_a"],
        replay_b=case["replay_b"],
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "mock_restore_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


class _MockData:
    def __init__(self) -> None:
        self.qpos = np.zeros(2, dtype=np.float64)
        self.qvel = np.zeros(2, dtype=np.float64)
        self.ctrl = np.zeros(1, dtype=np.float64)
        self.act = np.zeros(1, dtype=np.float64)
        self.mocap_pos = np.zeros((1, 3), dtype=np.float64)
        self.mocap_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.userdata = np.zeros(1, dtype=np.float64)
        self.time = 0.0


class _MockSim:
    def __init__(self) -> None:
        self.data = _MockData()

    def forward(self) -> None:
        return None


class _MockEnv:
    def __init__(self) -> None:
        self.sim = _MockSim()
        self.internal = {"step": 0, "success": False}

    def get_internal_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.internal)

    def set_internal_state(self, state: Mapping[str, Any]) -> None:
        self.internal = copy.deepcopy(dict(state))

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.internal["step"] += 1
        self.sim.data.time += 1.0
        self.sim.data.qpos[:] = np.asarray(action[:2], dtype=np.float64) + self.internal["step"]
        self.sim.data.qvel[:] = np.asarray(action[2:4], dtype=np.float64)
        obs = {
            "rgb": [self.internal["step"], int(round(float(action[-1]) * 10))],
            "proprio": {"qpos": self.sim.data.qpos.tolist()},
            "eef_pose": [float(self.internal["step"]), 0.0, 0.0],
            "gripper_width": float(abs(action[-1])),
        }
        done = self.internal["step"] >= 99
        return obs, float(self.internal["step"]), done, {"success": False}


class _MockPolicy:
    def act(self, obs: Any) -> tuple[Sequence[float], Sequence[int]]:
        step = int(obs.get("rgb", [0])[0]) if isinstance(obs, Mapping) else 0
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0 if step % 2 == 0 else 1.0]
        tokens = [31000 + step, 31001, 31002, 31003, 31004, 31005, 31872 if action[-1] > 0 else 31744]
        return action, tokens


class _MockStudent:
    def __init__(self) -> None:
        self.state = "EMITTED"
        self.armed_step = 58
        self.feature_history = [{"step": 56}, {"step": 57}, {"step": 58}]

    def snapshot_state(self) -> dict[str, Any]:
        return {"state": self.state, "armed_step": self.armed_step}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.state = str(state["state"])
        self.armed_step = int(state["armed_step"])


def build_mock_restore_case() -> dict[str, Any]:
    parent = Layer3ParentDependencyManifest(
        suite="libero_spatial",
        task_idx=0,
        state_id=20,
        eval_seed=0,
        parent_key="libero_spatial|0|20|0|CLEAN",
        openvla_model_sha256="a" * 64,
        unnorm_key="libero_spatial",
        layer2_dataset_sha256="b" * 64,
        detector_checkpoint_sha256="c" * 64,
        tau_corridor=0.3,
        tau_release=0.3,
        libero_version="mock",
        mujoco_version="mock",
        task_instruction_sha256="d" * 64,
    )
    env = _MockEnv()
    student = _MockStudent()
    policy = _MockPolicy()
    obs_t = {"rgb": [58, 0], "proprio": {"qpos": [0.0, 0.0]}, "eef_pose": [0, 0, 0], "gripper_width": 1.0}
    mujoco_state = capture_mujoco_state(env)
    env_state = capture_env_internal_state(env)
    policy_rng = {"mock_policy_rng": "stable"}
    student_state = capture_student_state(student)
    feature_history = copy.deepcopy(student.feature_history)
    prefix = build_prefix_snapshot(
        parent=parent,
        emit_step=58,
        observation=obs_t,
        mujoco_state=mujoco_state,
        policy_rng_state=policy_rng,
        student_state=student_state,
        feature_history=feature_history,
        source_episode_relpath="mock",
    )
    snapshot = ExactRestoreSnapshotPayload(
        prefix=prefix,
        parent_manifest=parent,
        mujoco_state=mujoco_state,
        env_internal_state=env_state,
        policy_rng_state=policy_rng,
        student_state=student_state,
        feature_history=feature_history,
        observation=obs_t,
        clean_action_t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
        clean_tokens_t=[1, 2, 3, 4, 5, 6, 31744],
    )
    reference_env = _MockEnv()
    restore_snapshot(reference_env, student, snapshot)
    reference = rollout_clean_steps(env=reference_env, student=student, policy=policy, initial_obs=obs_t, start_step=58)
    replay_env_a = _MockEnv()
    restore_snapshot(replay_env_a, student, snapshot)
    replay_a = rollout_clean_steps(env=replay_env_a, student=student, policy=policy, initial_obs=obs_t, start_step=58)
    replay_env_b = _MockEnv()
    restore_snapshot(replay_env_b, student, snapshot)
    replay_b = rollout_clean_steps(env=replay_env_b, student=student, policy=policy, initial_obs=obs_t, start_step=58)
    branch = BranchRunRecord(
        condition="CLEAN_REPLAY",
        prefix_snapshot_sha256=prefix.snapshot_sha256,
        branch_source="EXACT_PREFIX_RESTORE",
        restored_sim_state_sha256=prefix.sim_state_sha256,
        restored_observation_sha256=prefix.observation_sha256,
        restored_policy_rng_sha256=prefix.policy_rng_sha256,
        restored_detector_state_sha256=prefix.detector_state_sha256,
        restored_feature_history_sha256=prefix.feature_history_sha256,
        trigger_step=58,
        first_env_step=58,
    )
    return {"snapshot": snapshot, "branch_records": [branch], "reference": reference, "replay_a": replay_a, "replay_b": replay_b}


def main() -> None:
    args = parse_args()
    if not args.mock:
        raise SystemExit("Only --mock is implemented in this CPU PR. GPU/LIBERO restore smoke remains a later gate.")
    run_mock(Path(args.output_dir))


if __name__ == "__main__":
    main()
