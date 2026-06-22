#!/usr/bin/env python3
"""Clean-only exact-restore runner scaffolding for Layer3.

This file is deliberately attack-free.  It provides the state capture/restore
and five-step replay comparison primitives that the later GPU runner will use
before VIS/RAND are allowed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
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
SUPPORTED_SUITES = {"libero_spatial", "libero_goal", "libero_10"}
EXPECTED_LAYER2_DATASET_SHA256 = "6252fd699010005e48f5dff24c631262fb4939d9b76314a9afb82efe7f2cd0b2"
EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE = {
    "libero_spatial": "d229e3db0a3b15cf68712a4582817c30a1bedd9f424b1ea7c68120f00e61134a",
    "libero_goal": "3826a64530d25078441c214e2667d25b32eee98a05581da943bb978ff6bfee98",
    "libero_10": "9f6759b916b0ab612a1b3ebcef4186677197a27c73e55ee4c1653d7828c30df9",
}
MUJOCO_STATE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "time",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
)


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
        if self.suite not in SUPPORTED_SUITES:
            raise ExactRestoreError(f"unsupported suite: {self.suite}")
        for field in (
            "openvla_model_sha256",
            "layer2_dataset_sha256",
            "detector_checkpoint_sha256",
            "task_instruction_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if self.layer2_dataset_sha256 != EXPECTED_LAYER2_DATASET_SHA256:
            raise ExactRestoreError("layer2_dataset_sha256 does not match frozen v3 dataset")
        expected_ckpt = EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE[self.suite]
        if self.detector_checkpoint_sha256 != expected_ckpt:
            raise ExactRestoreError(
                f"detector_checkpoint_sha256 does not match frozen {self.suite} M2 checkpoint"
            )
        if not self.unnorm_key:
            raise ExactRestoreError("unnorm_key is required")
        if self.unnorm_key != self.suite:
            raise ExactRestoreError(f"unnorm_key {self.unnorm_key} does not match suite {self.suite}")
        if not self.parent_key:
            raise ExactRestoreError("parent_key is required")
        expected_prefix = f"{self.suite}|{self.task_idx}|{self.state_id}|{self.eval_seed}|"
        if not self.parent_key.startswith(expected_prefix):
            raise ExactRestoreError(f"parent_key does not match suite/task/state/eval_seed: {self.parent_key}")
        if not (math.isfinite(float(self.tau_corridor)) and 0.0 <= float(self.tau_corridor) <= 1.0):
            raise ExactRestoreError("tau_corridor must be finite and in [0,1]")
        if not (math.isfinite(float(self.tau_release)) and 0.0 <= float(self.tau_release) <= 1.0):
            raise ExactRestoreError("tau_release must be finite and in [0,1]")
        if not self.libero_version:
            raise ExactRestoreError("libero_version is required")
        if not self.mujoco_version:
            raise ExactRestoreError("mujoco_version is required")

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
    qpos_values: list[float]
    qvel_values: list[float]
    eef_pose_values: list[float]
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or directory dependency in a deterministic manifest form."""

    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ExactRestoreError(f"dependency path does not exist: {path}")
    rows: list[dict[str, str | int]] = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append(
            {
                "relpath": child.relative_to(path).as_posix(),
                "size": child.stat().st_size,
                "sha256": sha256_file(child),
            }
        )
    if not rows:
        raise ExactRestoreError(f"dependency directory is empty: {path}")
    return hash_jsonable(rows)


def validate_dependency_sha_values(
    parent: Layer3ParentDependencyManifest,
    *,
    actual_openvla_model_sha256: str,
    actual_detector_checkpoint_sha256: str,
) -> dict[str, Any]:
    require_sha256(actual_openvla_model_sha256, field="actual_openvla_model_sha256")
    require_sha256(actual_detector_checkpoint_sha256, field="actual_detector_checkpoint_sha256")
    if actual_openvla_model_sha256 != parent.openvla_model_sha256:
        raise ExactRestoreError("actual OpenVLA model SHA does not match parent manifest")
    if actual_detector_checkpoint_sha256 != parent.detector_checkpoint_sha256:
        raise ExactRestoreError("actual detector checkpoint SHA does not match parent manifest")
    return {
        "openvla_model_sha256": actual_openvla_model_sha256,
        "detector_checkpoint_sha256": actual_detector_checkpoint_sha256,
        "dependency_sha_validation_pass": True,
    }


def validate_dependency_files(
    parent: Layer3ParentDependencyManifest,
    *,
    openvla_model_path: str | Path,
    detector_checkpoint_path: str | Path,
) -> dict[str, Any]:
    model_path = Path(openvla_model_path)
    ckpt_path = Path(detector_checkpoint_path)
    return validate_dependency_sha_values(
        parent,
        actual_openvla_model_sha256=sha256_path(model_path),
        actual_detector_checkpoint_sha256=sha256_path(ckpt_path),
    )


def capture_python_rng_state() -> dict[str, Any]:
    return {"python_random_state": _json_clone(random.getstate())}


def _to_tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple_tree(v) for v in value)
    return value


def restore_python_rng_state(state: Mapping[str, Any]) -> None:
    if "python_random_state" not in state:
        raise ExactRestoreError("python RNG state missing")
    random.setstate(_to_tuple_tree(state["python_random_state"]))


def capture_numpy_rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def restore_numpy_rng_state(state: Mapping[str, Any]) -> None:
    required = {"bit_generator", "keys", "pos", "has_gauss", "cached_gaussian"}
    missing = required - set(state)
    if missing:
        raise ExactRestoreError(f"numpy RNG state missing: {','.join(sorted(missing))}")
    np.random.set_state(
        (
            str(state["bit_generator"]),
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["pos"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


def capture_torch_rng_state() -> dict[str, Any]:
    if torch is None:
        return {"torch_available": False}
    out: dict[str, Any] = {
        "torch_available": True,
        "cpu_rng_state": torch.get_rng_state().cpu().numpy().astype(np.uint8).tolist(),
    }
    if torch.cuda.is_available():
        out["cuda_rng_state"] = [x.cpu().numpy().astype(np.uint8).tolist() for x in torch.cuda.get_rng_state_all()]
    else:
        out["cuda_rng_state"] = []
    out["cpu_rng_sha256"] = hash_array(np.asarray(out["cpu_rng_state"], dtype=np.uint8))
    out["cuda_rng_sha256"] = [hash_array(np.asarray(x, dtype=np.uint8)) for x in out["cuda_rng_state"]]
    return out


def restore_torch_rng_state(state: Mapping[str, Any]) -> None:
    if torch is None:
        if state.get("torch_available"):
            raise ExactRestoreError("torch RNG state present but torch is unavailable")
        return
    if not state.get("torch_available", False):
        return
    if "cpu_rng_state" not in state:
        raise ExactRestoreError("torch CPU RNG state missing")
    torch.set_rng_state(torch.tensor(state["cpu_rng_state"], dtype=torch.uint8))
    cuda_states = list(state.get("cuda_rng_state", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise ExactRestoreError("torch CUDA RNG state present but CUDA is unavailable")
        torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in cuda_states])


def capture_policy_rng_state(policy: Any | None = None) -> dict[str, Any]:
    state = {
        "python": capture_python_rng_state(),
        "numpy": capture_numpy_rng_state(),
        "torch": capture_torch_rng_state(),
    }
    if policy is not None and hasattr(policy, "rng_state"):
        state["policy"] = _json_clone(policy.rng_state())
    return state


def restore_policy_rng_state(policy: Any | None, state: Mapping[str, Any], *, strict: bool = True) -> None:
    restore_python_rng_state(state.get("python", {}))
    restore_numpy_rng_state(state.get("numpy", {}))
    restore_torch_rng_state(state.get("torch", {}))
    if "policy" in state:
        if policy is None or not hasattr(policy, "set_rng_state"):
            raise ExactRestoreError("policy RNG state present but policy.set_rng_state is unavailable")
        policy.set_rng_state(copy.deepcopy(state["policy"]))
    elif strict and policy is not None and hasattr(policy, "rng_state"):
        raise ExactRestoreError("policy exposes rng_state but saved policy RNG state is missing")


def capture_mujoco_state(env: Any) -> dict[str, Any]:
    sim = env.sim
    data = sim.data
    state: dict[str, Any] = {}
    for name in MUJOCO_STATE_FIELDS:
        if hasattr(data, name):
            value = getattr(data, name)
            if name == "time":
                state[name] = float(value)
            else:
                state[name] = np.asarray(value).copy()
    for required in ("qpos", "qvel", "time"):
        if required not in state:
            raise ExactRestoreError(f"MuJoCo state missing required field: {required}")
    return state


def restore_mujoco_state(env: Any, state: Mapping[str, Any]) -> None:
    data = env.sim.data
    for name in MUJOCO_STATE_FIELDS:
        if name in state and hasattr(data, name):
            if name == "time":
                data.time = float(state["time"])
                continue
            target = getattr(data, name)
            target[...] = np.asarray(state[name], dtype=target.dtype)
    if hasattr(env.sim, "forward"):
        env.sim.forward()


def capture_env_internal_state(env: Any, *, strict: bool = True) -> dict[str, Any]:
    if hasattr(env, "get_internal_state"):
        return _json_clone(env.get_internal_state())
    if strict:
        raise ExactRestoreError("env adapter missing get_internal_state")
    return {}


def restore_env_internal_state(env: Any, state: Mapping[str, Any], *, strict: bool = True) -> None:
    if hasattr(env, "set_internal_state"):
        env.set_internal_state(copy.deepcopy(dict(state)))
        return
    if strict:
        raise ExactRestoreError("env adapter missing set_internal_state")


def capture_student_state(student: Any, *, strict: bool = True) -> dict[str, Any]:
    if hasattr(student, "snapshot_state"):
        return _json_clone(student.snapshot_state())
    if strict:
        raise ExactRestoreError("student adapter missing snapshot_state")
    return _json_clone(getattr(student, "__dict__", {}))


def restore_student_state(student: Any, state: Mapping[str, Any], *, strict: bool = True) -> None:
    if hasattr(student, "restore_state"):
        student.restore_state(copy.deepcopy(dict(state)))
        return
    if strict:
        raise ExactRestoreError("student adapter missing restore_state")
    student.__dict__.clear()
    student.__dict__.update(copy.deepcopy(dict(state)))


def capture_feature_history(student: Any, *, strict: bool = True) -> Any:
    if hasattr(student, "snapshot_feature_history"):
        return _json_clone(student.snapshot_feature_history())
    if strict:
        raise ExactRestoreError("student adapter missing snapshot_feature_history")
    return _json_clone(getattr(student, "feature_history", []))


def restore_feature_history(student: Any, history: Any, *, strict: bool = True) -> None:
    if hasattr(student, "restore_feature_history"):
        student.restore_feature_history(copy.deepcopy(history))
        return
    if strict:
        raise ExactRestoreError("student adapter missing restore_feature_history")
    student.feature_history = copy.deepcopy(history)


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

    def __post_init__(self) -> None:
        if self.prefix.snapshot_boundary != SNAPSHOT_BOUNDARY:
            raise ExactRestoreError("snapshot boundary mismatch")
        if self.prefix.sim_state_sha256 != hash_jsonable(self.mujoco_state):
            raise ExactRestoreError("payload mujoco_state hash does not match prefix")
        if self.prefix.policy_rng_sha256 != hash_jsonable(self.policy_rng_state):
            raise ExactRestoreError("payload policy_rng_state hash does not match prefix")
        if self.prefix.detector_state_sha256 != hash_jsonable(self.student_state):
            raise ExactRestoreError("payload student_state hash does not match prefix")
        if self.prefix.feature_history_sha256 != hash_jsonable(self.feature_history):
            raise ExactRestoreError("payload feature_history hash does not match prefix")
        if self.prefix.observation_sha256 != hash_jsonable(self.observation):
            raise ExactRestoreError("payload observation hash does not match prefix")
        if len(list(self.clean_action_t)) != 7 or not all(math.isfinite(float(x)) for x in self.clean_action_t):
            raise ExactRestoreError("clean_action_t must be exact finite 7D")
        if len(list(self.clean_tokens_t)) != 7:
            raise ExactRestoreError("clean_tokens_t must contain exactly 7 tokens")

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


@dataclass(frozen=True)
class Layer3RuntimeReceipt:
    cuda_visible_devices: str
    ordered_gpu_uuids: Sequence[str]
    device_count: int
    torch_version: str
    cuda_runtime: str
    driver_version: str
    libero_version: str
    mujoco_version: str
    openvla_generation_kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        if int(self.device_count) != len(list(self.ordered_gpu_uuids)):
            raise ExactRestoreError("runtime receipt device_count does not match ordered_gpu_uuids")
        if int(self.device_count) > 0 and not self.cuda_visible_devices:
            raise ExactRestoreError("runtime receipt missing CUDA_VISIBLE_DEVICES for CUDA run")
        for field in ("torch_version", "cuda_runtime", "driver_version", "libero_version", "mujoco_version"):
            if not str(getattr(self, field)):
                raise ExactRestoreError(f"runtime receipt missing {field}")
        if bool(self.openvla_generation_kwargs.get("do_sample", False)):
            raise ExactRestoreError("runtime receipt requires deterministic generation: do_sample must be false")
        temperature = self.openvla_generation_kwargs.get("temperature", 0.0)
        if temperature not in (None, 0, 0.0):
            raise ExactRestoreError("runtime receipt requires deterministic generation temperature")

    @property
    def receipt_sha256(self) -> str:
        return sha256_jsonable(asdict(self))


def capture_runtime_receipt(
    *,
    libero_version: str,
    mujoco_version: str,
    openvla_generation_kwargs: Mapping[str, Any],
    ordered_gpu_uuids: Sequence[str] | None = None,
    driver_version: str | None = None,
) -> Layer3RuntimeReceipt:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch is not None and torch.cuda.is_available():
        device_count = int(torch.cuda.device_count())
        cuda_runtime = str(torch.version.cuda or "")
    else:
        device_count = 0
        cuda_runtime = "cpu"
    uuids = list(ordered_gpu_uuids or [])
    if not uuids and device_count > 0:
        uuids = query_ordered_gpu_uuids()
    return Layer3RuntimeReceipt(
        cuda_visible_devices=cuda_visible,
        ordered_gpu_uuids=uuids,
        device_count=device_count,
        torch_version=str(getattr(torch, "__version__", "unavailable")) if torch is not None else "unavailable",
        cuda_runtime=cuda_runtime,
        driver_version=driver_version or query_nvidia_driver_version() or "cpu",
        libero_version=libero_version,
        mujoco_version=mujoco_version,
        openvla_generation_kwargs=_json_clone(dict(openvla_generation_kwargs)),
    )


def query_ordered_gpu_uuids() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def query_nvidia_driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return ""
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if versions else ""


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


def restore_snapshot(env: Any, student: Any, snapshot: ExactRestoreSnapshotPayload, policy: Any | None = None) -> None:
    restore_mujoco_state(env, snapshot.mujoco_state)
    restore_env_internal_state(env, snapshot.env_internal_state, strict=True)
    restore_student_state(student, snapshot.student_state, strict=True)
    restore_feature_history(student, snapshot.feature_history, strict=True)
    restore_policy_rng_state(policy, snapshot.policy_rng_state, strict=True)


def get_observation_after_restore(env: Any, snapshot: ExactRestoreSnapshotPayload) -> Any:
    """Rebuild obs_t from the restored env, then verify it matches the prefix.

    Real adapters must not feed the saved obs_t back into policy execution. They
    must reconstruct/render the observation from restored simulator/env state.
    """

    if hasattr(env, "get_observation_after_restore"):
        obs = env.get_observation_after_restore()
    elif hasattr(env, "get_observation"):
        obs = env.get_observation()
    else:
        raise ExactRestoreError("env adapter missing get_observation_after_restore")
    obs_hash = hash_jsonable(obs)
    if obs_hash != snapshot.prefix.observation_sha256:
        raise ExactRestoreError("restored observation hash does not match prefix")
    return _json_clone(obs)


def restore_snapshot_and_recapture_observation(
    env: Any,
    student: Any,
    snapshot: ExactRestoreSnapshotPayload,
    policy: Any | None = None,
) -> Any:
    restore_snapshot(env, student, snapshot, policy)
    return get_observation_after_restore(env, snapshot)


def recapture_branch_record(
    *,
    condition: str,
    snapshot: ExactRestoreSnapshotPayload,
    env: Any,
    student: Any,
    policy: Any | None,
) -> BranchRunRecord:
    """Construct a BranchRunRecord from actual post-restore state."""

    actual_mujoco = capture_mujoco_state(env)
    actual_policy_rng = capture_policy_rng_state(policy)
    actual_student = capture_student_state(student, strict=True)
    actual_history = capture_feature_history(student, strict=True)
    actual_observation = get_observation_after_restore(env, snapshot)
    return BranchRunRecord(
        condition=condition,
        prefix_snapshot_sha256=snapshot.prefix.snapshot_sha256,
        branch_source="EXACT_PREFIX_RESTORE",
        restored_sim_state_sha256=hash_jsonable(actual_mujoco),
        restored_observation_sha256=hash_jsonable(actual_observation),
        restored_policy_rng_sha256=hash_jsonable(actual_policy_rng),
        restored_detector_state_sha256=hash_jsonable(actual_student),
        restored_feature_history_sha256=hash_jsonable(actual_history),
        trigger_step=snapshot.prefix.emit_step,
        first_env_step=snapshot.prefix.emit_step,
    )


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
        qpos_values=[float(x) for x in np.asarray(getattr(env.sim.data, "qpos", [])).reshape(-1).tolist()],
        qvel_values=[float(x) for x in np.asarray(getattr(env.sim.data, "qvel", [])).reshape(-1).tolist()],
        eef_pose_values=[
            float(x)
            for x in np.asarray(obs.get("eef_pose", []) if isinstance(obs, Mapping) else []).reshape(-1).tolist()
        ],
        gripper_width=float(obs.get("gripper_width", 0.0) if isinstance(obs, Mapping) else 0.0),
        detector_state_sha256=hash_jsonable(capture_student_state(student, strict=True)),
        feature_history_sha256=hash_jsonable(feature_history),
    )


def update_student_for_step(student: Any, *, step: int, obs: Any, action: Sequence[float], tokens: Sequence[int]) -> Any:
    """Advance Student/FSM state for one clean step using an explicit adapter hook."""

    if hasattr(student, "update_for_step"):
        return student.update_for_step(step=step, obs=copy.deepcopy(obs), action=list(action), tokens=list(tokens))
    if hasattr(student, "step"):
        return student.step(step=step, obs=copy.deepcopy(obs), action=list(action), tokens=list(tokens))
    if hasattr(student, "update"):
        return student.update(step=step, obs=copy.deepcopy(obs), action=list(action), tokens=list(tokens))
    raise ExactRestoreError("student adapter missing per-step update hook")


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
        "step",
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
        qpos_diff = max_abs_diff(a.qpos_values, b.qpos_values)
        qvel_diff = max_abs_diff(a.qvel_values, b.qvel_values)
        eef_diff = max_abs_diff(a.eef_pose_values, b.eef_pose_values)
        if qpos_diff > float_tolerance:
            problems.append(f"step{idx}:qpos_max_abs_diff={qpos_diff:.9g}")
        if qvel_diff > float_tolerance:
            problems.append(f"step{idx}:qvel_max_abs_diff={qvel_diff:.9g}")
        if eef_diff > float_tolerance:
            problems.append(f"step{idx}:eef_pose_max_abs_diff={eef_diff:.9g}")
        if abs(a.reward - b.reward) > float_tolerance:
            problems.append(f"step{idx}:reward_mismatch")
        if abs(a.gripper_width - b.gripper_width) > float_tolerance:
            problems.append(f"step{idx}:gripper_width_mismatch")
    return problems


def max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def _assert_action_tokens_match_snapshot(
    *,
    action: Sequence[float],
    tokens: Sequence[int],
    expected_action: Sequence[float],
    expected_tokens: Sequence[int],
    float_tolerance: float = FLOAT_TOLERANCE,
) -> None:
    actual_action = [float(x) for x in action]
    expected_action_list = [float(x) for x in expected_action]
    if len(actual_action) != len(expected_action_list):
        raise ExactRestoreError(f"first action length mismatch: {len(actual_action)}!={len(expected_action_list)}")
    for idx, (actual, expected) in enumerate(zip(actual_action, expected_action_list)):
        if abs(actual - expected) > float_tolerance:
            raise ExactRestoreError(f"first action mismatch at dim {idx}: {actual}!={expected}")
    actual_tokens = [int(x) for x in tokens]
    expected_token_list = [int(x) for x in expected_tokens]
    if actual_tokens != expected_token_list:
        raise ExactRestoreError(f"first token mismatch: {actual_tokens}!={expected_token_list}")


def rollout_clean_steps(
    *,
    env: RestoreEnv,
    student: Any,
    policy: Policy,
    initial_obs: Any,
    start_step: int,
    count: int = RESTORE_STEPS,
    expected_first_action: Sequence[float] | None = None,
    expected_first_tokens: Sequence[int] | None = None,
) -> list[StepObservation]:
    obs = copy.deepcopy(initial_obs)
    rows: list[StepObservation] = []
    for offset in range(count):
        action, tokens = policy.act(obs)
        if offset == 0 and expected_first_action is not None and expected_first_tokens is not None:
            _assert_action_tokens_match_snapshot(
                action=action,
                tokens=tokens,
                expected_action=expected_first_action,
                expected_tokens=expected_first_tokens,
            )
        update_student_for_step(student, step=start_step + offset, obs=obs, action=action, tokens=tokens)
        next_obs, reward, done, info = env.step(action)
        success = bool(info.get("success", False)) if isinstance(info, Mapping) else False
        feature_history = capture_feature_history(student, strict=True)
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
            raise ExactRestoreError(f"restore replay ended before {count} steps at offset {offset}")
    if len(rows) != count:
        raise ExactRestoreError(f"restore replay produced {len(rows)} steps, expected {count}")
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
    for name, rows in (("reference", reference), ("replay_a", replay_a), ("replay_b", replay_b)):
        if len(rows) != RESTORE_STEPS:
            raise ExactRestoreError(f"{name} must contain exactly {RESTORE_STEPS} restore steps, got {len(rows)}")
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
        self.qacc_warmstart = np.zeros(2, dtype=np.float64)
        self.qfrc_applied = np.zeros(2, dtype=np.float64)
        self.xfrc_applied = np.zeros((1, 6), dtype=np.float64)
        self.eq_active = np.ones(1, dtype=np.int32)
        self.time = 0.0


class _MockSim:
    def __init__(self) -> None:
        self.data = _MockData()

    def forward(self) -> None:
        return None


class _MockEnv:
    def __init__(self, *, step: int = 0) -> None:
        self.sim = _MockSim()
        self.internal = {"step": int(step), "success": False}
        self.sim.data.time = float(step)

    def get_internal_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.internal)

    def set_internal_state(self, state: Mapping[str, Any]) -> None:
        self.internal = copy.deepcopy(dict(state))

    def get_observation_after_restore(self) -> dict[str, Any]:
        step = int(self.internal["step"])
        return {
            "rgb": [step, 0],
            "proprio": {"qpos": self.sim.data.qpos.tolist()},
            "eef_pose": [0, 0, 0],
            "gripper_width": 1.0,
        }

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
    def __init__(self) -> None:
        self._rng_state = {"counter": 0}

    def rng_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._rng_state)

    def set_rng_state(self, state: Mapping[str, Any]) -> None:
        self._rng_state = copy.deepcopy(dict(state))

    def act(self, obs: Any) -> tuple[Sequence[float], Sequence[int]]:
        step = int(obs.get("rgb", [0])[0]) if isinstance(obs, Mapping) else 0
        action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0 if step % 2 == 0 else 1.0]
        tokens = [31000 + step, 31001, 31002, 31003, 31004, 31005, 31872 if action[-1] > 0 else 31744]
        return action, tokens


class _MockStudent:
    def __init__(self) -> None:
        self.state = "EMITTED"
        self.armed_step = 58
        self.update_count = 0
        self.feature_history = [{"step": 56}, {"step": 57}, {"step": 58}]

    def snapshot_state(self) -> dict[str, Any]:
        return {"state": self.state, "armed_step": self.armed_step, "update_count": self.update_count}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.state = str(state["state"])
        self.armed_step = int(state["armed_step"])
        self.update_count = int(state.get("update_count", 0))

    def snapshot_feature_history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.feature_history)

    def restore_feature_history(self, history: Sequence[Mapping[str, Any]]) -> None:
        self.feature_history = [copy.deepcopy(dict(row)) for row in history]

    def update_for_step(self, *, step: int, obs: Any, action: Sequence[float], tokens: Sequence[int]) -> None:
        self.update_count += 1
        self.feature_history.append(
            {
                "step": int(step),
                "rgb0": int(obs.get("rgb", [0])[0]) if isinstance(obs, Mapping) else -1,
                "gripper_token": int(list(tokens)[-1]),
                "update_count": self.update_count,
            }
        )


def build_mock_restore_case() -> dict[str, Any]:
    parent = Layer3ParentDependencyManifest(
        suite="libero_spatial",
        task_idx=0,
        state_id=20,
        eval_seed=0,
        parent_key="libero_spatial|0|20|0|CLEAN",
        openvla_model_sha256="a" * 64,
        unnorm_key="libero_spatial",
        layer2_dataset_sha256=EXPECTED_LAYER2_DATASET_SHA256,
        detector_checkpoint_sha256=EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE["libero_spatial"],
        tau_corridor=0.3,
        tau_release=0.3,
        libero_version="mock",
        mujoco_version="mock",
        task_instruction_sha256="d" * 64,
    )
    env = _MockEnv(step=58)
    student = _MockStudent()
    policy = _MockPolicy()
    obs_t = env.get_observation_after_restore()
    clean_action_t, clean_tokens_t = policy.act(obs_t)
    mujoco_state = capture_mujoco_state(env)
    env_state = capture_env_internal_state(env)
    policy_rng = capture_policy_rng_state(policy)
    student_state = capture_student_state(student)
    feature_history = capture_feature_history(student)
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
        clean_action_t=clean_action_t,
        clean_tokens_t=clean_tokens_t,
    )
    reference_obs_t = get_observation_after_restore(env, snapshot)
    reference = rollout_clean_steps(
        env=env,
        student=student,
        policy=policy,
        initial_obs=reference_obs_t,
        start_step=58,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
    )
    replay_student_a = _MockStudent()
    replay_env_a = _MockEnv()
    replay_obs_a = restore_snapshot_and_recapture_observation(replay_env_a, replay_student_a, snapshot, policy)
    branch = recapture_branch_record(
        condition="CLEAN_REPLAY",
        snapshot=snapshot,
        env=replay_env_a,
        student=replay_student_a,
        policy=policy,
    )
    replay_a = rollout_clean_steps(
        env=replay_env_a,
        student=replay_student_a,
        policy=policy,
        initial_obs=replay_obs_a,
        start_step=58,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
    )
    replay_student_b = _MockStudent()
    replay_env_b = _MockEnv()
    replay_obs_b = restore_snapshot_and_recapture_observation(replay_env_b, replay_student_b, snapshot, policy)
    replay_b = rollout_clean_steps(
        env=replay_env_b,
        student=replay_student_b,
        policy=policy,
        initial_obs=replay_obs_b,
        start_step=58,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
    )
    return {"snapshot": snapshot, "branch_records": [branch], "reference": reference, "replay_a": replay_a, "replay_b": replay_b}


def main() -> None:
    args = parse_args()
    if not args.mock:
        raise SystemExit("Only --mock is implemented in this CPU PR. GPU/LIBERO restore smoke remains a later gate.")
    run_mock(Path(args.output_dir))


if __name__ == "__main__":
    main()
