#!/usr/bin/env python3
"""Clean-only exact-restore runner scaffolding for Layer3.

This file is deliberately attack-free.  It provides the state capture/restore
and five-step replay comparison primitives that the later GPU runner will use
before VIS/RAND are allowed.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

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
EXPECTED_OPENVLA_MODEL_DIR_BY_SUITE = {
    "libero_spatial": "openvla-7b-finetuned-libero-spatial",
    "libero_goal": "openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla-7b-finetuned-libero-10",
}
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def model_norm_stat_keys(model_path: str | Path) -> list[str]:
    stats_path = Path(model_path) / "dataset_statistics.json"
    if not stats_path.is_file():
        raise ExactRestoreError(f"missing dataset_statistics.json in model path: {model_path}")
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExactRestoreError(f"cannot read model dataset_statistics.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExactRestoreError("model dataset_statistics.json is not a mapping")
    return sorted(str(key) for key in payload)


def validate_real_openvla_model_binding(*, suite: str, model_path: str | Path, unnorm_key: str) -> dict[str, Any]:
    if suite not in SUPPORTED_SUITES:
        raise ExactRestoreError(f"unsupported suite: {suite}")
    path = Path(model_path)
    expected_dir = EXPECTED_OPENVLA_MODEL_DIR_BY_SUITE[suite]
    if path.name != expected_dir:
        raise ExactRestoreError(
            f"model_path {path} is not suite-matched for {suite}; expected directory name {expected_dir}"
        )
    keys = model_norm_stat_keys(path)
    if unnorm_key not in keys:
        raise ExactRestoreError(
            f"unnorm_key {unnorm_key} unavailable in {path / 'dataset_statistics.json'}; available={keys}"
        )
    return {
        "suite": suite,
        "model_path": str(path),
        "expected_model_dir": expected_dir,
        "unnorm_key": unnorm_key,
        "available_norm_keys": keys,
    }


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
        uuids = query_ordered_visible_gpu_uuids(cuda_visible)
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


def parse_cuda_visible_devices(cuda_visible_devices: str) -> list[str]:
    """Return CUDA_VISIBLE_DEVICES tokens in exact runtime order."""

    return [token.strip() for token in str(cuda_visible_devices).split(",") if token.strip()]


def query_ordered_visible_gpu_uuids(cuda_visible_devices: str | None = None) -> list[str]:
    """Query UUIDs in CUDA_VISIBLE_DEVICES order.

    Plain `nvidia-smi --query-gpu=uuid` returns all physical GPUs on the
    machine, which does not match `torch.cuda.device_count()` under a restricted
    CUDA_VISIBLE_DEVICES setting. Scientific receipts must record the ordered
    physical devices used by this process.
    """

    tokens = parse_cuda_visible_devices(cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not tokens:
        raise ExactRestoreError("CUDA_VISIBLE_DEVICES is required to derive ordered GPU UUIDs")
    uuids: list[str] = []
    for token in tokens:
        uuids.append(query_gpu_uuid_by_index(token))
    return uuids


def query_gpu_uuid_by_index(index_token: str) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(index_token), "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        raise ExactRestoreError(f"failed to query UUID for CUDA_VISIBLE_DEVICES token {index_token}") from exc
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(uuids) != 1:
        raise ExactRestoreError(f"expected one UUID for CUDA_VISIBLE_DEVICES token {index_token}, got {len(uuids)}")
    return uuids[0]


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
    first_step_student_already_updated: bool = False,
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
        if not (offset == 0 and first_step_student_already_updated):
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


REAL_PREPROCESS_KWARGS = {
    "libero_official_preprocess": False,
    "libero_preprocess_backend": "official_pil_lanczos",
    "center_crop": True,
    "resize_size": 224,
    "drop_attention_mask": True,
}
NUM_STEPS_WAIT = 10


def openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction}?\nOut:"


def model_float_dtype(model: Any) -> Any:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def postprocess_openvla_action_for_libero(action: Sequence[float]) -> np.ndarray:
    env_action = np.asarray(action, dtype=np.float32).copy()
    env_action[..., -1] = 2.0 * env_action[..., -1] - 1.0
    env_action[..., -1] = np.sign(env_action[..., -1])
    env_action[..., -1] = 1.0 if env_action[..., -1] == 0 else env_action[..., -1]
    env_action[..., -1] = -1.0 * env_action[..., -1]
    return np.clip(env_action, -1.0, 1.0).astype(np.float32)


def decode_action_from_token_ids(model: Any, token_ids: Sequence[int], unnorm_key: str) -> np.ndarray:
    token_array = np.asarray([int(x) for x in token_ids], dtype=np.int64)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    discretized = np.clip(vocab_size - token_array - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    norm_actions = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    return np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions).astype(np.float32)


class RealOpenVLAPolicyAdapter:
    def __init__(self, *, model: Any, processor: Any, device: str, instruction: str, unnorm_key: str, action_dim: int):
        self.model = model
        self.processor = processor
        self.device = device
        self.instruction = instruction
        self.unnorm_key = unnorm_key
        self.action_dim = int(action_dim)

    def rng_state(self) -> dict[str, Any]:
        return {"generation_kwargs": {"do_sample": False, "temperature": 0.0}}

    def set_rng_state(self, state: Mapping[str, Any]) -> None:
        kwargs = dict(state.get("generation_kwargs", {}))
        if kwargs.get("do_sample", False):
            raise ExactRestoreError("policy RNG restore rejected nondeterministic generation kwargs")

    def act(self, obs: Any) -> tuple[Sequence[float], Sequence[int]]:
        if not isinstance(obs, Mapping) or "agentview_image" not in obs:
            raise ExactRestoreError("observation missing agentview_image")
        from gripper_attack.openvla_preprocess import prepare_openvla_image
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        import torch

        raw = np.asarray(obs["agentview_image"]).copy()
        proc_image = prepare_openvla_image(
            raw,
            libero_official_preprocess=REAL_PREPROCESS_KWARGS["libero_official_preprocess"],
            center_crop=REAL_PREPROCESS_KWARGS["center_crop"],
            resize_size=REAL_PREPROCESS_KWARGS["resize_size"],
            libero_preprocess_backend=REAL_PREPROCESS_KWARGS["libero_preprocess_backend"],
        )
        inputs = self.processor(openvla_prompt(self.instruction.lower()), proc_image, return_tensors="pt")
        if REAL_PREPROCESS_KWARGS["drop_attention_mask"]:
            inputs.pop("attention_mask", None)
        for key, val in list(inputs.items()):
            if torch.is_tensor(val):
                if torch.is_floating_point(val):
                    inputs[key] = val.to(device=self.device, dtype=model_float_dtype(self.model))
                else:
                    inputs[key] = val.to(device=self.device)
        input_ids = inputs.get("input_ids")
        if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871], device=input_ids.device).long(), dim=0)),
                dim=1,
            )
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.action_dim,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        tokens = extract_exact_new_tokens(gen.sequences, prompt_len=prompt_len, expected_new_tokens=self.action_dim)
        action = decode_action_from_token_ids(self.model, tokens, self.unnorm_key)
        return [float(x) for x in action.tolist()], [int(x) for x in tokens]


class RealLiberoEnvAdapter:
    def __init__(self, env: Any):
        self.env = env
        self.sim = env.sim
        self.frames: list[np.ndarray] = []

    def get_internal_state(self) -> dict[str, Any]:
        inner = getattr(self.env, "env", self.env)
        state: dict[str, Any] = {"sim_flat_state_sha256": hash_array(self.env.get_sim_state())}
        for name in ("timestep", "_timestep", "cur_time", "_elapsed_steps", "done"):
            if hasattr(inner, name):
                value = getattr(inner, name)
                if isinstance(value, (int, float, bool, str)) or value is None:
                    state[name] = value
        return _json_clone(state)

    def set_internal_state(self, state: Mapping[str, Any]) -> None:
        inner = getattr(self.env, "env", self.env)
        for name, value in state.items():
            if name == "sim_flat_state_sha256":
                continue
            if hasattr(inner, name):
                setattr(inner, name, copy.deepcopy(value))

    def get_observation_after_restore(self) -> Any:
        inner = getattr(self.env, "env", self.env)
        self.env.sim.forward()
        if hasattr(self.env, "_post_process"):
            self.env._post_process()
        if hasattr(self.env, "_update_observables"):
            self.env._update_observables(force=True)
        if hasattr(inner, "_get_observations"):
            return inner._get_observations()
        raise ExactRestoreError("LIBERO env inner object missing _get_observations")

    def step(self, action: Sequence[float]) -> tuple[Any, float, bool, Mapping[str, Any]]:
        env_action = postprocess_openvla_action_for_libero(action)
        obs, reward, done, info = self.env.step(env_action)
        if isinstance(obs, Mapping) and "agentview_image" in obs:
            self.frames.append(np.asarray(obs["agentview_image"]).copy())
        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()


class RealSC5StudentAdapter:
    def __init__(self, *, detector: Any, streamer: Any, env_adapter: RealLiberoEnvAdapter):
        self.detector = detector
        self.streamer = streamer
        self.env_adapter = env_adapter
        self.prev_eef: tuple[float, float, float] | None = None
        self.invalid_steps = 0

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "detector_state": self.detector.state,
            "detector_arm_step": int(self.detector.arm_step),
            "detector_emit_step": int(self.detector.emit_step),
            "detector_emitted": bool(self.detector.emitted),
            "prev_eef": list(self.prev_eef) if self.prev_eef is not None else None,
            "invalid_steps": int(self.invalid_steps),
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self.detector.state = str(state["detector_state"])
        self.detector.arm_step = int(state["detector_arm_step"])
        self.detector.emit_step = int(state["detector_emit_step"])
        self.detector.emitted = bool(state["detector_emitted"])
        prev = state.get("prev_eef")
        self.prev_eef = tuple(float(x) for x in prev) if prev is not None else None
        self.invalid_steps = int(state.get("invalid_steps", 0))

    def snapshot_feature_history(self) -> dict[str, Any]:
        return {
            "streamer_history": _json_clone(getattr(self.streamer, "history", [])),
            "next_expected_step": int(getattr(self.streamer, "next_expected_step")),
            "close_streak": int(getattr(self.streamer, "_close_streak", 0)),
            "open_streak": int(getattr(self.streamer, "_open_streak", 0)),
            "flip_count": int(getattr(self.streamer, "_flip_count", 0)),
            "last_close_step": int(getattr(self.streamer, "_last_close_step", -1)),
            "prev_gripper_close": getattr(self.streamer, "_prev_gripper_close", None),
            "onset_detected": bool(getattr(self.streamer, "_onset_detected", False)),
        }

    def restore_feature_history(self, history: Mapping[str, Any]) -> None:
        self.streamer.history = copy.deepcopy(list(history["streamer_history"]))
        self.streamer._next_expected_step = int(history["next_expected_step"])
        self.streamer._close_streak = int(history["close_streak"])
        self.streamer._open_streak = int(history["open_streak"])
        self.streamer._flip_count = int(history["flip_count"])
        self.streamer._last_close_step = int(history["last_close_step"])
        self.streamer._prev_gripper_close = history["prev_gripper_close"]
        self.streamer._onset_detected = bool(history["onset_detected"])

    def update_for_step(self, *, step: int, obs: Any, action: Sequence[float], tokens: Sequence[int]) -> None:
        env_action = postprocess_openvla_action_for_libero(action)
        raw_grip = float(action[-1])
        env_grip = float(env_action[-1])
        qpos = np.asarray(getattr(self.env_adapter.env.sim.data, "qpos", []), dtype=np.float64).reshape(-1)
        qpos_sum = float(qpos[0] + qpos[1]) if qpos.size >= 2 else float("nan")
        opening_proxy = float(abs(qpos[0]) + abs(qpos[1])) if qpos.size >= 2 else float("nan")
        eef_pos = self.env_adapter.env.sim.data.site_xpos[
            self.env_adapter.env.sim.model.site_name2id("gripper0_grip_site")
        ]
        eef = (float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]))
        if self.prev_eef is None:
            eef_v = (0.0, 0.0, 0.0)
        else:
            eef_v = (eef[0] - self.prev_eef[0], eef[1] - self.prev_eef[1], eef[2] - self.prev_eef[2])
        self.prev_eef = eef
        feat_res = self.streamer.update(
            step_id=int(step),
            raw_gripper=raw_grip,
            env_gripper=env_grip,
            gripper_qpos=qpos_sum,
            gripper_opening_proxy=opening_proxy,
            eef_x=eef[0],
            eef_y=eef[1],
            eef_z=eef[2],
            eef_vx=eef_v[0],
            eef_vy=eef_v[1],
            eef_vz=eef_v[2],
            action_dx=float(action[0]),
            action_dy=float(action[1]),
            action_dz=float(action[2]),
            action_gripper=raw_grip,
        )
        if not bool(feat_res.get("valid", False)):
            self.invalid_steps += 1
            return
        self.detector.update(dict(feat_res["features"]), int(step))


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_real_video(path: Path, frames: Sequence[np.ndarray]) -> str:
    if not frames:
        return "NO_FRAMES"
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, list(frames), fps=10)
        return "WROTE"
    except Exception as exc:
        return f"VIDEO_WRITE_FAILED:{type(exc).__name__}:{exc}"


def recursive_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == "recursive_sha256_manifest.csv":
            continue
        rows.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_recursive_manifest(root: Path) -> str:
    rows = recursive_manifest(root)
    path = root / "recursive_sha256_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "size", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return hash_jsonable(rows)


def make_candidate_order(*, suite: str, state_start: int, state_end: int, task_count: int, eval_seed: int) -> list[dict[str, Any]]:
    rows = []
    protocol_id = "REAL_LIBERO_SINGLE_PARENT_CLEAN_RESTORE_R1"
    for task_idx in range(int(task_count)):
        for state_id in range(int(state_start), int(state_end) + 1):
            key = f"{protocol_id}|{suite}|{task_idx}|{state_id}|{eval_seed}"
            rows.append(
                {
                    "protocol_id": protocol_id,
                    "suite": suite,
                    "task_idx": task_idx,
                    "state_id": state_id,
                    "eval_seed": int(eval_seed),
                    "selection_hash": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                }
            )
    return sorted(rows, key=lambda r: r["selection_hash"])


def read_candidate_manifest(path: Path, *, suite: str, eval_seed: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ExactRestoreError("candidate manifest is empty")
    required = {"protocol_id", "suite", "task_idx", "state_id", "eval_seed", "selection_hash"}
    missing = required - set(rows[0])
    if missing:
        raise ExactRestoreError(f"candidate manifest missing columns: {','.join(sorted(missing))}")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if str(row["suite"]) != suite:
            raise ExactRestoreError(f"candidate manifest row {idx} suite mismatch: {row['suite']} != {suite}")
        if int(row["eval_seed"]) != int(eval_seed):
            raise ExactRestoreError(f"candidate manifest row {idx} eval_seed mismatch: {row['eval_seed']} != {eval_seed}")
        protocol_id = str(row["protocol_id"])
        task_idx = int(row["task_idx"])
        state_id = int(row["state_id"])
        expected_key = f"{protocol_id}|{suite}|{task_idx}|{state_id}|{int(eval_seed)}"
        expected_hash = hashlib.sha256(expected_key.encode("utf-8")).hexdigest()
        if str(row["selection_hash"]) != expected_hash:
            raise ExactRestoreError(f"candidate manifest row {idx} selection_hash mismatch")
        out.append(
            {
                "protocol_id": protocol_id,
                "suite": suite,
                "task_idx": task_idx,
                "state_id": state_id,
                "eval_seed": int(eval_seed),
                "selection_hash": expected_hash,
            }
        )
    return out


def write_candidate_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["protocol_id", "suite", "task_idx", "state_id", "eval_seed", "selection_hash", "status", "reason"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(payload)


def build_real_env_for_candidate(*, suite: str, task_idx: int, state_id: int, render_gpu: int, max_steps: int):
    from libero.libero import benchmark, get_libero_path
    from gripper_attack.libero_v4_env_factory import apply_dummy_wait, build_v4_exact_env

    bm = benchmark.get_benchmark_dict()
    suite_obj = bm[suite]()
    task_obj = suite_obj.get_task(int(task_idx))
    init_states = suite_obj.get_task_init_states(int(task_idx))
    if int(state_id) >= len(init_states):
        raise ExactRestoreError(f"state_id {state_id} out of range for {suite} task {task_idx}")
    bddl = os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)
    env, obs = build_v4_exact_env(bddl, int(render_gpu), int(max_steps), NUM_STEPS_WAIT)
    obs = env.set_init_state(init_states[int(state_id)])
    env, obs = apply_dummy_wait(env, obs, NUM_STEPS_WAIT)
    return env, obs, task_obj, bddl


def load_real_policy_and_student(args: argparse.Namespace, *, env_adapter: RealLiberoEnvAdapter, instruction: str):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    from gripper_attack.sc5_detector_runtime import SC5DetectorRuntime
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2

    np.random.seed(int(args.eval_seed))
    random.seed(int(args.eval_seed))
    torch.manual_seed(int(args.eval_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.eval_seed))

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    visible = torch.cuda.device_count()
    if visible <= 0:
        raise ExactRestoreError("no CUDA device visible")
    model = AutoModelCls.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={idx: "10000MiB" for idx in range(visible)} | {"cpu": "128GiB"},
        attn_implementation="eager",
    )
    model.eval()
    device = "cuda:0"
    for v in getattr(model, "hf_device_map", {}).values():
        if isinstance(v, int):
            device = f"cuda:{v}"
            break
    action_dim = int(model.get_action_dim(args.unnorm_key))
    policy = RealOpenVLAPolicyAdapter(
        model=model,
        processor=processor,
        device=device,
        instruction=instruction,
        unnorm_key=args.unnorm_key,
        action_dim=action_dim,
    )
    detector = SC5DetectorRuntime(args.detector_path, guard=5)
    student = RealSC5StudentAdapter(
        detector=detector,
        streamer=SC5StreamingFeatureAdapterV2(),
        env_adapter=env_adapter,
    )
    return policy, student, model, detector


def build_parent_manifest_for_candidate(
    args: argparse.Namespace,
    *,
    task_idx: int,
    state_id: int,
    instruction: str,
    tau_corridor: float,
    tau_release: float,
) -> Layer3ParentDependencyManifest:
    model_sha = sha256_path(Path(args.model_path))
    detector_sha = sha256_path(Path(args.detector_path))
    return Layer3ParentDependencyManifest(
        suite=args.suite,
        task_idx=int(task_idx),
        state_id=int(state_id),
        eval_seed=int(args.eval_seed),
        parent_key=f"{args.suite}|{int(task_idx)}|{int(state_id)}|{int(args.eval_seed)}|CLEAN",
        openvla_model_sha256=model_sha,
        unnorm_key=args.unnorm_key,
        layer2_dataset_sha256=EXPECTED_LAYER2_DATASET_SHA256,
        detector_checkpoint_sha256=detector_sha,
        tau_corridor=float(tau_corridor),
        tau_release=float(tau_release),
        libero_version="openvla_official_libero_20260525",
        mujoco_version="runtime_recorded",
        task_instruction_sha256=hash_jsonable({"instruction": instruction}),
    )


def find_emit_snapshot_for_candidate(
    *,
    args: argparse.Namespace,
    candidate: Mapping[str, Any],
    attempt_dir: Path,
    repetition: int,
) -> dict[str, Any]:
    env = None
    try:
        env, obs, task_obj, bddl = build_real_env_for_candidate(
            suite=args.suite,
            task_idx=int(candidate["task_idx"]),
            state_id=int(candidate["state_id"]),
            render_gpu=int(args.render_gpu),
            max_steps=int(args.max_steps),
        )
        env_adapter = RealLiberoEnvAdapter(env)
        instruction = str(task_obj.language)
        parent = build_parent_manifest_for_candidate(
            args,
            task_idx=int(candidate["task_idx"]),
            state_id=int(candidate["state_id"]),
            instruction=instruction,
            tau_corridor=float(detector.tau_c),
            tau_release=float(detector.tau_r),
        )
        dependency = validate_dependency_files(
            parent,
            openvla_model_path=args.model_path,
            detector_checkpoint_path=args.detector_path,
        )
        runtime = capture_runtime_receipt(
            libero_version=parent.libero_version,
            mujoco_version=parent.mujoco_version,
            openvla_generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 7},
        )
        selected: dict[str, Any] | None = None
        first_valid_step = -1
        for step in range(int(args.max_steps)):
            action, tokens = policy.act(obs)
            update_student_for_step(student, step=step, obs=obs, action=action, tokens=tokens)
            if first_valid_step < 0 and student.invalid_steps == 0:
                first_valid_step = step
            if detector.emitted and detector.emit_step == step:
                if int(args.max_steps) - step < RESTORE_STEPS:
                    raise ExactRestoreError("emit occurred without five remaining steps")
                mujoco_state = capture_mujoco_state(env_adapter)
                env_state = capture_env_internal_state(env_adapter)
                policy_rng = capture_policy_rng_state(policy)
                student_state = capture_student_state(student)
                feature_history = capture_feature_history(student)
                prefix = build_prefix_snapshot(
                    parent=parent,
                    emit_step=step,
                    observation=obs,
                    mujoco_state=mujoco_state,
                    policy_rng_state=policy_rng,
                    student_state=student_state,
                    feature_history=feature_history,
                    source_episode_relpath=f"{args.suite}/task{candidate['task_idx']}/state{candidate['state_id']}",
                )
                snapshot = ExactRestoreSnapshotPayload(
                    prefix=prefix,
                    parent_manifest=parent,
                    mujoco_state=mujoco_state,
                    env_internal_state=env_state,
                    policy_rng_state=policy_rng,
                    student_state=student_state,
                    feature_history=feature_history,
                    observation=obs,
                    clean_action_t=action,
                    clean_tokens_t=tokens,
                )
                selected = {
                    "snapshot": snapshot,
                    "policy": policy,
                    "student": student,
                    "env_adapter": env_adapter,
                    "obs_t": obs,
                    "parent": parent,
                    "runtime": runtime,
                    "dependency": dependency,
                    "instruction": instruction,
                    "bddl": bddl,
                    "first_valid_step": first_valid_step,
                }
                break
            obs, _reward, done, _info = env_adapter.step(action)
            if done:
                break
        if selected is None:
            raise ExactRestoreError("candidate did not produce eligible natural Student emit")
        return selected
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise


def new_env_policy_student_for_snapshot(args: argparse.Namespace, selected: Mapping[str, Any]):
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    env, _obs, task_obj, _bddl = build_real_env_for_candidate(
        suite=snapshot.parent_manifest.suite,
        task_idx=snapshot.parent_manifest.task_idx,
        state_id=snapshot.parent_manifest.state_id,
        render_gpu=int(args.render_gpu),
        max_steps=int(args.max_steps),
    )
    env_adapter = RealLiberoEnvAdapter(env)
    policy, student, _model, _detector = load_real_policy_and_student(
        args, env_adapter=env_adapter, instruction=str(task_obj.language)
    )
    return env_adapter, policy, student


def run_selected_parent_attempt(args: argparse.Namespace, selected: Mapping[str, Any], attempt_dir: Path, repetition: int) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    env_adapter: RealLiberoEnvAdapter = selected["env_adapter"]
    policy: RealOpenVLAPolicyAdapter = selected["policy"]
    student: RealSC5StudentAdapter = selected["student"]
    obs_t = selected["obs_t"]
    (attempt_dir / "stdout.log").write_text("", encoding="utf-8")
    (attempt_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(attempt_dir / "parent_dependency_manifest.json", asdict(selected["parent"]))
    write_json(attempt_dir / "runtime_receipt.json", asdict(selected["runtime"]))
    write_json(attempt_dir / "dependency_sha_receipt.json", selected["dependency"])
    write_json(attempt_dir / "prefix_snapshot_manifest.json", asdict(snapshot.prefix))

    reference = rollout_clean_steps(
        env=env_adapter,
        student=student,
        policy=policy,
        initial_obs=obs_t,
        start_step=snapshot.prefix.emit_step,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
        first_step_student_already_updated=True,
    )
    reference_frames = list(env_adapter.frames)
    env_adapter.close()

    replay_env_a, replay_policy_a, replay_student_a = new_env_policy_student_for_snapshot(args, selected)
    replay_obs_a = restore_snapshot_and_recapture_observation(replay_env_a, replay_student_a, snapshot, replay_policy_a)
    branch_record = recapture_branch_record(
        condition="CLEAN_REPLAY",
        snapshot=snapshot,
        env=replay_env_a,
        student=replay_student_a,
        policy=replay_policy_a,
    )
    replay_a = rollout_clean_steps(
        env=replay_env_a,
        student=replay_student_a,
        policy=replay_policy_a,
        initial_obs=replay_obs_a,
        start_step=snapshot.prefix.emit_step,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
        first_step_student_already_updated=True,
    )
    replay_a_frames = list(replay_env_a.frames)
    replay_env_a.close()

    replay_env_b, replay_policy_b, replay_student_b = new_env_policy_student_for_snapshot(args, selected)
    replay_obs_b = restore_snapshot_and_recapture_observation(replay_env_b, replay_student_b, snapshot, replay_policy_b)
    replay_b = rollout_clean_steps(
        env=replay_env_b,
        student=replay_student_b,
        policy=replay_policy_b,
        initial_obs=replay_obs_b,
        start_step=snapshot.prefix.emit_step,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
        first_step_student_already_updated=True,
    )
    replay_b_frames = list(replay_env_b.frames)
    replay_env_b.close()

    result = validate_clean_restore_pair(
        snapshot=snapshot,
        branch_records=[branch_record],
        reference=reference,
        replay_a=replay_a,
        replay_b=replay_b,
    )
    write_json(attempt_dir / "branch_records.json", [asdict(branch_record)])
    write_jsonl(attempt_dir / "reference_steps.jsonl", reference)
    write_jsonl(attempt_dir / "replay_a_steps.jsonl", replay_a)
    write_jsonl(attempt_dir / "replay_b_steps.jsonl", replay_b)
    write_json(attempt_dir / "numeric_drift_report.json", {"reference_vs_replay": [], "replay_a_vs_replay_b": []})
    write_real_video(attempt_dir / "raw_reference.mp4", reference_frames)
    write_real_video(attempt_dir / "raw_replay_a.mp4", replay_a_frames)
    write_real_video(attempt_dir / "raw_replay_b.mp4", replay_b_frames)
    summary = {
        **result,
        "attempt_id": attempt_dir.name,
        "repetition": repetition,
        "suite": snapshot.parent_manifest.suite,
        "task_idx": snapshot.parent_manifest.task_idx,
        "state_id": snapshot.parent_manifest.state_id,
        "emit_step": snapshot.prefix.emit_step,
    }
    write_json(attempt_dir / "attempt_summary.json", summary)
    write_recursive_manifest(attempt_dir)
    return summary


def run_real_libero_single_parent(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ExactRestoreError(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model_binding = validate_real_openvla_model_binding(
        suite=args.suite,
        model_path=args.model_path,
        unnorm_key=args.unnorm_key,
    )
    write_json(out / "openvla_model_binding_receipt.json", model_binding)
    if args.candidate_manifest:
        candidates = read_candidate_manifest(Path(args.candidate_manifest), suite=args.suite, eval_seed=args.eval_seed)
    else:
        candidates = make_candidate_order(
            suite=args.suite,
            state_start=args.state_start,
            state_end=args.state_end,
            task_count=args.task_count,
            eval_seed=args.eval_seed,
        )
    candidate_rows = [dict(row, status="PLANNED", reason="") for row in candidates]
    write_candidate_manifest(out / "candidate_manifest.csv", candidate_rows)
    selected: dict[str, Any] | None = None
    selected_idx = -1
    for idx, cand in enumerate(candidates):
        try:
            selected = find_emit_snapshot_for_candidate(args=args, candidate=cand, attempt_dir=out, repetition=0)
            selected_idx = idx
            candidate_rows[idx]["status"] = "SELECTED"
            break
        except Exception as exc:
            candidate_rows[idx]["status"] = "INELIGIBLE"
            candidate_rows[idx]["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
            write_candidate_manifest(out / "candidate_manifest.csv", candidate_rows)
            continue
    if selected is None:
        write_json(out / "single_parent_restore_qualification_summary.json", {"result": "NO_ELIGIBLE_GOAL_RESTORE_PARENT"})
        raise ExactRestoreError("NO_ELIGIBLE_GOAL_RESTORE_PARENT")
    write_candidate_manifest(out / "candidate_manifest.csv", candidate_rows)
    summaries = []
    try:
        first_summary = run_selected_parent_attempt(args, selected, out / "attempt_00", 0)
        summaries.append(first_summary)
        for rep in range(1, int(args.repetitions)):
            fresh_selected = find_emit_snapshot_for_candidate(
                args=args,
                candidate=candidates[selected_idx],
                attempt_dir=out,
                repetition=rep,
            )
            summaries.append(run_selected_parent_attempt(args, fresh_selected, out / f"attempt_{rep:02d}", rep))
    finally:
        try:
            selected["env_adapter"].close()
        except Exception:
            pass
    passed = sum(1 for s in summaries if s.get("clean_restore_pass") is True)
    final = {
        "stage": "REAL_LIBERO_SINGLE_PARENT_CLEAN_RESTORE_QUALIFICATION",
        "completed": len(summaries),
        "passed": passed,
        "failed": len(summaries) - passed,
        "result": "SINGLE_PARENT_REAL_LIBERO_RESTORE_ENGINEERING_PASS"
        if len(summaries) == int(args.repetitions) and passed == int(args.repetitions)
        else "SINGLE_PARENT_REAL_LIBERO_RESTORE_FAIL",
        "selected_candidate": candidates[selected_idx],
        "attempts": summaries,
    }
    write_json(out / "single_parent_restore_qualification_summary.json", final)
    report = [
        "# Single Parent Real LIBERO Clean Restore Qualification",
        "",
        f"- result: `{final['result']}`",
        f"- completed: {final['completed']}",
        f"- passed: {final['passed']}",
        f"- selected: `{candidates[selected_idx]}`",
        "",
        "Forbidden claims: VIS/RAND/shuffled/oracle/attack effectiveness.",
    ]
    (out / "single_parent_restore_qualification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    seal = write_recursive_manifest(out)
    final["recursive_sha256_manifest_sha256"] = seal
    write_json(out / "single_parent_restore_qualification_summary.json", final)
    print(json.dumps(final, sort_keys=True, default=str))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mock", action="store_true", help="Run the CPU mock restore smoke; no LIBERO/OpenVLA/GPU.")
    ap.add_argument(
        "--real-libero-single-parent",
        action="store_true",
        help="Run authorized clean-only real LIBERO single-parent exact-restore qualification.",
    )
    ap.add_argument("--suite", default="libero_goal", choices=sorted(SUPPORTED_SUITES))
    ap.add_argument("--model-path", default="")
    ap.add_argument("--unnorm-key", default="")
    ap.add_argument("--detector-path", default="")
    ap.add_argument("--render-gpu", type=int, default=-1)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--state-start", type=int, default=20)
    ap.add_argument("--state-end", type=int, default=23)
    ap.add_argument("--task-count", type=int, default=10)
    ap.add_argument("--candidate-manifest", default="")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--repetitions", type=int, default=3)
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
    if args.mock and args.real_libero_single_parent:
        raise SystemExit("choose exactly one of --mock or --real-libero-single-parent")
    if args.mock:
        run_mock(Path(args.output_dir))
        return
    if args.real_libero_single_parent:
        run_real_libero_single_parent(args)
        return
    raise SystemExit("Specify --mock or --real-libero-single-parent.")


if __name__ == "__main__":
    main()
