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
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
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
    ExactActionPrefixReplayPayload,
    Layer3BranchingContractError,
    PrefixBranchSnapshot,
    PrefixReplayStep,
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


class NoNaturalStudentEmit(ExactRestoreError):
    """Raised only when the authorized parent never reaches natural Student emit."""


class PrefixReplayDivergence(ExactRestoreError):
    """Raised when exact action-prefix replay diverges from the reference trace."""


class InfraInvalidError(ExactRestoreError):
    """Raised for infrastructure/provenance failures that invalidate the run."""


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


def _finite_max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float | str:
    if actual.shape != expected.shape:
        return ""
    try:
        if actual.size == 0:
            return 0.0
        return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    except Exception:
        return ""


def assert_array_exact(actual: Any, expected: Any, *, name: str) -> dict[str, Any]:
    """Fail-closed byte-exact array identity gate.

    Numeric closeness is reported only as diagnostics. Passing requires exact
    shape, exact dtype, exact element equality, and matching canonical byte SHA.
    """

    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    shape_match = tuple(actual_arr.shape) == tuple(expected_arr.shape)
    dtype_match = str(actual_arr.dtype) == str(expected_arr.dtype)
    actual_sha = hash_array(actual_arr)
    expected_sha = hash_array(expected_arr)
    array_equal = bool(shape_match and dtype_match and np.array_equal(actual_arr, expected_arr))
    nonzero_diff_count: int | str = ""
    if shape_match:
        try:
            nonzero_diff_count = int(np.count_nonzero(actual_arr != expected_arr))
        except Exception:
            nonzero_diff_count = ""
    report = {
        "name": str(name),
        "shape_match": bool(shape_match),
        "dtype_match": bool(dtype_match),
        "array_equal": bool(array_equal),
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "byte_sha_exact": bool(actual_sha == expected_sha),
        "max_abs_diff": _finite_max_abs_diff(actual_arr, expected_arr),
        "nonzero_diff_count": nonzero_diff_count,
    }
    report["exact"] = bool(
        report["shape_match"] and report["dtype_match"] and report["array_equal"] and report["byte_sha_exact"]
    )
    if not report["exact"]:
        raise PrefixReplayDivergence(
            f"{name} exact mismatch "
            f"(shape={report['shape_match']}, dtype={report['dtype_match']}, "
            f"array_equal={report['array_equal']}, sha={report['byte_sha_exact']})"
        )
    return report


def assert_tokens_exact(actual: Sequence[Any], expected: Sequence[Any], *, name: str) -> dict[str, Any]:
    actual_tokens = tuple(int(x) for x in actual)
    expected_tokens = tuple(int(x) for x in expected)
    if len(actual_tokens) != 7 or len(expected_tokens) != 7:
        raise PrefixReplayDivergence(f"{name} token sequences must both have exactly 7 tokens")
    actual_sha = hash_jsonable(list(actual_tokens))
    expected_sha = hash_jsonable(list(expected_tokens))
    exact = actual_tokens == expected_tokens and actual_sha == expected_sha
    report = {
        "name": str(name),
        "actual_tokens": list(actual_tokens),
        "expected_tokens": list(expected_tokens),
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "exact": bool(exact),
    }
    if not exact:
        raise PrefixReplayDivergence(f"{name} token mismatch")
    return report


def action_identity_report(candidate: Any, expected: Any) -> dict[str, Any]:
    """Strict action identity report for exact-restore branch gates.

    This intentionally does not cast either side before comparing. A numeric
    allclose pass is useful diagnostics, but it is not enough for exact replay.
    """
    candidate_arr = np.asarray(candidate)
    expected_arr = np.asarray(expected)
    same_shape = tuple(candidate_arr.shape) == tuple(expected_arr.shape)
    same_dtype = str(candidate_arr.dtype) == str(expected_arr.dtype)
    candidate_sha = hash_array(candidate_arr)
    expected_sha = hash_array(expected_arr)
    array_equal = bool(same_shape and same_dtype and np.array_equal(candidate_arr, expected_arr))
    max_abs_diff = ""
    if same_shape:
        try:
            max_abs_diff = float(np.max(np.abs(candidate_arr.astype(np.float64) - expected_arr.astype(np.float64))))
        except Exception:
            max_abs_diff = ""
    return {
        "shape_exact": bool(same_shape),
        "dtype_exact": bool(same_dtype),
        "array_equal": bool(array_equal),
        "candidate_action_sha256": candidate_sha,
        "expected_action_sha256": expected_sha,
        "byte_sha_exact": bool(candidate_sha == expected_sha),
        "max_abs_diff": max_abs_diff,
        "exact": bool(array_equal and candidate_sha == expected_sha),
    }


def require_action_byte_exact(candidate: Any, expected: Any, *, context: str) -> dict[str, Any]:
    report = action_identity_report(candidate, expected)
    if not report["exact"]:
        raise ExactRestoreError(
            f"{context} action byte identity mismatch "
            f"(shape_exact={report['shape_exact']}, dtype_exact={report['dtype_exact']}, "
            f"array_equal={report['array_equal']}, byte_sha_exact={report['byte_sha_exact']}, "
            f"max_abs_diff={report['max_abs_diff']})"
        )
    return report


def hash_jsonable(value: Any) -> str:
    return sha256_jsonable(_json_clone(value))


def typed_value_manifest(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hash_array(value),
        }
    if isinstance(value, Mapping):
        return {str(k): typed_value_manifest(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return {"__tuple__": [typed_value_manifest(v) for v in value]}
    if isinstance(value, list):
        return [typed_value_manifest(v) for v in value]
    return _json_clone(value)


def hash_typed_observation(value: Any) -> str:
    return sha256_jsonable(typed_value_manifest(value))


def clone_typed_observation(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {k: clone_typed_observation(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_typed_observation(v) for v in value)
    if isinstance(value, list):
        return [clone_typed_observation(v) for v in value]
    return copy.deepcopy(value)


def compact_state_value(value: Any, *, depth: int = 0, max_depth: int = 3) -> Any:
    if depth > max_depth:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        out: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hash_array(value),
        }
        if flat.size <= 16:
            out["values"] = flat.tolist()
        else:
            out["head"] = flat[:8].tolist()
            out["tail"] = flat[-8:].tolist()
        return out
    if torch is not None and torch.is_tensor(value):
        return compact_state_value(value.detach().cpu().numpy(), depth=depth, max_depth=max_depth)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _json_clone(value)
    if isinstance(value, Mapping):
        return {
            str(k): compact_state_value(v, depth=depth + 1, max_depth=max_depth)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        if len(value) <= 32:
            return [compact_state_value(v, depth=depth + 1, max_depth=max_depth) for v in value]
        return {
            "type": type(value).__name__,
            "len": len(value),
            "head": [compact_state_value(v, depth=depth + 1, max_depth=max_depth) for v in list(value)[:8]],
            "tail": [compact_state_value(v, depth=depth + 1, max_depth=max_depth) for v in list(value)[-8:]],
        }
    text = repr(value)[:240]
    payload = {"type": type(value).__name__, "module": getattr(value.__class__, "__module__", "")}
    if " object at 0x" not in text:
        payload["repr"] = text
    return payload


def capture_object_state(obj: Any, *, max_depth: int = 2) -> dict[str, Any]:
    state: dict[str, Any] = {
        "class": obj.__class__.__name__,
        "module": getattr(obj.__class__, "__module__", ""),
    }
    try:
        raw_attrs = vars(obj)
    except Exception:
        raw_attrs = {}
    attrs: dict[str, Any] = {}
    interesting = (
        "goal",
        "current",
        "start",
        "step",
        "total",
        "ramp",
        "last",
        "prev",
        "action",
        "control",
        "counter",
        "timestep",
        "interpolator",
        "controller",
        "torque",
        "qpos",
        "qvel",
        "input",
        "output",
    )
    for name, value in sorted(raw_attrs.items()):
        lname = str(name).lower()
        if name.startswith("__"):
            continue
        if not any(tok in lname for tok in interesting) and not isinstance(value, (int, float, bool, str, np.ndarray)):
            continue
        if callable(value):
            continue
        attrs[str(name)] = compact_state_value(value, max_depth=max_depth)
    state["attrs"] = attrs
    return state


def flatten_state(prefix: str, value: Any, out: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_state(next_prefix, value[key], out)
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            flatten_state(f"{prefix}[{idx}]", item, out)
        return
    out[prefix] = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def diff_state_dicts(reference: Mapping[str, Any], replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    left: dict[str, str] = {}
    right: dict[str, str] = {}
    flatten_state("", reference, left)
    flatten_state("", replay, right)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        lval = left.get(key, "")
        rval = right.get(key, "")
        if lval == rval:
            continue
        rows.append(
            {
                "field": key,
                "reference_present": key in left,
                "replay_present": key in right,
                "reference_sha256": hashlib.sha256(lval.encode("utf-8")).hexdigest() if lval else "",
                "replay_sha256": hashlib.sha256(rval.encode("utf-8")).hexdigest() if rval else "",
                "reference_value": lval[:1000],
                "replay_value": rval[:1000],
            }
        )
    return rows


def classify_transition_diff(field: str) -> str:
    name = field.lower()
    if "repr" in name:
        return "OBJECT_IDENTITY_OR_REPR_NOISE"
    if "j_full" in name or "j_pos" in name or "j_ori" in name or "mass_matrix" in name:
        return "CONTROLLER_DERIVED_CACHE"
    if "goal_pos" in name or "goal_ori" in name or "goal_orientation" in name or "goal_qpos" in name:
        return "CONTROLLER_MUTABLE_GOAL_STATE"
    if "interpolator" in name:
        return "INTERPOLATOR_MUTABLE_STATE"
    if "previous" in name or "prev" in name or "last_action" in name or "action_buffer" in name:
        return "ROBOT_ACTION_HISTORY"
    if "counter" in name or "timestep" in name or "elapsed" in name or "cur_time" in name or "control_freq" in name:
        return "CONTROL_LOOP_COUNTER"
    if name.startswith("mujoco.qacc_warmstart") or name.startswith("mujoco.qfrc") or name.startswith("mujoco.xfrc"):
        return "MUJOCO_WARMSTART_STATE"
    if name.startswith("mujoco.qacc"):
        return "MUJOCO_DERIVED_ACCELERATION"
    if name.startswith("mujoco.eq_active"):
        return "MUJOCO_WARMSTART_STATE"
    if (
        "input_min" in name
        or "input_max" in name
        or "output_min" in name
        or "output_max" in name
        or "control_limits" in name
        or "config" in name
    ):
        return "STATIC_CONFIG_DIFFERENCE"
    if "repeat" in name or "action_repeat" in name:
        return "CONTROL_LOOP_COUNTER"
    return "UNKNOWN_TRANSITION_STATE"


TRANSITION_CLASS_PRIORITY = (
    "CONTROLLER_MUTABLE_GOAL_STATE",
    "INTERPOLATOR_MUTABLE_STATE",
    "ROBOT_ACTION_HISTORY",
    "CONTROL_LOOP_COUNTER",
    "MUJOCO_WARMSTART_STATE",
    "MUJOCO_DERIVED_ACCELERATION",
    "CONTROLLER_DERIVED_CACHE",
    "STATIC_CONFIG_DIFFERENCE",
    "OBJECT_IDENTITY_OR_REPR_NOISE",
    "UNKNOWN_TRANSITION_STATE",
)


def annotate_transition_diffs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        annotated = dict(row)
        annotated["classification"] = classify_transition_diff(str(row.get("field", "")))
        out.append(annotated)
    return out


def transition_classification_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in TRANSITION_CLASS_PRIORITY}
    for row in rows:
        cls = str(row.get("classification") or classify_transition_diff(str(row.get("field", ""))))
        counts[cls] = counts.get(cls, 0) + 1
    return {key: value for key, value in counts.items() if value}


def primary_transition_diff(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    priority = {name: idx for idx, name in enumerate(TRANSITION_CLASS_PRIORITY)}
    return min(
        (dict(row) for row in rows),
        key=lambda row: (
            priority.get(str(row.get("classification")), 999),
            1 if str(row.get("field", "")).endswith(".repr") else 0,
            str(row.get("field", "")),
        ),
    )


def validate_transition_state_audit_known_parent(snapshot: ExactRestoreSnapshotPayload) -> None:
    parent = snapshot.parent_manifest
    if (
        parent.suite != "libero_goal"
        or int(parent.task_idx) != 4
        or int(parent.state_id) != 1
        or int(parent.eval_seed) != 0
    ):
        raise ExactRestoreError(
            "transition-state audit is authorized only for known parent "
            "libero_goal|4|1|0|CLEAN"
        )


def validate_known_goal_candidate(candidate: Mapping[str, Any]) -> None:
    if (
        str(candidate.get("suite")) != "libero_goal"
        or int(candidate.get("task_idx", -1)) != 4
        or int(candidate.get("state_id", -1)) != 1
        or int(candidate.get("eval_seed", -1)) != 0
    ):
        raise ExactRestoreError("C3 is authorized only for known parent libero_goal|4|1|0|CLEAN")


def _object_attr_state(obj: Any, attr_names: Sequence[str], *, max_depth: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in attr_names:
        if not hasattr(obj, name):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}:{exc}"}
            continue
        if callable(value):
            continue
        out[name] = compact_state_value(value, max_depth=max_depth)
    return out


def capture_robot_control_state(env_adapter: "RealLiberoEnvAdapter") -> list[dict[str, Any]]:
    env = env_adapter.env
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", getattr(env, "robots", []))
    rows: list[dict[str, Any]] = []
    for idx, robot in enumerate(list(robots) if robots is not None else []):
        row = capture_object_state(robot, max_depth=2)
        row["robot_index"] = int(idx)
        row["robot_selected_attrs"] = _object_attr_state(
            robot,
            (
                "recent_qpos",
                "recent_qvel",
                "recent_torques",
                "recent_ee_forcetorques",
                "recent_ee_pose",
                "recent_ee_vel",
                "_joint_positions",
                "_joint_velocities",
                "action_limits",
            ),
            max_depth=2,
        )
        for attr in ("controller", "gripper", "robot_model"):
            if hasattr(robot, attr):
                try:
                    child = getattr(robot, attr)
                except Exception as exc:
                    row[attr] = {"error": f"{type(exc).__name__}:{exc}"}
                    continue
                row[attr] = capture_object_state(child, max_depth=3)
                if attr == "controller":
                    row["controller_selected_attrs"] = _object_attr_state(
                        child,
                        (
                            "goal_pos",
                            "goal_ori",
                            "goal_orientation",
                            "input_min",
                            "input_max",
                            "output_min",
                            "output_max",
                            "control_limits",
                            "interpolator_pos",
                            "interpolator_ori",
                            "interpolator",
                            "_interpolator",
                        ),
                        max_depth=3,
                    )
        rows.append(row)
    return rows


CONTROL_MUTABLE_GOAL_ATTRS = ("goal_pos", "goal_ori", "goal_orientation", "goal_qpos")
INTERPOLATOR_MUTABLE_ATTRS = ("start", "goal", "step", "total_steps", "ori_interpolate")
ROBOT_ACTION_HISTORY_ATTRS = ("recent_actions",)
ENV_COUNTER_ATTRS = ("timestep", "_timestep", "cur_time", "_elapsed_steps")


def snapshot_simple_object_attrs(obj: Any, attrs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in attrs:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if isinstance(value, np.ndarray):
            out[name] = value.copy()
        elif isinstance(value, (int, float, bool, str)) or value is None:
            out[name] = copy.deepcopy(value)
    return out


def snapshot_mutable_interpolator_state(interpolator: Any | None) -> dict[str, Any] | None:
    if interpolator is None:
        return None
    return snapshot_simple_object_attrs(interpolator, INTERPOLATOR_MUTABLE_ATTRS)


def snapshot_action_history_buffer(buffer: Any | None) -> dict[str, Any] | None:
    if buffer is None:
        return None
    try:
        raw_attrs = vars(buffer)
    except Exception:
        return None
    out: dict[str, Any] = {}
    for name, value in raw_attrs.items():
        if isinstance(value, np.ndarray):
            out[name] = value.copy()
        elif isinstance(value, (list, tuple)):
            cloned = []
            ok = True
            for item in value:
                if isinstance(item, np.ndarray):
                    cloned.append(item.copy())
                elif isinstance(item, (int, float, bool, str)) or item is None:
                    cloned.append(copy.deepcopy(item))
                else:
                    ok = False
                    break
            if ok:
                out[name] = type(value)(cloned) if isinstance(value, tuple) else cloned
        elif isinstance(value, (int, float, bool, str)) or value is None:
            out[name] = copy.deepcopy(value)
    return out


def restore_simple_object_attrs(obj: Any, values: Mapping[str, Any]) -> list[str]:
    restored = []
    for name, value in values.items():
        if not hasattr(obj, name):
            continue
        current = getattr(obj, name)
        if isinstance(current, np.ndarray):
            current[...] = np.asarray(value, dtype=current.dtype)
        else:
            setattr(obj, name, copy.deepcopy(value))
        restored.append(str(name))
    return restored


def restore_action_history_buffer(buffer: Any | None, values: Mapping[str, Any] | None) -> list[str]:
    if buffer is None or values is None:
        return []
    restored = []
    for name, value in values.items():
        if not hasattr(buffer, name):
            continue
        current = getattr(buffer, name)
        if isinstance(current, np.ndarray):
            current[...] = np.asarray(value, dtype=current.dtype)
        else:
            setattr(buffer, name, copy.deepcopy(value))
        restored.append(str(name))
    return restored


def get_env_robots(env_adapter: "RealLiberoEnvAdapter") -> list[Any]:
    env = env_adapter.env
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", getattr(env, "robots", []))
    return list(robots) if robots is not None else []


def snapshot_control_ablation_state(env_adapter: "RealLiberoEnvAdapter") -> dict[str, Any]:
    env = env_adapter.env
    inner = getattr(env, "env", env)
    robots_out = []
    for robot in get_env_robots(env_adapter):
        ctrl = getattr(robot, "controller", None)
        row: dict[str, Any] = {
            "controller_goal": snapshot_simple_object_attrs(ctrl, CONTROL_MUTABLE_GOAL_ATTRS) if ctrl is not None else {},
            "interpolator_pos": snapshot_mutable_interpolator_state(getattr(ctrl, "interpolator_pos", None))
            if ctrl is not None
            else None,
            "interpolator_ori": snapshot_mutable_interpolator_state(getattr(ctrl, "interpolator_ori", None))
            if ctrl is not None
            else None,
            "action_history": {
                name: snapshot_action_history_buffer(getattr(robot, name, None))
                for name in ROBOT_ACTION_HISTORY_ATTRS
                if hasattr(robot, name)
            },
        }
        robots_out.append(row)
    qacc = None
    if hasattr(env.sim.data, "qacc"):
        qacc = np.asarray(env.sim.data.qacc).copy()
    return {
        "robots": robots_out,
        "env_counters": snapshot_simple_object_attrs(inner, ENV_COUNTER_ATTRS),
        "qacc": qacc,
    }


def refresh_derived_controller_state(env_adapter: "RealLiberoEnvAdapter") -> list[str]:
    refreshed = []
    for idx, robot in enumerate(get_env_robots(env_adapter)):
        ctrl = getattr(robot, "controller", None)
        if ctrl is not None and hasattr(ctrl, "update"):
            ctrl.update(force=True)
            refreshed.append(f"robot{idx}.controller.update(force=True)")
    return refreshed


def apply_control_ablation_state(
    env_adapter: "RealLiberoEnvAdapter",
    reference_state: Mapping[str, Any],
    *,
    restore_goal: bool = False,
    restore_interpolator: bool = False,
    restore_action_history: bool = False,
    restore_qacc: bool = False,
    refresh_derived: bool = False,
) -> dict[str, Any]:
    actions: list[str] = []
    robots = get_env_robots(env_adapter)
    ref_robots = list(reference_state.get("robots", []))
    for idx, robot in enumerate(robots):
        if idx >= len(ref_robots):
            continue
        ref_robot = ref_robots[idx]
        ctrl = getattr(robot, "controller", None)
        if restore_goal and ctrl is not None:
            for attr in restore_simple_object_attrs(ctrl, ref_robot.get("controller_goal", {})):
                actions.append(f"robot{idx}.controller.{attr}")
        if restore_interpolator and ctrl is not None:
            for name in ("interpolator_pos", "interpolator_ori"):
                interp = getattr(ctrl, name, None)
                state = ref_robot.get(name)
                if interp is not None and state is not None:
                    for attr in restore_simple_object_attrs(interp, state):
                        actions.append(f"robot{idx}.controller.{name}.{attr}")
        if restore_action_history:
            for name, state in dict(ref_robot.get("action_history", {})).items():
                for attr in restore_action_history_buffer(getattr(robot, name, None), state):
                    actions.append(f"robot{idx}.{name}.{attr}")
    if restore_action_history:
        inner = getattr(env_adapter.env, "env", env_adapter.env)
        for attr in restore_simple_object_attrs(inner, reference_state.get("env_counters", {})):
            actions.append(f"env_inner.{attr}")
    if refresh_derived:
        actions.extend(refresh_derived_controller_state(env_adapter))
    if restore_qacc and reference_state.get("qacc") is not None and hasattr(env_adapter.env.sim.data, "qacc"):
        env_adapter.env.sim.data.qacc[...] = np.asarray(reference_state["qacc"], dtype=env_adapter.env.sim.data.qacc.dtype)
        actions.append("mujoco.qacc")
    return {
        "restore_goal": bool(restore_goal),
        "restore_interpolator": bool(restore_interpolator),
        "restore_action_history": bool(restore_action_history),
        "restore_qacc": bool(restore_qacc),
        "refresh_derived": bool(refresh_derived),
        "actions": actions,
    }


def capture_transition_state(
    *,
    phase: str,
    env_adapter: "RealLiberoEnvAdapter",
    student: Any,
    policy: Any,
    obs: Any,
    action: Sequence[float],
    tokens: Sequence[int],
) -> dict[str, Any]:
    env = env_adapter.env
    inner = getattr(env, "env", env)
    env_action = postprocess_openvla_action_for_libero(action)
    policy_fingerprint: dict[str, Any] = {}
    if hasattr(policy, "policy_input_fingerprint"):
        try:
            policy_fingerprint = policy.policy_input_fingerprint(obs)
        except Exception as exc:
            policy_fingerprint = {"error": f"{type(exc).__name__}:{exc}"}
    sim = getattr(env, "sim", None)
    mujoco_state: dict[str, Any] = {}
    if sim is not None and hasattr(sim, "data"):
        data = sim.data
        for name in MUJOCO_STATE_FIELDS:
            if hasattr(data, name):
                try:
                    mujoco_state[name] = compact_state_value(getattr(data, name), max_depth=1)
                except Exception as exc:
                    mujoco_state[name] = {"error": f"{type(exc).__name__}:{exc}"}
        if hasattr(data, "qacc"):
            mujoco_state["qacc"] = compact_state_value(getattr(data, "qacc"), max_depth=1)
    flat_sim: dict[str, Any] = {}
    if hasattr(env, "get_sim_state"):
        try:
            flat_sim["get_sim_state"] = compact_state_value(np.asarray(env.get_sim_state()).copy(), max_depth=1)
        except Exception as exc:
            flat_sim["get_sim_state"] = {"error": f"{type(exc).__name__}:{exc}"}
    return {
        "phase": str(phase),
        "policy": {
            "observation_sha256": hash_typed_observation(obs),
            "policy_input_fingerprint": policy_fingerprint,
            "action": [float(x) for x in action],
            "action_sha256": hash_jsonable([float(x) for x in action]),
            "tokens": [int(x) for x in tokens],
            "tokens_sha256": hash_jsonable([int(x) for x in tokens]),
            "env_action": [float(x) for x in env_action.tolist()],
            "env_action_sha256": hash_jsonable([float(x) for x in env_action.tolist()]),
        },
        "mujoco": mujoco_state,
        "flat_sim_state": flat_sim,
        "env_wrapper": capture_object_state(env, max_depth=2),
        "env_inner": capture_object_state(inner, max_depth=2),
        "robots": capture_robot_control_state(env_adapter),
        "env_internal_state": capture_env_internal_state(env_adapter, strict=False),
        "student_state": capture_student_state(student, strict=False),
        "feature_history": capture_feature_history(student, strict=False),
        "frame_sha256": hash_array(np.asarray(obs["agentview_image"])) if isinstance(obs, Mapping) and "agentview_image" in obs else "",
    }


def first_transition_diff(pre_diff: Sequence[Mapping[str, Any]], post_diff: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if pre_diff:
        row = primary_transition_diff(pre_diff) or dict(pre_diff[0])
        row["first_divergence_phase"] = "PRE_STEP"
        row["classification"] = str(row.get("classification") or classify_transition_diff(str(row.get("field", ""))))
        return row
    if post_diff:
        row = primary_transition_diff(post_diff) or dict(post_diff[0])
        row["first_divergence_phase"] = "POST_STEP"
        row["classification"] = str(row.get("classification") or classify_transition_diff(str(row.get("field", ""))))
        return row
    return {
        "first_divergence_phase": "NONE",
        "field": "",
        "classification": "NO_TRANSITION_STATE_DIVERGENCE",
    }


def observation_value_summary(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(value).__name__}
    try:
        arr = np.asarray(value)
        if arr.dtype != object:
            out.update(
                {
                    "shape": "x".join(map(str, arr.shape)),
                    "dtype": str(arr.dtype),
                    "sha256": hash_array(arr),
                }
            )
            return out
    except Exception:
        pass
    try:
        out["sha256"] = hash_jsonable(value)
    except Exception as exc:
        out["sha256"] = f"UNHASHABLE:{type(exc).__name__}"
    return out


def compare_observation_values(a: Any, b: Any) -> dict[str, Any]:
    left = observation_value_summary(a)
    right = observation_value_summary(b)
    row: dict[str, Any] = {
        "left_type": left.get("type", ""),
        "right_type": right.get("type", ""),
        "left_shape": left.get("shape", ""),
        "right_shape": right.get("shape", ""),
        "left_dtype": left.get("dtype", ""),
        "right_dtype": right.get("dtype", ""),
        "left_sha256": left.get("sha256", ""),
        "right_sha256": right.get("sha256", ""),
        "sha_match": left.get("sha256") == right.get("sha256"),
        "max_abs_diff": "",
        "mean_abs_diff": "",
        "nonzero_diff_count": "",
        "first_diff_index": "",
    }
    try:
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape == bb.shape and aa.dtype != object and bb.dtype != object:
            diff = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
            nz = np.argwhere(diff != 0)
            row["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
            row["mean_abs_diff"] = float(diff.mean()) if diff.size else 0.0
            row["nonzero_diff_count"] = int(nz.shape[0])
            row["first_diff_index"] = nz[0].tolist() if nz.size else ""
    except Exception:
        pass
    return row


def observation_diff_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = sorted(set(left) | set(right))
    rows: list[dict[str, Any]] = []
    for key in keys:
        row = {"key": key, "left_present": key in left, "right_present": key in right}
        if key in left and key in right:
            row.update(compare_observation_values(left[key], right[key]))
        rows.append(row)
    return rows


def write_agentview_diff_artifacts(output_dir: Path, *, prefix: Mapping[str, Any], restored: Mapping[str, Any], label: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "agentview_image" not in prefix or "agentview_image" not in restored:
        return {"label": label, "agentview_available": False}
    a = np.asarray(prefix["agentview_image"])
    b = np.asarray(restored["agentview_image"])
    summary: dict[str, Any] = {
        "label": label,
        "agentview_available": True,
        "prefix_shape": list(a.shape),
        "restored_shape": list(b.shape),
        "prefix_sha256": hash_array(a),
        "restored_sha256": hash_array(b),
        "shape_match": a.shape == b.shape,
    }
    try:
        from PIL import Image

        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(output_dir / f"{label}_prefix_agentview.png")
        Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).save(output_dir / f"{label}_restored_agentview.png")
        if a.shape == b.shape:
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
            Image.fromarray(diff).save(output_dir / f"{label}_agentview_absdiff.png")
            summary.update(
                {
                    "pixel_diff_count": int(np.count_nonzero(diff)),
                    "pixel_max_abs_diff": int(diff.max()) if diff.size else 0,
                    "pixel_mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
                }
            )
    except Exception as exc:
        summary["png_write_error"] = f"{type(exc).__name__}:{exc}"
    return summary


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _safe_command_text(command: Sequence[str], *, timeout: int = 15) -> str:
    try:
        proc = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.stdout
    except Exception as exc:
        return f"COMMAND_FAILED:{type(exc).__name__}:{exc}\n"


def _safe_git_text(args: Sequence[str]) -> str:
    return _safe_command_text(["git", *args], timeout=10).strip()


def write_text_snapshot(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("$ " + " ".join(command) + "\n" + _safe_command_text(command), encoding="utf-8")


def write_gpu_and_kernel_snapshots(output_dir: Path, *, suffix: str) -> None:
    write_text_snapshot(output_dir / f"GPU_{suffix}.txt", ["nvidia-smi"])
    if os.name == "nt":
        (output_dir / f"dmesg_{suffix}.txt").write_text("NOT_AVAILABLE_ON_WINDOWS\n", encoding="utf-8")
    else:
        write_text_snapshot(output_dir / f"dmesg_{suffix}.txt", ["bash", "-lc", "dmesg --ctime | tail -n 200"])


def write_run_manifest(output_dir: Path, args: argparse.Namespace, *, stage: str) -> None:
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "OPENVLA_ATTN_IMPLEMENTATION",
        "TOKENIZERS_PARALLELISM",
    )
    manifest = {
        "stage": stage,
        "argv": list(sys.argv),
        "args": vars(args),
        "cwd": str(Path.cwd()),
        "repo": str(REPO),
        "git_head": _safe_git_text(["rev-parse", "HEAD"]),
        "git_branch": _safe_git_text(["branch", "--show-current"]),
        "git_status_short": _safe_git_text(["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "time_unix": time.time(),
        "environment": {key: os.environ.get(key, "") for key in env_keys},
    }
    write_json(output_dir / "run_manifest.json", manifest)


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
        if self.prefix.observation_sha256 != hash_typed_observation(self.observation):
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
                "observation_sha256": hash_typed_observation(self.observation),
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
        observation_sha256=hash_typed_observation(observation),
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
    obs_hash = hash_typed_observation(obs)
    if obs_hash != snapshot.prefix.observation_sha256:
        raise ExactRestoreError(
            "restored observation hash does not match prefix: "
            f"expected={snapshot.prefix.observation_sha256} actual={obs_hash}"
        )
    return clone_typed_observation(obs)


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
        restored_observation_sha256=hash_typed_observation(actual_observation),
        restored_policy_rng_sha256=hash_jsonable(actual_policy_rng),
        restored_detector_state_sha256=hash_jsonable(actual_student),
        restored_feature_history_sha256=hash_jsonable(actual_history),
        trigger_step=snapshot.prefix.emit_step,
        first_env_step=snapshot.prefix.emit_step,
    )


def captured_prefix_branch_record(
    *,
    condition: str,
    snapshot: ExactRestoreSnapshotPayload,
    env: Any,
    student: Any,
    policy: Any | None,
) -> BranchRunRecord:
    actual_mujoco = capture_mujoco_state(env)
    actual_policy_rng = capture_policy_rng_state(policy)
    actual_student = capture_student_state(student, strict=True)
    actual_history = capture_feature_history(student, strict=True)
    diagnostic_obs_sha = ""
    if hasattr(env, "get_observation_after_restore"):
        try:
            diagnostic_obs_sha = hash_typed_observation(env.get_observation_after_restore())
        except Exception as exc:
            diagnostic_obs_sha = f"DIAGNOSTIC_RECAPTURE_FAILED:{type(exc).__name__}"
    return BranchRunRecord(
        condition=condition,
        prefix_snapshot_sha256=snapshot.prefix.snapshot_sha256,
        branch_source="EXACT_PREFIX_RESTORE",
        restored_sim_state_sha256=hash_jsonable(actual_mujoco),
        restored_observation_sha256=snapshot.prefix.observation_sha256,
        restored_policy_rng_sha256=hash_jsonable(actual_policy_rng),
        restored_detector_state_sha256=hash_jsonable(actual_student),
        restored_feature_history_sha256=hash_jsonable(actual_history),
        trigger_step=snapshot.prefix.emit_step,
        first_env_step=snapshot.prefix.emit_step,
        branch_input_source="CAPTURED_PREFIX_OBSERVATION",
        branch_policy_input_sha256=snapshot.prefix.observation_sha256,
        diagnostic_recaptured_observation_sha256=diagnostic_obs_sha,
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
        observation_sha256=hash_typed_observation(obs),
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


def hash_flat_sim_state(env: Any) -> str:
    """Hash the flat simulator state when available, otherwise hash MuJoCo fields."""

    if hasattr(env, "get_sim_state"):
        return hash_array(np.asarray(env.get_sim_state()))
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "get_sim_state"):
        return hash_array(np.asarray(inner.get_sim_state()))
    return hash_jsonable(capture_mujoco_state(env))


def prefix_replay_state_hashes(*, env: Any, obs: Any, student: Any, policy: Any | None = None) -> dict[str, str]:
    out = {
        "observation_sha256": hash_typed_observation(obs),
        "qpos_sha256": hash_array(getattr(env.sim.data, "qpos", [])),
        "qvel_sha256": hash_array(getattr(env.sim.data, "qvel", [])),
        "flat_sim_state_sha256": hash_flat_sim_state(env),
        "student_state_sha256": hash_jsonable(capture_student_state(student, strict=True)),
        "feature_history_sha256": hash_jsonable(capture_feature_history(student, strict=True)),
    }
    if policy is not None:
        if not hasattr(policy, "policy_input_fingerprint"):
            raise InfraInvalidError("policy object missing policy_input_fingerprint for C3")
        out["policy_input_sha256"] = hash_jsonable(policy.policy_input_fingerprint(obs))
    return out


def _first_present(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


REQUIRED_PREFIX_STEP_FIELDS = (
    "step",
    "raw_action",
    "raw_action_sha256",
    "env_action",
    "env_action_sha256",
    "tokens",
    "tokens_sha256",
    "observation_sha256",
    "policy_input_sha256",
    "qpos_sha256",
    "qvel_sha256",
    "flat_sim_state_sha256",
    "student_state_sha256",
    "feature_history_sha256",
    "post_qpos_sha256",
    "post_qvel_sha256",
    "post_flat_sim_state_sha256",
    "next_observation_sha256",
    "post_student_state_sha256",
    "post_feature_history_sha256",
    "reward",
    "done",
)

REQUIRED_BRANCH_REFERENCE_FIELDS = (
    "observation_sha256",
    "policy_input_sha256",
    "qpos_sha256",
    "qvel_sha256",
    "flat_sim_state_sha256",
    "student_state_sha256",
    "feature_history_sha256",
    "branch_post_student_update_state_sha256",
    "branch_post_student_update_feature_history_sha256",
    "post_branch_qpos_sha256",
    "post_branch_qvel_sha256",
    "post_branch_flat_sim_state_sha256",
    "post_branch_observation_sha256",
    "post_branch_reward",
    "post_branch_done",
)


def _schema_invalid(message: str) -> PrefixReplayDivergence:
    return PrefixReplayDivergence(
        json.dumps({"failure_class": "PREFIX_REPLAY_SCHEMA_INVALID", "error": message}, sort_keys=True)
    )


def _require_fields(row: Mapping[str, Any], fields: Sequence[str], *, context: str) -> None:
    missing = [field for field in fields if field not in row or row[field] in (None, "")]
    if missing:
        raise _schema_invalid(f"{context} missing required fields: {','.join(missing)}")


def _array_from_trace(row: Mapping[str, Any], *, value_field: str, dtype_field: str) -> np.ndarray:
    value = row.get(value_field)
    if value is None:
        raise _schema_invalid(f"missing {value_field}")
    dtype = row.get(dtype_field)
    if dtype:
        try:
            return np.asarray(value, dtype=np.dtype(str(dtype)))
        except Exception as exc:
            raise _schema_invalid(f"{value_field} dtype decode failed: {exc}") from exc
    return np.asarray(value)


def _require_trace_content_hashes(record: Mapping[str, Any], *, step: int) -> None:
    raw_action = _array_from_trace(record, value_field="raw_action", dtype_field="raw_action_dtype")
    env_action = _array_from_trace(record, value_field="env_action", dtype_field="env_action_dtype")
    if hash_array(raw_action) != str(record["raw_action_sha256"]):
        raise _schema_invalid(f"step {step} raw_action_sha256 does not bind raw_action bytes")
    if hash_array(env_action) != str(record["env_action_sha256"]):
        raise _schema_invalid(f"step {step} env_action_sha256 does not bind env_action bytes")
    tokens = [int(x) for x in record["tokens"]]
    if len(tokens) != 7:
        raise _schema_invalid(f"step {step} tokens must have exactly 7 ids")
    if hash_jsonable(tokens) != str(record["tokens_sha256"]):
        raise _schema_invalid(f"step {step} tokens_sha256 does not bind token sequence")


def _require_hash_match(
    *,
    actual: Mapping[str, str],
    expected: Mapping[str, Any],
    actual_field: str,
    expected_field: str,
    step: int,
    phase: str,
    failure_class: str,
) -> None:
    expected_value = expected.get(expected_field, "")
    if not expected_value:
        return
    actual_value = actual.get(actual_field, "")
    if actual_value != expected_value:
        raise PrefixReplayDivergence(
            json.dumps(
                {
                    "failure_class": failure_class,
                    "first_divergence_step": int(step),
                    "first_divergence_phase": phase,
                    "first_divergence_field": expected_field,
                    "reference_sha256": expected_value,
                    "replay_sha256": actual_value,
                },
                sort_keys=True,
            )
        )


def _step_env_action_without_double_postprocess(env: Any, env_action: Sequence[float]) -> tuple[Any, float, bool, Mapping[str, Any]]:
    step_env_action = getattr(env, "step_env_action", None)
    if callable(step_env_action):
        return step_env_action(env_action)
    raise InfraInvalidError("C3 requires env adapter with step_env_action; no fallback to step() is allowed")


def _student_emit_status(student: Any, *, branch_step: int) -> dict[str, Any]:
    state = capture_student_state(student, strict=True)
    emitted = state.get("detector_emitted")
    emit_step = state.get("detector_emit_step")
    if emitted is None:
        emitted = str(state.get("state", "")).upper() == "EMITTED"
    if emit_step is None:
        emit_step = state.get("emit_step", branch_step)
    return {
        "state": state,
        "emitted": bool(emitted),
        "emit_step": int(emit_step),
    }


def run_exact_action_prefix_replay_from_trace(
    *,
    env: Any,
    student: Any,
    policy: Any,
    initial_obs: Any,
    prefix_steps: Sequence[Mapping[str, Any]],
    branch_step: int,
    expected_branch_action: Any,
    expected_branch_tokens: Sequence[int],
    expected_branch_env_action: Any,
    expected_prefix_trace_sha256: str | None = None,
    branch_reference: Mapping[str, Any] | None = None,
    expected_next_observations: Sequence[Mapping[str, Any]] | None = None,
    observation_drift_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Replay a recorded action prefix and execute one exact branch action.

    The prefix path never calls policy.act. It uses recorded raw action/tokens
    only for Student reconstruction, and recorded env_action for env.step.
    """

    if int(branch_step) <= 0:
        raise PrefixReplayDivergence("branch_step must be positive")
    if len(prefix_steps) != int(branch_step):
        raise PrefixReplayDivergence(f"prefix_steps must contain exactly branch_step rows ({branch_step})")
    if expected_branch_env_action is None:
        raise _schema_invalid("expected_branch_env_action is required")
    if expected_prefix_trace_sha256 is not None and sha256_jsonable([dict(row) for row in prefix_steps]) != expected_prefix_trace_sha256:
        raise _schema_invalid("prefix trace SHA does not bind prefix_steps")
    obs = copy.deepcopy(initial_obs)
    replay_rows: list[dict[str, Any]] = []
    for step, record in enumerate(prefix_steps):
        _require_fields(record, REQUIRED_PREFIX_STEP_FIELDS, context=f"prefix step {step}")
        if int(record.get("step", -1)) != step:
            raise PrefixReplayDivergence(f"prefix step order mismatch at index {step}")
        _require_trace_content_hashes(record, step=step)
        if bool(record["done"]):
            raise PrefixReplayDivergence(f"PREFIX_REPLAY_EARLY_DONE at step {step}")
        pre = prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
        for field in (
            "observation_sha256",
            "policy_input_sha256",
            "qpos_sha256",
            "qvel_sha256",
            "flat_sim_state_sha256",
            "student_state_sha256",
            "feature_history_sha256",
        ):
            _require_hash_match(
                actual=pre,
                expected=record,
                actual_field=field,
                expected_field=field,
                step=step,
                phase="pre_step",
                failure_class="PREFIX_REPLAY_PRE_STEP_DIVERGENCE",
            )
        raw_action = _first_present(record, "raw_action", "action")
        env_action = _first_present(record, "env_action")
        tokens = _first_present(record, "tokens")
        if raw_action is None or env_action is None or tokens is None:
            raise PrefixReplayDivergence(f"prefix step {step} missing raw_action/env_action/tokens")
        assert_tokens_exact(tokens, record.get("tokens", tokens), name=f"prefix_step_{step}_tokens")
        update_student_for_step(student, step=step, obs=obs, action=raw_action, tokens=tokens)
        obs_next, reward, done, info = _step_env_action_without_double_postprocess(env, env_action)
        post = prefix_replay_state_hashes(env=env, obs=obs_next, student=student, policy=policy)
        for actual_field, expected_field in (
            ("qpos_sha256", "post_qpos_sha256"),
            ("qvel_sha256", "post_qvel_sha256"),
            ("flat_sim_state_sha256", "post_flat_sim_state_sha256"),
            ("student_state_sha256", "post_student_state_sha256"),
            ("feature_history_sha256", "post_feature_history_sha256"),
        ):
            _require_hash_match(
                actual=post,
                expected=record,
                actual_field=actual_field,
                expected_field=expected_field,
                step=step,
                phase="post_step",
                failure_class="PREFIX_REPLAY_POST_STEP_DIVERGENCE",
            )
        if post["observation_sha256"] != record["next_observation_sha256"]:
            if (
                expected_next_observations is not None
                and step < len(expected_next_observations)
                and observation_drift_output_dir is not None
            ):
                drift_dir = observation_drift_output_dir / f"step_{step:04d}"
                expected_obs = expected_next_observations[step]
                write_dict_csv(
                    drift_dir / "observation_field_diff.csv",
                    observation_diff_rows(expected_obs, obs_next),
                )
                image_summary = write_agentview_diff_artifacts(
                    drift_dir,
                    prefix=expected_obs,
                    restored=obs_next,
                    label="reference_vs_replay",
                )
                write_json(
                    drift_dir / "observation_drift_summary.json",
                    {
                        "step": int(step),
                        "expected_observation_sha256": record["next_observation_sha256"],
                        "actual_observation_sha256": post["observation_sha256"],
                        "agentview": image_summary,
                    },
                )
            _require_hash_match(
                actual=post,
                expected=record,
                actual_field="observation_sha256",
                expected_field="next_observation_sha256",
                step=step,
                phase="post_step",
                failure_class="PREFIX_REPLAY_POST_STEP_DIVERGENCE",
            )
        if "reward" in record and record["reward"] is not None and float(record["reward"]) != float(reward):
            raise PrefixReplayDivergence(f"PREFIX_REPLAY_REWARD_MISMATCH at step {step}")
        if bool(done):
            raise PrefixReplayDivergence(f"PREFIX_REPLAY_EARLY_DONE at step {step}")
        if "done" in record and bool(record["done"]) != bool(done):
            raise PrefixReplayDivergence(f"PREFIX_REPLAY_EARLY_DONE at step {step}")
        replay_rows.append(
            {
                "step": step,
                "pre_observation_sha256": pre["observation_sha256"],
                "post_observation_sha256": post["observation_sha256"],
                "reward": float(reward),
                "done": bool(done),
                "info": _json_clone(info),
            }
        )
        obs = obs_next

    branch_reference = dict(branch_reference or {})
    _require_fields(branch_reference, REQUIRED_BRANCH_REFERENCE_FIELDS, context="branch_reference")
    branch_pre = prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
    for field in (
        "observation_sha256",
        "policy_input_sha256",
        "qpos_sha256",
        "qvel_sha256",
        "flat_sim_state_sha256",
        "student_state_sha256",
        "feature_history_sha256",
    ):
        _require_hash_match(
            actual=branch_pre,
            expected=branch_reference,
            actual_field=field,
            expected_field=field,
            step=int(branch_step),
            phase="branch_boundary",
            failure_class="PREFIX_REPLAY_POLICY_INPUT_MISMATCH"
            if field == "policy_input_sha256"
            else "PREFIX_REPLAY_PRE_STEP_DIVERGENCE",
        )

    branch_action, branch_tokens = policy.act(obs)
    token_report = assert_tokens_exact(branch_tokens, expected_branch_tokens, name="branch_action_tokens")
    action_report = assert_array_exact(branch_action, expected_branch_action, name="branch_raw_action")
    branch_env_action = postprocess_openvla_action_for_libero(branch_action)
    env_action_report = assert_array_exact(branch_env_action, expected_branch_env_action, name="branch_env_action")
    update_student_for_step(student, step=int(branch_step), obs=obs, action=branch_action, tokens=branch_tokens)
    branch_post_update = prefix_replay_state_hashes(env=env, obs=obs, student=student, policy=policy)
    _require_hash_match(
        actual=branch_post_update,
        expected=branch_reference,
        actual_field="student_state_sha256",
        expected_field="branch_post_student_update_state_sha256",
        step=int(branch_step),
        phase="branch_student_update",
        failure_class="PREFIX_REPLAY_STUDENT_DIVERGENCE",
    )
    _require_hash_match(
        actual=branch_post_update,
        expected=branch_reference,
        actual_field="feature_history_sha256",
        expected_field="branch_post_student_update_feature_history_sha256",
        step=int(branch_step),
        phase="branch_student_update",
        failure_class="PREFIX_REPLAY_STUDENT_DIVERGENCE",
    )
    emit_status = _student_emit_status(student, branch_step=int(branch_step))
    if not emit_status["emitted"] or int(emit_status["emit_step"]) != int(branch_step):
        raise PrefixReplayDivergence(
            json.dumps(
                {
                    "failure_class": "PREFIX_REPLAY_STUDENT_DIVERGENCE",
                    "first_divergence_step": int(branch_step),
                    "first_divergence_phase": "branch_student_update",
                    "first_divergence_field": "detector_emit_step",
                    "expected": int(branch_step),
                    "actual": emit_status,
                },
                sort_keys=True,
            )
        )
    obs_52, reward_51, done_51, info_51 = _step_env_action_without_double_postprocess(env, branch_env_action)
    post_branch = prefix_replay_state_hashes(env=env, obs=obs_52, student=student, policy=policy)
    for actual_field, expected_field in (
        ("qpos_sha256", "post_branch_qpos_sha256"),
        ("qvel_sha256", "post_branch_qvel_sha256"),
        ("flat_sim_state_sha256", "post_branch_flat_sim_state_sha256"),
        ("observation_sha256", "post_branch_observation_sha256"),
    ):
        _require_hash_match(
            actual=post_branch,
            expected=branch_reference,
            actual_field=actual_field,
            expected_field=expected_field,
            step=int(branch_step),
            phase="post_branch",
            failure_class="POST_BRANCH_TRANSITION_MISMATCH",
        )
    if "post_branch_reward" in branch_reference and float(branch_reference["post_branch_reward"]) != float(reward_51):
        raise PrefixReplayDivergence("POST_BRANCH_TRANSITION_MISMATCH reward")
    if "post_branch_done" in branch_reference and bool(branch_reference["post_branch_done"]) != bool(done_51):
        raise PrefixReplayDivergence("POST_BRANCH_TRANSITION_MISMATCH done")
    post_branch_diff_rows = []
    for actual_field, expected_field in (
        ("qpos_sha256", "post_branch_qpos_sha256"),
        ("qvel_sha256", "post_branch_qvel_sha256"),
        ("flat_sim_state_sha256", "post_branch_flat_sim_state_sha256"),
        ("observation_sha256", "post_branch_observation_sha256"),
    ):
        actual_value = post_branch.get(actual_field, "")
        expected_value = branch_reference.get(expected_field, "")
        post_branch_diff_rows.append(
            {
                "field": actual_field,
                "expected_field": expected_field,
                "actual": actual_value,
                "expected": expected_value,
                "exact": bool(actual_value == expected_value),
            }
        )
    return {
        "stage": "C3_EXACT_ACTION_PREFIX_REPLAY_ONE_STEP",
        "result": "PASS",
        "prefix_steps_completed": len(replay_rows),
        "branch_step": int(branch_step),
        "first_divergence": None,
        "branch_action_tokens_exact": bool(token_report["exact"]),
        "branch_action_exact": bool(action_report["exact"]),
        "branch_env_action_exact": bool(env_action_report["exact"]),
        "branch_student_emit_exact": True,
        "post_branch_qpos_exact": post_branch.get("qpos_sha256") == branch_reference.get("post_branch_qpos_sha256", post_branch.get("qpos_sha256")),
        "post_branch_qvel_exact": post_branch.get("qvel_sha256") == branch_reference.get("post_branch_qvel_sha256", post_branch.get("qvel_sha256")),
        "post_branch_sim_state_exact": post_branch.get("flat_sim_state_sha256")
        == branch_reference.get("post_branch_flat_sim_state_sha256", post_branch.get("flat_sim_state_sha256")),
        "replay_prefix_rows": replay_rows,
        "branch_action_exactness": {
            "tokens": token_report,
            "raw_action": action_report,
            "env_action": env_action_report,
            "student_emit_exact": True,
        },
        "post_branch_diff_rows": post_branch_diff_rows,
    }


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

    def policy_input_stages(self, obs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not isinstance(obs, Mapping) or "agentview_image" not in obs:
            raise ExactRestoreError("observation missing agentview_image")
        from gripper_attack.openvla_preprocess import prepare_openvla_image
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
        input_ids = inputs.get("input_ids")
        if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
            inputs["input_ids"] = torch.cat(
                (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0)),
                dim=1,
            )
        return raw, np.asarray(proc_image).copy(), inputs

    def policy_input_fingerprint(self, obs: Any) -> dict[str, Any]:
        raw, prepared, inputs = self.policy_input_stages(obs)
        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        out: dict[str, Any] = {
            "raw_agentview_sha256": hash_array(raw),
            "raw_agentview_shape": list(raw.shape),
            "raw_agentview_dtype": str(raw.dtype),
            "prepared_image_sha256": hash_array(prepared),
            "prepared_image_shape": list(prepared.shape),
            "prompt": openvla_prompt(self.instruction.lower()),
        }
        if input_ids is not None:
            out["input_ids_sha256"] = hash_array(input_ids.detach().cpu().numpy())
            out["input_ids_shape"] = list(input_ids.shape)
        if pixel_values is not None:
            out["pixel_values_sha256"] = hash_array(pixel_values.detach().cpu().numpy())
            out["pixel_values_shape"] = list(pixel_values.shape)
            out["pixel_values_dtype"] = str(pixel_values.dtype)
        return out

    def act(self, obs: Any) -> tuple[Sequence[float], Sequence[int]]:
        from gripper_attack.v3_generation_parity import extract_exact_new_tokens
        import torch

        _raw, _prepared, inputs = self.policy_input_stages(obs)
        for key, val in list(inputs.items()):
            if torch.is_tensor(val):
                if torch.is_floating_point(val):
                    inputs[key] = val.to(device=self.device, dtype=model_float_dtype(self.model))
                else:
                    inputs[key] = val.to(device=self.device)
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


def release_real_policy(policy: Any | None) -> None:
    if policy is not None:
        if hasattr(policy, "model"):
            policy.model = None
        if hasattr(policy, "processor"):
            policy.processor = None
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class RealLiberoEnvAdapter:
    def __init__(self, env: Any):
        self.env = env
        self.sim = env.sim
        self.frames: list[np.ndarray] = []

    def get_internal_state(self) -> dict[str, Any]:
        inner = getattr(self.env, "env", self.env)
        sim_flat = np.asarray(self.env.get_sim_state()).copy()
        state: dict[str, Any] = {
            "sim_flat_state_sha256": hash_array(sim_flat),
            "sim_flat_state_shape": list(sim_flat.shape),
            "sim_flat_state_dtype": str(sim_flat.dtype),
            "sim_flat_state_values": sim_flat.tolist(),
        }
        for name in ("timestep", "_timestep", "cur_time", "_elapsed_steps", "done"):
            if hasattr(inner, name):
                value = getattr(inner, name)
                if isinstance(value, (int, float, bool, str)) or value is None:
                    state[name] = value
        return _json_clone(state)

    def set_internal_state(self, state: Mapping[str, Any]) -> None:
        inner = getattr(self.env, "env", self.env)
        for name, value in state.items():
            if name in {"sim_flat_state_sha256", "sim_flat_state_shape", "sim_flat_state_dtype", "sim_flat_state_values"}:
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
        return self.step_env_action(env_action)

    def step_env_action(self, env_action: Sequence[float]) -> tuple[Any, float, bool, Mapping[str, Any]]:
        """Step using an already postprocessed LIBERO env action.

        C3 exact action-prefix replay must not run postprocess twice. The
        original continuous rollout records postprocessed env actions; fresh
        replay sends those bytes directly to the environment.
        """

        obs, reward, done, info = self.env.step(env_action)
        if isinstance(obs, Mapping) and "agentview_image" in obs:
            self.frames.append(np.asarray(obs["agentview_image"]).copy())
        return obs, reward, done, info

    def close(self) -> None:
        self.env.close()


class RealSC5StudentAdapter:
    def __init__(self, *, detector: Any, streamer: Any, env_adapter: RealLiberoEnvAdapter):
        from gripper_attack.sc5_online_feature_state import initialize_sc5_prev_eef

        self.detector = detector
        self.streamer = streamer
        self.env_adapter = env_adapter
        self.prev_eef: tuple[float, float, float] | None = initialize_sc5_prev_eef(env_adapter.env)
        self.invalid_steps = 0
        self.scan_telemetry: list[dict[str, Any]] = []

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
        from gripper_attack.sc5_online_feature_state import extract_sc5_physical_state

        env_action = postprocess_openvla_action_for_libero(action)
        raw_grip = float(action[-1])
        env_grip = float(env_action[-1])
        phys = extract_sc5_physical_state(env=self.env_adapter.env, obs=obs, prev_eef=self.prev_eef)
        self.prev_eef = phys.next_prev_eef
        feat_res = self.streamer.update(
            step_id=int(step),
            raw_gripper=raw_grip,
            env_gripper=env_grip,
            gripper_qpos=phys.gripper_qpos,
            gripper_opening_proxy=phys.gripper_opening_proxy,
            eef_x=phys.eef_x,
            eef_y=phys.eef_y,
            eef_z=phys.eef_z,
            eef_vx=phys.eef_vx,
            eef_vy=phys.eef_vy,
            eef_vz=phys.eef_vz,
            action_dx=float(action[0]),
            action_dy=float(action[1]),
            action_dz=float(action[2]),
            action_gripper=raw_grip,
        )
        row: dict[str, Any] = {
            "step": int(step),
            "raw_gripper": raw_grip,
            "env_gripper": env_grip,
            "action_dx": float(action[0]),
            "action_dy": float(action[1]),
            "action_dz": float(action[2]),
            "action_gripper": raw_grip,
            "exact_new_tokens_json": json.dumps([int(t) for t in tokens]),
            "feat_valid": bool(feat_res.get("valid", False)),
            "feat_error": str(feat_res.get("error", "")),
            "detector_state_before": self.detector.state,
            "detector_emit_step_before": int(self.detector.emit_step),
            "detector_emitted_before": bool(self.detector.emitted),
            **phys.as_dict(),
        }
        if not bool(feat_res.get("valid", False)):
            self.invalid_steps += 1
            row.update(
                {
                    "detector_state_after": self.detector.state,
                    "detector_emit_step_after": int(self.detector.emit_step),
                    "detector_emitted_after": bool(self.detector.emitted),
                    "corridor_p": "",
                    "release_p": "",
                    "pred_phase": "",
                }
            )
            self.scan_telemetry.append(row)
            return
        features = dict(feat_res["features"])
        decision = self.detector.update(features, int(step))
        row.update(
            {
                "detector_state_after": decision["state"],
                "detector_emit_step_after": int(decision["emit_step"]),
                "detector_emitted_after": bool(decision["emitted"]),
                "corridor_p": decision.get("corridor_p"),
                "release_p": decision.get("release_p"),
                "pred_phase": decision.get("pred_phase"),
            }
        )
        for name, value in features.items():
            row["f_" + name] = value
        self.scan_telemetry.append(row)


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def compare_policy_input_fingerprints(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        rows.append(
            {
                "key": key,
                "left": left.get(key, ""),
                "right": right.get(key, ""),
                "match": left.get(key) == right.get(key),
            }
        )
    return rows


def save_typed_prefix_observation_artifacts(
    output_dir: Path,
    *,
    snapshot: ExactRestoreSnapshotPayload,
    policy: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = clone_typed_observation(snapshot.observation)
    manifest: dict[str, Any] = {
        "branch_input_source": "CAPTURED_PREFIX_OBSERVATION",
        "captured_prefix_observation_sha256": hash_typed_observation(obs),
        "prefix_snapshot_observation_sha256": snapshot.prefix.observation_sha256,
        "typed_observation_manifest": typed_value_manifest(obs),
    }
    if manifest["captured_prefix_observation_sha256"] != snapshot.prefix.observation_sha256:
        raise ExactRestoreError("typed prefix observation hash does not match snapshot")
    if isinstance(obs, Mapping) and "agentview_image" in obs:
        agent = np.asarray(obs["agentview_image"])
        npy_path = output_dir / "prefix_agentview.npy"
        np.save(npy_path, agent)
        loaded = np.load(npy_path, allow_pickle=False)
        if loaded.shape != agent.shape or loaded.dtype != agent.dtype or hash_array(loaded) != hash_array(agent):
            raise ExactRestoreError("prefix agentview npy round-trip mismatch")
        manifest.update(
            {
                "prefix_agentview_npy": npy_path.name,
                "prefix_agentview_shape": list(agent.shape),
                "prefix_agentview_dtype": str(agent.dtype),
                "prefix_agentview_sha256": hash_array(agent),
                "prefix_agentview_roundtrip_sha256": hash_array(loaded),
                "prefix_agentview_roundtrip_exact": True,
            }
        )
        try:
            from PIL import Image

            Image.fromarray(np.clip(agent, 0, 255).astype(np.uint8)).save(output_dir / "prefix_agentview.png")
            manifest["prefix_agentview_png"] = "prefix_agentview.png"
        except Exception as exc:
            manifest["prefix_agentview_png_error"] = f"{type(exc).__name__}:{exc}"
    npz_arrays: dict[str, np.ndarray] = {}
    if isinstance(obs, Mapping):
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                safe_key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(key))
                npz_arrays[safe_key] = value
    if npz_arrays:
        np.savez_compressed(output_dir / "prefix_observation_diagnostic.npz", **npz_arrays)
        manifest["prefix_observation_diagnostic_npz"] = "prefix_observation_diagnostic.npz"
        manifest["prefix_observation_diagnostic_keys"] = sorted(npz_arrays)
    if policy is not None and hasattr(policy, "policy_input_fingerprint"):
        policy_fp = policy.policy_input_fingerprint(obs)
        write_json(output_dir / "prefix_policy_input_manifest.json", policy_fp)
        manifest["captured_policy_input_sha256"] = hash_jsonable(policy_fp)
        manifest["prefix_policy_input_manifest"] = "prefix_policy_input_manifest.json"
    write_json(output_dir / "prefix_typed_observation_manifest.json", manifest)
    return manifest


def run_single_observation_diff(
    *,
    output_dir: Path,
    label: str,
    prefix_obs: Mapping[str, Any],
    candidate_obs: Mapping[str, Any],
    policy: RealOpenVLAPolicyAdapter,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_rows = observation_diff_rows(prefix_obs, candidate_obs)
    write_dict_csv(output_dir / f"{label}_observation_field_diff.csv", obs_rows)
    image_summary = write_agentview_diff_artifacts(output_dir, prefix=prefix_obs, restored=candidate_obs, label=label)
    prefix_policy = policy.policy_input_fingerprint(prefix_obs)
    candidate_policy = policy.policy_input_fingerprint(candidate_obs)
    policy_rows = compare_policy_input_fingerprints(prefix_policy, candidate_policy)
    write_dict_csv(output_dir / f"{label}_policy_input_diff.csv", policy_rows)
    summary = {
        "label": label,
        "prefix_observation_sha256": hash_typed_observation(prefix_obs),
        "candidate_observation_sha256": hash_typed_observation(candidate_obs),
        "observation_hash_match": hash_typed_observation(prefix_obs) == hash_typed_observation(candidate_obs),
        "field_mismatch_count": sum(1 for r in obs_rows if not r.get("sha_match", False)),
        "policy_input_mismatch_count": sum(1 for r in policy_rows if not r.get("match", False)),
        "image": image_summary,
    }
    write_json(output_dir / f"{label}_summary.json", summary)
    return summary


def run_observation_reconstruction_audit(
    *,
    args: argparse.Namespace,
    selected: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    prefix_obs = clone_typed_observation(snapshot.observation)
    env_adapter: RealLiberoEnvAdapter = selected["env_adapter"]
    policy: RealOpenVLAPolicyAdapter = selected["policy"]
    student: RealSC5StudentAdapter = selected["student"]
    summaries: list[dict[str, Any]] = []

    sim_before_o1 = hash_array(env_adapter.env.get_sim_state())
    same_env_obs = env_adapter.get_observation_after_restore()
    sim_after_o1 = hash_array(env_adapter.env.get_sim_state())
    o1 = run_single_observation_diff(
        output_dir=output_dir,
        label="O1_same_env_recapture_no_restore",
        prefix_obs=prefix_obs,
        candidate_obs=same_env_obs,
        policy=policy,
    )
    o1["sim_state_before_sha256"] = sim_before_o1
    o1["sim_state_after_sha256"] = sim_after_o1
    o1["sim_state_changed_by_recapture"] = sim_before_o1 != sim_after_o1
    write_json(output_dir / "O1_same_env_recapture_no_restore_summary.json", o1)
    summaries.append(o1)

    # O2: perturb the current env by one clean step, then restore snapshot in the same env.
    try:
        env_adapter.step(snapshot.clean_action_t)
    except Exception as exc:
        write_json(output_dir / "O2_same_env_restore_error.json", {"stage": "perturb_step", "error": f"{type(exc).__name__}:{exc}"})
    restore_snapshot(env_adapter, student, snapshot, policy)
    same_env_restored_obs = env_adapter.get_observation_after_restore()
    o2 = run_single_observation_diff(
        output_dir=output_dir,
        label="O2_same_env_restore",
        prefix_obs=prefix_obs,
        candidate_obs=same_env_restored_obs,
        policy=policy,
    )
    o2["restored_flat_sim_state_sha256"] = hash_array(env_adapter.env.get_sim_state())
    o2["prefix_flat_sim_state_sha256"] = selected.get("prefix_flat_sim_state_sha256", "")
    summaries.append(o2)

    # The fresh-env path loads a second OpenVLA instance. Release the prefix
    # scan model before O3 so the audit does not fail from avoidable GPU OOM.
    try:
        env_adapter.close()
    except Exception:
        pass
    release_real_policy(policy)
    selected["policy"] = None
    selected["student"] = None
    selected["env_adapter"] = None

    # O3: fresh env restore, matching the formal path.
    fresh_env, fresh_policy, fresh_student = new_env_policy_student_for_snapshot(args, selected)
    restore_snapshot(fresh_env, fresh_student, snapshot, fresh_policy)
    fresh_obs = fresh_env.get_observation_after_restore()
    o3 = run_single_observation_diff(
        output_dir=output_dir,
        label="O3_fresh_env_restore",
        prefix_obs=prefix_obs,
        candidate_obs=fresh_obs,
        policy=fresh_policy,
    )
    o3["restored_flat_sim_state_sha256"] = hash_array(fresh_env.env.get_sim_state())
    summaries.append(o3)
    fresh_env.close()
    release_real_policy(fresh_policy)

    summary = {
        "stage": "OBSERVATION_RECONSTRUCTION_AUDIT_O1_O2_O3",
        "suite": snapshot.parent_manifest.suite,
        "task_idx": snapshot.parent_manifest.task_idx,
        "state_id": snapshot.parent_manifest.state_id,
        "emit_step": snapshot.prefix.emit_step,
        "summaries": summaries,
        "all_observation_hash_match": all(s.get("observation_hash_match") for s in summaries),
        "all_policy_inputs_match": all(s.get("policy_input_mismatch_count") == 0 for s in summaries),
    }
    write_json(output_dir / "observation_reconstruction_audit_summary.json", summary)
    return summary


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
        policy, student, _model, detector = load_real_policy_and_student(
            args, env_adapter=env_adapter, instruction=instruction
        )
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
        prefix_trace: list[dict[str, Any]] = []
        prefix_next_observations: list[dict[str, Any]] = []
        for step in range(int(args.max_steps)):
            pre_hashes = prefix_replay_state_hashes(env=env_adapter, obs=obs, student=student, policy=policy)
            action, tokens = policy.act(obs)
            raw_action_arr = np.asarray(action)
            env_action_arr = postprocess_openvla_action_for_libero(action)
            update_student_for_step(student, step=step, obs=obs, action=action, tokens=tokens)
            post_student_update_hashes = prefix_replay_state_hashes(
                env=env_adapter, obs=obs, student=student, policy=policy
            )
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
                    "prefix_flat_sim_state_sha256": hash_array(env.get_sim_state()),
                    "prefix_trace": prefix_trace,
                    "prefix_next_observations": prefix_next_observations,
                    "branch_pre_hashes": pre_hashes,
                    "branch_post_student_update_hashes": post_student_update_hashes,
                }
                break
            obs_next, reward, done, info = env_adapter.step_env_action(env_action_arr)
            post_hashes = prefix_replay_state_hashes(env=env_adapter, obs=obs_next, student=student, policy=policy)
            prefix_trace.append(
                {
                    "step": int(step),
                    "raw_action": raw_action_arr.tolist(),
                    "raw_action_dtype": str(raw_action_arr.dtype),
                    "raw_action_sha256": hash_array(raw_action_arr),
                    "env_action": env_action_arr.tolist(),
                    "env_action_dtype": str(env_action_arr.dtype),
                    "env_action_sha256": hash_array(env_action_arr),
                    "tokens": [int(x) for x in tokens],
                    "tokens_sha256": hash_jsonable([int(x) for x in tokens]),
                    **pre_hashes,
                    "post_qpos_sha256": post_hashes["qpos_sha256"],
                    "post_qvel_sha256": post_hashes["qvel_sha256"],
                    "post_flat_sim_state_sha256": post_hashes["flat_sim_state_sha256"],
                    "next_observation_sha256": post_hashes["observation_sha256"],
                    "post_student_state_sha256": post_hashes["student_state_sha256"],
                    "post_feature_history_sha256": post_hashes["feature_history_sha256"],
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(info.get("success", False)) if isinstance(info, Mapping) else False,
                }
            )
            prefix_next_observations.append(clone_typed_observation(obs_next))
            obs = obs_next
            if done:
                break
        telemetry_name = (
            f"candidate_scan_task{int(candidate['task_idx'])}_state{int(candidate['state_id'])}_"
            f"rep{int(repetition)}.csv"
        )
        write_dict_csv(attempt_dir / telemetry_name, getattr(student, "scan_telemetry", []))
        if selected is None:
            raise NoNaturalStudentEmit("candidate did not produce eligible natural Student emit")
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


def run_transition_state_audit(
    args: argparse.Namespace,
    selected: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    validate_transition_state_audit_known_parent(snapshot)

    env_adapter: RealLiberoEnvAdapter = selected["env_adapter"]
    policy: RealOpenVLAPolicyAdapter = selected["policy"]
    student: RealSC5StudentAdapter = selected["student"]
    obs_t = clone_typed_observation(selected["obs_t"])

    write_json(output_dir / "parent_dependency_manifest.json", asdict(selected["parent"]))
    write_json(output_dir / "runtime_receipt.json", asdict(selected["runtime"]))
    write_json(output_dir / "dependency_sha_receipt.json", selected["dependency"])
    write_json(output_dir / "prefix_snapshot_manifest.json", asdict(snapshot.prefix))
    typed_manifest = save_typed_prefix_observation_artifacts(
        output_dir / "captured_prefix",
        snapshot=snapshot,
        policy=policy,
    )

    reference_action, reference_tokens = policy.act(obs_t)
    if [int(x) for x in reference_tokens] != [int(x) for x in snapshot.clean_tokens_t]:
        raise ExactRestoreError("transition-state audit first token mismatch before reference step")
    require_action_byte_exact(
        reference_action,
        snapshot.clean_action_t,
        context="transition-state audit reference first",
    )

    reference_pre = capture_transition_state(
        phase="PRE_STEP",
        env_adapter=env_adapter,
        student=student,
        policy=policy,
        obs=obs_t,
        action=snapshot.clean_action_t,
        tokens=snapshot.clean_tokens_t,
    )
    reference_next_obs, reference_reward, reference_done, reference_info = env_adapter.step(snapshot.clean_action_t)
    reference_post = capture_transition_state(
        phase="POST_STEP",
        env_adapter=env_adapter,
        student=student,
        policy=policy,
        obs=reference_next_obs,
        action=snapshot.clean_action_t,
        tokens=snapshot.clean_tokens_t,
    )
    reference_step = {
        "reward": float(reference_reward),
        "done": bool(reference_done),
        "info": _json_clone(reference_info),
        "next_observation_sha256": hash_typed_observation(reference_next_obs),
    }
    env_adapter.close()
    release_real_policy(policy)
    selected["env_adapter"] = None
    selected["policy"] = None
    selected["student"] = None

    replay_env, replay_policy, replay_student = new_env_policy_student_for_snapshot(args, selected)
    try:
        restore_snapshot(replay_env, replay_student, snapshot, replay_policy)
        replay_obs = clone_typed_observation(snapshot.observation)
        replay_action, replay_tokens = replay_policy.act(replay_obs)
        if [int(x) for x in replay_tokens] != [int(x) for x in snapshot.clean_tokens_t]:
            raise ExactRestoreError("transition-state audit first token mismatch before replay step")
        require_action_byte_exact(
            replay_action,
            snapshot.clean_action_t,
            context="transition-state audit replay first",
        )

        replay_pre = capture_transition_state(
            phase="PRE_STEP",
            env_adapter=replay_env,
            student=replay_student,
            policy=replay_policy,
            obs=replay_obs,
            action=snapshot.clean_action_t,
            tokens=snapshot.clean_tokens_t,
        )
        replay_next_obs, replay_reward, replay_done, replay_info = replay_env.step(snapshot.clean_action_t)
        replay_post = capture_transition_state(
            phase="POST_STEP",
            env_adapter=replay_env,
            student=replay_student,
            policy=replay_policy,
            obs=replay_next_obs,
            action=snapshot.clean_action_t,
            tokens=snapshot.clean_tokens_t,
        )
        replay_step = {
            "reward": float(replay_reward),
            "done": bool(replay_done),
            "info": _json_clone(replay_info),
            "next_observation_sha256": hash_typed_observation(replay_next_obs),
        }
    finally:
        try:
            replay_env.close()
        except Exception:
            pass
        release_real_policy(replay_policy)

    pre_diff = annotate_transition_diffs(diff_state_dicts(reference_pre, replay_pre))
    post_diff = annotate_transition_diffs(diff_state_dicts(reference_post, replay_post))
    first = first_transition_diff(pre_diff, post_diff)
    summary = {
        "stage": "TRANSITION_STATE_AUDIT_ONLY",
        "result": "TRANSITION_STATE_AUDIT_COMPLETE",
        "forbidden_paths_not_run": [
            "formal_restore_3x",
            "R2",
            "VIS",
            "RAND",
            "SHUFFLED",
            "ORACLE",
            "ATTACK",
            "A800_FORMAL",
        ],
        "suite": snapshot.parent_manifest.suite,
        "task_idx": int(snapshot.parent_manifest.task_idx),
        "state_id": int(snapshot.parent_manifest.state_id),
        "eval_seed": int(snapshot.parent_manifest.eval_seed),
        "emit_step": int(snapshot.prefix.emit_step),
        "prefix_snapshot_sha256": snapshot.prefix.snapshot_sha256,
        "snapshot_payload_sha256": snapshot.payload_sha256,
        "typed_prefix_manifest_sha256": hash_jsonable(typed_manifest),
        "first_action_tokens_exact": True,
        "first_action_exact": True,
        "reference_step": reference_step,
        "replay_step": replay_step,
        "pre_diff_count": len(pre_diff),
        "post_diff_count": len(post_diff),
        "pre_diff_classification_counts": transition_classification_counts(pre_diff),
        "post_diff_classification_counts": transition_classification_counts(post_diff),
        "first_divergence_phase": first.get("first_divergence_phase"),
        "first_divergence_field": first.get("field"),
        "transition_state_classification": first.get("classification"),
        "first_divergence_reference_sha256": first.get("reference_sha256", ""),
        "first_divergence_replay_sha256": first.get("replay_sha256", ""),
    }
    write_json(output_dir / "transition_state_reference_pre.json", reference_pre)
    write_json(output_dir / "transition_state_replay_pre.json", replay_pre)
    write_json(output_dir / "transition_state_reference_post.json", reference_post)
    write_json(output_dir / "transition_state_replay_post.json", replay_post)
    write_dict_csv(output_dir / "transition_state_pre_diff.csv", pre_diff)
    write_dict_csv(output_dir / "transition_state_post_diff.csv", post_diff)
    write_json(output_dir / "transition_state_audit_summary.json", summary)
    seal = write_recursive_manifest(output_dir)
    summary["recursive_sha256_manifest_sha256"] = seal
    write_json(output_dir / "transition_state_audit_summary.json", summary)
    return summary


C2_ABLATIONS = (
    ("A0_BASELINE", {}),
    ("A1_DERIVED_RECOMPUTE", {"refresh_derived": True}),
    ("A2_GOAL_STATE", {"restore_goal": True, "refresh_derived": True}),
    (
        "A3_GOAL_INTERPOLATOR_STATE",
        {"restore_goal": True, "restore_interpolator": True, "refresh_derived": True},
    ),
    (
        "A4_GOAL_INTERPOLATOR_ACTION_HISTORY",
        {
            "restore_goal": True,
            "restore_interpolator": True,
            "restore_action_history": True,
            "refresh_derived": True,
        },
    ),
    (
        "A5_QACC_ABLATION",
        {
            "restore_goal": True,
            "restore_interpolator": True,
            "restore_action_history": True,
            "restore_qacc": True,
            "refresh_derived": True,
        },
    ),
)


def qpos_qvel_gate(reference_post: Mapping[str, Any], replay_post: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("qpos", "qvel"):
        ref = reference_post.get("mujoco", {}).get(name, {})
        rep = replay_post.get("mujoco", {}).get(name, {})
        out[f"{name}_sha_match"] = ref.get("sha256", "") == rep.get("sha256", "")
        ref_vals = ref.get("values")
        rep_vals = rep.get("values")
        if isinstance(ref_vals, list) and isinstance(rep_vals, list) and len(ref_vals) == len(rep_vals):
            diffs = [abs(float(a) - float(b)) for a, b in zip(ref_vals, rep_vals)]
            out[f"{name}_max_abs_diff"] = max(diffs) if diffs else 0.0
            out[f"{name}_nonzero_diff_count"] = sum(1 for value in diffs if value != 0)
        else:
            out[f"{name}_max_abs_diff"] = ""
            out[f"{name}_nonzero_diff_count"] = ""
    out["qpos_qvel_exact"] = bool(out.get("qpos_sha_match") and out.get("qvel_sha_match"))
    return out


def run_control_state_causal_ablation(
    args: argparse.Namespace,
    selected: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    validate_transition_state_audit_known_parent(snapshot)

    env_adapter: RealLiberoEnvAdapter = selected["env_adapter"]
    policy: RealOpenVLAPolicyAdapter = selected["policy"]
    student: RealSC5StudentAdapter = selected["student"]
    obs_t = clone_typed_observation(selected["obs_t"])

    write_json(output_dir / "parent_dependency_manifest.json", asdict(selected["parent"]))
    write_json(output_dir / "runtime_receipt.json", asdict(selected["runtime"]))
    write_json(output_dir / "dependency_sha_receipt.json", selected["dependency"])
    write_json(output_dir / "prefix_snapshot_manifest.json", asdict(snapshot.prefix))
    typed_manifest = save_typed_prefix_observation_artifacts(
        output_dir / "captured_prefix",
        snapshot=snapshot,
        policy=policy,
    )

    reference_action, reference_tokens = policy.act(obs_t)
    if [int(x) for x in reference_tokens] != [int(x) for x in snapshot.clean_tokens_t]:
        raise ExactRestoreError("C2 first token mismatch before reference step")
    reference_action_identity = require_action_byte_exact(
        reference_action,
        snapshot.clean_action_t,
        context="C2 reference first",
    )

    reference_mutable_state = snapshot_control_ablation_state(env_adapter)
    reference_pre = capture_transition_state(
        phase="PRE_STEP",
        env_adapter=env_adapter,
        student=student,
        policy=policy,
        obs=obs_t,
        action=snapshot.clean_action_t,
        tokens=snapshot.clean_tokens_t,
    )
    reference_next_obs, reference_reward, reference_done, reference_info = env_adapter.step(snapshot.clean_action_t)
    reference_post = capture_transition_state(
        phase="POST_STEP",
        env_adapter=env_adapter,
        student=student,
        policy=policy,
        obs=reference_next_obs,
        action=snapshot.clean_action_t,
        tokens=snapshot.clean_tokens_t,
    )
    reference_step = {
        "reward": float(reference_reward),
        "done": bool(reference_done),
        "info": _json_clone(reference_info),
        "next_observation_sha256": hash_typed_observation(reference_next_obs),
    }
    write_json(output_dir / "reference_mutable_control_state.json", compact_state_value(reference_mutable_state, max_depth=5))
    write_json(output_dir / "reference_pre.json", reference_pre)
    write_json(output_dir / "reference_post.json", reference_post)
    env_adapter.close()
    release_real_policy(policy)
    selected["env_adapter"] = None
    selected["policy"] = None
    selected["student"] = None

    rows: list[dict[str, Any]] = []
    for ablation_name, options in C2_ABLATIONS:
        ablation_dir = output_dir / ablation_name
        ablation_dir.mkdir(parents=True, exist_ok=False)
        replay_env, replay_policy, replay_student = new_env_policy_student_for_snapshot(args, selected)
        try:
            restore_snapshot(replay_env, replay_student, snapshot, replay_policy)
            replay_obs = clone_typed_observation(snapshot.observation)
            applied = apply_control_ablation_state(replay_env, reference_mutable_state, **options)
            replay_action, replay_tokens = replay_policy.act(replay_obs)
            tokens_exact = [int(x) for x in replay_tokens] == [int(x) for x in snapshot.clean_tokens_t]
            action_identity = action_identity_report(replay_action, snapshot.clean_action_t)
            action_exact = bool(action_identity["exact"])
            if not tokens_exact or not action_exact:
                raise ExactRestoreError(f"C2 {ablation_name} first action/tokens mismatch")
            replay_pre = capture_transition_state(
                phase="PRE_STEP",
                env_adapter=replay_env,
                student=replay_student,
                policy=replay_policy,
                obs=replay_obs,
                action=snapshot.clean_action_t,
                tokens=snapshot.clean_tokens_t,
            )
            replay_next_obs, replay_reward, replay_done, replay_info = replay_env.step(snapshot.clean_action_t)
            replay_post = capture_transition_state(
                phase="POST_STEP",
                env_adapter=replay_env,
                student=replay_student,
                policy=replay_policy,
                obs=replay_next_obs,
                action=snapshot.clean_action_t,
                tokens=snapshot.clean_tokens_t,
            )
            pre_diff = annotate_transition_diffs(diff_state_dicts(reference_pre, replay_pre))
            post_diff = annotate_transition_diffs(diff_state_dicts(reference_post, replay_post))
            first = first_transition_diff(pre_diff, post_diff)
            gate = qpos_qvel_gate(reference_post, replay_post)
            row = {
                "ablation": ablation_name,
                "tokens_exact": tokens_exact,
                "action_exact": action_exact,
                "action_shape_exact": action_identity["shape_exact"],
                "action_dtype_exact": action_identity["dtype_exact"],
                "action_array_equal": action_identity["array_equal"],
                "action_byte_sha_exact": action_identity["byte_sha_exact"],
                "action_candidate_sha256": action_identity["candidate_action_sha256"],
                "action_expected_sha256": action_identity["expected_action_sha256"],
                "action_max_abs_diff": action_identity["max_abs_diff"],
                "first_divergence_phase": first.get("first_divergence_phase"),
                "first_divergence_field": first.get("field"),
                "classification": first.get("classification"),
                "pre_diff_count": len(pre_diff),
                "post_diff_count": len(post_diff),
                "pre_diff_classification_counts": json.dumps(
                    transition_classification_counts(pre_diff), sort_keys=True
                ),
                "post_diff_classification_counts": json.dumps(
                    transition_classification_counts(post_diff), sort_keys=True
                ),
                "reference_next_observation_sha256": reference_step["next_observation_sha256"],
                "replay_next_observation_sha256": hash_typed_observation(replay_next_obs),
                "replay_reward": float(replay_reward),
                "replay_done": bool(replay_done),
                "replay_info": json.dumps(_json_clone(replay_info), sort_keys=True),
                "applied_actions": ";".join(applied["actions"]),
                **gate,
            }
            write_json(ablation_dir / "applied_ablation.json", applied)
            write_json(ablation_dir / "replay_pre.json", replay_pre)
            write_json(ablation_dir / "replay_post.json", replay_post)
            write_dict_csv(ablation_dir / "pre_diff.csv", pre_diff)
            write_dict_csv(ablation_dir / "post_diff.csv", post_diff)
            write_json(ablation_dir / "ablation_summary.json", row)
            rows.append(row)
        finally:
            try:
                replay_env.close()
            except Exception:
                pass
            release_real_policy(replay_policy)

    passed = [row for row in rows if row.get("qpos_qvel_exact") is True and row.get("replay_next_observation_sha256") == reference_step["next_observation_sha256"]]
    summary = {
        "stage": "C2_CONTROL_STATE_CAUSAL_ABLATION",
        "result": "C2_ONE_STEP_POST_ACTION_EXACT" if passed else "C2_ONE_STEP_POST_ACTION_STILL_DIVERGES",
        "forbidden_paths_not_run": [
            "formal_restore_3x",
            "R2",
            "VIS",
            "RAND",
            "SHUFFLED",
            "ORACLE",
            "ATTACK",
            "A800_FORMAL",
        ],
        "suite": snapshot.parent_manifest.suite,
        "task_idx": int(snapshot.parent_manifest.task_idx),
        "state_id": int(snapshot.parent_manifest.state_id),
        "eval_seed": int(snapshot.parent_manifest.eval_seed),
        "emit_step": int(snapshot.prefix.emit_step),
        "typed_prefix_manifest_sha256": hash_jsonable(typed_manifest),
        "reference_action_identity": reference_action_identity,
        "reference_step": reference_step,
        "ablation_count": len(rows),
        "passing_ablation_count": len(passed),
        "passing_ablations": [row["ablation"] for row in passed],
        "ablations": rows,
    }
    write_dict_csv(output_dir / "c2_ablation_results.csv", rows)
    write_json(output_dir / "c2_control_state_ablation_summary.json", summary)
    seal = write_recursive_manifest(output_dir)
    summary["recursive_sha256_manifest_sha256"] = seal
    write_json(output_dir / "c2_control_state_ablation_summary.json", summary)
    return summary


def run_exact_action_prefix_replay_canary(
    *,
    args: argparse.Namespace,
    selected: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(output_dir, args, stage="C3_EXACT_ACTION_PREFIX_REPLAY_ONE_STEP")
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    validate_transition_state_audit_known_parent(snapshot)
    prefix_trace = [dict(row) for row in selected.get("prefix_trace", [])]
    if len(prefix_trace) != int(snapshot.prefix.emit_step):
        raise PrefixReplayDivergence(
            f"prefix trace length {len(prefix_trace)} != emit step {snapshot.prefix.emit_step}"
        )
    write_json(output_dir / "parent_dependency_manifest.json", asdict(snapshot.parent_manifest))
    write_json(output_dir / "runtime_receipt.json", asdict(selected["runtime"]))
    write_json(output_dir / "dependency_receipt.json", selected["dependency"])
    write_jsonl(output_dir / "original_prefix_trace.jsonl", prefix_trace)
    write_jsonl(
        output_dir / "dummy_wait_trace.jsonl",
        [
            {
                "stage": "C3_EXACT_ACTION_PREFIX_REPLAY_ONE_STEP",
                "status": "NOT_RUN",
                "reason": "exact action-prefix replay uses recorded env_action prefix, not dummy wait restore",
            }
        ],
    )
    prefix_trace_sha = sha256_jsonable(prefix_trace)
    (output_dir / "prefix_trace_sha256.txt").write_text(prefix_trace_sha + "\n", encoding="utf-8")

    reference_env: RealLiberoEnvAdapter = selected["env_adapter"]
    reference_student: RealSC5StudentAdapter = selected["student"]
    reference_branch_pre = dict(selected.get("branch_pre_hashes", {}))
    reference_branch_post_update = dict(selected.get("branch_post_student_update_hashes", {}))
    if not reference_branch_pre or not reference_branch_post_update:
        raise _schema_invalid("selected parent missing branch pre/post Student update hashes")
    reference_next_obs, reference_reward, reference_done, reference_info = reference_env.step_env_action(
        postprocess_openvla_action_for_libero(snapshot.clean_action_t)
    )
    reference_post = prefix_replay_state_hashes(
        env=reference_env, obs=reference_next_obs, student=reference_student, policy=selected["policy"]
    )
    branch_reference = {
        "observation_sha256": reference_branch_pre["observation_sha256"],
        "policy_input_sha256": reference_branch_pre["policy_input_sha256"],
        "qpos_sha256": reference_branch_pre["qpos_sha256"],
        "qvel_sha256": reference_branch_pre["qvel_sha256"],
        "flat_sim_state_sha256": reference_branch_pre["flat_sim_state_sha256"],
        "student_state_sha256": reference_branch_pre["student_state_sha256"],
        "feature_history_sha256": reference_branch_pre["feature_history_sha256"],
        "branch_post_student_update_state_sha256": reference_branch_post_update["student_state_sha256"],
        "branch_post_student_update_feature_history_sha256": reference_branch_post_update["feature_history_sha256"],
        "post_branch_qpos_sha256": reference_post["qpos_sha256"],
        "post_branch_qvel_sha256": reference_post["qvel_sha256"],
        "post_branch_flat_sim_state_sha256": reference_post["flat_sim_state_sha256"],
        "post_branch_observation_sha256": reference_post["observation_sha256"],
        "post_branch_reward": float(reference_reward),
        "post_branch_done": bool(reference_done),
        "post_branch_info": _json_clone(reference_info),
    }
    write_json(output_dir / "branch_boundary_manifest.json", reference_branch_pre)
    write_json(output_dir / "post_branch_reference_state.json", branch_reference)
    try:
        reference_env.close()
    except Exception:
        pass
    release_real_policy(selected.get("policy"))
    if isinstance(selected, dict):
        selected["env_adapter"] = None
        selected["policy"] = None

    replay_env = None
    replay_policy = None
    try:
        replay_env, replay_initial_obs, _task_obj, _bddl = build_real_env_for_candidate(
            suite=snapshot.parent_manifest.suite,
            task_idx=snapshot.parent_manifest.task_idx,
            state_id=snapshot.parent_manifest.state_id,
            render_gpu=int(args.render_gpu),
            max_steps=int(args.max_steps),
        )
        replay_adapter = RealLiberoEnvAdapter(replay_env)
        replay_policy, replay_student, _model, _detector = load_real_policy_and_student(
            args, env_adapter=replay_adapter, instruction=str(selected["instruction"])
        )
        result = run_exact_action_prefix_replay_from_trace(
            env=replay_adapter,
            student=replay_student,
            policy=replay_policy,
            initial_obs=replay_initial_obs,
            prefix_steps=prefix_trace,
            branch_step=int(snapshot.prefix.emit_step),
            expected_branch_action=np.asarray(snapshot.clean_action_t),
            expected_branch_tokens=snapshot.clean_tokens_t,
            expected_branch_env_action=postprocess_openvla_action_for_libero(snapshot.clean_action_t),
            expected_prefix_trace_sha256=prefix_trace_sha,
            branch_reference=branch_reference,
            expected_next_observations=selected.get("prefix_next_observations"),
            observation_drift_output_dir=output_dir / "prefix_observation_drift",
        )
        result["parent_key"] = snapshot.parent_manifest.parent_key
        result["prefix_trace_sha256"] = prefix_trace_sha
        write_json(output_dir / "post_branch_replay_state.json", result)
        write_json(output_dir / "c3_prefix_replay_summary.json", result)
        write_dict_csv(output_dir / "prefix_replay_step_diff.csv", result["replay_prefix_rows"])
        write_jsonl(output_dir / "replay_prefix_trace.jsonl", result["replay_prefix_rows"])
        write_json(output_dir / "branch_action_exactness.json", result["branch_action_exactness"])
        write_dict_csv(output_dir / "post_branch_diff.csv", result["post_branch_diff_rows"])
        write_json(output_dir / "prefix_replay_first_divergence.json", {"first_divergence": None})
    except PrefixReplayDivergence as exc:
        failure = {
            "stage": "C3_EXACT_ACTION_PREFIX_REPLAY_ONE_STEP",
            "result": "FAIL",
            "failure_class": "PREFIX_REPLAY_DIVERGENCE",
            "error": str(exc),
            "parent_key": snapshot.parent_manifest.parent_key,
            "prefix_trace_sha256": prefix_trace_sha,
        }
        write_json(output_dir / "c3_prefix_replay_summary.json", failure)
        write_json(output_dir / "prefix_replay_first_divergence.json", failure)
        write_recursive_manifest(output_dir)
        raise
    finally:
        try:
            if replay_env is not None:
                replay_env.close()
        except Exception:
            pass
        release_real_policy(replay_policy)
    summary = json.loads((output_dir / "c3_prefix_replay_summary.json").read_text(encoding="utf-8"))
    write_recursive_manifest(output_dir)
    return summary


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
    release_real_policy(policy)
    selected["policy"] = None
    selected["student"] = None
    selected["env_adapter"] = None

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
    release_real_policy(replay_policy_a)

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
    release_real_policy(replay_policy_b)

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


def run_captured_prefix_canary_attempt(
    args: argparse.Namespace,
    selected: Mapping[str, Any],
    attempt_dir: Path,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    snapshot: ExactRestoreSnapshotPayload = selected["snapshot"]
    env_adapter: RealLiberoEnvAdapter = selected["env_adapter"]
    policy: RealOpenVLAPolicyAdapter = selected["policy"]
    student: RealSC5StudentAdapter = selected["student"]
    obs_t = clone_typed_observation(selected["obs_t"])
    (attempt_dir / "stdout.log").write_text("", encoding="utf-8")
    (attempt_dir / "stderr.log").write_text("", encoding="utf-8")
    write_json(attempt_dir / "parent_dependency_manifest.json", asdict(selected["parent"]))
    write_json(attempt_dir / "runtime_receipt.json", asdict(selected["runtime"]))
    write_json(attempt_dir / "dependency_sha_receipt.json", selected["dependency"])
    write_json(attempt_dir / "prefix_snapshot_manifest.json", asdict(snapshot.prefix))
    typed_manifest = save_typed_prefix_observation_artifacts(
        attempt_dir / "captured_prefix",
        snapshot=snapshot,
        policy=policy,
    )

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
    release_real_policy(policy)
    selected["policy"] = None
    selected["student"] = None
    selected["env_adapter"] = None

    replay_env, replay_policy, replay_student = new_env_policy_student_for_snapshot(args, selected)
    restore_snapshot(replay_env, replay_student, snapshot, replay_policy)
    branch = captured_prefix_branch_record(
        condition="CLEAN_REPLAY",
        snapshot=snapshot,
        env=replay_env,
        student=replay_student,
        policy=replay_policy,
    )
    replay = rollout_clean_steps(
        env=replay_env,
        student=replay_student,
        policy=replay_policy,
        initial_obs=clone_typed_observation(snapshot.observation),
        start_step=snapshot.prefix.emit_step,
        expected_first_action=snapshot.clean_action_t,
        expected_first_tokens=snapshot.clean_tokens_t,
        first_step_student_already_updated=True,
    )
    replay_frames = list(replay_env.frames)
    replay_env.close()
    release_real_policy(replay_policy)

    problems = [f"reference_vs_replay:{p}" for p in compare_step_sequences(reference, replay)]
    branch_summary = validate_branch_records(snapshot.prefix, [branch], required_conditions=("CLEAN_REPLAY",))
    result = {
        "captured_prefix_canary_pass": not problems,
        "restore_steps": len(reference),
        "prefix_snapshot_sha256": snapshot.prefix.snapshot_sha256,
        "snapshot_payload_sha256": snapshot.payload_sha256,
        "branch_summary": branch_summary,
        "reference_vs_replay_mismatch_count": len(problems),
        "reference_vs_replay_problems": problems,
        "typed_prefix_manifest_sha256": hash_jsonable(typed_manifest),
        "branch_input_source": "CAPTURED_PREFIX_OBSERVATION",
    }
    if problems:
        write_json(attempt_dir / "branch_records.json", [asdict(branch)])
        write_jsonl(attempt_dir / "reference_steps.jsonl", reference)
        write_jsonl(attempt_dir / "replay_steps.jsonl", replay)
        write_real_video(attempt_dir / "raw_reference.mp4", reference_frames)
        write_real_video(attempt_dir / "raw_replay.mp4", replay_frames)
        write_json(attempt_dir / "attempt_summary.json", result)
        write_recursive_manifest(attempt_dir)
        raise ExactRestoreError(";".join(problems))
    write_json(attempt_dir / "branch_records.json", [asdict(branch)])
    write_jsonl(attempt_dir / "reference_steps.jsonl", reference)
    write_jsonl(attempt_dir / "replay_steps.jsonl", replay)
    write_real_video(attempt_dir / "raw_reference.mp4", reference_frames)
    write_real_video(attempt_dir / "raw_replay.mp4", replay_frames)
    write_json(attempt_dir / "numeric_drift_report.json", {"reference_vs_replay": []})
    summary = {
        **result,
        "attempt_id": attempt_dir.name,
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
    write_run_manifest(out, args, stage="REAL_LIBERO_SINGLE_PARENT")
    write_gpu_and_kernel_snapshots(out, suffix="before")
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
    if args.exact_action_prefix_replay_canary_only:
        if not args.candidate_manifest:
            raise ExactRestoreError("C3 exact action-prefix replay requires an explicit candidate manifest")
        if len(candidates) != 1:
            raise ExactRestoreError("C3 exact action-prefix replay candidate manifest must have exactly one row")
        validate_known_goal_candidate(candidates[0])
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
            if args.exact_action_prefix_replay_canary_only and not isinstance(exc, NoNaturalStudentEmit):
                raise
            candidate_rows[idx]["status"] = "INELIGIBLE"
            candidate_rows[idx]["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
            write_candidate_manifest(out / "candidate_manifest.csv", candidate_rows)
            continue
    if selected is None:
        write_json(out / "single_parent_restore_qualification_summary.json", {"result": "NO_ELIGIBLE_GOAL_RESTORE_PARENT"})
        raise ExactRestoreError("NO_ELIGIBLE_GOAL_RESTORE_PARENT")
    write_candidate_manifest(out / "candidate_manifest.csv", candidate_rows)
    if args.transition_state_audit_only:
        try:
            audit_summary = run_transition_state_audit(
                args=args,
                selected=selected,
                output_dir=out / "transition_state_audit",
            )
        finally:
            try:
                if selected.get("env_adapter") is not None:
                    selected["env_adapter"].close()
            except Exception:
                pass
            release_real_policy(selected.get("policy"))
        final = {
            "stage": "TRANSITION_STATE_AUDIT_ONLY",
            "result": "TRANSITION_STATE_AUDIT_COMPLETE",
            "selected_candidate": candidates[selected_idx],
            "audit_summary": audit_summary,
        }
        write_gpu_and_kernel_snapshots(out, suffix="after")
        seal = write_recursive_manifest(out)
        final["recursive_sha256_manifest_sha256"] = seal
        write_json(out / "single_parent_restore_qualification_summary.json", final)
        print(json.dumps(final, sort_keys=True, default=str))
        return
    if args.control_state_ablation_only:
        try:
            ablation_summary = run_control_state_causal_ablation(
                args=args,
                selected=selected,
                output_dir=out / "control_state_causal_ablation",
            )
        finally:
            try:
                if selected.get("env_adapter") is not None:
                    selected["env_adapter"].close()
            except Exception:
                pass
            release_real_policy(selected.get("policy"))
        final = {
            "stage": "C2_CONTROL_STATE_CAUSAL_ABLATION",
            "result": ablation_summary.get("result"),
            "selected_candidate": candidates[selected_idx],
            "ablation_summary": ablation_summary,
        }
        write_gpu_and_kernel_snapshots(out, suffix="after")
        seal = write_recursive_manifest(out)
        final["recursive_sha256_manifest_sha256"] = seal
        write_json(out / "single_parent_restore_qualification_summary.json", final)
        print(json.dumps(final, sort_keys=True, default=str))
        return
    if args.exact_action_prefix_replay_canary_only:
        try:
            c3_summary = run_exact_action_prefix_replay_canary(
                args=args,
                selected=selected,
                output_dir=out / "exact_action_prefix_replay_canary",
            )
        finally:
            try:
                if selected.get("env_adapter") is not None:
                    selected["env_adapter"].close()
            except Exception:
                pass
            release_real_policy(selected.get("policy"))
        final = {
            "stage": "C3_EXACT_ACTION_PREFIX_REPLAY_ONE_STEP",
            "result": c3_summary.get("result"),
            "selected_candidate": candidates[selected_idx],
            "c3_summary": c3_summary,
            "forbidden_paths_not_run": [
                "five_step",
                "formal_restore_3x",
                "R2",
                "VIS",
                "RAND",
                "SHUFFLED",
                "ORACLE",
                "ATTACK",
                "A800_FORMAL",
            ],
        }
        write_gpu_and_kernel_snapshots(out, suffix="after")
        seal = write_recursive_manifest(out)
        final["recursive_sha256_manifest_sha256"] = seal
        write_json(out / "single_parent_restore_qualification_summary.json", final)
        print(json.dumps(final, sort_keys=True, default=str))
        return
    if args.observation_audit_only:
        audit_summary = run_observation_reconstruction_audit(
            args=args,
            selected=selected,
            output_dir=out / "observation_reconstruction_audit",
        )
        write_json(
            out / "single_parent_restore_qualification_summary.json",
            {
                "result": "OBSERVATION_RECONSTRUCTION_AUDIT_ONLY",
                "selected_candidate": candidates[selected_idx],
                "audit_summary": audit_summary,
            },
        )
        try:
            selected["env_adapter"].close()
        except Exception:
            pass
        release_real_policy(selected.get("policy"))
        write_gpu_and_kernel_snapshots(out, suffix="after")
        seal = write_recursive_manifest(out)
        write_json(
            out / "single_parent_restore_qualification_summary.json",
            {
                "result": "OBSERVATION_RECONSTRUCTION_AUDIT_ONLY",
                "selected_candidate": candidates[selected_idx],
                "audit_summary": audit_summary,
                "recursive_sha256_manifest_sha256": seal,
            },
        )
        return
    if args.captured_prefix_canary_only:
        try:
            canary_summary = run_captured_prefix_canary_attempt(args, selected, out / "captured_prefix_canary_00")
        finally:
            try:
                if selected.get("env_adapter") is not None:
                    selected["env_adapter"].close()
            except Exception:
                pass
            release_real_policy(selected.get("policy"))
        final = {
            "stage": "CAPTURED_PREFIX_SINGLE_RESTORE_CANARY",
            "result": "SINGLE_RESTORE_CANARY_PASS"
            if canary_summary.get("captured_prefix_canary_pass") is True
            else "SINGLE_RESTORE_CANARY_FAIL",
            "selected_candidate": candidates[selected_idx],
            "canary": canary_summary,
        }
        write_gpu_and_kernel_snapshots(out, suffix="after")
        seal = write_recursive_manifest(out)
        final["recursive_sha256_manifest_sha256"] = seal
        write_json(out / "single_parent_restore_qualification_summary.json", final)
        print(json.dumps(final, sort_keys=True, default=str))
        return
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
    write_gpu_and_kernel_snapshots(out, suffix="after")
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
    ap.add_argument("--observation-audit-only", action="store_true")
    ap.add_argument("--captured-prefix-canary-only", action="store_true")
    ap.add_argument("--transition-state-audit-only", action="store_true")
    ap.add_argument("--control-state-ablation-only", action="store_true")
    ap.add_argument("--exact-action-prefix-replay-canary-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--repetitions", type=int, default=3)
    return ap.parse_args()


def validate_mode_gates(args: argparse.Namespace) -> None:
    exact_modes = [
        "observation_audit_only",
        "captured_prefix_canary_only",
        "transition_state_audit_only",
        "control_state_ablation_only",
        "exact_action_prefix_replay_canary_only",
    ]
    enabled = [name for name in exact_modes if bool(getattr(args, name, False))]
    if len(enabled) > 1:
        raise ExactRestoreError(f"exact restore modes are mutually exclusive: {enabled}")
    if bool(getattr(args, "exact_action_prefix_replay_canary_only", False)):
        if int(getattr(args, "repetitions", 1)) != 1:
            raise ExactRestoreError("C3 exact action-prefix replay forbids formal repetitions; set --repetitions 1")
        if not bool(getattr(args, "real_libero_single_parent", False)):
            raise ExactRestoreError("C3 exact action-prefix replay must run under --real-libero-single-parent")
        if str(getattr(args, "suite", "")) != "libero_goal" or int(getattr(args, "eval_seed", -1)) != 0:
            raise ExactRestoreError("C3 exact action-prefix replay accepts only libero_goal eval_seed 0")


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

    def step_env_action(self, env_action: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self.step(env_action)


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

    def policy_input_fingerprint(self, obs: Any) -> dict[str, Any]:
        return {"mock_policy_input": _json_clone(obs)}


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
    validate_mode_gates(args)
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
