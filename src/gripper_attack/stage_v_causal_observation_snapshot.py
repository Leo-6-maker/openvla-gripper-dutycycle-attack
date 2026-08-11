"""Exact, file-backed causal observation snapshots for prospective M3.5 V1.4.

The V1.3 runner stored hashes but reconstructed observations through a fresh
renderer.  This module keeps the canonical bytes in sidecars and makes the
loaded object, not a fresh render, the primary input authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_v_canonical_execution_core import canonical_sha256, canonical_value


SNAPSHOT_SCHEMA = "STAGE_V_CAUSAL_PROBE_SNAPSHOT_V2"
SNAPSHOT_FILENAME = "CAUSAL_PROBE_SNAPSHOT_V2.json"
SNAPSHOT_SHA_FILENAME = "CAUSAL_PROBE_SNAPSHOT_V2.json.sha256"
ARRAY_DESCRIPTOR_SCHEMA = "STAGE_V_CAUSAL_ARRAY_DESCRIPTOR_V1"
ARRAY_MARKER = "__causal_array__"


class CausalSnapshotError(RuntimeError):
    """Raised when a snapshot cannot be proven exact."""


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _array_bytes(value: Any) -> tuple[str, str, list[int], bytes] | None:
    """Return backend, dtype, shape and C-contiguous bytes without pickle."""
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        dtype = str(getattr(tensor, "dtype", ""))
        shape = [int(item) for item in getattr(tensor, "shape", ())]
        try:
            raw = tensor.numpy().tobytes()
        except Exception:
            import torch

            raw = tensor.view(torch.uint8).numpy().tobytes()
        return "torch", dtype, shape, raw
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return "numpy", array.dtype.str, [int(item) for item in array.shape], array.tobytes(order="C")
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "value"


def _write_field(value: Any, root: Path, path: str, arrays: list[dict[str, Any]]) -> Any:
    raw = _array_bytes(value)
    logical_type = None
    if raw is None and hasattr(value, "tobytes") and hasattr(value, "mode") and hasattr(value, "size"):
        value = np.asarray(value)
        raw = _array_bytes(value)
        logical_type = "image"
    if raw is not None:
        backend, dtype, shape, payload = raw
        relative = Path("arrays") / f"{len(arrays):04d}_{_safe_name(path)}.bin"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        descriptor = {
            "schema": ARRAY_DESCRIPTOR_SCHEMA,
            "backend": backend,
            "dtype": dtype,
            "shape": shape,
            "byte_length": len(payload),
            "raw_sha256": _sha256(payload),
            "contiguous_order": "C",
            "binary_path": relative.as_posix(),
        }
        if logical_type is not None:
            descriptor["logical_type"] = logical_type
        arrays.append({"field": path, **descriptor})
        return {ARRAY_MARKER: len(arrays) - 1}
    if isinstance(value, Mapping):
        return {str(key): _write_field(item, root, f"{path}.{key}" if path else str(key), arrays) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_write_field(item, root, f"{path}[{index}]", arrays) for index, item in enumerate(value)]
    return canonical_value(value)


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "torch.bool": torch.bool,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.int16": torch.int16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise CausalSnapshotError(f"TORCH_DTYPE_UNSUPPORTED:{name}")
    return mapping[name]


def _read_field(value: Any, root: Path, arrays: Sequence[Mapping[str, Any]], *, materialize_torch: bool) -> Any:
    if isinstance(value, Mapping) and set(value) == {ARRAY_MARKER}:
        index = int(value[ARRAY_MARKER])
        if index < 0 or index >= len(arrays):
            raise CausalSnapshotError("ARRAY_DESCRIPTOR_INDEX_INVALID")
        descriptor = dict(arrays[index])
        target = root / str(descriptor["binary_path"])
        payload = target.read_bytes()
        if len(payload) != int(descriptor["byte_length"]) or _sha256(payload) != descriptor["raw_sha256"]:
            raise CausalSnapshotError(f"ARRAY_BYTES_SHA_MISMATCH:{descriptor.get('field', index)}")
        shape = tuple(int(item) for item in descriptor["shape"])
        if descriptor["backend"] == "torch" and materialize_torch:
            import torch

            result = torch.frombuffer(bytearray(payload), dtype=_torch_dtype(str(descriptor["dtype"]))).reshape(shape)
            return result.clone()
        dtype = np.dtype(str(descriptor["dtype"])) if descriptor["backend"] == "numpy" else np.uint8
        return np.frombuffer(payload, dtype=dtype).copy().reshape(shape)
    if isinstance(value, Mapping):
        return {str(key): _read_field(item, root, arrays, materialize_torch=materialize_torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_read_field(item, root, arrays, materialize_torch=materialize_torch) for item in value]
    return value


def write_snapshot(root: Path, payload: Mapping[str, Any], *, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Write one immutable snapshot package and return its manifest."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise CausalSnapshotError(f"SNAPSHOT_ROOT_NOT_EMPTY:{root}")
    root.mkdir(parents=True, exist_ok=True)
    arrays: list[dict[str, Any]] = []
    encoded_payload = _write_field(dict(payload), root, "payload", arrays)
    manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "SEALED_PROSPECTIVE_SNAPSHOT",
        "binding": canonical_value(dict(binding)),
        "payload": encoded_payload,
        "arrays": arrays,
        "array_count": len(arrays),
        "fresh_render_equality_gate_used": False,
        "primary_input_authority": "loaded_frozen_canonical_bytes",
    }
    manifest_path = root / SNAPSHOT_FILENAME
    manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    manifest_sha = _sha256(manifest_path.read_bytes())
    (root / SNAPSHOT_SHA_FILENAME).write_text(f"{manifest_sha}  {SNAPSHOT_FILENAME}\n", encoding="utf-8")
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def load_snapshot(root: Path, *, materialize_torch: bool = False) -> dict[str, Any]:
    """Verify and load all frozen bytes; any mismatch fails closed."""
    root = Path(root)
    manifest_path = root / SNAPSHOT_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SNAPSHOT_SCHEMA or manifest.get("status") != "SEALED_PROSPECTIVE_SNAPSHOT":
        raise CausalSnapshotError("SNAPSHOT_SCHEMA_OR_STATUS_INVALID")
    recorded = (root / SNAPSHOT_SHA_FILENAME).read_text(encoding="utf-8").strip().split()
    actual = _sha256(manifest_path.read_bytes())
    if len(recorded) != 2 or recorded[0] != actual or recorded[1] != SNAPSHOT_FILENAME:
        raise CausalSnapshotError("SNAPSHOT_MANIFEST_SHA_MISMATCH")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, list) or int(manifest.get("array_count", -1)) != len(arrays):
        raise CausalSnapshotError("SNAPSHOT_ARRAY_MANIFEST_INVALID")
    payload = _read_field(manifest.get("payload"), root, arrays, materialize_torch=materialize_torch)
    return {"manifest": manifest, "payload": payload}


def assert_exact(actual: Any, expected: Any, *, label: str) -> None:
    if canonical_value(actual) != canonical_value(expected):
        raise CausalSnapshotError(f"EXACT_BINDING_MISMATCH:{label}")


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state, not just a seed; consumers must bind its bytes."""
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_global": np.random.get_state(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    except ImportError:
        state["torch"] = "UNAVAILABLE"
    return state


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tuple_tree(item) for item in value)
    return value


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a previously captured RNG state before a fresh branch starts."""
    if "python_random" in state:
        random.setstate(_tuple_tree(state["python_random"]))
    if "numpy_global" in state:
        numpy_state = state["numpy_global"]
        if isinstance(numpy_state, list) and len(numpy_state) == 5:
            numpy_state = tuple(numpy_state)
        np.random.set_state(numpy_state)
    if "torch_cpu" in state or "torch_cuda_all" in state:
        import torch

        if "torch_cpu" in state:
            cpu_state = state["torch_cpu"]
            if not hasattr(cpu_state, "dtype"):
                cpu_state = torch.as_tensor(cpu_state, dtype=torch.uint8)
            torch.set_rng_state(cpu_state)
        if torch.cuda.is_available() and "torch_cuda_all" in state:
            cuda_states = []
            for item in state["torch_cuda_all"]:
                if not hasattr(item, "dtype"):
                    item = torch.as_tensor(item, dtype=torch.uint8)
                cuda_states.append(item)
            torch.cuda.set_rng_state_all(cuda_states)


def _capture_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _capture_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_capture_value(item) for item in value]
    if _array_bytes(value) is not None:
        return value
    if hasattr(value, "flatten") and callable(value.flatten):
        try:
            return np.asarray(value.flatten())
        except Exception:
            pass
    if hasattr(value, "buf"):
        return _attrs(value, ("buf", "index", "current", "length", "dim"))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"unhandled_runtime_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _capture_rng_object(value: Any) -> Any:
    """Capture common environment RNG containers without pickle."""
    if value is None:
        return None
    if hasattr(value, "get_state") and callable(value.get_state):
        return {"kind": "get_state", "state": _capture_value(value.get_state())}
    bit_generator = getattr(value, "bit_generator", None)
    state = getattr(bit_generator, "state", None) if bit_generator is not None else None
    if state is not None:
        return {"kind": "bit_generator_state", "state": _capture_value(state)}
    return _capture_value(value)


def _attrs(obj: Any, names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        if hasattr(obj, name):
            result[name] = _capture_value(getattr(obj, name))
    return result


def _controller_state(controller: Any) -> dict[str, Any]:
    names = ("goal_pos", "goal_ori", "relative_ori", "ori_ref", "goal_qpos", "goal_vel", "goal_torque", "current_pos", "current_ori", "current_vel", "current_torque", "torques", "kp", "kd", "summed_err", "derr_buf")
    result = _attrs(controller, names)
    for field in ("interpolator_pos", "interpolator_ori", "interpolator",):
        interpolator = getattr(controller, field, None)
        if interpolator is not None:
            result[field] = _attrs(interpolator, ("start", "goal", "step", "total_steps"))
    return result


def _observable_state(observables: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(observables, Mapping):
        return result
    for name, observable in sorted(observables.items(), key=lambda pair: str(pair[0])):
        result[str(name)] = _attrs(observable, ("_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled"))
    return result


def capture_runtime_state(env: Any, *, model: Any | None = None, adapter: Any | None = None) -> dict[str, Any]:
    """Capture mutable state known to affect the primary physical window."""
    inner = getattr(env, "env", None)
    target = inner if inner is not None else env
    environment = _attrs(target, ("timestep", "cur_time", "done", "horizon", "ignore_done", "control_freq", "control_timestep", "model_timestep", "sim_state_initial"))
    for name in ("np_random", "_np_random", "rng", "_rng", "random_state"):
        if hasattr(target, name):
            environment[name] = _capture_rng_object(getattr(target, name))
    robots: list[dict[str, Any]] = []
    for index, robot in enumerate(getattr(target, "robots", ()) or ()):
        row: dict[str, Any] = {"index": index, "recent_buffers": _attrs(robot, ("recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques", "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc"))}
        controller = getattr(robot, "controller", None)
        if controller is not None:
            row["controller"] = _controller_state(controller)
        robots.append(row)
    model_state = _attrs(model, ("training",)) if model is not None else {}
    adapter_state = _attrs(adapter, ("unnorm_key", "center_crop", "base_vla_name", "open_token_ids", "close_token_ids", "token_action_map")) if adapter is not None else {}
    return {
        "environment": environment,
        "observables": _observable_state(getattr(target, "_observables", None)),
        "robots": robots,
        "model_execution": model_state,
        "adapter_execution": adapter_state,
        "rng": capture_rng_state(),
    }


def capture_simulator_state(env: Any) -> dict[str, Any]:
    """Capture simulator arrays plus the wrapper's registered flat state."""
    sim = getattr(env, "sim", None)
    if sim is None:
        sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        raise CausalSnapshotError("SIMULATOR_HANDLE_MISSING")
    data = getattr(sim, "data", None)
    result: dict[str, Any] = {"schema": "STAGE_V_FULL_SIM_STATE_DIAGNOSTIC_V2", "data": {}}
    for field in ("qpos", "qvel", "qacc", "qacc_warmstart", "act", "ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat"):
        if data is not None and hasattr(data, field):
            value = getattr(data, field)
            result["data"][field] = value.copy() if hasattr(value, "copy") else value
    if hasattr(env, "get_sim_state"):
        result["registered_flat_state"] = np.asarray(env.get_sim_state()).copy()
    if hasattr(sim, "get_state"):
        state = sim.get_state()
        result["sim_state"] = {
            field: getattr(state, field).copy() if hasattr(getattr(state, field), "copy") else getattr(state, field)
            for field in ("time", "qpos", "qvel", "act", "udd_state")
            if hasattr(state, field)
        }
    return result


def assert_runtime_exact(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    """Compare captured mutable runtime state with exact canonical bytes."""
    assert_exact(actual, expected, label=label)


def reference_action_window(rows: Sequence[Mapping[str, Any]], *, start_step: int, length: int) -> list[dict[str, Any]]:
    selected = list(rows)[int(start_step) : int(start_step) + int(length)]
    if len(selected) != int(length) or any(int(row.get("step", -1)) != int(start_step) + index for index, row in enumerate(selected)):
        raise CausalSnapshotError("REFERENCE_ACTION_WINDOW_INCOMPLETE")
    result: list[dict[str, Any]] = []
    for row in selected:
        raw = list(row.get("raw_action", []))
        env = list(row.get("env_action", []))
        if len(raw) != 7 or len(env) != 7:
            raise CausalSnapshotError("REFERENCE_ACTION_DIMENSION_INVALID")
        result.append({"step": int(row["step"]), "raw_policy_action": raw, "env_action": env, "action_sha256": canonical_sha256({"raw": raw, "env": env})})
    return result


def matched_action(reference: Mapping[str, Any], *, forced_open: bool = False) -> dict[str, Any]:
    if not forced_open:
        raw = list(reference["raw_policy_action"])
        env = list(reference["env_action"])
        return {
            "raw_policy_action": raw,
            "normalized_action": raw,
            "env_action": env,
            "arm_delta": [0.0] * 6,
            "arm_delta_linf": 0.0,
            "gripper_delta_env": 0.0,
            "arm_source": "CANONICAL_CLEAN_REFERENCE",
        }
    from .stage_v_m3_5_physical_taxonomy import build_forced_open_action

    action = build_forced_open_action(reference["raw_policy_action"], reference["env_action"])
    action["arm_source"] = "MATCHED_CANONICAL_ARM_FORCED_OPEN_GRIPPER"
    return action
