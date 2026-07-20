"""Fail-closed R10.4D single-episode passive deployment runtime.

This module contains no attack path.  The detector and FSM may observe the
clean OpenVLA action stream, but ``executed_action`` is always an exact copy of
the official postprocessed clean action.

R10.4E Gate E-R2.5B: Added 4-type termination classification, recursive
safe_json_value, authorized_parents parameter, schema V1 upgrade.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn as nn

from .r10_4_runtime import (
    ACTION_DIM,
    FEATURE_NAMES,
    FEATURE_ORDER_SHA256,
    OfficialStreamingFeatureAdapter,
    R10_4ContractError,
    initialize_env_once,
    sha256_file,
    verify_checksum_manifest,
)


CHECKPOINT_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_PARENT = "libero_10/task_00/state_20"
FROZEN = {
    "input_dim": 25,
    "hidden_dim": 64,
    "num_layers": 2,
    "grasp_threshold": 0.5,
    "grasp_persistence": 3,
    "guard_type": "vertical_lift",
    "guard_param": 0.02,
    "max_episode_emits": 1,
}
REQUIRED_BUNDLE_FILES = {
    "full_fit_deploy.pt",
    "detector_config.json",
    "feature_contract.json",
    "route_contract.json",
    "fsm_config.json",
    "source_binding.json",
    "training_binding.json",
    "MANIFEST.json",
}


class R10_4DContractError(R10_4ContractError):
    """Fail-closed R10.4D contract violation."""


# ═══════════════════════════════════════════════════════════════════════════════
# Models (frozen — do not retrain)
# ═══════════════════════════════════════════════════════════════════════════════

class RoutedGraspDetector(nn.Module):
    """Frozen R10.3 dual-head GRU architecture (46,658 parameters)."""

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head_multi = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_single = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    @torch.no_grad()
    def step(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None,
        route: str,
    ) -> tuple[float, torch.Tensor]:
        if features.shape != (FROZEN["input_dim"],):
            raise R10_4DContractError(f"DETECTOR_FEATURE_SHAPE:{tuple(features.shape)}")
        if hidden is None:
            hidden = torch.zeros(
                self.encoder.num_layers,
                1,
                self.encoder.hidden_size,
                dtype=features.dtype,
                device=features.device,
            )
        _, next_hidden = self.encoder(features.reshape(1, 1, -1), hidden)
        last = next_hidden[-1]
        if route == "multi_object_transfer":
            logit = self.head_multi(last).reshape(-1)[0]
        elif route == "single_object_pick_place":
            logit = self.head_single(last).reshape(-1)[0]
        else:
            raise R10_4DContractError(f"DETECTOR_UNSUPPORTED_ROUTE:{route}")
        return float(logit.detach().cpu()), next_hidden


class DetectorRuntime:
    def __init__(self, model: RoutedGraspDetector, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.hidden: torch.Tensor | None = None

    def reset(self) -> None:
        self.hidden = None

    def step(self, features: np.ndarray, route: str) -> tuple[float, float]:
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if values.shape != (25,) or not np.isfinite(values).all():
            raise R10_4DContractError("DETECTOR_FEATURE_INVALID")
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        logit, self.hidden = self.model.step(tensor, self.hidden, route)
        probability = 1.0 / (1.0 + math.exp(-logit))
        return logit, probability


@dataclass
class FSMState:
    state: str = "IDLE"
    event_id: int = 0
    grasp_persist: int = 0
    anchor_step: int = -1
    anchor_eef_z: float = 0.0
    emitted_this_event: bool = False
    total_emits: int = 0


class EventFSM:
    """Frozen one-shot vertical-lift scheduler."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state = FSMState()

    def step(
        self,
        *,
        step: int,
        grasp_probability: float,
        eef_z: float,
        close_mask: bool,
        route: str,
    ) -> dict[str, Any]:
        if route != "multi_object_transfer":
            return {
                "emit": False,
                "fsm_state": "ABSTAIN",
                "event_id": 0,
                "anchor_step": -1,
                "vertical_lift": 0.0,
                "reason": f"route={route}",
            }

        detected = grasp_probability > FROZEN["grasp_threshold"]
        s = self._state

        if s.state == "IDLE" and close_mask:
            s.state = "CLOSE_CANDIDATE"
            s.event_id += 1
            s.grasp_persist = 0
            s.emitted_this_event = False

        if s.state == "CLOSE_CANDIDATE":
            if detected:
                s.grasp_persist += 1
                if s.grasp_persist == 1:
                    s.anchor_step = step
                    s.anchor_eef_z = float(eef_z)
            else:
                s.grasp_persist = 0
            if s.grasp_persist >= FROZEN["grasp_persistence"]:
                s.state = "ARMED"

        if s.state in {"ARMED", "EVENT_CANDIDATE", "EMITTED"} and not close_mask:
            s.state = "RESET"

        if s.state == "ARMED" and not s.emitted_this_event:
            if float(eef_z) - s.anchor_eef_z >= FROZEN["guard_param"]:
                s.state = "EVENT_CANDIDATE"

        emit = False
        if s.state == "EVENT_CANDIDATE" and not s.emitted_this_event:
            if s.total_emits < FROZEN["max_episode_emits"]:
                s.emitted_this_event = True
                s.total_emits += 1
                s.state = "EMITTED"
                emit = True

        if s.state == "RESET" and close_mask:
            s.state = "CLOSE_CANDIDATE"
            s.event_id += 1
            s.grasp_persist = 0
            s.anchor_step = -1
            s.anchor_eef_z = 0.0
            s.emitted_this_event = False

        vertical_lift = 0.0 if s.anchor_step < 0 else float(eef_z) - s.anchor_eef_z
        return {
            "emit": emit,
            "fsm_state": s.state,
            "event_id": s.event_id,
            "anchor_step": s.anchor_step,
            "vertical_lift": vertical_lift,
            "total_emits": s.total_emits,
        }


def parse_route(identity: str) -> str:
    parts = identity.split("/")
    if len(parts) != 3:
        return "unsupported_abstain"
    suite, task, _state = parts
    if suite == "libero_10" and task in {f"task_{index:02d}" for index in range(10)}:
        return "multi_object_transfer"
    return "unsupported_abstain"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_exact_feature_order(value: Any) -> bool:
    if isinstance(value, list) and value == list(FEATURE_NAMES):
        return True
    if isinstance(value, dict):
        return any(_contains_exact_feature_order(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_exact_feature_order(item) for item in value)
    return False


def safe_json_value(obj: Any) -> Any:
    """Recursively normalize any value to a JSON-serializable form.

    Handles: None, bool, int, float, str, NumPy scalar, NumPy array,
    tuple, set, frozenset, Path, bytes, dict, list, and unknown objects.
    The result is guaranteed to pass json.dumps().
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if np.isfinite(obj):
            return float(obj)
        return str(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isfinite(val):
            return val
        return str(val)
    if isinstance(obj, np.ndarray):
        return [safe_json_value(v) for v in obj.tolist()]
    if isinstance(obj, (tuple, set, frozenset)):
        return [safe_json_value(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): safe_json_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json_value(v) for v in obj]
    # Unknown — best-effort string repr
    try:
        return str(type(obj).__name__) + ":" + repr(obj)[:200]
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def _safe_info(info: Any) -> dict[str, Any]:
    """Serialize env.step info dict safely, guaranteeing json.dumps success."""
    result = safe_json_value(info)
    if not isinstance(result, dict):
        return {"_raw": result}
    # Verify JSON round-trip
    try:
        json.dumps(result, sort_keys=True, default=str)
    except Exception:
        return {"_serialize_error": str(type(info).__name__)}
    return result


def _classify_termination(
    step_records: list[dict[str, Any]],
    configured_horizon: int,
    env: Any,
    violations: list[str],
) -> dict[str, Any]:
    """Classify episode termination into exactly one of four types.

    SUCCESS_TERMINATION        — done=True AND check_success()=True
    HORIZON_TERMINATION        — done=True at configured horizon, not success
    FULL_LOOP_TASK_FAILURE     — ran full horizon without done, not success
    EARLY_DONE_WITHOUT_SUCCESS — done=True before horizon, not success (HARD FAILURE)
    NO_STEPS                   — empty step_records (HARD FAILURE)
    UNCLASSIFIED               — none of the above match (HARD FAILURE)
    """
    observed_horizon = len(step_records)
    last_done = bool(step_records[-1]["done"]) if step_records else False

    # Simulator timestep (best-effort, MuJoCo via Robosuite)
    simulator_timestep = -1.0
    try:
        sim = getattr(env, "sim", None)
        if sim is not None and hasattr(sim, "data"):
            simulator_timestep = float(sim.data.time)
    except Exception:
        pass

    # env.check_success() — may raise
    env_success = None
    check_success_error = None
    if not violations and hasattr(env, "check_success"):
        try:
            env_success = bool(env.check_success())
        except Exception as exc:
            env_success = None
            check_success_error = f"{type(exc).__name__}:{str(exc)[:200]}"

    # Classification
    # P0-9: check_success() exception → fail-closed (cannot determine success)
    if check_success_error is not None:
        termination_reason = "CHECK_SUCCESS_FAILURE"
        task_success = False
        is_hard_failure = True
    elif not step_records:
        termination_reason = "NO_STEPS"
        task_success = False
        is_hard_failure = True
    elif last_done and env_success is True:
        termination_reason = "SUCCESS_TERMINATION"
        task_success = True
        is_hard_failure = False
    elif last_done and observed_horizon >= configured_horizon and env_success is not True:
        termination_reason = "HORIZON_TERMINATION"
        task_success = False
        is_hard_failure = False
    elif not last_done and observed_horizon >= configured_horizon:
        termination_reason = "FULL_LOOP_TASK_FAILURE"
        task_success = False
        is_hard_failure = False
    elif last_done and observed_horizon < configured_horizon and env_success is not True:
        termination_reason = "EARLY_DONE_WITHOUT_SUCCESS"
        task_success = False
        is_hard_failure = True
    else:
        termination_reason = "UNCLASSIFIED"
        task_success = False
        is_hard_failure = True

    if is_hard_failure and termination_reason != "NO_STEPS":
        violations.append(f"HARD_FAILURE:{termination_reason}")

    return {
        "termination_reason": termination_reason,
        "task_success": task_success,
        "is_hard_failure": is_hard_failure,
        "configured_horizon": configured_horizon,
        "observed_horizon": observed_horizon,
        "done": last_done,
        "env_check_success": env_success,
        "check_success_error": check_success_error,
        "simulator_timestep": simulator_timestep,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Authorization receipt validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_authorization_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_head: str,
    expected_parent: str,
    expected_checkpoint_sha256: str,
    expected_bundle_sha256s: str,
    expected_model_tree_sha256: str,
) -> None:
    required = {
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
        "selected_parent": expected_parent,
        "source_commit": expected_head,
        "detector_checkpoint_sha256": expected_checkpoint_sha256,
        "bundle_sha256s_sha256": expected_bundle_sha256s,
        "model_tree_sha256": expected_model_tree_sha256,
        "r4c_classification": "CONTACT_DYNAMICS_REPLAY_DIVERGENCE",
        "feature_order_sha256": FEATURE_ORDER_SHA256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise R10_4DContractError(f"AUTH_RECEIPT_FIELD_FAIL:{key}")


# ═══════════════════════════════════════════════════════════════════════════════
# Detector bundle loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_detector_bundle(
    bundle_root: Path,
    *,
    device: torch.device,
    expected_checkpoint_sha256: str,
    expected_bundle_sha256s: str,
) -> tuple[DetectorRuntime, dict[str, Any]]:
    bundle_root = bundle_root.resolve()
    seal = verify_checksum_manifest(bundle_root)
    if seal["sha256sums_sha256"] != expected_bundle_sha256s:
        raise R10_4DContractError("DETECTOR_BUNDLE_SEAL_MISMATCH")
    missing = sorted(name for name in REQUIRED_BUNDLE_FILES if not (bundle_root / name).is_file())
    if missing:
        raise R10_4DContractError(f"DETECTOR_BUNDLE_REQUIRED_FILE_MISSING:{missing}")

    checkpoint_path = bundle_root / "full_fit_deploy.pt"
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha256:
        raise R10_4DContractError("DETECTOR_CHECKPOINT_SHA_MISMATCH")

    feature_contract = _json(bundle_root / "feature_contract.json")
    if not _contains_exact_feature_order(feature_contract):
        raise R10_4DContractError("DETECTOR_FEATURE_ORDER_NOT_BOUND")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model_state" not in checkpoint or "frozen" not in checkpoint:
        raise R10_4DContractError("DETECTOR_CHECKPOINT_SCHEMA_FAIL")
    frozen = checkpoint["frozen"]
    for key in ("input_dim", "hidden_dim", "num_layers"):
        if int(frozen.get(key, -1)) != int(FROZEN[key]):
            raise R10_4DContractError(f"DETECTOR_ARCH_FIELD_FAIL:{key}")
    for key in ("grasp_threshold", "grasp_persistence", "guard_param", "max_episode_emits"):
        if key in frozen and float(frozen[key]) != float(FROZEN[key]):
            raise R10_4DContractError(f"DETECTOR_FROZEN_FIELD_FAIL:{key}")

    for key in ("source_commit", "trainer_blob_sha256", "feature_contract_sha256"):
        value = checkpoint.get(key)
        expected_length = 40 if key == "source_commit" else 64
        if not isinstance(value, str) or len(value) != expected_length:
            raise R10_4DContractError(f"DETECTOR_PROVENANCE_FIELD_FAIL:{key}")
        if key != "source_commit" and not CHECKPOINT_SHA_RE.fullmatch(value):
            raise R10_4DContractError(f"DETECTOR_PROVENANCE_DIGEST_FAIL:{key}")

    model = RoutedGraspDetector(
        input_dim=int(frozen["input_dim"]),
        hidden_dim=int(frozen["hidden_dim"]),
        num_layers=int(frozen["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 46658:
        raise R10_4DContractError(f"DETECTOR_PARAM_COUNT_FAIL:{parameter_count}")

    metadata = {
        "bundle_root": str(bundle_root),
        "bundle_sha256s_sha256": seal["sha256sums_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha,
        "parameter_count": parameter_count,
        "source_commit": checkpoint["source_commit"],
        "trainer_blob_sha256": checkpoint["trainer_blob_sha256"],
        "feature_contract_sha256": checkpoint["feature_contract_sha256"],
    }
    return DetectorRuntime(model, device), metadata


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical action semantics (Gate F1.1c — do not change behavior)
# ═══════════════════════════════════════════════════════════════════════════════
# Feature contract: feature[0] = raw_gripper_command (OpenVLA space [0,1]).
# Feature[12] = duplicated raw value, NOT env action. This is frozen per the
# R10.3 training checkpoint input distribution. Do not reinterpret feature[12]
# as env-space gripper without retraining.

def raw_gripper_is_close(raw: float) -> bool:
    """OpenVLA raw action space: raw in [0,1], close = raw < 0.5."""
    return float(raw) < 0.5


def postprocess_gripper(raw: float) -> float:
    """Official OpenVLA→LIBERO gripper postprocess.

    env = -sign(2 * raw - 1)
    raw=0.0 → env=+1.0 (CLOSE)
    raw=0.5 → env= 0.0 (BOUNDARY — neither close nor open)
    raw=1.0 → env=-1.0 (OPEN)
    """
    return float(-np.sign(2.0 * float(raw) - 1.0))


def env_gripper_is_close(env: float) -> bool:
    """LIBERO action space: env in [-1,1], close = env > 0."""
    return float(env) > 0.0


def close_semantics_parity(raw: float, env: float) -> bool:
    """Both action spaces agree on close/open (excluding boundary raw=0.5)."""
    if abs(float(raw) - 0.5) <= 1e-6:
        return False  # boundary — neither close nor open in env space
    return raw_gripper_is_close(raw) == env_gripper_is_close(env)


# ═══════════════════════════════════════════════════════════════════════════════
# Core passive episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_passive_episode(
    *,
    env: Any,
    initial_state: Any,
    task_language: str,
    identity: str,
    openvla_adapter: Any,
    detector: DetectorRuntime,
    image_getter: Callable[[Any], np.ndarray],
    max_steps: int,
    authorized_parents: frozenset[str] | None = None,
    privileged_observer: Callable[[Any, Any, int], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run exactly one passive episode without changing any clean action.

    Gate E-R2.5B: Accepts explicit ``authorized_parents`` (frozenset).
    If None, falls back to ``SUPPORTED_PARENT`` for backward compatibility.
    Produces ``R10_4E_SINGLE_EPISODE_PASSIVE_RESULT_V1`` with termination
    classification, safe info, and extended metadata.
    """
    route = parse_route(identity)
    if route != "multi_object_transfer":
        raise R10_4DContractError(f"PASSIVE_PARENT_ROUTE_FAIL:{identity}:{route}")

    allowed = authorized_parents if authorized_parents is not None else frozenset({SUPPORTED_PARENT})
    if identity not in allowed:
        raise R10_4DContractError(f"PASSIVE_PARENT_NOT_AUTHORIZED:{identity}")

    if not isinstance(max_steps, int) or max_steps < 1:
        raise R10_4DContractError("PASSIVE_MAX_STEPS_INVALID")

    detector.reset()
    fsm = EventFSM()
    feature_adapter = OfficialStreamingFeatureAdapter()
    observation = initialize_env_once(env, initial_state)
    feature_adapter.reset()

    step_records: list[dict[str, Any]] = []
    detector_records: list[dict[str, Any]] = []
    privileged_records: list[dict[str, Any]] = []
    violations: list[str] = []

    for step in range(max_steps):
        image = np.asarray(image_getter(observation))
        if image.dtype != np.uint8:
            raise R10_4DContractError("PASSIVE_IMAGE_NOT_UINT8")

        raw_action, capture = openvla_adapter.predict_action(
            image_np=image,
            task_label=task_language,
            capture=True,
        )
        if not isinstance(capture, Mapping):
            raise R10_4DContractError("PASSIVE_GENERATION_METADATA_TYPE")
        generation_passes = capture.get("generation_passes_per_step")
        if isinstance(generation_passes, bool) or not isinstance(generation_passes, int) or generation_passes != 1:
            raise R10_4DContractError(f"PASSIVE_GENERATION_COUNT:{generation_passes!r}")

        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw_action.shape != (ACTION_DIM,) or not np.isfinite(raw_action).all():
            raise R10_4DContractError("PASSIVE_RAW_ACTION_INVALID")
        postprocess = getattr(openvla_adapter, "postprocess", None)
        if not callable(postprocess):
            raise R10_4DContractError("PASSIVE_OFFICIAL_POSTPROCESS_MISSING")
        clean_env_action = np.asarray(postprocess(raw_action), dtype=np.float32).reshape(-1)
        if clean_env_action.shape != (ACTION_DIM,) or not np.isfinite(clean_env_action).all():
            raise R10_4DContractError("PASSIVE_ENV_ACTION_INVALID")

        stream = feature_adapter.update_from_env(observation, env, raw_action, clean_env_action)
        if not isinstance(stream, Mapping) or stream.get("valid") is not True or not isinstance(stream.get("features"), Mapping):
            raise R10_4DContractError(f"PASSIVE_FEATURE_STREAM_INVALID:{stream.get('error', '') if isinstance(stream, Mapping) else ''}")
        features = np.asarray([stream["features"][name] for name in FEATURE_NAMES], dtype=np.float32)
        if features.shape != (25,) or not np.isfinite(features).all():
            raise R10_4DContractError("PASSIVE_FEATURE_VECTOR_INVALID")

        grasp_logit, grasp_probability = detector.step(features, route)
        close_mask = bool(float(raw_action[-1]) <= 0.5)
        fsm_result = fsm.step(
            step=step,
            grasp_probability=grasp_probability,
            eef_z=float(stream["features"]["eef_z"]),
            close_mask=close_mask,
            route=route,
        )

        executed_action = clean_env_action.copy()
        action_error = float(np.max(np.abs(executed_action - clean_env_action)))
        if action_error != 0.0 or not np.array_equal(executed_action, clean_env_action):
            raise R10_4DContractError(f"PASSIVE_ACTION_MUTATION:{action_error:.9g}")

        if privileged_observer is not None:
            sidecar = dict(privileged_observer(env, observation, step))
            sidecar["step"] = step
            sidecar["detector_input"] = False
            privileged_records.append(sidecar)

        next_observation, reward, done, info = env.step(executed_action.tolist())
        step_records.append(
            {
                "step": step,
                "generation_passes_per_step": generation_passes,
                "raw_action_7d": raw_action.tolist(),
                "clean_env_action_7d": clean_env_action.tolist(),
                "executed_action_7d": executed_action.tolist(),
                "action_max_abs_error": action_error,
                "features_25d": features.tolist(),
                "done": bool(done),
                "reward": float(reward),
                "info": _safe_info(info),
            }
        )
        env_gripper = float(postprocess_gripper(raw_action[-1])) if raw_action.size > 0 else 0.0
        detector_records.append(
            {
                "step": step,
                "route": route,
                "grasp_logit": grasp_logit,
                "grasp_probability": grasp_probability,
                "raw_gripper": float(raw_action[-1]) if raw_action.size > 0 else 0.0,
                "env_gripper": env_gripper,
                "raw_close": raw_gripper_is_close(float(raw_action[-1])) if raw_action.size > 0 else False,
                "env_close": env_gripper_is_close(env_gripper),
                "close_semantics_parity": close_semantics_parity(
                    float(raw_action[-1]) if raw_action.size > 0 else 0.5, env_gripper),
                "close_source": "OPENVLA_RAW_ACTION",
                "close_mask": close_mask,
                **fsm_result,
            }
        )
        observation = next_observation
        if done:
            break

    emit_count = sum(1 for row in detector_records if row.get("emit") is True)
    if emit_count > FROZEN["max_episode_emits"]:
        violations.append(f"DUPLICATE_EMIT:{emit_count}")
    if any(float(row["action_max_abs_error"]) != 0.0 for row in step_records):
        violations.append("ACTION_PARITY")
    if any(int(row["generation_passes_per_step"]) != 1 for row in step_records):
        violations.append("GENERATION_COUNT")

    # ── Gate E-R2.5B: proper termination classification ──
    termination = _classify_termination(
        step_records=step_records,
        configured_horizon=max_steps,
        env=env,
        violations=violations,
    )

    if termination["is_hard_failure"]:
        status = "FAIL_TERMINATION"
    elif violations:
        status = "FAIL_RUNTIME"
    elif emit_count > 0:
        status = "PASS_RUNTIME_EMIT_OBSERVED"
    else:
        status = "PASS_RUNTIME_NO_EMIT"

    return {
        "schema": "R10_4E_SINGLE_EPISODE_PASSIVE_RESULT_V1",
        "identity": identity,
        "status": status,
        "n_steps": len(step_records),
        "emit_count": emit_count,
        "task_success": termination["task_success"],
        "termination_reason": termination["termination_reason"],
        "configured_horizon": termination["configured_horizon"],
        "observed_horizon": termination["observed_horizon"],
        "done": termination["done"],
        "env_check_success": termination["env_check_success"],
        "check_success_error": termination["check_success_error"],
        "simulator_timestep": termination["simulator_timestep"],
        "violations": violations,
        "step_records": step_records,
        "detector_records": detector_records,
        "privileged_records": privileged_records,
        "privileged_runtime_input": False,
        "action_mutation": False,
    }


__all__ = [
    "DetectorRuntime",
    "EventFSM",
    "FROZEN",
    "R10_4DContractError",
    "RoutedGraspDetector",
    "SUPPORTED_PARENT",
    "close_semantics_parity",
    "env_gripper_is_close",
    "load_detector_bundle",
    "parse_route",
    "postprocess_gripper",
    "raw_gripper_is_close",
    "run_passive_episode",
    "safe_json_value",
    "validate_authorization_receipt",
]
