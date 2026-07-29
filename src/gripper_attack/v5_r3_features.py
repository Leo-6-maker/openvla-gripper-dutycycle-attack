"""Frozen FIT670 -> official 25D causal feature materialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .sc5_streaming_features_v2 import FEATURE_NAMES, SC5StreamingFeatureAdapterV2
from .action_contract import action_semantics_parity, raw_gripper_is_close


FEATURE_ORDER = tuple(FEATURE_NAMES)
ACTION_GRIPPER_SOURCE = "raw_action_7d[6] via SC5StreamingFeatureAdapterV2 implementation"


def load_feature_binding(binding_path: Path, source_root: Path) -> dict[str, Any]:
    """Validate the frozen R3 binding before any feature materialization."""
    data = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_order = list(FEATURE_ORDER)
    order_sha = hashlib.sha256(json.dumps(expected_order, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if data.get("schema") != "V5_R3_SC5_FEATURE_BINDING_V1":
        raise ValueError("unexpected R3 feature binding schema")
    if data.get("status") != "FROZEN_ENGINEERING_BINDING":
        raise ValueError("R3 feature binding is not frozen")
    if data.get("feature_order") != expected_order or data.get("feature_order_sha256") != order_sha:
        raise ValueError("R3 feature-order binding mismatch")
    adapter_source = data.get("adapter_source")
    if not isinstance(adapter_source, str) or Path(adapter_source).is_absolute() or ".." in Path(adapter_source).parts:
        raise ValueError("unsafe R3 adapter source path")
    adapter_path = source_root / adapter_source
    if adapter_path.is_symlink() or not adapter_path.is_file():
        raise ValueError("R3 adapter source is missing")
    if data.get("adapter_source_hash_algorithm") != "SHA256" or data.get("adapter_source_hash_normalization") != "UTF-8 text with CRLF normalized to LF":
        raise ValueError("R3 adapter source hash normalization is not frozen")
    canonical_source = adapter_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    adapter_sha = hashlib.sha256(canonical_source).hexdigest()
    if data.get("adapter_source_sha256") != adapter_sha:
        raise ValueError("R3 adapter source SHA mismatch")
    action = data.get("action_gripper")
    if not isinstance(action, dict) or action.get("source") != "raw_action_7d[6]" or action.get("range") != [0.0, 1.0]:
        raise ValueError("R3 action_gripper binding mismatch")
    if data.get("future_fields_used") is not False or data.get("teacher_fields_used") is not False or data.get("outcome_fields_used") is not False or data.get("attack_enabled") is not False:
        raise ValueError("R3 causal/attack binding is not fail-closed")
    legacy = data.get("legacy_schema_conflict")
    if not isinstance(legacy, dict) or legacy.get("path") != "configs/v2_sc5_schema_aliases.yaml" or legacy.get("status") != "NOT_CONSUMABLE_FOR_R3":
        raise ValueError("R3 legacy schema conflict is not explicit")
    return data


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return array


def materialize_fit670_features(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the exact SC5 25D stream from recorded causal runtime fields.

    Only the current step and the previously emitted EEF position are used.
    The adapter itself remains the single source of feature ordering and
    derived-feature semantics.
    """
    steps = episode.get("steps")
    telemetry = episode.get("telemetry")
    if not isinstance(steps, list) or not isinstance(telemetry, list) or len(steps) != len(telemetry) or not steps:
        raise ValueError("FIT670 steps/telemetry closure is incomplete")
    adapter = SC5StreamingFeatureAdapterV2()
    previous_eef: np.ndarray | None = None
    output: list[dict[str, Any]] = []
    for expected_step, (step, state) in enumerate(zip(steps, telemetry)):
        if not isinstance(step, Mapping) or not isinstance(state, Mapping):
            raise ValueError(f"malformed FIT670 feature row at step {expected_step}")
        if step.get("step") != expected_step or state.get("step") != expected_step:
            raise ValueError(f"FIT670 feature step closure failed at {expected_step}")
        raw_action = _finite_vector(step.get("raw_action_7d"), 7, "raw_action_7d")
        duplicate_raw = step.get("action_raw_7d")
        if duplicate_raw is not None and not np.array_equal(raw_action, _finite_vector(duplicate_raw, 7, "action_raw_7d")):
            raise ValueError(f"raw action aliases disagree at step {expected_step}")
        env_action = _finite_vector(step.get("action_env_7d"), 7, "action_env_7d")
        if not 0.0 <= float(raw_action[6]) <= 1.0:
            raise ValueError(f"raw gripper is outside [0,1] at step {expected_step}")
        if not -1.0 <= float(env_action[6]) <= 1.0:
            raise ValueError(f"env gripper is outside [-1,1] at step {expected_step}")
        if not action_semantics_parity(float(raw_action[6]), float(env_action[6])):
            raise ValueError(f"raw/env gripper semantics are boundary or inconsistent at step {expected_step}")
        qpos = _finite_vector(state.get("robot0_gripper_qpos"), 2, "robot0_gripper_qpos")
        eef = _finite_vector(state.get("robot0_eef_pos"), 3, "robot0_eef_pos")
        if previous_eef is None:
            velocity = np.zeros(3, dtype=np.float64)
        else:
            velocity = eef - previous_eef
        previous_eef = eef
        result = adapter.update(
            step_id=expected_step,
            raw_gripper=float(raw_action[6]),
            env_gripper=float(env_action[6]),
            gripper_qpos=float(abs(qpos[0]) + abs(qpos[1])),
            gripper_opening_proxy=float(abs(qpos[0]) + abs(qpos[1])),
            eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
            eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
            action_dx=float(raw_action[0]), action_dy=float(raw_action[1]), action_dz=float(raw_action[2]),
            action_gripper=float(raw_action[6]),
        )
        if not result.get("valid"):
            raise ValueError(f"SC5 feature adapter rejected step {expected_step}: {result.get('error')}")
        values = result.get("features")
        vector = np.asarray([values[name] for name in FEATURE_ORDER], dtype=np.float32)
        if vector.shape != (25,) or not np.isfinite(vector).all():
            raise ValueError(f"invalid 25D feature vector at step {expected_step}")
        action_gripper_index = FEATURE_ORDER.index("action_gripper")
        if not np.isclose(vector[action_gripper_index], raw_action[6], rtol=0.0, atol=1e-6):
            raise ValueError(f"SC5 action_gripper source mismatch at step {expected_step}")
        output.append({
            "step": expected_step,
            "features_25d": vector.tolist(),
            "candidate_close": bool(raw_gripper_is_close(float(raw_action[6]))),
            "feature_schema": "SC5StreamingFeatureAdapterV2_25D",
            "feature_order": list(FEATURE_ORDER),
            "feature_source": "FIT670.step+telemetry -> SC5StreamingFeatureAdapterV2",
            "action_gripper_source": ACTION_GRIPPER_SOURCE,
        })
    return output
