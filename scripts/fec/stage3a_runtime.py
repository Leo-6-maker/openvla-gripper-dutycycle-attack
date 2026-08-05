"""Frozen Stage 2 R2 shadow runtime for the Stage 3A rollout wrapper.

The runtime accepts only clean policy action and current observation fields.
It has no attack outcome, teacher, reward, success, future-state, or guard
hook, so its output cannot affect rollout control.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gripper_attack.action_contract import action_semantics_parity
from gripper_attack.d8_streaming_features_v3 import D8StreamingFeatureAdapterV3, FEATURE_NAMES
from detector_v5.d8_train_core import apply_normalization, create_model, load_checkpoint


class ShadowContractError(RuntimeError):
    """Raised when frozen detector provenance or causal inputs are invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ShadowContractError(f"{name} must be finite with shape ({size},)")
    return array


class FrozenStage2R2DetectorRuntime:
    """CPU shadow inference plus the exact frozen R2 scheduler."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        freeze_receipt_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_scheduler_sha256: str,
        expected_source_commit: str,
        expected_source_tree: str,
        episode_id: str = "",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        self.freeze_receipt_path = Path(freeze_receipt_path).resolve(strict=True)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.scheduler_sha256 = sha256_file(self.freeze_receipt_path)
        if self.checkpoint_sha256 != expected_checkpoint_sha256.lower():
            raise ShadowContractError(
                f"checkpoint SHA mismatch: {self.checkpoint_sha256} != {expected_checkpoint_sha256}"
            )
        if self.scheduler_sha256 != expected_scheduler_sha256.lower():
            raise ShadowContractError(
                f"scheduler SHA mismatch: {self.scheduler_sha256} != {expected_scheduler_sha256}"
            )

        receipt = json.loads(self.freeze_receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "D8_DETECTOR_FREEZE_RECEIPT_R2_V1":
            raise ShadowContractError("unexpected Stage 2 R2 receipt schema")
        if receipt.get("status") != "SHADOW_PROBE_ONLY":
            raise ShadowContractError("Stage 2 R2 receipt is not shadow-only")
        if receipt.get("authorization_mode") != "SHADOW_PROBE_ONLY":
            raise ShadowContractError("Stage 2 R2 receipt authorization is not shadow-only")
        if receipt.get("guard_deployment_authorized") is not False:
            raise ShadowContractError("guard deployment is unexpectedly authorized")
        if receipt.get("source_commit") != expected_source_commit or receipt.get("source_tree") != expected_source_tree:
            raise ShadowContractError("Stage 2 R2 source commit/tree binding mismatch")
        if receipt.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise ShadowContractError("receipt checkpoint SHA does not match checkpoint bytes")
        scheduler = receipt.get("scheduler")
        if not isinstance(scheduler, Mapping):
            raise ShadowContractError("receipt does not contain a scheduler freeze")
        self.receipt = receipt
        self.scheduler = {
            "threshold": float(scheduler["threshold"]),
            "persistence": int(scheduler["persistence"]),
            "hysteresis": float(scheduler["hysteresis"]),
            "cooldown": int(scheduler["cooldown"]),
        }
        if (
            not math.isfinite(self.scheduler["threshold"])
            or self.scheduler["persistence"] < 1
            or self.scheduler["hysteresis"] < 0
            or self.scheduler["cooldown"] < 0
        ):
            raise ShadowContractError("invalid frozen scheduler")

        schema_path = ROOT / "configs" / "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if schema.get("schema") != "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA_V2":
            raise ShadowContractError("25D causal schema is not frozen V2")
        if schema.get("dimensions") != 25 or schema.get("causal_only") is not True:
            raise ShadowContractError("25D causal schema dimensions/causality mismatch")
        if any(schema.get(key) != 0 for key in ("future_fields", "teacher_label_fields", "attack_outcome_fields")):
            raise ShadowContractError("frozen schema contains forbidden input fields")
        if list(FEATURE_NAMES) != [row["name"] for row in schema.get("features", [])]:
            raise ShadowContractError("feature order does not match frozen schema")

        self.model = create_model(seed=20260717).to("cpu")
        checkpoint = load_checkpoint(self.checkpoint_path, self.model, map_location="cpu")
        self.model.eval()
        if self.model.feature_dim != 25:
            raise ShadowContractError("checkpoint model feature dimension is not 25")
        if checkpoint.get("feature_schema_sha256") not in ("", sha256_file(schema_path)):
            raise ShadowContractError("checkpoint feature schema binding mismatch")
        if checkpoint.get("executable_source_commit") not in ("", expected_source_commit):
            raise ShadowContractError("checkpoint source commit binding mismatch")
        if checkpoint.get("executable_source_tree") not in ("", expected_source_tree):
            raise ShadowContractError("checkpoint source tree binding mismatch")
        norm = checkpoint.get("normalization")
        if not isinstance(norm, Mapping) or norm.get("schema") != "D8_NORMALIZATION_V2":
            raise ShadowContractError("checkpoint normalization schema mismatch")
        if norm.get("feature_dim") != 25:
            raise ShadowContractError("checkpoint normalization feature dimension is not 25")
        mean = np.asarray(norm.get("mean"), dtype=np.float64)
        std = np.asarray(norm.get("std"), dtype=np.float64)
        if mean.shape != (25,) or std.shape != (25,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ShadowContractError("checkpoint normalization is non-finite or malformed")
        self.normalization = dict(norm)
        self.episode_id = str(episode_id)
        self.reset_episode()

    def reset_episode(self, episode_id: str | None = None) -> None:
        if episode_id is not None:
            self.episode_id = str(episode_id)
        self.adapter = D8StreamingFeatureAdapterV3()
        self.previous_eef: np.ndarray | None = None
        self._previous_step: int | None = None
        self.consecutive = 0
        self.latched_active = False
        self.next_allowed = -10**9
        self._trace: list[dict[str, Any]] = []

    def step(
        self,
        *,
        episode_id: str,
        policy_step: int,
        raw_action: Any,
        env_action: Any,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Score clean policy inputs before any attack action is constructed."""
        if not isinstance(observation, Mapping):
            raise ShadowContractError("observation must be a mapping")
        step = int(policy_step)
        if step < 0 or (self._previous_step is not None and step != self._previous_step + 1):
            raise ShadowContractError("shadow policy steps are not contiguous")
        raw = _finite_vector(raw_action, 7, "raw_action")
        env = _finite_vector(env_action, 7, "env_action")
        if not 0.0 <= float(raw[6]) <= 1.0 or not -1.0 <= float(env[6]) <= 1.0:
            raise ShadowContractError("gripper action is outside the frozen semantic range")
        if not action_semantics_parity(float(raw[6]), float(env[6])):
            raise ShadowContractError("raw/env gripper semantics are inconsistent or boundary-valued")
        qpos = _finite_vector(observation.get("robot0_gripper_qpos"), 2, "robot0_gripper_qpos")
        eef = _finite_vector(observation.get("robot0_eef_pos"), 3, "robot0_eef_pos")
        velocity = np.zeros(3, dtype=np.float64) if self.previous_eef is None else eef - self.previous_eef
        self.previous_eef = eef
        feature_record = self.adapter.update(
            step_id=step,
            raw_gripper=float(raw[6]),
            env_gripper=float(env[6]),
            gripper_qpos=float(qpos[0] + qpos[1]),
            gripper_opening_proxy=float(abs(qpos[0]) + abs(qpos[1])),
            eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
            eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
            action_dx=float(raw[0]), action_dy=float(raw[1]), action_dz=float(raw[2]),
            action_gripper=float(env[6]),
        )
        if feature_record.get("valid") is not True:
            raise ShadowContractError(f"D8 V3 rejected causal inputs: {feature_record.get('error')}")
        features = np.asarray([feature_record["features"][name] for name in FEATURE_NAMES], dtype=np.float32)
        if features.shape != (25,) or not np.isfinite(features).all():
            raise ShadowContractError("runtime feature vector must be finite with dimension 25")

        x = torch.from_numpy(features).to(dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit_tensor = self.model(apply_normalization(x, self.normalization))
        logit = float(logit_tensor.reshape(-1)[0].item())
        if not math.isfinite(logit):
            raise ShadowContractError("runtime detector logit is non-finite")
        threshold = self.scheduler["threshold"]
        above = logit > threshold
        if above and self._previous_step is not None and step == self._previous_step + 1:
            self.consecutive += 1
        else:
            self.consecutive = 1 if above else 0
        release = self.latched_active and logit < threshold - self.scheduler["hysteresis"]
        if release:
            self.latched_active = False
        emission = False
        if not self.latched_active and self.consecutive >= self.scheduler["persistence"] and step >= self.next_allowed:
            emission = True
            self.latched_active = True
            self.next_allowed = step + self.scheduler["cooldown"]
        row = {
            "episode_id": str(episode_id),
            "policy_step": step,
            "raw_action": raw.astype(np.float32).tolist(),
            "env_action": env.astype(np.float32).tolist(),
            "input_action_source": "clean_policy_action_before_attack",
            "features_25d": features.tolist(),
            "feature_schema": "DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA_V2",
            "logit": logit,
            "threshold": threshold,
            "above_threshold": bool(above),
            "consecutive": int(self.consecutive),
            "consecutive_positive": int(self.consecutive),
            "latched_active": bool(self.latched_active),
            "emission": bool(emission),
            "release": bool(release),
            "cooldown_remaining": max(int(self.next_allowed) - step, 0),
            "evaluation_detector_affects_action": False,
            "evaluation_detector_affects_timing": False,
            "evaluation_detector_affects_termination": False,
        }
        self._trace.append(row)
        self._previous_step = step
        return row

    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

