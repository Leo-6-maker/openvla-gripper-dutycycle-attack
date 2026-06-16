"""D5 Frozen Online Detector v1 — complete live streaming pipeline.

Wraps D5FrozenFeatureAdapter with frozen normalization, D5 MLP, tau=0.050,
and first-trigger lock. This is the production-ready online detector.

Full chain:
  raw per-step observation
  → D5FrozenFeatureAdapter (frozen feature extraction + abstain)
  → frozen normalization (means/stdevs/impute from D5 checkpoint)
  → frozen D5 MLP (CandidateRanker)
  → abstain gate (abstained candidates NEVER emit)
  → tau=0.050 threshold
  → first-trigger lock (at most one emission per episode)

SHA-bound: refuses to initialize with wrong checkpoint, config, or schema.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Optional

import torch

from .d5_frozen_feature_adapter_v1 import D5FrozenFeatureAdapter, FEATURE_NAMES
from .d5_frozen_runtime_v1 import CandidateRankerV1, normalize_features_v1

# ── Frozen binding ──
FROZEN_CHECKPOINT_SHA = "7eea609f21eae7b91ff790631b656ec88949df8993a89b26b3588468a81e5ee5"
FROZEN_CONFIG_SHA = "d6f6af61e7ec86216e2f689b1806985cce12fdcc35134388b7c6b96789dde1d5"
FROZEN_RUNTIME_SHA = "d8621637287217c08595bf9df635d552e1266b4f6e149c3dd77572f33058811e"
FROZEN_ADAPTER_SHA = "81ee7fd31db5c02fc148b575599f64d29ca76b04739985b2371d34b2521743d3"
FROZEN_TAU = 0.050
DETECTOR_VERSION = "d5_frozen_online_v1"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def _sha256_file(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class D5FrozenOnlineDetectorV1:
    """D5 frozen online detector — complete streaming pipeline with SHA gates.

    Must be initialized with the frozen checkpoint path and config path.
    Refuses to start if any SHA mismatches.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        # ── SHA gates ──
        ckpt_sha = _sha256_file(checkpoint_path)
        if ckpt_sha != FROZEN_CHECKPOINT_SHA:
            raise RuntimeError(
                f"Checkpoint SHA mismatch: got {ckpt_sha[:16]}..., "
                f"expected {FROZEN_CHECKPOINT_SHA[:16]}..."
            )
        cfg_sha = _sha256_file(config_path)
        if cfg_sha != FROZEN_CONFIG_SHA:
            raise RuntimeError(
                f"Config SHA mismatch: got {cfg_sha[:16]}..., "
                f"expected {FROZEN_CONFIG_SHA[:16]}..."
            )

        # ── Load model + normalization ──
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.means = ckpt["means"]
        self.stdevs = ckpt["stdevs"]
        self.impute = ckpt["impute"]

        # Verify feature names
        ckpt_features = ckpt.get("feature_names", [])
        if ckpt_features and ckpt_features != FEATURE_NAMES:
            raise RuntimeError(
                f"Feature name mismatch: checkpoint has {ckpt_features[:3]}..., "
                f"expected {FEATURE_NAMES[:3]}..."
            )

        # Verify frozen runtime SHA (strict binding)
        runtime_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "d5_frozen_runtime_v1.py")
        runtime_sha = _sha256_file(runtime_path)
        if runtime_sha != FROZEN_RUNTIME_SHA:
            raise RuntimeError(
                f"Runtime SHA mismatch: got {runtime_sha[:16]}..., "
                f"expected {FROZEN_RUNTIME_SHA[:16]}..."
            )

        # Verify frozen adapter SHA (strict binding)
        adapter_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "d5_frozen_feature_adapter_v1.py")
        adapter_sha = _sha256_file(adapter_path)
        if adapter_sha != FROZEN_ADAPTER_SHA:
            raise RuntimeError(
                f"Adapter SHA mismatch: got {adapter_sha[:16]}..., "
                f"expected {FROZEN_ADAPTER_SHA[:16]}..."
            )

        self.model = CandidateRankerV1(n_features=16).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device
        self.tau = FROZEN_TAU

        # Config
        with open(config_path) as f:
            cfg = json.load(f)
        cfg_tau = float(cfg.get("tau", -1))
        if abs(cfg_tau - FROZEN_TAU) > 1e-9:
            raise RuntimeError(
                f"Config tau mismatch: got {cfg_tau}, expected {FROZEN_TAU}"
            )

        # Bind SHAs
        self._checkpoint_sha = ckpt_sha
        self._config_sha = cfg_sha
        self._runtime_sha = runtime_sha
        self._adapter_sha = adapter_sha
        self._checkpoint_path = checkpoint_path
        self._config_path = config_path

        self._reset_episode()

    def _reset_episode(self):
        self.adapter = D5FrozenFeatureAdapter()
        self.emit_step = -1
        self.emit_score = 0.0
        self.audit_records = []  # full per-candidate audit

    def reset(self):
        self._reset_episode()

    @property
    def next_expected_step(self) -> int:
        return self.adapter.next_expected_step

    @property
    def has_emitted(self) -> bool:
        return self.emit_step >= 0

    @property
    def history(self):
        """Compatibility with ProductionStreamingDetector interface."""
        return self.adapter.history

    @property
    def candidate_features(self):
        """Compatibility with ProductionStreamingDetector interface."""
        return self.adapter.candidate_features

    def update(self, step_id: int,
               raw_gripper: float, env_gripper: float, gripper_qpos: float,
               eef_x: float, eef_y: float, eef_z: float,
               decoded_open: int,
               raw_valid: bool = True, env_valid: bool = True,
               qpos_valid: bool = True, eef_valid: bool = True,
               gripper_semantics_valid: bool = True) -> Optional[dict]:
        """Process one step. Returns None (no candidate) or full audit dict.

        Raises ValueError on step sequence violation.
        """
        adapter_result = self.adapter.update(
            step_id=step_id,
            raw_gripper=raw_gripper, env_gripper=env_gripper,
            gripper_qpos=gripper_qpos,
            eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
            decoded_open=decoded_open,
            raw_valid=raw_valid, env_valid=env_valid,
            qpos_valid=qpos_valid, eef_valid=eef_valid,
            gripper_semantics_valid=gripper_semantics_valid,
        )

        if adapter_result is None:
            return None

        features = adapter_result["features"]
        abstain = adapter_result["abstain"]
        abstained = adapter_result["abstained"]

        # ── Normalize + score ──
        X = normalize_features_v1([features], self.means, self.stdevs, self.impute)
        X = X.to(self.device)
        norm_vec = [round(float(v), 10) for v in X[0].cpu().tolist()]

        with torch.no_grad():
            score = float(self.model(X).item())

        # ── First-trigger: abstained candidates NEVER emit ──
        emitted = False
        if not abstained and not self.has_emitted and score >= self.tau:
            self.emit_step = step_id
            self.emit_score = score
            emitted = True

        record = {
            "step": step_id,
            "is_candidate": True,
            "features": features,
            "normalized_features": norm_vec,
            "score": round(score, 6),
            "abstain": abstain,
            "abstained": abstained,
            "candidate_reason": adapter_result["candidate_reason"],
            "emitted": emitted,
            "first_emit_step": self.emit_step,
            "detector_version": DETECTOR_VERSION,
            "feature_schema_version": adapter_result["feature_schema_version"],
            "source_commit": adapter_result["source_commit"],
        }
        self.audit_records.append(record)
        return record

    @property
    def bound_manifest(self) -> dict:
        return {
            "detector_version": DETECTOR_VERSION,
            "checkpoint_sha": self._checkpoint_sha,
            "config_sha": self._config_sha,
            "runtime_sha": self._runtime_sha,
            "adapter_sha": self._adapter_sha,
            "checkpoint_path": self._checkpoint_path,
            "config_path": self._config_path,
            "tau": float(self.tau),
            "feature_schema": FEATURE_NAMES,
            "adapter_source_commit": "44bf7b86bafdda79837b4089dd5250901bb3ae75",
        }
