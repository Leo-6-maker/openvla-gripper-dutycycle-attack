"""Deployment runtime for the clean-only C2g gripper-critical-window detector.

The runtime is called after the clean OpenVLA decode and before any visual attack.
It consumes only clean causal history, clean gripper logits, the current clean RGB
frame, and task language.  A stateful fixed-burst scheduler selects the attack
start; the attack payload remains TokenPrefixPGDAttacker in the execution runner.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .c2g_clean_policy_signals import (
    CLEAN_POLICY_FEATURE_NAMES,
    summarize_clean_gripper_logits,
)
from .c2g_gripper_critical_window_detector import (
    C2gDetectorConfig,
    C2gGripperCriticalWindowDetector,
    FixedBurstTriggerScheduler,
)
from .openvla_libero_exec_spec import (
    close_token_ids_from_decoded_action,
    open_token_ids_from_decoded_action,
    validate_open_close_token_sets,
)

CHECKPOINT_SCHEMA_VERSION = "c2g.clean_window_checkpoint.2026-07-10.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_gripper_token_semantics(model: Any, unnorm_key: str) -> dict[str, Any]:
    """Derive and validate OPEN/CLOSE action-token sets from executable decoding.

    The mapping uses the model's own bin centers and per-suite action statistics.
    It never assumes a fixed target token id.
    """

    centers = np.asarray(model.bin_centers, dtype=np.float32).reshape(-1)
    stats = model.get_action_stats(unnorm_key)
    low = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    high = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool).reshape(-1)
    if low.size == 0 or high.shape != low.shape or mask.shape != low.shape:
        raise ValueError("invalid OpenVLA action statistics")
    gripper_index = low.size - 1
    if bool(mask[gripper_index]):
        decoded = 0.5 * (centers + 1.0) * (high[gripper_index] - low[gripper_index]) + low[gripper_index]
    else:
        decoded = centers.copy()
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    token_action_map = {
        int(vocab_size - bin_index - 1): float(raw)
        for bin_index, raw in enumerate(decoded)
    }
    open_ids = open_token_ids_from_decoded_action(token_action_map)
    close_ids = close_token_ids_from_decoded_action(token_action_map)
    validate_open_close_token_sets(open_ids, close_ids, token_action_map)
    canonical = "".join(f"{token}:{token_action_map[token]:.9g}\n" for token in sorted(token_action_map))
    return {
        "open_token_ids": open_ids,
        "close_token_ids": close_ids,
        "token_action_map": token_action_map,
        "token_semantics_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "unnorm_key": str(unnorm_key),
    }


class C2gCleanWindowRuntime:
    """Checkpoint-bound online feature extraction, inference, and scheduling."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        openvla_model: Any,
        openvla_processor: Any,
        unnorm_key: str,
        device: str = "cuda",
        burst_length: int = 10,
        require_clean_close: bool = True,
        minimum_open_minus_close_log_mass: float = -8.0,
        minimum_entropy: float = 0.0,
    ) -> None:
        checkpoint_path = Path(checkpoint_path).resolve()
        raw = torch.load(checkpoint_path, map_location="cpu")
        if str(raw.get("schema_version", "")) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported C2g clean-window checkpoint schema")
        config_dict = dict(raw["model_config"])
        self.config = C2gDetectorConfig(**config_dict)
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model = C2gGripperCriticalWindowDetector(self.config).to(self.device)
        self.model.load_state_dict(raw["model_state_dict"], strict=True)
        self.model.eval()
        self.window = int(raw["window"])
        if self.window <= 0:
            raise ValueError("checkpoint window must be positive")
        thresholds = dict(raw.get("thresholds", {}))
        self.scheduler = FixedBurstTriggerScheduler(
            burst_length=burst_length,
            tau_critical=float(thresholds.get("tau_critical", 0.5)),
            tau_release=float(thresholds.get("tau_release", 0.5)),
            tau_ground=float(thresholds.get("tau_ground", 0.5)),
            persistence_window=int(thresholds.get("persistence_window", 3)),
            persistence_required=int(thresholds.get("persistence_required", 2)),
            one_shot=True,
        )
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self._vla = openvla_model
        self._processor = openvla_processor
        self._unnorm_key = str(unnorm_key)
        self._token_semantics = derive_gripper_token_semantics(openvla_model, self._unnorm_key)
        self.require_clean_close = bool(require_clean_close)
        self.minimum_open_minus_close_log_mass = float(minimum_open_minus_close_log_mass)
        self.minimum_entropy = float(minimum_entropy)
        if not math.isfinite(self.minimum_open_minus_close_log_mass):
            raise ValueError("minimum_open_minus_close_log_mass must be finite")
        if not 0.0 <= self.minimum_entropy <= 1.0:
            raise ValueError("minimum_entropy must be in [0,1]")
        self._text_embedding = None
        self._language_cache: dict[str, np.ndarray] = {}
        self.reset()

    @property
    def token_semantics(self) -> Mapping[str, Any]:
        return self._token_semantics

    def reset(self) -> None:
        self._proprio_history: list[np.ndarray] = []
        self._policy_history: list[np.ndarray] = []
        self.scheduler.reset()

    def _encode_text(self, text: str) -> np.ndarray:
        text = str(text).strip()
        if not text:
            raise ValueError("task language cannot be empty")
        if text in self._language_cache:
            return self._language_cache[text]
        if self._text_embedding is None:
            self._text_embedding = self._vla.language_model.get_input_embeddings()
        tokens = self._processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=64,
        )
        input_ids = tokens["input_ids"].to(self.device)
        with torch.no_grad():
            embedding = self._text_embedding(input_ids).float().mean(dim=1)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        value = embedding.cpu().numpy()[0].astype(np.float32)
        if value.shape != (self.config.language_dim,):
            raise ValueError(
                f"runtime language dim {value.shape} differs from checkpoint {self.config.language_dim}"
            )
        self._language_cache[text] = value
        return value

    def _encode_image(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError("RGB must have shape [height,width,channels]")
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if np.nanmax(image) <= 1.0 else image, 0, 255).astype(np.uint8)
        processed = self._processor.image_processor(
            images=Image.fromarray(image[..., :3]).convert("RGB"),
            return_tensors="pt",
        )["pixel_values"]
        try:
            model_dtype = next(self._vla.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        processed = processed.to(device=self.device, dtype=model_dtype)
        with torch.no_grad():
            output = self._vla.vision_backbone(processed)
            if torch.is_tensor(output):
                embedding = output
            elif getattr(output, "pooler_output", None) is not None:
                embedding = output.pooler_output
            elif getattr(output, "last_hidden_state", None) is not None:
                embedding = output.last_hidden_state.mean(dim=1)
            else:
                embedding = output[0]
            if embedding.ndim > 2:
                embedding = embedding.mean(dim=1)
            embedding = embedding.float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        value = embedding.cpu().numpy()[0].astype(np.float32)
        if value.shape != (self.config.visual_dim,):
            raise ValueError(f"runtime visual dim {value.shape} differs from checkpoint {self.config.visual_dim}")
        return value

    def _policy_features(self, clean_gripper_logits: Tensor | np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        logits = torch.as_tensor(clean_gripper_logits, dtype=torch.float32)
        if logits.ndim > 1:
            logits = logits.reshape(-1, logits.shape[-1])[-1]
        summary = summarize_clean_gripper_logits(
            logits,
            open_token_ids=self._token_semantics["open_token_ids"],
            close_token_ids=self._token_semantics["close_token_ids"],
        )
        values = np.asarray(
            [float(summary[name].detach().cpu()) for name in CLEAN_POLICY_FEATURE_NAMES],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            raise ValueError("clean policy-intent features are non-finite")
        return values, {name: float(values[index]) for index, name in enumerate(CLEAN_POLICY_FEATURE_NAMES)}

    def predict(
        self,
        *,
        features_25d: Sequence[float],
        rgb: np.ndarray,
        task_language: str,
        clean_gripper_logits: Tensor | np.ndarray,
    ) -> Dict[str, Any]:
        proprio = np.asarray(features_25d, dtype=np.float32).reshape(-1)
        if proprio.shape != (25,) or not np.isfinite(proprio).all():
            raise ValueError("features_25d must be a finite vector of length 25")
        policy, policy_summary = self._policy_features(clean_gripper_logits)
        self._proprio_history.append(proprio)
        self._policy_history.append(policy)
        self._proprio_history = self._proprio_history[-self.window :]
        self._policy_history = self._policy_history[-self.window :]
        if len(self._proprio_history) < self.window:
            decision = self.scheduler.update(
                critical_probability=0.0,
                release_safe_probability=1.0,
                grounding_confidence_probability=0.0,
                valid=False,
            )
            return {
                "ready": False,
                "outputs": {},
                "policy": policy_summary,
                "susceptibility_gate": False,
                "decision": decision,
            }

        language = self._encode_text(task_language)
        visual = self._encode_image(rgb) if self.config.use_visual else None
        proprio_tensor = torch.from_numpy(np.asarray(self._proprio_history)[None]).to(self.device)
        policy_tensor = torch.from_numpy(np.asarray(self._policy_history)[None]).to(self.device)
        language_tensor = torch.from_numpy(language[None]).to(self.device)
        visual_tensor = torch.from_numpy(visual[None]).to(self.device) if visual is not None else None
        with torch.no_grad():
            logits = self.model(
                proprio_tensor,
                language_tensor,
                policy_intent=policy_tensor if self.config.use_policy_intent else None,
                siglip_visual=visual_tensor if self.config.use_visual else None,
                return_sequence=False,
            )
        probabilities = {name: float(torch.sigmoid(value)[0].cpu()) for name, value in logits.items()}
        clean_close = policy_summary["clean_top1_is_close"] >= 0.5
        margin_ok = (
            policy_summary["clean_open_minus_close_log_mass"]
            >= self.minimum_open_minus_close_log_mass
        )
        entropy_ok = (
            policy_summary["clean_action_token_entropy_normalized"] >= self.minimum_entropy
        )
        susceptibility_gate = bool((clean_close or not self.require_clean_close) and margin_ok and entropy_ok)
        decision = self.scheduler.update(
            critical_probability=probabilities["critical_window"],
            release_safe_probability=probabilities["release_safe"],
            grounding_confidence_probability=probabilities["grounding_confidence"],
            valid=susceptibility_gate,
        )
        return {
            "ready": True,
            "outputs": probabilities,
            "policy": policy_summary,
            "susceptibility_gate": susceptibility_gate,
            "decision": decision,
        }
