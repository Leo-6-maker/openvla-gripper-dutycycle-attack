"""OpenVLA action re-decode helpers for adversarial prepared inputs.

TokenPrefixPGD intentionally returns ``action_adv=None`` because the attacked
image must be sent through OpenVLA generation again.  This module provides the
small, explicit re-decode path used by diagnostics so callers do not silently
fall back to zero actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class OpenVLARedecodeResult:
    """Decoded OpenVLA action and generation metadata."""

    action: np.ndarray
    token_ids: np.ndarray
    runtime_sec: float
    generation: Any
    model_dtype: str
    pixel_values_dtype: str


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("OpenVLA model has no parameters; cannot infer decode device") from exc
    except AttributeError as exc:
        raise ValueError("OpenVLA model does not expose parameters(); cannot infer decode device") from exc


def _model_float_dtype(model: Any) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration as exc:
        raise ValueError("OpenVLA model has no parameters; cannot infer decode dtype") from exc
    except AttributeError as exc:
        raise ValueError("OpenVLA model does not expose parameters(); cannot infer decode dtype") from exc


def validate_adv_inputs(adv_inputs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate and copy adversarial prepared inputs.

    The returned mapping is shallow-copied so decode-time device moves cannot
    mutate ``AttackResult.debug["adv_inputs"]``.
    """

    if not isinstance(adv_inputs, Mapping):
        raise ValueError("adv_inputs must be a mapping with pixel_values and input_ids")
    missing = [key for key in ("pixel_values", "input_ids") if key not in adv_inputs]
    if missing:
        raise ValueError(f"adv_inputs missing required keys: {missing}")
    pixel_values = adv_inputs["pixel_values"]
    input_ids = adv_inputs["input_ids"]
    if not torch.is_tensor(pixel_values):
        raise ValueError("adv_inputs['pixel_values'] must be a torch.Tensor")
    if not torch.is_tensor(input_ids):
        raise ValueError("adv_inputs['input_ids'] must be a torch.Tensor")
    if not torch.is_floating_point(pixel_values):
        raise ValueError("adv_inputs['pixel_values'] must be floating point")
    if input_ids.ndim != 2:
        raise ValueError("adv_inputs['input_ids'] must have shape [batch, tokens]")
    if pixel_values.numel() == 0 or input_ids.numel() == 0:
        raise ValueError("adv_inputs tensors must be non-empty")
    return dict(adv_inputs)


def _move_inputs_preserving_float_dtype(inputs: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _ensure_trailing_action_prefix_token(inputs: Dict[str, Any], token_id: int = 29871) -> None:
    input_ids = inputs.get("input_ids")
    if not torch.is_tensor(input_ids):
        return
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    if not torch.all(input_ids[:, -1] == token_id):
        suffix = torch.full((input_ids.shape[0], 1), int(token_id), dtype=input_ids.dtype, device=input_ids.device)
        inputs["input_ids"] = torch.cat((input_ids, suffix), dim=1)


def decode_openvla_generation_to_action(model: Any, generation: Any, unnorm_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode OpenVLA generated action tokens into a continuous action."""

    if generation is None or not hasattr(generation, "sequences"):
        raise ValueError("OpenVLA generation result missing sequences")
    action_dim = int(model.get_action_dim(unnorm_key))
    if action_dim <= 0:
        raise ValueError(f"invalid OpenVLA action_dim={action_dim}")
    sequences = generation.sequences
    if not torch.is_tensor(sequences):
        raise ValueError("OpenVLA generation sequences must be a torch.Tensor")
    if sequences.ndim != 2 or sequences.shape[0] < 1 or sequences.shape[1] < action_dim:
        raise ValueError("OpenVLA generation sequences do not contain enough action tokens")

    token_ids = sequences[0, -action_dim:].detach().cpu().numpy().astype(np.int64)
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    bin_centers = np.asarray(model.bin_centers, dtype=np.float32)
    if bin_centers.ndim != 1 or bin_centers.size == 0:
        raise ValueError("model.bin_centers must be a non-empty 1D array")
    discretized = np.clip(vocab_size - token_ids - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    norm_actions = bin_centers[discretized]

    stats = model.get_action_stats(unnorm_key)
    if "q01" not in stats or "q99" not in stats:
        raise ValueError("OpenVLA action stats must contain q01 and q99")
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    if low.shape[0] != action_dim or high.shape[0] != action_dim or mask.shape[0] != action_dim:
        raise ValueError("OpenVLA action stats dimension mismatch")

    action = np.where(mask, 0.5 * (norm_actions + 1.0) * (high - low) + low, norm_actions).astype(np.float32)
    if action.shape[0] != action_dim:
        raise ValueError("decoded OpenVLA action dimension mismatch")
    if not np.all(np.isfinite(action)):
        raise ValueError("decoded OpenVLA action contains NaN/Inf")
    return action, token_ids


def redecode_openvla_action_from_adv_inputs(
    *,
    model: Any,
    processor: Any = None,
    adv_inputs: Mapping[str, Any],
    instruction: str = "",
    unnorm_key: str,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
) -> OpenVLARedecodeResult:
    """Generate and decode OpenVLA action from adversarial prepared inputs.

    ``processor`` and ``instruction`` are accepted for call-site parity with the
    normal OpenVLA runner; the actual re-decode consumes already-prepared
    ``input_ids`` and ``pixel_values`` from ``adv_inputs``.
    """

    _ = processor, instruction
    inputs = validate_adv_inputs(adv_inputs)
    device = _model_device(model)
    model_dtype = _model_float_dtype(model)
    pixel_dtype = str(inputs["pixel_values"].dtype)
    inputs = _move_inputs_preserving_float_dtype(inputs, device)
    _ensure_trailing_action_prefix_token(inputs)

    action_dim = int(model.get_action_dim(unnorm_key))
    gen_kwargs = dict(generation_kwargs or {})
    gen_kwargs.setdefault("max_new_tokens", action_dim)
    gen_kwargs.setdefault("do_sample", False)
    gen_kwargs.setdefault("return_dict_in_generate", True)
    gen_kwargs.setdefault("output_scores", True)

    with torch.inference_mode():
        t0 = time.time()
        generation = model.generate(**inputs, **gen_kwargs)
        runtime_sec = time.time() - t0
    action, token_ids = decode_openvla_generation_to_action(model, generation, unnorm_key)
    return OpenVLARedecodeResult(
        action=action,
        token_ids=token_ids,
        runtime_sec=float(runtime_sec),
        generation=generation,
        model_dtype=str(model_dtype),
        pixel_values_dtype=pixel_dtype,
    )
