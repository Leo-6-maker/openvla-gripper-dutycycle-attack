"""Prospective Stage-X native action-token authority.

This module deliberately does not modify ``TokenPrefixPGDAttacker``.  The
historical helper uses nearest bin centres and remains available for immutable
reproduction only.  Prospective code must bind a checkpoint-local native
``ActionTokenizer`` through :class:`NativeActionTokenAuthorityV2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .openvla_libero_exec_spec import (
    OPEN_THRESHOLD_RAW,
    classify_env_gripper,
    classify_raw_gripper,
    raw_gripper_to_env_gripper,
)


NATIVE_ACTION_TOKEN_AUTHORITY_VERSION = "STAGE_X_X1R_T1_NATIVE_ACTION_TOKEN_AUTHORITY_V2"
NATIVE_ACTION_TOKEN_ALGORITHM = "clip_normalized_then_np_digitize_native_bins_then_vocab_minus_index"
LEGACY_HELPER_STATUS = "HISTORICAL_COMPATIBILITY_ONLY"


def _array(values: Sequence[float] | np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise ValueError("TOKEN_AUTHORITY_VECTOR_MUST_BE_1D")
    return result


def _mask(stats: Mapping[str, Any], size: int) -> np.ndarray:
    value = stats.get("mask")
    if value is None:
        return np.ones(size, dtype=bool)
    result = np.asarray(value, dtype=bool)
    if result.shape != (size,):
        raise ValueError("TOKEN_AUTHORITY_MASK_SHAPE_MISMATCH")
    return result


@dataclass(frozen=True)
class SuiteActionTokenBinding:
    """Immutable identity and native quantizer data for one suite checkpoint."""

    suite: str
    checkpoint_path: str
    checkpoint_config_sha256: str
    tokenizer_source: str
    tokenizer_source_sha256: str
    model_decoder_source_sha256: str
    tokenizer_files: tuple[tuple[str, str], ...]
    tokenizer_vocab_size: int
    n_action_bins: int
    bins: tuple[float, ...]
    bin_centers: tuple[float, ...]
    q01: tuple[float, ...]
    q99: tuple[float, ...]
    mask: tuple[bool, ...]

    @classmethod
    def from_native_info(
        cls,
        native_info: Mapping[str, Any],
        *,
        checkpoint_config_sha256: str,
        tokenizer_source: str,
        tokenizer_source_sha256: str,
        model_decoder_source_sha256: str,
    ) -> "SuiteActionTokenBinding":
        stats = native_info["stats"]
        bins = _array(native_info["bins"])
        centers = _array(native_info["bin_centers"])
        q01 = _array(stats["q01"])
        q99 = _array(stats["q99"])
        mask = _mask(stats, q01.size)
        tokenizer_files = tuple(sorted((str(k), str(v)) for k, v in (native_info.get("tokenizer_files") or {}).items()))
        binding = cls(
            suite=str(native_info["suite"]),
            checkpoint_path=str(native_info["model_path"]),
            checkpoint_config_sha256=str(checkpoint_config_sha256),
            tokenizer_source=str(tokenizer_source),
            tokenizer_source_sha256=str(tokenizer_source_sha256),
            model_decoder_source_sha256=str(model_decoder_source_sha256),
            tokenizer_files=tokenizer_files,
            tokenizer_vocab_size=int(native_info["tokenizer_vocab_size"]),
            n_action_bins=int(native_info["native"].n_bins),
            bins=tuple(float(x) for x in bins),
            bin_centers=tuple(float(x) for x in centers),
            q01=tuple(float(x) for x in q01),
            q99=tuple(float(x) for x in q99),
            mask=tuple(bool(x) for x in mask),
        )
        binding.validate()
        return binding

    def validate(self) -> None:
        if not self.suite:
            raise ValueError("TOKEN_AUTHORITY_SUITE_REQUIRED")
        if not self.checkpoint_config_sha256 or not self.tokenizer_source_sha256 or not self.model_decoder_source_sha256:
            raise ValueError("TOKEN_AUTHORITY_SOURCE_BINDING_REQUIRED")
        if self.n_action_bins < 2 or len(self.bins) != self.n_action_bins:
            raise ValueError("TOKEN_AUTHORITY_NATIVE_BIN_COUNT_MISMATCH")
        if len(self.bin_centers) != self.n_action_bins - 1:
            raise ValueError("TOKEN_AUTHORITY_NATIVE_CENTER_COUNT_MISMATCH")
        if not (len(self.q01) == len(self.q99) == len(self.mask)):
            raise ValueError("TOKEN_AUTHORITY_STATS_SHAPE_MISMATCH")
        if any(self.bins[i] >= self.bins[i + 1] for i in range(len(self.bins) - 1)):
            raise ValueError("TOKEN_AUTHORITY_BINS_NOT_STRICTLY_INCREASING")


class NativeActionTokenAuthorityV2:
    """Checkpoint-bound implementation of the official native quantizer."""

    version = NATIVE_ACTION_TOKEN_AUTHORITY_VERSION

    def __init__(self, binding: SuiteActionTokenBinding):
        binding.validate()
        self.binding = binding
        self._bins = np.asarray(binding.bins, dtype=np.float64)
        self._centers = np.asarray(binding.bin_centers, dtype=np.float64)
        self._q01 = np.asarray(binding.q01, dtype=np.float64)
        self._q99 = np.asarray(binding.q99, dtype=np.float64)
        self._mask = np.asarray(binding.mask, dtype=bool)

    @property
    def vocab_size(self) -> int:
        return int(self.binding.tokenizer_vocab_size)

    @property
    def action_dim(self) -> int:
        return int(self._q01.size)

    def normalize_raw(self, raw_action: Sequence[float] | np.ndarray) -> np.ndarray:
        raw = _array(raw_action)
        if raw.size != self.action_dim:
            raise ValueError("TOKEN_AUTHORITY_ACTION_DIM_MISMATCH")
        return np.where(self._mask, 2.0 * (raw - self._q01) / np.maximum(self._q99 - self._q01, 1e-6) - 1.0, raw)

    def denormalize(self, normalized_action: Sequence[float] | np.ndarray) -> np.ndarray:
        norm = _array(normalized_action)
        if norm.size != self.action_dim:
            raise ValueError("TOKEN_AUTHORITY_ACTION_DIM_MISMATCH")
        return np.where(self._mask, 0.5 * (norm + 1.0) * (self._q99 - self._q01) + self._q01, norm)

    def encode_normalized(self, normalized_action: Sequence[float] | np.ndarray) -> np.ndarray:
        norm = _array(normalized_action)
        if norm.size != self.action_dim:
            raise ValueError("TOKEN_AUTHORITY_ACTION_DIM_MISMATCH")
        clipped = np.clip(norm, -1.0, 1.0)
        # This is intentionally np.digitize, including its endpoint behavior.
        discretized = np.digitize(clipped, self._bins)
        return (self.vocab_size - discretized).astype(np.int64)

    def encode_raw(self, raw_action: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.encode_normalized(self.normalize_raw(raw_action))

    def decode_normalized(self, token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        tokens = _array(token_ids, dtype=np.int64)
        if tokens.size != self.action_dim:
            raise ValueError("TOKEN_AUTHORITY_TOKEN_DIM_MISMATCH")
        discretized = np.clip(self.vocab_size - tokens - 1, 0, self._centers.size - 1)
        return self._centers[discretized]

    def decode_raw(self, token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        return self.denormalize(self.decode_normalized(token_ids))

    def decode_gripper(self, token_id: int) -> dict[str, Any]:
        tokens = np.zeros(self.action_dim, dtype=np.int64)
        tokens[-1] = int(token_id)
        normalized = float(self.decode_normalized(tokens)[-1])
        raw = float(self.decode_raw(tokens)[-1])
        env = raw_gripper_to_env_gripper(raw)
        return {
            "suite": self.binding.suite,
            "token_id": int(token_id),
            "normalized": normalized,
            "raw": raw,
            "env": env,
            "raw_class": classify_raw_gripper(raw),
            "env_class": classify_env_gripper(env),
        }

    def token_action_map(self) -> dict[int, float]:
        ids = np.arange(self.vocab_size - self.binding.n_action_bins, self.vocab_size + 1, dtype=np.int64)
        normalized = np.clip(self.vocab_size - ids - 1, 0, self._centers.size - 1)
        raw = self.denormalize(np.full(self.action_dim, 0.0))
        gripper_norm = self._centers[normalized]
        if self._mask[-1]:
            gripper_raw = 0.5 * (gripper_norm + 1.0) * (self._q99[-1] - self._q01[-1]) + self._q01[-1]
        else:
            gripper_raw = gripper_norm
        del raw
        return {int(token): float(value) for token, value in zip(ids, gripper_raw)}

    def open_token_id(self, *, raw_target: float = 1.0) -> int:
        if float(raw_target) <= OPEN_THRESHOLD_RAW:
            raise ValueError("TOKEN_AUTHORITY_OPEN_TARGET_MUST_BE_ABOVE_BOUNDARY")
        raw = np.zeros(self.action_dim, dtype=np.float64)
        raw[-1] = float(raw_target)
        token = int(self.encode_raw(raw)[-1])
        decoded = self.decode_gripper(token)
        if decoded["raw_class"] != "open" or decoded["env_class"] != "open":
            raise ValueError(f"TOKEN_AUTHORITY_OPEN_SEMANTIC_MISMATCH:{decoded}")
        return token

    def endpoint_receipt(self) -> dict[str, Any]:
        low = np.full(self.action_dim, -1.0, dtype=np.float64)
        high = np.full(self.action_dim, 1.0, dtype=np.float64)
        low_token = int(self.encode_normalized(low)[-1])
        high_token = int(self.encode_normalized(high)[-1])
        low_decoded = float(self.decode_normalized(np.r_[np.zeros(self.action_dim - 1), low_token])[-1])
        high_decoded = float(self.decode_normalized(np.r_[np.zeros(self.action_dim - 1), high_token])[-1])
        return {
            "native_endpoint_non_bijective": True,
            "normalized_endpoints": [-1.0, 1.0],
            "endpoint_tokens": [low_token, high_token],
            "decoded_normalized_centers": [low_decoded, high_decoded],
            "roundtrip_endpoint_equality_required": False,
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "STAGE_X_X1R_T1_NATIVE_ACTION_TOKEN_AUTHORITY_RECEIPT_V2",
            "authority_version": self.version,
            "suite": self.binding.suite,
            "checkpoint_path": self.binding.checkpoint_path,
            "checkpoint_config_sha256": self.binding.checkpoint_config_sha256,
            "tokenizer_source": self.binding.tokenizer_source,
            "tokenizer_source_sha256": self.binding.tokenizer_source_sha256,
            "model_decoder_source_sha256": self.binding.model_decoder_source_sha256,
            "tokenizer_files": dict(self.binding.tokenizer_files),
            "tokenizer_vocab_size": self.vocab_size,
            "n_action_bins": self.binding.n_action_bins,
            "algorithm": NATIVE_ACTION_TOKEN_ALGORITHM,
            "legacy_helper_status": LEGACY_HELPER_STATUS,
            "open_token_id": self.open_token_id(),
            "endpoint": self.endpoint_receipt(),
        }
