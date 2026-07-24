#!/usr/bin/env python3
"""C2e3 GRU detector deployment runtime with normalization contract.

Binds checkpoint + config + normalization stats + context lookup into a single
deployable bundle. Guarantees training/deployment equivalence.

Standard path::

    from gripper_attack.c2e3_gru_detector_runtime import C2e3GRUDetectorRuntime
    det = C2e3GRUDetectorRuntime("d4c2e3_25d_baseline_package/")
    emit_p, supp_p, emitted = det.predict(window_25d, suite, task_index)

CPU-only. No env.step, no OpenVLA, no MuJoCo.
"""

from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# ── canonical 25D SC5_V2_FEATURES order (must match C2e1 training) ──
CANONICAL_25D_FEATURES: List[str] = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close",
    "eef_speed", "eef_z_delta_since_close",
    "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5",
    "eef_speed_variance_5",
]

# ── canonical 108D context feature order (must match C2e1 training) ──
CANONICAL_108D_CONTEXT_FEATURES: List[str] = [
    "ctx_suite_libero_10", "ctx_suite_libero_goal",
    "ctx_suite_libero_object", "ctx_suite_libero_spatial",
] + [
    f"ctx_suite_task_hash_{i:02d}" for i in range(32)
] + [
    f"ctx_task_index_hash_{i:02d}" for i in range(32)
] + [
    f"ctx_task_index_onehot_libero_10_{i:02d}" for i in range(10)
] + [
    f"ctx_task_index_onehot_libero_goal_{i:02d}" for i in range(10)
] + [
    f"ctx_task_index_onehot_libero_object_{i:02d}" for i in range(10)
] + [
    f"ctx_task_index_onehot_libero_spatial_{i:02d}" for i in range(10)
]


class GRUModel(nn.Module):
    """GRU detector matching C2e2K training architecture."""
    def __init__(self, nf: int = 25, nc: int = 108, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(nf, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden + nc, 2)

    def forward(self, xt: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(xt)
        return self.head(torch.cat([h[-1], xc], dim=1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float_list(obj: Any, expected_len: int, label: str) -> List[float]:
    vals = [float(v) for v in obj]
    if len(vals) != expected_len:
        raise ValueError(f"{label}: expected {expected_len} values, got {len(vals)}")
    return vals


class C2e3GRUDetectorRuntime:
    """Unified C2e3 GRU detector with built-in normalization.

    Usage::

        det = C2e3GRUDetectorRuntime("/path/to/d4c2e3_25d_baseline_package/")
        emit_p, supp_p, emitted = det.predict(window_raw_25d, suite="libero_object", task_index=3)

    ``window_raw_25d`` must be a float32 array of shape ``(W, 25)`` where W=window
    (default 16) and the 25 features follow ``CANONICAL_25D_FEATURES`` order.
    """

    def __init__(
        self,
        package_dir: str,
        *,
        checkpoint_filename: str = "c2e3_selected_baseline_model.pt",
        config_filename: str = "c2e3_selected_baseline_config.json",
        norm_stats_filename: str = "c2e3_normalization_stats_train_only.json",
        context_lookup_filename: str = "c2e3_context_lookup.json",
    ):
        pkg = Path(package_dir)
        self._pkg = pkg
        self._checkpoint_path = pkg / checkpoint_filename
        self._config_path = pkg / config_filename
        self._norm_path = pkg / norm_stats_filename
        self._ctx_lookup_path = pkg / context_lookup_filename

        # ── load checkpoint ──
        ckpt = torch.load(str(self._checkpoint_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if state is None:
            raise ValueError("checkpoint missing model_state_dict and state_dict")
        cfg = ckpt.get("config", {})
        cfg_th = ckpt.get("threshold", {})

        self._tau_emit = float(cfg_th.get("tau_emit", 0.33))
        self._tau_suppress = float(cfg_th.get("tau_suppress", 0.67))
        self._window = int(cfg.get("window", 16))
        self._hidden = int(cfg.get("channels", cfg.get("hidden", 128)))
        self._n_features = int(cfg.get("n_features", 25))
        self._n_context = int(cfg.get("n_context", 108))

        if self._n_features != 25:
            raise ValueError(f"Expected 25 temporal features, got {self._n_features}")
        if self._n_context != 108:
            raise ValueError(f"Expected 108 context features, got {self._n_context}")

        self._model = GRUModel(nf=self._n_features, nc=self._n_context, hidden=self._hidden)
        self._model.load_state_dict(state)
        self._model.cpu().eval()

        # ── load config ──
        self._config: Dict[str, Any] = {}
        if self._config_path.exists():
            self._config = json.loads(self._config_path.read_text())

        # ── load normalization stats ──
        norm = json.loads(self._norm_path.read_text())
        self._temporal_mean = np.array(
            _safe_float_list(norm["temporal_feature_mean"], self._n_features, "temporal_feature_mean"),
            dtype=np.float32,
        )
        self._temporal_std = np.array(
            _safe_float_list(norm["temporal_feature_std"], self._n_features, "temporal_feature_std"),
            dtype=np.float32,
        )
        self._context_mean = np.array(
            _safe_float_list(norm["context_feature_mean"], self._n_context, "context_feature_mean"),
            dtype=np.float32,
        )
        self._context_std = np.array(
            _safe_float_list(norm["context_feature_std"], self._n_context, "context_feature_std"),
            dtype=np.float32,
        )

        # ── load context lookup ──
        ctx_lookup_raw = json.loads(self._ctx_lookup_path.read_text())
        self._context_lookup: Dict[Tuple[str, int], np.ndarray] = {}
        for key, vec in ctx_lookup_raw["lookup"].items():
            suite, task_str = key.split("|")
            task_idx = int(task_str.replace("task_", ""))
            self._context_lookup[(suite, task_idx)] = np.array(vec, dtype=np.float32)

        # ── provenance ──
        self._checkpoint_sha = sha256_file(self._checkpoint_path)
        self._norm_sha = sha256_file(self._norm_path)
        self._config_sha = sha256_file(self._config_path) if self._config_path.exists() else ""
        self._ctx_lookup_sha = sha256_file(self._ctx_lookup_path) if self._ctx_lookup_path.exists() else ""

    # ── public API ──

    def predict(
        self,
        window_raw: np.ndarray,
        suite: str,
        task_index: int,
    ) -> Tuple[float, float, bool]:
        """Run GRU on a normalized temporal window and context.

        Args:
            window_raw: shape ``(W, 25)`` float32, raw SC5 streaming features.
            suite: one of ``libero_10``, ``libero_goal``, ``libero_object``, ``libero_spatial``.
            task_index: 0–9 LIBERO task index.

        Returns:
            ``(emit_p, suppress_p, emitted)`` where ``emitted`` is True when
            ``emit_p >= tau_emit AND suppress_p <= tau_suppress``.
        """
        if window_raw.shape[1] != self._n_features:
            raise ValueError(
                f"window_raw has {window_raw.shape[1]} features, expected {self._n_features}"
            )

        # normalize temporal window
        tm = self._temporal_mean.reshape(1, 1, -1)
        ts = np.maximum(self._temporal_std.reshape(1, 1, -1), 1e-8)
        window_norm = (window_raw.astype(np.float32) - tm) / ts

        # lookup + normalize context
        ctx_key = (suite, task_index)
        if ctx_key not in self._context_lookup:
            raise KeyError(f"no context for ({suite}, task_{task_index:02d})")
        ctx_raw = self._context_lookup[ctx_key].reshape(1, -1)
        cm = self._context_mean.reshape(1, -1)
        cs = np.maximum(self._context_std.reshape(1, -1), 1e-8)
        ctx_norm = (ctx_raw - cm) / cs

        # forward
        with torch.no_grad():
            logits = self._model(
                torch.from_numpy(window_norm),
                torch.from_numpy(ctx_norm),
            ).numpy()[0]

        emit_p = float(1.0 / (1.0 + np.exp(-np.clip(float(logits[0]), -50, 50))))
        supp_p = float(1.0 / (1.0 + np.exp(-np.clip(float(logits[1]), -50, 50))))
        emitted = bool(emit_p >= self._tau_emit and supp_p <= self._tau_suppress)

        return emit_p, supp_p, emitted

    # ── properties ──

    @property
    def tau_emit(self) -> float:
        return self._tau_emit

    @property
    def tau_suppress(self) -> float:
        return self._tau_suppress

    @property
    def window(self) -> int:
        return self._window

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_context(self) -> int:
        return self._n_context

    @property
    def hidden(self) -> int:
        return self._hidden

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha

    @property
    def normalization_sha256(self) -> str:
        return self._norm_sha

    @property
    def config_sha256(self) -> str:
        return self._config_sha

    @property
    def context_lookup_sha256(self) -> str:
        return self._ctx_lookup_sha

    @property
    def provenance(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": str(self._checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha,
            "config_sha256": self._config_sha,
            "normalization_sha256": self._norm_sha,
            "context_lookup_sha256": self._ctx_lookup_sha,
            "tau_emit": self._tau_emit,
            "tau_suppress": self._tau_suppress,
            "window": self._window,
            "n_features": self._n_features,
            "n_context": self._n_context,
            "hidden": self._hidden,
            "normalization_applied": True,
            "context_policy": "lookup_from_c2e1_dataset",
            "feature_order": CANONICAL_25D_FEATURES,
        }
