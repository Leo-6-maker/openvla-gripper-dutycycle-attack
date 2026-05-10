from __future__ import annotations
from typing import Any, Optional
import numpy as np


def _as_prefix_logits(prefix_logits: Any) -> np.ndarray:
    x = np.asarray(prefix_logits, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] < 2:
        raise ValueError(f"prefix_logits must have shape [K,V] with V>=2; got {x.shape}")
    return x


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def prefix_entropy(prefix_logits: np.ndarray) -> float:
    x = _as_prefix_logits(prefix_logits)
    p = np.clip(softmax_np(x, axis=-1), 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=-1)
    return float(np.mean(h))


def prefix_position_entropy(prefix_logits: np.ndarray) -> np.ndarray:
    x = _as_prefix_logits(prefix_logits)
    p = np.clip(softmax_np(x, axis=-1), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def prefix_entropy_positions(prefix_logits: np.ndarray, positions: list[int] | tuple[int, ...]) -> float:
    h = prefix_position_entropy(prefix_logits)
    idx = np.asarray([int(i) for i in positions if 0 <= int(i) < h.shape[0]], dtype=np.int64)
    if idx.size == 0:
        return prefix_entropy(prefix_logits)
    return float(np.mean(h[idx]))


def xyz_entropy(prefix_logits: np.ndarray) -> float:
    return prefix_entropy_positions(prefix_logits, [0, 1, 2])


def arm_entropy(prefix_logits: np.ndarray) -> float:
    return prefix_entropy_positions(prefix_logits, [0, 1, 2, 3, 4, 5])


def gripper_entropy(prefix_logits: np.ndarray) -> float:
    x = _as_prefix_logits(prefix_logits)
    return prefix_entropy_positions(x, [x.shape[0] - 1])


def _motion_norm(clean_action, dims: list[int] | tuple[int, ...]) -> float:
    if clean_action is None:
        return 0.0
    a = np.asarray(clean_action, dtype=np.float32).reshape(-1)
    idx = np.asarray([int(i) for i in dims if 0 <= int(i) < a.shape[0]], dtype=np.int64)
    if idx.size == 0:
        return 0.0
    return float(np.linalg.norm(a[idx]))


def motion_weighted_xyz_entropy(prefix_logits: np.ndarray, clean_action) -> float:
    return float(xyz_entropy(prefix_logits) * _motion_norm(clean_action, [0, 1, 2]))


def motion_weighted_arm_entropy(prefix_logits: np.ndarray, clean_action) -> float:
    return float(arm_entropy(prefix_logits) * _motion_norm(clean_action, [0, 1, 2, 3, 4, 5]))


def prefix_top2_margin(prefix_logits: np.ndarray) -> float:
    x = _as_prefix_logits(prefix_logits)
    top2 = np.partition(x, kth=-2, axis=-1)[:, -2:]
    return float(np.mean(top2[:, 1] - top2[:, 0]))


def extract_prefix_logits(model_output: Any, k: int) -> Optional[np.ndarray]:
    # Extract generated-token logits from HF generate output or forward output.
    if model_output is None:
        return None
    scores = getattr(model_output, "scores", None)
    if scores is not None:
        rows = []
        for s in list(scores)[: int(k)]:
            if hasattr(s, "detach"):
                s = s.detach().float().cpu().numpy()
            s = np.asarray(s)
            if s.ndim == 2:
                s = s[0]
            rows.append(s.astype(np.float32))
        return np.stack(rows, axis=0) if rows else None
    logits = getattr(model_output, "logits", None)
    if logits is not None:
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        arr = np.asarray(logits)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[0] >= int(k):
            return arr[-int(k):].astype(np.float32)
    return None
