from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import yaml


@dataclass
class DirectionSpec:
    direction_id: str
    vector_full: np.ndarray
    dims: List[int]
    g_hat: np.ndarray


def normalize_direction(vector: np.ndarray, dims: list[int]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    idx = np.asarray(dims, dtype=np.int64)
    sub = v[idx]
    n = float(np.linalg.norm(sub))
    if n < 1e-12:
        raise ValueError("cannot normalize zero direction on selected dims")
    out = np.zeros_like(v, dtype=np.float32)
    out[idx] = sub / n
    return out


def load_direction_spec(path: str, direction_id: str) -> DirectionSpec:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["directions"][direction_id]
    vec = np.asarray(d["vector_full_7d"], dtype=np.float32)
    dims = [int(x) for x in d["dims"]]
    g = normalize_direction(vec, dims) if d.get("normalize_on_dims", True) else vec
    return DirectionSpec(direction_id=direction_id, vector_full=vec, dims=dims, g_hat=g)


def compute_delta_action(a_exec: np.ndarray, a_clean: np.ndarray) -> np.ndarray:
    return np.asarray(a_exec, dtype=np.float32) - np.asarray(a_clean, dtype=np.float32)


def compute_alignment(delta: np.ndarray, g_hat: np.ndarray, dims: list[int]) -> dict:
    d = np.asarray(delta, dtype=np.float32)
    g = np.asarray(g_hat, dtype=np.float32)
    idx = np.asarray(dims, dtype=np.int64)
    alignment = float(np.dot(d[idx], g[idx]))
    cos = float(alignment / (np.linalg.norm(d[idx]) * np.linalg.norm(g[idx]) + 1e-8))
    return {
        "alignment": alignment,
        "alignment_cos": cos,
        "delta_l2": float(np.linalg.norm(d)),
        "delta_linf": float(np.max(np.abs(d))) if d.size else 0.0,
    }


def build_target_action(a_clean: np.ndarray, g_hat: np.ndarray, dims: list[int], alpha: float) -> np.ndarray:
    a = np.asarray(a_clean, dtype=np.float32).copy()
    idx = np.asarray(dims, dtype=np.int64)
    a[idx] = a[idx] + float(alpha) * np.asarray(g_hat, dtype=np.float32)[idx]
    return a
