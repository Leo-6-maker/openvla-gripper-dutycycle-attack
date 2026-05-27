"""Proprio-only causal student baseline utilities for Milestone 2C."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PHASE_CLASSES = [
    "approach",
    "grasp_close",
    "lift",
    "carry",
    "pre_release_hazard",
    "release_safe",
    "other",
]

NUMERIC_FEATURES = [
    "gripper_command",
    "gripper_qpos",
    "gripper_width",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_vx",
    "eef_vy",
    "eef_vz",
    "action_dx",
    "action_dy",
    "action_dz",
    "action_gripper",
    "recent_close_streak",
    "recent_open_streak",
    "recent_gripper_flip_count",
    "normalized_step",
]

CATEGORICAL_FEATURES = ["mechanism_type", "parse_confidence"]

EVAL_ONLY_COLUMNS = [
    "teacher_window_start",
    "teacher_window_end",
    "teacher_anchor_step",
    "teacher_detector_mode",
]

IDENTITY_COLUMNS = ["run_id", "suite", "task_id", "task_name", "state_id", "seed", "episode_key", "step_idx"]

FORBIDDEN_INPUT_SUBSTRINGS = [
    "object_pose",
    "target_pose",
    "object_to_target_distance",
    "target_distance",
    "done_future",
    "success_done",
    "attack_outcome",
    "oracle_outcome",
    "random_outcome",
    "manual_outcome",
    "teacher_window_start",
    "teacher_window_end",
    "teacher_anchor_step",
    "task_id",
    "state_id",
    "run_id",
    "episode_key",
    "image_path",
    "visual_feature",
]


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def assert_feature_whitelist(input_features: Iterable[str]) -> None:
    features = list(input_features)
    allowed = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    unknown = [f for f in features if f not in allowed]
    if unknown:
        raise ValueError(f"Unknown model input features: {unknown}")
    bad = [f for f in features for token in FORBIDDEN_INPUT_SUBSTRINGS if token in f.lower()]
    if bad:
        raise ValueError(f"Forbidden model input features: {sorted(set(bad))}")


def phase_index(phase: str) -> int:
    return PHASE_CLASSES.index(phase) if phase in PHASE_CLASSES else PHASE_CLASSES.index("other")


def split_keys(keys: list[str], seed: int, train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, str]:
    unique = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n and n >= 3:
        n_train = n - 2
        n_val = 1
    mapping: dict[str, str] = {}
    for i, key in enumerate(unique):
        mapping[key] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return mapping


def assign_splits(rows: list[dict[str, str]], split_mode: str, seed: int) -> dict[str, str]:
    if split_mode not in {"task_id", "episode_key"}:
        raise ValueError("split_mode must be task_id or episode_key")
    key_field = split_mode
    keys_by_suite: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        keys_by_suite[row.get("suite", "unknown")].append(row[key_field])
    mapping: dict[str, str] = {}
    for suite, keys in sorted(keys_by_suite.items()):
        suite_seed = seed + sum(ord(c) for c in suite)
        mapping.update(split_keys(keys, suite_seed))
    return {r["episode_key"]: mapping[r[key_field]] for r in rows}


def split_summary(rows: list[dict[str, str]], split_mode: str, seed: int) -> list[dict[str, Any]]:
    split_by_episode = assign_splits(rows, split_mode, seed)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(split_by_episode[row["episode_key"]], row["episode_key"])].append(row)
    out = []
    for (split_name, episode_key), group in sorted(groups.items()):
        first = group[0]
        out.append(
            {
                "split_name": split_name,
                "split_mode": split_mode,
                "suite": first["suite"],
                "task_id": first["task_id"],
                "episode_key": episode_key,
                "n_rows": len(group),
                "n_hazard_positive": sum(boolish(r["teacher_hazard"]) for r in group),
                "n_phase_labels": len({r["teacher_phase"] for r in group}),
            }
        )
    return out


@dataclass
class EncodedDataset:
    rows: list[dict[str, str]]
    x: torch.Tensor
    phase: torch.Tensor
    hazard: torch.Tensor
    release: torch.Tensor
    confidence: torch.Tensor
    feature_names: list[str]
    split_by_episode: dict[str, str]
    category_maps: dict[str, list[str]]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]


def _stats(values: list[float | None]) -> tuple[float, float]:
    present = [v for v in values if v is not None]
    if not present:
        return 0.0, 1.0
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / max(1, len(present) - 1)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return mean, std


def encode_dataset(rows: list[dict[str, str]], split_mode: str, seed: int) -> EncodedDataset:
    assert_feature_whitelist(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    split_by_episode = assign_splits(rows, split_mode, seed)
    train_rows = [r for r in rows if split_by_episode[r["episode_key"]] == "train"]
    numeric_mean: dict[str, float] = {}
    numeric_std: dict[str, float] = {}
    for f in NUMERIC_FEATURES:
        mean, std = _stats([as_float(r.get(f)) for r in train_rows])
        numeric_mean[f] = mean
        numeric_std[f] = std
    category_maps = {
        f: sorted({r.get(f, "missing") or "missing" for r in rows})
        for f in CATEGORICAL_FEATURES
    }
    feature_names = list(NUMERIC_FEATURES)
    for f, vals in category_maps.items():
        feature_names.extend([f"{f}={v}" for v in vals])

    encoded: list[list[float]] = []
    for r in rows:
        row_vec: list[float] = []
        for f in NUMERIC_FEATURES:
            val = as_float(r.get(f))
            val = numeric_mean[f] if val is None else val
            row_vec.append((val - numeric_mean[f]) / numeric_std[f])
        for f in CATEGORICAL_FEATURES:
            val = r.get(f, "missing") or "missing"
            row_vec.extend([1.0 if val == candidate else 0.0 for candidate in category_maps[f]])
        encoded.append(row_vec)
    x = torch.tensor(encoded, dtype=torch.float32)
    phase = torch.tensor([phase_index(r.get("teacher_phase", "other")) for r in rows], dtype=torch.long)
    hazard = torch.tensor([1.0 if boolish(r.get("teacher_hazard")) else 0.0 for r in rows], dtype=torch.float32)
    release = torch.tensor([1.0 if boolish(r.get("teacher_release_safe")) else 0.0 for r in rows], dtype=torch.float32)
    confidence = torch.tensor([1.0 if r.get("teacher_confidence") in {"medium", "high"} else 0.0 for r in rows], dtype=torch.float32)
    return EncodedDataset(rows, x, phase, hazard, release, confidence, feature_names, split_by_episode, category_maps, numeric_mean, numeric_std)


class ProprioCausalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, phase_classes: int = len(PHASE_CLASSES)) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, phase_classes)
        self.hazard_head = nn.Linear(hidden_dim, 1)
        self.release_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "phase_logits": self.phase_head(h),
            "hazard_logit": self.hazard_head(h).squeeze(-1),
            "release_safe_logit": self.release_head(h).squeeze(-1),
            "confidence_logit": self.confidence_head(h).squeeze(-1),
        }


def make_loader(data: EncodedDataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        data.x[indices],
        data.phase[indices],
        data.hazard[indices],
        data.release[indices],
        data.confidence[indices],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def indices_for_split(data: EncodedDataset, split_name: str) -> list[int]:
    return [i for i, r in enumerate(data.rows) if data.split_by_episode[r["episode_key"]] == split_name]


def compute_loss(
    outputs: dict[str, torch.Tensor],
    phase: torch.Tensor,
    hazard: torch.Tensor,
    release: torch.Tensor,
    confidence: torch.Tensor,
    lambda_hazard: float = 1.0,
    lambda_release: float = 0.5,
    lambda_conf: float = 0.2,
) -> torch.Tensor:
    ce = nn.functional.cross_entropy(outputs["phase_logits"], phase)
    hazard_bce = nn.functional.binary_cross_entropy_with_logits(outputs["hazard_logit"], hazard)
    release_bce = nn.functional.binary_cross_entropy_with_logits(outputs["release_safe_logit"], release)
    conf_bce = nn.functional.binary_cross_entropy_with_logits(outputs["confidence_logit"], confidence)
    return ce + lambda_hazard * hazard_bce + lambda_release * release_bce + lambda_conf * conf_bce


def binary_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": sum(y_true) / len(y_true) if y_true else 0.0,
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
    }


def auroc(y_true: list[int], y_score: list[float]) -> float:
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum = sum(rank for rank, (_, y) in enumerate(pairs, start=1) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def auprc(y_true: list[int], y_score: list[float]) -> float:
    if not y_true or sum(y_true) == 0:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    total_pos = sum(y_true)
    for _, y in pairs:
        if y:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def phase_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    rows = []
    f1s = []
    for idx, label in enumerate(PHASE_CLASSES):
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == idx and p == idx)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != idx and p == idx)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == idx and p != idx)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        rows.append({"phase": label, "precision": precision, "recall": recall, "f1": f1, "support": sum(1 for y in y_true if y == idx)})
    accuracy = sum(1 for y, p in zip(y_true, y_pred) if y == p) / len(y_true) if y_true else 0.0
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s), "per_class": rows}


def majority_baseline(rows: list[dict[str, str]], indices: list[int]) -> dict[str, str]:
    phase = Counter(rows[i].get("teacher_phase", "other") for i in indices).most_common(1)[0][0]
    hazard = Counter(boolish(rows[i].get("teacher_hazard")) for i in indices).most_common(1)[0][0]
    return {"phase": phase, "hazard": str(hazard)}
