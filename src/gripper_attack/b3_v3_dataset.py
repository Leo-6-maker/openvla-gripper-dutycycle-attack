"""Strict Official V3 S1 dataset and episode-balanced batching.

The loader consumes only a sealed S1 root and a formal registry.  It never
selects on task success, never opens source CLEAN evidence, and never treats
unknown Teacher labels as negatives.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .b3_formal import (
    B3_FEATURES_25D,
    B3_POLICY_INTENT_FEATURES_9D,
    B3_HEADS,
    B3Normalization,
    json_sha,
)
from .b3_official_v3_s1 import STUDENT_FIELDS, TEACHER_FIELDS
from .b3_teacher_training_adapter import adapt_teacher_batch
from .official_v3_contract import FORMAL_PROVENANCE_CLASSES, SUITES, canonical_key, expected_split


FIT_SPLIT = "FIT_TRAIN"
FORMAL_PROVENANCE = FORMAL_PROVENANCE_CLASSES
STUDENT_SCHEMA = "B3_OFFICIAL_V3_STUDENT_INPUT_V1"
TEACHER_SCHEMA = "B3_OFFICIAL_V3_TEACHER_RECORD_V1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_no}")
        rows.append(value)
    return rows


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _row_identity(row: dict[str, Any], registry: dict[str, Any]) -> None:
    expected = {
        "suite": registry["suite"],
        "task_idx": int(registry["task_idx"]),
        "state_id": int(registry["state_id"]),
        "canonical_parent_key": registry["canonical_parent_key"],
    }
    actual = {name: row.get(name) for name in expected}
    try:
        actual["task_idx"] = int(actual["task_idx"])
        actual["state_id"] = int(actual["state_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError("episode identity columns are not integers") from exc
    if actual != expected:
        raise ValueError(f"episode identity mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True)
class B3Episode:
    canonical_parent_key: str
    suite: str
    task_idx: int
    state_id: int
    split: str
    task_success: bool
    features_25d: Tensor
    targets: dict[str, Tensor]
    known_masks: dict[str, Tensor]
    valid_mask: Tensor
    features_9d: Tensor | None = None
    event_ids: tuple[int, ...] = ()
    source_artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.features_25d.ndim != 2 or self.features_25d.shape[1] != 25:
            raise ValueError("features_25d must have shape [T, 25]")
        if self.features_9d is not None and (self.features_9d.ndim != 2 or self.features_9d.shape != (len(self.features_25d), 9)):
            raise ValueError("features_9d must have shape [T, 9]")
        if self.valid_mask.shape != (len(self.features_25d),) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [T]")
        for head in B3_HEADS:
            if head not in self.targets or head not in self.known_masks:
                raise ValueError(f"missing head in episode: {head}")
            if self.targets[head].shape != self.valid_mask.shape or self.known_masks[head].shape != self.valid_mask.shape:
                raise ValueError(f"head shape mismatch: {head}")
            if self.known_masks[head].dtype != torch.bool:
                raise ValueError(f"known mask must be bool: {head}")


@dataclass(frozen=True)
class B3Batch:
    x25: Tensor
    x9: Tensor | None
    targets: dict[str, Tensor]
    known_masks: dict[str, Tensor]
    episode_valid_mask: Tensor
    padding_mask: Tensor
    episodes: tuple[B3Episode, ...]


def _validate_registry_rows(rows: Sequence[dict[str, Any]], *, require_a_only: bool = True) -> list[dict[str, Any]]:
    if len(rows) != 2000:
        raise ValueError(f"formal registry must contain 2000 rows, got {len(rows)}")
    seen: set[str] = set()
    fit: list[dict[str, Any]] = []
    for row in rows:
        required = {"canonical_parent_key", "suite", "task_idx", "state_id", "split", "formal_selected", "formal_eligible", "provenance_class", "selected_artifact_root"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"registry row missing fields: {missing}")
        suite, task, state = row["suite"], int(row["task_idx"]), int(row["state_id"])
        key = row["canonical_parent_key"]
        if suite not in SUITES or not 0 <= task < 10 or not 0 <= state < 50 or key != canonical_key(suite, task, state):
            raise ValueError(f"registry identity mismatch: {key}")
        if key in seen:
            raise ValueError(f"duplicate registry identity: {key}")
        seen.add(key)
        if row["split"] != expected_split(state):
            raise ValueError(f"registry split mismatch: {key}")
        if row["split"] == FIT_SPLIT:
            if row["formal_selected"] not in (True, "True", "true", 1, "1") or row["formal_eligible"] not in (True, "True", "true", 1, "1"):
                raise ValueError(f"FIT row is not formally selected: {key}")
            if require_a_only and row["provenance_class"] not in FORMAL_PROVENANCE:
                raise ValueError(f"FIT row is not current-head provenance: {key}")
            fit.append(dict(row, task_idx=task, state_id=state))
    if len(fit) != 800:
        raise ValueError(f"formal FIT registry must contain 800 FIT rows, got {len(fit)}")
    suite_counts = {suite: sum(row["suite"] == suite for row in fit) for suite in SUITES}
    if suite_counts != {suite: 200 for suite in SUITES}:
        raise ValueError(f"FIT suite quota mismatch: {suite_counts}")
    task_counts = {(suite, task): 0 for suite in SUITES for task in range(10)}
    for row in fit:
        task_counts[(row["suite"], row["task_idx"])] += 1
    if any(value != 20 for value in task_counts.values()):
        raise ValueError("FIT task quota mismatch")
    return sorted(fit, key=lambda row: row["canonical_parent_key"])


def load_formal_registry_csv(path: Path, *, require_a_only: bool = True) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return _validate_registry_rows(rows, require_a_only=require_a_only)


def _load_9d(root: Path, registry: dict[str, Any], expected_steps: int) -> Tensor:
    rows = _jsonl(root / "policy_intent_9d_records.jsonl")
    if len(rows) != expected_steps:
        raise ValueError("9D episode length mismatch")
    values: list[list[float]] = []
    for index, row in enumerate(rows):
        _row_identity(row, registry)
        if int(row.get("step", -1)) != index or row.get("schema") != "B3_OFFICIAL_V3_POLICY_INTENT_9D_V1":
            raise ValueError(f"9D schema/step mismatch at {index}")
        vector = row.get("clean_policy_intent_9d")
        if not isinstance(vector, list) or len(vector) != 9:
            raise ValueError(f"9D vector mismatch at {index}")
        values.append([float(item) for item in vector])
    tensor = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("9D vector contains non-finite values")
    return tensor


def load_episode(root: Path, registry: dict[str, Any], *, include_9d_root: Path | None = None) -> B3Episode:
    """Load one already-audited S1 episode without reading CLEAN source files."""

    manifest = json.loads((root / "materialization_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "B3_OFFICIAL_V3_S1_EPISODE_V1" or manifest.get("teacher_materialization") != "COMPLETED":
        raise ValueError("unexpected S1 episode manifest")
    _row_identity(manifest["source_identity"], registry)
    if manifest.get("source_identity", {}).get("canonical_parent_key") != registry["canonical_parent_key"]:
        raise ValueError("S1 manifest identity mismatch")
    students = _jsonl(root / "student_input_records.jsonl")
    teachers = _jsonl(root / "teacher_retention_records.jsonl")
    events = json.loads((root / "retention_events.json").read_text(encoding="utf-8"))
    if not isinstance(events, list) or len(students) == 0 or len(students) != len(teachers):
        raise ValueError("S1 episode streams are empty or misaligned")
    features: list[list[float]] = []
    valid_values: list[bool] = []
    for index, row in enumerate(students):
        if set(row) != set(STUDENT_FIELDS):
            raise ValueError(f"Student field whitelist mismatch at {index}")
        _row_identity(row, registry)
        if row.get("schema") != STUDENT_SCHEMA or row.get("source_schema") != "OFFICIAL_25D_V1" or row.get("feature_order_sha256") != json_sha(list(B3_FEATURES_25D)):
            raise ValueError(f"Student schema/order mismatch at {index}")
        if int(row.get("step", -1)) != index or not isinstance(row.get("features_25d"), list) or len(row["features_25d"]) != 25:
            raise ValueError(f"Student vector/step mismatch at {index}")
        vector = [float(item) for item in row["features_25d"]]
        if not bool(torch.isfinite(torch.tensor(vector)).all()):
            raise ValueError(f"Student vector is non-finite at {index}")
        features.append(vector)
        valid_values.append(_as_bool(row["valid"], f"student.valid[{index}]"))
    required_teacher = set(TEACHER_FIELDS)
    for index, row in enumerate(teachers):
        if set(row) != required_teacher:
            raise ValueError(f"Teacher field whitelist mismatch at {index}")
        _row_identity(row, registry)
        if row.get("schema") != TEACHER_SCHEMA or int(row.get("step", -1)) != index:
            raise ValueError(f"Teacher schema/step mismatch at {index}")
        if row.get("source_artifact_sha256") != manifest.get("source_artifact_sha256"):
            raise ValueError(f"Teacher source provenance mismatch at {index}")
    targets, masks = adapt_teacher_batch([teachers], padding_mask=torch.tensor([valid_values], dtype=torch.bool))
    if "task_success" not in registry and "success" not in registry:
        raise ValueError("formal registry row is missing task_success")
    task_success = registry.get("task_success", registry.get("success"))
    if isinstance(task_success, str):
        task_success = task_success.lower() == "true"
    task_success = _as_bool(task_success, "task_success")
    features_25d = torch.tensor(features, dtype=torch.float32)
    valid_mask = torch.tensor(valid_values, dtype=torch.bool)
    features_9d = _load_9d(include_9d_root, registry, len(features)) if include_9d_root is not None else None
    event_ids = tuple(int(row.get("event_id", -1)) for row in teachers)
    return B3Episode(
        canonical_parent_key=registry["canonical_parent_key"], suite=registry["suite"], task_idx=int(registry["task_idx"]),
        state_id=int(registry["state_id"]), split=registry["split"], task_success=task_success,
        features_25d=features_25d, targets={key: value.squeeze(0) for key, value in targets.items()},
        known_masks={key: value.squeeze(0) for key, value in masks.items()}, valid_mask=valid_mask,
        features_9d=features_9d, event_ids=event_ids, source_artifact_sha256=str(manifest.get("source_artifact_sha256", "")),
    )


def pad_episode_batch(episodes: Sequence[B3Episode]) -> B3Batch:
    if not episodes:
        raise ValueError("at least one episode is required")
    max_steps = max(len(item.features_25d) for item in episodes)
    batch = len(episodes)
    x25 = torch.zeros(batch, max_steps, 25, dtype=torch.float32)
    any_9d = any(item.features_9d is not None for item in episodes)
    if any_9d and any(item.features_9d is None for item in episodes):
        raise ValueError("a batch cannot mix 25D and 25D9D episodes")
    x9 = torch.zeros(batch, max_steps, 9, dtype=torch.float32) if any_9d else None
    episode_valid = torch.zeros(batch, max_steps, dtype=torch.bool)
    padding = torch.zeros(batch, max_steps, dtype=torch.bool)
    targets = {head: torch.zeros(batch, max_steps, dtype=torch.float32) for head in B3_HEADS}
    masks = {head: torch.zeros(batch, max_steps, dtype=torch.bool) for head in B3_HEADS}
    for batch_index, episode in enumerate(episodes):
        steps = len(episode.features_25d)
        x25[batch_index, :steps] = episode.features_25d
        if x9 is not None and episode.features_9d is not None:
            x9[batch_index, :steps] = episode.features_9d
        episode_valid[batch_index, :steps] = episode.valid_mask
        padding[batch_index, :steps] = True
        for head in B3_HEADS:
            targets[head][batch_index, :steps] = episode.targets[head]
            masks[head][batch_index, :steps] = episode.known_masks[head]
    return B3Batch(x25, x9, targets, masks, episode_valid, padding, tuple(episodes))


class B3EpisodeSampler:
    """Deterministic suite -> task -> episode sampler; never samples steps."""

    def __init__(self, episodes: Sequence[B3Episode], *, seed: int = 0) -> None:
        if not episodes:
            raise ValueError("sampler requires episodes")
        self.episodes = tuple(episodes)
        self.seed = int(seed)
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            groups[(episode.suite, episode.task_idx)].append(index)
        self.groups = {key: sorted(value, key=lambda index: self.episodes[index].canonical_parent_key) for key, value in groups.items()}

    def ordered_indices(self, *, shuffle: bool = True) -> list[int]:
        rng = random.Random(self.seed)
        queues = {key: list(value) for key, value in sorted(self.groups.items())}
        if shuffle:
            for values in queues.values():
                rng.shuffle(values)
        order: list[int] = []
        while any(queues.values()):
            for key in sorted(queues):
                if queues[key]:
                    order.append(queues[key].pop(0))
        return order


def compute_fit_normalization(episodes: Iterable[B3Episode], *, include_9d: bool = False) -> B3Normalization:
    episodes = tuple(episodes)
    if not episodes or any(item.split != FIT_SPLIT or not 0 <= item.state_id < 20 for item in episodes):
        raise ValueError("normalization must be computed from FIT_TRAIN states 0-19 only")
    x25 = torch.cat([item.features_25d[item.valid_mask] for item in episodes], dim=0)
    if len(x25) == 0:
        raise ValueError("FIT normalization has no valid steps")
    mean25 = x25.mean(dim=0)
    std25 = x25.std(dim=0, unbiased=False).clamp_min(1e-6)
    if not include_9d:
        return B3Normalization(tuple(mean25.tolist()), tuple(std25.tolist()))
    if any(item.features_9d is None for item in episodes):
        raise ValueError("9D normalization requires 9D data for every episode")
    x9 = torch.cat([item.features_9d[item.valid_mask] for item in episodes if item.features_9d is not None], dim=0)
    mean9 = x9.mean(dim=0)
    std9 = x9.std(dim=0, unbiased=False).clamp_min(1e-6)
    return B3Normalization(tuple(mean25.tolist()), tuple(std25.tolist()), tuple(mean9.tolist()), tuple(std9.tolist()))


def select_fit_fold_episodes(
    episodes: Sequence[B3Episode], fold_manifest: dict[str, Any], *, fold_id: int, partition: str,
) -> list[B3Episode]:
    """Select the pre-frozen 600/200 episode partition without reordering identities."""

    if partition not in {"train", "validation"}:
        raise ValueError("partition must be train or validation")
    folds = {int(item["fold_id"]): item for item in fold_manifest.get("folds", [])}
    if fold_id not in folds:
        raise ValueError(f"fold is not present in manifest: {fold_id}")
    field = "train_identities" if partition == "train" else "validation_identities"
    selected = set(folds[fold_id][field])
    by_key = {episode.canonical_parent_key: episode for episode in episodes}
    if set(by_key) != set().union(*(set(item["train_identities"]) | set(item["validation_identities"]) for item in folds.values())):
        raise ValueError("episodes do not match the sealed fold identity universe")
    result = [by_key[key] for key in sorted(selected)]
    expected_count = 600 if partition == "train" else 200
    if len(result) != expected_count:
        raise ValueError(f"fold {fold_id} {partition} count mismatch: {len(result)}")
    return result


__all__ = [
    "FIT_SPLIT", "FORMAL_PROVENANCE", "B3Episode", "B3Batch", "B3EpisodeSampler",
    "load_formal_registry_csv", "load_episode", "pad_episode_batch", "compute_fit_normalization",
    "select_fit_fold_episodes",
]
