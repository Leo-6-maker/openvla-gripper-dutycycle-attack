"""FIT-only V5-A dataset loader and window aggregation helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .v5_protocol import V5Window, validate_phase_windows, validate_student_features, validate_teacher_row


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fit_registry(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fit: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("canonical_parent_key", ""))
        state = int(row.get("state_id", -1))
        task = int(row.get("task_idx", -1))
        parts = key.split("/")
        if len(parts) != 3 or state not in range(20) or task not in range(10):
            continue
        if key in seen:
            raise ValueError(f"duplicate FIT identity: {key}")
        if key != f"{row['suite']}/task_{task:02d}/state_{state:02d}":
            raise ValueError(f"registry identity mismatch: {key}")
        seen.add(key)
        fit.append(dict(row, task_idx=task, state_id=state, canonical_parent_key=key))
    if len(fit) != 800:
        raise ValueError(f"V5-A requires exactly 800 FIT identities, got {len(fit)}")
    return sorted(fit, key=lambda row: row["canonical_parent_key"])


@dataclass(frozen=True)
class V5Episode:
    canonical_parent_key: str
    suite: str
    task_idx: int
    state_id: int
    features_25d: Tensor
    valid_mask: Tensor
    candidate_close: Tensor
    utility_tier: Tensor
    known_mask: Tensor
    release_imminent: Tensor
    regrasp_or_unstable: Tensor
    release_known_mask: Tensor
    regrasp_known_mask: Tensor
    windows: tuple[V5Window, ...]

    def __post_init__(self) -> None:
        steps = self.features_25d.shape[0]
        if self.features_25d.shape != (steps, 25):
            raise ValueError("V5-A features must have shape [T,25]")
        for name, value in (
            ("valid_mask", self.valid_mask),
            ("candidate_close", self.candidate_close),
            ("utility_tier", self.utility_tier),
            ("known_mask", self.known_mask),
            ("release_imminent", self.release_imminent),
            ("regrasp_or_unstable", self.regrasp_or_unstable),
            ("release_known_mask", self.release_known_mask),
            ("regrasp_known_mask", self.regrasp_known_mask),
        ):
            if value.shape != (steps,):
                raise ValueError(f"{name} shape mismatch")
        if self.valid_mask.dtype != torch.bool or self.candidate_close.dtype != torch.bool or self.known_mask.dtype != torch.bool:
            raise TypeError("V5-A masks must be bool")
        if not bool(torch.isfinite(self.features_25d).all()):
            raise ValueError("V5-A features contain NaN/Inf")
        validate_phase_windows(self.windows)


def _episode_root(root: Path, row: dict[str, Any]) -> Path:
    return root / str(row["suite"]) / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"


def load_v5_episode(s1_root: Path, teacher_root: Path, row: dict[str, Any]) -> V5Episode:
    identity = str(row["canonical_parent_key"])
    s1 = _episode_root(s1_root, row)
    teacher = _episode_root(teacher_root, row)
    students = _jsonl(s1 / "student_input_records.jsonl")
    teachers = _jsonl(teacher / "v5_teacher_utility.jsonl")
    if not students or len(students) != len(teachers):
        raise ValueError(f"V5-A stream mismatch: {identity}")
    features: list[list[float]] = []
    valid: list[bool] = []
    candidate: list[bool] = []
    tiers: list[int] = []
    known: list[bool] = []
    release: list[bool] = []
    regrasp: list[bool] = []
    window_rows: list[dict[str, Any]] = []
    for index, (student, teacher_row) in enumerate(zip(students, teachers)):
        validate_student_features(student)
        validate_teacher_row(teacher_row)
        if student.get("canonical_parent_key") != identity or int(student.get("step", -1)) != index:
            raise ValueError(f"V5-A Student identity/step mismatch: {identity}:{index}")
        if teacher_row.get("canonical_parent_key") != identity or int(teacher_row.get("step", -1)) != index:
            raise ValueError(f"V5-A Teacher identity/step mismatch: {identity}:{index}")
        features.append([float(value) for value in student["features_25d"]])
        valid.append(bool(student["valid"]))
        candidate.append(bool(teacher_row["candidate_close"]))
        known.append(bool(teacher_row["known_mask"]))
        tiers.append(-1 if teacher_row["utility_tier"] is None else int(teacher_row["utility_tier"]))
        release.append(bool(teacher_row["release_imminent"]))
        regrasp.append(bool(teacher_row["regrasp_or_unstable"]))
        window_rows.append({
            "index": index,
            "rankable": bool(student["valid"])
            and bool(teacher_row["candidate_close"])
            and bool(teacher_row["known_mask"])
            and str(teacher_row["phase_name"]) != "UNKNOWN"
            and not str(teacher_row["window_id"]).startswith("none:"),
            "window_id": str(teacher_row["window_id"]),
            "phase_name": str(teacher_row["phase_name"]),
            "utility_tier": None if teacher_row["utility_tier"] is None else int(teacher_row["utility_tier"]),
        })
    windows: list[V5Window] = []
    active: dict[str, Any] | None = None
    segment_counts: dict[str, int] = {}
    for item in window_rows + [{"rankable": False}]:
        contiguous = bool(
            active
            and item.get("rankable")
            and int(item["index"]) == int(active["indices"][-1]) + 1
            and item["window_id"] == active["window_id"]
            and item["phase_name"] == active["phase_name"]
            and item["utility_tier"] == active["utility_tier"]
        )
        if contiguous:
            active["indices"].append(int(item["index"]))
            continue
        if active:
            base_id = str(active["window_id"])
            ordinal = segment_counts.get(base_id, 0)
            segment_counts[base_id] = ordinal + 1
            window_id = base_id if ordinal == 0 else f"{base_id}#segment{ordinal}"
            indices = tuple(active["indices"])
            windows.append(V5Window(
                episode_id=identity,
                window_id=window_id,
                start=indices[0],
                end=indices[-1],
                phase_name=str(active["phase_name"]),
                utility_tier=active["utility_tier"],
                known=True,
                candidate_close=True,
                step_indices=indices,
            ))
            active = None
        if item.get("rankable"):
            active = {
                "window_id": item["window_id"],
                "phase_name": item["phase_name"],
                "utility_tier": item["utility_tier"],
                "indices": [int(item["index"])],
            }
    return V5Episode(
        canonical_parent_key=identity,
        suite=str(row["suite"]),
        task_idx=int(row["task_idx"]),
        state_id=int(row["state_id"]),
        features_25d=torch.tensor(features, dtype=torch.float32),
        valid_mask=torch.tensor(valid, dtype=torch.bool),
        candidate_close=torch.tensor(candidate, dtype=torch.bool),
        utility_tier=torch.tensor(tiers, dtype=torch.int64),
        known_mask=torch.tensor(known, dtype=torch.bool),
        release_imminent=torch.tensor(release, dtype=torch.bool),
        regrasp_or_unstable=torch.tensor(regrasp, dtype=torch.bool),
        release_known_mask=torch.tensor([bool(v and c and k) for v, c, k in zip(valid, candidate, known)], dtype=torch.bool),
        regrasp_known_mask=torch.tensor([bool(v and c and k) for v, c, k in zip(valid, candidate, known)], dtype=torch.bool),
        windows=tuple(windows),
    )


def load_v5_episodes(s1_root: Path, teacher_root: Path, rows: Iterable[dict[str, Any]]) -> list[V5Episode]:
    return [load_v5_episode(s1_root, teacher_root, row) for row in rows]


def classify_v5_episode_windows(windows: Sequence[V5Window]) -> str:
    """Return the strict category used by all V5 diagnostics."""

    tiers = [int(window.utility_tier) for window in windows if window.rankable and window.utility_tier is not None]
    has_positive = any(tier >= 2 for tier in tiers)
    has_negative = any(tier <= 1 for tier in tiers)
    if not tiers:
        return "NO_CANDIDATE"
    if has_positive and has_negative:
        return "TRUE_MIXED"
    if has_positive:
        return "POSITIVE_ONLY"
    return "PURE_NEGATIVE"


def compute_v5_normalization(episodes: Iterable[V5Episode]) -> tuple[Tensor, Tensor]:
    values = torch.cat([episode.features_25d[episode.valid_mask] for episode in episodes], dim=0)
    if values.numel() == 0:
        raise ValueError("cannot normalize an empty V5-A training set")
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _aggregate_window_scores(utility_logits: Tensor, episode: V5Episode, *, causal: bool) -> tuple[Tensor, list[dict[str, object]]]:
    if utility_logits.ndim != 1 or utility_logits.shape[0] != len(episode.features_25d):
        raise ValueError("utility logits do not match episode length")
    scores: list[Tensor] = []
    rows: list[dict[str, object]] = []
    for window in episode.windows:
        indices = list(window.step_indices)
        if causal:
            indices = [step for step in indices if step <= int(window.decision_anchor_step)]
        indices = [step for step in indices if 0 <= step < len(utility_logits)]
        if not indices:
            continue
        scores.append(utility_logits[indices].mean())
        rows.append({
            "episode_id": episode.canonical_parent_key,
            "window_id": window.window_id,
            "phase_name": window.phase_name,
            "known": window.known,
            "utility_tier": window.utility_tier,
            "decision_anchor_step": window.decision_anchor_step,
            "step_indices": tuple(indices),
            "minimum_dwell_met": window.minimum_dwell_met,
        })
    if not scores:
        return utility_logits[:0], rows
    return torch.stack(scores), rows


def aggregate_retrospective_window_scores(utility_logits: Tensor, episode: V5Episode) -> tuple[Tensor, list[dict[str, object]]]:
    return _aggregate_window_scores(utility_logits, episode, causal=False)


def causal_window_anchor_scores(utility_logits: Tensor, episode: V5Episode) -> tuple[Tensor, list[dict[str, object]]]:
    return _aggregate_window_scores(utility_logits, episode, causal=True)


aggregate_window_scores = aggregate_retrospective_window_scores


__all__ = [
    "V5Episode",
    "load_fit_registry",
    "load_v5_episode",
    "load_v5_episodes",
    "classify_v5_episode_windows",
    "compute_v5_normalization",
    "aggregate_window_scores",
    "aggregate_retrospective_window_scores",
    "causal_window_anchor_scores",
]
