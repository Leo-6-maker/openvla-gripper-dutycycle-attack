"""Corrected FIT-only V4 dataset and causal feature construction.

The old prototype used positional guesses for the Official 25D vector.  This
module derives every base field through the frozen ``SC5_FEATURES`` names and
uses a last-valid-observation policy for dynamic history.  Invalid steps never
update feature state and never receive supervision.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor

from .v4_contract import (
    FEATURE_INDEX,
    FIT_STATES,
    PHASE_INDEX,
    SUITES,
    VIEW_FEATURE_COUNTS,
)

# Keep the public names used by the earlier prototype.
FIT_SPLIT = "FIT_TRAIN"


def _zero_row(device: torch.device) -> list[float]:
    return [0.0] * 8


def derive_dynamic_features(
    base_25d: Tensor,
    view: str,
    student_valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Build View A/B/C using the official feature names and valid history.

    ``student_valid_mask`` defaults to all-valid for small engineering fixtures.
    For real data it is always passed by ``load_v4_episode``.  Invalid rows
    retain the original 25D row but all derived values are neutral and do not
    update the last-valid state used by later rows.
    """
    if view not in VIEW_FEATURE_COUNTS:
        raise ValueError(f"unknown feature view: {view}")
    if base_25d.ndim != 2 or base_25d.shape[1] != 25:
        raise ValueError(f"expected [T,25], got {tuple(base_25d.shape)}")
    if not torch.isfinite(base_25d).all():
        raise ValueError("base 25D contains NaN/Inf")
    T = int(base_25d.shape[0])
    valid = (
        torch.ones(T, dtype=torch.bool, device=base_25d.device)
        if student_valid_mask is None
        else student_valid_mask.to(device=base_25d.device, dtype=torch.bool)
    )
    if valid.shape != (T,):
        raise ValueError(f"student_valid_mask shape mismatch: {tuple(valid.shape)}")
    if view == "A":
        return base_25d

    # Name-bound access.  No raw numeric feature index is used here.
    qpos = base_25d[:, FEATURE_INDEX["gripper_qpos"]]
    command = base_25d[:, FEATURE_INDEX["gripper_command"]]
    eef = torch.stack(
        [
            base_25d[:, FEATURE_INDEX["eef_x"]],
            base_25d[:, FEATURE_INDEX["eef_y"]],
            base_25d[:, FEATURE_INDEX["eef_z"]],
        ],
        dim=1,
    )
    time_since_close = base_25d[:, FEATURE_INDEX["time_since_close"]]

    b_rows: list[list[float]] = []
    c_rows: list[list[float]] = []
    command_history: list[float] = []
    qpos_history: list[float] = []
    eef_history: list[Tensor] = []
    last_qpos: Optional[float] = None
    last_dq = 0.0
    previous_closed = False
    dwell = 0
    close_count = 0
    close_onset_position: Optional[Tensor] = None
    opening_ema = 0.0
    previous_speed = 0.0

    for t in range(T):
        if not bool(valid[t]):
            b_rows.append(_zero_row(base_25d.device))
            if view == "C":
                c_rows.append([0.0] * 6)
            continue

        q = float(qpos[t])
        cmd = float(command[t])
        closed = float(time_since_close[t]) >= 0.0
        if last_qpos is None:
            dq = 0.0
        else:
            dq = q - last_qpos
        d2q = dq - last_dq
        deviation = abs(cmd - q)
        if closed:
            dwell = dwell + 1 if previous_closed else 1
            if not previous_closed:
                close_count += 1
                close_onset_position = eef[t].detach().clone()
        else:
            dwell = 0
            close_onset_position = None
        onset_time = float(time_since_close[t]) if closed else -1.0
        opening_ema = 0.9 * opening_ema + 0.1 * dq
        command_history.append(cmd)
        qpos_history.append(q)
        eef_history.append(eef[t].detach().clone())
        command_window = command_history[-10:]
        command_mean = sum(command_window) / len(command_window)
        command_variance = sum((x - command_mean) ** 2 for x in command_window) / len(command_window)

        b_rows.append([
            dq,
            d2q,
            deviation,
            float(dwell),
            onset_time,
            float(close_count),
            opening_ema,
            command_variance,
        ])

        if view == "C":
            if len(eef_history) >= 2:
                delta = eef_history[-1] - eef_history[-2]
                speed = float(torch.linalg.vector_norm(delta))
                vertical = float(delta[2])
            else:
                speed = 0.0
                vertical = 0.0
            acceleration = speed - previous_speed
            history = eef_history[-20:]
            positions = torch.stack(history)
            stability = float(torch.sqrt(positions.var(dim=0, unbiased=False).sum())) if len(history) >= 2 else 0.0
            if close_onset_position is None:
                displacement = 0.0
            else:
                displacement = float(torch.linalg.vector_norm(eef[t] - close_onset_position))
            counts: dict[float, int] = {}
            for value in command_window:
                counts[value] = counts.get(value, 0) + 1
            consistency = max(counts.values()) / len(command_window)
            c_rows.append([speed, acceleration, vertical, stability, displacement, consistency])
            previous_speed = speed

        last_qpos = q
        last_dq = dq
        previous_closed = closed

    device = base_25d.device
    b = torch.tensor(b_rows, dtype=base_25d.dtype, device=device)
    features = [base_25d, b]
    if view == "C":
        features.append(torch.tensor(c_rows, dtype=base_25d.dtype, device=device))
    out = torch.cat(features, dim=1)
    if out.shape[1] != VIEW_FEATURE_COUNTS[view]:
        raise AssertionError(f"{view} generated {out.shape[1]} features")
    if not torch.isfinite(out).all():
        raise ValueError(f"{view} contains NaN/Inf")
    return out


# Backward-compatible private name used by old callers.
_derive_dynamic_features = derive_dynamic_features


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class V4Episode:
    canonical_parent_key: str
    suite: str
    task_idx: int
    state_id: int
    split: str
    features: Tensor
    student_valid_mask: Tensor
    candidate_close: Tensor
    quality_target: Tensor
    quality_supervision_mask: Tensor
    release_target: Tensor
    release_supervision_mask: Tensor
    event_id: Tensor
    phase_id: Tensor
    window_id: Tensor
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        T = len(self.features)
        if self.features.ndim != 2:
            raise ValueError("features must be [T,F]")
        for name, value in (
            ("student_valid_mask", self.student_valid_mask),
            ("candidate_close", self.candidate_close),
            ("quality_supervision_mask", self.quality_supervision_mask),
            ("release_supervision_mask", self.release_supervision_mask),
        ):
            if value.shape != (T,) or value.dtype != torch.bool:
                raise ValueError(f"{name} shape/dtype mismatch: {value.shape} {value.dtype}")
        for name, value in (
            ("quality_target", self.quality_target),
            ("release_target", self.release_target),
        ):
            if value.shape != (T,):
                raise ValueError(f"{name} shape mismatch: {value.shape}")
        for name, value in (("event_id", self.event_id), ("phase_id", self.phase_id), ("window_id", self.window_id)):
            if value.shape != (T,):
                raise ValueError(f"{name} shape mismatch: {value.shape}")

    @property
    def fold_id(self) -> int:
        return self.state_id // 5

    @property
    def n_steps(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class V4Batch:
    features: Tensor
    student_valid_mask: Tensor
    candidate_close: Tensor
    quality_target: Tensor
    quality_supervision_mask: Tensor
    release_target: Tensor
    release_supervision_mask: Tensor
    event_id: Tensor
    phase_id: Tensor
    window_id: Tensor
    padding_mask: Tensor
    episode_boundaries: Tensor
    episodes: tuple[V4Episode, ...]


def _load_phase_bounds(path: Path) -> dict[int, tuple[int, int]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, tuple[int, int]] = {}
    for phase in data.get("phases", []):
        event_id = int(phase.get("event_id", -1))
        if event_id >= 0:
            result[event_id] = (int(phase.get("start_step", 0)), int(phase.get("end_step", 0)))
    return result


def load_v4_episode(
    s1_root: Path,
    teacher_root: Path,
    suite: str,
    task: int,
    state: int,
    view: str,
) -> Optional[V4Episode]:
    if suite not in SUITES or state not in FIT_STATES:
        raise ValueError(f"FIT-only loader received {suite}/task_{task:02d}/state_{state:02d}")
    ident = f"{suite}/task_{task:02d}/state_{state:02d}"
    ident_dir = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    s1_path = ident_dir / "student_input_records.jsonl"
    teacher_dir = teacher_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
    teacher_path = next(
        (teacher_dir / name for name in ("teacher_v213_labels.jsonl", "teacher_v212_labels.jsonl", "teacher_v21_labels.jsonl") if (teacher_dir / name).is_file()),
        teacher_dir / "teacher_v213_labels.jsonl",
    )
    phases_path = teacher_path.parent / "close_phases.json"
    if not s1_path.is_file() or not teacher_path.is_file():
        return None
    students, labels = _jsonl(s1_path), _jsonl(teacher_path)
    if not students or len(students) != len(labels):
        raise ValueError(f"{ident}: student/teacher length mismatch")
    T = len(students)
    for index, row in enumerate(students):
        if row.get("canonical_parent_key", ident) != ident or int(row.get("step", index)) != index:
            raise ValueError(f"{ident}: student identity/step mismatch at {index}")
        if len(row.get("features_25d", [])) != 25:
            raise ValueError(f"{ident}: non-25D student row at {index}")
    for index, row in enumerate(labels):
        if int(row.get("step", index)) != index:
            raise ValueError(f"{ident}: teacher step mismatch at {index}")

    base = torch.tensor([[float(value) for value in row["features_25d"]] for row in students], dtype=torch.float32)
    student_valid = torch.tensor([bool(row.get("valid", True)) for row in students], dtype=torch.bool)
    features = derive_dynamic_features(base, view, student_valid)

    candidate_close = torch.tensor([bool(row.get("candidate_close", False)) for row in labels], dtype=torch.bool)
    quality_valid = torch.tensor([bool(row.get("quality_valid", row.get("valid_retention", False))) for row in labels])
    veto_invalid = torch.tensor([bool(row.get("veto_invalid", row.get("false_trigger_veto", False))) for row in labels])
    known = torch.tensor([bool(row.get("known_mask", row.get("event_valid_mask", False))) for row in labels])
    exclusive = quality_valid ^ veto_invalid
    quality_mask = student_valid & candidate_close & known & exclusive
    quality_target = torch.full((T,), -1.0, dtype=torch.float32)
    quality_target[quality_mask] = quality_valid[quality_mask].float()

    event_id = torch.tensor([int(row.get("event_id", -1)) for row in labels], dtype=torch.long)
    phase_id = torch.tensor(
        [int(row.get("phase_id", PHASE_INDEX.get(str(row.get("phase_name", row.get("phase", "UNKNOWN"))), PHASE_INDEX["UNKNOWN"]))) for row in labels],
        dtype=torch.long,
    )
    window_id = torch.tensor([int(row.get("window_id", int(row.get("event_id", -1)))) for row in labels], dtype=torch.long)
    release_known = torch.tensor(
        [bool(row.get("release_known", row.get("known_mask", False))) for row in labels], dtype=torch.bool
    )
    release_target = torch.full((T,), -1.0, dtype=torch.float32)
    release_target[release_known] = torch.tensor(
        [bool(row.get("release_imminent", False)) for row in labels], dtype=torch.float32
    )[release_known]
    release_mask = student_valid & release_known & (event_id >= 0)

    source_sha = str(students[0].get("source_artifact_sha256", ""))
    return V4Episode(
        canonical_parent_key=ident,
        suite=suite,
        task_idx=task,
        state_id=state,
        split=FIT_SPLIT,
        features=features,
        student_valid_mask=student_valid,
        candidate_close=candidate_close,
        quality_target=quality_target,
        quality_supervision_mask=quality_mask,
        release_target=release_target,
        release_supervision_mask=release_mask,
        event_id=event_id,
        phase_id=phase_id,
        window_id=window_id,
        source_artifact_sha256=source_sha,
    )


def pad_v4_episode_batch(episodes: Sequence[V4Episode]) -> V4Batch:
    if not episodes:
        raise ValueError("at least one episode required")
    batch_size = len(episodes)
    max_steps = max(ep.n_steps for ep in episodes)
    feature_count = episodes[0].features.shape[1]
    device = episodes[0].features.device
    features = torch.zeros(batch_size, max_steps, feature_count, device=device)
    student_valid = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    candidate_close = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    quality_target = torch.full((batch_size, max_steps), -1.0, device=device)
    quality_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    release_target = torch.full((batch_size, max_steps), -1.0, device=device)
    release_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    event_id = torch.full((batch_size, max_steps), -1, dtype=torch.long, device=device)
    phase_id = torch.full((batch_size, max_steps), PHASE_INDEX["UNKNOWN"], dtype=torch.long, device=device)
    window_id = torch.full((batch_size, max_steps), -1, dtype=torch.long, device=device)
    padding = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    boundaries = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)
    for batch_index, episode in enumerate(episodes):
        n = episode.n_steps
        features[batch_index, :n] = episode.features
        student_valid[batch_index, :n] = episode.student_valid_mask
        candidate_close[batch_index, :n] = episode.candidate_close
        quality_target[batch_index, :n] = episode.quality_target
        quality_mask[batch_index, :n] = episode.quality_supervision_mask
        release_target[batch_index, :n] = episode.release_target
        release_mask[batch_index, :n] = episode.release_supervision_mask
        event_id[batch_index, :n] = episode.event_id
        phase_id[batch_index, :n] = episode.phase_id
        window_id[batch_index, :n] = episode.window_id
        padding[batch_index, :n] = True
        boundaries[batch_index, 0] = True
    return V4Batch(
        features,
        student_valid,
        candidate_close,
        quality_target,
        quality_mask,
        release_target,
        release_mask,
        event_id,
        phase_id,
        window_id,
        padding,
        boundaries,
        tuple(episodes),
    )


def compute_v4_fold_normalization(episodes: Iterable[V4Episode], view: str):
    from .v4_formal import V4Normalization

    episodes = tuple(episodes)
    if not episodes or any(ep.state_id not in FIT_STATES for ep in episodes):
        raise ValueError("normalization accepts FIT episodes only")
    values = torch.cat([ep.features[ep.student_valid_mask] for ep in episodes], dim=0)
    if values.numel() == 0:
        raise ValueError("no valid feature rows")
    mean = tuple(float(value) for value in values.mean(dim=0))
    std = tuple(float(value) for value in values.std(dim=0, unbiased=False).clamp_min(1e-6))
    return V4Normalization(mean, std, VIEW_FEATURE_COUNTS[view], view)


def select_fold_episodes(episodes: Sequence[V4Episode], fold_id: int, split: str) -> list[V4Episode]:
    if fold_id not in range(4):
        raise ValueError("fold_id must be 0..3")
    valid_states = set(range(fold_id * 5, fold_id * 5 + 5))
    if split == "validation":
        return [ep for ep in episodes if ep.state_id in valid_states]
    if split == "train":
        return [ep for ep in episodes if ep.state_id in FIT_STATES and ep.state_id not in valid_states]
    raise ValueError(f"unknown fold split: {split}")


class V4EpisodeSampler:
    """Deterministic suite/task/episode round-robin ordering."""

    def __init__(self, episodes: Sequence[V4Episode], base_seed: int = 20260717):
        self.episodes = tuple(episodes)
        self.base_seed = int(base_seed)
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, episode in enumerate(self.episodes):
            groups[(episode.suite, episode.task_idx)].append(index)
        self.groups = {key: sorted(value, key=lambda i: self.episodes[i].canonical_parent_key) for key, value in groups.items()}

    def ordered_indices(self, epoch: int, shuffle: bool = True) -> list[int]:
        rng = random.Random(self.base_seed + int(epoch) * 1000)
        queues = {key: list(value) for key, value in sorted(self.groups.items())}
        if shuffle:
            for value in queues.values():
                rng.shuffle(value)
        order: list[int] = []
        while any(queues.values()):
            for key in sorted(queues):
                if queues[key]:
                    order.append(queues[key].pop(0))
        return order
