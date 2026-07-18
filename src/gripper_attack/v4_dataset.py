"""Detector V4 dataset: episode loading, batch padding, fold selection, sampler.

Loads S1 student records + V2.1.1 Teacher labels. Derives dynamic features.
Enforces student_valid_mask, candidate_close gate, and quality supervision mask.
"""

from __future__ import annotations

import json, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor

from .v4_formal import V4Normalization, VIEW_FEATURE_COUNTS

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FIT_SPLIT = "FIT_TRAIN"
FIT_STATES = set(range(0, 20))

# ── dynamic feature derivation (from V4 prototype) ────────────────────
def _derive_dynamic_features(base_25d: Tensor, view: str) -> Tensor:
    T = base_25d.shape[0]
    if view == "A":
        return base_25d

    IDX_QPOS, IDX_CMD = 0, 1
    IDX_X, IDX_Y, IDX_Z = 4, 5, 6
    IDX_TSC = 23
    qpos, cmd = base_25d[:, IDX_QPOS], base_25d[:, IDX_CMD]
    eef_x, eef_y, eef_z = base_25d[:, IDX_X], base_25d[:, IDX_Y], base_25d[:, IDX_Z]
    tsc = base_25d[:, IDX_TSC]
    feats = [base_25d]

    if view in ("B", "C"):
        dq = torch.zeros(T); dq[1:] = qpos[1:] - qpos[:-1]; feats.append(dq.unsqueeze(1))
        d2q = torch.zeros(T); d2q[2:] = dq[2:] - dq[1:-1]; feats.append(d2q.unsqueeze(1))
        dev = (cmd - qpos).abs(); feats.append(dev.unsqueeze(1))
        dwell = torch.zeros(T); c = 0
        for t in range(T):
            c = c + 1 if tsc[t] >= 0 else 0; dwell[t] = float(c)
        feats.append(dwell.unsqueeze(1))
        tsco = torch.full((T,), -1.0); lo = -1
        for t in range(T):
            if t == 0 and tsc[t] >= 0: lo = t
            elif t > 0 and tsc[t] >= 0 and tsc[t-1] < 0: lo = t
            if lo >= 0: tsco[t] = float(t - lo)
        feats.append(tsco.unsqueeze(1))
        rcc = torch.zeros(T); oc = 0; prev = False
        for t in range(T):
            cl = tsc[t] >= 0
            if cl and not prev: oc += 1
            prev = cl; rcc[t] = float(oc)
        feats.append(rcc.unsqueeze(1))
        trend = torch.zeros(T); ema = 0.0
        for t in range(T): ema = 0.9 * ema + 0.1 * dq[t]; trend[t] = ema
        feats.append(trend.unsqueeze(1))
        rcv = torch.zeros(T)
        for t in range(T):
            w = cmd[max(0, t-9):t+1]; rcv[t] = float(w.var(unbiased=False)) if len(w) >= 2 else 0.0
        feats.append(rcv.unsqueeze(1))

    if view == "C":
        vel = torch.zeros(T)
        if T >= 2:
            dx = eef_x[1:]-eef_x[:-1]; dy = eef_y[1:]-eef_y[:-1]; dz = eef_z[1:]-eef_z[:-1]
            vel[1:] = torch.sqrt(dx*dx+dy*dy+dz*dz)
        feats.append(vel.unsqueeze(1))
        acc = torch.zeros(T); acc[2:] = vel[2:]-vel[1:-1]; feats.append(acc.unsqueeze(1))
        vz = torch.zeros(T); vz[1:] = eef_z[1:]-eef_z[:-1]; feats.append(vz.unsqueeze(1))
        stab = torch.zeros(T)
        for t in range(T):
            w = min(20, t+1)
            if w >= 2:
                px = eef_x[max(0,t-w+1):t+1]; py = eef_y[max(0,t-w+1):t+1]; pz = eef_z[max(0,t-w+1):t+1]
                stab[t] = float(torch.sqrt(px.var(unbiased=False)+py.var(unbiased=False)+pz.var(unbiased=False)))
        feats.append(stab.unsqueeze(1))
        disp = torch.zeros(T); op = None
        for t in range(T):
            if t == 0 and tsc[t] >= 0: op = (eef_x[t], eef_y[t], eef_z[t])
            elif t > 0 and tsc[t] >= 0 and tsc[t-1] < 0: op = (eef_x[t], eef_y[t], eef_z[t])
            if op is not None:
                dx = eef_x[t]-op[0]; dy = eef_y[t]-op[1]; dz = eef_z[t]-op[2]
                disp[t] = float(torch.sqrt(dx*dx+dy*dy+dz*dz))
        feats.append(disp.unsqueeze(1))
        ac = torch.zeros(T)
        for t in range(T):
            w = cmd[max(0,t-9):t+1]
            if len(w) >= 2:
                mv = w.mode().values; ac[t] = float((w == mv).sum()) / len(w)
            else: ac[t] = 1.0
        feats.append(ac.unsqueeze(1))

    return torch.cat(feats, dim=1)


# ── episode ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class V4Episode:
    canonical_parent_key: str
    suite: str
    task_idx: int
    state_id: int
    split: str
    features: Tensor              # [T, F]
    student_valid_mask: Tensor    # [T] bool
    candidate_close: Tensor       # [T] bool
    quality_target: Tensor        # [T] float, 1=valid, 0=invalid, -1=masked
    quality_known_mask: Tensor    # [T] bool
    release_target: Tensor        # [T] float, 1=release_imminent, 0=not, -1=no label
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        T = len(self.features)
        for name, t in [("student_valid_mask", self.student_valid_mask),
                         ("candidate_close", self.candidate_close),
                         ("quality_known_mask", self.quality_known_mask)]:
            if t.shape != (T,) or t.dtype != torch.bool:
                raise ValueError(f"{name} shape/dtype mismatch: {t.shape} {t.dtype}")
        for name, t in [("quality_target", self.quality_target),
                         ("release_target", self.release_target)]:
            if t.shape != (T,):
                raise ValueError(f"{name} shape mismatch: {t.shape}")

    @property
    def fold_id(self) -> int:
        return self.state_id // 5


@dataclass(frozen=True)
class V4Batch:
    features: Tensor                # [B, T_max, F]
    student_valid_mask: Tensor      # [B, T_max]
    candidate_close: Tensor         # [B, T_max]
    quality_target: Tensor          # [B, T_max]
    quality_supervision_mask: Tensor  # [B, T_max] = student_valid & candidate_close & known
    release_target: Tensor          # [B, T_max]
    padding_mask: Tensor            # [B, T_max]
    episode_boundaries: Tensor      # [B, T_max] — True at step 0 of each episode
    episodes: tuple[V4Episode, ...]


# ── loading ─────────────────────────────────────────────────────────────
def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_v4_episode(s1_root: Path, v21_root: Path,
                    suite: str, task: int, state: int,
                    view: str) -> Optional[V4Episode]:
    s1_path = s1_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "student_input_records.jsonl"
    v21_path = v21_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "teacher_v21_labels.jsonl"
    if not s1_path.exists() or not v21_path.exists():
        return None

    students = _jsonl(s1_path)
    v21_labels = _jsonl(v21_path)
    if not students:
        return None

    cid = f"{suite}/task_{task:02d}/state_{state:02d}"
    T = len(students)

    features_25d = torch.tensor(
        [[float(v) for v in r["features_25d"]] for r in students], dtype=torch.float32)
    features = _derive_dynamic_features(features_25d, view)

    student_valid = torch.tensor(
        [bool(r.get("valid", True)) for r in students], dtype=torch.bool)

    quality_target = torch.full((T,), -1.0)
    quality_known = torch.zeros(T, dtype=torch.bool)
    cand_close = torch.zeros(T, dtype=torch.bool)
    release_target = torch.full((T,), -1.0)

    for i, lab in enumerate(v21_labels):
        if i >= T:
            break
        cand_close[i] = bool(lab.get("candidate_close", False))
        km = bool(lab.get("known_mask", False))
        if km:
            quality_known[i] = True
            quality_target[i] = 1.0 if lab.get("quality_valid") else 0.0
            release_target[i] = 1.0 if lab.get("release_imminent") else 0.0

    # Quality supervision mask: student_valid AND candidate_close AND known
    quality_supervision = student_valid & cand_close & quality_known

    # Source artifact SHA from first student record
    source_sha = str(students[0].get("source_artifact_sha256", ""))

    return V4Episode(
        canonical_parent_key=cid, suite=suite, task_idx=task, state_id=state,
        split=FIT_SPLIT, features=features, student_valid_mask=student_valid,
        candidate_close=cand_close, quality_target=quality_target,
        quality_known_mask=quality_supervision, release_target=release_target,
        source_artifact_sha256=source_sha,
    )


# ── batch ───────────────────────────────────────────────────────────────
def pad_v4_episode_batch(episodes: Sequence[V4Episode]) -> V4Batch:
    if not episodes:
        raise ValueError("at least one episode required")
    B = len(episodes)
    T_max = max(ep.features.shape[0] for ep in episodes)
    F = episodes[0].features.shape[1]

    features = torch.zeros(B, T_max, F)
    svm = torch.zeros(B, T_max, dtype=torch.bool)
    cc = torch.zeros(B, T_max, dtype=torch.bool)
    qt = torch.full((B, T_max), -1.0)
    qsm = torch.zeros(B, T_max, dtype=torch.bool)
    rt = torch.full((B, T_max), -1.0)
    padding = torch.zeros(B, T_max, dtype=torch.bool)
    boundaries = torch.zeros(B, T_max, dtype=torch.bool)

    for b, ep in enumerate(episodes):
        T_ep = ep.features.shape[0]
        features[b, :T_ep] = ep.features
        svm[b, :T_ep] = ep.student_valid_mask
        cc[b, :T_ep] = ep.candidate_close
        qt[b, :T_ep] = ep.quality_target
        qsm[b, :T_ep] = ep.quality_known_mask
        rt[b, :T_ep] = ep.release_target
        padding[b, :T_ep] = True
        boundaries[b, 0] = True  # episode start at step 0

    return V4Batch(features, svm, cc, qt, qsm, rt, padding, boundaries, tuple(episodes))


# ── normalization ──────────────────────────────────────────────────────
def compute_v4_fold_normalization(episodes: Iterable[V4Episode],
                                  view: str) -> V4Normalization:
    episodes = tuple(episodes)
    if not episodes:
        raise ValueError("no episodes for normalization")
    for ep in episodes:
        if ep.state_id not in FIT_STATES:
            raise ValueError(f"normalization requires FIT states 0-19, got {ep.state_id}")

    # Only use student-valid steps
    valid_features = torch.cat(
        [ep.features[ep.student_valid_mask] for ep in episodes], dim=0)
    if len(valid_features) == 0:
        raise ValueError("no valid steps for normalization")

    mean = tuple(float(v) for v in valid_features.mean(dim=0))
    std = tuple(float(v) for v in valid_features.std(dim=0, unbiased=False).clamp_min(1e-6))
    return V4Normalization(mean, std, VIEW_FEATURE_COUNTS[view], view)


# ── fold selection ─────────────────────────────────────────────────────
def select_fold_episodes(all_episodes: list[V4Episode], fold_id: int,
                         split: str = "train") -> list[V4Episode]:
    val_states = set(range(fold_id * 5, (fold_id + 1) * 5))
    if split == "validation":
        return [ep for ep in all_episodes if ep.state_id in val_states]
    else:
        return [ep for ep in all_episodes if ep.state_id not in val_states
                and ep.state_id in FIT_STATES]


# ── sampler ─────────────────────────────────────────────────────────────
class V4EpisodeSampler:
    """Suite/task-balanced sampler with per-epoch seed variation."""

    def __init__(self, episodes: Sequence[V4Episode], base_seed: int = 20260717):
        self.episodes = tuple(episodes)
        self.base_seed = base_seed
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i, ep in enumerate(self.episodes):
            groups[(ep.suite, ep.task_idx)].append(i)
        self.groups = {k: sorted(v, key=lambda i: self.episodes[i].canonical_parent_key)
                      for k, v in groups.items()}

    def ordered_indices(self, epoch: int, shuffle: bool = True) -> list[int]:
        rng = random.Random(self.base_seed + epoch * 1000)
        queues = {k: list(v) for k, v in sorted(self.groups.items())}
        if shuffle:
            for v in queues.values():
                rng.shuffle(v)
        order = []
        while any(queues.values()):
            for k in sorted(queues):
                if queues[k]:
                    order.append(queues[k].pop(0))
        return order
