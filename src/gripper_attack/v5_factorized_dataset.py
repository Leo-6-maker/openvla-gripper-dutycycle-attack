"""Factorized Student dataset adapter (Gate S2).

Narrow adapter on mature V5 dataset infrastructure.
Loads Factorized Teacher V1 labels, produces three-head training targets
with per-head known masks, mechanism route, and event metadata.
"""

from __future__ import annotations

import csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .b3_training_protocol import sha256_file, verify_sealed_directory

FACTORIZED_LABEL_FILENAME = "factorized_teacher_v1.jsonl"
SUPPORTED_ROUTES = {"single_object_pick_place", "multi_object_transfer"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@dataclass(frozen=True)
class FactorizedEpisode:
    canonical_parent_key: str
    suite: str
    task_idx: int
    state_id: int
    mechanism_route: str
    route_supported: bool

    features_25d: Tensor       # [T, 25]
    valid_mask: Tensor          # [T]

    grasp_target: Tensor        # [T] bool
    grasp_known_mask: Tensor    # [T] bool
    manipulation_target: Tensor
    manipulation_known_mask: Tensor
    release_target: Tensor
    release_known_mask: Tensor

    event_id: Tensor            # [T] int
    event_role: list[str]       # [T]
    active_object_name: list[str | None]  # [T]

    k10_feasible: Tensor        # [T] bool, eval only
    k10_known_mask: Tensor

    policy_intent_9d: Tensor    # [T, 9] or empty

    def __post_init__(self):
        T = self.features_25d.shape[0]
        for name, t in [
            ("valid_mask", self.valid_mask),
            ("grasp_target", self.grasp_target), ("grasp_known_mask", self.grasp_known_mask),
            ("manipulation_target", self.manipulation_target), ("manipulation_known_mask", self.manipulation_known_mask),
            ("release_target", self.release_target), ("release_known_mask", self.release_known_mask),
            ("k10_feasible", self.k10_feasible), ("k10_known_mask", self.k10_known_mask),
        ]:
            if t.shape != (T,):
                raise ValueError(f"{name} shape mismatch: expected [{T}], got {t.shape}")
            if t.dtype not in (torch.bool, torch.int64) and name.endswith("_mask"):
                if t.dtype != torch.bool:
                    raise TypeError(f"{name} must be bool")
        if not torch.isfinite(self.features_25d).all():
            raise ValueError("features contain NaN/Inf")
        if self.policy_intent_9d.numel() > 0 and self.policy_intent_9d.shape != (T, 9):
            raise ValueError(f"policy_intent_9d shape mismatch: {self.policy_intent_9d.shape}")


def load_factorized_episode(
    s1_root: Path,
    teacher_root: Path,
    row: dict[str, Any],
    *,
    policy_index: dict[str, list[dict[str, Any]]] | None = None,
) -> FactorizedEpisode:
    identity = row["canonical_parent_key"]
    suite, task_name, state_name = identity.split("/")

    s1_path = s1_root / suite / task_name / state_name / "student_input_records.jsonl"
    students = _jsonl(s1_path)

    teacher_path = (teacher_root / "labels" / suite / task_name / state_name /
                    FACTORIZED_LABEL_FILENAME)
    teachers = _jsonl(teacher_path)

    if len(students) != len(teachers):
        raise ValueError(f"stream length mismatch: {identity}")

    policy_rows = None
    if policy_index is not None:
        policy_rows = policy_index.get(identity)
        if policy_rows is None or len(policy_rows) != len(students):
            raise ValueError(f"policy stream mismatch: {identity}")

    feats, valid = [], []
    g_tgt, g_km = [], []
    m_tgt, m_km = [], []
    r_tgt, r_km = [], []
    eids, eroles, aobjs = [], [], []
    k10_f, k10_km = [], []
    pol_9d, pol_v = [], []

    for i, (stu, tea) in enumerate(zip(students, teachers)):
        if stu.get("canonical_parent_key") != identity or int(stu.get("step", -1)) != i:
            raise ValueError(f"student identity/step: {identity}:{i}")
        if int(tea.get("step", -1)) != i:
            raise ValueError(f"teacher step: {identity}:{i}")

        feats.append([float(v) for v in stu["features_25d"]])
        valid.append(bool(stu.get("valid", True)))

        g_tgt.append(bool(tea["grasp_established"]))
        g_km.append(bool(tea["grasp_established_known_mask"]))
        m_tgt.append(bool(tea["manipulation_active"]))
        m_km.append(bool(tea["manipulation_active_known_mask"]))
        r_tgt.append(bool(tea["release_or_instability"]))
        r_km.append(bool(tea["release_or_instability_known_mask"]))

        eids.append(int(tea.get("event_id", -1)))
        eroles.append(str(tea.get("event_role", "NONE")))
        aobjs.append(tea.get("active_object_name"))

        k10_f.append(bool(tea.get("strict_k10_feasible", False)))
        k10_km.append(bool(tea.get("strict_k10_known_mask", False)))

        if policy_rows is not None:
            pr = policy_rows[i]
            pol_9d.append([float(v) for v in pr["clean_policy_intent_9d"]])
            pol_v.append(bool(pr["valid_intent"]))

    route = str(teachers[0].get("mechanism_type", "unknown_or_ambiguous"))

    # Validate contracts
    for i in range(len(feats)):
        if g_km[i] is False:
            if g_tgt[i]:
                raise ValueError(f"unknown grasp cannot be positive: {identity}:{i}")
        if m_km[i] is False:
            if m_tgt[i]:
                raise ValueError(f"unknown manipulation cannot be positive: {identity}:{i}")
        if m_tgt[i] and not g_tgt[i]:
            raise ValueError(f"manipulation implies grasp: {identity}:{i}")

    return FactorizedEpisode(
        canonical_parent_key=identity, suite=suite, task_idx=int(task_name.split("_")[1]),
        state_id=int(state_name.split("_")[1]),
        mechanism_route=route, route_supported=route in SUPPORTED_ROUTES,
        features_25d=torch.tensor(feats, dtype=torch.float32),
        valid_mask=torch.tensor(valid, dtype=torch.bool),
        grasp_target=torch.tensor(g_tgt, dtype=torch.bool),
        grasp_known_mask=torch.tensor(g_km, dtype=torch.bool),
        manipulation_target=torch.tensor(m_tgt, dtype=torch.bool),
        manipulation_known_mask=torch.tensor(m_km, dtype=torch.bool),
        release_target=torch.tensor(r_tgt, dtype=torch.bool),
        release_known_mask=torch.tensor(r_km, dtype=torch.bool),
        event_id=torch.tensor(eids, dtype=torch.int64),
        event_role=eroles, active_object_name=aobjs,
        k10_feasible=torch.tensor(k10_f, dtype=torch.bool),
        k10_known_mask=torch.tensor(k10_km, dtype=torch.bool),
        policy_intent_9d=torch.tensor(pol_9d, dtype=torch.float32) if pol_9d else torch.empty((0, 9)),
    )


def load_factorized_episodes(
    s1_root: Path, teacher_root: Path,
    rows: list[dict[str, Any]],
    policy_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[FactorizedEpisode]:
    return [load_factorized_episode(s1_root, teacher_root, r, policy_index=policy_index) for r in rows]


def compute_factorized_normalization(episodes: list[FactorizedEpisode]) -> tuple[Tensor, Tensor]:
    values = torch.cat([ep.features_25d[ep.valid_mask] for ep in episodes], dim=0)
    if values.numel() == 0:
        raise ValueError("empty normalization set")
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6)


class FactorizedLoss:
    """Masked BCE per head + consistency, event/route-balanced."""

    def __init__(self, consistency_weight: float = 0.1):
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.consistency_weight = consistency_weight

    def __call__(self, logits: dict[str, Tensor], episode: FactorizedEpisode) -> tuple[Tensor, dict[str, float]]:
        g_logits = logits["grasp"]
        m_logits = logits["manipulation"]
        r_logits = logits["release"]

        g_loss = (self.bce(g_logits, episode.grasp_target.float()) * episode.grasp_known_mask.float()).sum() / max(1, episode.grasp_known_mask.sum())
        m_loss = (self.bce(m_logits, episode.manipulation_target.float()) * episode.manipulation_known_mask.float()).sum() / max(1, episode.manipulation_known_mask.sum())
        r_loss = (self.bce(r_logits, episode.release_target.float()) * episode.release_known_mask.float()).sum() / max(1, episode.release_known_mask.sum())

        p_g = torch.sigmoid(g_logits)
        p_m = torch.sigmoid(m_logits)
        consistency = torch.relu(p_m - p_g).mean()

        total = g_loss + m_loss + r_loss + self.consistency_weight * consistency
        return total, {"grasp": g_loss.item(), "manipulation": m_loss.item(),
                        "release": r_loss.item(), "consistency": consistency.item()}

    def to(self, device):
        return self  # stateless


__all__ = ["FactorizedEpisode", "FactorizedLoss",
           "load_factorized_episode", "load_factorized_episodes",
           "compute_factorized_normalization",
           "FACTORIZED_LABEL_FILENAME", "SUPPORTED_ROUTES"]
