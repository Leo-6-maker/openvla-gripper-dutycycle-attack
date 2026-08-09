"""Factorized Student dataset adapter (Gate S2.1).

Narrow adapter on mature V5 dataset infrastructure.
Loads Factorized Teacher V1 labels, produces three-head training targets
with per-head known masks, mechanism route, event metadata, and policy-intent
valid masks for 25D9D fusion.

Contracts enforced per episode:
  - manipulation → grasp (logical)
  - unknown → not positive
  - unsupported route → all heads unknown
  - features_25d finite, shape [T,25]
  - policy_intent_9d finite when present
  - targets bool dtype, event_id int64
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


def verify_factorized_source_roots(
    s1_root: Path, teacher_root: Path,
    policy_intent_root: Path | None = None,
) -> dict[str, str]:
    """Batch-level seal verification — run once before loading any episode."""
    seals = {
        "s1_root_seal": sha256_file(s1_root / "SHA256SUMS"),
        "teacher_root_seal": sha256_file(teacher_root / "SHA256SUMS"),
    }
    verify_sealed_directory(s1_root)
    verify_sealed_directory(teacher_root)
    if policy_intent_root is not None:
        seals["policy_intent_root_seal"] = sha256_file(policy_intent_root / "SHA256SUMS")
        verify_sealed_directory(policy_intent_root)
    return seals


@dataclass(frozen=True)
class FactorizedEpisode:
    canonical_parent_key: str
    suite: str
    task_idx: int
    state_id: int
    mechanism_route: str
    route_supported: bool

    features_25d: Tensor            # [T, 25]
    valid_mask: Tensor               # [T]

    grasp_target: Tensor             # [T] bool
    grasp_known_mask: Tensor         # [T] bool
    manipulation_target: Tensor      # [T] bool
    manipulation_known_mask: Tensor  # [T] bool
    release_target: Tensor           # [T] bool
    release_known_mask: Tensor       # [T] bool

    event_id: Tensor                 # [T] int64
    event_role: list[str]            # [T]
    active_object_name: list[str | None]  # [T]

    k10_feasible: Tensor             # [T] bool (eval only)
    k10_known_mask: Tensor           # [T] bool

    policy_intent_9d: Tensor         # [T, 9] or empty
    policy_intent_valid_mask: Tensor # [T] bool (or all-False if no 9D)

    def __post_init__(self):
        T = self.features_25d.shape[0]
        for name, t, expected_shape in [
            ("features_25d", self.features_25d, (T, 25)),
            ("valid_mask", self.valid_mask, (T,)),
            ("grasp_target", self.grasp_target, (T,)),
            ("grasp_known_mask", self.grasp_known_mask, (T,)),
            ("manipulation_target", self.manipulation_target, (T,)),
            ("manipulation_known_mask", self.manipulation_known_mask, (T,)),
            ("release_target", self.release_target, (T,)),
            ("release_known_mask", self.release_known_mask, (T,)),
            ("event_id", self.event_id, (T,)),
            ("k10_feasible", self.k10_feasible, (T,)),
            ("k10_known_mask", self.k10_known_mask, (T,)),
            ("policy_intent_valid_mask", self.policy_intent_valid_mask, (T,)),
        ]:
            if t.shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: {t.shape} != {expected_shape}")
        for name, t in [
            ("grasp_target", self.grasp_target), ("grasp_known_mask", self.grasp_known_mask),
            ("manipulation_target", self.manipulation_target), ("manipulation_known_mask", self.manipulation_known_mask),
            ("release_target", self.release_target), ("release_known_mask", self.release_known_mask),
            ("valid_mask", self.valid_mask), ("k10_feasible", self.k10_feasible),
            ("k10_known_mask", self.k10_known_mask), ("policy_intent_valid_mask", self.policy_intent_valid_mask),
        ]:
            if t.dtype != torch.bool:
                raise TypeError(f"{name} must be bool, got {t.dtype}")
        if self.event_id.dtype != torch.int64:
            raise TypeError(f"event_id must be int64, got {self.event_id.dtype}")
        if not torch.isfinite(self.features_25d).all():
            raise ValueError("features_25d contain NaN/Inf")
        if self.policy_intent_9d.numel() > 0:
            if self.policy_intent_9d.shape != (T, 9):
                raise ValueError(f"policy_intent_9d shape: {self.policy_intent_9d.shape}")
            if not torch.isfinite(self.policy_intent_9d).all():
                raise ValueError("policy_intent_9d contain NaN/Inf")
        if len(self.event_role) != T or len(self.active_object_name) != T:
            raise ValueError(f"event_role/active_object_name length != {T}")
        # Logical contracts
        if (self.manipulation_target & ~self.grasp_target).any():
            raise ValueError(f"manipulation without grasp: {self.canonical_parent_key}")
        if (~self.grasp_known_mask & self.grasp_target).any():
            raise ValueError(f"unknown grasp positive: {self.canonical_parent_key}")
        if (~self.manipulation_known_mask & self.manipulation_target).any():
            raise ValueError(f"unknown manipulation positive: {self.canonical_parent_key}")
        if (~self.release_known_mask & self.release_target).any():
            raise ValueError(f"unknown release positive: {self.canonical_parent_key}")
        if not self.route_supported:
            if self.grasp_known_mask.any() or self.manipulation_known_mask.any() or self.release_known_mask.any():
                raise ValueError(f"unsupported route has known heads: {self.canonical_parent_key}")


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
        policy_intent_valid_mask=torch.tensor(pol_v if pol_v else valid, dtype=torch.bool),
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


__all__ = ["FactorizedEpisode",
           "load_factorized_episode", "load_factorized_episodes",
           "compute_factorized_normalization",
           "verify_factorized_source_roots",
           "FACTORIZED_LABEL_FILENAME", "SUPPORTED_ROUTES"]
