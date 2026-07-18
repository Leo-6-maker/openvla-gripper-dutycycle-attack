"""FIT-only V5-A dataset loader and window aggregation helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .b3_training_protocol import sha256_file, verify_sealed_directory
from .v5_protocol import (
    V5_FEATURES_9D,
    V5_STUDENT_FORBIDDEN_FIELDS,
    V5Window,
    feature_order_sha,
    validate_phase_windows,
    validate_student_features,
    validate_teacher_row,
)
from .v5_physics import PHYSICS_TEACHER_FIELDS


_POLICY_INTENT_REQUIRED_FIELDS = frozenset(
    {
        "step",
        "clean_policy_intent_9d",
        "clean_open_probability_mass",
        "clean_close_probability_mass",
        "clean_open_minus_close_log_mass",
        "clean_action_token_entropy_normalized",
        "clean_top1_probability",
        "clean_top1_is_open",
        "clean_top1_is_close",
        "clean_best_open_rank_normalized",
        "clean_best_close_rank_normalized",
        "action_token_ids",
        "clean_action_token_top_ids",
        "clean_action_token_top_logits",
        "score_head_summary",
        "generation_passes_per_step",
        "single_generation_parity_pass",
        "score_adapter_parity_pass",
        "valid_intent",
    }
)


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
    policy_intent_9d: Tensor = field(default_factory=lambda: torch.empty((0, 9), dtype=torch.float32))
    intent_valid_mask: Tensor = field(default_factory=lambda: torch.empty((0,), dtype=torch.bool))

    def __post_init__(self) -> None:
        steps = self.features_25d.shape[0]
        if self.features_25d.shape != (steps, 25):
            raise ValueError("V5-A features must have shape [T,25]")
        if self.policy_intent_9d.numel() == 0 and self.intent_valid_mask.numel() == 0:
            object.__setattr__(self, "policy_intent_9d", torch.zeros((steps, 9), dtype=torch.float32))
            object.__setattr__(self, "intent_valid_mask", torch.zeros((steps,), dtype=torch.bool))
        if self.policy_intent_9d.shape != (steps, 9) or self.intent_valid_mask.shape != (steps,):
            raise ValueError("V5 policy-intent stream must have shape [T,9] and [T]")
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
        if self.intent_valid_mask.dtype != torch.bool:
            raise TypeError("intent_valid_mask must be bool")
        if not bool(torch.isfinite(self.policy_intent_9d).all()):
            raise ValueError("V5 policy-intent features contain NaN/Inf")
        validate_phase_windows(self.windows)


def _episode_root(root: Path, row: dict[str, Any]) -> Path:
    return root / str(row["suite"]) / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"


def _load_physics_protocol(root: Path) -> dict[str, Any]:
    value = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V1":
        raise ValueError("unexpected Physics Teacher protocol")
    constants = value.get("fixed_constants")
    if not isinstance(constants, dict):
        raise ValueError("Physics Teacher protocol lacks fixed constants")
    for name in ("tier2_max_release_risk", "tier2_max_regrasp_risk"):
        if name not in constants:
            raise ValueError(f"Physics Teacher protocol lacks {name}")
    return value


def _physics_row_as_v5_teacher(row: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    if set(row) != PHYSICS_TEACHER_FIELDS:
        missing = sorted(PHYSICS_TEACHER_FIELDS - set(row))
        extra = sorted(set(row) - PHYSICS_TEACHER_FIELDS)
        raise ValueError(f"invalid Physics Teacher row: missing={missing}, extra={extra}")
    known = bool(row["known_mask"])
    tier = None if row["utility_tier"] is None else int(row["utility_tier"])
    if known != (tier is not None):
        raise ValueError("Physics Teacher known/tier mismatch")
    constants = protocol["fixed_constants"]
    release = known and float(row["release_risk"]) >= float(constants["tier2_max_release_risk"])
    regrasp = known and float(row["regrasp_or_instability_risk"]) >= float(constants["tier2_max_regrasp_risk"])
    quality = known and tier >= 2
    veto = known and tier <= 1
    return {
        "canonical_parent_key": row["canonical_parent_key"],
        "step": int(row["step"]),
        "event_id": -1,
        "phase_id": -1,
        "window_id": str(row["window_id"]),
        "phase_name": str(row["phase_name"]),
        "window_start": int(row["window_start"]),
        "window_end": int(row["window_end"]),
        "candidate_close": bool(row["candidate_close"]),
        "quality_valid": quality,
        "veto_invalid": veto,
        "release_imminent": release,
        "regrasp_or_unstable": regrasp,
        "known_mask": known,
        "utility_tier": tier,
        "ranking_group": str(row["window_id"]),
    }


def load_policy_intent_root(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load the independently sealed FIT-only policy-intent derivative.

    The binder already performed source-artifact parity checks.  This loader
    still rechecks the root seal, row schema, identity/step closure, finite 9D
    values, and the explicit Student-only boundary before a V5-B run consumes
    any bytes.
    """

    root = root.resolve()
    verify_sealed_directory(root)
    manifest_path = root / "OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1":
        raise ValueError("unexpected policy-intent manifest schema")
    if manifest.get("status") != "PASS" or manifest.get("formal_training_authorized") is not False or manifest.get("formal_attack_authorized") is not False:
        raise ValueError("policy-intent root is not a clean-only PASS")
    if manifest.get("policy_feature_order") != list(V5_FEATURES_9D):
        raise ValueError("policy-intent feature order mismatch")
    if manifest.get("policy_step_count") != manifest.get("valid_intent_step_count"):
        raise ValueError("policy-intent valid-step count mismatch")
    for name in ("source_artifact_index_sha256",):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid policy-intent binding: {name}")

    records_path = root / "policy_intent_records.jsonl"
    records = _jsonl(records_path)
    if len(records) != int(manifest.get("policy_step_count", -1)):
        raise ValueError("policy-intent record count does not match manifest")
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("canonical_parent_key", ""))
        if len(key.split("/")) != 3:
            raise ValueError(f"invalid policy-intent identity: {key}")
        if set(record) != ({"canonical_parent_key", "task_language"} | _POLICY_INTENT_REQUIRED_FIELDS):
            raise ValueError(f"policy-intent row schema mismatch: {key}")
        if record.get("valid_intent") is not True:
            raise ValueError(f"invalid policy-intent row: {key}")
        if any(field in record for field in V5_STUDENT_FORBIDDEN_FIELDS):
            raise ValueError(f"policy-intent row contains forbidden field: {key}")
        values = record.get("clean_policy_intent_9d")
        if not isinstance(values, list) or len(values) != 9:
            raise ValueError(f"policy-intent width mismatch: {key}")
        try:
            tensor = torch.tensor(values, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"policy-intent values are not numeric: {key}") from exc
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"policy-intent values are non-finite: {key}")
        index.setdefault(key, []).append(record)
    if len(index) != int(manifest.get("fit_identity_count", -1)) or len(index) != 800:
        raise ValueError("policy-intent root does not cover exactly 800 FIT identities")
    for key, rows in index.items():
        steps = [int(item.get("step", -1)) for item in rows]
        if steps != list(range(len(rows))):
            raise ValueError(f"policy-intent step closure failed: {key}")
    meta = {
        "policy_root_sha256s_sha256": sha256_file(root / "SHA256SUMS"),
        "policy_manifest_sha256": sha256_file(manifest_path),
        "policy_feature_order_sha256": feature_order_sha(V5_FEATURES_9D),
        "policy_source_artifact_index_sha256": manifest["source_artifact_index_sha256"],
        "policy_identity_count": len(index),
        "policy_step_count": len(records),
    }
    return index, meta


def load_v5_episode(
    s1_root: Path,
    teacher_root: Path,
    row: dict[str, Any],
    *,
    policy_index: dict[str, list[dict[str, Any]]] | None = None,
) -> V5Episode:
    identity = str(row["canonical_parent_key"])
    s1 = _episode_root(s1_root, row)
    teacher = _episode_root(teacher_root, row)
    students = _jsonl(s1 / "student_input_records.jsonl")
    physics_teacher = teacher_root / "labels" / str(row["suite"]) / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
    physics_path = physics_teacher / "physics_teacher_v2.jsonl"
    legacy_path = teacher / "v5_teacher_utility.jsonl"
    physics_protocol = _load_physics_protocol(teacher_root) if physics_path.is_file() else None
    teacher_path = physics_path if physics_path.is_file() else legacy_path
    teachers = _jsonl(teacher_path)
    if not students or len(students) != len(teachers):
        raise ValueError(f"V5-A stream mismatch: {identity}")
    policy_rows = None if policy_index is None else policy_index.get(identity)
    if policy_index is not None and (policy_rows is None or len(policy_rows) != len(students)):
        raise ValueError(f"V5-B policy-intent stream mismatch: {identity}")
    features: list[list[float]] = []
    valid: list[bool] = []
    candidate: list[bool] = []
    tiers: list[int] = []
    known: list[bool] = []
    release: list[bool] = []
    regrasp: list[bool] = []
    policy_features: list[list[float]] = []
    policy_valid: list[bool] = []
    window_rows: list[dict[str, Any]] = []
    for index, (student, teacher_row) in enumerate(zip(students, teachers)):
        if physics_protocol is not None:
            teacher_row = _physics_row_as_v5_teacher(teacher_row, physics_protocol)
        validate_student_features(student)
        validate_teacher_row(teacher_row)
        if student.get("canonical_parent_key") != identity or int(student.get("step", -1)) != index:
            raise ValueError(f"V5-A Student identity/step mismatch: {identity}:{index}")
        if teacher_row.get("canonical_parent_key") != identity or int(teacher_row.get("step", -1)) != index:
            raise ValueError(f"V5-A Teacher identity/step mismatch: {identity}:{index}")
        if policy_rows is not None:
            policy_row = policy_rows[index]
            if policy_row.get("canonical_parent_key") != identity or int(policy_row.get("step", -1)) != index:
                raise ValueError(f"V5-B policy identity/step mismatch: {identity}:{index}")
            values = policy_row["clean_policy_intent_9d"]
            policy_features.append([float(value) for value in values])
            policy_valid.append(bool(policy_row["valid_intent"]))
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
        policy_intent_9d=torch.tensor(policy_features, dtype=torch.float32) if policy_rows is not None else torch.empty((0, 9), dtype=torch.float32),
        intent_valid_mask=torch.tensor(policy_valid, dtype=torch.bool) if policy_rows is not None else torch.empty((0,), dtype=torch.bool),
    )


def load_v5_episodes(
    s1_root: Path,
    teacher_root: Path,
    rows: Iterable[dict[str, Any]],
    *,
    policy_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[V5Episode]:
    return [load_v5_episode(s1_root, teacher_root, row, policy_index=policy_index) for row in rows]


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


def compute_v5_intent_normalization(episodes: Iterable[V5Episode]) -> tuple[Tensor, Tensor]:
    values = torch.cat([episode.policy_intent_9d[episode.intent_valid_mask] for episode in episodes], dim=0)
    if values.numel() == 0:
        raise ValueError("cannot normalize an empty V5-B policy-intent training set")
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
    "load_policy_intent_root",
    "load_v5_episode",
    "load_v5_episodes",
    "classify_v5_episode_windows",
    "compute_v5_normalization",
    "compute_v5_intent_normalization",
    "aggregate_window_scores",
    "aggregate_retrospective_window_scores",
    "causal_window_anchor_scores",
]
