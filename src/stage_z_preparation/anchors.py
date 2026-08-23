"""Outcome-blind, deterministic clean-trajectory anchor selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contract import StageZHold


FORBIDDEN_OUTCOME_FIELDS = frozenset(
    {
        "v_phys",
        "command_open",
        "task_success",
        "contact_loss",
        "object_displacement",
        "manual_label",
        "video_label",
        "intervention_outcome",
        "student_emit",
        "detector_score",
        "detector_emit",
    }
)


@dataclass(frozen=True)
class AnchorCandidate:
    parent_key: str
    model_id: str
    step: int
    anchor_class: str
    legal_branch: bool = True
    fresh_boundary: bool = True
    horizon_legal: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_outcome_blind(self) -> None:
        forbidden = sorted(FORBIDDEN_OUTCOME_FIELDS.intersection(self.metadata))
        if forbidden:
            raise StageZHold(f"ANCHOR_OUTCOME_LEAKAGE:{','.join(forbidden)}")
        if any(key.lower().startswith("student") or key.lower().startswith("detector") for key in self.metadata):
            raise StageZHold("ANCHOR_STUDENT_DETECTOR_LEAKAGE")
        if self.anchor_class not in {"CRITICAL", "NONCRITICAL"}:
            raise StageZHold("UNKNOWN_ANCHOR_CLASS")
        if int(self.step) < 0:
            raise StageZHold("NEGATIVE_ANCHOR_STEP")


@dataclass(frozen=True)
class AnchorSelection:
    status: str
    anchor_class: str
    parent_key: str
    model_id: str
    selected: AnchorCandidate | None = None
    rank_digest: str | None = None


def select_anchor(
    candidates: Sequence[AnchorCandidate],
    *,
    salt: str,
    model_id: str,
    parent_key: str,
    anchor_class: str,
) -> AnchorSelection:
    """Select exactly one eligible clean anchor, or abstain without replacement."""

    if not salt or not model_id or not parent_key:
        raise StageZHold("ANCHOR_SELECTION_BINDING_MISSING")
    if anchor_class not in {"CRITICAL", "NONCRITICAL"}:
        raise StageZHold("UNKNOWN_ANCHOR_CLASS")
    eligible: list[tuple[str, AnchorCandidate]] = []
    for candidate in candidates:
        candidate.validate_outcome_blind()
        if (
            candidate.model_id == model_id
            and candidate.parent_key == parent_key
            and candidate.anchor_class == anchor_class
            and candidate.legal_branch
            and candidate.fresh_boundary
            and candidate.horizon_legal
        ):
            digest = hashlib.sha256(f"{salt}|{model_id}|{parent_key}|{candidate.step}".encode()).hexdigest()
            eligible.append((digest, candidate))
    if not eligible:
        return AnchorSelection(
            status=f"NO_{anchor_class}_ANCHOR",
            anchor_class=anchor_class,
            parent_key=parent_key,
            model_id=model_id,
        )
    digest, selected = min(eligible, key=lambda item: (item[0], item[1].step))
    return AnchorSelection(
        status=f"SELECTED_{anchor_class}_ANCHOR",
        anchor_class=anchor_class,
        parent_key=parent_key,
        model_id=model_id,
        selected=selected,
        rank_digest=digest,
    )


__all__ = ["AnchorCandidate", "AnchorSelection", "FORBIDDEN_OUTCOME_FIELDS", "select_anchor"]
