"""Frozen WindowProposal contract for Layer1/2 → Layer3 handoff.

A WindowProposal is the ONLY interface through which Layer3 receives window
specifications. It must be generated without access to attack outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class WindowProposal:
    """Immutable window proposal for Layer3 evaluation.

    All fields are frozen at creation. The proposal records its own provenance
    so Layer3 can verify it was generated clean-only.
    """

    # ── Identity ──
    proposal_id: str
    selector_version: str
    source_commit: str
    source_trace_path: str
    source_trace_sha256: str = ""

    # ── Task context ──
    suite: str = "libero_object"
    task_key: str = ""
    task_id: str = ""
    state_id: int = 0
    eval_seed: int = 0
    instruction: str = ""

    # ── Window specification ──
    window_start: int = 0
    window_end: int = 10
    anchor_step: int = 5
    predicted_first_close_step: int = -1
    first_close_horizon: int = 0

    # ── Selector confidence ──
    phase_label: str = ""
    phase_confidence: float = 0.0
    closure_criticality: float = 0.0
    release_safe_score: float = 0.0
    selector_score: float = 0.0

    # ── Selection mode ──
    selection_mode: str = "offline_clean_repeat"  # "offline_clean_repeat" or "online_streaming"
    is_online: bool = False

    # ── Mechanism ──
    mechanism_type: str = ""
    mechanism_eligible: bool = False
    eligible: bool = False
    abstain_reason: str = ""

    # ── Prediction mode ──
    prediction_mode: str = ""  # "observed_close_interception" or "future_close_forecast"

    # ── Provenance invariants ──
    uses_clean_only: bool = True
    uses_attack_outcome: bool = False
    uses_random_outcome: bool = False
    uses_privileged_state: bool = False
    features_are_causal: bool = True    # per-step feature extraction is causal
    selection_is_causal: bool = False   # window-selection process is online-causal
    history_length: int = 16

    # ── Config provenance ──
    selector_config_sha256: str = ""
    feature_schema_version: str = "l12_v1"

    # ── Teacher-only marker ──
    selector_role: str = "student"

    def validate(self) -> list[str]:
        """Return list of contract violations (empty = valid)."""
        issues = []
        if not self.proposal_id:
            issues.append("proposal_id:EMPTY")
        if not self.uses_clean_only:
            issues.append("uses_clean_only:FALSE")
        if self.uses_attack_outcome:
            issues.append("uses_attack_outcome:TRUE")
        if self.uses_random_outcome:
            issues.append("uses_random_outcome:TRUE")
        if self.selector_role == "student" and self.uses_privileged_state:
            issues.append("student_uses_privileged_state")
        if self.selector_role == "student" and not self.features_are_causal:
            issues.append("student_features_not_causal")
        # Offline mode: features_are_causal=True, selection_is_causal=False (OK)
        # Online mode: selection_is_causal must be True
        if self.is_online and not self.selection_is_causal:
            issues.append("online_mode_requires_selection_is_causal")
        if self.window_start < 0:
            issues.append("window_start:NEGATIVE")
        if self.window_end <= self.window_start:
            issues.append("window_end:NOT_GT_START")
        if not self.source_commit:
            issues.append("source_commit:EMPTY")
        return issues

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "selector_version": self.selector_version,
            "source_commit": self.source_commit,
            "source_trace_path": self.source_trace_path,
            "source_trace_sha256": self.source_trace_sha256,
            "suite": self.suite,
            "task_key": self.task_key,
            "task_id": self.task_id,
            "state_id": self.state_id,
            "eval_seed": self.eval_seed,
            "instruction": self.instruction,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "anchor_step": self.anchor_step,
            "predicted_first_close_step": self.predicted_first_close_step,
            "first_close_horizon": self.first_close_horizon,
            "phase_label": self.phase_label,
            "phase_confidence": self.phase_confidence,
            "closure_criticality": self.closure_criticality,
            "release_safe_score": self.release_safe_score,
            "selector_score": self.selector_score,
            "selection_mode": self.selection_mode,
            "is_online": self.is_online,
            "mechanism_type": self.mechanism_type,
            "mechanism_eligible": self.mechanism_eligible,
            "eligible": self.eligible,
            "abstain_reason": self.abstain_reason,
            "uses_clean_only": self.uses_clean_only,
            "uses_attack_outcome": self.uses_attack_outcome,
            "uses_random_outcome": self.uses_random_outcome,
            "uses_privileged_state": self.uses_privileged_state,
            "features_are_causal": self.features_are_causal,
            "selection_is_causal": self.selection_is_causal,
            "prediction_mode": self.prediction_mode,
            "history_length": self.history_length,
            "selector_config_sha256": self.selector_config_sha256,
            "feature_schema_version": self.feature_schema_version,
            "selector_role": self.selector_role,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WindowProposal":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def validate_proposals(proposals: list[WindowProposal]) -> tuple[list[str], bool]:
    """Validate a list of proposals. Returns (issues, all_valid)."""
    all_issues = []
    seen_ids = set()
    for p in proposals:
        if p.proposal_id in seen_ids:
            all_issues.append(f"DUPLICATE_ID:{p.proposal_id}")
        seen_ids.add(p.proposal_id)
        all_issues.extend(p.validate())
    return all_issues, len(all_issues) == 0


def compute_source_trace_sha256(trace_path: str) -> str:
    """Compute SHA256 of a trace file for provenance."""
    if not os.path.exists(trace_path):
        return "MISSING"
    h = hashlib.sha256()
    with open(trace_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
