"""Frozen static schema for C2g counterfactual replay manifests."""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .c2g_teacher_v2_schema import CANDIDATE_STRATA, TEACHER_SCHEMA_VERSION


COUNTERFACTUAL_MANIFEST_VERSION = "c2g.counterfactual_manifest.2026-07-10.v1"
COMPARISON_TIERS = ("TIER_A_MATCHED_ACTION_SHORT_HORIZON", "TIER_B_CLOSED_LOOP_CONTINUATION")
UNKNOWN_REASONS = frozenset({
    "RESTORE_MISMATCH",
    "ACTION_ALIGNMENT_FAILED",
    "AMBIGUOUS_EFFECT",
    "NOT_REPLAYED",
    "TARGET_GROUNDING_FAILED",
    "SNAPSHOT_INCOMPLETE",
    "INCOMPLETE_ATTACK_DELIVERY",
})
REQUIRED_SNAPSHOT_FIELDS = frozenset({"qpos", "qvel", "mocap_pos", "mocap_quat"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_FIELDS = (
    "manifest_version",
    "comparison_tier",
    "run_id",
    "episode_key",
    "suite",
    "task_index",
    "state_id",
    "step",
    "candidate_stratum",
    "candidate_reason",
    "snapshot_hash",
    "snapshot_fields_present",
    "restore_state_hash",
    "restore_parity_pass",
    "restore_parity_metrics",
    "clean_action_source",
    "matched_action_alignment_pass",
    "short_horizon",
    "closed_loop_continuation_enabled",
    "attack_horizon",
    "delivered_attack_steps",
    "force_open_raw_command",
    "force_open_env_command",
    "clean_continuation_hash",
    "attack_continuation_hash",
    "label_known_mask",
    "unknown_reason",
    "effect_thresholds",
    "progress_metric_version",
    "teacher_schema_version",
    "code_commit",
    "git_clean",
    "simulator_version",
    "libero_version",
    "policy_model_manifest_sha256",
    "processor_manifest_sha256",
    "random_seed",
    "created_at",
)


def _require_fields(row: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError("missing required manifest fields: " + ", ".join(missing))


def _sha(value: Any, field: str, *, allow_empty: bool = False) -> None:
    text = str(value or "")
    if allow_empty and not text:
        return
    if not SHA256_RE.fullmatch(text):
        raise ValueError(f"{field} must be a full SHA256")


def validate_counterfactual_manifest(row: Mapping[str, Any]) -> None:
    """Validate provenance, restore/action parity, and unknown-safe replay status."""
    _require_fields(row, REQUIRED_FIELDS)
    if row["manifest_version"] != COUNTERFACTUAL_MANIFEST_VERSION:
        raise ValueError("manifest_version mismatch")
    tier = str(row["comparison_tier"])
    if tier not in COMPARISON_TIERS:
        raise ValueError("unknown comparison_tier")
    if row["candidate_stratum"] not in CANDIDATE_STRATA:
        raise ValueError("unknown candidate_stratum")
    if row["teacher_schema_version"] != TEACHER_SCHEMA_VERSION:
        raise ValueError("teacher_schema_version mismatch")
    if not COMMIT_RE.fullmatch(str(row["code_commit"])) or row["git_clean"] is not True:
        raise ValueError("counterfactual provenance requires clean full git commit")
    for field in ("snapshot_hash", "restore_state_hash", "policy_model_manifest_sha256", "processor_manifest_sha256"):
        _sha(row[field], field)
    known = row["label_known_mask"] in (1, True)
    if row["label_known_mask"] not in (0, 1, False, True):
        raise ValueError("label_known_mask must be binary")
    for field in ("clean_continuation_hash", "attack_continuation_hash"):
        _sha(row[field], field, allow_empty=not known)
    snapshot_fields = set(str(value) for value in row["snapshot_fields_present"])
    snapshot_complete = REQUIRED_SNAPSHOT_FIELDS <= snapshot_fields
    if not isinstance(row["restore_parity_metrics"], Mapping) or not isinstance(row["effect_thresholds"], Mapping):
        raise ValueError("restore_parity_metrics and effect_thresholds must be mappings")
    if not str(row["clean_action_source"]).strip():
        raise ValueError("clean_action_source is required")
    if type(row["restore_parity_pass"]) is not bool or type(row["matched_action_alignment_pass"]) is not bool:
        raise ValueError("restore/action parity fields must be booleans")
    if type(row["closed_loop_continuation_enabled"]) is not bool:
        raise ValueError("closed_loop_continuation_enabled must be boolean")
    if tier == "TIER_A_MATCHED_ACTION_SHORT_HORIZON" and row["closed_loop_continuation_enabled"]:
        raise ValueError("Tier A cannot enable closed-loop continuation")
    if tier == "TIER_B_CLOSED_LOOP_CONTINUATION" and not row["closed_loop_continuation_enabled"]:
        raise ValueError("Tier B requires closed-loop continuation")
    horizon = int(row["attack_horizon"])
    delivered = int(row["delivered_attack_steps"])
    if horizon <= 0 or int(row["short_horizon"]) <= 0 or not 0 <= delivered <= horizon:
        raise ValueError("invalid replay or attack horizon")
    for field in ("force_open_raw_command", "force_open_env_command"):
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if type(row["random_seed"]) is not int or row["random_seed"] < 0:
        raise ValueError("random_seed must be a non-negative integer")
    for field in ("run_id", "episode_key", "suite", "candidate_reason", "progress_metric_version", "simulator_version", "libero_version", "created_at"):
        if not str(row[field]).strip():
            raise ValueError(f"{field} is required")

    failures = []
    if not snapshot_complete:
        failures.append("SNAPSHOT_INCOMPLETE")
    if not row["restore_parity_pass"]:
        failures.append("RESTORE_MISMATCH")
    if not row["matched_action_alignment_pass"]:
        failures.append("ACTION_ALIGNMENT_FAILED")
    if delivered != horizon:
        failures.append("INCOMPLETE_ATTACK_DELIVERY")
    unknown_reason = str(row["unknown_reason"] or "")
    if known:
        if failures:
            raise ValueError("known label has invalid replay evidence: " + ", ".join(failures))
        if unknown_reason:
            raise ValueError("known label cannot carry unknown_reason")
    else:
        if not unknown_reason or unknown_reason not in UNKNOWN_REASONS:
            raise ValueError("unknown replay requires an explicit supported unknown_reason")
        if failures and unknown_reason not in failures:
            raise ValueError("unknown_reason must identify the replay evidence failure")
