"""D8-1 Event Consolidator: merge TRUE spans separated by short REL_UNK gaps.

Does NOT modify step labels. Produces event_group sidecar.
UNKNOWN steps keep mask=false, weight=0.

Formal mode (default): relations MUST be provided for identity checks.
Diagnostic mode (--diagnostic-unbound-relations): relations=None allowed,
output marked consumer_eligible=false.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Frozen protocol constants ---
BRIDGE_REASON_ALLOWLIST = {"RELATION_EVIDENCE_UNKNOWN"}
FORBIDDEN_REASONS = {
    "GEOMETRY_NOT_APPLICABLE", "RIGHT_CENSORED", "NONFINITE",
    "IDENTITY_UNRESOLVED", "KNOWN_FALSE",
}
ARTICULATED_TASKS = {"libero_goal/task_00", "libero_goal/task_07"}
HEAD = "physical_criticality"


def _step_is_known_true(lab: dict | None) -> bool:
    if not isinstance(lab, dict):
        return False
    return bool(lab.get("value") == "TRUE" and lab.get("mask") and lab.get("valid_mask"))


def _step_is_known_false(lab: dict | None) -> bool:
    if not isinstance(lab, dict):
        return False
    return bool(lab.get("value") == "FALSE" and lab.get("mask") and lab.get("valid_mask"))


def _step_is_unk_allowed(lab: dict | None) -> bool:
    if not isinstance(lab, dict):
        return False
    if lab.get("value") != "UNKNOWN":
        return False
    reason = lab.get("reason", "")
    if reason in FORBIDDEN_REASONS:
        return False
    return reason in BRIDGE_REASON_ALLOWLIST


def _relation_signature(ep_rel: dict | None) -> str:
    """Stable signature. Returns empty string if any field is missing/empty (fail-closed)."""
    if not isinstance(ep_rel, dict):
        return ""
    fields = [
        "logical_object", "logical_target", "selected_relation",
        "binding_identity", "entity_role", "entity_type",
        "object_entity_id", "target_entity_id",
    ]
    values = []
    for f in fields:
        v = ep_rel.get(f, "")
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return ""  # fail-closed: missing field
        values.append(str(v))
    return hashlib.sha256("|".join(values).encode()).hexdigest()


def _validate_steps(labels: Dict[int, dict]) -> Optional[str]:
    """Validate step integrity. Returns error message or None."""
    if not labels:
        return None
    steps = sorted(labels.keys())
    prev = steps[0] - 1
    seen = set()
    for s in steps:
        if not isinstance(s, int) or isinstance(s, bool):
            return f"non-integer step: {s}"
        if s in seen:
            return f"duplicate step: {s}"
        if s < 0:
            return f"negative step: {s}"
        if not np.isfinite(float(s)):
            return f"non-finite step: {s}"
        if s != prev + 1 and prev >= 0:
            return f"missing step: expected {prev+1}, got {s}"
        seen.add(s)
        prev = s
    return None


def consolidate_physical_events(
    episode_id: str,
    labels: Dict[int, dict],
    relations: Optional[List[dict]] = None,
    G: int = 3,
    diagnostic_unbound_relations: bool = False,
) -> Dict[str, Any]:
    """Consolidate TRUE spans for one episode.

    Args:
        episode_id: e.g. "libero_10/task_02/state_05"
        labels: step -> physical_criticality label dict (contiguous, zero-based)
        relations: per-step or episode-level relation records (REQUIRED in formal mode)
        G: max gap length for bridge
        diagnostic_unbound_relations: if True, skip identity checks (DIAGNOSTIC ONLY)

    Returns:
        event_groups sidecar dict
    """
    # Validate step integrity
    step_err = _validate_steps(labels)
    if step_err:
        raise ValueError(f"step integrity violation in {episode_id}: {step_err}")

    if relations is None and not diagnostic_unbound_relations:
        raise ValueError(
            f"relations required for formal consolidation in {episode_id}; "
            f"use diagnostic_unbound_relations=True for diagnostic mode"
        )

    suite, task_str, state_str = _parse_episode_id(episode_id)
    task_key = f"{suite}/{task_str}"
    is_articulated = task_key in ARTICULATED_TASKS

    steps_sorted = sorted(labels.keys())
    if not steps_sorted:
        return _empty_result(episode_id, G, is_articulated, diagnostic_unbound_relations)

    n = len(steps_sorted)

    # Find raw TRUE spans
    raw_spans: List[Tuple[int, int, int, int]] = []
    s_start = None
    for idx, step in enumerate(steps_sorted):
        lab = labels[step]
        if _step_is_known_true(lab):
            if s_start is None:
                s_start = (step, idx)
        else:
            if s_start is not None:
                raw_spans.append((s_start[0], steps_sorted[idx - 1], s_start[1], idx - 1))
                s_start = None
    if s_start is not None:
        raw_spans.append((s_start[0], steps_sorted[-1], s_start[1], n - 1))

    if not raw_spans:
        return _empty_result(episode_id, G, is_articulated, diagnostic_unbound_relations)

    # Merge spans across bridgeable gaps
    merged_groups = []
    i = 0
    while i < len(raw_spans):
        cs, ce_val, cs_idx, ce_idx = raw_spans[i]
        fragments = [(cs, ce_val)]
        gaps = []
        j = i + 1
        while j < len(raw_spans):
            gap_start_step = raw_spans[j - 1][1] + 1
            gap_end_step = raw_spans[j][0] - 1
            if gap_start_step > gap_end_step:
                # Adjacent spans
                j += 1
                fragments.append((raw_spans[j - 1][0], raw_spans[j - 1][1]))
                continue

            gap_len = gap_end_step - gap_start_step + 1
            if gap_len > G:
                break

            gap_valid, gap_reason = _validate_gap(
                labels, steps_sorted, gap_start_step, gap_end_step,
                raw_spans[j - 1], raw_spans[j], relations, diagnostic_unbound_relations,
            )
            if not gap_valid:
                gaps.append({
                    "start": gap_start_step, "end": gap_end_step,
                    "length": gap_len, "bridgeable": False,
                    "reject_reason": gap_reason,
                })
                break

            gaps.append({
                "start": gap_start_step, "end": gap_end_step,
                "length": gap_len, "bridgeable": True,
                "reason": gap_reason,
            })
            fragments.append((raw_spans[j][0], raw_spans[j][1]))
            j += 1

        merged_groups.append({
            "consolidated_event_id": len(merged_groups),
            "fragment_ranges": fragments,
            "fragment_count": len(fragments),
            "bridged_gaps": [g for g in gaps if g["bridgeable"]],
            "rejected_gaps": [g for g in gaps if not g["bridgeable"]],
            "raw_true_step_count": sum(e - s + 1 for s, e in fragments),
            "unknown_gap_step_count": sum(g["length"] for g in gaps if g["bridgeable"]),
            "total_span_steps": fragments[-1][1] - fragments[0][0] + 1,
        })
        i = j

    return {
        "episode_id": episode_id,
        "event_groups": merged_groups,
        "head": HEAD, "G": G,
        "articulated": is_articulated,
        "applicable": not is_articulated,
        "consumer_eligible": not is_articulated and not diagnostic_unbound_relations,
        "diagnostic_unbound_relations": diagnostic_unbound_relations,
        "identity_checks_performed": relations is not None,
        "raw_true_span_count": len(raw_spans),
        "consolidated_event_count": len(merged_groups),
        "total_bridged_gaps": sum(len(g["bridged_gaps"]) for g in merged_groups),
        "total_rejected_gaps": sum(len(g["rejected_gaps"]) for g in merged_groups),
    }


def _empty_result(episode_id, G, is_articulated, diagnostic):
    return {
        "episode_id": episode_id, "event_groups": [], "head": HEAD, "G": G,
        "raw_true_span_count": 0, "consolidated_event_count": 0,
        "total_bridged_gaps": 0, "total_rejected_gaps": 0,
        "articulated": is_articulated, "applicable": not is_articulated,
        "consumer_eligible": not is_articulated and not diagnostic,
        "diagnostic_unbound_relations": diagnostic,
        "identity_checks_performed": not diagnostic,
    }


def _validate_gap(
    labels, steps_sorted, gap_start, gap_end,
    left_span, right_span, relations, diagnostic,
) -> Tuple[bool, str]:
    gap_steps = [s for s in steps_sorted if gap_start <= s <= gap_end]
    for step in gap_steps:
        lab = labels.get(step)
        if lab is None:
            return False, "MISSING_LABEL"
        if _step_is_known_false(lab):
            return False, "KNOWN_FALSE_IN_GAP"
        if lab.get("reason") == "GEOMETRY_NOT_APPLICABLE":
            return False, "GEOMETRY_NOT_APPLICABLE_IN_GAP"
        if lab.get("reason") == "RIGHT_CENSORED" or lab.get("right_censored"):
            return False, "RIGHT_CENSORED_IN_GAP"
        if not _step_is_unk_allowed(lab):
            return False, f"GAP_STEP_NOT_ALLOWED: reason={lab.get('reason')} value={lab.get('value')}"

    # Identity checks (formal mode only)
    if relations is not None:
        rel_left = _episode_relation_at(labels, left_span[0], relations)
        rel_right = _episode_relation_at(labels, right_span[0], relations)
        sig_left = _relation_signature(rel_left)
        sig_right = _relation_signature(rel_right)
        if not sig_left or not sig_right:
            return False, "RELATION_SIGNATURE_EMPTY"
        if sig_left != sig_right:
            return False, "RELATION_SIGNATURE_MISMATCH"
        # Verify gap candidate ledger contains same relation
        for step in gap_steps:
            rel_gap = _episode_relation_at(labels, step, relations)
            if rel_gap and _relation_signature(rel_gap) not in (sig_left, ""):
                return False, "RELATION_CANDIDATE_CONFLICT_IN_GAP"
    elif not diagnostic:
        return False, "IDENTITY_CHECKS_SKIPPED_FORMAL_MODE"

    return True, "REL_UNK_BRIDGE"


def _episode_relation_at(labels, step, relations) -> dict | None:
    """Find relation record active at a given step.

    Supports two formats:
    1. Per-step dict: relations[step] = sidecar entry with per_relation list
    2. List of dicts with 'step' field (legacy)
    """
    if not relations:
        return None
    # Format 1: dict keyed by step
    if isinstance(relations, dict):
        entry = relations.get(step)
        if not isinstance(entry, dict):
            return None
        # Get the selected relation from the sidecar entry
        per_rel = entry.get("per_relation", [])
        sel_idx = entry.get("selected_relation_index")
        if sel_idx is not None and sel_idx < len(per_rel):
            return per_rel[sel_idx]
        return None
    # Format 2: list of dicts with step field (legacy)
    has_steps = any(isinstance(r, dict) and "step" in r for r in relations)
    if has_steps:
        for r in relations:
            if isinstance(r, dict) and r.get("step") == step:
                return r
        return None
    raise ValueError("relation records lack per-step resolution")


def _parse_episode_id(eid: str) -> Tuple[str, str, str]:
    parts = eid.split("/")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return eid, "", ""


def build_physical_event_weights(
    labels: np.ndarray,
    masks: np.ndarray,
    consolidated_events: Dict[str, Any],
) -> np.ndarray:
    """Teacher-event-based weights.

    Each consolidated TRUE event gets equal total positive weight (shared across
    all fragments). Known FALSE spans get equal total negative weight.
    UNKNOWN/articulated steps get zero weight.

    Precondition: labels and masks are contiguous zero-based arrays where
    index i corresponds to step i.
    """
    n = len(labels)
    weights = np.zeros(n, dtype=np.float32)
    event_groups = consolidated_events.get("event_groups", [])

    if not event_groups:
        # Fallback: raw contiguous TRUE/FALSE spans
        return _fallback_weights(labels, masks)

    num_events = len(event_groups)
    pos_weight_per_event = 1.0 / max(num_events, 1)

    # Positive weights: each consolidated event gets equal share,
    # distributed evenly across ALL true steps in that event
    event_true_steps = set()
    for group in event_groups:
        group_steps = []
        for frag_start, frag_end in group["fragment_ranges"]:
            for i in range(frag_start, frag_end + 1):
                if 0 <= i < n and masks[i] and labels[i] == 1:
                    group_steps.append(i)
        if group_steps:
            per_step = pos_weight_per_event / len(group_steps)
            for i in group_steps:
                weights[i] = per_step
                event_true_steps.add(i)

    # Negative weights: contiguous known FALSE spans
    i = 0
    neg_spans = []
    while i < n:
        if masks[i] and labels[i] == 0:
            j = i + 1
            while j < n and masks[j] and labels[j] == 0:
                j += 1
            neg_spans.append((i, j))
            i = j
        else:
            i += 1

    if neg_spans:
        neg_weight_per_span = 1.0 / len(neg_spans)
        for s, e in neg_spans:
            span_len = e - s
            if span_len > 0:
                for i in range(s, e):
                    weights[i] = neg_weight_per_span / span_len

    return weights


def _fallback_weights(labels, masks):
    """Weights without consolidated events: equal per contiguous span."""
    n = len(labels)
    weights = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        if masks[i] and labels[i] == 1:
            j = i + 1
            while j < n and masks[j] and labels[j] == 1:
                j += 1
            span_len = j - i
            if span_len > 0:
                weights[i:j] = 1.0 / float(span_len)
            i = j
        elif masks[i] and labels[i] == 0:
            j = i + 1
            while j < n and masks[j] and labels[j] == 0:
                j += 1
            span_len = j - i
            if span_len > 0:
                weights[i:j] = 1.0 / float(span_len)
            i = j
        else:
            i += 1
    return weights


def compute_consolidation_digest(result: Dict[str, Any]) -> str:
    canonical = json.dumps(result, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
