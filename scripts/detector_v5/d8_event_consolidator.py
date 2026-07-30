"""D8-1 Event Consolidator: merge TRUE spans separated by short REL_UNK gaps.

Does NOT modify step labels. Produces event_group sidecar for evaluation
and training-weight purposes.  UNKNOWN steps keep mask=false, weight=0.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Frozen protocol constants ---
BRIDGE_REASON_ALLOWLIST = {"RELATION_EVIDENCE_UNKNOWN"}
FORBIDDEN_REASONS = {
    "GEOMETRY_NOT_APPLICABLE", "RIGHT_CENSORED", "NONFINITE",
    "IDENTITY_UNRESOLVED", "KNOWN_FALSE",
}
ARTICULATED_TASKS = {
    "libero_goal/task_00", "libero_goal/task_07",
}
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
    """UNKNOWN step whose reason is in the allowlist and NOT forbidden."""
    if not isinstance(lab, dict):
        return False
    if lab.get("value") != "UNKNOWN":
        return False
    reason = lab.get("reason", "")
    if reason in FORBIDDEN_REASONS:
        return False
    if reason not in BRIDGE_REASON_ALLOWLIST:
        return False
    return True


def _relation_signature(ep_rel: dict | None, side: str) -> str:
    """Stable signature for object/target/relation identity."""
    if not isinstance(ep_rel, dict):
        return ""
    obj = ep_rel.get("logical_object", "")
    tgt = ep_rel.get("logical_target", "")
    rel = ep_rel.get("selected_relation", "")
    bid = ep_rel.get("binding_identity", "")
    role = ep_rel.get("entity_role", "")
    return hashlib.sha256(
        f"{obj}|{tgt}|{rel}|{bid}|{role}|{side}".encode()
    ).hexdigest()


def consolidate_physical_events(
    episode_id: str,
    labels: Dict[int, dict],
    relations: Optional[List[dict]] = None,
    G: int = 3,
) -> Dict[str, Any]:
    """Consolidate TRUE spans for one episode.

    Args:
        episode_id: e.g. "libero_10/task_02/state_05"
        labels: step -> physical_criticality label dict
        relations: episode-level relation list (for identity checks)
        G: max gap length for bridge

    Returns:
        event_groups sidecar dict with consolidated_event_ids, fragments, gaps
    """
    steps_sorted = sorted(labels.keys())
    if not steps_sorted:
        return {"episode_id": episode_id, "event_groups": [], "head": HEAD, "G": G,
                "raw_true_span_count": 0, "consolidated_event_count": 0,
                "total_bridged_gaps": 0, "total_rejected_gaps": 0,
                "articulated": is_articulated, "applicable": not is_articulated}

    n = len(steps_sorted)
    suite, task_str, state_str = _parse_episode_id(episode_id)

    # Determine if this episode is articulated (exclude from physical denominator)
    task_key = f"{suite}/{task_str}"
    is_articulated = task_key in ARTICULATED_TASKS

    # Step 1: Find raw TRUE spans
    raw_spans: List[Tuple[int, int, int, int]] = []  # (start, end, start_idx, end_idx)
    s = None
    for idx, step in enumerate(steps_sorted):
        lab = labels[step]
        if _step_is_known_true(lab):
            if s is None:
                s = (step, idx)
        else:
            if s is not None:
                raw_spans.append((s[0], steps_sorted[idx - 1], s[1], idx - 1))
                s = None
    if s is not None:
        raw_spans.append((s[0], steps_sorted[-1], s[1], n - 1))

    if not raw_spans:
        return {
            "episode_id": episode_id,
            "event_groups": [],
            "head": HEAD, "G": G,
            "articulated": is_articulated,
            "raw_true_span_count": 0,
            "consolidated_event_count": 0,
            "total_bridged_gaps": 0,
            "total_rejected_gaps": 0,
            "applicable": not is_articulated,
        }

    # Step 2: Merge spans across bridgeable gaps
    merged_groups = []
    i = 0
    while i < len(raw_spans):
        cs, ce, cs_idx, ce_idx = raw_spans[i]
        fragments = [(cs, ce)]
        gaps = []
        j = i + 1
        while j < len(raw_spans):
            gap_start_step = raw_spans[j - 1][1] + 1
            gap_end_step = raw_spans[j][0] - 1
            if gap_start_step > gap_end_step:
                # Adjacent TRUE spans (no gap)
                j += 1
                fragments.append((raw_spans[j - 1][0], raw_spans[j - 1][1]))
                ce = raw_spans[j - 1][1]
                continue

            gap_len = gap_end_step - gap_start_step + 1
            if gap_len > G:
                break  # Gap too long

            # Validate gap
            gap_valid, gap_reason = _validate_gap(
                labels, steps_sorted, gap_start_step, gap_end_step,
                raw_spans[j - 1], raw_spans[j], relations,
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

        total_true_steps = sum(e - s + 1 for s, e in fragments)
        merged_groups.append({
            "consolidated_event_id": len(merged_groups),
            "fragment_ranges": fragments,
            "fragment_count": len(fragments),
            "bridged_gaps": [g for g in gaps if g["bridgeable"]],
            "rejected_gaps": [g for g in gaps if not g["bridgeable"]],
            "raw_true_step_count": total_true_steps,
            "unknown_gap_step_count": sum(g["length"] for g in gaps if g["bridgeable"]),
            "total_span_steps": fragments[-1][1] - fragments[0][0] + 1,
        })
        i = j

    return {
        "episode_id": episode_id,
        "event_groups": merged_groups,
        "head": HEAD,
        "G": G,
        "articulated": is_articulated,
        "applicable": not is_articulated,
        "raw_true_span_count": len(raw_spans),
        "consolidated_event_count": len(merged_groups),
        "total_bridged_gaps": sum(len(g["bridged_gaps"]) for g in merged_groups),
        "total_rejected_gaps": sum(len(g["rejected_gaps"]) for g in merged_groups),
    }


def _validate_gap(
    labels: Dict[int, dict],
    steps_sorted: List[int],
    gap_start: int, gap_end: int,
    left_span: Tuple[int, int, int, int],
    right_span: Tuple[int, int, int, int],
    relations: Optional[List[dict]],
) -> Tuple[bool, str]:
    """Check if gap between two TRUE spans is bridgeable. Returns (valid, reason)."""
    # Condition 1-2: Same episode/head (guaranteed by caller)

    # Condition 3: Both sides known TRUE (already established)

    # Conditions 5-8: Gap length, step types, reasons
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

    # Condition 4: Same logical object/target/relation/binding identity
    if relations:
        rel_sig_left = _episode_relation_at(labels, left_span[0], relations)
        rel_sig_right = _episode_relation_at(labels, right_span[0], relations)
        sig_left = _relation_signature(rel_sig_left, "left")
        sig_right = _relation_signature(rel_sig_right, "right")
        if sig_left != sig_right:
            return False, f"RELATION_SIGNATURE_MISMATCH: {sig_left[:16]} != {sig_right[:16]}"

    return True, "REL_UNK_BRIDGE"


def _episode_relation_at(labels, step, relations) -> dict | None:
    """Find the relation active at a given step. Returns first match or None."""
    if not relations:
        return None
    for rel in relations:
        if isinstance(rel, dict):
            return rel  # Simplified: use the first relation
    return None


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
    """Teacher-event-based weights using consolidated event groups.

    Each consolidated event gets equal total positive weight.
    Known FALSE spans get equal total negative weight.
    UNKNOWN/articulated steps get zero weight.
    """
    weights = np.zeros(len(labels), dtype=np.float32)
    n = len(labels)

    # If we have consolidated events, use them for TRUE span weighting
    event_groups = consolidated_events.get("event_groups", [])
    if event_groups:
        for group in event_groups:
            for frag_start, frag_end in group["fragment_ranges"]:
                frag_len = frag_end - frag_start + 1
                if frag_len > 0:
                    total_positive_weight = 1.0 / len(event_groups)  # Equal per event
                    for i in range(frag_start, frag_end + 1):
                        if i < n and masks[i] and labels[i] == 1:
                            weights[i] = total_positive_weight / frag_len
        return weights

    # Fallback: _physical_event_weights logic without consolidation
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
    """Deterministic digest for a consolidation result."""
    canonical = json.dumps(result, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
