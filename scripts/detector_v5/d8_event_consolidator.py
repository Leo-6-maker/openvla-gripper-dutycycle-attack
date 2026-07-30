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


# --- Canonical relation binding digest ---

_CANONICAL_RELATION_FIELDS = [
    "logical_object", "logical_target", "selected_relation",
    "object_entity_id", "target_entity_id",
    "entity_role", "entity_type",
]


def canonical_relation_binding_digest(rel: dict | None) -> str:
    """Canonical digest from all available relation identity fields.

    Only entity_type is genuinely optional (never populated in current
    Teacher data). All other fields must be present and non-empty.
    Returns empty string on any critical field violation.
    """
    if not isinstance(rel, dict):
        return ""
    # entity_type is the only field known to be absent from Teacher identity data
    critical = [
        "logical_object", "logical_target", "selected_relation",
        "object_entity_id", "target_entity_id", "entity_role",
    ]
    values = []
    for f in critical:
        v = rel.get(f)
        if v is None:
            return ""
        s = str(v).strip()
        if not s:
            return ""
        values.append(s)
    # entity_type: optional, may be empty
    et = rel.get("entity_type")
    if et is None:
        et = ""
    values.append(str(et).strip())
    return hashlib.sha256("|".join(values).encode()).hexdigest()


# Backward-compatible alias used by tests
_relation_signature = canonical_relation_binding_digest


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
        relations: per-step dict {step: sidecar_entry} (REQUIRED in formal mode)
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
                j += 1
                fragments.append((raw_spans[j - 1][0], raw_spans[j - 1][1]))
                continue

            gap_len = gap_end_step - gap_start_step + 1
            if gap_len > G:
                gaps.append({
                    "start": gap_start_step, "end": gap_end_step,
                    "length": gap_len, "bridgeable": False,
                    "reject_reason": "GAP_EXCEEDS_G",
                })
                break

            gap_valid, gap_evidence = _validate_gap(
                labels, steps_sorted, gap_start_step, gap_end_step,
                raw_spans[j - 1], raw_spans[j], relations, diagnostic_unbound_relations,
            )
            if not gap_valid:
                gap_entry = {
                    "start": gap_start_step, "end": gap_end_step,
                    "length": gap_len, "bridgeable": False,
                    "reject_reason": gap_evidence.get("reject_reason", "OTHER"),
                    "step_evidence": gap_evidence.get("step_evidence", []),
                }
                gaps.append(gap_entry)
                break

            gap_entry = {
                "start": gap_start_step, "end": gap_end_step,
                "length": gap_len, "bridgeable": True,
                "reason": "REL_UNK_BRIDGE",
                "step_evidence": gap_evidence.get("step_evidence", []),
            }
            gaps.append(gap_entry)
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


def _build_step_evidence(labels, step, relations) -> dict:
    """Build per-step evidence record for gap step."""
    lab = labels.get(step, {})
    evidence = {
        "step": step,
        "label_value": lab.get("value", "?"),
        "label_reason": lab.get("reason", "?"),
        "right_censored": bool(lab.get("right_censored", False)),
        "mask": bool(lab.get("mask", False)),
        "valid_mask": bool(lab.get("valid_mask", False)),
    }
    # Include candidate ledger info from sidecar
    entry = _get_sidecar_entry(relations, step)
    if entry is not None:
        evidence["selection_status"] = entry.get("selection_status", "?")
        evidence["candidate_relation_indices"] = entry.get("candidate_relation_indices", [])
        evidence["selected_relation_id"] = entry.get("selected_relation_id")
        # Per-relation verdicts
        per_rel = entry.get("per_relation", [])
        evidence["per_relation_verdicts"] = [
            {"relation_index": r.get("relation_index"), "verdict": r.get("verdict")}
            for r in per_rel
        ]
    return evidence


def _validate_gap(
    labels, steps_sorted, gap_start, gap_end,
    left_span, right_span, relations, diagnostic,
) -> Tuple[bool, dict]:
    """Validate a gap for bridging.

    Returns (valid, evidence_dict). evidence_dict contains reject_reason
    and per-step evidence when rejected.
    """
    gap_steps = [s for s in steps_sorted if gap_start <= s <= gap_end]
    gap_evidence = []

    # Phase 1: Basic label checks on gap steps
    for step in gap_steps:
        lab = labels.get(step)
        ev = _build_step_evidence(labels, step, relations)
        if lab is None:
            ev["gap_check"] = "MISSING_LABEL"
            gap_evidence.append(ev)
            return False, {"reject_reason": "MISSING_LABEL", "step_evidence": gap_evidence}
        if _step_is_known_false(lab):
            ev["gap_check"] = "KNOWN_FALSE"
            gap_evidence.append(ev)
            return False, {"reject_reason": "KNOWN_FALSE_IN_GAP", "step_evidence": gap_evidence}
        if lab.get("reason") == "GEOMETRY_NOT_APPLICABLE":
            ev["gap_check"] = "GEOMETRY_NOT_APPLICABLE"
            gap_evidence.append(ev)
            return False, {"reject_reason": "GEOMETRY_NOT_APPLICABLE_IN_GAP", "step_evidence": gap_evidence}
        if lab.get("reason") == "RIGHT_CENSORED" or lab.get("right_censored"):
            ev["gap_check"] = "RIGHT_CENSORED"
            gap_evidence.append(ev)
            return False, {"reject_reason": "RIGHT_CENSORED_IN_GAP", "step_evidence": gap_evidence}
        if not _step_is_unk_allowed(lab):
            ev["gap_check"] = "NOT_ALLOWED"
            gap_evidence.append(ev)
            return False, {
                "reject_reason": f"GAP_STEP_NOT_ALLOWED: reason={lab.get('reason')} value={lab.get('value')}",
                "step_evidence": gap_evidence,
            }
        gap_evidence.append(ev)

    # Phase 2: Identity checks (formal mode only) — real candidate ledger verification
    if relations is not None:
        # Get boundary relation IDs (not list positions — P0-2 fix)
        left_rel_id = _boundary_relation_id(relations, left_span[0])
        right_rel_id = _boundary_relation_id(relations, right_span[0])

        if left_rel_id is None or right_rel_id is None:
            return False, {"reject_reason": "BOUNDARY_RELATION_UNRESOLVED", "step_evidence": gap_evidence}

        if left_rel_id != right_rel_id:
            return False, {"reject_reason": "BOUNDARY_RELATION_ID_MISMATCH", "step_evidence": gap_evidence}

        boundary_rel_id = left_rel_id

        # Get relation signatures for the boundary relation
        rel_left = _find_relation_by_id(relations, left_span[0], boundary_rel_id)
        rel_right = _find_relation_by_id(relations, right_span[0], boundary_rel_id)

        sig_left = canonical_relation_binding_digest(rel_left)
        sig_right = canonical_relation_binding_digest(rel_right)
        if not sig_left or not sig_right:
            return False, {"reject_reason": "RELATION_SIGNATURE_EMPTY", "step_evidence": gap_evidence}
        if sig_left != sig_right:
            return False, {"reject_reason": "RELATION_SIGNATURE_MISMATCH", "step_evidence": gap_evidence}

        # P0-1 fix: Real candidate ledger verification
        # The boundary relation_id must appear in candidate_relation_indices of
        # every gap step. Also verify no competing TRUE relation exists.
        for step in gap_steps:
            entry = _get_sidecar_entry(relations, step)
            if entry is None:
                return False, {
                    "reject_reason": "GAP_CANDIDATE_MISSING_ENTRY",
                    "step_evidence": gap_evidence,
                }

            candidate_ids = entry.get("candidate_relation_indices", [])
            if boundary_rel_id not in candidate_ids:
                return False, {
                    "reject_reason": "GAP_CANDIDATE_MISSING_RELATION",
                    "step_evidence": gap_evidence,
                }

            # Check no competing TRUE support in gap
            per_rel = entry.get("per_relation", [])
            for r in per_rel:
                if r.get("relation_index") == boundary_rel_id:
                    continue  # Our candidate relation is fine
                if r.get("verdict") == "TRUE":
                    return False, {
                        "reject_reason": "GAP_COMPETING_TRUE_RELATION",
                        "step_evidence": gap_evidence,
                    }

            # R6 P0-2 fix: Gap relation detail MUST exist with matching binding.
            # The aggregate label already confirmed RELATION_EVIDENCE_UNKNOWN for
            # the step. Per-relation verdict may differ from aggregate (different
            # semantic levels). What matters: the same relation binding persisted
            # through the gap with non-forbidden identity.
            gap_rel = _find_relation_by_id(relations, step, boundary_rel_id)
            if gap_rel is None:
                return False, {
                    "reject_reason": "GAP_MISSING_RELATION_DETAIL",
                    "step_evidence": gap_evidence,
                }

            # Gap relation must not have forbidden reason at per-relation level
            gap_reason = gap_rel.get("reason", "")
            if gap_reason in FORBIDDEN_REASONS:
                return False, {
                    "reject_reason": f"GAP_RELATION_FORBIDDEN_REASON: {gap_reason}",
                    "step_evidence": gap_evidence,
                }

            # Gap relation binding must match boundary (fail-closed on empty sig)
            sig_gap = canonical_relation_binding_digest(gap_rel)
            if not sig_gap:
                return False, {
                    "reject_reason": "GAP_RELATION_SIGNATURE_EMPTY",
                    "step_evidence": gap_evidence,
                }
            if sig_gap != sig_left:
                return False, {
                    "reject_reason": "RELATION_BINDING_CHANGED_IN_GAP",
                    "step_evidence": gap_evidence,
                }

        # Store boundary evidence in gap_evidence last element
        if gap_evidence:
            gap_evidence[-1]["boundary_relation_id"] = boundary_rel_id
            gap_evidence[-1]["boundary_signature"] = sig_left[:16]

    elif not diagnostic:
        return False, {"reject_reason": "IDENTITY_CHECKS_SKIPPED_FORMAL_MODE", "step_evidence": gap_evidence}

    return True, {"step_evidence": gap_evidence}


# --- Relation lookup helpers (P0-2 fix: relation_id ≠ list position) ---

def _get_sidecar_entry(relations, step) -> dict | None:
    """Get the raw sidecar entry dict for a step."""
    if not isinstance(relations, dict):
        return None
    entry = relations.get(step)
    if isinstance(entry, dict):
        return entry
    return None


def _find_relation_by_id(relations, step, relation_id) -> dict | None:
    """Find a specific relation by its relation_index/ID (not list position)."""
    entry = _get_sidecar_entry(relations, step)
    if entry is None:
        return None
    per_rel = entry.get("per_relation", [])
    for r in per_rel:
        if r.get("relation_index") == relation_id:
            return r
    return None


def _boundary_relation_id(relations, step) -> int | None:
    """Get the unique relation ID for a TRUE boundary step.

    Only UNIQUE_SUPPORT boundary steps are eligible (P0-2 fix).
    MULTI_SUPPORT or AMBIGUOUS boundaries are rejected.
    """
    entry = _get_sidecar_entry(relations, step)
    if entry is None:
        return None
    status = entry.get("selection_status", "")
    if status != "UNIQUE_SUPPORT":
        return None
    # R6 P0-3: Formal mode only — no legacy fallback
    rid = entry.get("selected_relation_id")
    if rid is None:
        return None
    # Verify the ID actually exists in per_relation
    per_rel = entry.get("per_relation", [])
    for r in per_rel:
        if r.get("relation_index") == rid:
            return rid
    return None


# Legacy compatibility: old function renamed internally
def _episode_relation_at(labels, step, relations) -> dict | None:
    """Legacy: returns relation for a step using selected_relation_index as list position.

    Deprecated by _find_relation_by_id and _boundary_relation_id.
    Kept for backward compatibility with existing tests.
    """
    if not relations:
        return None
    if isinstance(relations, dict):
        entry = relations.get(step)
        if not isinstance(entry, dict):
            return None
        per_rel = entry.get("per_relation", [])
        sel_idx = entry.get("selected_relation_index")
        if sel_idx is not None and sel_idx < len(per_rel):
            return per_rel[sel_idx]
        return None
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
    right_censored: np.ndarray | None = None,
    geom_na: np.ndarray | None = None,
) -> np.ndarray:
    """Teacher-event-based weights.

    Weight hierarchy (frozen):
      Per-episode: each episode gets equal total positive weight (1.0).
        Within each episode, each consolidated TRUE event gets equal share
        (distributed evenly across its fragments).
      Known FALSE spans get equal total negative weight per episode.
      RIGHT_CENSORED, GEOMETRY_NOT_APPLICABLE, UNKNOWN, articulated steps
      get zero training weight regardless of mask.

    Precondition: labels and masks are contiguous zero-based arrays where
    index i corresponds to step i.
    """
    n = len(labels)
    weights = np.zeros(n, dtype=np.float32)

    effective_mask = masks.copy()
    if right_censored is not None:
        effective_mask = effective_mask & (~right_censored)
    if geom_na is not None:
        effective_mask = effective_mask & (~geom_na)

    event_groups = consolidated_events.get("event_groups", [])

    if not event_groups:
        return _fallback_weights(labels, effective_mask)

    num_events = len(event_groups)
    pos_weight_per_event = 1.0 / max(num_events, 1)

    for group in event_groups:
        group_steps = []
        for frag_start, frag_end in group["fragment_ranges"]:
            for i in range(frag_start, frag_end + 1):
                if 0 <= i < n and effective_mask[i] and labels[i] == 1:
                    group_steps.append(i)
        if group_steps:
            per_step = pos_weight_per_event / len(group_steps)
            for i in group_steps:
                weights[i] = per_step

    i = 0
    neg_spans = []
    while i < n:
        if effective_mask[i] and labels[i] == 0:
            j = i + 1
            while j < n and effective_mask[j] and labels[j] == 0:
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
