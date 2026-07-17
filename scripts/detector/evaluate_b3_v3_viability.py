#!/usr/bin/env python3
"""FIT-only event-level viability metrics; no CAL/CHECK/attack execution."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def validate_fit_only(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    if any(int(row.get("state_id", -1)) not in range(20) or row.get("split") != "FIT_TRAIN" for row in rows):
        raise ValueError("viability evaluator accepts FIT_TRAIN states 0-19 only")
    return rows


def event_level_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute pre-registered, threshold-independent bookkeeping metrics.

    Each row is one step and must contain ``canonical_parent_key``,
    ``event_id``, ``event_ordinal``, ``target_t10_known``, ``target_t10``,
    ``pred_emit`` and ``release_imminent``.
    """

    rows = validate_fit_only(records)
    events: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["canonical_parent_key"]), int(row["event_id"]))
        events[key].append(row)
        episodes[str(row["canonical_parent_key"])].append(row)
    eligible = 0
    hit = 0
    later_eligible = 0
    later_hit = 0
    release_overlap = 0
    for event_rows in events.values():
        event_rows.sort(key=lambda row: int(row["step"]))
        positive = [row for row in event_rows if row.get("target_t10_known") is True and row.get("target_t10") is True]
        predicted = [row for row in event_rows if row.get("pred_emit") is True]
        if positive:
            eligible += 1
            is_hit = bool(predicted)
            hit += int(is_hit)
            ordinal = int(event_rows[0].get("event_ordinal", 0))
            if ordinal >= 1:
                later_eligible += 1
                later_hit += int(is_hit)
        release_overlap += sum(int(row.get("pred_emit") is True and row.get("release_imminent") is True) for row in event_rows)
    negative_episode_any_emit = sum(
        int(not any(row.get("target_t10_known") is True and row.get("target_t10") is True for row in rows_for_episode) and any(row.get("pred_emit") is True for row in rows_for_episode))
        for rows_for_episode in episodes.values()
    )
    return {
        "event_count": len(events),
        "eligible_t10_event_count": eligible,
        "full_t10_event_hit_count": hit,
        "full_t10_event_hit_rate": hit / eligible if eligible else None,
        "negative_episode_any_emit_count": negative_episode_any_emit,
        "release_overlap_count": release_overlap,
        "later_event_eligible_count": later_eligible,
        "later_event_hit_count": later_hit,
        "later_event_hit_rate": later_hit / later_eligible if later_eligible else None,
        "effectiveness_metrics_are_not_attack_results": True,
    }


__all__ = ["validate_fit_only", "event_level_metrics"]
