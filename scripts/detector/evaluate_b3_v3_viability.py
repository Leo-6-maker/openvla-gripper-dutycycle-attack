#!/usr/bin/env python3
"""FIT-only, event-level viability metrics.

This evaluator consumes already materialized predictions and labels.  It does
not choose a threshold, read CAL/CHECK, or execute an attack.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping


def validate_fit_only(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("viability evaluator requires at least one FIT row")
    for row in rows:
        if int(row.get("state_id", -1)) not in range(20) or row.get("split") != "FIT_TRAIN":
            raise ValueError("viability evaluator accepts FIT_TRAIN states 0-19 only")
        for name in ("canonical_parent_key", "event_id", "step", "target_t10_known", "target_t10", "pred_emit", "release_imminent"):
            if name not in row:
                raise ValueError(f"viability row is missing {name}")
        if not isinstance(row["pred_emit"], bool) or not isinstance(row["release_imminent"], bool):
            raise ValueError("pred_emit and release_imminent must be bool")
    return rows


def _group_key(row: Mapping[str, Any], level: str) -> str:
    if level == "suite":
        return str(row.get("suite", "UNKNOWN"))
    if level == "task":
        return f"{row.get('suite', 'UNKNOWN')}/task_{int(row.get('task_idx', -1)):02d}"
    if level == "fold":
        return str(row.get("fold_id", "UNKNOWN"))
    if level == "seed":
        return str(row.get("seed", "UNKNOWN"))
    raise ValueError(f"unsupported group level: {level}")


def _event_summary(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_rows.sort(key=lambda row: int(row["step"]))
    positive_steps = sorted(int(row["step"]) for row in event_rows if row["target_t10_known"] is True and row["target_t10"] is True)
    predicted_steps = sorted(int(row["step"]) for row in event_rows if row["pred_emit"] is True)
    predicted_in_window = [step for step in predicted_steps if step in set(positive_steps)]
    first_positive = positive_steps[0] if positive_steps else None
    event_start = min((int(row.get("event_start_step")) for row in event_rows if row.get("event_start_step") is not None), default=first_positive)
    first_pred = predicted_steps[0] if predicted_steps else None
    return {
        "eligible": bool(positive_steps),
        "hit": bool(predicted_in_window),
        "early": bool(first_pred is not None and first_positive is not None and first_pred < first_positive),
        "late": bool(predicted_steps and positive_steps and predicted_steps[-1] > positive_steps[-1]),
        "onset_latency": (min(predicted_in_window) - event_start) if predicted_in_window and event_start is not None else None,
        "ordinal": int(event_rows[0].get("event_ordinal", -1)),
        "release_overlap": sum(int(row["pred_emit"] and row["release_imminent"]) for row in event_rows),
    }


def _aggregate_event_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in summaries if item["eligible"]]
    latency = [item["onset_latency"] for item in eligible if item["onset_latency"] is not None]
    later = [item for item in eligible if item["ordinal"] >= 1]
    return {
        "event_count": len(summaries),
        "eligible_t10_event_count": len(eligible),
        "full_t10_event_hit_count": sum(int(item["hit"]) for item in eligible),
        "full_t10_event_hit_rate": (sum(int(item["hit"]) for item in eligible) / len(eligible)) if eligible else None,
        "early_emit_event_count": sum(int(item["early"]) for item in eligible),
        "late_emit_event_count": sum(int(item["late"]) for item in eligible),
        "onset_latency_mean_steps": mean(latency) if latency else None,
        "release_overlap_count": sum(item["release_overlap"] for item in summaries),
        "later_event_eligible_count": len(later),
        "later_event_hit_count": sum(int(item["hit"]) for item in later),
        "later_event_hit_rate": (sum(int(item["hit"]) for item in later) / len(later)) if later else None,
    }


def event_level_metrics(records: Iterable[dict[str, Any]], *, baseline_metrics: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = validate_fit_only(records)
    events: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = str(row["canonical_parent_key"])
        episodes[identity].append(row)
        event_id = int(row["event_id"])
        if event_id >= 0:
            events[(identity, event_id)].append(row)
    summaries = [_event_summary(event_rows) for event_rows in events.values()]
    metrics = _aggregate_event_summaries(summaries)
    positive_episode_ids = {
        identity for identity, rows_for_episode in episodes.items()
        if any(row["target_t10_known"] is True and row["target_t10"] is True for row in rows_for_episode)
    }
    negative_emit = sum(
        int(identity not in positive_episode_ids and any(row["pred_emit"] is True for row in rows_for_episode))
        for identity, rows_for_episode in episodes.items()
    )
    metrics.update({
        "episode_count": len(episodes),
        "negative_episode_any_emit_count": negative_emit,
        "negative_episode_any_emit_rate": negative_emit / max(len(episodes) - len(positive_episode_ids), 1),
        "event_id_minus_one_excluded": True,
        "effectiveness_metrics_are_not_attack_results": True,
    })
    for level in ("suite", "task", "fold", "seed"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key, event_rows in events.items():
            row = event_rows[0]
            grouped[_group_key(row, level)].append(_event_summary(event_rows))
        metrics[f"by_{level}"] = {name: _aggregate_event_summaries(items) for name, items in sorted(grouped.items())}
    if baseline_metrics is not None:
        metrics["baseline_comparison"] = {
            name: {"full_t10_event_hit_rate": value.get("full_t10_event_hit_rate"), "negative_episode_any_emit_rate": value.get("negative_episode_any_emit_rate")}
            for name, value in sorted(baseline_metrics.items())
        }
    return metrics


__all__ = ["validate_fit_only", "event_level_metrics"]
