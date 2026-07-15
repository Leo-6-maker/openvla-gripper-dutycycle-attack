#!/usr/bin/env python3
"""Summarize S1 Teacher-label availability and apply pre-registered HOLD gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_b3_teacher_invariants import (  # noqa: E402
    HEADS,
    MASKS,
    audit_episode,
    load_episode,
)


def _known(head: str, row: dict[str, Any]) -> bool:
    mask = MASKS[head]
    return not row[mask] if mask == "retention_unknown_mask" else row[mask]


def _bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    head_stats = {}
    for head in HEADS:
        known = [row for row in records if _known(head, row)]
        positives = sum(bool(row.get(head)) for row in known)
        head_stats[head] = {
            "known": len(known),
            "positive": positives,
            "negative": len(known) - positives,
            "unknown": len(records) - len(known),
        }
    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if isinstance(row.get("event_id"), int) and row["event_id"] >= 0:
            events[row["event_id"]].append(row)
    event_rows = []
    positive_anchor_distances = []
    positive_release_distances = []
    for event_id, rows in sorted(events.items()):
        start = min(row["step"] for row in rows)
        positives = [row for row in rows if row.get("retention_continuation_t10") is True]
        release = next((row["step"] for row in records if row.get("event_release_onset") and row.get("released_event_id") == event_id), None)
        for row in positives:
            positive_anchor_distances.append(row["step"] - start)
            if isinstance(release, int):
                positive_release_distances.append(release - row["step"])
        event_rows.append({
            "event_id": event_id,
            "event_ordinal": event_id,
            "step_count": len(rows),
            "supported": any(row.get("event_support") is True for row in rows),
            "t10_positive_anchors": len(positives),
            "release_step": release,
        })
    return {
        "episodes": 1,
        "steps": len(records),
        "event_count": len(event_rows),
        "event_ordinal_counts": dict(sorted(Counter(row["event_ordinal"] for row in event_rows).items())),
        "event_rows": event_rows,
        "head_stats": head_stats,
        "t10_positive_anchor_count": sum(row.get("retention_continuation_t10") is True for row in records),
        "t10_positive_anchor_distances": positive_anchor_distances,
        "t10_positive_release_distances": positive_release_distances,
        "static_close_positive_count": sum(
            row.get("grasp_support") is True and row.get("retention_active") is False for row in records
        ),
        "release_overlap_count": sum(
            row.get("retention_continuation_t10") is True and row.get("release_imminent") is True for row in records
        ),
        "t10_all_unknown": all(row.get("retention_unknown_mask") is True for row in records),
    }


def audit_distribution(materialized_root: Path, protocol_path: Path, expected_episodes: int = 800) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    episode_roots = sorted(path.parent for path in materialized_root.rglob("materialization_manifest.json"))
    per_episode = []
    invariant_reports = []
    for episode_root in episode_roots:
        try:
            rows, events = load_episode(episode_root)
            invariant = audit_episode(rows, events)
            stats = _bucket(rows)
            manifest = json.loads((episode_root / "materialization_manifest.json").read_text(encoding="utf-8"))
            identity = manifest.get("source_identity", {})
            stats.update({"episode_root": str(episode_root), **identity})
        except Exception as exc:  # noqa: BLE001 - retain every failed identity
            invariant = {"status": "HOLD", "violations": [f"LOAD_ERROR:{type(exc).__name__}:{exc}"]}
            stats = {"episode_root": str(episode_root), "episodes": 1, "event_count": 0, "head_stats": {}}
        per_episode.append(stats)
        invariant_reports.append(invariant)

    totals = {
        "episodes": len(per_episode),
        "events": sum(row.get("event_count", 0) for row in per_episode),
        "event_ordinal_counts": dict(sorted(Counter(
            event["event_ordinal"] for row in per_episode for event in row.get("event_rows", [])
        ).items())),
        "heads": {},
        "t10_positive_anchors": sum(row.get("t10_positive_anchor_count", 0) for row in per_episode),
        "zero_positive_events": sum(
            event["t10_positive_anchors"] == 0
            for row in per_episode for event in row.get("event_rows", [])
        ),
        "supported_events": sum(
            event["supported"] for row in per_episode for event in row.get("event_rows", [])
        ),
        "supported_events_without_anchor": sum(
            event["supported"] and event["t10_positive_anchors"] == 0
            for row in per_episode for event in row.get("event_rows", [])
        ),
        "positive_anchor_distance_mean": mean(
            distance for row in per_episode for distance in row.get("t10_positive_anchor_distances", [])
        ) if any(row.get("t10_positive_anchor_distances") for row in per_episode) else None,
        "positive_release_distance_mean": mean(
            distance for row in per_episode for distance in row.get("t10_positive_release_distances", [])
        ) if any(row.get("t10_positive_release_distances") for row in per_episode) else None,
        "static_close_positive_count": sum(row.get("static_close_positive_count", 0) for row in per_episode),
        "release_overlap_count": sum(row.get("release_overlap_count", 0) for row in per_episode),
        "task_all_unknown": [],
        "task_t10_rollup": {},
        "suite_episode_counts": {},
        "task_episode_counts": {},
        "suite_t10_positive_anchors": {},
        "l10_later_event_known_positive": 0,
    }
    for head in HEADS:
        totals["heads"][head] = {
            key: sum(row.get("head_stats", {}).get(head, {}).get(key, 0) for row in per_episode)
            for key in ("known", "positive", "negative", "unknown")
        }
    for row in per_episode:
        suite = row.get("suite")
        if suite:
            totals["suite_t10_positive_anchors"][suite] = totals["suite_t10_positive_anchors"].get(suite, 0) + row.get("t10_positive_anchor_count", 0)
        if row.get("suite") == "libero_10":
            totals["l10_later_event_known_positive"] += sum(
                event["t10_positive_anchors"] > 0 and event["event_ordinal"] >= 1
                for event in row.get("event_rows", [])
            )

        suite = row.get("suite")
        task_idx = row.get("task_idx")
        if isinstance(suite, str):
            totals["suite_episode_counts"][suite] = totals["suite_episode_counts"].get(suite, 0) + 1
        if isinstance(suite, str) and isinstance(task_idx, int):
            task_key = f"{suite}/task_{task_idx:02d}"
            totals["task_episode_counts"][task_key] = totals["task_episode_counts"].get(task_key, 0) + 1
            t10 = row.get("head_stats", {}).get("retention_continuation_t10", {})
            rollup = totals["task_t10_rollup"].setdefault(task_key, {"known": 0, "unknown": 0, "positive": 0})
            rollup["known"] += t10.get("known", 0)
            rollup["unknown"] += t10.get("unknown", 0)
            rollup["positive"] += t10.get("positive", 0)

    totals["task_all_unknown"] = sorted(
        task_key for task_key, rollup in totals["task_t10_rollup"].items()
        if rollup["unknown"] > 0 and rollup["known"] == 0
    )

    hold_reasons = []
    gates = protocol.get("hold_conditions", {})
    if totals["episodes"] != expected_episodes:
        hold_reasons.append("FIT_EPISODE_COUNT_INCOMPLETE")
    if any(totals["suite_episode_counts"].get(suite, 0) != 200 for suite in protocol.get("suites", [])):
        hold_reasons.append("FIT_SUITE_EPISODE_COUNT_INCOMPLETE")
    if any(count != 20 for count in totals["task_episode_counts"].values()) or len(totals["task_episode_counts"]) != 40:
        hold_reasons.append("FIT_TASK_EPISODE_COUNT_INCOMPLETE")
    if not totals["heads"]["retention_continuation_t10"]["positive"]:
        hold_reasons.append("NO_T10_POSITIVE_ANYWHERE")
    if gates.get("suite_without_t10_positive") and any(
        totals["suite_t10_positive_anchors"].get(suite, 0) == 0
        for suite in protocol.get("suites", [])
    ):
        hold_reasons.append("SUITE_WITHOUT_T10_POSITIVE")
    if totals["l10_later_event_known_positive"] == 0 and gates.get("l10_later_event_without_known_positive"):
        hold_reasons.append("L10_LATER_EVENT_WITHOUT_KNOWN_POSITIVE")
    if totals["task_all_unknown"] and gates.get("task_all_unknown"):
        hold_reasons.append("TASK_ALL_UNKNOWN")
    if any(report.get("status") != "PASS" for report in invariant_reports) and gates.get("label_invariant_violation"):
        hold_reasons.append("LABEL_INVARIANT_VIOLATION")
    if totals["supported_events"]:
        fraction = totals["supported_events_without_anchor"] / totals["supported_events"]
        if fraction > float(gates.get("supported_events_without_anchor_fraction_gt", 1.0)):
            hold_reasons.append("SUPPORTED_EVENTS_WITHOUT_ANCHOR_TOO_HIGH")
    t10 = totals["heads"]["retention_continuation_t10"]
    ratio = float("inf") if t10["negative"] == 0 and t10["positive"] else t10["positive"] / max(1, t10["negative"])
    if ratio > float(gates.get("positive_negative_ratio_gt", float("inf"))):
        hold_reasons.append("POSITIVE_NEGATIVE_RATIO_TOO_HIGH")
    if totals["release_overlap_count"]:
        hold_reasons.append("T10_RELEASE_OVERLAP_PRESENT")

    return {
        "schema": "B3_S1_LABEL_DISTRIBUTION_AUDIT_V1",
        "status": "PASS" if not hold_reasons else "HOLD",
        "materialized_root": str(materialized_root.resolve()),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "teacher_labels_read": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "hold_reasons": sorted(set(hold_reasons)),
        "totals": totals,
        "invariant_episode_count": len(invariant_reports),
        "invariant_violation_episode_count": sum(report.get("status") != "PASS" for report in invariant_reports),
        "episodes": per_episode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=800)
    args = parser.parse_args()
    report = audit_distribution(args.materialized_root.resolve(), args.protocol.resolve(), args.expected_episodes)
    args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "hold_reasons")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
