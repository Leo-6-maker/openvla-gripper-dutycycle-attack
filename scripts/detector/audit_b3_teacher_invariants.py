#!/usr/bin/env python3
"""Audit materialized B3 Teacher rows without training or model selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

HEADS = (
    "grasp_support",
    "retention_active",
    "retention_continuation_t10",
    "release_imminent",
)
MASKS = {
    "grasp_support": "grasp_support_mask",
    "retention_active": "retention_active_mask",
    "retention_continuation_t10": "retention_unknown_mask",
    "release_imminent": "release_imminent_mask",
}


def _bool(value: Any) -> bool:
    return isinstance(value, bool)


def _known(head: str, row: dict[str, Any]) -> bool:
    mask = MASKS[head]
    if mask == "retention_unknown_mask":
        return not row[mask]
    return row[mask]


def _unknown_reason(index: int, rows: list[dict[str, Any]]) -> str:
    explicit = rows[index].get("retention_unknown_reason")
    if isinstance(explicit, str) and explicit:
        return explicit
    if len(rows) - index < 10:
        return "INSUFFICIENT_FUTURE_HORIZON"
    future = rows[index:index + 10]
    if any(item.get("valid") is not True or item.get("event_evidence_valid") is not True for item in future):
        return "MISSING_OR_INVALID_EVIDENCE"
    return "UNKNOWN_UNEXPLAINED"


def audit_episode(rows: list[dict[str, Any]], events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    violations: list[str] = []
    if not rows:
        return {"status": "HOLD", "violations": ["EMPTY_TEACHER_STREAM"]}
    steps = [row.get("step") for row in rows]
    if steps != list(range(len(rows))):
        violations.append("NON_CONTIGUOUS_STEPS")
    for index, row in enumerate(rows):
        for head in HEADS:
            mask = MASKS[head]
            if not _bool(row.get(mask)):
                violations.append(f"STEP_{index}_{mask}_NOT_BOOLEAN")
                continue
            known = _known(head, row)
            value = row.get(head)
            if known and value not in (False, True, 0, 1, 0.0, 1.0):
                violations.append(f"STEP_{index}_{head}_NOT_BINARY")
            if not known and value is not None:
                violations.append(f"STEP_{index}_{head}_UNKNOWN_NOT_NULL")
        if not _bool(row.get("valid")):
            violations.append(f"STEP_{index}_VALID_NOT_BOOLEAN")
        if not _bool(row.get("event_evidence_valid")):
            violations.append(f"STEP_{index}_EVENT_EVIDENCE_VALID_NOT_BOOLEAN")
        if not _known("retention_continuation_t10", row):
            row_reason = _unknown_reason(index, rows)
            if row_reason == "UNKNOWN_UNEXPLAINED":
                violations.append(f"STEP_{index}_UNKNOWN_REASON_UNEXPLAINED")

    for index, row in enumerate(rows):
        if row.get("retention_continuation_t10") is not True:
            continue
        future = rows[index:index + 10]
        if len(future) != 10:
            violations.append(f"STEP_{index}_T10_POSITIVE_SHORT_FUTURE")
            continue
        event_id = row.get("event_id", -1)
        if not isinstance(event_id, int) or event_id < 0:
            violations.append(f"STEP_{index}_T10_POSITIVE_WITHOUT_EVENT")
        if any(
            item.get("valid") is not True or item.get("event_evidence_valid") is not True
            for item in future
        ):
            violations.append(f"STEP_{index}_T10_POSITIVE_HAS_UNKNOWN_FUTURE_EVIDENCE")
        if any(item.get("event_id") != event_id for item in future):
            violations.append(f"STEP_{index}_T10_POSITIVE_CROSSES_EVENT")
        if row.get("release_imminent") is True:
            violations.append(f"STEP_{index}_T10_POSITIVE_RELEASE_CONFLICT")

    event_ids = sorted({row.get("event_id") for row in rows if isinstance(row.get("event_id"), int) and row["event_id"] >= 0})
    if event_ids and event_ids != list(range(event_ids[-1] + 1)):
        violations.append("EVENT_ORDINAL_NOT_CONTIGUOUS")
    if events is not None:
        previous_end = -1
        for expected_id, event in enumerate(events):
            if event.get("event_id") != expected_id:
                violations.append("EVENT_FILE_ORDINAL_NOT_CONTIGUOUS")
            start, end = event.get("start_step"), event.get("end_step")
            if not isinstance(start, int) or not isinstance(end, int) or end < start:
                violations.append(f"EVENT_{expected_id}_INVALID_BOUNDS")
            elif start <= previous_end:
                violations.append(f"EVENT_{expected_id}_OVERLAPS_PREVIOUS")
            previous_end = end if isinstance(end, int) else previous_end
    for index in range(max(0, len(rows) - 9), len(rows)):
        if rows[index].get("retention_unknown_mask") is not True:
            violations.append(f"STEP_{index}_TAIL_T10_NOT_UNKNOWN")

    reasons = Counter(
        _unknown_reason(index, rows)
        for index, row in enumerate(rows)
        if row.get("retention_unknown_mask") is True
    )
    return {
        "status": "PASS" if not violations else "HOLD",
        "step_count": len(rows),
        "event_count": len(event_ids),
        "t10_positive_count": sum(row.get("retention_continuation_t10") is True for row in rows),
        "unknown_t10_count": sum(row.get("retention_unknown_mask") is True for row in rows),
        "unknown_reason_counts": dict(sorted(reasons.items())),
        "violations": sorted(set(violations)),
    }


def load_episode(episode_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((episode_root / "materialization_manifest.json").read_text(encoding="utf-8"))
    allowed = (
        manifest.get("mode") == "fit-label-materialization"
        and manifest.get("teacher_materialization") == "COMPLETED"
    ) or (
        manifest.get("schema") == "B3_CAUSAL_25D_S1_MATERIALIZED_EPISODE_V1"
        and manifest.get("mode") == "fit-label-materialization-25d-causal"
        and manifest.get("teacher_materialization") == "COMPLETED"
    )
    if not allowed:
        raise ValueError(f"not a completed FIT materialization: {episode_root}")
    rows = [json.loads(line) for line in (episode_root / "teacher_retention_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    events = json.loads((episode_root / "retention_events.json").read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError(f"invalid retention_events.json: {episode_root}")
    return rows, events


def audit_root(materialized_root: Path) -> dict[str, Any]:
    episode_roots = sorted(path.parent for path in materialized_root.rglob("materialization_manifest.json"))
    records = []
    for episode_root in episode_roots:
        try:
            rows, events = load_episode(episode_root)
            result = audit_episode(rows, events)
        except Exception as exc:  # noqa: BLE001 - retain a report for every identity
            result = {"status": "HOLD", "violations": [f"LOAD_ERROR:{type(exc).__name__}:{exc}"]}
        result["episode_root"] = str(episode_root)
        records.append(result)
    return {
        "schema": "B3_S1_TEACHER_INVARIANT_AUDIT_V1",
        "status": "PASS" if records and all(row["status"] == "PASS" for row in records) else "HOLD",
        "episode_count": len(records),
        "violation_episode_count": sum(row["status"] != "PASS" for row in records),
        "teacher_labels_read": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "episodes": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_root(args.materialized_root.resolve())
    args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "episode_count", "violation_episode_count")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
