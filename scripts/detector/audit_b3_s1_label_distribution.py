#!/usr/bin/env python3
"""Summarize S1 Teacher-label availability and apply pre-registered HOLD gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from audit_b3_teacher_invariants import (  # noqa: E402
    HEADS,
    MASKS,
    audit_episode,
    load_episode,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return {"runner_git_head": head, "runner_worktree_clean": not dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"runner_git_head": None, "runner_worktree_clean": False}


def _canonical_key(identity: dict[str, Any]) -> str:
    suite = identity.get("suite")
    task_idx = identity.get("task_idx")
    state_id = identity.get("state_id")
    try:
        return f"{suite}/task_{int(task_idx):02d}/state_{int(state_id):02d}"
    except (TypeError, ValueError):
        return ""


def _t10_ratios(positive: int, negative: int) -> dict[str, float]:
    return {
        "negative_to_positive": float("inf") if positive == 0 and negative else negative / max(1, positive),
        "positive_to_negative": float("inf") if negative == 0 and positive else positive / max(1, negative),
    }


def _load_census(census_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with census_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    allowed = {
        "RUNTIME_VALID_SOURCE_PRESENT",
        "RUNTIME_VALID_MATERIALIZATION_DRYRUN_PASS",
        "MATERIALIZATION_DRYRUN_HOLD",
        "RUNTIME_INVALID",
        "MISSING",
        "PROTOCOL_HOLD",
    }
    for row in rows:
        key = row.get("canonical_parent_key", "")
        if not key or key in seen:
            errors.append(f"DUPLICATE_OR_MISSING_CENSUS_KEY:{key}")
            continue
        seen.add(key)
        if row.get("status") not in allowed:
            errors.append(f"UNKNOWN_CENSUS_STATUS:{key}:{row.get('status')}")
            continue
        if row.get("status") != "RUNTIME_VALID_MATERIALIZATION_DRYRUN_PASS":
            continue
        if not row.get("source_artifact_sha256") or not row.get("materializer_config_sha256"):
            errors.append(f"PASS_CENSUS_ROW_MISSING_SHA:{key}")
            continue
        expected[key] = row
    return expected, {
        "census_sha256": sha256_file(census_path),
        "census_row_count": len(rows),
        "census_unique_key_count": len({row.get("canonical_parent_key") for row in rows}),
        "census_errors": sorted(set(errors)),
        "materializable_count": len(expected),
    }


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


def audit_distribution(
    materialized_root: Path,
    protocol_path: Path,
    census_path: Path,
    expected_episodes: int | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected, census_info = _load_census(census_path)
    expected_keys = set(expected)
    expected_episodes = len(expected) if expected_episodes is None else expected_episodes
    per_episode = []
    invariant_reports = []
    actual_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    source_sha_mismatches: list[str] = []
    config_sha_mismatches: list[str] = []
    materializer_shas: set[str] = set()
    for manifest_path in sorted(materialized_root.rglob("materialization_manifest.json")):
        episode_root = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = manifest.get("source_identity", {})
            key = _canonical_key(identity)
            if not key or key in actual_keys:
                duplicate_keys.add(key or f"INVALID_MANIFEST:{episode_root}")
            actual_keys.add(key)
            if key not in expected:
                source_sha_mismatches.append(f"UNEXPECTED_IDENTITY:{key}")
            else:
                if manifest.get("source_artifact_sha256") != expected[key].get("source_artifact_sha256"):
                    source_sha_mismatches.append(key)
                if manifest.get("config_sha256") != expected[key].get("materializer_config_sha256"):
                    config_sha_mismatches.append(key)
            if isinstance(manifest.get("materializer_sha256"), str):
                materializer_shas.add(manifest["materializer_sha256"])
            rows, events = load_episode(episode_root)
            invariant = audit_episode(rows, events)
            stats = _bucket(rows)
            stats.update({"episode_root": str(episode_root), **identity, "canonical_parent_key": key})
        except Exception as exc:  # noqa: BLE001 - retain every failed identity
            invariant = {"status": "HOLD", "violations": [f"LOAD_ERROR:{type(exc).__name__}:{exc}"]}
            stats = {"episode_root": str(episode_root), "episodes": 1, "event_count": 0, "head_stats": {}}
        per_episode.append(stats)
        invariant_reports.append(invariant)

    actual_suite_counts = Counter(row.get("suite") for row in per_episode if row.get("suite"))
    actual_task_counts = Counter(
        f"{row.get('suite')}/task_{int(row.get('task_idx')):02d}"
        for row in per_episode
        if row.get("suite") is not None and isinstance(row.get("task_idx"), int)
    )
    expected_suite_counts = Counter(row.get("suite") for row in expected.values())
    expected_task_counts = Counter(
        f"{row.get('suite')}/task_{int(row.get('task_idx')):02d}" for row in expected.values()
    )

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
        "suite_episode_counts": dict(sorted(actual_suite_counts.items())),
        "task_episode_counts": dict(sorted(actual_task_counts.items())),
        "expected_suite_episode_counts": dict(sorted(expected_suite_counts.items())),
        "expected_task_episode_counts": dict(sorted(expected_task_counts.items())),
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

        task_idx = row.get("task_idx")
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
        hold_reasons.append("MATERIALIZED_IDENTITY_COUNT_MISMATCH")
    if actual_keys != expected_keys:
        hold_reasons.append("MATERIALIZED_IDENTITY_SET_MISMATCH")
    if duplicate_keys:
        hold_reasons.append("DUPLICATE_MATERIALIZED_IDENTITY")
    if census_info["census_row_count"] != 800 or census_info["census_unique_key_count"] != 800:
        hold_reasons.append("CENSUS_IDENTITY_SET_INVALID")
    if census_info["census_errors"]:
        hold_reasons.append("CENSUS_PROVENANCE_INVALID")
    if not expected_keys:
        hold_reasons.append("NO_MATERIALIZABLE_CENSUS_IDENTITIES")
    if actual_suite_counts != expected_suite_counts or actual_task_counts != expected_task_counts:
        hold_reasons.append("MATERIALIZED_SUITE_TASK_SET_MISMATCH")
    if source_sha_mismatches:
        hold_reasons.append("SOURCE_ARTIFACT_SHA_MISMATCH")
    if config_sha_mismatches:
        hold_reasons.append("MATERIALIZER_CONFIG_SHA_MISMATCH")
    runner = _git_provenance()
    if not runner["runner_git_head"] or not runner["runner_worktree_clean"]:
        hold_reasons.append("RUNNER_PROVENANCE_NOT_CLEAN")
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
    ratios = _t10_ratios(t10["positive"], t10["negative"])
    if ratios["negative_to_positive"] > float(gates.get("negative_positive_ratio_gt", float("inf"))):
        hold_reasons.append("NEGATIVE_POSITIVE_RATIO_TOO_HIGH")
    if totals["release_overlap_count"]:
        hold_reasons.append("T10_RELEASE_OVERLAP_PRESENT")

    return {
        "schema": "B3_S1_LABEL_DISTRIBUTION_AUDIT_V1",
        "status": "PASS" if not hold_reasons else "HOLD",
        "materialized_root": str(materialized_root.resolve()),
        "census_path": str(census_path.resolve()),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "census_sha256": census_info["census_sha256"],
        "materializer_config_sha256": sorted({row.get("materializer_config_sha256") for row in expected.values()}),
        "materializer_sha256": sorted(materializer_shas),
        "runner_git_head": runner["runner_git_head"],
        "runner_worktree_clean": runner["runner_worktree_clean"],
        "auditor_script_sha256": sha256_file(Path(__file__).resolve()),
        "identity_binding": {
            "expected_materializable_count": len(expected),
            "actual_materialized_count": len(actual_keys),
            "duplicate_keys": sorted(duplicate_keys),
            "source_sha_mismatches": sorted(set(source_sha_mismatches)),
            "config_sha_mismatches": sorted(set(config_sha_mismatches)),
            "census_errors": census_info["census_errors"],
        },
        "teacher_labels_read": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "hold_reasons": sorted(set(hold_reasons)),
        "totals": totals,
        "t10_negative_to_positive_ratio": ratios["negative_to_positive"],
        "t10_positive_to_negative_ratio": ratios["positive_to_negative"],
        "invariant_episode_count": len(invariant_reports),
        "invariant_violation_episode_count": sum(report.get("status") != "PASS" for report in invariant_reports),
        "episodes": per_episode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_distribution(args.materialized_root.resolve(), args.protocol.resolve(), args.census.resolve())
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"audit output already exists: {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output.with_name(output.name + ".sha256")).write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "hold_reasons")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
