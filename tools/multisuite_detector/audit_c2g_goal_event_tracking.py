#!/usr/bin/env python3
"""CPU-only scientific audit for clean per-step C2g goal-event tracking.

The audit distinguishes three cases that must not be conflated:

* no current goal-target contact (allowed to remain unknown for multi-target tasks);
* one current goal target resolved with explicit progress/release evidence;
* ambiguous/unresolved target contact, which is a HOLD rather than a negative label.

No OpenVLA model, LIBERO environment, GPU, attack, or attacked outcome is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_clean_event_tracking import goal_event_bindings
from src.gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)


ELIGIBLE_MECHANISMS = {
    "pick_place_transfer",
    "multi_object_transfer",
    "articulated_object",
    "constrained_manipulation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid/nonempty JSONL required: {path}")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if len(set(steps)) != len(steps):
        raise ValueError(f"duplicate step ids in {path}")
    return ordered


def _progress_known(row: Mapping[str, Any]) -> bool:
    return any(
        key in row and row[key] is not None
        for key in (
            "manipulation_progress_active",
            "constrained_manipulation_active",
            "object_relative_lift",
            "target_distance_decrease",
            "fixture_joint_motion",
        )
    )


def _release_known(row: Mapping[str, Any]) -> bool:
    return row.get("release_safe") is not None


def _contact_positive(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("active_target_contact")
        or row.get("active_target_bilateral_contact")
        or row.get("contacted_goal_targets")
    )


def audit(input_root: Path, burst_length: int) -> dict[str, Any]:
    metadata_paths = sorted(input_root.rglob("episode_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no episode metadata under {input_root}")

    episodes: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    suite_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    thresholds = CleanTeacherThresholds(burst_length=burst_length)

    for metadata_path in metadata_paths:
        step_path = metadata_path.with_name("step_records.jsonl")
        if not step_path.is_file():
            violations.append({"path": str(metadata_path), "reason": "MISSING_STEP_RECORDS"})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            violations.append({"path": str(metadata_path), "reason": "INVALID_METADATA"})
            continue
        rows = read_jsonl(step_path)
        resolution = resolve_task_targets(metadata)
        mechanism = infer_clean_mechanism_type(metadata, resolution=resolution)
        bindings = goal_event_bindings(resolution)
        episode_key = str(metadata.get("episode_key") or metadata.get("parent_key") or metadata_path.parent)
        suite = str(metadata.get("suite", ""))
        suite_counts[suite] += 1
        for artifact in (metadata_path, step_path):
            manifest.append({
                "path": artifact.relative_to(input_root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            })

        active_known = sum(bool(row.get("active_target_known")) for row in rows)
        contact_positive = sum(_contact_positive(row) for row in rows)
        bilateral = sum(bool(row.get("active_target_bilateral_contact")) for row in rows)
        progress_known = sum(_progress_known(row) for row in rows)
        release_known = sum(_release_known(row) for row in rows)
        contacted_unresolved = sum(
            _contact_positive(row) and not bool(row.get("active_target_known"))
            for row in rows
        )
        active_progress_unresolved = sum(
            bool(row.get("active_target_known"))
            and _contact_positive(row)
            and not _progress_known(row)
            for row in rows
        )
        labels = build_clean_teacher_episode(rows, metadata, thresholds=thresholds)
        known_rows = sum(bool(row["label_known_mask"]) for row in labels)
        positives = sum(
            bool(row["y_gripper_critical_window"])
            for row in labels
            if row["label_known_mask"]
        )
        triggerable = sum(
            bool(row["y_burst_feasible"])
            for row in labels
            if row["label_known_mask"]
        )
        starts = sum(
            bool(row["y_attack_start_b"])
            for row in labels
            if row["label_known_mask"]
        )
        for row in rows:
            reason_counts[str(row.get("active_target_reason", "MISSING"))] += 1

        if mechanism in ELIGIBLE_MECHANISMS and not bindings:
            violations.append({
                "episode_key": episode_key,
                "reason": "ELIGIBLE_EPISODE_HAS_NO_GOAL_EVENT_BINDINGS",
            })
        if mechanism in ELIGIBLE_MECHANISMS and known_rows == 0:
            violations.append({
                "episode_key": episode_key,
                "reason": "ELIGIBLE_EPISODE_HAS_ZERO_KNOWN_TEACHER_ROWS",
            })
        if contacted_unresolved:
            violations.append({
                "episode_key": episode_key,
                "reason": "CONTACTED_GOAL_TARGET_UNRESOLVED",
                "count": contacted_unresolved,
            })
        if active_progress_unresolved:
            violations.append({
                "episode_key": episode_key,
                "reason": "ACTIVE_TARGET_PROGRESS_UNRESOLVED",
                "count": active_progress_unresolved,
            })
        if starts > 1:
            violations.append({
                "episode_key": episode_key,
                "reason": "MULTIPLE_ATTACK_START_ROWS",
                "count": starts,
            })

        multi_target = len({binding.target_entity for binding in bindings}) > 1
        articulated = mechanism == "articulated_object"
        row_summary = {
            "episode_key": episode_key,
            "suite": suite,
            "task_index": int(metadata.get("task_index", -1)),
            "mechanism_type": mechanism,
            "goal_event_count": len(bindings),
            "goal_target_count": len({binding.target_entity for binding in bindings}),
            "multi_target": multi_target,
            "articulated": articulated,
            "row_count": len(rows),
            "active_target_known_rows": active_known,
            "contact_positive_rows": contact_positive,
            "bilateral_contact_rows": bilateral,
            "progress_known_rows": progress_known,
            "release_known_rows": release_known,
            "contacted_unresolved_rows": contacted_unresolved,
            "active_progress_unresolved_rows": active_progress_unresolved,
            "known_teacher_rows": known_rows,
            "critical_positive_rows": positives,
            "burst_feasible_rows": triggerable,
            "attack_start_rows": starts,
        }
        episodes.append(row_summary)
        for key, value in row_summary.items():
            if isinstance(value, int) and key not in {"task_index"}:
                totals[key] += value
        totals["multi_target_episode_count"] += int(multi_target)
        totals["articulated_episode_count"] += int(articulated)
        if multi_target:
            totals["multi_target_contact_positive_rows"] += contact_positive
            totals["multi_target_active_target_known_rows"] += active_known
            totals["multi_target_progress_known_rows"] += progress_known
            totals["multi_target_known_teacher_rows"] += known_rows
            totals["multi_target_critical_positive_rows"] += positives
            totals["multi_target_burst_feasible_rows"] += triggerable
        if articulated:
            totals["articulated_contact_positive_rows"] += contact_positive
            totals["articulated_progress_known_rows"] += progress_known
            totals["articulated_known_teacher_rows"] += known_rows
            totals["articulated_critical_positive_rows"] += positives

    canonical_manifest = sorted(manifest, key=lambda row: row["path"])
    manifest_sha = hashlib.sha256(
        "".join(
            f"{row['path']}|{row['bytes']}|{row['sha256']}\n"
            for row in canonical_manifest
        ).encode("utf-8")
    ).hexdigest()
    status = (
        "PASS_C2G_GOAL_EVENT_TRACKING_AUDIT"
        if episodes and not violations
        else "HOLD_C2G_GOAL_EVENT_TRACKING_AUDIT"
    )
    return {
        "gate": "C2G_GOAL_EVENT_TRACKING_AUDIT",
        "status": status,
        "input_root": str(input_root.resolve()),
        "episode_count": len(episodes),
        "suite_episode_counts": dict(sorted(suite_counts.items())),
        "totals": dict(sorted(totals.items())),
        "active_target_reason_counts": dict(sorted(reason_counts.items())),
        "violations": violations,
        "violation_count": len(violations),
        "episodes": episodes,
        "input_manifest_file_count": len(canonical_manifest),
        "input_manifest_sha256": manifest_sha,
        "uses_attack_outcomes": False,
        "openvla_model_loads": 0,
        "libero_environments_created": 0,
        "attacks_launched": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--burst-length", type=int, default=10)
    args = parser.parse_args(argv)
    report = audit(args.input_root.resolve(), args.burst_length)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
