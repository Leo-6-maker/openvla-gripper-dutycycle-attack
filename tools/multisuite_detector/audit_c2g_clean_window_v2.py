#!/usr/bin/env python3
"""CPU-only, read-only audit for the clean C2g Detector-v2 label path."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from src.gripper_attack.c2g_clean_mechanism import infer_clean_mechanism_type
from src.gripper_attack.c2g_clean_window_schema import LABEL_FIELDS
from src.gripper_attack.c2g_teacher_v2_target_resolution import resolve_task_targets
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
ELIGIBLE_MECHANISMS = {
    "pick_place_transfer",
    "multi_object_transfer",
    "articulated_object",
    "constrained_manipulation",
}
_PROGRESS_FIELDS = {
    "lift_transport_or_constraint",
    "manipulation_progress_active",
    "constrained_manipulation_active",
    "object_relative_lift",
    "target_object_relative_lift",
    "relative_lift",
    "target_distance_decrease",
    "object_target_progress",
    "target_relative_progress",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no step rows")
    return rows


def _suite_from_path(path: Path) -> str:
    return next((part for part in path.parts if part in SUITES), "")


def _git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(result.stdout.strip()).resolve()


def _assert_external_output(output_dir: Path, repo_root: Path | None) -> None:
    if repo_root is None:
        return
    output = output_dir.resolve()
    repository = repo_root.resolve()
    if output == repository or output.is_relative_to(repository):
        raise ValueError("audit output_dir must be outside the repository worktree")


def _episode_record(input_root: Path, step_path: Path) -> dict[str, Any]:
    metadata_path = step_path.with_name("episode_metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing episode metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    suite = str(metadata.get("suite") or _suite_from_path(step_path))
    task_index = int(metadata.get("task_index", metadata.get("task_id", -1)))
    if suite not in SUITES or task_index < 0:
        raise ValueError(f"invalid suite/task identity for {step_path}")
    relative_parent = step_path.parent.relative_to(input_root).as_posix()
    metadata = dict(metadata)
    metadata["suite"] = suite
    metadata["task_index"] = task_index
    metadata.setdefault("episode_key", relative_parent)
    resolution = resolve_task_targets(metadata)
    mechanism_type = infer_clean_mechanism_type(metadata, resolution=resolution)
    metadata["mechanism_type"] = mechanism_type
    return {
        "suite": suite,
        "task_index": task_index,
        "episode_key": str(metadata["episode_key"]),
        "mechanism_type": mechanism_type,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "step_path": step_path,
        "relative_parent": relative_parent,
    }


def discover_episode_records(input_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for step_path in sorted(input_root.rglob("step_records.jsonl")):
        try:
            records.append(_episode_record(input_root, step_path))
        except Exception as exc:  # preserve all malformed sources in the audit ledger
            errors.append({"path": str(step_path), "error": f"{type(exc).__name__}: {exc}"})
    return records, errors


def select_balanced_dry_run(
    records: Sequence[dict[str, Any]],
    *,
    episodes_per_suite: int,
) -> list[dict[str, Any]]:
    """Select eligible then boundary episodes deterministically within each suite."""

    if episodes_per_suite <= 0:
        raise ValueError("episodes_per_suite must be positive")
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_suite[str(record["suite"])].append(record)
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        candidates = sorted(by_suite.get(suite, []), key=lambda item: item["relative_parent"])
        eligible = [item for item in candidates if item["mechanism_type"] in ELIGIBLE_MECHANISMS]
        boundary = [item for item in candidates if item["mechanism_type"] not in ELIGIBLE_MECHANISMS]
        ordered: list[dict[str, Any]] = []
        if eligible:
            ordered.append(eligible[0])
        if boundary:
            ordered.append(boundary[0])
        ordered.extend(item for item in candidates if item not in ordered)
        selected.extend(ordered[:episodes_per_suite])
    return selected


def _row_has_only_absolute_z_progress(source: Mapping[str, Any]) -> bool:
    has_absolute_z = any(key in source for key in ("eef_z", "eef_pos_z", "end_effector_z"))
    has_valid_progress = any(key in source and source[key] is not None for key in _PROGRESS_FIELDS)
    return has_absolute_z and not has_valid_progress


def audit_clean_window_v2(
    input_root: Path,
    *,
    episodes_per_suite: int = 2,
    burst_length: int = 10,
    strict_four_suites: bool = False,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit a tiny balanced set without running OpenVLA, LIBERO, or a GPU."""

    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    if output_dir is not None:
        _assert_external_output(output_dir, repo_root or _git_root(Path.cwd()))

    discovered, discovery_errors = discover_episode_records(input_root)
    selected = select_balanced_dry_run(discovered, episodes_per_suite=episodes_per_suite)
    read_errors = list(discovery_errors)
    episode_summaries: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    suite_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    mechanism_counts: Counter[str] = Counter()
    violations: list[dict[str, Any]] = []

    thresholds = CleanTeacherThresholds(burst_length=burst_length)
    for record in selected:
        metadata_path = Path(record["metadata_path"])
        step_path = Path(record["step_path"])
        for artifact in (metadata_path, step_path):
            manifest_entries.append(
                {
                    "path": artifact.relative_to(input_root).as_posix(),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            )
        try:
            source_rows = read_jsonl(step_path)
            built = build_clean_teacher_episode(
                source_rows,
                record["metadata"],
                thresholds=thresholds,
            )
        except Exception as exc:
            read_errors.append(
                {
                    "path": str(step_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        suite = str(record["suite"])
        mechanism = str(record["mechanism_type"])
        suite_counts[suite] += 1
        mechanism_counts[mechanism] += 1
        known = sum(bool(row["label_known_mask"]) for row in built)
        unknown = len(built) - known
        positives = sum(
            bool(row["y_gripper_critical_window"])
            for row in built
            if row["label_known_mask"]
        )
        negatives = known - positives
        starts = sum(
            bool(row["y_attack_start_b"])
            for row in built
            if row["label_known_mask"]
        )
        release_safe = sum(
            bool(row["y_release_safe"])
            for row in built
            if row["label_known_mask"]
        )
        distractor = sum(row["teacher_reason_code"] == "DISTRACTOR_CONTACT" for row in built)
        for row in built:
            reason_counts[str(row["teacher_reason_code"])] += 1
        if mechanism in ELIGIBLE_MECHANISMS and known == 0:
            violations.append(
                {
                    "episode_key": record["episode_key"],
                    "reason": "ELIGIBLE_EPISODE_HAS_ZERO_KNOWN_ROWS",
                }
            )
        if starts > 1:
            violations.append(
                {
                    "episode_key": record["episode_key"],
                    "reason": "MULTIPLE_ATTACK_START_ROWS",
                    "count": starts,
                }
            )
        for source, label in zip(source_rows, built):
            if _row_has_only_absolute_z_progress(source) and label.get("y_gripper_critical_window") is True:
                violations.append(
                    {
                        "episode_key": record["episode_key"],
                        "step": label["step"],
                        "reason": "ABSOLUTE_Z_ONLY_CRITICAL_POSITIVE",
                    }
                )
            if label.get("y_release_safe") is True and label.get("y_gripper_critical_window") is True:
                violations.append(
                    {
                        "episode_key": record["episode_key"],
                        "step": label["step"],
                        "reason": "RELEASE_SAFE_CRITICAL_POSITIVE",
                    }
                )

        episode_summaries.append(
            {
                "episode_key": record["episode_key"],
                "suite": suite,
                "task_index": record["task_index"],
                "mechanism_type": mechanism,
                "row_count": len(built),
                "known_rows": known,
                "unknown_rows": unknown,
                "critical_positive_rows": positives,
                "known_negative_rows": negatives,
                "attack_start_rows": starts,
                "release_safe_rows": release_safe,
                "distractor_rows": distractor,
            }
        )
        label_rows.extend(built)

    canonical_manifest = sorted(manifest_entries, key=lambda item: item["path"])
    manifest_sha256 = hashlib.sha256(
        "".join(
            f"{item['path']}|{item['bytes']}|{item['sha256']}\n"
            for item in canonical_manifest
        ).encode("utf-8")
    ).hexdigest()
    selected_suites = set(suite_counts)
    missing_suites = sorted(set(SUITES) - selected_suites)
    complete = bool(selected) and not read_errors and not violations
    if strict_four_suites:
        complete = complete and not missing_suites

    report = {
        "gate": "C2G_CLEAN_WINDOW_V2_DRY_AUDIT",
        "status": "PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT" if complete else "HOLD_C2G_CLEAN_WINDOW_V2_DRY_AUDIT",
        "input_root": str(input_root),
        "episodes_per_suite_requested": episodes_per_suite,
        "burst_length": burst_length,
        "strict_four_suites": strict_four_suites,
        "discovered_episode_count": len(discovered),
        "selected_episode_count": len(selected),
        "processed_episode_count": len(episode_summaries),
        "label_row_count": len(label_rows),
        "known_row_count": sum(item["known_rows"] for item in episode_summaries),
        "unknown_row_count": sum(item["unknown_rows"] for item in episode_summaries),
        "critical_positive_row_count": sum(item["critical_positive_rows"] for item in episode_summaries),
        "known_negative_row_count": sum(item["known_negative_rows"] for item in episode_summaries),
        "attack_start_row_count": sum(item["attack_start_rows"] for item in episode_summaries),
        "release_safe_row_count": sum(item["release_safe_rows"] for item in episode_summaries),
        "distractor_row_count": sum(item["distractor_rows"] for item in episode_summaries),
        "suite_episode_counts": dict(sorted(suite_counts.items())),
        "mechanism_episode_counts": dict(sorted(mechanism_counts.items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "missing_suites": missing_suites,
        "read_error_count": len(read_errors),
        "violation_count": len(violations),
        "input_manifest_file_count": len(canonical_manifest),
        "input_manifest_total_bytes": sum(item["bytes"] for item in canonical_manifest),
        "input_manifest_sha256": manifest_sha256,
        "uses_attack_outcome": False,
        "openvla_inference_runs": 0,
        "libero_rollouts_launched": 0,
        "gpu_episodes_launched": 0,
        "detectors_trained": 0,
        "datasets_materialized": 0,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "clean_window_v2_audit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "clean_window_v2_episode_summary.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_summaries),
            encoding="utf-8",
        )
        (output_dir / "clean_window_v2_dry_labels.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows),
            encoding="utf-8",
        )
        (output_dir / "clean_window_v2_input_manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in canonical_manifest),
            encoding="utf-8",
        )
        (output_dir / "clean_window_v2_read_errors.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in read_errors),
            encoding="utf-8",
        )
        (output_dir / "clean_window_v2_violations.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in violations),
            encoding="utf-8",
        )

    return report, episode_summaries, read_errors + violations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--episodes-per-suite", type=int, default=2)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--strict-four-suites", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, _, problems = audit_clean_window_v2(
            args.input_root,
            episodes_per_suite=args.episodes_per_suite,
            burst_length=args.burst_length,
            strict_four_suites=args.strict_four_suites,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
        )
    except Exception as exc:
        print(json.dumps({"status": "HOLD_C2G_CLEAN_WINDOW_V2_DRY_AUDIT", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if problems:
        print(json.dumps({"problems": problems}, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
