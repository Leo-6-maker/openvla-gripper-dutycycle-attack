#!/usr/bin/env python3
"""CPU-only R6.1 diagnosis for the sealed Physics smoke.

This script never trains, reads protected splits, or consumes attack evidence.
It compares the Physics Teacher candidate segments with the current loader
windows and replays sealed utility/release/regrasp probabilities with vetoes
ablated one at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory
from gripper_attack.v5_dataset import classify_v5_episode_windows, load_fit_registry, load_v5_episodes
from gripper_attack.v5_scheduler import V5OneShotScheduler, V5SchedulerConfig


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_segments(teacher_root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    path = teacher_root / "labels" / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}" / "physics_teacher_v2.jsonl"
    rows = _jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        if bool(item.get("candidate_close")):
            grouped[str(item["window_id"])].append(item)
    result: list[dict[str, Any]] = []
    for window_id, members in sorted(grouped.items()):
        steps = [int(item["step"]) for item in members]
        tiers = [int(item["utility_tier"]) for item in members if item.get("utility_tier") is not None]
        phases = sorted({str(item["phase_name"]) for item in members})
        result.append({
            "window_id": window_id,
            "start_step": min(steps),
            "end_step": max(steps),
            "step_count": len(steps),
            "contiguous": steps == list(range(min(steps), max(steps) + 1)),
            "tier_set": sorted(set(tiers)),
            "phase_set": phases,
            "max_tier": max(tiers) if tiers else None,
            "tier2_or_3_onset": next((steps[index] for index, item in enumerate(members) if item.get("utility_tier") is not None and int(item["utility_tier"]) >= 2), None),
        })
    return result


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _geometry(episodes: list[Any], rows_by_key: dict[str, dict[str, Any]], teacher_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_segment_count = 0
    raw_transition_count = 0
    raw_internal_mixed_count = 0
    loader_split_count = 0
    true_mixed_internal_count = 0
    true_mixed_independent_count = 0
    length_values: list[float] = []
    tier_values: list[float] = []
    for episode in episodes:
        source_row = rows_by_key[episode.canonical_parent_key]
        raw = _raw_segments(teacher_root, source_row)
        raw_segment_count += len(raw)
        by_base: dict[str, list[Any]] = defaultdict(list)
        for window in episode.windows:
            by_base[str(window.window_id).split("#segment", 1)[0]].append(window)
        loader_tiers_by_base = {key: [int(window.utility_tier) for window in value] for key, value in by_base.items()}
        for segment in raw:
            if len(segment["tier_set"]) > 1 or len(segment["phase_set"]) > 1:
                raw_transition_count += 1
            if any(tier >= 2 for tier in segment["tier_set"]) and any(tier <= 1 for tier in segment["tier_set"]):
                raw_internal_mixed_count += 1
            split = len(by_base.get(segment["window_id"], []))
            if split > 1:
                loader_split_count += 1
            length_values.append(float(segment["step_count"]))
            tier_values.append(float(segment["max_tier"] if segment["max_tier"] is not None else -1))
        category = classify_v5_episode_windows(episode.windows)
        if category == "TRUE_MIXED":
            mixed_bases = [tiers for tiers in loader_tiers_by_base.values() if any(tier >= 2 for tier in tiers) and any(tier <= 1 for tier in tiers)]
            if mixed_bases:
                true_mixed_internal_count += 1
            raw_max = [int(item["max_tier"]) for item in raw if item["max_tier"] is not None]
            if len(raw_max) > 1 and any(tier >= 2 for tier in raw_max) and any(tier <= 1 for tier in raw_max):
                true_mixed_independent_count += 1
        rows.append({
            "canonical_parent_key": episode.canonical_parent_key,
            "raw_candidate_segment_count": len(raw),
            "loader_rankable_window_count": len(episode.windows),
            "loader_window_split_count": sum(max(0, len(by_base.get(item["window_id"], [])) - 1) for item in raw),
            "category_from_loader_windows": category,
            "raw_segments_with_tier_or_phase_transition": sum(len(item["tier_set"]) > 1 or len(item["phase_set"]) > 1 for item in raw),
            "raw_segments_with_internal_tier_mixture": sum(any(tier >= 2 for tier in item["tier_set"]) and any(tier <= 1 for tier in item["tier_set"]) for item in raw),
            "raw_segments": raw,
        })
    summary = {
        "schema": "DETECTOR_V5_PHYSICS_R61_WINDOW_GEOMETRY_V1",
        "identity_count": len(episodes),
        "raw_candidate_segment_count": raw_segment_count,
        "loader_rankable_window_count": sum(int(item["loader_rankable_window_count"]) for item in rows),
        "raw_segments_with_tier_or_phase_transition": raw_transition_count,
        "raw_segments_with_internal_tier_mixture": raw_internal_mixed_count,
        "raw_segments_split_by_loader": loader_split_count,
        "true_mixed_episode_count": sum(item["category_from_loader_windows"] == "TRUE_MIXED" for item in rows),
        "true_mixed_from_same_raw_segment_count": true_mixed_internal_count,
        "true_mixed_with_multiple_raw_segments_count": true_mixed_independent_count,
        "window_length_vs_max_tier_pearson": _pearson(length_values, tier_values),
        "category_counts": dict(Counter(item["category_from_loader_windows"] for item in rows)),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    return summary, rows


def _prediction_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    records = _jsonl(root / "prediction_records.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["canonical_parent_key"])].append(row)
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in values] != list(range(len(values))):
            raise ValueError(f"prediction step closure failed: {key}")
    return dict(grouped)


def _window_for_step(episode: Any, step: int) -> Any | None:
    return next((window for window in episode.windows if step in window.step_indices), None)


def _veto_ablation(episodes: list[Any], prediction_root: Path) -> dict[str, Any]:
    predictions = _prediction_rows(prediction_root)
    if set(predictions) != {episode.canonical_parent_key for episode in episodes}:
        raise ValueError("prediction root identity set does not match validation episodes")
    modes = {
        "U": (False, False),
        "U+R": (True, False),
        "U+G": (False, True),
        "U+R+G": (True, True),
    }
    output: dict[str, Any] = {"schema": "DETECTOR_V5_PHYSICS_R61_VETO_ABLATION_V1", "prediction_root_sha256s_sha256": sha256_file(prediction_root / "SHA256SUMS"), "modes": {}}
    for mode, (release_enabled, regrasp_enabled) in modes.items():
        thresholds: list[dict[str, Any]] = []
        for index in range(5, 96, 5):
            threshold = index / 100.0
            episode_rows: list[dict[str, Any]] = []
            for episode in episodes:
                scheduler = V5OneShotScheduler(V5SchedulerConfig(
                    utility_threshold=threshold,
                    release_veto_enabled=release_enabled,
                    regrasp_veto_enabled=regrasp_enabled,
                    uncertainty_veto_enabled=False,
                ))
                emitted: list[int] = []
                for item in predictions[episode.canonical_parent_key]:
                    result = scheduler.update(
                        step=int(item["step"]),
                        candidate_close=bool(item["candidate_close"]),
                        valid=bool(item["student_valid"]),
                        utility_probability=float(item["utility_probability"]),
                        release_probability=float(item["release_probability"]),
                        regrasp_probability=float(item["regrasp_probability"]),
                        uncertainty_probability=0.0,
                    )
                    if result["emit"]:
                        emitted.append(int(item["step"]))
                selected = _window_for_step(episode, emitted[0]) if emitted else None
                category = classify_v5_episode_windows(episode.windows)
                best_tier = max((int(window.utility_tier) for window in episode.windows), default=None)
                selected_tier = None if selected is None else int(selected.utility_tier)
                episode_rows.append({
                    "canonical_parent_key": episode.canonical_parent_key,
                    "category": category,
                    "emit": bool(emitted),
                    "emit_count": len(emitted),
                    "selected_tier": selected_tier,
                    "best_tier": best_tier,
                    "selected_tier_ge2": selected_tier is not None and selected_tier >= 2,
                    "selected_highest_tier": selected_tier is not None and best_tier is not None and selected_tier == best_tier,
                    "outside_rankable": bool(emitted) and selected is None,
                    "release_trigger": bool(emitted and episode.release_imminent[emitted[0]]),
                    "regrasp_trigger": bool(emitted and episode.regrasp_or_unstable[emitted[0]]),
                })
            positive = [row for row in episode_rows if row["category"] in {"TRUE_MIXED", "POSITIVE_ONLY"}]
            mixed = [row for row in episode_rows if row["category"] == "TRUE_MIXED"]
            pure = [row for row in episode_rows if row["category"] == "PURE_NEGATIVE"]
            selected = [row for row in episode_rows if row["emit"] and not row["outside_rankable"]]
            thresholds.append({
                "threshold": threshold,
                "critical_window_recall": sum(row["selected_tier_ge2"] for row in positive) / len(positive) if positive else None,
                "mixed_scheduler_correct_selection": sum(row["selected_highest_tier"] for row in mixed) / len(mixed) if mixed else None,
                "pure_negative_abstention": sum(not row["emit"] for row in pure) / len(pure) if pure else None,
                "selected_tier_ge2_precision": sum(row["selected_tier_ge2"] for row in selected) / len(selected) if selected else None,
                "total_emits": sum(row["emit"] for row in episode_rows),
                "outside_rankable_emits": sum(row["outside_rankable"] for row in episode_rows),
                "release_triggers": sum(row["release_trigger"] for row in episode_rows),
                "regrasp_triggers": sum(row["regrasp_trigger"] for row in episode_rows),
                "one_shot_compliance": all(row["emit_count"] <= 1 for row in episode_rows),
            })
        eligible = [row for row in thresholds if row["critical_window_recall"] is not None and row["critical_window_recall"] >= 0.95]
        output["modes"][mode] = {
            "release_veto_enabled": release_enabled,
            "regrasp_veto_enabled": regrasp_enabled,
            "working_point": max(eligible, key=lambda row: row["threshold"]) if eligible else None,
            "thresholds": thresholds,
        }
    return output


def _subset_summary(episodes: list[Any], subset_path: Path) -> dict[str, Any]:
    identities = _json(subset_path)
    if isinstance(identities, dict):
        identities = identities.get("identities", [])
    selected = [episode for episode in episodes if episode.canonical_parent_key in set(identities)]
    return {
        "schema": "DETECTOR_V5_PHYSICS_R61_TRAINING_SUBSET_GEOMETRY_V1",
        "identity_count": len(selected),
        "identity_sha256": sha256_file(subset_path),
        "category_counts": dict(Counter(classify_v5_episode_windows(episode.windows) for episode in selected)),
        "window_count": sum(len(episode.windows) for episode in selected),
        "tier_counts": dict(Counter(int(window.utility_tier) for episode in selected for window in episode.windows)),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }


def _write_output(output: Path, payloads: dict[str, str]) -> None:
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        for name, value in payloads.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value, encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, default=0, choices=range(4))
    parser.add_argument("--subset-identities", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for root in (args.registry_root, args.s1_root, args.teacher_root, args.fold_root):
        verify_sealed_directory(root.resolve())
    rows = load_fit_registry(args.registry_csv.resolve())
    rows_by_key = {row["canonical_parent_key"]: row for row in rows}
    fold = _json(args.fold_root.resolve() / "B3_OFFICIAL_V3_FIT_FOLD_MANIFEST_V1.json")
    fold_row = next(item for item in fold["folds"] if int(item["fold_id"]) == args.fold_id)
    validation = [rows_by_key[key] for key in fold_row["validation_identities"]]
    episodes = load_v5_episodes(args.s1_root.resolve(), args.teacher_root.resolve(), rows)
    by_key = {episode.canonical_parent_key: episode for episode in episodes}
    geometry, geometry_rows = _geometry(episodes, rows_by_key, args.teacher_root.resolve())
    subset = _subset_summary(episodes, args.subset_identities.resolve())
    payloads = {
        "window_geometry.json": json.dumps(geometry, indent=2, sort_keys=True) + "\n",
        "window_geometry_rows.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in geometry_rows),
        "training_subset_geometry.json": json.dumps(subset, indent=2, sort_keys=True) + "\n",
        "input_binding.json": json.dumps({
            "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
            "registry_root_sha256s_sha256": sha256_file(args.registry_root.resolve() / "SHA256SUMS"),
            "s1_root_sha256s_sha256": sha256_file(args.s1_root.resolve() / "SHA256SUMS"),
            "teacher_root_sha256s_sha256": sha256_file(args.teacher_root.resolve() / "SHA256SUMS"),
            "fold_root_sha256s_sha256": sha256_file(args.fold_root.resolve() / "SHA256SUMS"),
            "protected_splits_read": [],
        }, indent=2, sort_keys=True) + "\n",
    }
    if args.prediction_root:
        predictions = _prediction_rows(args.prediction_root.resolve())
        validation_episodes = [by_key[key] for key in fold_row["validation_identities"]]
        if set(predictions) != {episode.canonical_parent_key for episode in validation_episodes}:
            raise ValueError("prediction root does not match fold validation identities")
        payloads["veto_ablation.json"] = json.dumps(_veto_ablation(validation_episodes, args.prediction_root.resolve()), indent=2, sort_keys=True) + "\n"
        payloads["prediction_root_binding.json"] = json.dumps({"prediction_root_sha256s_sha256": sha256_file(args.prediction_root.resolve() / "SHA256SUMS"), "validation_identity_count": len(validation_episodes)}, indent=2, sort_keys=True) + "\n"
    _write_output(args.output_root.resolve(), payloads)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "identity_count": len(episodes), "validation_count": len(validation)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
