"""Clean-only Stage 2 R2 state-aware scheduler evaluation.

This is intentionally separate from the superseded R1 evaluator.  It binds
the sealed G=3 Teacher events to Cache A, searches only clean OOF scores, and
never opens Eval160, protected-eval, or attack roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_d8_3b_run import verify_sha256_seal
from audit_r3_contact_input import sha256_file, verify_seal
from run_detector_clean_freeze import (
    CONFIG,
    DEPLOYMENT_SEED,
    EPOCHS,
    EVENT_G,
    H1_SOURCE_COMMIT,
    H1_SOURCE_TREE,
    LINEAGE_DIGEST,
    SOURCE_SNAPSHOT_SHA256,
    SIDECAR_SEAL,
    TEACHER_SEAL,
    atomic_bytes,
    atomic_json,
    cache_effective_rows,
    load_cache,
    load_clean_event_groups,
    load_oof,
    metric_summary,
    python_environment,
    sha256_json,
    train_final_detector,
    utc_now,
)

PRE_EVENT_WINDOW = 2
POST_EVENT_TOLERANCE = 2
R2_GATE = {
    "false_onset_episode_rate_max": 0.10,
    "negative_active_step_rate_max": 0.05,
    "active_overlap_event_recall_min": 0.70,
    "median_first_activation_delay_max": 2.0,
}
PERSISTENCE_CANDIDATES = (1, 2, 3, 4, 5)
HYSTERESIS_CANDIDATES = (0.0, 0.25, 0.5, 1.0)
COOLDOWN_CANDIDATES = (0, 2, 5, 10)
R1_SOURCE_COMMIT = "4201224f1e8bcb1c47ad63fd1d513c745540eebf"
R1_RAW_SPANS = 734
R2_VERDICTS = {
    "FULL": "STAGE2_R2_FULL_FREEZE_ELIGIBLE",
    "SHADOW": "STAGE2_R2_SHADOW_PROBE_ONLY",
    "FAIL": "STAGE2_R2_SCIENTIFIC_FAIL",
}


def assert_clean_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    forbidden = {"eval160", "protected_eval", "protected", "attack", "attacks"}
    if any(part.lower() in forbidden for part in resolved.parts):
        raise RuntimeError(f"forbidden evaluation/attack path: {resolved}")
    return resolved


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def scheduler_trace(
    sequence: list[dict[str, Any]],
    threshold: float,
    persistence: int,
    hysteresis: float,
    cooldown: int,
) -> list[dict[str, Any]]:
    """Return the complete state trace, not only activation emissions."""
    if persistence < 1 or hysteresis < 0 or cooldown < 0:
        raise ValueError("invalid scheduler configuration")
    trace: list[dict[str, Any]] = []
    consecutive = 0
    previous_step: int | None = None
    latched = False
    next_allowed = -10**9
    for row in sorted(sequence, key=lambda item: int(item["step"])):
        step = int(row["step"])
        score = float(row["score"])
        if not math.isfinite(score):
            raise RuntimeError(f"non-finite score at step {step}")
        above = score > threshold
        if above and previous_step is not None and step == previous_step + 1:
            consecutive += 1
        else:
            consecutive = 1 if above else 0
        previous_step = step

        release = latched and score < threshold - hysteresis
        if release:
            latched = False
        emission = False
        if not latched and consecutive >= persistence and step >= next_allowed:
            emission = True
            latched = True
            next_allowed = step + cooldown
        trace.append(
            {
                "episode_id": str(row["episode_id"]),
                "step": step,
                "score": score,
                "target": float(row["target"]),
                "above_threshold": bool(above),
                "consecutive_positive": int(consecutive),
                "latched_active": bool(latched),
                "emission": bool(emission),
                "release": bool(release),
                "cooldown_remaining": max(int(next_allowed) - step, 0),
            }
        )
    return trace


def build_aggregate_rows(
    rows: list[dict[str, Any]], seed_scores: Mapping[int, Mapping[tuple[str, int], float]]
) -> list[dict[str, Any]]:
    effective = cache_effective_rows(rows)
    by_key = {(str(row["episode_id"]), int(row["step"])): row for row in effective}
    if not by_key:
        raise RuntimeError("no effective Cache A rows")
    aggregate: list[dict[str, Any]] = []
    seeds = sorted(seed_scores)
    for key, row in sorted(by_key.items()):
        scores = [float(seed_scores[seed][key]) for seed in seeds]
        if not all(math.isfinite(value) for value in scores):
            raise RuntimeError(f"non-finite clean OOF score: {key}")
        aggregate.append(
            dict(row, target=float(row["physical_target"]), score=float(np.mean(scores)))
        )
    return aggregate


def threshold_candidates(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise RuntimeError("aggregate OOF scores are empty or non-finite")
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 401)).tolist()
    values = sorted({float(value) for value in [*quantiles, 0.0]})
    if not values:
        raise RuntimeError("threshold grid is empty")
    return values


def event_index(
    aggregate_rows: list[dict[str, Any]], event_groups: Mapping[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_episode[str(row["episode_id"])].append(row)
    for sequence in by_episode.values():
        sequence.sort(key=lambda row: int(row["step"]))

    events: list[dict[str, Any]] = []
    indexed: dict[str, list[dict[str, Any]]] = {}
    for episode_id in sorted(by_episode):
        indexed[episode_id] = []
        for group in event_groups.get(episode_id, []):
            fragments = [(int(start), int(end)) for start, end in group["fragment_ranges"]]
            if not fragments:
                raise RuntimeError(f"empty formal event: {episode_id}")
            start = min(fragment[0] for fragment in fragments)
            end = max(fragment[1] for fragment in fragments)
            item = {
                "event_index": len(events),
                "episode_id": episode_id,
                "suite": episode_id.split("/", 1)[0],
                "consolidated_event_id": int(group["consolidated_event_id"]),
                "fragment_ranges": fragments,
                "fragment_count": int(group["fragment_count"]),
                "event_start": start,
                "event_end": end,
                "event_length": sum(end - start + 1 for start, end in fragments),
            }
            events.append(item)
            indexed[episode_id].append(item)
    if not events:
        raise RuntimeError("formal event index is empty")
    return events, indexed


def _step_map(event_rows: list[dict[str, Any]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    fragments: dict[int, list[int]] = defaultdict(list)
    protected: dict[int, list[int]] = defaultdict(list)
    for event in event_rows:
        index = int(event["event_index"])
        for start, end in event["fragment_ranges"]:
            for step in range(int(start), int(end) + 1):
                fragments[step].append(index)
        for step in range(
            int(event["event_start"]) - PRE_EVENT_WINDOW,
            int(event["event_end"]) + POST_EVENT_TOLERANCE + 1,
        ):
            protected[step].append(index)
    return fragments, protected


def _candidate_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in threshold_candidates(rows):
        for persistence in PERSISTENCE_CANDIDATES:
            for hysteresis in HYSTERESIS_CANDIDATES:
                for cooldown in COOLDOWN_CANDIDATES:
                    candidates.append(
                        {
                            "candidate_id": len(candidates),
                            "threshold": threshold,
                            "persistence": persistence,
                            "hysteresis": hysteresis,
                            "cooldown": cooldown,
                        }
                    )
    return candidates


def _search(
    aggregate_rows: list[dict[str, Any]], events: list[dict[str, Any]], event_groups: Mapping[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _candidate_grid(aggregate_rows)
    count = len(candidates)
    event_count = len(events)
    thresholds = np.asarray([row["threshold"] for row in candidates], dtype=np.float64)
    persistence = np.asarray([row["persistence"] for row in candidates], dtype=np.int16)
    hysteresis = np.asarray([row["hysteresis"] for row in candidates], dtype=np.float64)
    cooldown = np.asarray([row["cooldown"] for row in candidates], dtype=np.int32)
    active_hits = np.zeros((count, event_count), dtype=bool)
    emission_hits = np.zeros((count, event_count), dtype=bool)
    first_active = np.full((count, event_count), -1, dtype=np.int32)
    false_episodes = np.zeros(count, dtype=bool)
    false_onsets = np.zeros(count, dtype=np.int64)
    negative_active_steps = np.zeros(count, dtype=np.int64)
    negative_steps = 0

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_episode[str(row["episode_id"])].append(row)
    for sequence in by_episode.values():
        sequence.sort(key=lambda row: int(row["step"]))

    for episode_id in sorted(by_episode):
        episode_events = list(event_groups.get(episode_id, []))
        event_rows = [event for event in events if event["episode_id"] == episode_id]
        fragments_at, protected_at = _step_map(event_rows)
        consecutive = np.zeros(count, dtype=np.int16)
        latched = np.zeros(count, dtype=bool)
        next_allowed = np.full(count, -10**9, dtype=np.int32)
        previous_step: int | None = None
        for row in by_episode[episode_id]:
            step = int(row["step"])
            score = float(row["score"])
            above = score > thresholds
            if previous_step is not None and step == previous_step + 1:
                consecutive = np.where(above, consecutive + 1, 0).astype(np.int16)
            else:
                consecutive = np.where(above, 1, 0).astype(np.int16)
            previous_step = step
            release = latched & (score < thresholds - hysteresis)
            latched &= ~release
            emission = (~latched) & (consecutive >= persistence) & (step >= next_allowed)
            latched |= emission
            next_allowed = np.where(emission, step + cooldown, next_allowed).astype(np.int32)

            for event_index in fragments_at.get(step, ()):
                active_hits[:, event_index] |= latched
                emission_hits[:, event_index] |= emission
            for event_index in protected_at.get(step, ()):
                newly_active = latched & (first_active[:, event_index] < 0)
                first_active[:, event_index] = np.where(
                    newly_active, step, first_active[:, event_index]
                )
            if not protected_at.get(step):
                false_episodes |= emission
                false_onsets += emission.astype(np.int64)
                if float(row["target"]) == 0.0:
                    negative_steps += 1
                    negative_active_steps += latched.astype(np.int64)

    starts = np.asarray([int(event["event_start"]) for event in events], dtype=np.int32)
    ends = np.asarray([int(event["event_end"]) for event in events], dtype=np.int32)
    event_lengths = np.asarray([int(event["event_length"]) for event in events], dtype=np.int32)
    fragment_counts = np.asarray([int(event["fragment_count"]) for event in events], dtype=np.int32)
    active_count = active_hits.sum(axis=1).astype(np.int64)
    emission_count = emission_hits.sum(axis=1).astype(np.int64)
    delay_mask = active_hits & (first_active >= 0)
    delays = np.where(delay_mask, first_active - starts[np.newaxis, :], np.nan)
    medians = np.full(count, np.nan, dtype=np.float64)
    p25 = np.full(count, np.nan, dtype=np.float64)
    p75 = np.full(count, np.nan, dtype=np.float64)
    before_count = np.zeros(count, dtype=np.int64)
    during_count = np.zeros(count, dtype=np.int64)
    after_count = np.zeros(count, dtype=np.int64)
    anticipation_mask = delay_mask & (first_active >= starts[np.newaxis, :] - PRE_EVENT_WINDOW) & (first_active < starts[np.newaxis, :])
    during_mask = delay_mask & (first_active >= starts[np.newaxis, :]) & (first_active <= ends[np.newaxis, :])
    after_mask = (first_active > ends[np.newaxis, :]) & (first_active <= ends[np.newaxis, :] + POST_EVENT_TOLERANCE)
    before_count = anticipation_mask.sum(axis=1).astype(np.int64)
    during_count = during_mask.sum(axis=1).astype(np.int64)
    after_count = after_mask.sum(axis=1).astype(np.int64)
    for index in np.flatnonzero(np.any(delay_mask, axis=1)):
        values = delays[index, delay_mask[index]]
        medians[index] = float(np.median(values))
        p25[index] = float(np.quantile(values, 0.25))
        p75[index] = float(np.quantile(values, 0.75))

    all_suites = sorted({str(row["episode_id"]).split("/", 1)[0] for row in aggregate_rows})
    suite_event_counts = {suite: sum(event["suite"] == suite for event in events) for suite in all_suites}
    all_suites_covered = bool(all_suites) and all(suite_event_counts[suite] > 0 for suite in all_suites)
    safe_release_status = "UNAVAILABLE_NOT_GATED"
    metrics: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        median_delay = None if not finite(medians[index]) else float(medians[index])
        values = {
            **candidate,
            "event_count": event_count,
            "active_overlap_event_count": int(active_count[index]),
            "active_overlap_event_recall": float(active_count[index] / max(event_count, 1)),
            "emission_event_count": int(emission_count[index]),
            "emission_event_recall": float(emission_count[index] / max(event_count, 1)),
            "anticipatory_event_count_at_2": int(before_count[index]),
            "anticipatory_event_recall_at_2": float(before_count[index] / max(event_count, 1)),
            "detected_before_event_count": int(before_count[index]),
            "detected_during_event_count": int(during_count[index]),
            "detected_after_event_count": int(after_count[index]),
            "median_first_activation_delay": median_delay,
            "p25_first_activation_delay": None if not finite(p25[index]) else float(p25[index]),
            "p75_first_activation_delay": None if not finite(p75[index]) else float(p75[index]),
            "first_activation_samples": int(np.sum(delay_mask[index])),
            "false_onset_episode_count": int(false_episodes[index]),
            "false_onset_episode_rate": float(false_episodes[index] / max(len(by_episode), 1)),
            "negative_active_step_count": int(negative_active_steps[index]),
            "negative_step_count": int(negative_steps),
            "negative_active_step_rate": None if negative_steps == 0 else float(negative_active_steps[index] / negative_steps),
            "false_onset_count": int(false_onsets[index]),
            "false_onset_per_1000_negative_steps": None if negative_steps == 0 else float(false_onsets[index] / negative_steps * 1000.0),
            "safe_release_metric": {"status": safe_release_status},
            "all_suites_have_valid_coverage": all_suites_covered,
        }
        finite_metrics = all(
            finite(values[key])
            for key in (
                "active_overlap_event_recall",
                "emission_event_recall",
                "anticipatory_event_recall_at_2",
                "false_onset_episode_rate",
                "negative_active_step_rate",
                "false_onset_per_1000_negative_steps",
            )
        ) and (median_delay is not None)
        values["finite"] = bool(finite_metrics)
        values["s1_pass"] = bool(
            finite_metrics
            and all_suites_covered
            and values["false_onset_episode_rate"] <= R2_GATE["false_onset_episode_rate_max"]
            and values["negative_active_step_rate"] <= R2_GATE["negative_active_step_rate_max"]
        )
        values["s2_pass"] = bool(
            finite_metrics
            and values["active_overlap_event_recall"] >= R2_GATE["active_overlap_event_recall_min"]
            and values["median_first_activation_delay"] <= R2_GATE["median_first_activation_delay_max"]
        )
        values["pass"] = bool(values["s1_pass"] and values["s2_pass"])
        metrics.append(values)

    def _rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -float(row["active_overlap_event_recall"]),
            float(row["false_onset_episode_rate"]),
            float(row["median_first_activation_delay"] if row["median_first_activation_delay"] is not None else 10**9),
            int(row["persistence"]),
            abs(float(row["threshold"])),
            float(row["threshold"]),
        )

    def _best(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        return min(values, key=_rank) if values else None

    s1 = [row for row in metrics if row["s1_pass"]]
    full = [row for row in s1 if row["s2_pass"]]
    selected = _best(full)
    shadow = _best(s1)
    distances = []
    for row in metrics:
        distances.append(
            (
                max(0.0, 0.70 - row["active_overlap_event_recall"]) / 0.70
                + max(0.0, row["false_onset_episode_rate"] - 0.10) / 0.10
                + max(0.0, row["negative_active_step_rate"] - 0.05) / 0.05
                + max(0.0, (row["median_first_activation_delay"] or 10**9) - 2.0) / 2.0,
                row["candidate_id"],
            )
        )
    closest = metrics[min(distances)[1]]
    search = {
        "schema": "D8_STAGE2_R2_SCHEDULER_SEARCH_V1",
        "threshold_generation": "401 equally spaced quantiles of aggregate clean OOF scores plus threshold=0, deduplicated and sorted",
        "threshold_count": len(threshold_candidates(aggregate_rows)),
        "scheduler_grid": {
            "persistence": list(PERSISTENCE_CANDIDATES),
            "hysteresis": list(HYSTERESIS_CANDIDATES),
            "cooldown": list(COOLDOWN_CANDIDATES),
        },
        "candidate_count": len(metrics),
        "candidates": metrics,
        "selection_rule": "S1 then S2; max active-overlap recall; min false onset; min median delay; min persistence; threshold closest to zero",
        "selected_full_freeze": selected,
        "selected_shadow_probe": shadow,
        "closest_to_original_gate": closest,
        "safe_release_metric": {"status": safe_release_status},
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }
    return metrics, search


def _candidate_by_id(metrics: list[dict[str, Any]], candidate_id: int) -> dict[str, Any]:
    for row in metrics:
        if int(row["candidate_id"]) == int(candidate_id):
            return row
    raise RuntimeError(f"candidate not found: {candidate_id}")


def detailed_candidate_metrics(
    aggregate_rows: list[dict[str, Any]],
    event_groups: Mapping[str, list[dict[str, Any]]],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        by_episode[str(row["episode_id"])].append(row)
    traces: dict[str, list[dict[str, Any]]] = {}
    event_records: list[dict[str, Any]] = []
    false_episode_by_suite: dict[str, set[str]] = defaultdict(set)
    episode_counts: dict[str, int] = defaultdict(int)
    false_onsets = 0
    negative_steps = 0
    negative_active = 0
    for episode_id in sorted(by_episode):
        sequence = sorted(by_episode[episode_id], key=lambda row: int(row["step"]))
        trace = scheduler_trace(sequence, **{key: candidate[key] for key in ("threshold", "persistence", "hysteresis", "cooldown")})
        traces[episode_id] = trace
        suite = episode_id.split("/", 1)[0]
        episode_counts[suite] += 1
        events = []
        for group in event_groups.get(episode_id, []):
            fragments = [(int(start), int(end)) for start, end in group["fragment_ranges"]]
            start = min(item[0] for item in fragments)
            end = max(item[1] for item in fragments)
            events.append((fragments, start, end, int(group["fragment_count"])))
        protected = {
            step
            for _, start, end, _ in events
            for step in range(start - PRE_EVENT_WINDOW, end + POST_EVENT_TOLERANCE + 1)
        }
        for row in trace:
            step = int(row["step"])
            if step not in protected and row["emission"]:
                false_onsets += 1
                false_episode_by_suite[suite].add(episode_id)
            if step not in protected and float(row["target"]) == 0.0:
                negative_steps += 1
                negative_active += int(row["latched_active"])
        for fragments, start, end, fragment_count in events:
            in_frag = [row for row in trace if any(a <= int(row["step"]) <= b for a, b in fragments)]
            in_window = [row for row in trace if start - PRE_EVENT_WINDOW <= int(row["step"]) <= end + POST_EVENT_TOLERANCE and row["latched_active"]]
            active_hit = bool(any(row["latched_active"] for row in in_frag))
            emission_hit = bool(any(row["emission"] for row in in_frag))
            first_active = int(in_window[0]["step"]) if in_window else None
            event_records.append(
                {
                    "suite": suite,
                    "event_length": sum(b - a + 1 for a, b in fragments),
                    "fragment_count": fragment_count,
                    "active_hit": active_hit,
                    "emission_hit": emission_hit,
                    "first_active_step": first_active,
                    "lead_time": None if first_active is None else first_active - start,
                    "before": bool(active_hit and first_active is not None and start - PRE_EVENT_WINDOW <= first_active < start),
                    "during": bool(active_hit and first_active is not None and start <= first_active <= end),
                    "after": bool(first_active is not None and end < first_active <= end + POST_EVENT_TOLERANCE),
                }
            )

    event_count = len(event_records)
    active_hits = sum(row["active_hit"] for row in event_records)
    emission_hits = sum(row["emission_hit"] for row in event_records)
    delays = [row["lead_time"] for row in event_records if row["active_hit"] and row["lead_time"] is not None]
    suites = sorted({row["suite"] for row in event_records})
    suite_breakdown: dict[str, Any] = {}
    for suite in suites:
        subset = [row for row in event_records if row["suite"] == suite]
        suite_breakdown[suite] = {
            "event_count": len(subset),
            "active_overlap_event_recall": sum(row["active_hit"] for row in subset) / len(subset),
            "emission_event_recall": sum(row["emission_hit"] for row in subset) / len(subset),
            "false_onset_episode_count": len(false_episode_by_suite.get(suite, set())),
            "false_onset_episode_rate": len(false_episode_by_suite.get(suite, set())) / max(episode_counts[suite], 1),
        }

    def grouped(field: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key in sorted({str(row[field]) for row in event_records}, key=lambda value: (float(value), value)):
            subset = [row for row in event_records if str(row[field]) == key]
            output[key] = {
                "event_count": len(subset),
                "active_overlap_event_recall": sum(row["active_hit"] for row in subset) / len(subset),
                "emission_event_recall": sum(row["emission_hit"] for row in subset) / len(subset),
            }
        return output

    metrics = {
        **dict(candidate),
        "event_count": event_count,
        "active_overlap_event_count": active_hits,
        "active_overlap_event_recall": active_hits / max(event_count, 1),
        "emission_event_count": emission_hits,
        "emission_event_recall": emission_hits / max(event_count, 1),
        "anticipatory_event_count_at_2": sum(row["before"] for row in event_records),
        "anticipatory_event_recall_at_2": sum(row["before"] for row in event_records) / max(event_count, 1),
        "detected_before_event_count": sum(row["before"] for row in event_records),
        "detected_during_event_count": sum(row["during"] for row in event_records),
        "detected_after_event_count": sum(row["after"] for row in event_records),
        "median_first_activation_delay": None if not delays else float(statistics.median(delays)),
        "p25_first_activation_delay": None if not delays else float(np.quantile(delays, 0.25)),
        "p75_first_activation_delay": None if not delays else float(np.quantile(delays, 0.75)),
        "first_activation_samples": len(delays),
        "false_onset_count": false_onsets,
        "false_onset_episode_count": sum(len(value) for value in false_episode_by_suite.values()),
        "false_onset_episode_rate": sum(len(value) for value in false_episode_by_suite.values()) / max(len(by_episode), 1),
        "negative_active_step_count": negative_active,
        "negative_step_count": negative_steps,
        "negative_active_step_rate": None if not negative_steps else negative_active / negative_steps,
        "false_onset_per_1000_negative_steps": None if not negative_steps else false_onsets / negative_steps * 1000.0,
        "suite_breakdown": suite_breakdown,
        "event_length_breakdown": grouped("event_length"),
        "event_fragment_count_breakdown": grouped("fragment_count"),
        "safe_release_metric": {"status": "UNAVAILABLE_NOT_GATED"},
    }
    return metrics, traces


def audit_safe_release_labels(
    cache_root: Path,
    formal_root: Path,
    sidecar_root: Path,
    teacher_root: Path,
    cache_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    roots = [cache_root, formal_root, sidecar_root, teacher_root]
    for root in roots:
        assert_clean_path(root)
        verify_seal(root)
    fields = sorted({str(key) for row in cache_rows for key in row})
    candidate_fields = [field for field in fields if "safe_release" in field or "task_success" in field]
    status = "AVAILABLE" if candidate_fields else "UNAVAILABLE_NOT_GATED"
    return {
        "schema": "D8_STAGE2_R2_SAFE_RELEASE_LABEL_AUDIT_V1",
        "status": status,
        "search_scope": [str(root) for root in roots],
        "sealed_roots_verified": True,
        "candidate_fields_in_clean_cache_rows": candidate_fields,
        "formal_safe_release_label_root": None,
        "formal_safe_release_label_seal": None,
        "reason": "No formal causal safe-release label was present in the sealed clean Cache A row schema." if not candidate_fields else "Candidate fields require explicit identity-closure review before gating.",
        "metric_status": "UNAVAILABLE_NOT_GATED" if not candidate_fields else "PENDING_BINDING_AUDIT",
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def write_trace(path: Path, candidate: Mapping[str, Any], traces: Mapping[str, list[dict[str, Any]]]) -> str:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for episode_id in sorted(traces):
                for row in traces[episode_id]:
                    handle.write(json.dumps({"candidate": dict(candidate), **row}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def seal_directory(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    lines = "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files)
    atomic_bytes(root / "SHA256SUMS", lines.encode("utf-8"))
    sums_sha = sha256_file(root / "SHA256SUMS")
    atomic_bytes(root / "SHA256SUMS.sha256", f"{sums_sha}  SHA256SUMS\n".encode("utf-8"))
    return {"sha256sums_sha256": sums_sha, "file_count": len(files)}


def _r1_reclassification(root: Path) -> dict[str, Any]:
    root = assert_clean_path(root)
    seal = verify_sha256_seal(root)
    return {
        "schema": "D8_STAGE2_R1_RECLASSIFICATION_V1",
        "status": "SUPERSEDED_PROTOCOL_EVALUATION_R1",
        "evidence_status": "NOT_FINAL_SCIENTIFIC_EVIDENCE",
        "root": str(root),
        "root_sha256sums_sha256": seal["sha256sums_sha256"],
        "source_commit": R1_SOURCE_COMMIT,
        "event_count": R1_RAW_SPANS,
        "event_count_definition": "734 raw contiguous positive spans",
        "formal_event_definition": "G=3 consolidated Teacher events",
        "formal_consolidated_event_count": 675,
        "scheduler_metric": "emission-only",
        "safe_release_metric": "unlabelled post-event proxy",
        "clean_task_success_field": "unavailable",
        "original_artifact_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def _artifact_index(root: Path, run_root: Path, run_seal: Mapping[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(run_root).as_posix(), "sha256": sha256_file(path)})
    return {
        "schema": "D8_ARTIFACT_INDEX_R2_V1",
        "run_root": str(run_root),
        "run_root_seal": dict(run_seal),
        "files": files,
        "stage1_retrained": False,
        "stage1_artifact_modified": False,
        "stage2_r1_artifact_modified": False,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
    }


def _write_goal_state(
    goal_root: Path,
    run_root: Path,
    run_seal: Mapping[str, Any],
    result: Mapping[str, Any],
    r1: Mapping[str, Any],
) -> None:
    goal_root.mkdir(parents=True, exist_ok=True)
    verdict = str(result["verdict"])
    authorization = str(result["scheduler_authorization_mode"])
    status = {
        "schema": "D8_GOAL_STATUS_R2_V1",
        "current_stage": "Stage 2 R2 Detector build",
        "stage1": {"status": "FORMAL_PASS", "frozen": True, "retrained": False, "artifact_modified": False},
        "stage2_r1": {"status": "SUPERSEDED_PROTOCOL_EVALUATION", "evidence_status": "NOT_FINAL_SCIENTIFIC_EVIDENCE", "artifact_modified": False},
        "stage2_r2": {"status": verdict, "scheduler_authorization_mode": authorization, "run_root": str(run_root), "run_root_seal": run_seal["sha256sums_sha256"]},
        "stage3a": {"status": "NOT_RUN", "rollouts": 0, "reason": "Current request explicitly defers attack experiments."},
        "stage3b": {"status": "NOT_RUN", "rollouts": 0},
        "stage4": {"status": "NOT_RUN", "rollouts": 0},
        "final_checkpoint_trained": bool(result["final_checkpoint_trained"]),
        "guard_deployment_authorized": bool(result["guard_deployment_authorized"]),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "other_user_processes_terminated": False,
        "other_user_artifacts_modified": False,
        "r1_reclassification": r1["status"],
    }
    atomic_json(goal_root / "GOAL_STATUS_R2.json", status)
    atomic_bytes(
        goal_root / "NIGHTLY_PROGRESS_R2.md",
        (
            f"# D8 Goal R2\n\n"
            f"- Stage 1: FORMAL_PASS / FROZEN\n"
            f"- Stage 2 R1: SUPERSEDED_PROTOCOL_EVALUATION_R1\n"
            f"- Stage 2 R2: {verdict}\n"
            f"- Scheduler authorization: {authorization}\n"
            f"- Final checkpoint trained: {bool(result['final_checkpoint_trained'])}\n"
            f"- Stage 3A/3B and Stage 4: NOT_RUN (attack explicitly deferred)\n"
            f"- Eval160 reads: 0; protected eval reads: 0; attack rollouts: 0\n"
        ).encode("utf-8"),
    )
    with (goal_root / "RESOURCE_LEDGER_R2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "status", "run_root", "worker_count", "eval160_reads", "protected_eval_reads", "attack_rollouts"])
        writer.writerow(["Stage 2 R2", verdict, str(run_root), 0, 0, 0, 0])
        writer.writerow(["Stage 3A", "NOT_RUN", "", 0, 0, 0, 0])
        writer.writerow(["Stage 3B", "NOT_RUN", "", 0, 0, 0, 0])
        writer.writerow(["Stage 4", "NOT_RUN", "", 0, 0, 0, 0])
    atomic_json(goal_root / "ARTIFACT_INDEX_R2.json", _artifact_index(goal_root, run_root, run_seal))


def self_test() -> None:
    def rows(scores: list[float]) -> list[dict[str, Any]]:
        return [{"episode_id": "suite/task/state", "step": i, "score": score, "target": 0.0} for i, score in enumerate(scores)]

    early = scheduler_trace(rows([0.0, 1.0, 1.0, 1.0, 0.0]), 0.5, 1, 0.25, 0)
    assert early[1]["emission"] and all(early[i]["latched_active"] for i in (2, 3))
    assert set(("episode_id", "step", "score", "target", "above_threshold", "consecutive_positive", "latched_active", "emission", "release", "cooldown_remaining")) <= early[0].keys()

    during = scheduler_trace(rows([0.0, 0.0, 1.0, 1.0, 0.0]), 0.5, 1, 0.25, 0)
    assert during[2]["emission"] and during[2]["latched_active"]

    after = scheduler_trace(rows([0.0, 0.0, 0.0, 0.0, 1.0]), 0.5, 1, 0.25, 0)
    assert not any(row["latched_active"] for row in after[:2]) and after[4]["emission"]

    bridged = scheduler_trace(
        [{"episode_id": "suite/task/state", "step": i, "score": 1.0, "target": 1.0} for i in (1, 2, 3)],
        0.5,
        1,
        0.0,
        0,
    )
    assert any(row["latched_active"] for row in bridged if row["step"] in (1, 3))
    print("SELF_TEST_PASS")


def run(args: argparse.Namespace) -> int:
    actual_commit = subprocess_git("rev-parse", "HEAD")
    actual_tree = subprocess_git("rev-parse", "HEAD^{tree}")
    if subprocess_git("status", "--porcelain"):
        raise RuntimeError("clean worktree required")
    if actual_commit != args.expected_source_commit or actual_tree != args.expected_source_tree:
        raise RuntimeError("source commit/tree binding mismatch")

    cache_root = assert_clean_path(args.cache_root)
    formal_root = assert_clean_path(args.formal_root)
    sidecar_root = assert_clean_path(args.sidecar_root)
    teacher_root = assert_clean_path(args.teacher_root)
    r1_root = assert_clean_path(args.r1_root)
    output_root = args.output_root.resolve()
    goal_root = args.goal_root.resolve()
    if output_root.exists():
        raise FileExistsError(str(output_root))
    if goal_root.exists() and any(goal_root.iterdir()):
        raise FileExistsError(f"goal root must be new and empty: {goal_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(str(staging))
    staging.mkdir(parents=True)
    started = utc_now()
    try:
        cache_rows, cache_manifest, cache_seal = load_cache(cache_root, args.expected_cache_seal)
        event_groups, event_binding = load_clean_event_groups(sidecar_root, teacher_root, cache_rows)
        seed_scores, oof_meta = load_oof(formal_root, cache_rows, args.expected_source_commit, args.expected_source_tree)
        aggregate_rows = build_aggregate_rows(cache_rows, seed_scores)
        events, indexed_events = event_index(aggregate_rows, event_groups)
        if event_binding["G"] != EVENT_G or event_binding["raw_true_spans"] != 734 or event_binding["consolidated_events"] != 675 or event_binding["bridged_gaps"] != 59:
            raise RuntimeError("formal G=3 closure mismatch")
        positive_identity_digest = sha256_json(sorted(f"{row['episode_id']}::{int(row['step'])}" for row in cache_effective_rows(cache_rows) if float(row["physical_target"]) == 1.0))
        event_binding_report = {
            **event_binding,
            "schema": "D8_STAGE2_R2_EVENT_BINDING_V1",
            "event_identity_digest": event_binding["event_group_digest"],
            "positive_step_identity_digest": positive_identity_digest,
            "cache_root": str(cache_root),
            "cache_seal": cache_seal["sha256sums_sha256"],
            "formal_root": str(formal_root),
            "formal_seal": oof_meta["formal_seal"]["sha256sums_sha256"],
            "event_identity_closure": True,
            "cache_positive_effective_identity_closure": True,
            "right_censored_explicitly_excluded_from_cache_positive_closure": True,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 0,
        }
        safe_release = audit_safe_release_labels(cache_root, formal_root, sidecar_root, teacher_root, cache_rows)
        metrics, search = _search(aggregate_rows, events, event_groups)
        full = search["selected_full_freeze"]
        shadow = search["selected_shadow_probe"]
        closest = search["closest_to_original_gate"]
        selected = full or shadow or closest
        if selected is None:
            raise RuntimeError("scheduler search returned no candidate")
        selected_details, traces = detailed_candidate_metrics(aggregate_rows, event_groups, selected)
        for key in ("active_overlap_event_recall", "emission_event_recall", "false_onset_episode_rate"):
            if abs(float(selected_details[key]) - float(selected[key])) > 1e-12:
                raise RuntimeError(f"batched/detail metric mismatch: {key}")
        trace_path = staging / "STAGE2_R2_SCHEDULER_TRACE.jsonl"
        trace_sha = write_trace(trace_path, selected, traces)
        r1 = _r1_reclassification(r1_root)
        s1_pass = shadow is not None
        s2_pass = full is not None
        verdict = R2_VERDICTS["FULL"] if s2_pass else R2_VERDICTS["SHADOW"] if s1_pass else R2_VERDICTS["FAIL"]
        authorization = "FULL_FREEZE_ELIGIBLE" if s2_pass else "SHADOW_PROBE_ONLY" if s1_pass else "SCIENTIFIC_FAIL"
        common = {
            "source_commit": actual_commit,
            "source_tree": actual_tree,
            "h1_source_commit": H1_SOURCE_COMMIT,
            "h1_source_tree": H1_SOURCE_TREE,
            "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
            "lineage_digest": LINEAGE_DIGEST,
            "cache_root": str(cache_root),
            "cache_seal": cache_seal["sha256sums_sha256"],
            "formal_root": str(formal_root),
            "formal_seal": oof_meta["formal_seal"]["sha256sums_sha256"],
            "sidecar_root": str(sidecar_root),
            "sidecar_seal": SIDECAR_SEAL,
            "teacher_root": str(teacher_root),
            "teacher_seal": TEACHER_SEAL,
            "event_binding_digest": event_binding["event_group_digest"],
            "safe_release_label_audit": "SAFE_RELEASE_LABEL_AUDIT.json",
            "clean_script_sha256": sha256_file(Path(__file__).resolve()),
            "python_environment": python_environment(),
            "started_utc": started,
        }
        atomic_json(staging / "STAGE2_R1_RECLASSIFICATION.json", r1)
        atomic_json(staging / "STAGE2_R2_EVENT_BINDING.json", event_binding_report)
        atomic_json(staging / "SAFE_RELEASE_LABEL_AUDIT.json", safe_release)
        atomic_json(staging / "STAGE2_R2_SCHEDULER_SEARCH.json", {**search, "provenance": common, "selected_candidate_detail": selected_details, "scheduler_trace_sha256": trace_sha})
        atomic_json(
            staging / "STAGE2_R2_PARETO_FRONTIER.json",
            {
                "schema": "D8_STAGE2_R2_PARETO_FRONTIER_V1",
                "frontier_definition": "non-dominated on false_onset_episode_rate, active_overlap_event_recall, and median_first_activation_delay",
                "max_recall_at_false_onset_le_0.10": max((row for row in metrics if row["false_onset_episode_rate"] <= 0.10), key=lambda row: (row["active_overlap_event_recall"], -row["median_first_activation_delay"], -row["persistence"]), default=None),
                "min_false_onset_at_recall_ge_0.70": min((row for row in metrics if row["active_overlap_event_recall"] >= 0.70), key=lambda row: (row["false_onset_episode_rate"], row["median_first_activation_delay"], row["persistence"]), default=None),
                "closest_to_original_gate": closest,
                "best_by_persistence": {str(persistence): max((row for row in metrics if row["persistence"] == persistence and row["s1_pass"]), key=lambda row: (row["active_overlap_event_recall"], -row["false_onset_episode_rate"], -row["median_first_activation_delay"]), default=None) for persistence in PERSISTENCE_CANDIDATES},
                "selected_candidate_detail": selected_details,
                "event_length_groups": selected_details["event_length_breakdown"],
                "event_fragment_count_groups": selected_details["event_fragment_count_breakdown"],
                "suite_breakdown": selected_details["suite_breakdown"],
                "provenance": common,
                "eval160_reads": 0,
                "protected_eval_reads": 0,
                "attack_rollouts": 0,
            },
        )

        final_checkpoint_trained = False
        guard_deployment_authorized = bool(s1_pass and s2_pass)
        replay = None
        if s1_pass:
            final_metrics = train_final_detector(cache_rows, staging, common)
            final_checkpoint_trained = True
            checkpoint = torch.load(str(staging / "FINAL_DETECTOR_CHECKPOINT.pt"), map_location="cpu", weights_only=False)
            from d8_train_core import apply_normalization, create_model

            model = create_model(DEPLOYMENT_SEED)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            effective = cache_effective_rows(cache_rows)
            tensor = torch.tensor([row["features_25d_raw"] for row in effective], dtype=torch.float32)
            with torch.no_grad():
                score_a = model(apply_normalization(tensor, checkpoint["normalization"])).numpy().reshape(-1)
                score_b = model(apply_normalization(tensor, checkpoint["normalization"])).numpy().reshape(-1)
            replay_rows = [dict(row, target=float(row["physical_target"]), score=float(score)) for row, score in zip(effective, score_a)]
            replay_detail, replay_traces = detailed_candidate_metrics(replay_rows, event_groups, selected)
            _, replay_traces_again = detailed_candidate_metrics(replay_rows, event_groups, selected)
            replay = {
                "schema": "D8_STAGE2_R2_CLEAN_REPLAY_V1",
                "checkpoint_restore": True,
                "score_determinism": bool(np.array_equal(score_a, score_b)),
                "scheduler_determinism": sha256_json(replay_traces) == sha256_json(replay_traces_again),
                "scheduler": dict(selected),
                "metrics": replay_detail,
                "clean_task_success": {"status": "PENDING_ONLINE_SMALL_MATRIX", "available": False},
                "eval160_reads": 0,
                "protected_eval_reads": 0,
                "attack_rollouts": 0,
            }
            atomic_json(staging / "CLEAN_REPLAY_R2.json", replay)
            atomic_json(
                staging / "DETECTOR_FREEZE_RECEIPT_R2.json",
                {
                    "schema": "D8_DETECTOR_FREEZE_RECEIPT_R2_V1",
                    "status": "PASS_CLEAN_ONLY_OFFLINE" if s2_pass else "SHADOW_PROBE_ONLY",
                    "source_commit": actual_commit,
                    "source_tree": actual_tree,
                    "checkpoint": "FINAL_DETECTOR_CHECKPOINT.pt",
                    "checkpoint_sha256": final_metrics["checkpoint_sha256"],
                    "scheduler": dict(selected),
                    "s1_verdict": "PASS",
                    "s2_verdict": "PASS" if s2_pass else "FAIL",
                    "authorization_mode": authorization,
                    "guard_deployment_authorized": guard_deployment_authorized,
                    "clean_task_success": {"status": "PENDING_ONLINE_SMALL_MATRIX"},
                    "replay": "CLEAN_REPLAY_R2.json",
                    "provenance": common,
                    "eval160_reads": 0,
                    "protected_eval_reads": 0,
                    "attack_rollouts": 0,
                },
            )
        result = {
            "schema": "D8_STAGE2_R2_RESULT_V1",
            "stage": "Stage 2 R2 Detector build",
            "verdict": verdict,
            "scheduler_authorization_mode": authorization,
            "s1_pass": s1_pass,
            "s2_pass": s2_pass,
            "selected_candidate": selected,
            "final_checkpoint_trained": final_checkpoint_trained,
            "guard_deployment_authorized": guard_deployment_authorized,
            "clean_task_success": {"status": "PENDING_ONLINE_SMALL_MATRIX"},
            "stage3a_authorized_by_gate": bool(s1_pass),
            "stage3b_authorized_by_gate": bool(s2_pass),
            "stage3a_rollouts": 0,
            "stage3b_rollouts": 0,
            "stage4_rollouts": 0,
            "event_binding": event_binding_report,
            "safe_release_label_audit": safe_release,
            "provenance": common,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 0,
            "finish_utc": utc_now(),
        }
        if verdict == R2_VERDICTS["FAIL"]:
            result["stop_reason"] = "Stage 2 R2 S1 clean safety Gate failed; no attack is authorized."
        atomic_json(staging / "STAGE2_R2_RESULT.json", result)
        run_seal = seal_directory(staging)
        os.rename(staging, output_root)
        _write_goal_state(goal_root, output_root, run_seal, result, r1)
        print(json.dumps({"output_root": str(output_root), "verdict": verdict, "authorization": authorization, "seal": run_seal}, sort_keys=True))
        return 20 if verdict == R2_VERDICTS["FAIL"] else 0
    except Exception:
        for path in staging.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
        staging.rmdir()
        raise


def subprocess_git(*args: str) -> str:
    import subprocess

    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--formal-root", type=Path)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--r1-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--goal-root", type=Path)
    parser.add_argument("--expected-cache-seal", required=False)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-tree")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (args.cache_root, args.formal_root, args.sidecar_root, args.teacher_root, args.r1_root, args.output_root, args.goal_root, args.expected_cache_seal, args.expected_source_commit, args.expected_source_tree)
    if any(value is None for value in required):
        parser.error("all execution arguments are required unless --self-test is used")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
