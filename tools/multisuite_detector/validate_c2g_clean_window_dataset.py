#!/usr/bin/env python3
"""Fail-closed trainability and leakage audit for a materialized C2g dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.multisuite_detector.train_c2g_clean_window_detector import load_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persistence_support(values: np.ndarray, known: np.ndarray, window: int, required: int) -> int:
    count = 0
    for index in range(window - 1, len(values)):
        local_values = values[index - window + 1 : index + 1]
        local_known = known[index - window + 1 : index + 1]
        if int(np.sum(local_values & local_known)) >= required:
            count += 1
    return count


def audit_dataset(
    data: Mapping[str, np.ndarray],
    *,
    persistence_window: int = 3,
    persistence_required: int = 2,
    require_test_support: bool = True,
) -> dict[str, Any]:
    if persistence_required < 1 or persistence_window < persistence_required:
        raise ValueError("invalid persistence configuration")
    splits = data["split"].astype(str)
    episodes = data["episode_key"].astype(str)
    steps = data["step"].astype(np.int64)
    current_target = data["y_critical_window"][:, -1] > 0.5
    current_known = data["m_critical_window"][:, -1].astype(bool)
    suites = data["suite"].astype(str)
    tasks = data["task_index"].astype(np.int64)

    episode_splits: dict[str, set[str]] = defaultdict(set)
    for episode, split in zip(episodes, splits):
        episode_splits[episode].add(split)
    leakage = {episode: sorted(values) for episode, values in episode_splits.items() if len(values) != 1}

    split_reports: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        indices = np.flatnonzero(splits == split_name)
        known = current_known[indices]
        target = current_target[indices]
        positive = int(np.sum(known & target))
        negative = int(np.sum(known & ~target))
        episode_keys = sorted(set(episodes[indices].tolist()))
        triggerable_episodes = 0
        positive_episodes = 0
        persistent_windows = 0
        for episode in episode_keys:
            local = indices[episodes[indices] == episode]
            local = local[np.argsort(steps[local])]
            values = current_target[local]
            known_values = current_known[local]
            if bool(np.any(values & known_values)):
                positive_episodes += 1
            support = persistence_support(
                values,
                known_values,
                persistence_window,
                persistence_required,
            )
            persistent_windows += support
            triggerable_episodes += int(support > 0)
        report = {
            "sample_count": int(indices.size),
            "episode_count": len(episode_keys),
            "known_positive_count": positive,
            "known_negative_count": negative,
            "unknown_count": int(indices.size - int(np.sum(known))),
            "positive_episode_count": positive_episodes,
            "triggerable_positive_episode_count": triggerable_episodes,
            "persistent_positive_window_count": persistent_windows,
            "suite_count": len(set(suites[indices].tolist())),
            "task_count": len(set(zip(suites[indices].tolist(), tasks[indices].tolist()))),
        }
        split_reports[split_name] = report
        support_required = split_name in {"train", "val"} or require_test_support
        if support_required:
            for field in (
                "sample_count",
                "episode_count",
                "known_positive_count",
                "known_negative_count",
                "triggerable_positive_episode_count",
            ):
                if int(report[field]) <= 0:
                    violations.append(
                        {
                            "split": split_name,
                            "reason": "INSUFFICIENT_SPLIT_SUPPORT",
                            "field": field,
                            "value": report[field],
                        }
                    )

    if leakage:
        violations.append(
            {
                "reason": "EPISODE_SPLIT_LEAKAGE",
                "episode_count": len(leakage),
                "examples": dict(list(leakage.items())[:20]),
            }
        )
    return {
        "gate": "C2G_CLEAN_WINDOW_DATASET_TRAINABILITY",
        "status": "PASS_C2G_DATASET_TRAINABILITY" if not violations else "HOLD_C2G_DATASET_TRAINABILITY",
        "sample_count": int(len(splits)),
        "episode_count": len(episode_splits),
        "split_reports": split_reports,
        "episode_split_leakage_count": len(leakage),
        "violation_count": len(violations),
        "violations": violations,
        "persistence_window": persistence_window,
        "persistence_required": persistence_required,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--persistence-window", type=int, default=3)
    parser.add_argument("--persistence-required", type=int, default=2)
    parser.add_argument("--require-test-support", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    data = load_dataset(args.dataset.resolve())
    report = audit_dataset(
        data,
        persistence_window=args.persistence_window,
        persistence_required=args.persistence_required,
        require_test_support=args.require_test_support,
    )
    report["dataset"] = str(args.dataset.resolve())
    report["dataset_sha256"] = sha256_file(args.dataset.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
