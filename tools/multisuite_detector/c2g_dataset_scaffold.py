"""Pure-CPU split, context, weighting, triggerability, and diagnostic helpers for C2g."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

CONTEXT_MODES = ("no_context", "suite_only", "full_context_legacy")
SPLIT_MODES = ("within-task", "leave-one-task-out", "leave-one-suite-out")
DIAGNOSTICS = ("shuffled-language", "permuted-task-context", "wrong-language-cross-task")


def context_feature_names(mode: str, available: Sequence[str]) -> List[str]:
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode: {mode}")
    if mode == "no_context":
        return []
    suite = [name for name in available if name.startswith(("ctx_suite_", "suite_onehot_"))]
    if mode == "suite_only":
        return sorted(suite)
    return sorted(name for name in available if name.startswith(("ctx_", "suite_onehot_")))


def _bucket(material: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % modulus


def assign_episode_splits(
    rows: Sequence[Dict[str, Any]],
    mode: str,
    *,
    seed: int = 0,
    held_out_task: str = "",
    held_out_suite: str = "",
) -> Dict[str, str]:
    if mode not in SPLIT_MODES:
        raise ValueError(f"unknown split mode: {mode}")
    episodes: Dict[str, Dict[str, str]] = {}
    for row in rows:
        episode = str(row["episode_key"])
        identity = {"suite": str(row["suite"]), "task": str(row["task_index"])}
        if episode in episodes and episodes[episode] != identity:
            raise ValueError(f"episode identity changed: {episode}")
        episodes[episode] = identity
    out: Dict[str, str] = {}
    if mode == "within-task":
        by_task: Dict[str, List[str]] = defaultdict(list)
        for episode, identity in episodes.items():
            by_task[f"{identity['suite']}:{identity['task']}"].append(episode)
        for task, task_episodes in sorted(by_task.items()):
            ordered = sorted(task_episodes, key=lambda episode: _bucket(f"{seed}|{task}|{episode}", 2**64))
            n = len(ordered)
            n_val = max(1, round(n * 0.1)) if n >= 3 else 0
            n_test = max(1, round(n * 0.1)) if n >= 3 else 0
            for index, episode in enumerate(ordered):
                out[episode] = "test" if index >= n - n_test else "val" if index >= n - n_test - n_val else "train"
        return out
    for episode, identity in sorted(episodes.items()):
        suite, task = identity["suite"], identity["task"]
        material = f"{seed}|{suite}|{task}|{episode}"
        if mode == "leave-one-task-out":
            if not held_out_task:
                raise ValueError("held_out_task is required")
            out[episode] = "test" if f"{suite}:{task}" == held_out_task else ("train" if _bucket(material, 5) else "val")
        else:
            if not held_out_suite:
                raise ValueError("held_out_suite is required")
            out[episode] = "test" if suite == held_out_suite else ("train" if _bucket(material, 5) else "val")
    return out


def assert_no_episode_leakage(rows: Iterable[Dict[str, Any]]) -> None:
    seen: Dict[str, str] = {}
    for row in rows:
        episode, split = str(row["episode_key"]), str(row["split"])
        if episode in seen and seen[episode] != split:
            raise ValueError(f"episode leakage: {episode} in {seen[episode]} and {split}")
        seen[episode] = split


def _positive_intervals(sequence: Sequence[tuple[int, bool, bool]]) -> list[list[int]]:
    intervals: list[list[int]] = []
    current: list[int] = []
    for step, known, positive in sorted(sequence):
        if known and positive:
            if current and step == current[-1] + 1:
                current.append(step)
            else:
                if current:
                    intervals.append(current)
                current = [step]
        elif current:
            intervals.append(current)
            current = []
    if current:
        intervals.append(current)
    return intervals


def _persistent_positive_windows(
    sequence: Sequence[tuple[int, bool, bool]], *, window: int = 3, required: int = 2,
) -> int:
    if required < 1 or window < required:
        raise ValueError("persistence requires 1 <= required <= window")
    ordered = sorted(sequence)
    count = 0
    for start in range(len(ordered)):
        start_step = ordered[start][0]
        eligible = 0
        for step, known, positive in ordered[start:]:
            if step - start_step >= window:
                break
            eligible += int(known and positive)
        count += int(eligible >= required)
    return count


def split_label_coverage(
    rows: Sequence[Dict[str, Any]],
    *,
    split_key: str = "split",
    known_key: str = "label_known_mask",
    label_key: str = "y_cmdopen_vulnerable",
    step_key: str = "step",
    persistence_window: int = 3,
    persistence_required: int = 2,
) -> Dict[str, Dict[str, int]]:
    """Summarize fold viability without converting unknown labels to negatives."""
    coverage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "rows": 0,
        "episodes": set(),
        "tasks": set(),
        "suites": set(),
        "episode_labels": {},
        "known_positive": 0,
        "known_negative": 0,
        "unknown": 0,
    })
    for index, row in enumerate(rows):
        split = str(row[split_key])
        bucket = coverage[split]
        bucket["rows"] += 1
        episode = str(row["episode_key"])
        bucket["episodes"].add(episode)
        bucket["tasks"].add(f"{row['suite']}:{row['task_index']}")
        bucket["suites"].add(str(row["suite"]))
        known = bool(row.get(known_key, False))
        positive = bool(row.get(label_key, False)) if known else False
        state = bucket["episode_labels"].setdefault(episode, {
            "all_known": True,
            "positive": False,
            "sequence": [],
        })
        state["all_known"] = bool(state["all_known"] and known)
        state["positive"] = bool(state["positive"] or positive)
        state["sequence"].append((int(row.get(step_key, index)), known, positive))
        if not known:
            bucket["unknown"] += 1
        elif positive:
            bucket["known_positive"] += 1
        else:
            bucket["known_negative"] += 1

    result: Dict[str, Dict[str, int]] = {}
    for split, values in sorted(coverage.items()):
        episode_states = values["episode_labels"]
        interval_lengths: list[int] = []
        triggerable = 0
        persistent_window_count = 0
        for state in episode_states.values():
            intervals = _positive_intervals(state["sequence"])
            interval_lengths.extend(len(interval) for interval in intervals)
            windows = _persistent_positive_windows(
                state["sequence"], window=persistence_window, required=persistence_required,
            )
            persistent_window_count += windows
            triggerable += int(state["positive"] and windows > 0)
        attackable = sum(1 for state in episode_states.values() if state["positive"])
        result[split] = {
            "rows": int(values["rows"]),
            "episodes": len(values["episodes"]),
            "tasks": len(values["tasks"]),
            "suites": len(values["suites"]),
            "attackable_episodes": attackable,
            "triggerable_attackable_episodes": triggerable,
            "untriggerable_positive_episodes": attackable - triggerable,
            "fully_known_negative_episodes": sum(
                1 for state in episode_states.values() if state["all_known"] and not state["positive"]
            ),
            "positive_interval_count": len(interval_lengths),
            "max_positive_interval_length": max(interval_lengths, default=0),
            "persistent_positive_window_count": persistent_window_count,
            "known_positive": int(values["known_positive"]),
            "known_negative": int(values["known_negative"]),
            "unknown": int(values["unknown"]),
        }
    return result


def assert_split_viability(
    coverage: Mapping[str, Mapping[str, int]],
    *,
    required_splits: Sequence[str] = ("train", "val", "test"),
    min_episodes: int = 1,
    min_known_positive: int = 1,
    min_known_negative: int = 1,
    min_tasks: int = 1,
    min_suites: int = 1,
    min_attackable_episodes: int = 1,
    min_fully_known_negative_episodes: int = 1,
    min_triggerable_attackable_episodes: int = 0,
) -> None:
    """Hard-gate folds that cannot support training, calibration, or evaluation."""
    problems: List[str] = []
    for split in required_splits:
        values = coverage.get(split)
        if values is None:
            problems.append(f"{split}:missing")
            continue
        checks = {
            "episodes": min_episodes,
            "known_positive": min_known_positive,
            "known_negative": min_known_negative,
            "tasks": min_tasks,
            "suites": min_suites,
            "attackable_episodes": min_attackable_episodes,
            "fully_known_negative_episodes": min_fully_known_negative_episodes,
            "triggerable_attackable_episodes": min_triggerable_attackable_episodes,
        }
        for key, minimum in checks.items():
            if int(values.get(key, 0)) < minimum:
                problems.append(f"{split}:{key}<{minimum}")
    if problems:
        raise ValueError("non-viable C2g split: " + ", ".join(problems))


def task_episode_balanced_weights(rows: Sequence[Dict[str, Any]]) -> List[float]:
    episode_rows = Counter(str(row["episode_key"]) for row in rows)
    episode_task: Dict[str, str] = {}
    task_episodes: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        episode = str(row["episode_key"])
        task = f"{row['suite']}:{row['task_index']}"
        if episode in episode_task and episode_task[episode] != task:
            raise ValueError(f"episode task changed: {episode}")
        episode_task[episode] = task
        task_episodes[task].add(episode)
    n_tasks = max(1, len(task_episodes))
    raw = [
        1.0 / (n_tasks * len(task_episodes[episode_task[str(row["episode_key"])]]) * episode_rows[str(row["episode_key"])])
        for row in rows
    ]
    scale = len(raw) / sum(raw) if raw else 1.0
    return [value * scale for value in raw]


def diagnostic_episode_permutation(rows: Sequence[Dict[str, Any]], *, seed: int, diagnostic: str) -> Dict[str, str]:
    if diagnostic not in DIAGNOSTICS:
        raise ValueError(f"unknown diagnostic: {diagnostic}")
    by_split: Dict[str, List[str]] = defaultdict(list)
    identity: Dict[str, str] = {}
    for row in rows:
        episode = str(row["episode_key"])
        split = str(row["split"])
        task = f"{row['suite']}:{row['task_index']}"
        if episode in identity and identity[episode] != task:
            raise ValueError(f"episode task changed: {episode}")
        identity[episode] = task
        if episode not in by_split[split]:
            by_split[split].append(episode)
    out: Dict[str, str] = {}
    for split, episodes in sorted(by_split.items()):
        order = sorted(episodes)
        if diagnostic == "wrong-language-cross-task":
            for source in order:
                candidates = [candidate for candidate in order if identity[candidate] != identity[source]]
                if not candidates:
                    raise ValueError(f"split {split} has no cross-task language donor for {source}")
                donor = min(candidates, key=lambda candidate: _bucket(f"{seed}|{diagnostic}|{split}|{source}|{candidate}", 2**64))
                out[source] = donor
            continue
        random.Random(f"{seed}|{diagnostic}|{split}").shuffle(order)
        donors = order[1:] + order[:1] if len(order) > 1 else order
        out.update(dict(zip(order, donors)))
    return out
