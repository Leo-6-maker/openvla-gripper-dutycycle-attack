"""Pure-CPU split, context, weighting, and diagnostic helpers for C2g."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence

CONTEXT_MODES = ("no_context", "suite_only", "full_context_legacy")
SPLIT_MODES = ("within-task", "leave-one-task-out", "leave-one-suite-out")


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
    if diagnostic not in {"shuffled-language", "permuted-task-context"}:
        raise ValueError(f"unknown diagnostic: {diagnostic}")
    by_split: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        episode = str(row["episode_key"])
        split = str(row["split"])
        if episode not in by_split[split]:
            by_split[split].append(episode)
    out: Dict[str, str] = {}
    for split, episodes in sorted(by_split.items()):
        order = sorted(episodes)
        random.Random(f"{seed}|{diagnostic}|{split}").shuffle(order)
        donors = order[1:] + order[:1] if len(order) > 1 else order
        out.update(dict(zip(order, donors)))
    return out
