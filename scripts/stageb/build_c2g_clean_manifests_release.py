#!/usr/bin/env python3
"""Build strict runtime-compatible C2g train/evaluation parent manifests.

Parent keys contain exactly five slash-separated components so they remain
compatible with the repository's established Stage-B worker parsers:

    suite/task_i/state_j/cohort/episode_k
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def deterministic_order(suite: str, task_index: int, state_ids: Sequence[int], seed: int) -> list[int]:
    return sorted(
        (int(value) for value in state_ids),
        key=lambda value: hashlib.sha256(
            f"C2G_PARENT_SELECTION|{seed}|{suite}|{task_index}|{value}".encode("utf-8")
        ).digest(),
    )


def task_count(suite_obj: Any) -> int:
    value = getattr(suite_obj, "n_tasks", None)
    if value is not None:
        return int(value)
    method = getattr(suite_obj, "get_num_tasks", None)
    if callable(method):
        return int(method())
    tasks = getattr(suite_obj, "tasks", None)
    if tasks is not None:
        return len(tasks)
    raise AttributeError("LIBERO suite exposes neither n_tasks, get_num_tasks, nor tasks")


def parent_key(suite: str, task_index: int, state_id: int, cohort: str, local_index: int) -> str:
    value = f"{suite}/task_{task_index}/state_{state_id}/{cohort}/episode_{local_index:03d}"
    if len(value.split("/")) != 5:
        raise AssertionError("parent key must contain exactly five components")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", action="append", choices=SUITES, default=[])
    parser.add_argument("--train-states-per-task", type=int, default=40)
    parser.add_argument("--eval-states-per-task", type=int, default=10)
    parser.add_argument("--max-tasks-per-suite", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_states_per_task <= 0 or args.eval_states_per_task <= 0:
        raise ValueError("train/eval states per task must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    from libero.libero import benchmark

    selected_suites = tuple(args.suite or SUITES)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    task_reports: list[dict[str, Any]] = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite in selected_suites:
        suite_obj = benchmark_dict[suite]()
        count = task_count(suite_obj)
        if args.max_tasks_per_suite > 0:
            count = min(count, args.max_tasks_per_suite)
        for task_index in range(count):
            states = suite_obj.get_task_init_states(task_index)
            available = len(states)
            required = args.train_states_per_task + args.eval_states_per_task
            if available < required:
                raise RuntimeError(
                    f"{suite} task {task_index} has {available} init states, requires {required}"
                )
            ordered = deterministic_order(suite, task_index, range(available), args.seed)
            train_ids = ordered[: args.train_states_per_task]
            eval_ids = ordered[args.train_states_per_task : required]
            for local_index, state_id in enumerate(train_ids):
                train_rows.append(
                    {
                        "parent_key": parent_key(
                            suite, task_index, state_id, "train", local_index
                        ),
                        "suite": suite,
                        "task_index": task_index,
                        "state_id": state_id,
                        "max_steps": args.max_steps,
                        "selection_seed": args.seed,
                        "cohort": "TRAIN_CLEAN",
                    }
                )
            for local_index, state_id in enumerate(eval_ids):
                eval_rows.append(
                    {
                        "parent_key": parent_key(
                            suite, task_index, state_id, "eval", local_index
                        ),
                        "suite": suite,
                        "task_index": task_index,
                        "state_id": state_id,
                        "eval_seed": args.seed + 100000 + task_index * 1000 + local_index,
                        "max_steps": args.max_steps,
                        "selection_seed": args.seed,
                        "cohort": "EVAL_PREREGISTERED",
                    }
                )
            task_reports.append(
                {
                    "suite": suite,
                    "task_index": task_index,
                    "available_init_states": available,
                    "train_state_ids": train_ids,
                    "eval_state_ids": eval_ids,
                }
            )

    train_identity = {(row["suite"], row["task_index"], row["state_id"]) for row in train_rows}
    eval_identity = {(row["suite"], row["task_index"], row["state_id"]) for row in eval_rows}
    overlap = sorted(train_identity & eval_identity)
    if overlap:
        raise RuntimeError(f"train/eval parent overlap: {overlap[:20]}")
    if len({row["parent_key"] for row in train_rows}) != len(train_rows):
        raise RuntimeError("duplicate training parent_key")
    if len({row["parent_key"] for row in eval_rows}) != len(eval_rows):
        raise RuntimeError("duplicate evaluation parent_key")
    if any(len(row["parent_key"].split("/")) != 5 for row in train_rows + eval_rows):
        raise RuntimeError("noncanonical parent key detected")

    output_dir = args.output_dir.resolve()
    train_path = output_dir / "c2g_train_clean_parents.jsonl"
    eval_path = output_dir / "c2g_eval_preregistered_parents.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    report = {
        "gate": "C2G_RELEASE_PARENT_MANIFESTS",
        "status": "PASS_C2G_RELEASE_PARENT_MANIFESTS",
        "parent_key_schema": "suite/task_i/state_j/cohort/episode_k",
        "selection_seed": args.seed,
        "suites": list(selected_suites),
        "train_parent_count": len(train_rows),
        "eval_parent_count": len(eval_rows),
        "train_eval_overlap_count": 0,
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "eval_manifest": str(eval_path),
        "eval_manifest_sha256": sha256_file(eval_path),
        "tasks": task_reports,
        "boundaries": {
            "uses_attack_outcomes": False,
            "rollouts_launched": 0,
            "gpu_jobs_launched": 0,
        },
    }
    report_path = output_dir / "c2g_clean_parent_manifest_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
