#!/usr/bin/env python3
"""Build a frozen 24-parent, train-only R8T Teacher-v2 collection canary.

The planner consumes the accepted R7 registry plus the accepted R8S semantic/replay
report.  It never redraws identities and never selects validation, clean-test, or
attack-evaluation parents.  The resulting four suite-local shards are planning
artifacts only; execution still requires the exact R8T authorization token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    DETECTOR_TRAIN,
    PASS_STATUS as R7_PASS,
    SCHEMA as R7_SCHEMA,
    SUITES,
)

SCHEMA = "c2g.r8t.teacher_v2_canary_plan.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R8T_TEACHER_V2_CANARY_PLAN"
AUTHORIZATION_TOKEN = "R8T_TEACHER_V2_CANARY_COLLECTION_AUTHORIZED"
EXPECTED_R8S_DECISION = "GO_AUXILIARY_LEGACY_SUPERVISION_ONLY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def assert_hash(path: Path, expected: str, name: str) -> str:
    expected = require_sha256(expected, name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must contain an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_index"]), int(row["state_id"])


def stable_rank(seed: int, *parts: object) -> bytes:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def assert_external_new_output(output_dir: Path, repo: Path) -> Path:
    output_dir = output_dir.resolve()
    repo = repo.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == repo or repo in output_dir.parents:
        raise ValueError("R8T plan output must be outside the repository")
    return output_dir


def build_plan(
    *,
    repo: Path,
    expected_git_commit: str,
    registry_path: Path,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    reusable_manifest_path: Path,
    expected_reusable_manifest_sha256: str,
    r8s_report_path: Path,
    expected_r8s_report_sha256: str,
    output_dir: Path,
    tasks_per_suite: int = 2,
    states_per_task: int = 3,
    selection_seed: int = 20260711,
) -> dict[str, Any]:
    output_dir = assert_external_new_output(output_dir, repo)
    expected_git_commit = str(expected_git_commit).strip().lower()
    if len(expected_git_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_git_commit):
        raise ValueError("expected_git_commit must be a 40-character lowercase SHA")
    if tasks_per_suite <= 0 or states_per_task <= 0:
        raise ValueError("tasks_per_suite and states_per_task must be positive")

    registry_path = registry_path.resolve()
    plan_report_path = plan_report_path.resolve()
    reusable_manifest_path = reusable_manifest_path.resolve()
    r8s_report_path = r8s_report_path.resolve()
    for path in (registry_path, plan_report_path, reusable_manifest_path, r8s_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    assert_hash(plan_report_path, expected_plan_report_sha256, "R7 plan report")
    assert_hash(reusable_manifest_path, expected_reusable_manifest_sha256, "R7 reusable manifest")
    assert_hash(r8s_report_path, expected_r8s_report_sha256, "R8S report")

    r7_plan = read_json(plan_report_path)
    if r7_plan.get("schema") != R7_SCHEMA or r7_plan.get("status") != R7_PASS:
        raise ValueError("R7 plan is not accepted")
    if Path(str(r7_plan.get("registry", ""))).resolve() != registry_path:
        raise ValueError("R7 plan binds another registry")
    if str(r7_plan.get("registry_sha256", "")) != sha256_file(registry_path):
        raise ValueError("R7 registry hash differs from plan")

    r8s = read_json(r8s_report_path)
    if r8s.get("final_decision") != EXPECTED_R8S_DECISION:
        raise ValueError("R8T requires the accepted R8S auxiliary-only decision")
    if int(r8s.get("episode_count", -1)) != 2000:
        raise ValueError("R8S episode cardinality is not 2,000")
    if int(r8s.get("strict_replay_ready_count", -1)) != 0:
        raise ValueError("R8T collection is not justified while replay-ready episodes exist")
    if int(r8s.get("exact_equivalent_mapping_count", -1)) != 0:
        raise ValueError("R8S unexpectedly claims exact Teacher-v2 equivalence")

    registry = read_jsonl(registry_path)
    reusable = read_jsonl(reusable_manifest_path)
    reusable_ids = {identity(row) for row in reusable}
    registry_ids: set[tuple[str, int, int]] = set()
    train_by_suite_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(registry):
        row = dict(raw)
        key = identity(row)
        if key in registry_ids:
            raise ValueError(f"duplicate R7 identity: {key}")
        registry_ids.add(key)
        if row.get("cohort") == DETECTOR_TRAIN and key not in reusable_ids:
            row["registry_index"] = index
            train_by_suite_task[(key[0], key[1])].append(row)

    selected: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for suite in SUITES:
        eligible_tasks = sorted(
            [
                task
                for (row_suite, task), members in train_by_suite_task.items()
                if row_suite == suite and len(members) >= states_per_task
            ],
            key=lambda task: stable_rank(selection_seed, "task", suite, task),
        )
        chosen_tasks = eligible_tasks[:tasks_per_suite]
        if len(chosen_tasks) != tasks_per_suite:
            raise ValueError(f"{suite} lacks enough train-only canary tasks")
        suite_rows: list[dict[str, Any]] = []
        for task in chosen_tasks:
            candidates = sorted(
                train_by_suite_task[(suite, task)],
                key=lambda row: stable_rank(selection_seed, "state", row["parent_key"]),
            )[:states_per_task]
            if len(candidates) != states_per_task:
                raise ValueError(f"{suite}/task_{task} lacks enough train states")
            suite_rows.extend(dict(row) for row in candidates)
        selected.extend(suite_rows)
        selection[suite] = {
            "eligible_task_count": len(eligible_tasks),
            "selected_task_indices": chosen_tasks,
            "selected_parent_count": len(suite_rows),
        }

    selected.sort(key=lambda row: (SUITES.index(str(row["suite"])), int(row["task_index"]), int(row["state_id"])))
    expected_count = len(SUITES) * tasks_per_suite * states_per_task
    if len(selected) != expected_count:
        raise AssertionError("R8T canary cardinality mismatch")
    if any(row.get("cohort") != DETECTOR_TRAIN or row.get("split") != "train" for row in selected):
        raise AssertionError("R8T canary contains non-train parent")
    if len({identity(row) for row in selected}) != len(selected):
        raise AssertionError("R8T canary contains duplicate identity")
    if {row["suite"] for row in selected} != set(SUITES):
        raise AssertionError("R8T canary does not cover all suites")

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "c2g_r8t_teacher_v2_canary.jsonl"
    write_jsonl(manifest_path, selected)
    shards: list[dict[str, Any]] = []
    for suite in SUITES:
        rows = [row for row in selected if row["suite"] == suite]
        shard_path = output_dir / "shards" / suite / "shard_000.jsonl"
        write_jsonl(shard_path, rows)
        shards.append({
            "shard_id": f"r8t_teacher_v2_canary__{suite}__000",
            "suite": suite,
            "episode_count": len(rows),
            "manifest": str(shard_path.resolve()),
            "manifest_sha256": sha256_file(shard_path),
            "cohort_counts": dict(Counter(str(row["cohort"]) for row in rows)),
            "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        })
    shard_index = output_dir / "c2g_r8t_teacher_v2_canary_shards.jsonl"
    write_jsonl(shard_index, shards)

    report_path = output_dir / "c2g_r8t_teacher_v2_canary_plan.json"
    report = {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "expected_git_commit": expected_git_commit,
        "r7_registry": str(registry_path),
        "r7_registry_sha256": sha256_file(registry_path),
        "r7_plan_report": str(plan_report_path),
        "r7_plan_report_sha256": sha256_file(plan_report_path),
        "r7_reusable_manifest": str(reusable_manifest_path),
        "r7_reusable_manifest_sha256": sha256_file(reusable_manifest_path),
        "r8s_report": str(r8s_report_path),
        "r8s_report_sha256": sha256_file(r8s_report_path),
        "selection_seed": selection_seed,
        "tasks_per_suite": tasks_per_suite,
        "states_per_task": states_per_task,
        "episode_count": len(selected),
        "selection": selection,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "shard_index": str(shard_index.resolve()),
        "shard_index_sha256": sha256_file(shard_index),
        "shards": shards,
        "authorization_token": AUTHORIZATION_TOKEN,
        "collection_authorization": "AUTHORIZED_BY_USER_FOR_R8T_24EP_TRAIN_ONLY_CANARY",
        "post_canary_collection_authorization": "HOLD_PENDING_R8T_CANARY_AUDIT",
        "training_authorization": "HOLD_PENDING_R8T_CANARY_AND_CORPUS_AUDIT",
        "invariants": {
            "train_only": True,
            "validation_parent_count": 0,
            "clean_test_parent_count": 0,
            "attack_eval_parent_count": 0,
            "suite_count": len(SUITES),
            "episode_cardinality_closed": len(selected) == expected_count,
        },
        "boundaries": {
            "registry_redrawn": False,
            "attack_outcomes_read": False,
            "models_loaded": 0,
            "environments_created": 0,
            "rollouts_launched": 0,
            "training_epochs": 0,
        },
    }
    write_json(report_path, report)
    return {**report, "report": str(report_path.resolve()), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--reusable-manifest", type=Path, required=True)
    parser.add_argument("--expected-reusable-manifest-sha256", required=True)
    parser.add_argument("--r8s-report", type=Path, required=True)
    parser.add_argument("--expected-r8s-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-suite", type=int, default=2)
    parser.add_argument("--states-per-task", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=20260711)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_plan(
        repo=args.repo,
        expected_git_commit=args.expected_git_commit,
        registry_path=args.registry,
        plan_report_path=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        reusable_manifest_path=args.reusable_manifest,
        expected_reusable_manifest_sha256=args.expected_reusable_manifest_sha256,
        r8s_report_path=args.r8s_report,
        expected_r8s_report_sha256=args.expected_r8s_report_sha256,
        output_dir=args.output_dir,
        tasks_per_suite=args.tasks_per_suite,
        states_per_task=args.states_per_task,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
