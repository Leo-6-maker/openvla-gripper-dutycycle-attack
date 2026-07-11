#!/usr/bin/env python3
"""Plan a leakage-safe Detector-v2 clean corpus without launching rollouts.

The planner freezes one parent registry over official LIBERO init states.  The
same registry supports three distinct scientific questions without resampling:

* pooled within-task generalization;
* leave-one-task-out (LOTO) generalization;
* leave-one-suite-out (LOSO) generalization.

The default allocation consumes the 50 official states of every LIBERO task as
30 detector-train, 5 detector-validation, 5 within-task clean-test, and 10
preregistered downstream attack-evaluation parents.  Attack-evaluation parents
are never eligible for detector fitting, checkpoint selection, or threshold
calibration.

This program imports LIBERO benchmark metadata only when ``--from-libero`` is
used.  It never creates an environment, loads an OpenVLA model, launches a
rollout, reads attack outcomes, or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
SCHEMA = "c2g.r7.scientific_corpus_plan.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R7_SCIENTIFIC_CORPUS_PLAN"

DETECTOR_TRAIN = "DETECTOR_TRAIN"
DETECTOR_VAL = "DETECTOR_VAL"
DETECTOR_TEST = "DETECTOR_TEST_WITHIN_TASK"
ATTACK_EVAL = "ATTACK_EVAL_PREREGISTERED"
COHORTS = (DETECTOR_TRAIN, DETECTOR_VAL, DETECTOR_TEST, ATTACK_EVAL)
COHORT_TO_SPLIT = {
    DETECTOR_TRAIN: "train",
    DETECTOR_VAL: "val",
    DETECTOR_TEST: "test",
    ATTACK_EVAL: "attack_eval",
}
COHORT_SLUG = {
    DETECTOR_TRAIN: "detector_train",
    DETECTOR_VAL: "detector_val",
    DETECTOR_TEST: "detector_test",
    ATTACK_EVAL: "attack_eval",
}


@dataclass(frozen=True)
class CohortCounts:
    train: int = 30
    val: int = 5
    test: int = 5
    attack_eval: int = 10

    @property
    def total(self) -> int:
        return self.train + self.val + self.test + self.attack_eval

    @property
    def detector_total(self) -> int:
        return self.train + self.val + self.test

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if int(value) <= 0:
                raise ValueError(f"{name} states per task must be positive")

    def by_cohort(self) -> dict[str, int]:
        return {
            DETECTOR_TRAIN: self.train,
            DETECTOR_VAL: self.val,
            DETECTOR_TEST: self.test,
            ATTACK_EVAL: self.attack_eval,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def deterministic_order(
    suite: str,
    task_index: int,
    state_ids: Sequence[int],
    seed: int,
) -> list[int]:
    return sorted(
        (int(value) for value in state_ids),
        key=lambda value: hashlib.sha256(
            f"C2G_R7_PARENT_SELECTION|{seed}|{suite}|{task_index}|{value}".encode(
                "utf-8"
            )
        ).digest(),
    )


def parent_key(
    suite: str,
    task_index: int,
    state_id: int,
    cohort: str,
    local_index: int,
) -> str:
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort: {cohort}")
    value = (
        f"{suite}/task_{task_index}/state_{state_id}/"
        f"{COHORT_SLUG[cohort]}/episode_{local_index:03d}"
    )
    if len(value.split("/")) != 5:
        raise AssertionError("parent key must contain exactly five components")
    return value


def _task_count(suite_obj: Any) -> int:
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


def official_libero_inventory() -> list[dict[str, Any]]:
    """Read official task/state cardinalities without creating environments."""

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    rows: list[dict[str, Any]] = []
    for suite in SUITES:
        suite_obj = benchmark_dict[suite]()
        for task_index in range(_task_count(suite_obj)):
            states = suite_obj.get_task_init_states(task_index)
            rows.append(
                {
                    "suite": suite,
                    "task_index": task_index,
                    "state_ids": list(range(len(states))),
                }
            )
    return rows


def load_inventory(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("tasks")
    if not isinstance(value, list):
        raise ValueError("inventory JSON must be a list or an object with a tasks list")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"inventory row {index} is not an object")
        suite = str(raw.get("suite", ""))
        task_index = int(raw.get("task_index", -1))
        if suite not in SUITES or task_index < 0:
            raise ValueError(f"invalid inventory identity at row {index}")
        if "state_ids" in raw:
            state_ids = [int(item) for item in raw["state_ids"]]
        else:
            count = int(raw.get("state_count", raw.get("available_init_states", -1)))
            if count < 0:
                raise ValueError(f"inventory row {index} lacks state_ids/state_count")
            state_ids = list(range(count))
        if len(state_ids) != len(set(state_ids)) or any(item < 0 for item in state_ids):
            raise ValueError(f"invalid state IDs at inventory row {index}")
        rows.append(
            {"suite": suite, "task_index": task_index, "state_ids": sorted(state_ids)}
        )
    return rows


def normalize_inventory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identities: set[tuple[str, int]] = set()
    output: list[dict[str, Any]] = []
    for raw in rows:
        suite = str(raw["suite"])
        task_index = int(raw["task_index"])
        state_ids = [int(value) for value in raw["state_ids"]]
        identity = (suite, task_index)
        if suite not in SUITES or task_index < 0:
            raise ValueError(f"invalid task identity: {identity}")
        if identity in identities:
            raise ValueError(f"duplicate inventory task: {identity}")
        identities.add(identity)
        if len(state_ids) != len(set(state_ids)) or any(value < 0 for value in state_ids):
            raise ValueError(f"invalid state IDs for {identity}")
        output.append(
            {"suite": suite, "task_index": task_index, "state_ids": sorted(state_ids)}
        )
    output.sort(key=lambda row: (SUITES.index(row["suite"]), row["task_index"]))
    present_suites = {row["suite"] for row in output}
    if present_suites != set(SUITES):
        raise ValueError(
            f"inventory suite closure mismatch: {sorted(present_suites)} != {list(SUITES)}"
        )
    return output


def build_registry(
    inventory: Sequence[Mapping[str, Any]],
    *,
    counts: CohortCounts,
    seed: int,
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts.validate()
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    rows: list[dict[str, Any]] = []
    task_reports: list[dict[str, Any]] = []
    for task in normalize_inventory(inventory):
        suite = str(task["suite"])
        task_index = int(task["task_index"])
        state_ids = list(task["state_ids"])
        if len(state_ids) < counts.total:
            raise ValueError(
                f"{suite} task {task_index} has {len(state_ids)} states; "
                f"requires {counts.total}"
            )
        ordered = deterministic_order(suite, task_index, state_ids, seed)
        cursor = 0
        selected: dict[str, list[int]] = {}
        for cohort, count in counts.by_cohort().items():
            cohort_ids = ordered[cursor : cursor + count]
            cursor += count
            selected[cohort] = cohort_ids
            for local_index, state_id in enumerate(cohort_ids):
                rows.append(
                    {
                        "parent_key": parent_key(
                            suite, task_index, state_id, cohort, local_index
                        ),
                        "suite": suite,
                        "task_index": task_index,
                        "state_id": state_id,
                        "cohort": cohort,
                        "split": COHORT_TO_SPLIT[cohort],
                        "max_steps": max_steps,
                        "selection_seed": seed,
                        "eligible_for_detector_fit": cohort == DETECTOR_TRAIN,
                        "eligible_for_checkpoint_selection": cohort == DETECTOR_VAL,
                        "eligible_for_threshold_calibration": cohort == DETECTOR_VAL,
                        "eligible_for_clean_test": cohort == DETECTOR_TEST,
                        "eligible_for_attack_evaluation": cohort == ATTACK_EVAL,
                    }
                )
        task_reports.append(
            {
                "suite": suite,
                "task_index": task_index,
                "available_state_count": len(state_ids),
                "selected_state_ids": selected,
                "unused_state_ids": ordered[counts.total :],
            }
        )

    identities = [(row["suite"], row["task_index"], row["state_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("state identity appears in more than one cohort")
    keys = [str(row["parent_key"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate parent_key")
    return rows, task_reports


def build_fold_plans(
    registry: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detector_rows = [row for row in registry if row["cohort"] != ATTACK_EVAL]
    attack_eval_count = sum(row["cohort"] == ATTACK_EVAL for row in registry)
    task_ids = sorted(
        {(str(row["suite"]), int(row["task_index"])) for row in registry},
        key=lambda item: (SUITES.index(item[0]), item[1]),
    )
    loto: list[dict[str, Any]] = []
    for suite, task_index in task_ids:
        train = sum(
            row["cohort"] == DETECTOR_TRAIN
            and (row["suite"], row["task_index"]) != (suite, task_index)
            for row in detector_rows
        )
        val = sum(
            row["cohort"] == DETECTOR_VAL
            and (row["suite"], row["task_index"]) != (suite, task_index)
            for row in detector_rows
        )
        test = sum(
            (row["suite"], row["task_index"]) == (suite, task_index)
            for row in detector_rows
        )
        loto.append(
            {
                "fold_id": f"loto__{suite}__task_{task_index}",
                "held_out_suite": suite,
                "held_out_task_index": task_index,
                "train_rule": "DETECTOR_TRAIN excluding held-out task",
                "val_rule": "DETECTOR_VAL excluding held-out task",
                "test_rule": "all detector cohorts from held-out task",
                "train_episode_count": train,
                "val_episode_count": val,
                "test_episode_count": test,
                "excluded_attack_eval_episode_count": attack_eval_count,
            }
        )

    loso: list[dict[str, Any]] = []
    for held_out_suite in SUITES:
        train = sum(
            row["cohort"] == DETECTOR_TRAIN and row["suite"] != held_out_suite
            for row in detector_rows
        )
        val = sum(
            row["cohort"] == DETECTOR_VAL and row["suite"] != held_out_suite
            for row in detector_rows
        )
        test = sum(row["suite"] == held_out_suite for row in detector_rows)
        loso.append(
            {
                "fold_id": f"loso__{held_out_suite}",
                "held_out_suite": held_out_suite,
                "train_rule": "DETECTOR_TRAIN from non-held suites",
                "val_rule": "DETECTOR_VAL from non-held suites",
                "test_rule": "all detector cohorts from held-out suite",
                "train_episode_count": train,
                "val_episode_count": val,
                "test_episode_count": test,
                "excluded_attack_eval_episode_count": attack_eval_count,
            }
        )
    return loto, loso


def summarize_registry(registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cohort = Counter(str(row["cohort"]) for row in registry)
    by_split = Counter(str(row["split"]) for row in registry)
    by_suite: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        suite_rows = [row for row in registry if row["suite"] == suite]
        by_suite[suite] = {
            "episode_count": len(suite_rows),
            "task_count": len({int(row["task_index"]) for row in suite_rows}),
            "cohort_counts": dict(
                sorted(Counter(str(row["cohort"]) for row in suite_rows).items())
            ),
            "split_counts": dict(
                sorted(Counter(str(row["split"]) for row in suite_rows).items())
            ),
        }
    return {
        "episode_count": len(registry),
        "task_count": len(
            {(str(row["suite"]), int(row["task_index"])) for row in registry}
        ),
        "suite_count": len({str(row["suite"]) for row in registry}),
        "cohort_counts": dict(sorted(by_cohort.items())),
        "split_counts": dict(sorted(by_split.items())),
        "per_suite": by_suite,
    }


def _assert_external_new_output(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.exists():
        raise FileExistsError(f"output directory already exists: {resolved}")
    repo = REPO.resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError("runtime corpus plan output must be outside the repository")
    return resolved


def materialize_plan(
    inventory: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    counts: CohortCounts,
    seed: int,
    max_steps: int,
    expected_git_commit: str,
    inventory_source: str,
) -> dict[str, Any]:
    output_dir = _assert_external_new_output(output_dir)
    registry, task_reports = build_registry(
        inventory, counts=counts, seed=seed, max_steps=max_steps
    )
    loto, loso = build_fold_plans(registry)
    output_dir.mkdir(parents=True, exist_ok=False)

    registry_path = output_dir / "c2g_parent_registry.jsonl"
    write_jsonl(registry_path, registry)
    cohort_paths: dict[str, Path] = {}
    for cohort in COHORTS:
        path = output_dir / f"c2g_{COHORT_SLUG[cohort]}_parents.jsonl"
        write_jsonl(path, [row for row in registry if row["cohort"] == cohort])
        cohort_paths[cohort] = path
    loto_path = output_dir / "c2g_loto_fold_plan.json"
    loso_path = output_dir / "c2g_loso_fold_plan.json"
    write_json(loto_path, {"schema": SCHEMA, "folds": loto})
    write_json(loso_path, {"schema": SCHEMA, "folds": loso})

    summary = summarize_registry(registry)
    report = {
        "schema": SCHEMA,
        "gate": "C2G_R7_SCIENTIFIC_CORPUS_PLAN",
        "status": PASS_STATUS,
        "expected_git_commit": expected_git_commit,
        "inventory_source": inventory_source,
        "selection_seed": seed,
        "max_steps": max_steps,
        "cohort_counts_per_task": asdict(counts),
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "cohort_manifests": {
            cohort: {"path": str(path), "sha256": sha256_file(path)}
            for cohort, path in cohort_paths.items()
        },
        "loto_fold_plan": str(loto_path),
        "loto_fold_plan_sha256": sha256_file(loto_path),
        "loto_fold_count": len(loto),
        "loso_fold_plan": str(loso_path),
        "loso_fold_plan_sha256": sha256_file(loso_path),
        "loso_fold_count": len(loso),
        "summary": summary,
        "tasks": task_reports,
        "training_authorization": "HOLD_PENDING_SOURCE_ELIGIBILITY_AND_LABEL_AUDIT",
        "next_stage": "R7_SOURCE_INVENTORY_AUDIT",
        "boundaries": {
            "libero_environment_created": False,
            "openvla_models_loaded": 0,
            "clean_rollouts_launched": 0,
            "attacks_launched": 0,
            "attack_outcomes_read": False,
            "training_epochs": 0,
            "calibration_runs": 0,
            "attack_eval_parents_excluded_from_detector_development": True,
        },
    }
    report_path = output_dir / "c2g_scientific_corpus_plan_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-libero", action="store_true")
    source.add_argument("--inventory-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-states-per-task", type=int, default=30)
    parser.add_argument("--val-states-per-task", type=int, default=5)
    parser.add_argument("--test-states-per-task", type=int, default=5)
    parser.add_argument("--attack-eval-states-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = (
        official_libero_inventory()
        if args.from_libero
        else load_inventory(args.inventory_json.resolve())
    )
    counts = CohortCounts(
        train=args.train_states_per_task,
        val=args.val_states_per_task,
        test=args.test_states_per_task,
        attack_eval=args.attack_eval_states_per_task,
    )
    report = materialize_plan(
        inventory,
        output_dir=args.output_dir,
        counts=counts,
        seed=args.seed,
        max_steps=args.max_steps,
        expected_git_commit=args.expected_git_commit,
        inventory_source="official_libero_benchmark" if args.from_libero else str(args.inventory_json.resolve()),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
