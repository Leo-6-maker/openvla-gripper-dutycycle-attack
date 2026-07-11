#!/usr/bin/env python3
"""Read-only R6 audit for an R5 provenance-bound C2g dataset.

This stage verifies that the exact materialized bytes remain bound to the accepted
R5 reports, reconstructs unique episode timelines from overlapping windows, and
separates three questions that must not be conflated:

1. provenance / schema integrity;
2. engineering one-epoch smoke viability;
3. scientific trainability across suites, tasks, episodes, and splits.

A scientific trainability HOLD is a successful audit outcome. The program exits
non-zero only when provenance or dataset integrity is invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.gripper_attack.c2g_clean_policy_signals import (  # noqa: E402
    CLEAN_POLICY_FEATURE_NAMES,
)
from tools.multisuite_detector.materialize_c2g_clean_window_dataset import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    HEADS,
    SUITES,
)
from tools.multisuite_detector.train_c2g_clean_window_detector import (  # noqa: E402
    load_dataset,
)
from tools.multisuite_detector.validate_c2g_clean_window_dataset import (  # noqa: E402
    audit_dataset,
)

SCHEMA = "c2g.r6.bound_dataset_audit.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R6_BOUND_DATASET_AUDIT"
INTEGRITY_PASS = "PASS_C2G_R6_DATASET_INTEGRITY"
ENGINEERING_PASS = "PASS_C2G_R6_ENGINEERING_SMOKE_VIABILITY"
ENGINEERING_HOLD = "HOLD_C2G_R6_ENGINEERING_SMOKE_VIABILITY"
SCIENTIFIC_PASS = "PASS_C2G_R6_SCIENTIFIC_TRAINABILITY"
SCIENTIFIC_HOLD = "HOLD_C2G_R6_SCIENTIFIC_TRAINABILITY"

R5_BOUND_SCHEMA = "c2g.r5.bound_materialization.2026-07-11.v1"
R5_BOUND_STATUS = "PASS_C2G_R5_BOUND_MATERIALIZATION"
R5_BASE_STATUS = "PASS_C2G_MULTISUITE_DATASET_MATERIALIZED"

FORBIDDEN_DATASET_KEY_TOKENS = (
    "attack_outcome",
    "attacked",
    "post_intervention",
    "counterfactual",
    "manual_failure",
    "vis_success",
    "random_success",
    "qpos_delta_after",
    "open_count_after",
)


@dataclass(frozen=True)
class TrainabilityThresholds:
    """Conservative lower bounds for authorizing scientific training.

    These are governance minima, not a claim of statistical sufficiency.
    """

    min_total_episodes: int = 12
    min_total_tasks: int = 8
    min_episodes_per_suite: int = 3
    min_tasks_per_suite: int = 2
    min_splits_per_suite: int = 3
    min_train_episodes: int = 4
    min_val_episodes: int = 2
    min_test_episodes: int = 2
    min_train_suites: int = 4
    min_val_suites: int = 2
    min_test_suites: int = 2

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def _require_file(path: Path, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name}: {resolved}")
    return resolved


def _same_path(recorded: Any, actual: Path, *, name: str) -> None:
    if Path(str(recorded)).resolve() != actual.resolve():
        raise ValueError(f"{name} binds another path")


def _assert_hash(path: Path, expected: str, *, name: str) -> str:
    expected = _require_sha256(expected, name=name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _assert_external_new_report(report_path: Path, dataset_root: Path) -> None:
    report_path = report_path.resolve()
    dataset_root = dataset_root.resolve()
    if _is_within(report_path, dataset_root):
        raise ValueError("R6 audit report must be outside the immutable R5 output tree")
    if report_path.exists():
        raise FileExistsError(f"R6 audit report already exists: {report_path}")
    repo = REPO.resolve()
    if _is_within(report_path, repo):
        raise ValueError("R6 runtime evidence report must be outside the repository")


def expected_npz_fields() -> set[str]:
    fields = {
        "schema_version",
        "X_proprio",
        "X_policy",
        "X_visual",
        "X_language",
        "suite",
        "task_index",
        "episode_key",
        "step",
        "split",
        "episode_fully_known_negative",
        "sample_weight",
        "feature_names_policy",
    }
    for head in HEADS:
        fields.add(f"y_{head}")
        fields.add(f"m_{head}")
    return fields


def _validate_feature_contract(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    actual_fields = set(data)
    expected_fields = expected_npz_fields()
    if actual_fields != expected_fields:
        raise ValueError(
            "dataset field closure mismatch "
            f"missing={sorted(expected_fields - actual_fields)} "
            f"unexpected={sorted(actual_fields - expected_fields)}"
        )
    forbidden = sorted(
        key
        for key in actual_fields
        if any(token in key.lower() for token in FORBIDDEN_DATASET_KEY_TOKENS)
    )
    if forbidden:
        raise ValueError("dataset contains forbidden outcome fields: " + ", ".join(forbidden))
    if str(data["schema_version"]) != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset schema version mismatch")
    if data["X_proprio"].ndim != 3 or data["X_proprio"].shape[-1] != 25:
        raise ValueError("X_proprio must be [sample,time,25]")
    if data["X_policy"].ndim != 3 or data["X_policy"].shape[-1] != len(
        CLEAN_POLICY_FEATURE_NAMES
    ):
        raise ValueError("X_policy must be [sample,time,9]")
    if data["X_visual"].ndim != 2:
        raise ValueError("X_visual must be [sample,visual_dim]")
    if data["X_language"].ndim != 2:
        raise ValueError("X_language must be [sample,language_dim]")
    policy_names = tuple(str(value) for value in data["feature_names_policy"].tolist())
    if policy_names != tuple(CLEAN_POLICY_FEATURE_NAMES):
        raise ValueError("policy feature order differs from the frozen clean contract")
    suites = set(data["suite"].astype(str).tolist())
    if suites != set(SUITES):
        raise ValueError(f"combined dataset suite closure mismatch: {sorted(suites)}")
    if any(int(value) < 0 for value in data["task_index"].astype(np.int64)):
        raise ValueError("task_index metadata must be non-negative")
    return {
        "field_count": len(actual_fields),
        "proprio_shape": list(data["X_proprio"].shape),
        "policy_shape": list(data["X_policy"].shape),
        "visual_shape": list(data["X_visual"].shape),
        "language_shape": list(data["X_language"].shape),
        "policy_feature_names": list(policy_names),
        "suite_task_identity_used_as_model_feature": False,
        "attack_or_post_intervention_fields_present": False,
    }


def _check_head_overlap(
    previous_targets: Mapping[str, np.ndarray],
    previous_masks: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    *,
    episode: str,
    step: int,
) -> None:
    for head in HEADS:
        if not np.array_equal(previous_masks[head][1:], masks[head][:-1]):
            raise ValueError(
                f"overlapping mask mismatch for {episode} step={step} head={head}"
            )
        known = masks[head][:-1].astype(bool)
        if not np.array_equal(
            previous_targets[head][1:][known],
            targets[head][:-1][known],
        ):
            raise ValueError(
                f"overlapping target mismatch for {episode} step={step} head={head}"
            )


def reconstruct_episodes(data: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Recover unique episode timelines from overlapping fixed-length windows."""

    episodes = data["episode_key"].astype(str)
    end_steps = data["step"].astype(np.int64)
    suites = data["suite"].astype(str)
    tasks = data["task_index"].astype(np.int64)
    splits = data["split"].astype(str)
    negative_flags = data["episode_fully_known_negative"].astype(bool)
    output: list[dict[str, Any]] = []
    for episode in sorted(set(episodes.tolist())):
        indices = np.flatnonzero(episodes == episode)
        indices = indices[np.argsort(end_steps[indices])]
        if indices.size == 0:
            continue
        suite_values = set(suites[indices].tolist())
        task_values = set(tasks[indices].tolist())
        split_values = set(splits[indices].tolist())
        negative_values = set(negative_flags[indices].tolist())
        if len(suite_values) != 1 or len(task_values) != 1 or len(split_values) != 1:
            raise ValueError(f"episode metadata changed across windows: {episode}")
        if len(negative_values) != 1:
            raise ValueError(f"episode negative flag changed across windows: {episode}")
        ordered_end = end_steps[indices]
        if ordered_end.size > 1 and not np.all(np.diff(ordered_end) == 1):
            raise ValueError(f"episode window endpoints are not contiguous: {episode}")

        first = int(indices[0])
        window = int(data["X_proprio"].shape[1])
        targets = {
            head: data[f"y_{head}"][first].astype(np.float32).copy()
            for head in HEADS
        }
        masks = {
            head: data[f"m_{head}"][first].astype(bool).copy()
            for head in HEADS
        }
        first_step = int(end_steps[first]) - window + 1
        unique_steps = list(range(first_step, int(end_steps[first]) + 1))
        previous_targets = {
            head: data[f"y_{head}"][first].astype(np.float32)
            for head in HEADS
        }
        previous_masks = {
            head: data[f"m_{head}"][first].astype(bool)
            for head in HEADS
        }
        for raw_index in indices[1:]:
            index = int(raw_index)
            current_targets = {
                head: data[f"y_{head}"][index].astype(np.float32)
                for head in HEADS
            }
            current_masks = {
                head: data[f"m_{head}"][index].astype(bool)
                for head in HEADS
            }
            _check_head_overlap(
                previous_targets,
                previous_masks,
                current_targets,
                current_masks,
                episode=episode,
                step=int(end_steps[index]),
            )
            for head in HEADS:
                targets[head] = np.concatenate(
                    [targets[head], current_targets[head][-1:]]
                )
                masks[head] = np.concatenate(
                    [masks[head], current_masks[head][-1:]]
                )
            unique_steps.append(int(end_steps[index]))
            previous_targets = current_targets
            previous_masks = current_masks

        critical_known = masks["critical_window"]
        critical_positive = (
            targets["critical_window"] > 0.5
        ) & critical_known
        persistent_count = 0
        for end in range(2, len(critical_positive)):
            persistent_count += int(
                int(np.sum(critical_positive[end - 2 : end + 1])) >= 2
            )
        reconstructed_negative = bool(
            critical_known.all() and not bool(critical_positive.any())
        )
        recorded_negative = bool(next(iter(negative_values)))
        if reconstructed_negative != recorded_negative:
            raise ValueError(
                f"episode fully-known-negative flag contradicts labels: {episode}"
            )
        output.append(
            {
                "episode_key": episode,
                "suite": next(iter(suite_values)),
                "task_index": int(next(iter(task_values))),
                "split": next(iter(split_values)),
                "sample_count": int(indices.size),
                "first_step": int(unique_steps[0]),
                "last_step": int(unique_steps[-1]),
                "unique_step_count": len(unique_steps),
                "known_positive_step_count": int(critical_positive.sum()),
                "known_negative_step_count": int(
                    np.sum(critical_known & ~critical_positive)
                ),
                "unknown_step_count": int(np.sum(~critical_known)),
                "positive_episode": bool(critical_positive.any()),
                "triggerable_positive_episode": bool(persistent_count > 0),
                "persistent_positive_window_count": int(persistent_count),
                "fully_known_negative_episode": reconstructed_negative,
            }
        )
    return output


def _aggregate_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "episode_count": len(rows),
        "sample_count": int(sum(int(row["sample_count"]) for row in rows)),
        "task_count": len(
            {(str(row["suite"]), int(row["task_index"])) for row in rows}
        ),
        "suite_count": len({str(row["suite"]) for row in rows}),
        "split_count": len({str(row["split"]) for row in rows}),
        "split_counts": {
            split: sum(str(row["split"]) == split for row in rows)
            for split in ("train", "val", "test")
        },
        "known_positive_step_count": int(
            sum(int(row["known_positive_step_count"]) for row in rows)
        ),
        "known_negative_step_count": int(
            sum(int(row["known_negative_step_count"]) for row in rows)
        ),
        "unknown_step_count": int(
            sum(int(row["unknown_step_count"]) for row in rows)
        ),
        "positive_episode_count": sum(bool(row["positive_episode"]) for row in rows),
        "triggerable_positive_episode_count": sum(
            bool(row["triggerable_positive_episode"]) for row in rows
        ),
        "fully_known_negative_episode_count": sum(
            bool(row["fully_known_negative_episode"]) for row in rows
        ),
        "persistent_positive_window_count": int(
            sum(int(row["persistent_positive_window_count"]) for row in rows)
        ),
    }


def summarize_episode_support(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall = _aggregate_episode_rows(rows, name="overall")
    per_suite = {
        suite: _aggregate_episode_rows(
            [row for row in rows if str(row["suite"]) == suite],
            name=suite,
        )
        for suite in SUITES
    }
    per_split = {
        split: _aggregate_episode_rows(
            [row for row in rows if str(row["split"]) == split],
            name=split,
        )
        for split in ("train", "val", "test")
    }
    return overall, per_suite, per_split


def scientific_trainability_violations(
    overall: Mapping[str, Any],
    per_suite: Mapping[str, Mapping[str, Any]],
    per_split: Mapping[str, Mapping[str, Any]],
    thresholds: TrainabilityThresholds,
) -> list[dict[str, Any]]:
    thresholds.validate()
    violations: list[dict[str, Any]] = []

    def minimum(scope: str, field: str, actual: int, required: int) -> None:
        if int(actual) < int(required):
            violations.append(
                {
                    "scope": scope,
                    "reason": "MINIMUM_SUPPORT_NOT_MET",
                    "field": field,
                    "actual": int(actual),
                    "required": int(required),
                }
            )

    minimum(
        "overall",
        "episode_count",
        int(overall["episode_count"]),
        thresholds.min_total_episodes,
    )
    minimum(
        "overall",
        "task_count",
        int(overall["task_count"]),
        thresholds.min_total_tasks,
    )
    for suite in SUITES:
        report = per_suite[suite]
        minimum(
            suite,
            "episode_count",
            int(report["episode_count"]),
            thresholds.min_episodes_per_suite,
        )
        minimum(
            suite,
            "task_count",
            int(report["task_count"]),
            thresholds.min_tasks_per_suite,
        )
        minimum(
            suite,
            "split_count",
            int(report["split_count"]),
            thresholds.min_splits_per_suite,
        )
        for field in (
            "known_positive_step_count",
            "known_negative_step_count",
            "positive_episode_count",
            "triggerable_positive_episode_count",
        ):
            minimum(suite, field, int(report[field]), 1)

    split_requirements = {
        "train": (
            thresholds.min_train_episodes,
            thresholds.min_train_suites,
        ),
        "val": (
            thresholds.min_val_episodes,
            thresholds.min_val_suites,
        ),
        "test": (
            thresholds.min_test_episodes,
            thresholds.min_test_suites,
        ),
    }
    for split, (min_episodes, min_suites) in split_requirements.items():
        report = per_split[split]
        minimum(split, "episode_count", int(report["episode_count"]), min_episodes)
        minimum(split, "suite_count", int(report["suite_count"]), min_suites)
        for field in (
            "known_positive_step_count",
            "known_negative_step_count",
            "positive_episode_count",
            "triggerable_positive_episode_count",
        ):
            minimum(split, field, int(report[field]), 1)
    return violations


def _verify_per_suite_artifacts(
    base_report: Mapping[str, Any],
    *,
    dataset_root: Path,
) -> tuple[dict[str, Any], set[Path]]:
    per_suite = base_report.get("per_suite")
    if not isinstance(per_suite, Mapping) or set(per_suite) != set(SUITES):
        raise ValueError("base report must bind exactly the four frozen suites")
    verified: dict[str, Any] = {}
    expected_files: set[Path] = set()
    for suite in SUITES:
        binding = per_suite[suite]
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid per-suite binding for {suite}")
        dataset_path = _require_file(
            Path(str(binding["dataset_path"])),
            name=f"{suite} dataset",
        )
        report_path = _require_file(
            Path(str(binding["report_path"])),
            name=f"{suite} report",
        )
        if not _is_within(dataset_path, dataset_root) or not _is_within(
            report_path, dataset_root
        ):
            raise ValueError(f"{suite} artifacts escape the R5 output tree")
        dataset_sha = _assert_hash(
            dataset_path,
            str(binding["dataset_sha256"]),
            name=f"{suite} dataset",
        )
        report_sha = _assert_hash(
            report_path,
            str(binding["report_sha256"]),
            name=f"{suite} report",
        )
        suite_report = _read_json(report_path)
        if not str(suite_report.get("status", "")).startswith("PASS_"):
            raise ValueError(f"{suite} materialization report is not PASS")
        if int(suite_report.get("n_episode_errors", -1)) != 0:
            raise ValueError(f"{suite} materialization recorded episode errors")
        if int(suite_report.get("n_windows", -1)) != int(binding["n_windows"]):
            raise ValueError(f"{suite} window count differs across reports")
        if int(suite_report.get("n_episodes_processed", -1)) != int(
            binding["n_episodes_processed"]
        ):
            raise ValueError(f"{suite} episode count differs across reports")
        manifest_path = _require_file(
            Path(str(suite_report["input_manifest_path"])),
            name=f"{suite} input manifest",
        )
        error_path = _require_file(
            Path(str(suite_report["error_ledger_path"])),
            name=f"{suite} error ledger",
        )
        if not _is_within(manifest_path, dataset_root) or not _is_within(
            error_path, dataset_root
        ):
            raise ValueError(f"{suite} sidecars escape the R5 output tree")
        if "input_manifest_sha256" in suite_report:
            _assert_hash(
                manifest_path,
                str(suite_report["input_manifest_sha256"]),
                name=f"{suite} input manifest",
            )
        if error_path.stat().st_size != 0:
            raise ValueError(f"{suite} error ledger is not empty")
        with np.load(dataset_path, allow_pickle=False) as archive:
            suite_values = archive["suite"].astype(str)
            sample_count = int(archive["X_proprio"].shape[0])
        if sample_count != int(binding["n_windows"]):
            raise ValueError(f"{suite} dataset cardinality differs from report")
        if set(suite_values.tolist()) != {suite}:
            raise ValueError(f"{suite} dataset contains another suite identity")
        expected_files.update(
            {dataset_path, report_path, manifest_path, error_path}
        )
        verified[suite] = {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha,
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "n_windows": sample_count,
            "n_episodes_processed": int(binding["n_episodes_processed"]),
            "error_ledger_empty": True,
        }
    return verified, expected_files


def audit_bound_dataset(
    *,
    dataset_path: Path,
    bound_report_path: Path,
    base_report_path: Path,
    expected_dataset_sha256: str,
    expected_bound_report_sha256: str,
    expected_base_report_sha256: str,
    expected_materialization_head: str,
    audit_head: str,
    thresholds: TrainabilityThresholds = TrainabilityThresholds(),
    persistence_window: int = 3,
    persistence_required: int = 2,
) -> dict[str, Any]:
    if (int(persistence_window), int(persistence_required)) != (3, 2):
        raise ValueError("R6 audit is frozen to the deployed 2-of-3 persistence rule")
    dataset_path = _require_file(dataset_path, name="combined dataset")
    bound_report_path = _require_file(
        bound_report_path, name="R5 bound materialization report"
    )
    base_report_path = _require_file(
        base_report_path, name="R5 base materialization report"
    )
    dataset_root = dataset_path.parent.resolve()
    expected_materialization_head = str(expected_materialization_head).strip()
    audit_head = str(audit_head).strip()
    if len(expected_materialization_head) != 40 or len(audit_head) != 40:
        raise ValueError("materialization and audit heads must be full commit SHAs")

    dataset_sha = _assert_hash(
        dataset_path, expected_dataset_sha256, name="expected combined dataset"
    )
    bound_report_sha = _assert_hash(
        bound_report_path,
        expected_bound_report_sha256,
        name="expected bound report",
    )
    base_report_sha = _assert_hash(
        base_report_path,
        expected_base_report_sha256,
        name="expected base report",
    )
    bound_report = _read_json(bound_report_path)
    base_report = _read_json(base_report_path)

    if bound_report.get("schema") != R5_BOUND_SCHEMA:
        raise ValueError("R5 bound report schema mismatch")
    if bound_report.get("status") != R5_BOUND_STATUS:
        raise ValueError("R5 bound report is not PASS")
    if bound_report.get("audit_head") != expected_materialization_head:
        raise ValueError("R5 report was produced by another materialization head")
    _same_path(
        bound_report.get("combined_dataset"),
        dataset_path,
        name="R5 bound dataset",
    )
    if bound_report.get("combined_dataset_sha256") != dataset_sha:
        raise ValueError("R5 bound report records another dataset hash")
    _same_path(
        bound_report.get("base_report"),
        base_report_path,
        name="R5 bound base report",
    )
    if bound_report.get("base_report_sha256") != base_report_sha:
        raise ValueError("R5 bound report records another base report hash")
    if bound_report.get("per_suite") != base_report.get("per_suite"):
        raise ValueError("R5 bound and base reports disagree on per-suite artifacts")
    r4_binding_path = _require_file(
        Path(str(bound_report.get("r4_provenance_binding", ""))),
        name="R4 provenance binding",
    )
    _assert_hash(
        r4_binding_path,
        str(bound_report.get("r4_provenance_binding_sha256", "")),
        name="R4 provenance binding",
    )
    r4_binding = _read_json(r4_binding_path)
    if r4_binding.get("status") != "PASS_C2G_R4_DUAL_HEAD_PROVENANCE_BINDING":
        raise ValueError("R4 provenance binding is not PASS")
    if r4_binding.get("audit_head") != expected_materialization_head:
        raise ValueError("R4 provenance binding belongs to another audit head")
    boundaries = bound_report.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("R5 bound report lacks boundary attestations")
    required_boundaries = {
        "clean_only": True,
        "attack_outcomes_read": False,
        "counterfactual_read": False,
        "suite_task_identity_used_as_model_feature": False,
        "libero_rollouts_launched": 0,
        "attacks_launched": 0,
        "training_epochs": 0,
    }
    for key, expected in required_boundaries.items():
        if boundaries.get(key) != expected:
            raise ValueError(f"R5 boundary mismatch: {key}")

    if base_report.get("status") != R5_BASE_STATUS:
        raise ValueError("R5 base materialization report is not PASS")
    _same_path(
        base_report.get("combined_dataset"),
        dataset_path,
        name="base combined dataset",
    )
    if base_report.get("combined_dataset_sha256") != dataset_sha:
        raise ValueError("base report records another dataset hash")
    base_boundaries = base_report.get("boundaries")
    if not isinstance(base_boundaries, Mapping):
        raise ValueError("base report lacks boundary attestations")
    if base_boundaries.get("clean_only") is not True:
        raise ValueError("base report is not clean-only")
    if base_boundaries.get("attack_outcomes_read") is not False:
        raise ValueError("base report records attacked-outcome access")
    if base_boundaries.get("suite_task_identity_used_as_model_feature") is not False:
        raise ValueError("base report records suite/task shortcut features")

    data = load_dataset(dataset_path)
    feature_contract = _validate_feature_contract(data)
    sample_count = int(data["X_proprio"].shape[0])
    if int(bound_report.get("combined_samples", -1)) != sample_count:
        raise ValueError("bound report sample count differs from dataset")
    if int(base_report.get("combined_samples", -1)) != sample_count:
        raise ValueError("base report sample count differs from dataset")
    actual_split_counts = {
        split: int(np.sum(data["split"].astype(str) == split))
        for split in ("train", "val", "test")
    }
    if bound_report.get("split_counts") != actual_split_counts:
        raise ValueError("bound report split counts differ from dataset")
    if base_report.get("split_counts") != actual_split_counts:
        raise ValueError("base report split counts differ from dataset")

    per_suite_artifacts, expected_files = _verify_per_suite_artifacts(
        base_report,
        dataset_root=dataset_root,
    )
    expected_files.update(
        {dataset_path, base_report_path, bound_report_path}
    )
    actual_files = {path.resolve() for path in dataset_root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "R5 output file closure mismatch "
            f"missing={sorted(str(path) for path in expected_files - actual_files)} "
            f"unexpected={sorted(str(path) for path in actual_files - expected_files)}"
        )

    reconstructed = reconstruct_episodes(data)
    if sum(int(row["sample_count"]) for row in reconstructed) != sample_count:
        raise ValueError("reconstructed episode cardinality differs from dataset")
    overall, per_suite, per_split = summarize_episode_support(reconstructed)
    for suite in SUITES:
        if int(per_suite[suite]["sample_count"]) != int(
            per_suite_artifacts[suite]["n_windows"]
        ):
            raise ValueError(f"{suite} reconstructed sample count mismatch")

    engineering = audit_dataset(
        data,
        persistence_window=persistence_window,
        persistence_required=persistence_required,
        require_test_support=True,
    )
    engineering_status = (
        ENGINEERING_PASS
        if engineering["status"] == "PASS_C2G_DATASET_TRAINABILITY"
        else ENGINEERING_HOLD
    )
    scientific_violations = scientific_trainability_violations(
        overall,
        per_suite,
        per_split,
        thresholds,
    )
    scientific_status = (
        SCIENTIFIC_PASS if not scientific_violations else SCIENTIFIC_HOLD
    )
    return {
        "schema": SCHEMA,
        "status": PASS_STATUS,
        "integrity_status": INTEGRITY_PASS,
        "engineering_smoke_status": engineering_status,
        "scientific_trainability_status": scientific_status,
        "training_authorization": (
            "HOLD_PENDING_EXPLICIT_TRAINING_AUTHORIZATION"
            if scientific_status == SCIENTIFIC_PASS
            else "HOLD_INSUFFICIENT_SCIENTIFIC_SUPPORT"
        ),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "bound_materialization_report": str(bound_report_path),
        "bound_materialization_report_sha256": bound_report_sha,
        "base_materialization_report": str(base_report_path),
        "base_materialization_report_sha256": base_report_sha,
        "materialization_head": expected_materialization_head,
        "audit_head": audit_head,
        "audit_tool": str(Path(__file__).resolve()),
        "audit_tool_sha256": sha256_file(Path(__file__).resolve()),
        "sample_count": sample_count,
        "episode_count": int(overall["episode_count"]),
        "split_counts": actual_split_counts,
        "output_file_count": len(actual_files),
        "feature_contract": feature_contract,
        "per_suite_artifacts": per_suite_artifacts,
        "episode_support": {
            "overall": overall,
            "per_suite": per_suite,
            "per_split": per_split,
        },
        "engineering_smoke_audit": engineering,
        "scientific_trainability_thresholds": asdict(thresholds),
        "scientific_trainability_violation_count": len(scientific_violations),
        "scientific_trainability_violations": scientific_violations,
        "next_stage": (
            "STOP_FOR_TRAINABILITY_REVIEW"
            if scientific_violations
            else "STOP_FOR_EXPLICIT_TRAINING_AUTHORIZATION"
        ),
        "boundaries": {
            "read_only_dataset_audit": True,
            "dataset_files_modified": 0,
            "model_loads": 0,
            "libero_environments": 0,
            "clean_rollouts": 0,
            "attacked_rollouts": 0,
            "training_epochs": 0,
            "calibration_runs": 0,
            "d7_table1_modified": False,
            "scientific_contract_changes": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bound-materialization-report", type=Path, required=True)
    parser.add_argument("--base-materialization-report", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-bound-report-sha256", required=True)
    parser.add_argument("--expected-base-report-sha256", required=True)
    parser.add_argument("--expected-materialization-head", required=True)
    parser.add_argument("--audit-head", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--persistence-window", type=int, default=3)
    parser.add_argument("--persistence-required", type=int, default=2)
    parser.add_argument("--min-total-episodes", type=int, default=12)
    parser.add_argument("--min-total-tasks", type=int, default=8)
    parser.add_argument("--min-episodes-per-suite", type=int, default=3)
    parser.add_argument("--min-tasks-per-suite", type=int, default=2)
    parser.add_argument("--min-splits-per-suite", type=int, default=3)
    parser.add_argument("--min-train-episodes", type=int, default=4)
    parser.add_argument("--min-val-episodes", type=int, default=2)
    parser.add_argument("--min-test-episodes", type=int, default=2)
    parser.add_argument("--min-train-suites", type=int, default=4)
    parser.add_argument("--min-val-suites", type=int, default=2)
    parser.add_argument("--min-test-suites", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_report = args.output_report.resolve()
    _assert_external_new_report(output_report, args.dataset.resolve().parent)
    thresholds = TrainabilityThresholds(
        min_total_episodes=args.min_total_episodes,
        min_total_tasks=args.min_total_tasks,
        min_episodes_per_suite=args.min_episodes_per_suite,
        min_tasks_per_suite=args.min_tasks_per_suite,
        min_splits_per_suite=args.min_splits_per_suite,
        min_train_episodes=args.min_train_episodes,
        min_val_episodes=args.min_val_episodes,
        min_test_episodes=args.min_test_episodes,
        min_train_suites=args.min_train_suites,
        min_val_suites=args.min_val_suites,
        min_test_suites=args.min_test_suites,
    )
    report = audit_bound_dataset(
        dataset_path=args.dataset,
        bound_report_path=args.bound_materialization_report,
        base_report_path=args.base_materialization_report,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_bound_report_sha256=args.expected_bound_report_sha256,
        expected_base_report_sha256=args.expected_base_report_sha256,
        expected_materialization_head=args.expected_materialization_head,
        audit_head=args.audit_head,
        thresholds=thresholds,
        persistence_window=args.persistence_window,
        persistence_required=args.persistence_required,
    )
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
