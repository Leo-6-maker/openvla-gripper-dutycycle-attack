#!/usr/bin/env python3
"""Audit existing clean Detector-v2 source episodes against an R7 registry.

This is a read-only structural and label-support census.  It discovers clean
``episode_metadata.json`` + ``step_records.jsonl`` pairs, verifies the frozen
25D/9D causal feature contract and RGB availability, rebuilds Teacher-v2 labels
from clean privileged fields, and reports which preregistered parents can be
reused before any new collection is authorized.

An incomplete source corpus is an expected successful audit outcome.  The tool
never loads OpenVLA, creates LIBERO environments, launches rollouts, trains a
model, calibrates thresholds, or reads attacked outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.gripper_attack.c2g_clean_policy_signals import (  # noqa: E402
    CLEAN_POLICY_FEATURE_NAMES,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (  # noqa: E402
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)
from tools.multisuite_detector.plan_c2g_scientific_corpus import (  # noqa: E402
    ATTACK_EVAL,
    COHORTS,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    PASS_STATUS as PLAN_PASS_STATUS,
    SCHEMA as PLAN_SCHEMA,
    SUITES,
    sha256_file,
)

SCHEMA = "c2g.r7.clean_source_inventory_audit.2026-07-11.v1"
PASS_STATUS = "PASS_C2G_R7_CLEAN_SOURCE_INVENTORY_AUDIT"
READY = "PASS_C2G_R7_SOURCE_CORPUS_COMPLETE"
HOLD = "HOLD_C2G_R7_SOURCE_CORPUS_INCOMPLETE"

FORBIDDEN_KEY_TOKENS = (
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def _assert_hash(path: Path, expected: str, *, name: str) -> str:
    expected = _require_sha256(expected, name=name)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def _assert_external_new_output(report: Path, manifest: Path) -> None:
    repo = REPO.resolve()
    for path in (report.resolve(), manifest.resolve()):
        if path.exists():
            raise FileExistsError(path)
        if path == repo or repo in path.parents:
            raise ValueError("runtime inventory outputs must be outside the repository")
    if report.resolve() == manifest.resolve():
        raise ValueError("report and reusable manifest paths must differ")


def load_registry(
    registry_path: Path,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_path = registry_path.resolve()
    plan_report_path = plan_report_path.resolve()
    _assert_hash(
        plan_report_path,
        expected_plan_report_sha256,
        name="scientific corpus plan report",
    )
    plan = read_json(plan_report_path)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_PASS_STATUS:
        raise ValueError("corpus plan report is not an accepted R7 plan")
    if Path(str(plan.get("registry", ""))).resolve() != registry_path:
        raise ValueError("corpus plan report binds another registry path")
    if str(plan.get("registry_sha256", "")) != sha256_file(registry_path):
        raise ValueError("registry bytes differ from the corpus plan report")
    rows = read_jsonl(registry_path)
    identities: set[tuple[str, int, int]] = set()
    for index, row in enumerate(rows):
        identity = (
            str(row.get("suite", "")),
            int(row.get("task_index", -1)),
            int(row.get("state_id", -1)),
        )
        if identity[0] not in SUITES or identity[1] < 0 or identity[2] < 0:
            raise ValueError(f"invalid registry identity at row {index}")
        if identity in identities:
            raise ValueError(f"duplicate registry identity: {identity}")
        identities.add(identity)
        if str(row.get("cohort", "")) not in COHORTS:
            raise ValueError(f"invalid registry cohort at row {index}")
    return rows, plan


def discover_sources(roots: Sequence[Path]) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for step_path in sorted(root.rglob("step_records.jsonl")):
            metadata_path = step_path.with_name("episode_metadata.json")
            if metadata_path.is_file():
                pairs.append((root, metadata_path, step_path))
    return pairs


def _forbidden_keys(mapping: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(key)
        for key in mapping
        if any(token in str(key).lower() for token in FORBIDDEN_KEY_TOKENS)
    )


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return vector


def _policy_vector(row: Mapping[str, Any]) -> np.ndarray:
    for key in ("clean_policy_intent_9d", "clean_policy_features", "policy_intent"):
        if key in row and row[key] is not None:
            return _finite_vector(row[key], len(CLEAN_POLICY_FEATURE_NAMES), key)
    if all(name in row and row[name] is not None for name in CLEAN_POLICY_FEATURE_NAMES):
        return _finite_vector(
            [row[name] for name in CLEAN_POLICY_FEATURE_NAMES],
            len(CLEAN_POLICY_FEATURE_NAMES),
            "named clean policy features",
        )
    raise KeyError("clean policy-intent features are absent")


def _ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if len(steps) != len(set(steps)):
        raise ValueError("duplicate step IDs")
    if len(steps) > 1 and any(right != left + 1 for left, right in zip(steps, steps[1:])):
        raise ValueError("step IDs are not contiguous")
    return ordered


def _tree_digest(entries: Sequence[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for relative, size, file_hash in sorted(entries):
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _triggerable(positive: Sequence[bool], window: int, required: int) -> bool:
    if required < 1 or window < required:
        raise ValueError("persistence requires 1 <= required <= window")
    values = np.asarray(positive, dtype=np.bool_)
    for end in range(window - 1, len(values)):
        if int(values[end - window + 1 : end + 1].sum()) >= required:
            return True
    return False


def audit_episode(
    root: Path,
    metadata_path: Path,
    step_path: Path,
    *,
    registry_lookup: Mapping[tuple[str, int, int], Mapping[str, Any]],
    thresholds: CleanTeacherThresholds,
    persistence_window: int,
    persistence_required: int,
    hash_rgb: bool,
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    rows = _ordered_rows(read_jsonl(step_path))
    suite = str(metadata.get("suite", ""))
    task_index = int(metadata.get("task_index", metadata.get("task_id", -1)))
    state_id = int(metadata.get("state_id", -1))
    identity = (suite, task_index, state_id)
    episode_key = str(
        metadata.get("episode_key")
        or metadata.get("parent_key")
        or metadata_path.parent.as_posix()
    )
    result: dict[str, Any] = {
        "episode_key": episode_key,
        "suite": suite,
        "task_index": task_index,
        "state_id": state_id,
        "source_root": str(root),
        "metadata_path": str(metadata_path),
        "step_records_path": str(step_path),
        "metadata_sha256": sha256_file(metadata_path),
        "step_records_sha256": sha256_file(step_path),
        "registered": identity in registry_lookup,
        "registry_parent_key": None,
        "cohort": None,
        "split": None,
        "structurally_eligible": False,
        "reusable": False,
        "failure_reason": None,
    }
    if identity in registry_lookup:
        registered = registry_lookup[identity]
        result.update(
            registry_parent_key=str(registered["parent_key"]),
            cohort=str(registered["cohort"]),
            split=str(registered["split"]),
        )
    try:
        if suite not in SUITES or task_index < 0 or state_id < 0:
            raise ValueError("invalid suite/task/state identity")
        if metadata.get("runtime_valid") is not True:
            raise ValueError("runtime_valid is not true")
        if str(metadata.get("condition", "")).upper() != "CLEAN":
            raise ValueError("episode condition is not CLEAN")
        forbidden_metadata = _forbidden_keys(metadata)
        if forbidden_metadata:
            raise ValueError("forbidden metadata keys: " + ", ".join(forbidden_metadata))
        if len(rows) < 16:
            raise ValueError("episode has fewer than 16 steps")
        language = str(metadata.get("task_language") or rows[0].get("task_language", "")).strip()
        if not language:
            raise ValueError("task language is empty")

        rgb_entries: list[tuple[str, int, str]] = []
        for row_index, row in enumerate(rows):
            forbidden = _forbidden_keys(row)
            if forbidden:
                raise ValueError(
                    f"forbidden step keys at row {row_index}: " + ", ".join(forbidden)
                )
            _finite_vector(row.get("features_25d"), 25, "features_25d")
            _policy_vector(row)
            rgb_value = str(row.get("rgb_path", "")).strip()
            if not rgb_value:
                raise ValueError(f"missing rgb_path at row {row_index}")
            rgb_path = (step_path.parent / rgb_value).resolve()
            if not rgb_path.is_file():
                raise FileNotFoundError(rgb_path)
            relative = rgb_path.relative_to(root).as_posix()
            rgb_entries.append(
                (
                    relative,
                    rgb_path.stat().st_size,
                    sha256_file(rgb_path) if hash_rgb else "NOT_HASHED",
                )
            )

        labels = build_clean_teacher_episode(rows, metadata, thresholds=thresholds)
        if len(labels) != len(rows):
            raise RuntimeError("Teacher-v2 label count differs from source steps")
        known = np.asarray([bool(row["label_known_mask"]) for row in labels])
        positive = np.asarray(
            [bool(row["y_gripper_critical_window"]) if known[index] else False for index, row in enumerate(labels)]
        )
        known_positive = int(np.sum(known & positive))
        known_negative = int(np.sum(known & ~positive))
        unknown = int(len(labels) - int(np.sum(known)))
        fully_negative = bool(known.all() and not positive.any())
        triggerable = _triggerable(
            (known & positive).tolist(), persistence_window, persistence_required
        )
        result.update(
            n_steps=len(rows),
            rgb_count=len(rgb_entries),
            rgb_bytes=sum(item[1] for item in rgb_entries),
            rgb_tree_sha256=_tree_digest(rgb_entries),
            rgb_files_hashed=bool(hash_rgb),
            known_positive_steps=known_positive,
            known_negative_steps=known_negative,
            unknown_steps=unknown,
            positive_episode=bool(known_positive > 0),
            fully_known_negative_episode=fully_negative,
            triggerable_positive_episode=triggerable,
            clean_success_observed=bool(metadata.get("clean_success_observed", False)),
            mechanism_type=str(metadata.get("mechanism_type", "")),
            structurally_eligible=True,
        )
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(rows),
        "task_count": len({(row["suite"], row["task_index"]) for row in rows}),
        "positive_episode_count": sum(bool(row.get("positive_episode")) for row in rows),
        "fully_known_negative_episode_count": sum(
            bool(row.get("fully_known_negative_episode")) for row in rows
        ),
        "unknown_containing_episode_count": sum(
            int(row.get("unknown_steps", 0)) > 0 for row in rows
        ),
        "triggerable_positive_episode_count": sum(
            bool(row.get("triggerable_positive_episode")) for row in rows
        ),
        "known_positive_steps": sum(int(row.get("known_positive_steps", 0)) for row in rows),
        "known_negative_steps": sum(int(row.get("known_negative_steps", 0)) for row in rows),
        "unknown_steps": sum(int(row.get("unknown_steps", 0)) for row in rows),
    }


def audit_inventory(
    *,
    registry_path: Path,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    source_roots: Sequence[Path],
    output_report: Path,
    reusable_manifest: Path,
    persistence_window: int = 3,
    persistence_required: int = 2,
    burst_length: int = 10,
    hash_rgb: bool = True,
    audit_head: str = "",
) -> dict[str, Any]:
    _assert_external_new_output(output_report, reusable_manifest)
    registry, plan = load_registry(
        registry_path, plan_report_path, expected_plan_report_sha256
    )
    registry_lookup = {
        (str(row["suite"]), int(row["task_index"]), int(row["state_id"])): row
        for row in registry
    }
    pairs = discover_sources(source_roots)
    thresholds = CleanTeacherThresholds(burst_length=burst_length)
    episodes = [
        audit_episode(
            root,
            metadata,
            steps,
            registry_lookup=registry_lookup,
            thresholds=thresholds,
            persistence_window=persistence_window,
            persistence_required=persistence_required,
            hash_rgb=hash_rgb,
        )
        for root, metadata, steps in pairs
    ]

    identity_counts = Counter(
        (row["suite"], row["task_index"], row["state_id"]) for row in episodes
    )
    for row in episodes:
        identity = (row["suite"], row["task_index"], row["state_id"])
        duplicate = identity_counts[identity] > 1
        row["duplicate_identity"] = duplicate
        row["reusable"] = bool(
            row["registered"] and row["structurally_eligible"] and not duplicate
        )
        if duplicate and row.get("failure_reason") is None:
            row["failure_reason"] = "DUPLICATE_SUITE_TASK_STATE_IDENTITY"
        elif not row["registered"] and row.get("failure_reason") is None:
            row["failure_reason"] = "UNREGISTERED_CLEAN_ASSET"

    reusable = sorted(
        (row for row in episodes if row["reusable"]),
        key=lambda row: (SUITES.index(row["suite"]), row["task_index"], row["state_id"]),
    )
    reusable_identities = {
        (row["suite"], row["task_index"], row["state_id"]) for row in reusable
    }
    deficits = [
        {
            "parent_key": row["parent_key"],
            "suite": row["suite"],
            "task_index": row["task_index"],
            "state_id": row["state_id"],
            "cohort": row["cohort"],
            "split": row["split"],
        }
        for row in registry
        if (row["suite"], row["task_index"], row["state_id"])
        not in reusable_identities
    ]

    per_cohort: dict[str, Any] = {}
    for cohort in COHORTS:
        rows = [row for row in reusable if row["cohort"] == cohort]
        required = sum(registry_row["cohort"] == cohort for registry_row in registry)
        per_cohort[cohort] = {
            **_support(rows),
            "required_episode_count": required,
            "missing_episode_count": required - len(rows),
        }
    per_suite: dict[str, Any] = {}
    for suite in SUITES:
        rows = [row for row in reusable if row["suite"] == suite]
        required = sum(registry_row["suite"] == suite for registry_row in registry)
        per_suite[suite] = {
            **_support(rows),
            "required_episode_count": required,
            "missing_episode_count": required - len(rows),
            "cohort_counts": dict(sorted(Counter(row["cohort"] for row in rows).items())),
        }
    per_split: dict[str, Any] = {}
    for split in ("train", "val", "test", "attack_eval"):
        rows = [row for row in reusable if row["split"] == split]
        required = sum(registry_row["split"] == split for registry_row in registry)
        per_split[split] = {
            **_support(rows),
            "suite_count": len({row["suite"] for row in rows}),
            "required_episode_count": required,
            "missing_episode_count": required - len(rows),
        }

    detector_required = sum(row["cohort"] != ATTACK_EVAL for row in registry)
    detector_available = sum(row["cohort"] != ATTACK_EVAL for row in reusable)
    attack_required = sum(row["cohort"] == ATTACK_EVAL for row in registry)
    attack_available = sum(row["cohort"] == ATTACK_EVAL for row in reusable)
    detector_complete = detector_available == detector_required
    attack_complete = attack_available == attack_required

    reusable_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(reusable_manifest, reusable)
    report = {
        "schema": SCHEMA,
        "gate": "C2G_R7_CLEAN_SOURCE_INVENTORY_AUDIT",
        "status": PASS_STATUS,
        "audit_head": audit_head,
        "plan_report": str(plan_report_path.resolve()),
        "plan_report_sha256": sha256_file(plan_report_path.resolve()),
        "registry": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path.resolve()),
        "source_roots": [str(path.resolve()) for path in source_roots],
        "source_episode_candidate_count": len(episodes),
        "registered_reusable_episode_count": len(reusable),
        "registered_ineligible_episode_count": sum(
            row["registered"] and not row["reusable"] for row in episodes
        ),
        "unregistered_episode_count": sum(not row["registered"] for row in episodes),
        "duplicate_identity_count": sum(count > 1 for count in identity_counts.values()),
        "detector_source_corpus_status": READY if detector_complete else HOLD,
        "attack_eval_source_corpus_status": READY if attack_complete else HOLD,
        "detector_required_episode_count": detector_required,
        "detector_available_episode_count": detector_available,
        "detector_missing_episode_count": detector_required - detector_available,
        "attack_eval_required_episode_count": attack_required,
        "attack_eval_available_episode_count": attack_available,
        "attack_eval_missing_episode_count": attack_required - attack_available,
        "per_cohort": per_cohort,
        "per_suite": per_suite,
        "per_split": per_split,
        "episode_audits": episodes,
        "missing_registry_parents": deficits,
        "reusable_manifest": str(reusable_manifest.resolve()),
        "reusable_manifest_sha256": sha256_file(reusable_manifest.resolve()),
        "within_task_generalization_ready": bool(
            detector_complete
            and all(
                per_split[split]["suite_count"] == len(SUITES)
                and per_split[split]["positive_episode_count"] > 0
                and per_split[split]["fully_known_negative_episode_count"] > 0
                and per_split[split]["triggerable_positive_episode_count"] > 0
                for split in ("train", "val", "test")
            )
        ),
        "loto_generalization_ready": False,
        "loso_generalization_ready": False,
        "training_authorization": "HOLD_PENDING_FULL_CORPUS_MATERIALIZATION_AND_AUDIT",
        "next_stage": "STOP_FOR_R7_SOURCE_INVENTORY_REVIEW",
        "boundaries": {
            "clean_only": True,
            "attack_outcomes_read": False,
            "openvla_models_loaded": 0,
            "libero_environments_created": 0,
            "clean_rollouts_launched": 0,
            "attacks_launched": 0,
            "training_epochs": 0,
            "calibration_runs": 0,
            "source_assets_modified": False,
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_report, report)
    return {**report, "output_report_sha256": sha256_file(output_report)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--reusable-manifest", type=Path, required=True)
    parser.add_argument("--persistence-window", type=int, default=3)
    parser.add_argument("--persistence-required", type=int, default=2)
    parser.add_argument("--burst-length", type=int, default=10)
    parser.add_argument("--hash-rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-head", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_inventory(
        registry_path=args.registry,
        plan_report_path=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        source_roots=args.source_root,
        output_report=args.output_report,
        reusable_manifest=args.reusable_manifest,
        persistence_window=args.persistence_window,
        persistence_required=args.persistence_required,
        burst_length=args.burst_length,
        hash_rgb=args.hash_rgb,
        audit_head=args.audit_head,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
