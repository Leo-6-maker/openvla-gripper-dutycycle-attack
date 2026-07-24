#!/usr/bin/env python3
"""CPU-only official-horizon derivation and Teacher-v2 label rebuilding for R8Z."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.stageb.run_c2g_r8w_full_clean_shard import validate_episode_receipt
from tools.multisuite_detector.audit_c2g_r8w_full_clean_2000 import (
    POST_STEP_SCHEMA,
    STEP_REQUIRED_KEYS,
    finite_vector,
    post_step_complete,
)
from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
    PASS_STATUS as R8W_PLAN_PASS_STATUS,
    read_json,
    read_jsonl,
    sha256_file,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)
from tools.multisuite_detector.c2g_official_suite_horizons import (
    official_max_policy_steps,
)


R8Z_SCHEMA = "c2g.r8z.ogs_official_horizon.2026-07-12.v1"
R8Z_RECEIPT_SCHEMA = "c2g.r8z.episode_receipt.2026-07-12.v1"
SOURCE_HORIZON = 300
BURST_LENGTH = 10
TARGET_SUITES = ("libero_spatial", "libero_object", "libero_goal")
TRAIN_COHORT = "DETECTOR_TRAIN"
COHORTS = (
    "DETECTOR_TRAIN",
    "DETECTOR_VAL",
    "DETECTOR_TEST_WITHIN_TASK",
    "ATTACK_EVAL_PREREGISTERED",
)
EXPECTED_COHORT_COUNTS = {
    "DETECTOR_TRAIN": 900,
    "DETECTOR_VAL": 150,
    "DETECTOR_TEST_WITHIN_TASK": 150,
    "ATTACK_EVAL_PREREGISTERED": 300,
}
EXPECTED_SUITE_COHORT_COUNTS = {
    "DETECTOR_TRAIN": 300,
    "DETECTOR_VAL": 50,
    "DETECTOR_TEST_WITHIN_TASK": 50,
    "ATTACK_EVAL_PREREGISTERED": 100,
}
CANARY_SALT = "C2G_R8Z_OGS_LABEL_CANARY_2026-07-12.v1"
SOURCE_STEP_SCHEMA = "c2g.teacher_v2.raw_privileged_evidence.2026-07-11.v1"
SOURCE_WORKER_PASS = "PASS_C2G_R8W_FULL_CLEAN_SHARD_RUN"
SOURCE_COLLECTION_PASS = "PASS_C2G_R8W_TEACHER_V2_CLEAN_SHARD_COLLECTION"

LABEL_FIELDS = (
    "y_target_relevant",
    "y_gripper_dependency",
    "y_clean_close_intent",
    "y_manipulation_progress_active",
    "y_release_safe",
    "y_gripper_critical_window",
    "y_burst_feasible",
    "y_attack_start_B",
)
CANONICAL_LABEL_FIELDS = (
    "y_target_relevant",
    "y_contact_or_grasp_stable",
    "y_gripper_dependency",
    "y_clean_close_intent",
    "y_lift_transport_or_constraint",
    "y_release_safe",
    "y_gripper_critical_window",
    "y_burst_feasible",
    "y_attack_start_b",
)
TEACHER_EXCLUDED_METADATA_FIELDS = frozenset(
    {
        "clean_success_observed",
        "clean_success_first_step",
        "final_env_check_success",
        "termination_reason",
        "done_first_step",
        "reward_sum",
        "reward_max",
        "reward_nonzero_step_count",
        "late_success_in_extended_source",
        "uses_source_final_outcome_for_teacher",
        "uses_future_step_for_teacher",
        "uses_attack_outcome",
    }
)
TEACHER_EXCLUDED_STEP_FIELDS = frozenset(
    {
        "reward_after_step",
        "done_after_step",
        "env_check_success_after_step",
        "info_success_after_step",
        "info_task_success_after_step",
        "info_is_success_after_step",
    }
)
DERIVED_FILES = (
    "derived_episode_metadata.json",
    "step_records_prefix.jsonl",
    "teacher_v2_labels.jsonl",
    "teacher_v2_episode_summary.json",
    "rgb_reference_manifest.jsonl",
    "source_binding.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def assert_sha256(path: Path, expected: str, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def require_new_output_root(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    return path


def write_report_sidecar(report_path: Path) -> Path:
    sidecar = report_path.with_name(report_path.name + ".sha256")
    sidecar.write_text(f"{sha256_file(report_path)}  {report_path.name}\n", encoding="ascii")
    return sidecar


def write_checksums(root: Path) -> tuple[str, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
        ),
        encoding="ascii",
    )
    sidecar = root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="ascii")
    return sha256_file(sums), sha256_file(sidecar)


def verify_checksums(root: Path) -> tuple[bool, str]:
    try:
        sums = root / "SHA256SUMS"
        sidecar = root / "SHA256SUMS.sha256"
        seen: set[str] = set()
        for line_no, line in enumerate(sums.read_text(encoding="ascii").splitlines(), 1):
            digest, rel = line.split("  ", 1)
            if rel in seen or Path(rel).is_absolute() or ".." in Path(rel).parts:
                return False, f"unsafe or duplicate SHA256SUMS entry at line {line_no}"
            seen.add(rel)
            path = root / rel
            if not path.is_file() or sha256_file(path) != digest:
                return False, f"SHA256SUMS mismatch: {rel}"
        digest, name = sidecar.read_text(encoding="ascii").strip().split("  ", 1)
        if name != "SHA256SUMS" or sha256_file(sums) != digest:
            return False, "SHA256SUMS.sha256 mismatch"
        return True, "PASS"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_index"]), int(row["state_id"])


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 2000:
        raise ValueError(f"R8W master manifest must contain 2000 rows, found {len(rows)}")
    ids = [identity(row) for row in rows]
    parents = [str(row["parent_key"]) for row in rows]
    if len(set(ids)) != 2000 or len(set(parents)) != 2000:
        raise ValueError("R8W master manifest contains duplicate identity or parent_key")
    selected = [row for row in rows if row.get("suite") in TARGET_SUITES]
    if len(selected) != 1500:
        raise ValueError(f"OGS source cardinality mismatch: {len(selected)}")
    suites = Counter(str(row["suite"]) for row in selected)
    if suites != Counter({suite: 500 for suite in TARGET_SUITES}):
        raise ValueError(f"OGS source suite counts mismatch: {dict(suites)}")
    if any(int(row.get("max_steps", -1)) != SOURCE_HORIZON for row in selected):
        raise ValueError("OGS source max_steps must be exactly 300")
    cohorts = Counter(str(row["cohort"]) for row in selected)
    if cohorts != Counter(EXPECTED_COHORT_COUNTS):
        raise ValueError(f"OGS source cohort counts mismatch: {dict(cohorts)}")
    for suite in TARGET_SUITES:
        suite_cohorts = Counter(
            str(row["cohort"]) for row in selected if row["suite"] == suite
        )
        if suite_cohorts != Counter(EXPECTED_SUITE_COHORT_COUNTS):
            raise ValueError(f"{suite} cohort counts mismatch: {dict(suite_cohorts)}")


@dataclass
class WorkerBinding:
    worker_id: str
    suite: str
    shard_id: str
    physical_gpu: int
    shard_manifest_sha256: str
    worker_receipt_path: Path
    worker_receipt_sha256: str
    collection_report_path: Path
    collection_report_sha256: str
    artifact_manifest_path: Path
    artifact_manifest_sha256: str
    artifact_hashes: dict[str, str]


@dataclass
class SourceContext:
    plan_report_path: Path
    plan_report_sha256: str
    master_manifest_path: Path
    master_manifest_sha256: str
    run_root: Path
    scheduler_report_path: Path
    scheduler_report_sha256: str
    source_git_head: str
    plan: dict[str, Any]
    scheduler: dict[str, Any]
    rows: list[dict[str, Any]]
    workers: dict[str, WorkerBinding]
    scheduler_caveat: str

    @property
    def ogs_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["suite"] in TARGET_SUITES]


def _artifact_hash_map(path: Path) -> dict[str, str]:
    rows = read_jsonl(path)
    result: dict[str, str] = {}
    for row in rows:
        rel = str(row["path"])
        if rel in result:
            raise ValueError(f"duplicate worker artifact entry: {rel}")
        result[rel] = str(row["sha256"])
    return result


def load_source_context(
    *,
    plan_report_path: Path,
    expected_plan_report_sha256: str,
    master_manifest_path: Path,
    expected_master_manifest_sha256: str,
    run_root: Path,
    scheduler_report_path: Path,
    expected_scheduler_report_sha256: str,
    expected_source_git_head: str,
) -> SourceContext:
    plan_report_path = plan_report_path.resolve()
    master_manifest_path = master_manifest_path.resolve()
    run_root = run_root.resolve()
    scheduler_report_path = scheduler_report_path.resolve()
    assert_sha256(plan_report_path, expected_plan_report_sha256, "R8W plan report")
    assert_sha256(master_manifest_path, expected_master_manifest_sha256, "R8W master manifest")
    assert_sha256(scheduler_report_path, expected_scheduler_report_sha256, "R8W scheduler report")
    if scheduler_report_path != run_root / "c2g_r8w_16worker_scheduler_report.json":
        raise ValueError("scheduler report must be the explicitly bound report under source run root")

    plan = read_json(plan_report_path)
    scheduler = read_json(scheduler_report_path)
    if plan.get("status") != R8W_PLAN_PASS_STATUS or plan.get("episode_count") != 2000:
        raise ValueError("R8W plan report is not the accepted 2000-episode plan")
    if Path(str(plan.get("manifest", ""))).resolve() != master_manifest_path:
        raise ValueError("R8W plan report points to a different master manifest")
    if plan.get("manifest_sha256") != expected_master_manifest_sha256:
        raise ValueError("R8W plan report master manifest SHA mismatch")
    if plan.get("expected_git_commit") != expected_source_git_head:
        raise ValueError("R8W plan source git head mismatch")
    if scheduler.get("git_head") != expected_source_git_head:
        raise ValueError("R8W scheduler source git head mismatch")
    if scheduler.get("plan_report_sha256") != expected_plan_report_sha256:
        raise ValueError("R8W scheduler plan-report binding mismatch")

    rows = read_jsonl(master_manifest_path)
    validate_manifest_rows(rows)
    selected_workers = {str(row["assigned_worker_id"]) for row in rows if row["suite"] in TARGET_SUITES}
    shard_by_worker = {str(row["worker_id"]): row for row in plan.get("shards", [])}
    if set(selected_workers) - set(shard_by_worker):
        raise ValueError("R8W plan is missing OGS shard declarations")

    receipt_index = {
        str(row["worker_id"]): row for row in scheduler.get("worker_receipts", [])
    }
    failures = {str(row.get("worker_id")) for row in scheduler.get("worker_failures", [])}
    unlaunched = {str(item) for item in scheduler.get("unlaunched_workers", [])}
    if failures & selected_workers or unlaunched & selected_workers:
        raise ValueError("R8W scheduler has an OGS worker failure or unlaunched worker")
    if set(selected_workers) - set(receipt_index):
        raise ValueError("R8W scheduler is missing OGS worker receipts")
    scheduler_caveat = ""
    if scheduler.get("status") != "PASS_C2G_R8W_16WORKER_COLLECTION":
        if failures != {"g7_l10"} or unlaunched:
            raise ValueError("R8W scheduler HOLD is not isolated to the accepted g7_l10 failure")
        scheduler_caveat = "GLOBAL_R8W_HOLD_ISOLATED_TO_G7_L10; OGS_12_WORKERS_VERIFIED"

    workers: dict[str, WorkerBinding] = {}
    for worker_id in sorted(selected_workers):
        shard = shard_by_worker[worker_id]
        suite = str(shard["suite"])
        worker_root = run_root / "workers" / worker_id
        receipt_path = worker_root / "worker_receipt.json"
        collection_root = worker_root / "collection"
        report_path = collection_root / "c2g_r8w_collection_report.json"
        artifact_path = collection_root / "c2g_r8w_collection_artifacts.jsonl"
        receipt = read_json(receipt_path)
        report = read_json(report_path)
        expected_receipt_sha = str(receipt_index[worker_id]["receipt_sha256"])
        if sha256_file(receipt_path) != expected_receipt_sha:
            raise ValueError(f"{worker_id} worker receipt SHA mismatch")
        if (
            receipt.get("status") != SOURCE_WORKER_PASS
            or receipt.get("git_head") != expected_source_git_head
            or int(receipt.get("episode_count", -1)) != 125
            or int(receipt.get("runtime_valid_episode_count", -1)) != 125
        ):
            raise ValueError(f"{worker_id} worker receipt is not a 125/125 PASS")
        if (
            report.get("status") != SOURCE_COLLECTION_PASS
            or report.get("suite") != suite
            or int(report.get("episode_count", -1)) != 125
            or int(report.get("runtime_valid_episode_count", -1)) != 125
        ):
            raise ValueError(f"{worker_id} collection report is not a 125/125 PASS")
        if Path(str(report.get("artifact_manifest", ""))).resolve() != artifact_path:
            raise ValueError(f"{worker_id} artifact manifest path mismatch")
        artifact_sha = sha256_file(artifact_path)
        if report.get("artifact_manifest_sha256") != artifact_sha:
            raise ValueError(f"{worker_id} artifact manifest SHA mismatch")
        workers[worker_id] = WorkerBinding(
            worker_id=worker_id,
            suite=suite,
            shard_id=str(shard["shard_id"]),
            physical_gpu=int(shard["physical_gpu"]),
            shard_manifest_sha256=str(shard["manifest_sha256"]),
            worker_receipt_path=receipt_path,
            worker_receipt_sha256=expected_receipt_sha,
            collection_report_path=report_path,
            collection_report_sha256=sha256_file(report_path),
            artifact_manifest_path=artifact_path,
            artifact_manifest_sha256=artifact_sha,
            artifact_hashes=_artifact_hash_map(artifact_path),
        )
    return SourceContext(
        plan_report_path=plan_report_path,
        plan_report_sha256=expected_plan_report_sha256,
        master_manifest_path=master_manifest_path,
        master_manifest_sha256=expected_master_manifest_sha256,
        run_root=run_root,
        scheduler_report_path=scheduler_report_path,
        scheduler_report_sha256=expected_scheduler_report_sha256,
        source_git_head=expected_source_git_head,
        plan=plan,
        scheduler=scheduler,
        rows=rows,
        workers=workers,
        scheduler_caveat=scheduler_caveat,
    )


@dataclass
class SourceEpisode:
    row: dict[str, Any]
    episode_dir: Path
    metadata: dict[str, Any]
    steps: list[dict[str, Any]]
    rgb_manifest: list[dict[str, Any]]
    receipt: dict[str, Any]
    worker: WorkerBinding


def source_episode_dir(context: SourceContext, row: Mapping[str, Any]) -> Path:
    return (
        context.run_root
        / "workers"
        / str(row["assigned_worker_id"])
        / "collection"
        / "episodes"
        / str(row["suite"])
        / str(row["parent_key"])
    )


def _verify_worker_artifact_entries(source: SourceEpisode) -> None:
    collection_root = source.worker.artifact_manifest_path.parent
    for name in ("episode_metadata.json", "step_records.jsonl", "rgb_manifest.jsonl", "episode_receipt.json"):
        path = source.episode_dir / name
        rel = path.relative_to(collection_root).as_posix()
        if source.worker.artifact_hashes.get(rel) != sha256_file(path):
            raise ValueError(f"worker artifact manifest mismatch: {rel}")


def validate_source_episode(
    context: SourceContext,
    row: Mapping[str, Any],
    *,
    verify_rgb: bool = True,
) -> SourceEpisode:
    row = dict(row)
    worker = context.workers[str(row["assigned_worker_id"])]
    episode_dir = source_episode_dir(context, row)
    valid, reason = validate_episode_receipt(
        episode_dir,
        expected_parent_key=str(row["parent_key"]),
        expected_worker_id=worker.worker_id,
        expected_shard_id=worker.shard_id,
        expected_git_head=context.source_git_head,
        expected_manifest_sha=worker.shard_manifest_sha256,
    )
    if not valid:
        raise ValueError(f"source episode receipt failed: {reason}")
    metadata = read_json(episode_dir / "episode_metadata.json")
    steps = read_jsonl(episode_dir / "step_records.jsonl")
    rgb_manifest = read_jsonl(episode_dir / "rgb_manifest.jsonl")
    receipt = read_json(episode_dir / "episode_receipt.json")
    for key in ("suite", "task_index", "state_id", "parent_key", "cohort", "split"):
        if metadata.get(key) != row.get(key):
            raise ValueError(f"source metadata identity mismatch: {key}")
    if (
        metadata.get("runtime_valid") is not True
        or metadata.get("condition") != "CLEAN"
        or metadata.get("git_commit") != context.source_git_head
        or metadata.get("git_clean") is not True
        or int(metadata.get("max_steps", -1)) != SOURCE_HORIZON
        or metadata.get("post_step_outcome_complete") is not True
        or metadata.get("post_step_outcome_schema_version") != POST_STEP_SCHEMA
    ):
        raise ValueError("source metadata clean/runtime/provenance contract mismatch")
    indices = [int(step["step"]) for step in steps]
    if not steps or indices != list(range(len(steps))) or len(steps) > SOURCE_HORIZON:
        raise ValueError("source steps are empty, discontinuous, or exceed source horizon")
    if len(rgb_manifest) != len(steps):
        raise ValueError("source RGB/step cardinality mismatch")
    for index, step in enumerate(steps):
        if step.get("teacher_schema_version") != SOURCE_STEP_SCHEMA:
            raise ValueError("source Teacher-v2 raw schema mismatch")
        if not post_step_complete(step):
            raise ValueError("source post-step schema incomplete")
        if not (
            finite_vector(step.get("features_25d"), 25)
            and finite_vector(step.get("clean_policy_intent_9d"), 9)
            and finite_vector(step.get("clean_action_raw_7d"), 7)
            and finite_vector(step.get("applied_action_7d"), 7)
        ):
            raise ValueError("source feature/action vector is incomplete or non-finite")
        expected_name = f"frame_{index:06d}.png"
        if str(rgb_manifest[index].get("path")) != expected_name:
            raise ValueError("source RGB manifest is not step aligned")
        if verify_rgb:
            # validate_episode_receipt already rehashed every frame. Keep this
            # pass structural so each build/audit hashes the source RGB once.
            frame = episode_dir / "rgb" / expected_name
            if not frame.is_file() or frame.stat().st_size != int(rgb_manifest[index]["bytes"]):
                raise ValueError("source RGB frame missing or wrong size")
    source = SourceEpisode(row, episode_dir, metadata, steps, rgb_manifest, receipt, worker)
    _verify_worker_artifact_entries(source)
    return source


@dataclass
class PrefixResult:
    rows: list[dict[str, Any]]
    official_horizon: int
    canonical_success: bool
    first_success_step: int | None
    termination_reason: str
    late_success_in_extended_source: bool


def derive_official_prefix(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    official_horizon: int,
) -> PrefixResult:
    if official_horizon <= 0:
        raise ValueError("official_horizon must be positive")
    ordered = sorted((dict(row) for row in step_rows), key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    if steps != list(range(len(ordered))):
        raise ValueError("source policy steps must be contiguous from zero")
    prefix = [row for row in ordered if int(row["step"]) < official_horizon]
    success_steps = [
        int(row["step"])
        for row in prefix
        if row.get("env_check_success_after_step") is True
    ]
    if success_steps:
        first_success = min(success_steps)
        prefix = [row for row in prefix if int(row["step"]) <= first_success]
        success = True
        termination = "ENV_CHECK_SUCCESS"
    else:
        first_success = None
        success = False
        termination = f"MAX_POLICY_STEPS_AT_{official_horizon}"
        if len(prefix) != official_horizon:
            raise ValueError(
                "source trajectory ended before the official horizon without canonical success"
            )
    late_success = any(
        int(row["step"]) >= official_horizon
        and row.get("env_check_success_after_step") is True
        for row in ordered
    )
    return PrefixResult(prefix, official_horizon, success, first_success, termination, late_success)


def _teacher_metadata(source: SourceEpisode, prefix: PrefixResult, r8z_head: str) -> dict[str, Any]:
    metadata = dict(source.metadata)
    metadata.update(
        {
            "schema": R8Z_SCHEMA,
            "max_steps": prefix.official_horizon,
            "official_horizon": prefix.official_horizon,
            "source_horizon": SOURCE_HORIZON,
            "n_steps": len(prefix.rows),
            "clean_success_observed": prefix.canonical_success,
            "clean_success_first_step": prefix.first_success_step,
            "final_env_check_success": prefix.canonical_success,
            "termination_reason": prefix.termination_reason,
            "done_first_step": next(
                (int(row["step"]) for row in prefix.rows if row.get("done_after_step") is True),
                None,
            ),
            "reward_sum": sum(float(row["reward_after_step"]) for row in prefix.rows),
            "reward_max": max(float(row["reward_after_step"]) for row in prefix.rows),
            "reward_nonzero_step_count": sum(
                float(row["reward_after_step"]) != 0.0 for row in prefix.rows
            ),
            "late_success_in_extended_source": prefix.late_success_in_extended_source,
            "derived_from_source_git_head": source.metadata["git_commit"],
            "r8z_code_head": r8z_head,
            "uses_source_final_outcome_for_teacher": False,
            "uses_future_step_for_teacher": False,
            "uses_attack_outcome": False,
        }
    )
    return metadata


def rebuild_teacher_labels(
    prefix_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    teacher_rows = [
        {key: value for key, value in row.items() if key not in TEACHER_EXCLUDED_STEP_FIELDS}
        for row in prefix_rows
    ]
    teacher_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in TEACHER_EXCLUDED_METADATA_FIELDS
    }
    labels = build_clean_teacher_episode(
        teacher_rows,
        teacher_metadata,
        thresholds=CleanTeacherThresholds(burst_length=BURST_LENGTH),
    )
    source_by_step = {int(row["step"]): row for row in prefix_rows}
    for label in labels:
        source = source_by_step[int(label["step"])]
        label["y_manipulation_progress_active"] = label[
            "y_lift_transport_or_constraint"
        ]
        label["y_attack_start_B"] = label["y_attack_start_b"]
        label["active_goal_event_index"] = source.get("active_subgoal_index")
        label["official_horizon"] = int(metadata["official_horizon"])
        label["burst_length"] = BURST_LENGTH
    return labels


def label_summary(
    labels: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    known = [row["label_known_mask"] is True for row in labels]
    critical = [
        known[index] and row["y_gripper_critical_window"] is True
        for index, row in enumerate(labels)
    ]
    return {
        "schema": R8Z_SCHEMA,
        "parent_key": metadata["parent_key"],
        "suite": metadata["suite"],
        "task_index": metadata["task_index"],
        "state_id": metadata["state_id"],
        "cohort": metadata["cohort"],
        "split": metadata["split"],
        "row_count": len(labels),
        "known_step_count": sum(known),
        "unknown_step_count": len(labels) - sum(known),
        "critical_active_step_count": sum(critical),
        "start_positive_step_count": sum(
            row.get("y_attack_start_b") is True for row in labels
        ),
        "burst_feasible_step_count": sum(
            row.get("y_burst_feasible") is True for row in labels
        ),
        "release_safe_step_count": sum(
            row.get("label_known_mask") is True and row.get("y_release_safe") is True
            for row in labels
        ),
        "target_grounding_known_step_count": sum(
            row.get("label_known_mask") is True
            and row.get("y_target_relevant") is not None
            for row in labels
        ),
        "start_positive_episode": any(row.get("y_attack_start_b") is True for row in labels),
        "burst_feasible_episode": any(row.get("y_burst_feasible") is True for row in labels),
        "fully_known_hard_negative_episode": all(known) and not any(critical),
        "reason_code_counts": dict(
            sorted(Counter(str(row["teacher_reason_code"]) for row in labels).items())
        ),
        "derived_canonical_success": metadata["clean_success_observed"],
        "late_success_in_extended_source": metadata[
            "late_success_in_extended_source"
        ],
    }


def _source_binding(
    context: SourceContext,
    source: SourceEpisode,
    prefix: PrefixResult,
    r8z_head: str,
) -> dict[str, Any]:
    episode_dir = source.episode_dir
    return {
        "schema": R8Z_SCHEMA,
        "source_episode_path": str(episode_dir),
        "source_parent_key": source.row["parent_key"],
        "source_suite": source.row["suite"],
        "source_task_index": source.row["task_index"],
        "source_state_id": source.row["state_id"],
        "source_cohort": source.row["cohort"],
        "source_split": source.row["split"],
        "source_physical_gpu": source.row["assigned_physical_gpu"],
        "source_worker_id": source.row["assigned_worker_id"],
        "source_git_head": context.source_git_head,
        "source_plan_report": str(context.plan_report_path),
        "source_plan_report_sha256": context.plan_report_sha256,
        "source_master_manifest": str(context.master_manifest_path),
        "source_master_manifest_sha256": context.master_manifest_sha256,
        "source_scheduler_report": str(context.scheduler_report_path),
        "source_scheduler_report_sha256": context.scheduler_report_sha256,
        "source_manifest_sha256": source.worker.shard_manifest_sha256,
        "source_episode_metadata_sha256": sha256_file(
            episode_dir / "episode_metadata.json"
        ),
        "source_step_records_sha256": sha256_file(episode_dir / "step_records.jsonl"),
        "source_rgb_manifest_sha256": sha256_file(episode_dir / "rgb_manifest.jsonl"),
        "source_artifact_manifest": str(source.worker.artifact_manifest_path),
        "source_artifact_manifest_sha256": source.worker.artifact_manifest_sha256,
        "source_receipt_sha256": sha256_file(episode_dir / "episode_receipt.json"),
        "official_horizon": prefix.official_horizon,
        "source_horizon": SOURCE_HORIZON,
        "r8z_code_head": r8z_head,
        "source_scheduler_caveat": context.scheduler_caveat,
    }


def _rgb_references(source: SourceEpisode, prefix: PrefixResult) -> list[dict[str, Any]]:
    result = []
    for row in prefix.rows:
        step = int(row["step"])
        source_manifest = source.rgb_manifest[step]
        source_path = source.episode_dir / "rgb" / str(source_manifest["path"])
        result.append(
            {
                "policy_step": step,
                "source_rgb_path": str(source_path),
                "source_rgb_sha256": str(source_manifest["sha256"]),
                "source_rgb_bytes": int(source_manifest["bytes"]),
                "source_artifact_manifest_sha256": source.worker.artifact_manifest_sha256,
                "allowed_by_official_horizon": step < prefix.official_horizon,
            }
        )
    return result


def derived_episode_dir(output_root: Path, row: Mapping[str, Any]) -> Path:
    return output_root / "episodes" / str(row["suite"]) / str(row["parent_key"])


def derive_episode(
    context: SourceContext,
    row: Mapping[str, Any],
    *,
    output_root: Path,
    r8z_head: str,
) -> dict[str, Any]:
    source = validate_source_episode(context, row, verify_rgb=True)
    horizon = official_max_policy_steps(str(row["suite"]))
    prefix = derive_official_prefix(source.steps, official_horizon=horizon)
    metadata = _teacher_metadata(source, prefix, r8z_head)
    labels = rebuild_teacher_labels(prefix.rows, metadata)
    if len(labels) != len(prefix.rows):
        raise ValueError("Teacher-v2 label cardinality mismatch")
    summary = label_summary(labels, metadata)
    binding = _source_binding(context, source, prefix, r8z_head)
    rgb_references = _rgb_references(source, prefix)
    episode_dir = derived_episode_dir(output_root, row)
    if episode_dir.exists():
        raise FileExistsError(episode_dir)
    episode_dir.mkdir(parents=True)
    paths = {
        "metadata": episode_dir / "derived_episode_metadata.json",
        "steps": episode_dir / "step_records_prefix.jsonl",
        "labels": episode_dir / "teacher_v2_labels.jsonl",
        "summary": episode_dir / "teacher_v2_episode_summary.json",
        "rgb": episode_dir / "rgb_reference_manifest.jsonl",
        "binding": episode_dir / "source_binding.json",
    }
    write_json(paths["metadata"], metadata)
    write_jsonl(paths["steps"], prefix.rows)
    write_jsonl(paths["labels"], labels)
    write_json(paths["summary"], summary)
    write_jsonl(paths["rgb"], rgb_references)
    write_json(paths["binding"], binding)
    receipt = {
        "schema": R8Z_RECEIPT_SCHEMA,
        "status": "PASS_C2G_R8Z_EPISODE_DERIVATION",
        "parent_key": row["parent_key"],
        "suite": row["suite"],
        "task_index": row["task_index"],
        "state_id": row["state_id"],
        "cohort": row["cohort"],
        "split": row["split"],
        "official_horizon": horizon,
        "source_horizon": SOURCE_HORIZON,
        "r8z_code_head": r8z_head,
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "derived_files": {
            path.name: sha256_file(path) for path in paths.values()
        },
        "runtime_valid": True,
        "completion_timestamp": utc_now(),
    }
    receipt_path = episode_dir / "episode_receipt.json"
    write_json(receipt_path, receipt)
    return {
        "suite": row["suite"],
        "task_index": row["task_index"],
        "state_id": row["state_id"],
        "parent_key": row["parent_key"],
        "cohort": row["cohort"],
        "split": row["split"],
        "official_horizon": horizon,
        "source_horizon": SOURCE_HORIZON,
        "derived_step_count": len(prefix.rows),
        "label_row_count": len(labels),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "label_file": str(paths["labels"]),
        "label_file_sha256": sha256_file(paths["labels"]),
        "source_binding": str(paths["binding"]),
        "source_binding_sha256": sha256_file(paths["binding"]),
        "summary": summary,
    }


def canary_rank(parent_key: str, salt: str = CANARY_SALT) -> str:
    return hashlib.sha256(f"{parent_key}{salt}".encode("utf-8")).hexdigest()


def select_canary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    salt: str = CANARY_SALT,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for suite in TARGET_SUITES:
        train = [
            dict(row)
            for row in rows
            if row.get("suite") == suite and row.get("cohort") == TRAIN_COHORT
        ]
        by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in train:
            by_task[int(row["task_index"])].append(row)
        ranked_tasks = sorted(
            by_task,
            key=lambda task: (
                min(canary_rank(str(row["parent_key"]), salt) for row in by_task[task]),
                task,
            ),
        )
        if len(ranked_tasks) < 2:
            raise ValueError(f"{suite} has fewer than two train tasks")
        for task in ranked_tasks[:2]:
            task_rows = sorted(
                by_task[task],
                key=lambda row: (canary_rank(str(row["parent_key"]), salt), str(row["parent_key"])),
            )
            if len(task_rows) < 2:
                raise ValueError(f"{suite} task {task} has fewer than two train states")
            selected.extend(task_rows[:2])
    if len(selected) != 12 or len({identity(row) for row in selected}) != 12:
        raise ValueError("R8Z canary selection did not close at 12 unique identities")
    return selected


def verify_git_head(expected_head: str, repo: Path) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != expected_head:
        raise ValueError(f"executed git head mismatch: expected={expected_head} actual={actual}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    if dirty:
        raise ValueError("R8Z execution requires a clean worktree")
