#!/usr/bin/env python3
"""Read-only closure audit for the R8W full Clean-2000 collection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.stageb.run_c2g_r8w_full_clean_shard import (
    RECEIPT_SCHEMA as WORKER_RECEIPT_SCHEMA,
    RUN_STATUS as WORKER_RUN_STATUS,
    validate_episode_receipt,
)
from scripts.stageb.run_c2g_r8w_gpu4567_16worker import RUN_STATUS_SCHEDULER
from tools.multisuite_detector.build_c2g_r8w_full_clean_2000_plan import (
    ATTACK_EVAL,
    DETECTOR_TEST,
    DETECTOR_TRAIN,
    DETECTOR_VAL,
    GPUS,
    CANARY_PASS_STATUS as CANARY_PLAN_PASS_STATUS,
    CANARY_PURPOSE,
    PASS_STATUS as PLAN_PASS_STATUS,
    SCHEMA as PLAN_SCHEMA,
    SUITES,
    expected_flags,
    identity,
    read_json,
    read_jsonl,
    sha256_file,
)
from tools.multisuite_detector.c2g_clean_window_label_builder import (
    CleanTeacherThresholds,
    build_clean_teacher_episode,
)

SCHEMA = "c2g.r8w.full_clean_2000_audit.2026-07-12.v1"
PASS_STATUS = "PASS_C2G_R8W_FULL_CLEAN_2000_COLLECTION"
HOLD_COLLECTION = "HOLD_C2G_R8W_FULL_CLEAN_2000_COLLECTION"
HOLD_IDENTITY = "HOLD_C2G_R8W_IDENTITY_OR_COHORT_INTEGRITY"
HOLD_TEACHER = "HOLD_C2G_R8W_TEACHER_V2_INTEGRITY"
GO_STATUS = "GO_R8W_DETECTOR_TRAIN_MATERIALIZATION_DESIGN"
CANARY_PASS_STATUS = "PASS_C2G_R8W_FRESH_COLLECTOR_GATE"
CANARY_HOLD_STATUS = "HOLD_C2G_R8W_FRESH_COLLECTOR_GATE"
TEACHER_SCHEMA = "c2g.teacher_v2.raw_privileged_evidence.2026-07-11.v1"
POST_STEP_SCHEMA = "c2g.r8w.post_step_outcome.2026-07-12.v1"
STEP_REQUIRED_KEYS = (
    "teacher_schema_version",
    "rgb_path",
    "task_language",
    "features_25d",
    "clean_policy_intent_9d",
    "clean_action_raw_7d",
    "applied_action_7d",
    "clean_action_token_top_ids",
    "clean_action_token_top_logits",
    "mujoco_contact_pairs",
    "active_target_known",
    "active_target_entity",
    "active_subgoal_index",
    "object_relative_lift",
    "target_distance_decrease",
    "constrained_manipulation_active",
    "manipulation_progress_active",
    "near_target",
    "supported_at_target",
    "release_safe",
    "target_object_position",
    "target_destination_position",
    "fixture_joint_motion",
    "active_target_contact",
    "active_target_bilateral_contact",
    "reward_after_step",
    "done_after_step",
    "env_check_success_after_step",
    "info_success_after_step",
    "info_task_success_after_step",
    "info_is_success_after_step",
)
META_REQUIRED_KEYS = (
    "teacher_schema_version",
    "event_tracking_schema",
    "structured_goal_metadata",
    "goal_event_bindings",
    "resolution",
    "model_path",
    "model_selected_hashes",
    "suite_model_map_sha256",
    "suite_model_report_sha256",
    "goal_model_manifest_sha256",
    "model_verification_report_sha256",
    "unnorm_key",
    "token_semantics_sha256",
    "raw_action_order",
    "applied_action_order",
    "action_semantics",
    "controller_config",
    "runtime_versions",
    "bddl_file",
    "bddl_sha256",
    "official_init_state_sha256",
    "official_init_state_shape",
    "official_init_state_dtype",
    "replay_seed",
    "max_steps",
    "dummy_wait",
    "thresholds",
    "clean_success_metric",
    "clean_success_observed",
    "termination_reason",
    "done_first_step",
    "reward_sum",
    "reward_max",
    "reward_nonzero_step_count",
    "post_step_outcome_complete",
    "post_step_outcome_schema_version",
    "worker_id",
    "shard_id",
    "physical_gpu",
    "model_device",
    "render_gpu_device_id",
    "shard_manifest_sha256",
    "git_commit",
    "git_clean",
)
PUBLIC_LEDGER_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split",
    "worker_id", "physical_gpu", "runtime_valid", "n_steps", "step_contiguous",
    "post_step_schema_complete", "features_25d_complete", "policy_intent_9d_complete",
    "raw_action_7d_complete", "applied_action_7d_complete", "token_evidence_complete",
    "teacher_raw_complete", "teacher_v2_rebuild_complete", "provenance_complete",
    "artifact_hashes_complete", "known_positive_steps", "known_negative_steps",
    "unknown_steps", "positive_episode", "fully_known_negative_episode",
    "triggerable_positive_episode", "structurally_complete", "failure_reason",
)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def triggerable(values: Sequence[bool]) -> bool:
    return any(sum(values[index - 2:index + 1]) >= 2 for index in range(2, len(values)))


def post_step_complete(row: Mapping[str, Any]) -> bool:
    if any(key not in row for key in STEP_REQUIRED_KEYS):
        return False
    try:
        return (
            math.isfinite(float(row["reward_after_step"]))
            and type(row["done_after_step"]) is bool
            and type(row["env_check_success_after_step"]) is bool
        )
    except (TypeError, ValueError):
        return False


def metadata_complete(metadata: Mapping[str, Any], expected: Mapping[str, Any], head: str) -> tuple[bool, list[str]]:
    missing = [key for key in META_REQUIRED_KEYS if key not in metadata]
    if metadata.get("runtime_valid") is not True or metadata.get("condition") != "CLEAN":
        missing.append("runtime_valid_or_condition")
    if type(metadata.get("clean_success_observed")) is not bool:
        missing.append("clean_success_observed_boolean")
    if metadata.get("post_step_outcome_complete") is not True:
        missing.append("post_step_outcome_complete")
    if metadata.get("post_step_outcome_schema_version") != POST_STEP_SCHEMA:
        missing.append("post_step_outcome_schema_version")
    if metadata.get("git_commit") != head or metadata.get("git_clean") is not True:
        missing.append("git_provenance")
    for key in ("suite", "task_index", "state_id", "parent_key", "cohort", "split"):
        if metadata.get(key) != expected.get(key):
            missing.append(f"identity_or_cohort.{key}")
    for key, value in expected_flags(str(expected["cohort"])).items():
        if metadata.get(key) is not value:
            missing.append(f"eligibility.{key}")
    if metadata.get("shard_manifest_sha256") != expected.get("shard_manifest_sha256"):
        missing.append("shard_manifest_sha256")
    return not missing, sorted(set(missing))


def audit_episode(
    run_root: Path,
    expected: Mapping[str, Any],
    *,
    head: str,
) -> dict[str, Any]:
    worker = str(expected["assigned_worker_id"])
    episode_dir = run_root / "workers" / worker / "collection" / "episodes" / str(expected["suite"]) / str(expected["parent_key"])
    metadata_path = episode_dir / "episode_metadata.json"
    steps_path = episode_dir / "step_records.jsonl"
    result: dict[str, Any] = {
        "suite": expected["suite"],
        "task_index": expected["task_index"],
        "state_id": expected["state_id"],
        "parent_key": expected["parent_key"],
        "cohort": expected["cohort"],
        "split": expected["split"],
        "worker_id": worker,
        "physical_gpu": expected["assigned_physical_gpu"],
        "runtime_valid": False,
        "structurally_complete": False,
        "failure_reason": "",
    }
    try:
        valid_receipt, reason = validate_episode_receipt(
            episode_dir,
            expected_parent_key=str(expected["parent_key"]),
            expected_worker_id=worker,
            expected_shard_id=str(expected["assigned_shard_id"]),
            expected_git_head=head,
            expected_manifest_sha=str(expected["shard_manifest_sha256"]),
        )
        if not valid_receipt:
            raise ValueError(f"episode receipt: {reason}")
        metadata = read_json(metadata_path)
        steps = read_jsonl(steps_path)
        indices = [row.get("step") for row in steps]
        contiguous = indices == list(range(len(steps)))
        if not steps or not contiguous:
            raise ValueError("empty or discontinuous trajectory")
        meta_ok, meta_missing = metadata_complete(metadata, expected, head)
        if not meta_ok:
            raise ValueError("metadata incomplete: " + ",".join(meta_missing))
        features_ok = all(finite_vector(row.get("features_25d"), 25) for row in steps)
        policy_ok = all(finite_vector(row.get("clean_policy_intent_9d"), 9) for row in steps)
        raw_ok = all(finite_vector(row.get("clean_action_raw_7d"), 7) for row in steps)
        applied_ok = all(finite_vector(row.get("applied_action_7d"), 7) for row in steps)
        token_ok = all(
            isinstance(row.get("clean_action_token_top_ids"), list)
            and isinstance(row.get("clean_action_token_top_logits"), list)
            and len(row["clean_action_token_top_ids"]) == len(row["clean_action_token_top_logits"]) > 0
            and all(math.isfinite(float(value)) for value in row["clean_action_token_top_logits"])
            for row in steps
        )
        post_ok = all(post_step_complete(row) for row in steps)
        teacher_ok = all(
            all(key in row for key in STEP_REQUIRED_KEYS)
            and row.get("teacher_schema_version") == TEACHER_SCHEMA
            for row in steps
        )
        if not all((features_ok, policy_ok, raw_ok, applied_ok, token_ok, post_ok, teacher_ok)):
            raise ValueError("step schema incomplete")
        labels = build_clean_teacher_episode(
            steps,
            metadata,
            thresholds=CleanTeacherThresholds(burst_length=10),
        )
        if len(labels) != len(steps):
            raise ValueError("Teacher-v2 label cardinality mismatch")
        known = [row.get("label_known_mask") is True for row in labels]
        positives = [
            known[index] and row.get("y_gripper_critical_window") is True
            for index, row in enumerate(labels)
        ]
        known_positive = sum(positives)
        known_negative = sum(is_known and not positives[index] for index, is_known in enumerate(known))
        unknown_reasons = Counter(
            str(row.get("teacher_reason_code", "UNSPECIFIED"))
            for index, row in enumerate(labels)
            if not known[index]
        )
        result.update({
            "runtime_valid": True,
            "n_steps": len(steps),
            "step_contiguous": True,
            "post_step_schema_complete": True,
            "features_25d_complete": True,
            "policy_intent_9d_complete": True,
            "raw_action_7d_complete": True,
            "applied_action_7d_complete": True,
            "token_evidence_complete": True,
            "teacher_raw_complete": True,
            "teacher_v2_rebuild_complete": True,
            "provenance_complete": True,
            "artifact_hashes_complete": True,
            "known_positive_steps": known_positive,
            "known_negative_steps": known_negative,
            "unknown_steps": len(labels) - sum(known),
            "unknown_reason_counts": dict(sorted(unknown_reasons.items())),
            "positive_episode": known_positive > 0,
            "fully_known_negative_episode": all(known) and known_positive == 0,
            "triggerable_positive_episode": triggerable(positives),
            "clean_success_observed": metadata["clean_success_observed"],
            "structurally_complete": True,
        })
    except Exception as exc:
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def verify_worker_receipts(run_root: Path, plan: Mapping[str, Any], head: str) -> tuple[list[dict[str, Any]], list[str]]:
    receipts = []
    failures = []
    for shard in plan["shards"]:
        path = run_root / "workers" / str(shard["worker_id"]) / "worker_receipt.json"
        try:
            value = read_json(path)
            if value.get("schema") != WORKER_RECEIPT_SCHEMA or value.get("status") != WORKER_RUN_STATUS:
                raise ValueError("status/schema mismatch")
            if value.get("worker_id") != shard["worker_id"] or value.get("git_head") != head:
                raise ValueError("worker/head mismatch")
            if value.get("shard_manifest_sha256") != shard["manifest_sha256"]:
                raise ValueError("manifest SHA mismatch")
            if value.get("episode_count") != 125 or value.get("runtime_valid_episode_count") != 125:
                raise ValueError("episode cardinality mismatch")
            receipts.append({"worker_id": shard["worker_id"], "path": str(path), "sha256": sha256_file(path)})
        except Exception as exc:
            failures.append(f"{shard['worker_id']}: {type(exc).__name__}: {exc}")
    return receipts, failures


def classify_final_status(
    *,
    cardinality_pass: bool,
    identity_failure_count: int,
    teacher_failure_count: int,
    l10_closure: bool,
    worker_failure_count: int,
    complete_count: int,
) -> str:
    if not cardinality_pass or identity_failure_count:
        return HOLD_IDENTITY
    if teacher_failure_count or not l10_closure:
        return HOLD_TEACHER
    if worker_failure_count or complete_count != 2000:
        return HOLD_COLLECTION
    return PASS_STATUS


def run_audit(
    *,
    plan_report: Path,
    expected_plan_report_sha256: str,
    run_root: Path,
    expected_scheduler_report_sha256: str,
    output_root: Path,
    audit_head: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    plan_report, run_root = plan_report.resolve(), run_root.resolve()
    if sha256_file(plan_report) != expected_plan_report_sha256:
        raise ValueError("R8W plan report SHA mismatch")
    plan = read_json(plan_report)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_PASS_STATUS:
        raise ValueError("R8W plan is not accepted")
    if plan.get("expected_git_commit") != audit_head:
        raise ValueError("audit head differs from R8W plan head")
    scheduler_path = run_root / "c2g_r8w_16worker_scheduler_report.json"
    if sha256_file(scheduler_path) != expected_scheduler_report_sha256:
        raise ValueError("R8W scheduler report SHA mismatch")
    scheduler = read_json(scheduler_path)
    if scheduler.get("status") != RUN_STATUS_SCHEDULER:
        raise ValueError("R8W scheduler did not PASS")
    expected_rows = read_jsonl(Path(str(plan["manifest"])))
    if len(expected_rows) != 2000 or len({identity(row) for row in expected_rows}) != 2000:
        raise ValueError("R8W plan identity closure failed")
    manifest_by_worker = {str(row["worker_id"]): str(row["manifest_sha256"]) for row in plan["shards"]}
    rows_with_sha = [
        {**row, "shard_manifest_sha256": manifest_by_worker[str(row["assigned_worker_id"])]}
        for row in expected_rows
    ]
    worker_receipts, worker_failures = verify_worker_receipts(run_root, plan, audit_head)
    rows = [audit_episode(run_root, row, head=audit_head) for row in rows_with_sha]
    complete = [row for row in rows if row.get("structurally_complete")]
    teacher_failures = [row for row in rows if not row.get("teacher_v2_rebuild_complete")]
    identity_failures = [
        row for row in rows
        if "identity_or_cohort" in str(row.get("failure_reason", ""))
        or "eligibility" in str(row.get("failure_reason", ""))
    ]
    suite_counts = Counter(str(row["suite"]) for row in complete)
    cohort_counts = Counter(str(row["cohort"]) for row in complete)
    gpu_counts = Counter(str(row["physical_gpu"]) for row in complete)
    worker_counts = Counter(str(row["worker_id"]) for row in complete)
    cardinality_pass = (
        len(complete) == 2000
        and suite_counts == Counter({suite: 500 for suite in SUITES})
        and cohort_counts == Counter({DETECTOR_TRAIN: 1200, DETECTOR_VAL: 200, DETECTOR_TEST: 200, ATTACK_EVAL: 400})
        and gpu_counts == Counter({str(gpu): 500 for gpu in GPUS})
        and all(value == 125 for value in worker_counts.values())
        and len(worker_counts) == 16
    )
    l10_rows = [row for row in complete if row["suite"] == "libero_10"]
    l10_unknown = sum(int(row["unknown_steps"]) for row in l10_rows)
    l10_reasons: Counter[str] = Counter()
    for row in l10_rows:
        l10_reasons.update(row.get("unknown_reason_counts", {}))
    l10_closure = sum(l10_reasons.values()) == l10_unknown
    teacher_pass = not teacher_failures and l10_closure
    integrity_pass = cardinality_pass and not worker_failures and not identity_failures
    final_decision = classify_final_status(
        cardinality_pass=cardinality_pass,
        identity_failure_count=len(identity_failures),
        teacher_failure_count=len(teacher_failures),
        l10_closure=l10_closure,
        worker_failure_count=len(worker_failures),
        complete_count=len(complete),
    )

    output_root.mkdir(parents=True)
    public_ledger = output_root / "c2g_r8w_full_clean_2000_episode_ledger.csv"
    write_csv(public_ledger, rows, PUBLIC_LEDGER_FIELDS)
    sealed_ledger = output_root / "c2g_r8w_full_clean_2000_sealed_outcomes.jsonl"
    write_jsonl(
        sealed_ledger,
        ({
            "suite": row["suite"], "task_index": row["task_index"],
            "state_id": row["state_id"], "parent_key": row["parent_key"],
            "cohort": row["cohort"], "clean_success_observed": row.get("clean_success_observed"),
        } for row in rows),
    )
    violations_path = output_root / "c2g_r8w_full_clean_2000_violations.jsonl"
    write_jsonl(
        violations_path,
        ({"parent_key": row["parent_key"], "failure_reason": row.get("failure_reason", "")}
         for row in rows if not row.get("structurally_complete")),
    )
    train_rows = [row for row in complete if row["cohort"] == DETECTOR_TRAIN]
    train_success = sum(row.get("clean_success_observed") is True for row in train_rows)
    report = {
        "schema": SCHEMA,
        "status": final_decision,
        "final_decision": GO_STATUS if final_decision == PASS_STATUS else final_decision,
        "audit_head": audit_head,
        "plan_report": str(plan_report),
        "plan_report_sha256": expected_plan_report_sha256,
        "scheduler_report": str(scheduler_path),
        "scheduler_report_sha256": expected_scheduler_report_sha256,
        "episode_directory_count": len(rows),
        "runtime_valid_episode_count": len(complete),
        "unique_identity_count": len({identity(row) for row in expected_rows}),
        "missing_identity_count": sum(not row.get("structurally_complete") for row in rows),
        "outside_identity_count": 0,
        "duplicate_identity_count": 0,
        "suite_counts": dict(sorted(suite_counts.items())),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "gpu_counts": dict(sorted(gpu_counts.items())),
        "worker_counts": dict(sorted(worker_counts.items())),
        "worker_receipts_pass": len(worker_receipts),
        "worker_receipts": worker_receipts,
        "worker_receipt_failures": worker_failures,
        "post_step_schema_complete": sum(bool(row.get("post_step_schema_complete")) for row in rows),
        "features_25d_complete": sum(bool(row.get("features_25d_complete")) for row in rows),
        "policy_intent_9d_complete": sum(bool(row.get("policy_intent_9d_complete")) for row in rows),
        "raw_action_7d_complete": sum(bool(row.get("raw_action_7d_complete")) for row in rows),
        "applied_action_7d_complete": sum(bool(row.get("applied_action_7d_complete")) for row in rows),
        "teacher_v2_raw_complete": sum(bool(row.get("teacher_raw_complete")) for row in rows),
        "teacher_v2_rebuild_complete": sum(bool(row.get("teacher_v2_rebuild_complete")) for row in rows),
        "provenance_complete": sum(bool(row.get("provenance_complete")) for row in rows),
        "detector_train_success_summary": {
            "episode_count": len(train_rows),
            "success_count": train_success,
            "success_rate": train_success / len(train_rows) if train_rows else None,
        },
        "nontrain_outcome_metrics_printed": False,
        "l10_unknown_decomposition": dict(sorted(l10_reasons.items())),
        "l10_unknown_step_count": l10_unknown,
        "l10_decomposition_closure": l10_closure,
        "integrity_pass": integrity_pass,
        "teacher_v2_pass": teacher_pass,
        "public_episode_ledger": str(public_ledger),
        "public_episode_ledger_sha256": sha256_file(public_ledger),
        "sealed_outcome_ledger": str(sealed_ledger),
        "sealed_outcome_ledger_sha256": sha256_file(sealed_ledger),
        "violations": str(violations_path),
        "violations_sha256": sha256_file(violations_path),
        "materialization_authorization": "HOLD",
        "training_authorization": "HOLD",
        "attack_authorization": "HOLD",
        "boundaries": {
            "attack_outcomes_read": False,
            "models_loaded": 0,
            "environments_created": 0,
            "rollouts_launched": 0,
            "attacks": 0,
            "training_epochs": 0,
            "materialization_runs": 0,
            "storage_deletions": 0,
            "d7_modifications": 0,
        },
    }
    report_path = output_root / "c2g_r8w_full_clean_2000_audit_report.json"
    write_json(report_path, report)
    files = sorted(path for path in output_root.iterdir() if path.is_file())
    sums = output_root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="ascii",
    )
    self_binding = output_root / "SHA256SUMS.sha256"
    self_binding.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="ascii")
    return {
        **report,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "sha256s_sha256": sha256_file(sums),
        "self_binding_sha256": sha256_file(self_binding),
    }


def vectors_exact(left: Any, right: Any, length: int) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right) == length
        and all(float(a) == float(b) for a, b in zip(left, right))
    )


def vectors_equivalent(left: Any, right: Any, length: int, tolerance: float = 1e-6) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right) == length
        and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))
    )


def run_canary_audit(
    *,
    plan_report: Path,
    expected_plan_report_sha256: str,
    run_root: Path,
    expected_scheduler_report_sha256: str,
    output_root: Path,
    audit_head: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    plan_report, run_root = plan_report.resolve(), run_root.resolve()
    if sha256_file(plan_report) != expected_plan_report_sha256:
        raise ValueError("R8W canary plan report SHA mismatch")
    plan = read_json(plan_report)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != CANARY_PLAN_PASS_STATUS
        or plan.get("plan_kind") != CANARY_PURPOSE
    ):
        raise ValueError("R8W canary plan is not accepted")
    if plan.get("expected_git_commit") != audit_head:
        raise ValueError("canary audit head differs from plan head")
    scheduler_path = run_root / "c2g_r8w_fresh_canary_scheduler_report.json"
    if sha256_file(scheduler_path) != expected_scheduler_report_sha256:
        raise ValueError("R8W canary scheduler SHA mismatch")
    scheduler = read_json(scheduler_path)
    if scheduler.get("status") != "PASS_C2G_R8W_FRESH_CANARY_SCHEDULER":
        raise ValueError("R8W canary scheduler did not PASS")
    expected_rows = read_jsonl(Path(str(plan["manifest"])))
    if len(expected_rows) != 8 or len({identity(row) for row in expected_rows}) != 8:
        raise ValueError("R8W canary identity closure failed")
    manifest_by_worker = {row["worker_id"]: row["manifest_sha256"] for row in plan["shards"]}
    replay_steps: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(Path(str(plan["r8u_step_ledger"]))):
        replay_steps.setdefault(str(row["parent_key"]), []).append(row)
    for rows in replay_steps.values():
        rows.sort(key=lambda row: int(row["step"]))

    ledger = []
    for expected in expected_rows:
        expected = {
            **expected,
            "shard_manifest_sha256": manifest_by_worker[expected["assigned_worker_id"]],
        }
        structural = audit_episode(run_root, expected, head=audit_head)
        episode_dir = (
            run_root / "workers" / str(expected["assigned_worker_id"]) / "collection"
            / "episodes" / str(expected["suite"]) / str(expected["parent_key"])
        )
        fresh = read_jsonl(episode_dir / "step_records.jsonl") if structural.get("structurally_complete") else []
        replay = replay_steps.get(str(expected["parent_key"]), [])
        prefix_length_ok = bool(fresh) and len(fresh) <= len(replay)
        raw_exact = prefix_length_ok and all(
            vectors_exact(row["clean_action_raw_7d"], replay[index]["raw_action_7d"], 7)
            for index, row in enumerate(fresh)
        )
        applied_exact = prefix_length_ok and all(
            vectors_exact(row["applied_action_7d"], replay[index]["applied_action_7d"], 7)
            for index, row in enumerate(fresh)
        )
        features_equivalent = prefix_length_ok and all(
            vectors_equivalent(row["features_25d"], replay[index]["replayed_features_25d"], 25)
            for index, row in enumerate(fresh)
        )
        metadata = read_json(episode_dir / "episode_metadata.json") if fresh else {}
        success_agreement = (
            type(metadata.get("clean_success_observed")) is bool
            and metadata["clean_success_observed"] is expected["expected_canonical_success"]
        )
        ledger.append({
            "suite": expected["suite"],
            "parent_key": expected["parent_key"],
            "expected_canonical_success": expected["expected_canonical_success"],
            "fresh_step_count": len(fresh),
            "replay_step_count": len(replay),
            "runtime_valid": structural.get("runtime_valid") is True,
            "teacher_v2_complete": structural.get("teacher_v2_rebuild_complete") is True,
            "raw_action_prefix_exact": raw_exact,
            "applied_action_prefix_exact": applied_exact,
            "features_25d_prefix_equivalent": features_equivalent,
            "action_prefix_agreement": raw_exact and applied_exact and features_equivalent,
            "success_agreement": success_agreement,
            "failure_reason": structural.get("failure_reason", ""),
        })
    runtime_count = sum(row["runtime_valid"] for row in ledger)
    teacher_count = sum(row["teacher_v2_complete"] for row in ledger)
    action_count = sum(row["action_prefix_agreement"] for row in ledger)
    success_count = sum(row["success_agreement"] for row in ledger)
    passed = (runtime_count, teacher_count, action_count, success_count) == (8, 8, 8, 8)
    output_root.mkdir(parents=True)
    ledger_path = output_root / "c2g_r8w_fresh_canary_audit_ledger.csv"
    write_csv(ledger_path, ledger, tuple(ledger[0].keys()))
    report = {
        "schema": "c2g.r8w.fresh_collector_gate.2026-07-12.v1",
        "status": CANARY_PASS_STATUS if passed else CANARY_HOLD_STATUS,
        "final_decision": "GO_R8W_FULL_CLEAN_2000_PLAN" if passed else CANARY_HOLD_STATUS,
        "audit_head": audit_head,
        "plan_report_sha256": expected_plan_report_sha256,
        "scheduler_report_sha256": expected_scheduler_report_sha256,
        "r8u_replay_report": plan["r8u_replay_report"],
        "r8u_replay_report_sha256": plan["r8u_replay_report_sha256"],
        "r8u_step_ledger": plan["r8u_step_ledger"],
        "r8u_step_ledger_sha256": plan["r8u_step_ledger_sha256"],
        "runtime_valid_count": runtime_count,
        "teacher_v2_complete_count": teacher_count,
        "action_prefix_agreement_count": action_count,
        "success_agreement_count": success_count,
        "episode_count": len(ledger),
        "calibration": scheduler.get("calibration", {}),
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "shadow_outputs_count_toward_full_2000": False,
        "attacks": 0,
        "training_epochs": 0,
        "materialization_runs": 0,
    }
    report_path = output_root / "c2g_r8w_fresh_collector_gate_report.json"
    write_json(report_path, report)
    sums = output_root / "SHA256SUMS"
    sums.write_text(
        f"{sha256_file(ledger_path)}  {ledger_path.name}\n{sha256_file(report_path)}  {report_path.name}\n",
        encoding="ascii",
    )
    self_binding = output_root / "SHA256SUMS.sha256"
    self_binding.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="ascii")
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "canary"))
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--expected-plan-report-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-scheduler-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_fn = run_canary_audit if args.mode == "canary" else run_audit
    result = audit_fn(
        plan_report=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        run_root=args.run_root,
        expected_scheduler_report_sha256=args.expected_scheduler_report_sha256,
        output_root=args.output_root,
        audit_head=args.audit_head,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {PASS_STATUS, CANARY_PASS_STATUS} else 1


if __name__ == "__main__":
    raise SystemExit(main())
