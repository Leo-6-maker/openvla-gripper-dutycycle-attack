#!/usr/bin/env python3
"""Audit the complete 1,500-episode R8Z official-horizon OGS corpus."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.gripper_attack.c2g_clean_window_schema import validate_clean_teacher_row
from tools.multisuite_detector.build_c2g_r8z_ogs_official_views import (
    SUITE_BUILD_PASS,
    _train_health,
    add_source_arguments,
    source_context_from_args,
    source_provenance,
)
from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    BURST_LENGTH,
    CANONICAL_LABEL_FIELDS,
    COHORTS,
    EXPECTED_COHORT_COUNTS,
    EXPECTED_SUITE_COHORT_COUNTS,
    R8Z_RECEIPT_SCHEMA,
    R8Z_SCHEMA,
    SOURCE_HORIZON,
    TARGET_SUITES,
    TRAIN_COHORT,
    SourceContext,
    derive_official_prefix,
    derived_episode_dir,
    identity,
    label_summary,
    read_json,
    read_jsonl,
    rebuild_teacher_labels,
    require_new_output_root,
    sha256_file,
    validate_source_episode,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
    write_report_sidecar,
)


PASS_STATUS = "PASS_C2G_R8Z_OGS_OFFICIAL_HORIZON_LABELS"
HOLD_STATUS = "HOLD_C2G_R8Z_OGS_OFFICIAL_HORIZON_LABELS"
HOLD_LABEL_SUPPORT = "HOLD_LABEL_SUPPORT"
NEXT_STAGE = "HOLD_PENDING_R8Y_L10_520_AND_FOUR_SUITE_COMPOSITE"
PUBLIC_FORBIDDEN_METRIC_TOKENS = (
    "late_success",
    "canonical_success",
    "positive_rate",
    "success_rate",
    "reason_code_counts",
    "known_step_fraction",
    "unknown_step_fraction",
)


def assert_public_report_sealed(report: Mapping[str, Any]) -> None:
    for key in report:
        lower = str(key).lower()
        if key == "train_only_label_health":
            continue
        if any(token in lower for token in PUBLIC_FORBIDDEN_METRIC_TOKENS):
            raise ValueError(f"public report exposes nontrain-capable metric key: {key}")
    if report.get("nontrain_metrics_exposed") is not False:
        raise ValueError("suite report must declare nontrain_metrics_exposed=false")


def _verify_derived_receipt(episode_dir: Path, expected: Mapping[str, Any], r8z_head: str) -> dict[str, Any]:
    receipt_path = episode_dir / "episode_receipt.json"
    receipt = read_json(receipt_path)
    expected_values = {
        "schema": R8Z_RECEIPT_SCHEMA,
        "status": "PASS_C2G_R8Z_EPISODE_DERIVATION",
        "parent_key": expected["parent_key"],
        "suite": expected["suite"],
        "task_index": expected["task_index"],
        "state_id": expected["state_id"],
        "cohort": expected["cohort"],
        "split": expected["split"],
        "source_horizon": SOURCE_HORIZON,
        "r8z_code_head": r8z_head,
        "runtime_valid": True,
    }
    for key, value in expected_values.items():
        if receipt.get(key) != value:
            raise ValueError(f"derived receipt {key} mismatch")
    files = receipt.get("derived_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("derived receipt file map missing")
    for name, digest in files.items():
        path = episode_dir / str(name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"derived receipt hash mismatch: {name}")
    return receipt


def _compare_labels(actual: Sequence[Mapping[str, Any]], rebuilt: Sequence[Mapping[str, Any]]) -> None:
    if len(actual) != len(rebuilt):
        raise ValueError("Teacher-v2 rebuilt label cardinality mismatch")
    fields = (
        "step",
        "label_known_mask",
        "teacher_reason_code",
        "active_goal_event_index",
        "y_manipulation_progress_active",
        "y_attack_start_B",
        *CANONICAL_LABEL_FIELDS,
    )
    for index, (left, right) in enumerate(zip(actual, rebuilt)):
        for field in fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"Teacher-v2 rebuild mismatch at row {index}: {field}")


def validate_r8z_teacher_row(row: Mapping[str, Any]) -> None:
    """Validate the frozen row plus the R8Z presentation aliases.

    The frozen schema already owns ``y_attack_start_b``.  R8Z additionally emits
    the task-requested ``y_attack_start_B`` spelling, which must mirror the frozen
    field but must not be presented to the older leakage-token validator.
    """

    if row.get("y_attack_start_B") != row.get("y_attack_start_b"):
        raise ValueError("R8Z attack-start alias differs from frozen Teacher field")
    if row.get("y_manipulation_progress_active") != row.get(
        "y_lift_transport_or_constraint"
    ):
        raise ValueError("R8Z progress alias differs from frozen Teacher field")
    canonical = dict(row)
    canonical.pop("y_attack_start_B", None)
    validate_clean_teacher_row(canonical)


def validate_derived_episode(
    context: SourceContext,
    expected: Mapping[str, Any],
    *,
    suite_root: Path,
    r8z_head: str,
    expose_train_metrics: bool,
) -> dict[str, Any]:
    episode_dir = derived_episode_dir(suite_root, expected)
    receipt = _verify_derived_receipt(episode_dir, expected, r8z_head)
    metadata = read_json(episode_dir / "derived_episode_metadata.json")
    steps = read_jsonl(episode_dir / "step_records_prefix.jsonl")
    labels = read_jsonl(episode_dir / "teacher_v2_labels.jsonl")
    summary = read_json(episode_dir / "teacher_v2_episode_summary.json")
    references = read_jsonl(episode_dir / "rgb_reference_manifest.jsonl")
    binding = read_json(episode_dir / "source_binding.json")
    horizon = int(receipt["official_horizon"])
    if horizon != int(metadata.get("official_horizon", -1)):
        raise ValueError("derived metadata/receipt horizon mismatch")
    if horizon != {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300}[str(expected["suite"])]:
        raise ValueError("derived official horizon mismatch")
    for key in ("suite", "task_index", "state_id", "parent_key", "cohort", "split"):
        if metadata.get(key) != expected.get(key):
            raise ValueError(f"derived metadata identity mismatch: {key}")
    indices = [int(row["step"]) for row in steps]
    if not steps or indices != list(range(len(steps))) or any(step >= horizon for step in indices):
        raise ValueError("derived prefix is empty, discontinuous, or crosses horizon")
    if len(labels) != len(steps) or len(references) != len(steps):
        raise ValueError("derived steps/labels/RGB-reference cardinality mismatch")
    expected_prefix = derive_official_prefix(steps, official_horizon=horizon)
    if expected_prefix.canonical_success != metadata.get("clean_success_observed"):
        raise ValueError("derived canonical success was not recomputed from prefix")
    if expected_prefix.first_success_step != metadata.get("clean_success_first_step"):
        raise ValueError("derived first-success step mismatch")
    if expected_prefix.termination_reason != metadata.get("termination_reason"):
        raise ValueError("derived termination reason mismatch")
    if metadata.get("uses_source_final_outcome_for_teacher") is not False:
        raise ValueError("derived metadata does not reject source-final outcome use")
    if metadata.get("uses_future_step_for_teacher") is not False:
        raise ValueError("derived metadata does not reject future-step use")

    rebuilt = rebuild_teacher_labels(steps, metadata)
    _compare_labels(labels, rebuilt)
    for label in labels:
        validate_r8z_teacher_row(label)
        if label.get("uses_future_student_input") is not False:
            raise ValueError("Teacher-v2 label claims future student input")
        if label.get("y_burst_feasible") is True and int(label["step"]) > horizon - BURST_LENGTH:
            raise ValueError("burst-feasible label crosses the derived official horizon")

    source = validate_source_episode(context, expected, verify_rgb=True)
    source_files = {
        "source_episode_metadata_sha256": source.episode_dir / "episode_metadata.json",
        "source_step_records_sha256": source.episode_dir / "step_records.jsonl",
        "source_rgb_manifest_sha256": source.episode_dir / "rgb_manifest.jsonl",
        "source_receipt_sha256": source.episode_dir / "episode_receipt.json",
        "source_artifact_manifest_sha256": source.worker.artifact_manifest_path,
    }
    if binding.get("source_episode_path") != str(source.episode_dir):
        raise ValueError("source binding episode path mismatch")
    if binding.get("source_git_head") != context.source_git_head:
        raise ValueError("source binding git head mismatch")
    if binding.get("source_plan_report_sha256") != context.plan_report_sha256:
        raise ValueError("source binding plan SHA mismatch")
    if binding.get("source_master_manifest_sha256") != context.master_manifest_sha256:
        raise ValueError("source binding manifest SHA mismatch")
    if binding.get("source_scheduler_report_sha256") != context.scheduler_report_sha256:
        raise ValueError("source binding scheduler SHA mismatch")
    for field, path in source_files.items():
        if binding.get(field) != sha256_file(path):
            raise ValueError(f"source mutation or binding mismatch: {field}")
    if receipt.get("source_receipt_sha256") != binding.get("source_receipt_sha256"):
        raise ValueError("derived receipt/source binding mismatch")
    for step, reference in enumerate(references):
        expected_rgb = source.episode_dir / "rgb" / f"frame_{step:06d}.png"
        if (
            int(reference.get("policy_step", -1)) != step
            or Path(str(reference.get("source_rgb_path", ""))).resolve() != expected_rgb
            or reference.get("source_rgb_sha256") != source.rgb_manifest[step]["sha256"]
            or reference.get("source_artifact_manifest_sha256")
            != source.worker.artifact_manifest_sha256
            or reference.get("allowed_by_official_horizon") is not True
        ):
            raise ValueError("RGB source reference mismatch")
    rebuilt_summary = label_summary(rebuilt, metadata)
    if summary != rebuilt_summary:
        raise ValueError("Teacher-v2 episode summary mismatch")
    result = {
        "suite": expected["suite"],
        "task_index": expected["task_index"],
        "state_id": expected["state_id"],
        "parent_key": expected["parent_key"],
        "cohort": expected["cohort"],
        "split": expected["split"],
        "official_horizon": horizon,
        "source_horizon": SOURCE_HORIZON,
        "derived_step_count": len(steps),
        "label_row_count": len(labels),
        "receipt_sha256": sha256_file(episode_dir / "episode_receipt.json"),
        "source_binding_sha256": sha256_file(episode_dir / "source_binding.json"),
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "source_mutation": False,
        "future_leakage": False,
        "structurally_complete": True,
    }
    if expose_train_metrics:
        result["summary"] = summary
    return result


def _train_calibration(validated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    enriched = [row for row in validated if row["cohort"] == TRAIN_COHORT]
    suites: dict[str, Any] = {}
    per_task: dict[str, Any] = {}
    support_failures: list[str] = []
    for suite in TARGET_SUITES:
        rows = [row for row in enriched if row["suite"] == suite]
        health = _train_health(rows)
        suites[suite] = health
        for field in (
            "start_positive_episode_count",
            "fully_known_hard_negative_episode_count",
            "known_step_count",
            "release_safe_step_count",
        ):
            if int(health[field]) <= 0:
                support_failures.append(f"{suite}:{field}=0")
        for task, count in health["per_task_episode_count"].items():
            per_task[f"{suite}:task_{task}"] = {"episode_count": count}
            if count != 30:
                support_failures.append(f"{suite}:task_{task}:episode_count={count}")
    return {
        "schema": R8Z_SCHEMA,
        "scope": "DETECTOR_TRAIN_ONLY",
        "episode_count": len(enriched),
        "suite_metrics": suites,
        "per_task_support": per_task,
        "support_failures": support_failures,
        "status": "PASS_TRAIN_ONLY_LABEL_SUPPORT" if not support_failures else HOLD_LABEL_SUPPORT,
        "nontrain_metrics_read_for_parameter_selection": False,
    }


def run_audit(
    context: SourceContext,
    *,
    suite_roots: Mapping[str, Path],
    composite_root: Path,
    r8z_head: str,
) -> dict[str, Any]:
    composite_root = require_new_output_root(composite_root)
    for suite in TARGET_SUITES:
        if suite not in suite_roots:
            raise ValueError(f"missing suite root: {suite}")
        root = suite_roots[suite].resolve()
        report = read_json(root / "suite_report.json")
        if report.get("status") != SUITE_BUILD_PASS or report.get("r8z_code_head") != r8z_head:
            raise ValueError(f"{suite} suite build report did not PASS at current head")
        assert_public_report_sealed(report)
        checksums_ok, reason = verify_checksums(root)
        if not checksums_ok:
            raise ValueError(f"{suite} checksum closure failed: {reason}")
    composite_root.mkdir(parents=True)
    validated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for expected in context.ogs_rows:
        suite = str(expected["suite"])
        try:
            validated.append(
                validate_derived_episode(
                    context,
                    expected,
                    suite_root=suite_roots[suite].resolve(),
                    r8z_head=r8z_head,
                    expose_train_metrics=expected["cohort"] == TRAIN_COHORT,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "suite": expected["suite"],
                    "task_index": expected["task_index"],
                    "state_id": expected["state_id"],
                    "parent_key": expected["parent_key"],
                    "cohort": expected["cohort"],
                    "split": expected["split"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    public_ledger = []
    for row in validated:
        public = {key: value for key, value in row.items() if key != "summary"}
        if row["cohort"] != TRAIN_COHORT:
            public.pop("derived_step_count", None)
            public.pop("label_row_count", None)
        public_ledger.append(public)
    ledger_path = composite_root / "composite_episode_ledger.jsonl"
    error_path = composite_root / "error_ledger.jsonl"
    source_hash_path = composite_root / "source_hash_ledger.jsonl"
    write_jsonl(ledger_path, public_ledger)
    write_jsonl(error_path, errors)
    write_jsonl(
        source_hash_path,
        (
            {
                "suite": row["suite"],
                "task_index": row["task_index"],
                "state_id": row["state_id"],
                "parent_key": row["parent_key"],
                "cohort": row["cohort"],
                "split": row["split"],
                "source_receipt_sha256": row["source_receipt_sha256"],
                "source_binding_sha256": row["source_binding_sha256"],
            }
            for row in validated
        ),
    )
    calibration = _train_calibration(validated)
    calibration_path = composite_root / "train_only_label_calibration.json"
    write_json(calibration_path, calibration)
    cohort_counts = Counter(row["cohort"] for row in validated)
    sealed = {
        "schema": R8Z_SCHEMA,
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "nontrain_episode_count": sum(
            count for cohort, count in cohort_counts.items() if cohort != TRAIN_COHORT
        ),
        "nontrain_metrics_exposed": False,
        "sealed_cohorts": [cohort for cohort in COHORTS if cohort != TRAIN_COHORT],
    }
    sealed_path = composite_root / "sealed_cohort_closure.json"
    write_json(sealed_path, sealed)
    expected_episode_dirs = {
        derived_episode_dir(suite_roots[str(row["suite"])].resolve(), row)
        for row in context.ogs_rows
    }
    actual_episode_dirs = {
        path.parent
        for root in suite_roots.values()
        for path in root.resolve().rglob("episode_receipt.json")
    }
    outside_episode_dirs = sorted(str(path) for path in actual_episode_dirs - expected_episode_dirs)
    suite_counts = Counter(row["suite"] for row in validated)
    unique_count = len({identity(row) for row in validated})
    structural_pass = (
        len(validated) == 1500
        and unique_count == 1500
        and not errors
        and suite_counts == Counter({suite: 500 for suite in TARGET_SUITES})
        and cohort_counts == Counter(EXPECTED_COHORT_COUNTS)
        and not outside_episode_dirs
        and all(row["source_mutation"] is False for row in validated)
        and all(row["future_leakage"] is False for row in validated)
    )
    if not structural_pass:
        status = HOLD_STATUS
    elif calibration["status"] != "PASS_TRAIN_ONLY_LABEL_SUPPORT":
        status = HOLD_LABEL_SUPPORT
    else:
        status = PASS_STATUS
    report = {
        "schema": R8Z_SCHEMA,
        "status": status,
        "final_decision": status,
        "next_stage": NEXT_STAGE,
        "r8z_code_head": r8z_head,
        "suite_roots": {suite: str(suite_roots[suite].resolve()) for suite in TARGET_SUITES},
        "episode_count": len(validated),
        "expected_episode_count": 1500,
        "unique_identity_count": unique_count,
        "suite_counts": dict(sorted(suite_counts.items())),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "missing_count": 1500 - len(validated),
        "duplicate_count": len(validated) - unique_count,
        "outside_count": len(outside_episode_dirs),
        "outside_episode_dirs": outside_episode_dirs,
        "source_mutation_count": sum(bool(row["source_mutation"]) for row in validated)
        + sum("source mutation" in row["error_message"].lower() for row in errors),
        "future_leakage_count": sum(bool(row["future_leakage"]) for row in validated)
        + sum("future" in row["error_message"].lower() for row in errors),
        "label_rebuild_failure_count": len(errors),
        "receipt_failure_count": len(errors),
        "nontrain_metrics_exposed": False,
        "composite_episode_ledger": str(ledger_path),
        "composite_episode_ledger_sha256": sha256_file(ledger_path),
        "train_only_label_calibration": str(calibration_path),
        "train_only_label_calibration_sha256": sha256_file(calibration_path),
        "sealed_cohort_closure": str(sealed_path),
        "sealed_cohort_closure_sha256": sha256_file(sealed_path),
        "source_hash_ledger": str(source_hash_path),
        "source_hash_ledger_sha256": sha256_file(source_hash_path),
        "error_ledger": str(error_path),
        "error_ledger_sha256": sha256_file(error_path),
        "boundaries": {
            "openvla_loads": 0,
            "libero_steps": 0,
            "gpu_jobs": 0,
            "training_epochs": 0,
            "materialization_runs": 0,
            "attacks": 0,
            "storage_deletions": 0,
            "attack_outcomes_read": False,
        },
        **source_provenance(context),
    }
    report_path = composite_root / "c2g_r8z_ogs_composite_report.json"
    write_json(report_path, report)
    write_report_sidecar(report_path)
    write_checksums(composite_root)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--spatial220-root", type=Path, required=True)
    parser.add_argument("--object280-root", type=Path, required=True)
    parser.add_argument("--goal300-root", type=Path, required=True)
    parser.add_argument("--composite-root", type=Path, required=True)
    parser.add_argument("--r8z-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = source_context_from_args(args)
    report = run_audit(
        context,
        suite_roots={
            "libero_spatial": args.spatial220_root,
            "libero_object": args.object280_root,
            "libero_goal": args.goal300_root,
        },
        composite_root=args.composite_root,
        r8z_head=args.r8z_head,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
