#!/usr/bin/env python3
"""Build hash-bound R8Z official-horizon views from the frozen R8W clean corpus."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    CANARY_SALT,
    COHORTS,
    R8Z_SCHEMA,
    TARGET_SUITES,
    TRAIN_COHORT,
    SourceContext,
    derive_episode,
    derived_episode_dir,
    load_source_context,
    require_new_output_root,
    select_canary_rows,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
    write_report_sidecar,
)


CANARY_BUILD_PASS = "PASS_C2G_R8Z_OGS_CANARY_BUILD"
CANARY_BUILD_HOLD = "HOLD_C2G_R8Z_OGS_CANARY_BUILD"
SUITE_BUILD_PASS = "PASS_C2G_R8Z_OGS_SUITE_BUILD"
SUITE_BUILD_HOLD = "HOLD_C2G_R8Z_OGS_SUITE_BUILD"


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-r8w-plan-report", type=Path, required=True)
    parser.add_argument("--expected-source-r8w-plan-report-sha256", required=True)
    parser.add_argument("--source-r8w-master-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-r8w-master-manifest-sha256", required=True)
    parser.add_argument("--source-r8w-run-root", type=Path, required=True)
    parser.add_argument("--source-r8w-scheduler-report", type=Path, required=True)
    parser.add_argument("--expected-source-r8w-scheduler-report-sha256", required=True)
    parser.add_argument("--expected-source-git-head", required=True)


def source_context_from_args(args: argparse.Namespace) -> SourceContext:
    return load_source_context(
        plan_report_path=args.source_r8w_plan_report,
        expected_plan_report_sha256=args.expected_source_r8w_plan_report_sha256,
        master_manifest_path=args.source_r8w_master_manifest,
        expected_master_manifest_sha256=args.expected_source_r8w_master_manifest_sha256,
        run_root=args.source_r8w_run_root,
        scheduler_report_path=args.source_r8w_scheduler_report,
        expected_scheduler_report_sha256=args.expected_source_r8w_scheduler_report_sha256,
        expected_source_git_head=args.expected_source_git_head,
    )


def source_provenance(context: SourceContext) -> dict[str, Any]:
    return {
        "source_r8w_plan_report": str(context.plan_report_path),
        "source_r8w_plan_report_sha256": context.plan_report_sha256,
        "source_r8w_master_manifest": str(context.master_manifest_path),
        "source_r8w_master_manifest_sha256": context.master_manifest_sha256,
        "source_r8w_run_root": str(context.run_root),
        "source_r8w_scheduler_report": str(context.scheduler_report_path),
        "source_r8w_scheduler_report_sha256": context.scheduler_report_sha256,
        "source_r8w_git_head": context.source_git_head,
        "source_scheduler_caveat": context.scheduler_caveat,
    }


def _train_health(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train = [row for row in results if row["cohort"] == TRAIN_COHORT]
    by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train:
        by_task[int(row["task_index"])].append(row)
    total_steps = sum(int(row["summary"]["row_count"]) for row in train)
    known_steps = sum(int(row["summary"]["known_step_count"]) for row in train)
    return {
        "episode_count": len(train),
        "step_count": total_steps,
        "known_step_count": known_steps,
        "unknown_step_count": total_steps - known_steps,
        "known_step_fraction": known_steps / total_steps if total_steps else 0.0,
        "unknown_step_fraction": (total_steps - known_steps) / total_steps if total_steps else 0.0,
        "critical_active_step_count": sum(
            int(row["summary"]["critical_active_step_count"]) for row in train
        ),
        "start_positive_episode_count": sum(
            bool(row["summary"]["start_positive_episode"]) for row in train
        ),
        "burst_feasible_episode_count": sum(
            bool(row["summary"]["burst_feasible_episode"]) for row in train
        ),
        "fully_known_hard_negative_episode_count": sum(
            bool(row["summary"]["fully_known_hard_negative_episode"]) for row in train
        ),
        "release_safe_step_count": sum(
            int(row["summary"]["release_safe_step_count"]) for row in train
        ),
        "target_grounding_known_step_count": sum(
            int(row["summary"]["target_grounding_known_step_count"]) for row in train
        ),
        "per_task_episode_count": {
            str(task): len(rows) for task, rows in sorted(by_task.items())
        },
        "teacher_reason_code_counts": dict(
            sorted(
                sum(
                    (
                        Counter(row["summary"]["reason_code_counts"])
                        for row in train
                    ),
                    Counter(),
                ).items()
            )
        ),
    }


def public_episode_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: result[key]
        for key in (
            "suite",
            "task_index",
            "state_id",
            "parent_key",
            "cohort",
            "split",
            "official_horizon",
            "source_horizon",
            "receipt",
            "receipt_sha256",
        )
    }
    if result["cohort"] == TRAIN_COHORT:
        row.update(
            derived_step_count=result["derived_step_count"],
            label_row_count=result["label_row_count"],
        )
    return row


def _label_ledger_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "suite": result["suite"],
        "task_index": result["task_index"],
        "state_id": result["state_id"],
        "parent_key": result["parent_key"],
        "cohort": result["cohort"],
        "split": result["split"],
        "label_file": result["label_file"],
        "label_file_sha256": result["label_file_sha256"],
    }
    if result["cohort"] == TRAIN_COHORT:
        row["label_row_count"] = result["label_row_count"]
    return row


def _binding_ledger_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "suite": result["suite"],
        "task_index": result["task_index"],
        "state_id": result["state_id"],
        "parent_key": result["parent_key"],
        "cohort": result["cohort"],
        "split": result["split"],
        "source_binding": result["source_binding"],
        "source_binding_sha256": result["source_binding_sha256"],
    }


def build_canary(
    context: SourceContext,
    *,
    output_root: Path,
    r8z_head: str,
    salt: str = CANARY_SALT,
) -> dict[str, Any]:
    output_root = require_new_output_root(output_root)
    output_root.mkdir(parents=True)
    selected = select_canary_rows(context.rows, salt=salt)
    selection_path = output_root / "canary_selection_manifest.jsonl"
    write_jsonl(
        selection_path,
        (
            {
                "suite": row["suite"],
                "task_index": row["task_index"],
                "state_id": row["state_id"],
                "parent_key": row["parent_key"],
                "cohort": row["cohort"],
                "split": row["split"],
                "selection_method": "sha256(parent_key + fixed_canary_salt)",
                "selection_salt": salt,
                "selection_uses_outcome_or_label": False,
            }
            for row in selected
        ),
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in selected:
        try:
            results.append(derive_episode(context, row, output_root=output_root, r8z_head=r8z_head))
        except Exception as exc:
            errors.append(
                {
                    "suite": row["suite"],
                    "task_index": row["task_index"],
                    "state_id": row["state_id"],
                    "parent_key": row["parent_key"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    ledger_path = output_root / "canary_build_ledger.jsonl"
    error_path = output_root / "error_ledger.jsonl"
    write_jsonl(ledger_path, (public_episode_row(row) for row in results))
    write_jsonl(error_path, errors)
    status = CANARY_BUILD_PASS if len(results) == 12 and not errors else CANARY_BUILD_HOLD
    report = {
        "schema": R8Z_SCHEMA,
        "status": status,
        "r8z_code_head": r8z_head,
        "selection_method": "sha256(parent_key + fixed_canary_salt)",
        "selection_salt": salt,
        "selection_uses_outcome_or_label": False,
        "episode_count": len(results),
        "expected_episode_count": 12,
        "nontrain_count": sum(row["cohort"] != TRAIN_COHORT for row in results),
        "failure_count": len(errors),
        "suite_counts": dict(sorted(Counter(row["suite"] for row in results).items())),
        "task_counts_by_suite": {
            suite: len({row["task_index"] for row in results if row["suite"] == suite})
            for suite in TARGET_SUITES
        },
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "ledger": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
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
    report_path = output_root / "canary_build_report.json"
    write_json(report_path, report)
    write_report_sidecar(report_path)
    write_checksums(output_root)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def build_suite(
    context: SourceContext,
    *,
    suite: str,
    output_root: Path,
    r8z_head: str,
) -> dict[str, Any]:
    if suite not in TARGET_SUITES:
        raise ValueError(f"unsupported R8Z OGS suite: {suite}")
    output_root = require_new_output_root(output_root)
    output_root.mkdir(parents=True)
    expected = [row for row in context.rows if row["suite"] == suite]
    if len(expected) != 500:
        raise ValueError(f"{suite} source count is not 500")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in expected:
        try:
            results.append(derive_episode(context, row, output_root=output_root, r8z_head=r8z_head))
        except Exception as exc:
            errors.append(
                {
                    "suite": row["suite"],
                    "task_index": row["task_index"],
                    "state_id": row["state_id"],
                    "parent_key": row["parent_key"],
                    "cohort": row["cohort"],
                    "split": row["split"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    episode_ledger = output_root / "episode_ledger.jsonl"
    label_ledger = output_root / "label_file_ledger.jsonl"
    binding_ledger = output_root / "source_binding_ledger.jsonl"
    error_ledger = output_root / "error_ledger.jsonl"
    sealed_manifest = output_root / "sealed_nontrain_manifest.jsonl"
    write_jsonl(episode_ledger, (public_episode_row(row) for row in results))
    write_jsonl(label_ledger, (_label_ledger_row(row) for row in results))
    write_jsonl(binding_ledger, (_binding_ledger_row(row) for row in results))
    write_jsonl(error_ledger, errors)
    write_jsonl(
        sealed_manifest,
        (
            {
                "suite": row["suite"],
                "task_index": row["task_index"],
                "state_id": row["state_id"],
                "parent_key": row["parent_key"],
                "cohort": row["cohort"],
                "split": row["split"],
                "episode_receipt": row["receipt"],
                "episode_receipt_sha256": row["receipt_sha256"],
                "sealed": True,
            }
            for row in results
            if row["cohort"] != TRAIN_COHORT
        ),
    )
    status = SUITE_BUILD_PASS if len(results) == 500 and not errors else SUITE_BUILD_HOLD
    report = {
        "schema": R8Z_SCHEMA,
        "status": status,
        "suite": suite,
        "r8z_code_head": r8z_head,
        "official_horizon": results[0]["official_horizon"] if results else None,
        "source_horizon": 300,
        "expected_episode_count": 500,
        "episode_count": len(results),
        "unique_identity_count": len(
            {(row["suite"], row["task_index"], row["state_id"]) for row in results}
        ),
        "failure_count": len(errors),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in results).items())),
        "train_only_label_health": _train_health(results),
        "nontrain_metrics_exposed": False,
        "episode_ledger": str(episode_ledger),
        "episode_ledger_sha256": sha256_file(episode_ledger),
        "label_file_ledger": str(label_ledger),
        "label_file_ledger_sha256": sha256_file(label_ledger),
        "source_binding_ledger": str(binding_ledger),
        "source_binding_ledger_sha256": sha256_file(binding_ledger),
        "error_ledger": str(error_ledger),
        "error_ledger_sha256": sha256_file(error_ledger),
        "sealed_nontrain_manifest": str(sealed_manifest),
        "sealed_nontrain_manifest_sha256": sha256_file(sealed_manifest),
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
    report_path = output_root / "suite_report.json"
    write_json(report_path, report)
    write_report_sidecar(report_path)
    write_checksums(output_root)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("canary", "suite"))
    add_source_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r8z-head", required=True)
    parser.add_argument("--suite", choices=TARGET_SUITES)
    parser.add_argument("--canary-salt", default=CANARY_SALT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = source_context_from_args(args)
    if args.mode == "canary":
        report = build_canary(
            context,
            output_root=args.output_root,
            r8z_head=args.r8z_head,
            salt=args.canary_salt,
        )
        passed = report["status"] == CANARY_BUILD_PASS
    else:
        if not args.suite:
            raise ValueError("--suite is required in suite mode")
        report = build_suite(
            context,
            suite=args.suite,
            output_root=args.output_root,
            r8z_head=args.r8z_head,
        )
        passed = report["status"] == SUITE_BUILD_PASS
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
