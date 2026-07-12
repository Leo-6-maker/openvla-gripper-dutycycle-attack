#!/usr/bin/env python3
"""Audit the deterministic 12-episode R8Z train-only official-horizon canary."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from tools.multisuite_detector.audit_c2g_r8z_ogs_full1500 import (
    validate_derived_episode,
)
from tools.multisuite_detector.build_c2g_r8z_ogs_official_views import (
    CANARY_BUILD_PASS,
    add_source_arguments,
    source_context_from_args,
    source_provenance,
)
from tools.multisuite_detector.rebuild_c2g_r8z_teacher_v2_labels import (
    CANARY_SALT,
    R8Z_SCHEMA,
    TARGET_SUITES,
    TRAIN_COHORT,
    identity,
    derived_episode_dir,
    read_json,
    read_jsonl,
    select_canary_rows,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
    write_report_sidecar,
)


PASS_STATUS = "PASS_C2G_R8Z_OGS_LABEL_CANARY"
HOLD_STATUS = "HOLD_C2G_R8Z_OGS_LABEL_CANARY"


def run_audit(
    context,
    *,
    canary_root: Path,
    r8z_head: str,
    salt: str = CANARY_SALT,
) -> dict[str, Any]:
    canary_root = canary_root.resolve()
    report_path = canary_root / "c2g_r8z_ogs_canary_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    checksums_ok, reason = verify_checksums(canary_root)
    if not checksums_ok:
        raise ValueError(f"canary build checksum closure failed: {reason}")
    build_report = read_json(canary_root / "canary_build_report.json")
    if build_report.get("status") != CANARY_BUILD_PASS:
        raise ValueError("canary build report did not PASS")
    actual_selection = read_jsonl(canary_root / "canary_selection_manifest.jsonl")
    expected_selection = select_canary_rows(context.rows, salt=salt)
    if [identity(row) for row in actual_selection] != [identity(row) for row in expected_selection]:
        raise ValueError("canary selection differs from deterministic source-only selection")
    if any(row.get("cohort") != TRAIN_COHORT for row in actual_selection):
        raise ValueError("canary contains a nontrain parent")
    validated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for expected in expected_selection:
        try:
            validated.append(
                validate_derived_episode(
                    context,
                    expected,
                    suite_root=canary_root,
                    r8z_head=r8z_head,
                    expose_train_metrics=False,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "suite": expected["suite"],
                    "task_index": expected["task_index"],
                    "state_id": expected["state_id"],
                    "parent_key": expected["parent_key"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    ledger_path = canary_root / "canary_audit_ledger.jsonl"
    error_path = canary_root / "canary_audit_errors.jsonl"
    write_jsonl(ledger_path, validated)
    write_jsonl(error_path, errors)
    suite_counts = Counter(row["suite"] for row in validated)
    expected_episode_dirs = {
        derived_episode_dir(canary_root, row) for row in expected_selection
    }
    actual_episode_dirs = {
        path.parent for path in canary_root.rglob("episode_receipt.json")
    }
    outside_episode_dirs = sorted(str(path) for path in actual_episode_dirs - expected_episode_dirs)
    passed = (
        len(validated) == 12
        and not errors
        and len({identity(row) for row in validated}) == 12
        and suite_counts == Counter({suite: 4 for suite in TARGET_SUITES})
        and all(row["cohort"] == TRAIN_COHORT for row in validated)
        and all(row["source_mutation"] is False for row in validated)
        and all(row["future_leakage"] is False for row in validated)
        and not outside_episode_dirs
    )
    report = {
        "schema": R8Z_SCHEMA,
        "status": PASS_STATUS if passed else HOLD_STATUS,
        "r8z_code_head": r8z_head,
        "episode_count": len(validated),
        "expected_episode_count": 12,
        "nontrain_count": sum(row["cohort"] != TRAIN_COHORT for row in validated),
        "failure_count": len(errors),
        "outside_count": len(outside_episode_dirs),
        "suite_counts": dict(sorted(suite_counts.items())),
        "source_hash_verified_count": len(validated),
        "prefix_closure_count": len(validated),
        "label_rebuild_complete_count": len(validated),
        "receipt_pass_count": len(validated),
        "source_mutation_count": sum(bool(row["source_mutation"]) for row in validated),
        "future_leakage_count": sum(bool(row["future_leakage"]) for row in validated),
        "outcome_based_selection_count": 0,
        "selection_method": "sha256(parent_key + fixed_canary_salt)",
        "selection_salt": salt,
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
    write_json(report_path, report)
    write_report_sidecar(report_path)
    write_checksums(canary_root)
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--r8z-head", required=True)
    parser.add_argument("--canary-salt", default=CANARY_SALT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    context = source_context_from_args(args)
    report = run_audit(
        context,
        canary_root=args.canary_root,
        r8z_head=args.r8z_head,
        salt=args.canary_salt,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
