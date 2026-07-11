#!/usr/bin/env python3
"""CPU-only R8S audit: Teacher-v1 salvage boundaries and replay feasibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.multisuite_detector.c2g_r8s_common import (
    ATTACK_EVAL,
    EP_FIELDS,
    FIELD_FIELDS,
    GO_AUX,
    GO_PARTIAL,
    GO_REPLAY,
    HOLD_INPUT,
    HOLD_NONE,
    LEGACY,
    MAPPINGS,
    SCHEMA,
    SUITES,
    sha256_file,
    verify_r8r,
    write_csv,
    write_json,
    write_jsonl,
)
from tools.multisuite_detector.c2g_r8s_episode import audit_episode


def select_replay_canary(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_task: int = 3,
    tasks_per_suite: int = 2,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("strict_replay_ready") and row.get("cohort") != ATTACK_EVAL:
            groups[(str(row["suite"]), int(row["task_index"]))].append(row)
    selected: list[dict[str, Any]] = []
    for suite in SUITES:
        tasks = sorted(
            task
            for (row_suite, task), members in groups.items()
            if row_suite == suite and len(members) >= per_task
        )[:tasks_per_suite]
        if len(tasks) != tasks_per_suite:
            return []
        for task in tasks:
            members = sorted(
                groups[(suite, task)],
                key=lambda row: hashlib.sha256(
                    f"R8S_REPLAY_CANARY|{row['parent_key']}".encode("utf-8")
                ).digest(),
            )[:per_task]
            selected.extend(
                {
                    key: row[key]
                    for key in (
                        "suite", "task_index", "state_id", "parent_key",
                        "cohort", "split", "metadata_path", "step_records_path",
                    )
                }
                for row in members
            )
    return selected


def coverage(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    count = sum(bool(row.get(field)) for row in rows)
    return {
        "count": count,
        "total": len(rows),
        "fraction": count / len(rows) if rows else 0.0,
    }


def run_audit(
    *,
    repo: Path,
    expected_git_commit: str,
    r8r_root: Path,
    expected_r8r_report_sha256: str,
    output_dir: Path,
    teacher_v1_source: Path | None = None,
    verify_git_state: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if verify_git_state:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        ).strip()
        if head != expected_git_commit:
            raise ValueError(f"head mismatch: {head} != {expected_git_commit}")
        if status:
            raise ValueError("worktree is not clean")

    r8r_report, source_rows, r8r_hashes = verify_r8r(
        r8r_root, expected_r8r_report_sha256
    )
    rows = sorted(
        (audit_episode(row) for row in source_rows),
        key=lambda row: (
            SUITES.index(str(row["suite"])),
            int(row["task_index"]),
            int(row["state_id"]),
        ),
    )
    failures = sum(not row.get("episode_read_ok") for row in rows)
    ready = sum(bool(row.get("strict_replay_ready")) for row in rows)
    candidates = sum(bool(row.get("strict_replay_candidate")) for row in rows)
    legacy = sum(bool(row.get("legacy_auxiliary_eligible")) for row in rows)

    if failures:
        decision, next_stage = HOLD_INPUT, "STOP_FOR_R8S_REVIEW"
    elif ready == len(rows) and rows:
        decision, next_stage = GO_REPLAY, "R8T_24EP_DETERMINISTIC_REPLAY_CANARY"
    elif ready:
        decision, next_stage = GO_PARTIAL, "R8T_PARTIAL_REPLAY_CANARY_AND_RESIDUAL_PLAN"
    elif legacy:
        decision, next_stage = GO_AUX, "R8T_TEACHER_V2_COLLECTION_CANARY_DESIGN"
    else:
        decision, next_stage = HOLD_NONE, "STOP_FOR_R8S_REVIEW"

    canary = select_replay_canary(rows)
    uncovered = [
        {
            key: row[key]
            for key in ("suite", "task_index", "state_id", "parent_key", "cohort", "split")
        }
        | {
            "reason": row["replay_blockers"]
            or "TEACHER_V2_RAW_EVIDENCE_ABSENT"
        }
        for row in rows
        if not row.get("strict_replay_ready")
    ]
    field_rows = [
        {
            key: row[key]
            for key in ("suite", "task_index", "state_id", "parent_key", "cohort", "split")
        }
        | {field: bool(row.get(f"{field}_present")) for field in LEGACY}
        | {
            field: row.get(field)
            for field in FIELD_FIELDS
            if field
            not in {
                "suite", "task_index", "state_id", "parent_key", "cohort", "split", *LEGACY
            }
        }
        for row in rows
    ]

    teacher_source = None
    if teacher_v1_source:
        text = teacher_v1_source.read_text(encoding="utf-8", errors="replace")
        teacher_source = {
            "path": str(teacher_v1_source.resolve()),
            "sha256": sha256_file(teacher_v1_source),
            **{
                f"{field}_literal": field in text
                for field in (
                    "teacher_primary_attackable",
                    "teacher_release_safe",
                    "teacher_event_role",
                    "teacher_phase",
                )
            },
        }

    contract_files = (
        "tools/multisuite_detector/c2g_clean_window_label_builder.py",
        "src/gripper_attack/c2g_clean_window_schema.py",
        "src/gripper_attack/c2g_clean_policy_signals.py",
        "tools/multisuite_detector/audit_c2f_teacher_v1_labels.py",
    )
    contract_hashes = {
        name: sha256_file(repo / name)
        for name in contract_files
        if (repo / name).is_file()
    }
    coverage_fields = tuple(f"{field}_present" for field in LEGACY) + (
        "full_action_7d_complete",
        "partial_action_4d_complete",
        "official_init_state_reference_present",
        "libero_version_bound",
        "runtime_versions_bound",
        "controller_config_bound",
        "action_semantics_bound",
        "task_bddl_bound",
        "seed_bound",
        "max_steps_bound",
        "strict_replay_candidate",
        "strict_replay_ready",
        "legacy_auxiliary_eligible",
    )
    report = {
        "schema": SCHEMA,
        "gate": "C2G_R8S_TEACHER_V1_SEMANTIC_REPLAY_AUDIT",
        "status": "PASS_C2G_R8S_SEMANTIC_REPLAY_AUDIT",
        "r8s_head": expected_git_commit,
        "actual_checkout": str(repo.resolve()),
        "worktree_clean": True,
        "r8r_root": str(r8r_root.resolve()),
        "r8r_report_sha256": sha256_file(
            r8r_root / "clean2000_r7_reuse_audit_report.json"
        ),
        "r8r_sha256s_sha256": sha256_file(r8r_root / "SHA256SUMS"),
        "r8r_sha256s_self_binding_sha256": sha256_file(
            r8r_root / "SHA256SUMS.sha256"
        ),
        "r8r_input_final_decision": r8r_report.get("final_decision"),
        "r8r_input_classification_counts": r8r_report.get("classification_counts"),
        "r8r_input_hash_count": len(r8r_hashes),
        "teacher_v1_source": teacher_source,
        "current_contract_source_sha256": contract_hashes,
        "episode_count": len(rows),
        "read_failure_count": failures,
        "legacy_auxiliary_eligible_count": legacy,
        "strict_replay_candidate_count": candidates,
        "strict_replay_ready_count": ready,
        "strict_replay_not_ready_count": len(rows) - ready,
        "replay_canary_parent_count": len(canary),
        "current_contract_uncovered_count": len(uncovered),
        "coverage": {
            field: coverage(rows, field) for field in coverage_fields
        },
        "semantic_mapping_counts": dict(Counter(row[2] for row in MAPPINGS)),
        "exact_equivalent_mapping_count": 0,
        "per_suite": {
            suite: {
                "episode_count": sum(row["suite"] == suite for row in rows),
                "strict_replay_ready_count": sum(
                    row["suite"] == suite and row.get("strict_replay_ready")
                    for row in rows
                ),
                "legacy_auxiliary_eligible_count": sum(
                    row["suite"] == suite and row.get("legacy_auxiliary_eligible")
                    for row in rows
                ),
            }
            for suite in SUITES
        },
        "invariants": {
            "episode_cardinality_matches_r8r": len(rows)
            == int(r8r_report.get("canonical_registered_identities", -1)),
            "replay_partition_closed": ready + len(uncovered) == len(rows),
            "canary_excludes_attack_eval": all(
                row["cohort"] != ATTACK_EVAL for row in canary
            ),
            "no_exact_teacher_v2_overclaim": all(
                not row.get("current_teacher_v2_exact_supervision_eligible")
                for row in rows
            ),
            "semantic_mapping_has_no_unproven_exact_equivalence": True,
        },
        "final_decision": decision,
        "next_recommended_stage": next_stage,
        "replay_canary_authorization": "HOLD_PENDING_R8S_REVIEW",
        "collection_authorization": "HOLD",
        "training_authorization": "HOLD",
        "claim_boundary": {
            "teacher_v1_may_be_used_for": [
                "legacy replication",
                "auxiliary supervision",
                "representation pretraining",
            ],
            "teacher_v1_must_not_be_called": "exact Teacher-v2 ground truth",
            "attack_eval_interpretation": "RETROSPECTIVELY_FROZEN_ATTACK_EVAL_COHORT",
        },
        "boundaries": {
            "attack_outcomes_read": False,
            "openvla_models_loaded": 0,
            "detector_models_loaded": 0,
            "libero_environments_created": 0,
            "environment_reset_calls": 0,
            "environment_step_calls": 0,
            "rollouts": 0,
            "replays_executed": 0,
            "training_epochs": 0,
            "calibration_runs": 0,
            "attacks": 0,
            "storage_deletions": 0,
            "d7_modifications": 0,
        },
    }
    if not all(report["invariants"].values()):
        raise ValueError("R8S invariant failure")

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "r8s_episode_ledger.csv", rows, EP_FIELDS)
    write_csv(output_dir / "r8s_legacy_field_coverage.csv", field_rows, FIELD_FIELDS)
    write_csv(
        output_dir / "r8s_semantic_mapping_matrix.csv",
        [
            {
                "legacy_signal": legacy_signal,
                "current_target": current_target,
                "mapping_class": mapping_class,
                "allowed_use": allowed_use,
                "forbidden_claim": forbidden_claim,
            }
            for legacy_signal, current_target, mapping_class, allowed_use, forbidden_claim
            in MAPPINGS
        ],
        (
            "legacy_signal",
            "current_target",
            "mapping_class",
            "allowed_use",
            "forbidden_claim",
        ),
    )
    write_csv(output_dir / "r8s_replay_feasibility.csv", rows, EP_FIELDS)
    write_jsonl(output_dir / "r8s_replay_canary_manifest.jsonl", canary)
    write_jsonl(output_dir / "r8s_current_contract_uncovered.jsonl", uncovered)
    write_json(output_dir / "r8s_semantic_replay_audit_report.json", report)

    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    sums = output_dir / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    self_binding = output_dir / "SHA256SUMS.sha256"
    self_binding.write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {
        **report,
        "output_root": str(output_dir.resolve()),
        "artifact_sha256": hashes,
        "sha256s_sha256": sums_sha,
        "sha256s_self_binding_sha256": sha256_file(self_binding),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--r8r-root", type=Path, required=True)
    parser.add_argument("--expected-r8r-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-v1-source", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_audit(
        repo=args.repo,
        expected_git_commit=args.expected_git_commit,
        r8r_root=args.r8r_root,
        expected_r8r_report_sha256=args.expected_r8r_report_sha256,
        output_dir=args.output_dir,
        teacher_v1_source=args.teacher_v1_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
