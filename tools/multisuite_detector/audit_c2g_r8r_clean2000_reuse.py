#!/usr/bin/env python3
"""Hash-bound, read-only R8R Clean2000 reuse audit.

This audit reconciles physical source views with the frozen R7 identity registry,
selects exactly one canonical source per parent, applies the current 25D/9D and
Teacher-v2 contracts, writes all required ledgers, and fails closed on ambiguity.
It never loads OpenVLA or detector checkpoints, creates LIBERO environments,
launches rollouts, trains, calibrates, or reads attack outcomes.
"""
from __future__ import annotations
import argparse, json, subprocess
from collections import Counter
from pathlib import Path
from typing import Sequence

from tools.multisuite_detector.plan_c2g_scientific_corpus import ATTACK_EVAL, SUITES
from tools.multisuite_detector.c2g_r8r_common import (
    IDENTITY_KEY, assert_sha, build_source_view_ledger, load_registry,
    load_source_spec, new_output, select_canonical, sha256_file, write_json,
)
from tools.multisuite_detector.c2g_r8r_episode import (
    A_DIRECT, B_AUGMENT, C_LEGACY, D_RECOLLECT, CLASSIFICATIONS, audit_episode,
)
from tools.multisuite_detector.c2g_r8r_outputs import (
    GO_AUGMENT, GO_DIRECT, GO_PARTIAL, HOLD_IDENTITY, HOLD_PROVENANCE, HOLD_TEACHER,
    candidate_root_rows, classification_counts, coverage, final_decision,
    task_classification_counts, write_hash_ledgers, write_primary_artifacts,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "c2g.r8r.clean2000_reuse_audit.2026-07-11.v2"
PASS_STATUS = "PASS_C2G_R8R_CLEAN2000_REUSE_AUDIT"
SOURCE_SPEC_SCHEMA = "c2g.r8r.clean2000_source_spec.2026-07-11.v1"


def _git_state(expected: str, verify: bool):
    expected = str(expected).strip().lower()
    if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected git commit invalid")
    if not verify:
        return expected, ""
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--short"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != expected:
        raise ValueError(f"head mismatch: {head} != {expected}")
    if status:
        raise ValueError("worktree is not clean")
    return head, status


def _missing_episode(registry_row):
    return {
        "suite": registry_row["suite"], "task_index": registry_row["task_index"],
        "state_id": registry_row["state_id"], "parent_key": registry_row["parent_key"],
        "cohort": registry_row["cohort"], "split": registry_row["split"],
        "classification": D_RECOLLECT,
        "classification_reason": "CANONICAL_SOURCE_MISSING_OR_CONFLICT",
        "n_steps": 0, "w16_window_count": 0, "rgb_count": 0,
        "legacy_semantic_salvage_candidate": False,
    }


def run_audit(*, registry_path: Path, plan_report_path: Path,
              expected_plan_report_sha256: str, source_audit_report_path: Path,
              expected_source_audit_report_sha256: str, reusable_manifest_path: Path,
              expected_reusable_manifest_sha256: str, source_spec_path: Path,
              output_dir: Path, expected_git_commit: str,
              verify_git_state: bool = True):
    head, status = _git_state(expected_git_commit, verify_git_state)
    registry, _ = load_registry(
        registry_path, plan_report_path, expected_plan_report_sha256,
    )
    source_audit_report_path = source_audit_report_path.resolve()
    reusable_manifest_path = reusable_manifest_path.resolve()
    assert_sha(source_audit_report_path, expected_source_audit_report_sha256, "R7 source audit")
    assert_sha(reusable_manifest_path, expected_reusable_manifest_sha256, "R7 reusable manifest")
    views, predecessors, source_spec = load_source_spec(source_spec_path)
    output = new_output(output_dir, REPO)
    lookup = {
        (row["suite"], row["task_index"], row["state_id"]): row for row in registry
    }
    source_rows, views_by_name = build_source_view_ledger(views, lookup)
    selected, reconciliation, conflicts = select_canonical(source_rows, registry)
    physical_counts = Counter(
        (row["suite"], row["task_index"], row["state_id"])
        for row in source_rows if row["registered"]
    )
    for row in source_rows:
        row["physical_view_count"] = (
            physical_counts[(row["suite"], row["task_index"], row["state_id"])]
            if row["registered"] else 1
        )
    episodes = []
    for registry_row in registry:
        identity = (
            registry_row["suite"], registry_row["task_index"], registry_row["state_id"]
        )
        source = selected.get(identity)
        episodes.append(
            audit_episode(source, views_by_name[source["source_view_name"]], registry_row)
            if source else _missing_episode(registry_row)
        )
    episodes.sort(key=IDENTITY_KEY)

    physical = len(source_rows)
    unregistered = sum(not row["registered"] for row in source_rows)
    registered = physical - unregistered
    canonical = len(selected)
    duplicate_views = registered - canonical
    noncanonical_views = physical - canonical
    identities_with_multiple_views = sum(count > 1 for count in physical_counts.values())
    missing = sum(row["identity_status"] == "MISSING" for row in reconciliation)
    replacement_views = [view for view in views if view.source_class == "REPLACEMENT_SOURCE"]
    replaced_identities = sum(
        row["selected_canonical"] and row["source_class"] == "REPLACEMENT_SOURCE"
        for row in source_rows
    )
    classes = Counter(row["classification"] for row in episodes)
    detector_required = sum(row["cohort"] != ATTACK_EVAL for row in registry)
    attack_required = sum(row["cohort"] == ATTACK_EVAL for row in registry)
    residual_detector = sum(
        row["cohort"] != ATTACK_EVAL
        and row["classification"] not in {A_DIRECT, B_AUGMENT}
        for row in episodes
    )
    residual_attack = sum(
        row["cohort"] == ATTACK_EVAL
        and row["classification"] not in {A_DIRECT, B_AUGMENT}
        for row in episodes
    )
    if detector_required + attack_required != len(registry):
        raise AssertionError("registry cohort partition failure")
    if residual_detector + residual_attack != classes[C_LEGACY] + classes[D_RECOLLECT]:
        raise AssertionError("current-contract residual partition failure")
    if canonical + duplicate_views + unregistered != physical:
        raise AssertionError("physical view accounting failure")

    frames = sum(int(row.get("rgb_count", 0)) for row in episodes)
    windows = sum(int(row.get("w16_window_count", 0)) for row in episodes)
    frame_expected = frames - 15 * canonical
    frame_closure = windows == frame_expected
    roots = candidate_root_rows(views, source_rows)
    decision = final_decision(len(registry), canonical, conflicts, unregistered, episodes)
    write_primary_artifacts(
        output, source_rows, reconciliation, episodes, roots, source_spec,
    )

    report = {
        "schema": SCHEMA, "status": PASS_STATUS,
        "gate": "R8R_CLEAN2000_REUSE_AUDIT", "r8r_head": expected_git_commit,
        "executed_head": head, "actual_checkout": str(REPO.resolve()),
        "worktree_clean": not bool(status),
        "plan_report": str(plan_report_path.resolve()),
        "plan_report_sha256": sha256_file(plan_report_path.resolve()),
        "registry": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path.resolve()),
        "source_audit_report": str(source_audit_report_path),
        "source_audit_report_sha256": sha256_file(source_audit_report_path),
        "r7_reusable_manifest": str(reusable_manifest_path),
        "r7_reusable_manifest_sha256": sha256_file(reusable_manifest_path),
        "source_spec": str(source_spec_path.resolve()),
        "source_spec_sha256": sha256_file(source_spec_path.resolve()),
        "output_root": str(output),
        "predecessor_r8r_roots": [str(path) for path in predecessors],
        "predecessor_root_status": {
            str(path): {
                "exists": path.exists(),
                "file_count": sum(1 for item in path.rglob("*") if item.is_file())
                if path.is_dir() else 0,
            } for path in predecessors
        },
        "authoritative_source_rule": {
            suite: next((view.name for view in views if suite in view.canonical_suites), None)
            for suite in SUITES
        },
        "candidate_roots": roots,
        "r7_registry_identities": len(registry),
        "physical_episode_views": physical,
        "registered_physical_episode_views": registered,
        "unique_registered_identities_seen": len(physical_counts),
        "canonical_registered_identities": canonical,
        "noncanonical_source_views": noncanonical_views,
        "duplicate_source_views": duplicate_views,
        "identities_with_multiple_views": identities_with_multiple_views,
        "duplicate_conflicts": conflicts,
        "unregistered_episode_views": unregistered,
        "missing_identities": missing,
        "replacement_lineages": len(replacement_views),
        "replaced_identity_count": replaced_identities,
        "classification_counts": {
            classification: int(classes[classification]) for classification in CLASSIFICATIONS
        },
        "per_suite_classification": classification_counts(episodes, "suite"),
        "per_task_classification": task_classification_counts(episodes),
        "per_cohort_classification": classification_counts(episodes, "cohort"),
        "per_split_classification": classification_counts(episodes, "split"),
        "coverage": {
            name: coverage(episodes, field) for name, field in (
                ("rgb", "rgb_complete"), ("task_language", "task_language_present"),
                ("canonical_25d", "canonical_25d_complete"),
                ("policy_intent_9d", "policy_intent_9d_complete"),
                ("raw_policy_logits", "raw_policy_logits_complete"),
                ("teacher_v1_labels", "teacher_v1_label_present"),
                ("teacher_v2_schema_marker", "teacher_v2_schema_marker_present"),
                ("teacher_v2_raw_privileged_evidence", "teacher_v2_raw_evidence_complete"),
                ("teacher_v2_rebuild_success", "teacher_v2_rebuild_success"),
                ("legacy_semantic_salvage_candidates", "legacy_semantic_salvage_candidate"),
                ("clean_success", "clean_success_observed"),
            )
        },
        "teacher_support": {
            "teacher_v2_rebuild_attempted": sum(
                bool(row.get("teacher_v2_rebuild_attempted")) for row in episodes
            ),
            "known_positive_steps": sum(int(row.get("known_positive_steps", 0)) for row in episodes),
            "known_negative_steps": sum(int(row.get("known_negative_steps", 0)) for row in episodes),
            "unknown_steps": sum(int(row.get("unknown_steps", 0)) for row in episodes),
            "positive_episodes": sum(bool(row.get("positive_episode")) for row in episodes),
            "fully_known_negative_episodes": sum(
                bool(row.get("fully_known_negative_episode")) for row in episodes
            ),
            "triggerable_positive_episodes": sum(
                bool(row.get("triggerable_positive_episode")) for row in episodes
            ),
        },
        "frame_count": frames, "w16_window_count": windows,
        "frame_to_window_closure": {
            "formula": "frame_count - canonical_episode_count * 15",
            "expected": frame_expected, "observed": windows, "pass": frame_closure,
        },
        "detector_required_parent_count": detector_required,
        "attack_eval_required_parent_count": attack_required,
        "residual_detector_collection_required": residual_detector,
        "residual_attack_eval_collection_required": residual_attack,
        "total_current_contract_deficit": residual_detector + residual_attack,
        "attack_eval_registry_field_name": ATTACK_EVAL,
        "paper_facing_attack_eval_interpretation": "RETROSPECTIVELY_FROZEN_ATTACK_EVAL_COHORT",
        "final_decision": decision,
        "next_recommended_read_only_stage": (
            "R8S_TEACHER_V1_TO_V2_SEMANTIC_SALVAGE_AUDIT"
            if decision == HOLD_TEACHER else "STOP_FOR_R8R_REVIEW"
        ),
        "r8_collection_authorization": "HOLD", "training_authorization": "HOLD",
        "boundaries": {
            "attack_outcomes_read": False, "openvla_models_loaded": 0,
            "detector_models_loaded": 0, "libero_environments_created": 0,
            "env_reset_calls": 0, "env_step_calls": 0, "rollouts": 0,
            "training_epochs": 0, "calibration_runs": 0, "attacks": 0,
            "storage_deletions": 0, "repository_writes": 0,
            "d7_modifications": 0, "source_assets_modified": False,
        },
        "invariants": {
            "registry_partition_closed": detector_required + attack_required == len(registry),
            "classification_partition_closed": sum(classes.values()) == len(registry),
            "physical_view_accounting_closed": canonical + duplicate_views + unregistered == physical,
            "current_contract_deficit_closed": (
                residual_detector + residual_attack == classes[C_LEGACY] + classes[D_RECOLLECT]
            ),
            "frame_to_window_closed": frame_closure,
        },
    }
    report_path = output / "clean2000_r7_reuse_audit_report.json"
    write_json(report_path, report)
    hashes, sums_sha, self_sha = write_hash_ledgers(output)
    declared = (output / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()[0]
    return {
        **report, "report_sha256": hashes[report_path.name],
        "artifact_sha256": hashes, "sha256sums_sha256": sums_sha,
        "sha256sums_self_binding_sha256": self_sha,
        "sha256sums_self_binding_pass": sha256_file(output / "SHA256SUMS") == declared,
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "registry", "plan-report", "source-audit-report", "reusable-manifest",
        "source-spec", "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-plan-report-sha256", "expected-source-audit-report-sha256",
        "expected-reusable-manifest-sha256", "expected-git-commit",
    ):
        parser.add_argument("--" + name, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        registry_path=args.registry, plan_report_path=args.plan_report,
        expected_plan_report_sha256=args.expected_plan_report_sha256,
        source_audit_report_path=args.source_audit_report,
        expected_source_audit_report_sha256=args.expected_source_audit_report_sha256,
        reusable_manifest_path=args.reusable_manifest,
        expected_reusable_manifest_sha256=args.expected_reusable_manifest_sha256,
        source_spec_path=args.source_spec, output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
