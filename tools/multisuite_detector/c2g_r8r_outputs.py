"""Stable report and artifact writers for the C2g R8R audit."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.multisuite_detector.c2g_r8r_common import sha256_file, write_csv, write_json, write_jsonl
from tools.multisuite_detector.c2g_r8r_episode import (
    A_DIRECT, B_AUGMENT, C_LEGACY, D_RECOLLECT, CLASSIFICATIONS,
)

HOLD_IDENTITY = "HOLD_CLEAN2000_IDENTITY_INTEGRITY"
HOLD_PROVENANCE = "HOLD_CLEAN2000_PROVENANCE"
HOLD_TEACHER = "HOLD_TEACHER_V2_RAW_EVIDENCE"
GO_DIRECT = "GO_DIRECT_REUSE_MATERIALIZATION"
GO_AUGMENT = "GO_OFFLINE_AUGMENTATION"
GO_PARTIAL = "GO_PARTIAL_REUSE_AND_RESIDUAL_COLLECTION"

EP_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split",
    "classification", "classification_reason", "source_view_name", "source_class",
    "source_root", "metadata_path", "step_records_path", "metadata_sha256",
    "step_records_sha256", "n_steps", "w16_window_count", "condition",
    "clean_boundary_valid", "runtime_valid", "error_record_present", "rgb_count",
    "rgb_missing_count", "rgb_complete", "task_language_present",
    "features_25d_shape_complete", "features_25d_names_exact",
    "feature_25d_order_bound_by_manifest", "canonical_25d_complete",
    "policy_intent_9d_complete", "raw_policy_logits_complete",
    "model_provenance_bound", "processor_provenance_bound",
    "derived_feature_reconstruction_possible", "teacher_v1_label_present",
    "legacy_label_fields_present", "teacher_v2_schema_marker_present",
    "teacher_v2_target_raw_present", "teacher_v2_contact_raw_present",
    "teacher_v2_progress_raw_present", "teacher_v2_release_raw_present",
    "teacher_v2_command_semantics_present", "teacher_v2_raw_evidence_complete",
    "teacher_v2_rebuild_attempted", "teacher_v2_rebuild_success",
    "teacher_v2_rebuild_error", "known_positive_steps", "known_negative_steps",
    "unknown_steps", "positive_episode", "fully_known_negative_episode",
    "triggerable_positive_episode", "clean_success_observed",
    "legacy_semantic_salvage_candidate",
)
VIEW_FIELDS = (
    "source_view_name", "source_root", "source_class", "priority",
    "canonical_for_suite", "suite", "task_index", "state_id", "registered",
    "parent_key", "cohort", "split", "identity_resolution_method",
    "identity_resolution_error", "physical_view_count", "selected_canonical",
    "canonical_conflict", "metadata_path", "step_records_path", "metadata_sha256",
    "step_records_sha256",
)
ID_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split",
    "physical_view_count", "canonical_candidate_count", "selected_source_view",
    "selected_metadata_path", "identity_status",
)
TEACHER_FIELDS = (
    "suite", "task_index", "state_id", "parent_key", "cohort", "split",
    "teacher_v1_label_present", "legacy_label_fields_present",
    "teacher_v2_schema_marker_present", "teacher_v2_target_raw_present",
    "teacher_v2_contact_raw_present", "teacher_v2_progress_raw_present",
    "teacher_v2_release_raw_present", "teacher_v2_command_semantics_present",
    "teacher_v2_raw_evidence_complete", "teacher_v2_rebuild_attempted",
    "teacher_v2_rebuild_success", "teacher_v2_rebuild_error",
    "known_positive_steps", "known_negative_steps", "unknown_steps",
    "positive_episode", "fully_known_negative_episode",
    "triggerable_positive_episode", "legacy_semantic_salvage_candidate",
)


def classification_counts(rows: Sequence[Mapping[str, Any]], field: str):
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[str(row[field])][str(row["classification"])] += 1
    return {
        key: {classification: int(counter[classification]) for classification in CLASSIFICATIONS}
        for key, counter in sorted(grouped.items())
    }


def task_classification_counts(rows: Sequence[Mapping[str, Any]]):
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[f"{row['suite']}/task_{row['task_index']}"][str(row["classification"])] += 1
    return {
        key: {classification: int(counter[classification]) for classification in CLASSIFICATIONS}
        for key, counter in sorted(grouped.items())
    }


def coverage(rows: Sequence[Mapping[str, Any]], field: str):
    count = sum(bool(row.get(field)) for row in rows)
    return {"count": count, "total": len(rows), "fraction": count / len(rows) if rows else 0.0}


def final_decision(registry_count: int, canonical_count: int, conflicts: int,
                   unregistered: int, rows: Sequence[Mapping[str, Any]]) -> str:
    if canonical_count != registry_count or conflicts or unregistered:
        return HOLD_IDENTITY
    if any(
        not row.get("clean_boundary_valid")
        or not row.get("model_provenance_bound")
        or not row.get("processor_provenance_bound")
        for row in rows
    ):
        return HOLD_PROVENANCE
    counts = Counter(str(row["classification"]) for row in rows)
    if counts[C_LEGACY]:
        return HOLD_TEACHER
    if counts[D_RECOLLECT]:
        return GO_PARTIAL
    if counts[B_AUGMENT]:
        return GO_AUGMENT
    return GO_DIRECT


def candidate_root_rows(views, source_rows):
    totals = Counter(row["source_view_name"] for row in source_rows)
    canonical = Counter(
        row["source_view_name"] for row in source_rows if row["selected_canonical"]
    )
    rows = []
    for view in sorted(views, key=lambda item: (item.source_class, item.name)):
        rows.append({
            "name": view.name,
            "root": str(view.root),
            "source_class": view.source_class,
            "canonical_suites": list(view.canonical_suites),
            "priority": view.priority,
            "episode_view_count": totals[view.name],
            "canonical_episode_count": canonical[view.name],
            "clean_only": view.clean_only,
            "runtime_valid_by_manifest": view.runtime_valid_by_manifest,
            "model_provenance_bound": view.model_provenance_bound,
            "processor_provenance_bound": view.processor_provenance_bound,
            "feature_25d_order_bound": view.feature_25d_order_bound,
            "evidence_paths": [str(path) for path in view.evidence_paths],
            "evidence_sha256": {str(path): sha256_file(path) for path in view.evidence_paths},
        })
    return rows


def write_primary_artifacts(output: Path, source_rows, reconciliation, episodes,
                            roots, source_spec):
    write_csv(output / "clean2000_r7_source_view_ledger.csv", source_rows, VIEW_FIELDS)
    write_csv(output / "clean2000_r7_identity_reconciliation.csv", reconciliation, ID_FIELDS)
    write_csv(output / "clean2000_r7_episode_ledger.csv", episodes, EP_FIELDS)
    write_csv(output / "clean2000_r7_field_coverage.csv", episodes, EP_FIELDS)
    write_csv(output / "clean2000_r7_teacher_v2_support.csv", episodes, TEACHER_FIELDS)
    for classification, name in (
        (A_DIRECT, "direct_reuse"), (B_AUGMENT, "offline_augmentation"),
        (C_LEGACY, "legacy_only"), (D_RECOLLECT, "recollect_required"),
    ):
        write_jsonl(
            output / f"clean2000_r7_{name}.jsonl",
            [row for row in episodes if row["classification"] == classification],
        )
    write_jsonl(
        output / "clean2000_r7_legacy_semantic_salvage_candidates.jsonl",
        [row for row in episodes if row.get("legacy_semantic_salvage_candidate")],
    )
    write_jsonl(output / "clean2000_candidate_roots.jsonl", roots)
    write_json(output / "bound_source_spec.json", source_spec)


def write_hash_ledgers(output: Path):
    files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    hashes = {path.name: sha256_file(path) for path in files}
    sums = output / "SHA256SUMS"
    sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    self_binding = output / "SHA256SUMS.sha256"
    self_binding.write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return hashes, sums_sha, sha256_file(self_binding)
