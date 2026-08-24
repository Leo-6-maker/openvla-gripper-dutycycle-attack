"""Audit owner labels and freeze the blind-mapped Stage X X1R cohort."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


OWNER_SHA = "76c835c292c76190b2a764da7be746a568697deea1cb4009d63e306fdc610c2c"
MAPPING_SHA = "3d7f59a736cc2c7bcb5ecdc49e9e57a7e8b547c9e7554251e88158017366f0fe"
SAFE_SHEET_SHA = "9c42b3d6486f2082c414ff799d9efe2f2633e797d2b92e1f45fb23426470a7b2"
CANDIDATE_SHA = "5f1f036b47b1c9a8c1bafe7a400b6be9269cd3e67587691018005c824dc8d89e"
ORDER_DIGEST = "30a73b0e4ab13e149d8c991906fc9067844797e39113201e9e76a10a8be40d67"
ORDER_SALT = "STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_ORDER_V1_20260818"
REVIEW_IDS = [f"M{i:03d}" for i in range(1, 15)]
LABELS = {"PASS", "FAIL", "ABSTAIN"}
FAIL_CODES = {
    "PRECONTACT_OR_APPROACH",
    "WRONG_OR_IRRELEVANT_OBJECT_PART",
    "RELEASE_ALREADY_STARTED",
    "RELEASE_SAFE_OR_INDEPENDENTLY_SUPPORTED",
    "CONTACT_ALREADY_LOST_OR_SLIPPING",
    "OTHER_CLEARLY_NON_GRIPPER_DEPENDENT",
}
IDENTITY_FIELDS = (
    "canonical_parent_key",
    "suite",
    "task_idx",
    "state_id",
    "ordinal",
    "task_instruction",
    "expected_clean_seed",
    "first_emit_step",
    "legal_horizon",
    "policy_horizon",
    "policy_steps_executed",
    "context_start",
    "context_end",
    "parent_receipt_path",
    "parent_receipt_sha256",
    "telemetry_path",
    "telemetry_sha256",
    "raw_clean_video_path",
    "raw_clean_video_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare_rows(left: dict[str, Any], right: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if str(left.get(field)) != str(right.get(field))]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    reports = root / "reports"
    owner_path = reports / "STAGE_X_X1R_T1D1M1_OWNER_LABEL_SUBMISSION_V1.csv"
    safe_path = reports / "STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1.csv"
    mapping_path = reports / "STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json"
    premanual_path = reports / "STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json"
    upstream_seal_path = reports / "STAGE_X_X1R_T1D1M0_ROOT_SEAL.json"
    binding_path = reports / "STAGE_X_X1R_T1D1M1_OWNER_BINDING_RECEIPT_V1.json"
    protocol_path = root / "configs" / "STAGE_X_X1R_PRIMARY_MATRIX_PROTOCOL_V1.json"

    # ponytail: one pass over the small sealed packet is enough; a database adds no value here.
    owner_raw_sha = sha256(owner_path)
    errors: list[str] = []
    if owner_raw_sha != OWNER_SHA:
        errors.append(f"owner raw SHA mismatch: {owner_raw_sha}")

    owner_fields, owner_rows = read_csv(owner_path)
    safe_fields, safe_rows = read_csv(safe_path)
    mapping = read_json(mapping_path)
    premanual = read_json(premanual_path)
    upstream_seal = read_json(upstream_seal_path)
    binding = read_json(binding_path)
    protocol = read_json(protocol_path)

    expected_owner_fields = [
        "review_id",
        "task_instruction",
        "review_clip_path",
        "review_clip_sha256",
        "review_frame_strip_path",
        "review_frame_strip_sha256",
        "contact_label",
        "reason_code",
        "reviewer",
        "review_timestamp",
        "optional_short_note",
    ]
    expected_safe_fields = expected_owner_fields[:6] + expected_owner_fields[6:]
    if owner_fields != expected_owner_fields:
        errors.append("owner CSV header mismatch")
    if safe_fields != expected_safe_fields:
        errors.append("safe review-sheet header mismatch")
    if [row.get("review_id") for row in owner_rows] != REVIEW_IDS:
        errors.append("owner review IDs are not exactly M001..M014 in order")
    if [row.get("review_id") for row in safe_rows] != REVIEW_IDS:
        errors.append("safe-sheet review IDs are not exactly M001..M014 in order")

    safe_by_id = {row.get("review_id"): row for row in safe_rows}
    owner_by_id = {row.get("review_id"): row for row in owner_rows}
    for review_id in REVIEW_IDS:
        owner = owner_by_id.get(review_id, {})
        safe = safe_by_id.get(review_id, {})
        for field in expected_owner_fields[:6]:
            if owner.get(field) != safe.get(field):
                errors.append(f"{review_id} safe-sheet field mismatch: {field}")
        label = owner.get("contact_label", "")
        reason = owner.get("reason_code", "")
        if label not in LABELS:
            errors.append(f"{review_id} invalid contact label: {label!r}")
        elif label == "PASS" and reason:
            errors.append(f"{review_id} PASS must have blank reason_code")
        elif label == "ABSTAIN" and reason:
            errors.append(f"{review_id} ABSTAIN must have blank reason_code")
        elif label == "FAIL" and reason not in FAIL_CODES:
            errors.append(f"{review_id} FAIL has invalid reason_code: {reason!r}")
        if not owner.get("reviewer") or not owner.get("review_timestamp"):
            errors.append(f"{review_id} missing reviewer/timestamp")

    mapping_rows = list(mapping.get("rows", []))
    premanual_rows = list(premanual.get("rows", []))
    if len(owner_rows) != 14 or len(safe_rows) != 14 or len(mapping_rows) != 14 or len(premanual_rows) != 14:
        errors.append("one or more frozen 14-row inputs have the wrong row count")
    if mapping.get("order_salt") != ORDER_SALT:
        errors.append("mapping order salt mismatch")

    mapping_by_id = {row.get("review_id"): row for row in mapping_rows}
    premanual_by_key = {row.get("canonical_parent_key"): row for row in premanual_rows}
    seen_keys: set[str] = set()
    for review_id in REVIEW_IDS:
        mapped = mapping_by_id.get(review_id)
        if mapped is None:
            errors.append(f"missing mapping row: {review_id}")
            continue
        key = str(mapped.get("canonical_parent_key"))
        if key in seen_keys:
            errors.append(f"duplicate mapped parent: {key}")
        seen_keys.add(key)
        expected_rank = hashlib.sha256(f"{ORDER_SALT}|{key}".encode()).hexdigest()
        if mapped.get("rank_key") != expected_rank:
            errors.append(f"rank mismatch: {review_id}")
        pre = premanual_by_key.get(key)
        if pre is None:
            errors.append(f"mapped parent missing from premanual ledger: {key}")
            continue
        mismatches = compare_rows(mapped, pre, IDENTITY_FIELDS)
        if mismatches:
            errors.append(f"mapping/premanual mismatch for {review_id}: {','.join(mismatches)}")
        if mapped.get("review_id") != review_id:
            errors.append(f"mapping review ID mismatch for {key}")

    sorted_mapping_ids = [row.get("review_id") for row in sorted(mapping_rows, key=lambda row: (row.get("rank_key", ""), row.get("canonical_parent_key", "")))]
    if sorted_mapping_ids != REVIEW_IDS:
        errors.append("mapping order is not the frozen ascending rank order")
    if upstream_seal.get("mapping_sha256") != MAPPING_SHA:
        errors.append("upstream mapping seal does not carry the expected SHA")
    if upstream_seal.get("candidate_ledger_sha256") != CANDIDATE_SHA:
        errors.append("upstream candidate-ledger SHA mismatch")
    if upstream_seal.get("order_digest") != ORDER_DIGEST:
        errors.append("upstream review-order digest mismatch")
    if binding.get("owner_submission", {}).get("raw_sha256") != owner_raw_sha:
        errors.append("owner binding receipt does not match the raw owner SHA")
    if protocol.get("status") != "FROZEN_PRE_LABEL_INGESTION":
        errors.append("primary protocol is not the pre-ingestion frozen version")

    dispositions: list[dict[str, Any]] = []
    cohort: list[dict[str, Any]] = []
    for review_id in REVIEW_IDS:
        owner = owner_by_id[review_id]
        mapped = mapping_by_id[review_id]
        label = owner["contact_label"]
        eligible = label == "PASS"
        disposition = {
            "review_id": review_id,
            "contact_label": label,
            "reason_code": owner.get("reason_code", ""),
            "reviewer": owner.get("reviewer", ""),
            "review_timestamp": owner.get("review_timestamp", ""),
            "canonical_parent_key": mapped.get("canonical_parent_key"),
            "suite": mapped.get("suite"),
            "task_idx": mapped.get("task_idx"),
            "state_id": mapped.get("state_id"),
            "ordinal": mapped.get("ordinal"),
            "attack_eligible": eligible,
            "selection_rule": "CONTACT_VALID PASS only; no score/outcome selection",
        }
        dispositions.append(disposition)
        if eligible:
            cohort.append({
                "review_id": review_id,
                "canonical_parent_key": mapped.get("canonical_parent_key"),
                "suite": mapped.get("suite"),
                "task_idx": mapped.get("task_idx"),
                "state_id": mapped.get("state_id"),
                "ordinal": mapped.get("ordinal"),
                "task_instruction": mapped.get("task_instruction"),
                "expected_clean_seed": mapped.get("expected_clean_seed"),
                "first_emit_step": mapped.get("first_emit_step"),
                "context_start": mapped.get("context_start"),
                "context_end": mapped.get("context_end"),
                "policy_horizon": mapped.get("policy_horizon"),
                "policy_steps_executed": mapped.get("policy_steps_executed"),
                "legal_horizon": mapped.get("legal_horizon"),
                "parent_receipt_path": mapped.get("parent_receipt_path"),
                "parent_receipt_sha256": mapped.get("parent_receipt_sha256"),
                "telemetry_path": mapped.get("telemetry_path"),
                "telemetry_sha256": mapped.get("telemetry_sha256"),
                "raw_clean_video_path": mapped.get("raw_clean_video_path"),
                "raw_clean_video_sha256": mapped.get("raw_clean_video_sha256"),
                "eligibility_source": "owner CONTACT_VALID PASS + frozen D1R premanual receipt",
            })

    audit = {
        "schema": "STAGE_X_X1R_T1D1M1_OWNER_LABEL_INTEGRITY_AUDIT_V1",
        "status": "PASS" if not errors else "HOLD_D1M1_LABEL_INTEGRITY",
        "scope": "owner label integrity, blind mapping, and final cohort freeze; no model or environment execution",
        "source_bindings": {
            "owner_submission_sha256": owner_raw_sha,
            "expected_owner_submission_sha256": OWNER_SHA,
            "safe_sheet_expected_sealed_sha256": SAFE_SHEET_SHA,
            "safe_sheet_worktree_sha256": sha256(safe_path),
            "mapping_expected_sealed_sha256": MAPPING_SHA,
            "mapping_worktree_sha256": sha256(mapping_path),
            "premanual_ledger_sha256": sha256(premanual_path),
            "d1m0_root_seal_sha256": sha256(upstream_seal_path),
            "candidate_ledger_sha256": CANDIDATE_SHA,
            "review_order_digest": ORDER_DIGEST,
        },
        "checks": {
            "raw_owner_sha_match": owner_raw_sha == OWNER_SHA,
            "safe_projection_fields_exact": not any("safe-sheet field mismatch" in error for error in errors),
            "label_grammar_valid": not any("invalid contact label" in error or "must have" in error or "FAIL has" in error for error in errors),
            "blind_order_recomputed": not any("rank mismatch" in error or "mapping order" in error for error in errors),
            "mapping_to_premanual_exact": not any("mapping/premanual mismatch" in error for error in errors),
            "one_to_one_parent_identity": len(seen_keys) == 14,
            "no_replacement": True,
            "no_outcome_based_selection": True,
        },
        "counts": {
            "submitted_rows": len(owner_rows),
            "mapping_rows": len(mapping_rows),
            "premanual_rows": len(premanual_rows),
            "pass": sum(row.get("contact_label") == "PASS" for row in owner_rows),
            "fail": sum(row.get("contact_label") == "FAIL" for row in owner_rows),
            "abstain": sum(row.get("contact_label") == "ABSTAIN" for row in owner_rows),
            "final_attack_eligible": len(cohort),
        },
        "errors": errors,
        "protected_boundary": {
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
            "model_inference": 0,
            "student_inference": 0,
            "env_step": 0,
            "pgd_calls": 0,
            "attacked_action_steps": 0,
            "physical_interventions": 0,
            "vphys_reads": 0,
            "attack_outcome_reads": 0,
            "eval160_reads": 0,
            "protected_reads": 0,
        },
        "next_gate": "STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_REQUIRED" if not errors and cohort else "OWNER_REVIEW_D1M1_AUDIT_REQUIRED",
    }
    write_json(reports / "STAGE_X_X1R_T1D1M1_OWNER_LABEL_INTEGRITY_AUDIT_V1.json", audit)
    if errors:
        return 2

    ledger = {
        "schema": "STAGE_X_X1R_T1D1M1_MANUAL_DISPOSITION_LEDGER_V1",
        "status": "FROZEN_AFTER_OWNER_LABEL_INTEGRITY_PASS",
        "selection_rule": "CONTACT_VALID PASS only; FAIL and ABSTAIN remain in the ledger and are non-eligible",
        "source": {
            "owner_submission_sha256": owner_raw_sha,
            "mapping_sha256": MAPPING_SHA,
            "candidate_ledger_sha256": CANDIDATE_SHA,
            "review_order_digest": ORDER_DIGEST,
        },
        "rows": dispositions,
        "protected_boundary": audit["protected_boundary"],
        "next_gate": "STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_REQUIRED",
    }
    cohort_manifest = {
        "schema": "STAGE_X_X1R_T1D1M1_FINAL_ATTACK_COHORT_V1",
        "status": "FROZEN_PRE_ATTACK_IMPLEMENTATION_AUDIT",
        "selection_rule": "owner CONTACT_VALID PASS only after exact blind mapping; no Student score, physical score, attack outcome, or replacement",
        "count": len(cohort),
        "rows": cohort,
        "source": {
            "owner_submission_sha256": owner_raw_sha,
            "mapping_sha256": MAPPING_SHA,
            "premanual_ledger_sha256": sha256(premanual_path),
            "manual_disposition_ledger": "reports/STAGE_X_X1R_T1D1M1_MANUAL_DISPOSITION_LEDGER_V1.json",
        },
        "protected_boundary": audit["protected_boundary"],
        "next_gate": "STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_REQUIRED",
    }
    write_json(reports / "STAGE_X_X1R_T1D1M1_MANUAL_DISPOSITION_LEDGER_V1.json", ledger)
    write_json(reports / "STAGE_X_X1R_T1D1M1_FINAL_ATTACK_COHORT_V1.json", cohort_manifest)

    report_paths = [
        reports / "STAGE_X_X1R_T1D1M1_OWNER_LABEL_INTEGRITY_AUDIT_V1.json",
        reports / "STAGE_X_X1R_T1D1M1_MANUAL_DISPOSITION_LEDGER_V1.json",
        reports / "STAGE_X_X1R_T1D1M1_FINAL_ATTACK_COHORT_V1.json",
    ]
    root_seal = {
        "schema": "STAGE_X_X1R_T1D1M1_ROOT_SEAL_V1",
        "status": "STAGE_X_X1R_T1D1M1_LABEL_INTEGRITY_PASS_COHORT_FROZEN",
        "ancestry": {
            "base_pr": 132,
            "base_commit": "14f7df98995262ede4fc129578001ffe40431582",
            "base_tree": "8e59bdeee666a92a776e9ef99f7c4ac1dd6801f2",
            "goal_branch": "codex/stage-x-x1r-goal-d1m1-primary-matrix-20260819",
        },
        "source_bindings": {
            "owner_submission_sha256": owner_raw_sha,
            "mapping_sha256": MAPPING_SHA,
            "candidate_ledger_sha256": CANDIDATE_SHA,
            "review_order_digest": ORDER_DIGEST,
            "primary_protocol": "configs/STAGE_X_X1R_PRIMARY_MATRIX_PROTOCOL_V1.json",
            "primary_protocol_sha256": sha256(protocol_path),
        },
        "counts": audit["counts"],
        "manual_labels_are_final_for_this_cohort": True,
        "teacher_student_changed": False,
        "replacement_or_rerank": False,
        "protected_boundary": audit["protected_boundary"],
        "included_report_sha256": {path.name: sha256(path) for path in report_paths},
        "next_gate": "STAGE_X_X1R_G2_ATTACK_IMPLEMENTATION_AUDIT_REQUIRED",
    }
    seal_path = reports / "STAGE_X_X1R_T1D1M1_ROOT_SEAL.json"
    write_json(seal_path, root_seal)
    (reports / "STAGE_X_X1R_T1D1M1_ROOT_SEAL.sha256").write_text(f"{sha256(seal_path)}  {seal_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "final_attack_eligible": len(cohort), "root_seal": str(seal_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
