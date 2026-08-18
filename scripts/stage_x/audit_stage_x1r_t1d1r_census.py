#!/usr/bin/env python3
"""Read-only aggregate audit for the frozen D1R SCREENING_CLEAN census."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "configs/STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1.json"
LEDGER_PATH = REPO / "reports/STAGE_X_X1R_T1D1R_CONTINUATION_LEDGER_V1.json"
HOLD_PATH = REPO / "reports/STAGE_X_X1R_T1D1_CANARY_RUNTIME_HOLD_V1.json"
HEAD_AUDIT_PATH = REPO / "reports/STAGE_X_X1R_T1D1R_HEAD_CONTRACT_AUDIT_V1.json"
FORBIDDEN_COUNTERS = (
    "pgd_calls", "attack_backward_calls", "adversarial_images",
    "physical_interventions", "vphys_reads", "attack_outcome_reads",
    "eval160_reads", "protected_reads", "attacked_env_steps",
)
EXPECTED_EXCLUDED = {1, 11, 20, 30}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_receipt() -> dict[str, str]:
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git("status", "--porcelain"),
        "branch": git("branch", "--show-current"),
    }


def fail(errors: list[str]) -> None:
    if errors:
        raise SystemExit("D1R_CENSUS_AUDIT_FAIL\n" + "\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    source = source_receipt()
    protocol = load_json(PROTOCOL_PATH)
    ledger = load_json(LEDGER_PATH)
    hold = load_json(HOLD_PATH)
    head_audit = load_json(HEAD_AUDIT_PATH)

    if source["status_porcelain"]:
        errors.append("WORKTREE_NOT_CLEAN_BEFORE_AUDIT")
    if protocol.get("schema") != "STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1":
        errors.append("PROTOCOL_SCHEMA_INVALID")
    if protocol.get("authorization", {}).get("pgd_authorized") is not False:
        errors.append("PGD_AUTHORITY_NOT_FALSE")
    if protocol.get("protected_boundary", {}).get("eval160") != "UNREAD":
        errors.append("EVAL160_NOT_UNREAD")
    if protocol.get("protected_boundary", {}).get("protected_evaluation") != "UNREAD":
        errors.append("PROTECTED_NOT_UNREAD")
    if sha256_file(HOLD_PATH) != protocol["historical_d1"]["hold_report_sha256"]:
        errors.append("D1_HOLD_REPORT_SHA_MISMATCH")
    if hold.get("status") != "HOLD_RUNTIME_INVALID_AFTER_FIRST_POLICY_DECISION":
        errors.append("D1_HOLD_STATUS_INVALID")
    if sorted(int(row["ordinal"]) for row in hold.get("canaries", [])) != sorted(EXPECTED_EXCLUDED):
        errors.append("D1_EXCLUDED_ORDINALS_INVALID")
    if head_audit.get("status") != "STAGE_X_X1R_T1D1R_HEAD_CONTRACT_PASS":
        errors.append("D1R_HEAD_CONTRACT_AUDIT_NOT_PASS")

    rows = ledger.get("rows", [])
    expected = {int(row["ordinal"]): row for row in rows}
    expected_ordinals = sorted(expected)
    if expected_ordinals != [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39]:
        errors.append("D1R_LEDGER_ORDINALS_INVALID")

    receipt_paths = sorted((root / "parents").glob("*/attempt_*/parent_receipt.json"))
    receipts: list[dict[str, Any]] = []
    by_ordinal: dict[int, dict[str, Any]] = {}
    artifact_rows: list[dict[str, Any]] = []
    evidence_runtime_source: dict[str, Any] | None = None
    for receipt_path in receipt_paths:
        try:
            receipt = load_json(receipt_path)
        except Exception as exc:
            errors.append(f"RECEIPT_JSON_INVALID:{receipt_path}:{type(exc).__name__}")
            continue
        ordinal = int(receipt.get("ordinal", -1))
        if ordinal in by_ordinal:
            errors.append(f"DUPLICATE_RECEIPT_ORDINAL:{ordinal}")
        by_ordinal[ordinal] = receipt
        receipts.append(receipt)
        expected_row = expected.get(ordinal)
        if expected_row is None:
            errors.append(f"UNEXPECTED_RECEIPT_ORDINAL:{ordinal}")
        elif receipt.get("canonical_parent_key") != expected_row.get("canonical_parent_key"):
            errors.append(f"PARENT_KEY_MISMATCH:{ordinal}")
        if receipt.get("schema") != "STAGE_X_X1R_T1D1_SCREENING_CLEAN_PARENT_RECEIPT_V1":
            errors.append(f"RECEIPT_SCHEMA_INVALID:{ordinal}")
        if receipt.get("status") != "PASS_SCREENING_CLEAN_EPISODE":
            errors.append(f"RECEIPT_STATUS_INVALID:{ordinal}")
        if receipt.get("condition") != "SCREENING_CLEAN" or receipt.get("screening_is_not_clean_eval") is not True:
            errors.append(f"CLEAN_SCOPE_INVALID:{ordinal}")
        suite = str(receipt.get("suite", ""))
        horizon = protocol.get("clean_runtime", {}).get("horizons", {}).get(suite)
        if horizon is None or int(receipt.get("policy_horizon", -1)) != int(horizon):
            errors.append(f"HORIZON_BINDING_INVALID:{ordinal}")
        if expected_row is not None and receipt.get("expected_clean_seed") != expected_row.get("expected_clean_seed"):
            errors.append(f"SEED_MISMATCH:{ordinal}")
        if receipt.get("clean_failure") != (not bool(receipt.get("clean_success"))):
            errors.append(f"SUCCESS_FAILURE_COMPLEMENT_INVALID:{ordinal}")
        if receipt.get("student_status") != "PASS_CAUSAL_TRACE":
            errors.append(f"STUDENT_TRACE_INVALID:{ordinal}")
        if receipt.get("no_emit_retained") != (receipt.get("first_emit_step") is None):
            errors.append(f"NO_EMIT_FLAG_INVALID:{ordinal}")
        if receipt.get("manual_clean_contact_review") != "REQUIRED":
            errors.append(f"MANUAL_REVIEW_FLAG_INVALID:{ordinal}")
        if receipt.get("forbidden_actions_executed") != []:
            errors.append(f"FORBIDDEN_ACTIONS_NONEMPTY:{ordinal}")
        counters = receipt.get("counters", {})
        for name in FORBIDDEN_COUNTERS:
            if int(counters.get(name, -1)) != 0:
                errors.append(f"FORBIDDEN_COUNTER_NONZERO:{ordinal}:{name}")
        boundary = receipt.get("protected_boundary", {})
        if boundary.get("eval160") != "UNREAD" or boundary.get("protected_evaluation") != "UNREAD":
            errors.append(f"PROTECTED_BOUNDARY_INVALID:{ordinal}")
        steps = int(receipt.get("policy_steps_executed", -1))
        if steps < 1 or (horizon is not None and steps > int(horizon)):
            errors.append(f"POLICY_STEP_COUNT_INVALID:{ordinal}")
        if int(counters.get("env_step_calls", -1)) != steps + 10:
            errors.append(f"ENV_STEP_ACCOUNTING_INVALID:{ordinal}")
        if int(counters.get("openvla_model_inference_calls", -1)) != steps:
            errors.append(f"OPENVLA_STEP_ACCOUNTING_INVALID:{ordinal}")
        if int(counters.get("prospective_parent_student_forward_calls", -1)) != 1:
            errors.append(f"STUDENT_FORWARD_ACCOUNTING_INVALID:{ordinal}")
        runtime = receipt.get("runtime_source_pre_evidence", {})
        if evidence_runtime_source is None:
            evidence_runtime_source = dict(runtime)
        elif runtime != evidence_runtime_source:
            errors.append(f"RUNTIME_SOURCE_NONUNIFORM:{ordinal}")
        if runtime.get("status_porcelain"):
            errors.append(f"RUNTIME_SOURCE_DIRTY:{ordinal}")

        video = receipt.get("video", {})
        video_path = Path(str(video.get("path", "")))
        if not video_path.is_absolute() or not video_path.is_file():
            errors.append(f"VIDEO_MISSING:{ordinal}")
        else:
            actual_video_sha = sha256_file(video_path)
            if actual_video_sha != video.get("sha256"):
                errors.append(f"VIDEO_SHA_MISMATCH:{ordinal}")
            if video_path.stat().st_size != int(video.get("bytes", -1)):
                errors.append(f"VIDEO_SIZE_MISMATCH:{ordinal}")
            artifact_rows.append({"path": str(video_path), "sha256": actual_video_sha, "kind": "video"})
        parent_dir = receipt_path.parent
        telemetry_path = parent_dir / "step_telemetry.jsonl"
        manifest_path = parent_dir / "episode_manifest.json"
        if not telemetry_path.is_file() or not manifest_path.is_file():
            errors.append(f"EPISODE_ARTIFACT_MISSING:{ordinal}")
        else:
            telemetry_rows = [line for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(telemetry_rows) != steps:
                errors.append(f"TELEMETRY_ROW_COUNT_INVALID:{ordinal}")
            manifest = load_json(manifest_path)
            if manifest.get("attack_enabled") is not False or manifest.get("physical_intervention") is not False or manifest.get("vphys_read") is not False:
                errors.append(f"MANIFEST_FORBIDDEN_SCOPE_INVALID:{ordinal}")
            artifact_rows.extend([
                {"path": str(telemetry_path), "sha256": sha256_file(telemetry_path), "kind": "telemetry"},
                {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "kind": "manifest"},
            ])
        artifact_rows.append({"path": str(receipt_path), "sha256": sha256_file(receipt_path), "kind": "receipt"})

    if len(receipts) != len(expected) or sorted(by_ordinal) != expected_ordinals:
        errors.append(f"RECEIPT_COVERAGE_INVALID:{len(receipts)}/{len(expected)}")
    if evidence_runtime_source is None:
        errors.append("RUNTIME_SOURCE_MISSING")
    worker_paths = sorted((root / "workers").glob("*.json"))
    worker_ordinals: list[int] = []
    for worker_path in worker_paths:
        worker = load_json(worker_path)
        if worker.get("status") != "PASS":
            errors.append(f"WORKER_STATUS_INVALID:{worker_path.name}")
        worker_ordinals.extend(int(value) for value in worker.get("ordinals", []))
        if any(int(worker.get("forbidden_counters", {}).get(name, -1)) != 0 for name in FORBIDDEN_COUNTERS):
            errors.append(f"WORKER_FORBIDDEN_COUNTER_NONZERO:{worker_path.name}")
        artifact_rows.append({"path": str(worker_path), "sha256": sha256_file(worker_path), "kind": "worker"})
    if sorted(worker_ordinals) != expected_ordinals:
        errors.append("WORKER_ORDINAL_COVERAGE_INVALID")

    counts = {
        "continuation_planned": len(expected),
        "runtime_valid": len(receipts),
        "runtime_invalid": 0,
        "clean_success": sum(bool(row.get("clean_success")) for row in receipts),
        "clean_failure": sum(bool(row.get("clean_failure")) for row in receipts),
        "first_emit": sum(row.get("first_emit_step") is not None for row in receipts),
        "no_emit": sum(row.get("first_emit_step") is None for row in receipts),
        "first_emit_legal": sum(row.get("first_emit_step") is not None and int(row["first_emit_step"]) + 5 + 10 <= int(row["policy_horizon"]) for row in receipts),
        "first_emit_illegal": sum(row.get("first_emit_step") is not None and int(row["first_emit_step"]) + 5 + 10 > int(row["policy_horizon"]) for row in receipts),
        "attack_eligible_pre_manual_review": sum(bool(row.get("attack_eligible_pre_manual_review")) for row in receipts),
    }
    if counts["runtime_valid"] != 35 or counts["runtime_invalid"] != 0 or counts["first_emit_illegal"] != 0:
        errors.append("D1R_CENSUS_GATE_INVALID")
    for suite, expected_count in protocol["parent_population"]["continuation_suite_counts"].items():
        actual = sum(row.get("suite") == suite for row in receipts)
        if actual != int(expected_count):
            errors.append(f"SUITE_COUNT_INVALID:{suite}:{actual}/{expected_count}")

    fail(errors)
    historical = {
        "pr129_live_head": {"commit": "4b0ceb65f8f7babdd29163e032c56fed3ba57526", "tree": "d7b688e82bf0b9c5e91c08b3ad15c3a6d94b89ad"},
        "d1_runtime_source": {"commit": protocol["historical_d1"]["source_pre_evidence_commit"], "tree": protocol["historical_d1"]["source_pre_evidence_tree"]},
        "d1r_repaired_source_pre_evidence": head_audit.get("source", {}),
        "d1r_evidence_runtime_source": evidence_runtime_source,
    }
    census = {
        "schema": "STAGE_X_X1R_T1D1R_CENSUS_AUDIT_V1",
        "status": "PASS_D1R_CONTINUATION_CENSUS_PRE_MANUAL_REVIEW",
        "scope": "SCREENING_CLEAN_ONLY",
        "source_before_evidence_outputs": source,
        "official_environment": protocol["official_environment"],
        "durable_root": str(root),
        "historical": historical,
        "population": {
            "nominal_design_cells": 40,
            "missing_cell": protocol["parent_population"]["missing_cell"],
            "frozen_executable_design": 39,
            "immutable_runtime_invalid_consumed_ordinals": sorted(EXPECTED_EXCLUDED),
            "continuation_ordinals": expected_ordinals,
            "continuation_count": len(expected),
            "replacement": False,
            "rerank": False,
        },
        "counts": counts,
        "suite_counts": {suite: sum(row.get("suite") == suite for row in receipts) for suite in sorted({row.get("suite") for row in receipts})},
        "student": {"head_contract_audit": "PASS", "checkpoint_sha256": protocol["student"]["checkpoint_sha256"], "source_raw_sha256": protocol["student"]["source_raw_sha256"], "thresholds_unchanged": True, "features_unchanged": True},
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "forbidden_counters_zero": True},
        "manual_review": {"clean_contact_validity": "NOT_REVIEWED", "video_review_required": True, "next_gate": "OWNER_REVIEW_D1R_CENSUS_AND_MANUAL_CONTACT_VALIDITY"},
        "artifact_rows": sorted(artifact_rows, key=lambda row: (row["kind"], row["path"])),
    }
    report_path = out / "STAGE_X_X1R_T1D1R_CENSUS_AUDIT_V1.json"
    report_path.write_text(json.dumps(census, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_artifacts = [PROTOCOL_PATH, LEDGER_PATH, HOLD_PATH, HEAD_AUDIT_PATH, report_path]
    manifest_rows = [{"path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(path), "kind": "repo"} for path in repo_artifacts]
    manifest_rows.extend(sorted(artifact_rows, key=lambda row: (row["kind"], row["path"])))
    sums_path = out / "STAGE_X_X1R_T1D1R_SHA256SUMS.txt"
    sums_path.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in manifest_rows), encoding="utf-8")
    sums_sha = sha256_file(sums_path)
    (out / "STAGE_X_X1R_T1D1R_SHA256SUMS.sha256").write_text(f"{sums_sha}  STAGE_X_X1R_T1D1R_SHA256SUMS.txt\n", encoding="utf-8")
    seal = {
        "schema": "STAGE_X_X1R_T1D1R_ROOT_SEAL_V1",
        "status": "PASS_D1R_CONTINUATION_CENSUS_PRE_MANUAL_REVIEW",
        "scope": "SCREENING_CLEAN_ONLY",
        "source_before_evidence_outputs": source,
        "historical": historical,
        "durable_root": str(root),
        "census_report": {"path": str(report_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(report_path)},
        "sha256sums": {"path": str(sums_path.relative_to(REPO)).replace("\\", "/"), "sha256": sums_sha},
        "population": census["population"],
        "counts": counts,
        "protected_boundary": census["protected_boundary"],
        "manual_review": census["manual_review"],
        "seal_note": "Git source/tree captured before evidence outputs; final GitHub evidence commit/tree is recorded by the subsequent handoff and is not inferred from this pre-output snapshot.",
    }
    seal_path = out / "STAGE_X_X1R_T1D1R_ROOT_SEAL.json"
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_sha = sha256_file(seal_path)
    (out / "STAGE_X_X1R_T1D1R_ROOT_SEAL.sha256").write_text(f"{seal_sha}  STAGE_X_X1R_T1D1R_ROOT_SEAL.json\n", encoding="utf-8")
    print(json.dumps({"status": census["status"], "continuation": len(receipts), "counts": counts, "census_sha256": sha256_file(report_path), "sha256sums_sha256": sums_sha, "root_seal_sha256": seal_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
