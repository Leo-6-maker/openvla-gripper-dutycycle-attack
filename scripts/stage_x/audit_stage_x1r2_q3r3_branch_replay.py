#!/usr/bin/env python3
"""Aggregate the four Q3R3 engineering branch-replay receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
EXPOSED = {
    "libero_10/task_08/state_44",
    "libero_goal/task_02/state_37",
    "libero_object/task_01/state_34",
    "libero_spatial/task_09/state_29",
}
ZERO_FIELDS = ("pgd_calls", "physical_interventions", "vphys_reads", "attack_outcome_reads", "attacked_env_steps", "protected_reads")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_zero(boundary: dict[str, Any], errors: list[str], label: str) -> None:
    for field in ZERO_FIELDS:
        if int(boundary.get(field, -1)) != 0:
            errors.append(f"{label}:{field}={boundary.get(field)!r}")
    if boundary.get("eval160") != "UNREAD" or boundary.get("protected_evaluation") != "UNREAD":
        errors.append(f"{label}:PROTECTED_STATUS_INVALID")


def audit_suite(path: Path, source_commit: str, source_tree: str, errors: list[str], repairs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"MISSING_SUITE_REPORT:{path}")
        return {"path": str(path), "status": "MISSING"}
    report = load(path)
    suite = str(report.get("suite", ""))
    if report.get("status") != "PASS_SUITE_BRANCH_REPLAY":
        errors.append(f"{suite}:SUITE_STATUS:{report.get('status')}")
    if report.get("scientific_authority") is not False:
        errors.append(f"{suite}:SCIENTIFIC_AUTHORITY_NOT_FALSE")
    observed_model = report.get("model_identity_observed", {}).get("identity", {})
    if not observed_model.get("tree_sha256") or int(observed_model.get("file_count", 0)) <= 0 or int(observed_model.get("bytes", 0)) <= 0:
        errors.append(f"{suite}:MODEL_IDENTITY_RECEIPT_MISSING")
    source = report.get("source", {})
    if source.get("commit") != source_commit or source.get("tree") != source_tree:
        errors.append(f"{suite}:SOURCE_ARGUMENT_BINDING_MISMATCH")
    if source.get("repository_observed_head") != source_commit or source.get("repository_observed_tree") != source_tree:
        errors.append(f"{suite}:SOURCE_OBSERVED_BINDING_MISMATCH")
    if source.get("status_porcelain"):
        errors.append(f"{suite}:SOURCE_WORKTREE_NOT_CLEAN")
    selected = report.get("selected_parent_key")
    if not selected or selected in EXPOSED:
        errors.append(f"{suite}:INVALID_SELECTED_PARENT:{selected}")
    scan = report.get("scan", [])
    if not scan:
        errors.append(f"{suite}:EMPTY_SCAN")
    if len({row.get("canonical_parent_key") for row in scan}) != len(scan):
        errors.append(f"{suite}:DUPLICATE_SCAN_IDENTITY")
    for row in scan:
        if row.get("canonical_parent_key") in EXPOSED:
            errors.append(f"{suite}:EXPOSED_SCAN_IDENTITY:{row.get('canonical_parent_key')}")
    branches = report.get("branch_receipts", [])
    if len(branches) != 2:
        errors.append(f"{suite}:BRANCH_REPEAT_COUNT:{len(branches)}")
    for branch in branches:
        label = f"{suite}:repeat_{branch.get('repeat')}"
        if branch.get("status") != "PASS_BRANCH_REPLAY":
            errors.append(f"{label}:STATUS:{branch.get('status')}")
        if not branch.get("state_audit", {}).get("equal"):
            errors.append(f"{label}:STATE_NOT_EQUAL")
        if branch.get("clean_direct_tokens_match") is not True:
            errors.append(f"{label}:DIRECT_TOKENS_NOT_EQUAL")
        if int(branch.get("prebranch_openvla_calls", -1)) != 0 or int(branch.get("prebranch_student_calls", -1)) != 0:
            errors.append(f"{label}:PREBRANCH_MODEL_CALL")
        if int(branch.get("branch_student_calls", -1)) != 0:
            errors.append(f"{label}:BRANCH_STUDENT_CALL")
        if int(branch.get("post_branch_steps", -1)) != 15:
            errors.append(f"{label}:POST_BRANCH_STEPS:{branch.get('post_branch_steps')}")
        if not branch.get("reference_observation_sha256") or not branch.get("live_branch_observation_sha256"):
            errors.append(f"{label}:OBSERVATION_RECEIPT_MISSING")
        boundary = dict(branch.get("protected_boundary", {}))
        if "protected_reads" not in boundary:
            repair = repairs.get((suite, int(branch.get("repeat", -1))))
            if not repair or repair.get("field_added") != "protected_reads" or int(repair.get("derived_value", -1)) != 0:
                errors.append(f"{label}:PROTECTED_READS_MISSING_WITHOUT_REPAIR")
            else:
                raw_path = path.parent / str(report.get("selected_fixture")) / f"branch_repeat_{int(branch['repeat'])}.json"
                if repair.get("raw_receipt_path") != raw_path.relative_to(path.parents[1]).as_posix() or repair.get("raw_receipt_sha256") != sha256_file(raw_path):
                    errors.append(f"{label}:REPAIR_RAW_SHA_MISMATCH")
                boundary["protected_reads"] = 0
        check_zero(boundary, errors, label)
    check_zero(report.get("protected_boundary", {}), errors, f"{suite}:suite")
    return {
        "suite": suite,
        "status": report.get("status"),
        "selected_parent_key": selected,
        "selected_fixture": report.get("selected_fixture"),
        "scan_count": len(scan),
        "branch_count": len(branches),
        "report_sha256": sha256_file(path),
    }


def write_root_seal(root: Path, source_commit: str, source_tree: str, suite_rows: list[dict[str, Any]]) -> tuple[Path, str]:
    seal = root / "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_ROOT_SEAL_V1.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != seal):
        files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    payload = {
        "schema": "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_ROOT_SEAL_V1",
        "status": "PASS_FOUR_SUITE_BRANCH_REPLAY",
        "source": {"commit": source_commit, "tree": source_tree},
        "suite_rows": suite_rows,
        "files": files,
        "protected_boundary": {"pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"},
        "scientific_authority": False,
        "next_gate": "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS",
    }
    seal.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seal, sha256_file(seal)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--schema-repair", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    repairs: dict[tuple[str, int], dict[str, Any]] = {}
    if args.schema_repair:
        repair_doc = load(args.schema_repair)
        if repair_doc.get("status") != "PASS_APPEND_ONLY_RECEIPT_SCHEMA_REPAIR" or not repair_doc.get("raw_receipts_unchanged"):
            errors.append("RECEIPT_SCHEMA_REPAIR_NOT_PASS")
        if repair_doc.get("execution_source") != {"commit": args.source_commit, "tree": args.source_tree}:
            errors.append("RECEIPT_SCHEMA_REPAIR_SOURCE_MISMATCH")
        for repair in repair_doc.get("repairs", []):
            repairs[(str(repair.get("suite")), int(repair.get("repeat", -1)))] = repair
    rows = [audit_suite(args.root / suite / "SUITE_BRANCH_REPLAY_REPORT_V1.json", args.source_commit, args.source_tree, errors, repairs) for suite in SUITES]
    selected = [row.get("selected_parent_key") for row in rows if row.get("selected_parent_key")]
    if len(selected) != len(set(selected)):
        errors.append("SELECTED_PARENT_DUPLICATE_ACROSS_SUITES")
    status = "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS" if not errors and len(rows) == 4 else "HOLD_Q3R3_FOUR_SUITE_BRANCH_REPLAY"
    root_seal = None
    root_seal_sha256 = None
    if status.endswith("PASS"):
        root_seal, root_seal_sha256 = write_root_seal(args.root, args.source_commit, args.source_tree, rows)
    result = {
        "schema": "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_AUDIT_V1",
        "status": status,
        "source": {"commit": args.source_commit, "tree": args.source_tree},
        "suites": rows,
        "errors": errors,
        "root_seal": str(root_seal) if root_seal else None,
        "root_seal_sha256": root_seal_sha256,
        "scientific_authority": False,
        "protected_boundary": {"pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0, "attacked_env_steps": 0, "protected_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"},
        "next_gate": "STAGE_X1R2_Q3R3_ENGINEERING_MATRIX" if status.endswith("PASS") else "OWNER_REVIEW_Q3R3_BRANCH_REPLAY_HOLD",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "errors": errors, "root_seal_sha256": root_seal_sha256}, ensure_ascii=False, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
