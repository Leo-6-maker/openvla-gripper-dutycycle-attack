#!/usr/bin/env python3
"""Audit the D1M0R safe human-facing projection without touching videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MAPPING = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json"
RENDER = REPO / "reports/STAGE_X_X1R_T1D1M0_REVIEW_RENDER_MANIFEST_V1.json"
D1M0_SUMS = REPO / "reports/STAGE_X_X1R_T1D1M0_SHA256SUMS.txt"
D1M0_SEAL = REPO / "reports/STAGE_X_X1R_T1D1M0_ROOT_SEAL.json"
D1M0_SEAL_SIDE = REPO / "reports/STAGE_X_X1R_T1D1M0_ROOT_SEAL.sha256"
SAFE_CSV = REPO / "reports/STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1.csv"
SAFE_JSON = REPO / "reports/STAGE_X_X1R_T1D1M0R_HUMAN_REVIEW_SHEET_V1.json"
INSTRUCTIONS = REPO / "docs/handoffs/STAGE_X_X1R_T1D1M0R_OWNER_REVIEW_INSTRUCTIONS_20260818.md"
OUT_AUDIT = REPO / "reports/STAGE_X_X1R_T1D1M0R_HUMAN_SHEET_AUDIT_V1.json"
EXPECTED_BASE = "f4da1c6683860757cc9775b573a158dd89505b15"
EXPECTED_PR131_HEAD = "6dbf1ee287660a5f70ee90a0680e7c859fa155a8"
EXPECTED_TREE = "e79c1aa77f3eab3a45e1c7e513e1239095c4c69c"
EXPECTED_LEDGER_SHA = "5f1f036b47b1c9a8c1bafe7a400b6be9269cd3e67587691018005c824dc8d89e"
EXPECTED_MAPPING_SHA = "3d7f59a736cc2c7bcb5ecdc49e9e57a7e8b547c9e7554251e88158017366f0fe"
EXPECTED_ORDER_DIGEST = "30a73b0e4ab13e149d8c991906fc9067844797e39113201e9e76a10a8be40d67"
FIELDS = [
    "review_id", "task_instruction", "review_clip_path", "review_clip_sha256",
    "review_frame_strip_path", "review_frame_strip_sha256", "contact_label",
    "reason_code", "reviewer", "review_timestamp", "optional_short_note",
]
FORBIDDEN_KEYS = {
    "suite", "task_idx", "state_id", "ordinal", "canonical_parent_key", "rank_key",
    "expected_clean_seed", "first_emit_step", "policy_horizon", "student_probabilities",
    "physical_criticality", "gripper_closing_state", "selection_rank", "parent_receipt_path",
    "telemetry_path", "vphys", "attack_outcome",
}
OLD_ARTIFACTS = [
    "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.csv",
    "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.json",
    "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json",
    "reports/STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json",
    "reports/STAGE_X_X1R_T1D1M0_PREVIDEO_FREEZE_V1.json",
    "configs/STAGE_X_X1R_T1D1M0_MANUAL_CONTACT_VALIDITY_PROTOCOL_V1.json",
    "docs/handoffs/STAGE_X_X1R_T1D1M0_MANUAL_CONTACT_RUBRIC_20260818.md",
    "reports/STAGE_X_X1R_T1D1M0_REVIEW_RENDER_MANIFEST_V1.json",
    "reports/STAGE_X_X1R_T1D1M0_REVIEW_PACKET_AUDIT_V1.json",
    "reports/STAGE_X_X1R_T1D1M0_ROOT_SEAL.json",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def pr131_blob_sha(relative: str) -> str:
    content = subprocess.run(["git", "show", f"{EXPECTED_PR131_HEAD}:{relative}"], cwd=REPO, check=True, capture_output=True).stdout
    return hashlib.sha256(content).hexdigest()


def fail(errors: list[str]) -> None:
    if errors:
        raise SystemExit("STAGE_X_X1R_T1D1M0R_HOLD_HUMAN_SHEET_BLINDING\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / "reports")
    args = parser.parse_args()
    errors: list[str] = []
    source = {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "branch": git("branch", "--show-current"), "status_porcelain": git("status", "--porcelain")}
    if source["status_porcelain"]:
        errors.append("WORKTREE_NOT_CLEAN_BEFORE_D1M0R_AUDIT")
    if not subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD"], cwd=REPO).returncode == 0:
        errors.append("ANCESTRY_NOT_FROM_PR131_HEAD")
    if git("rev-parse", "HEAD~0") == EXPECTED_BASE:
        errors.append("D1M0R_NO_NEW_COMMIT")

    sums = {}
    for line in D1M0_SUMS.read_text(encoding="utf-8").splitlines():
        digest, _, path = line.partition("  ")
        if digest and path:
            sums[path] = digest
    if sums.get("reports/STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json") != EXPECTED_LEDGER_SHA or sums.get("reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json") != EXPECTED_MAPPING_SHA:
        errors.append("D1M0_SHA_BINDING_INVALID")
    if sha(MAPPING) != EXPECTED_MAPPING_SHA:
        errors.append("D1M0_MAPPING_CHANGED")
    mapping, render, safe = load(MAPPING), load(RENDER), load(SAFE_JSON)
    mapping_rows = mapping.get("rows", [])
    render_rows = {row["review_id"]: row for row in render.get("rows", [])}
    if [row.get("review_id") for row in mapping_rows] != [f"M{i:03d}" for i in range(1, 15)]:
        errors.append("PRIVATE_MAPPING_ORDER_CHANGED")
    order_digest = hashlib.sha256(json.dumps([{"review_id": row["review_id"], "rank_key": row["rank_key"], "canonical_parent_key": row["canonical_parent_key"]} for row in mapping_rows], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if order_digest != EXPECTED_ORDER_DIGEST:
        errors.append("D1M0_ORDER_DIGEST_CHANGED")

    with SAFE_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        sheet_rows = list(reader)
        sheet_fields = list(reader.fieldnames or [])
    if sheet_fields != FIELDS or safe.get("fields") != FIELDS:
        errors.append("SAFE_SHEET_FIELDS_INVALID")
    if [row.get("review_id") for row in sheet_rows] != [f"M{i:03d}" for i in range(1, 15)] or [row.get("review_id") for row in safe.get("rows", [])] != [f"M{i:03d}" for i in range(1, 15)]:
        errors.append("SAFE_SHEET_ID_ORDER_INVALID")
    if len(sheet_rows) != 14 or len(safe.get("rows", [])) != 14:
        errors.append("SAFE_SHEET_COUNT_INVALID")
    safe_rows = {row["review_id"]: row for row in sheet_rows}
    safe_json_rows = {row["review_id"]: row for row in safe.get("rows", [])}
    private_rows = {row["review_id"]: row for row in mapping_rows}
    for review_id in [f"M{i:03d}" for i in range(1, 15)]:
        expected = private_rows[review_id]
        render_row = render_rows[review_id]
        for row in (safe_rows[review_id], safe_json_rows[review_id]):
            if row["task_instruction"] != expected["task_instruction"]:
                errors.append(f"TASK_INSTRUCTION_MISMATCH:{review_id}")
            for field in ("review_clip_path", "review_clip_sha256", "review_frame_strip_path", "review_frame_strip_sha256"):
                if row[field] != render_row[field]:
                    errors.append(f"REVIEW_COPY_PROJECTION_MISMATCH:{review_id}:{field}")
            if any(row.get(field, "") for field in ("contact_label", "reason_code", "reviewer", "review_timestamp", "optional_short_note")):
                errors.append(f"OWNER_LABEL_FIELD_NOT_BLANK:{review_id}")
        for path_field, sha_field in (("review_clip_path", "review_clip_sha256"), ("review_frame_strip_path", "review_frame_strip_sha256")):
            review_path = Path(str(render_row[path_field]))
            if not review_path.is_file():
                errors.append(f"REVIEW_COPY_MISSING:{review_id}:{path_field}")
            elif sha(review_path) != render_row[sha_field]:
                errors.append(f"REVIEW_COPY_SHA_MISMATCH:{review_id}:{path_field}")
    serialized = json.dumps({"fields": sheet_fields, "rows": sheet_rows}, ensure_ascii=False)
    if any(key in sheet_fields for key in FORBIDDEN_KEYS) or any(key in serialized for key in FORBIDDEN_KEYS):
        errors.append("FORBIDDEN_KEY_EXPOSED_IN_SAFE_SHEET")
    for private_row in mapping_rows:
        hidden_values = (private_row.get("canonical_parent_key", ""), private_row.get("parent_receipt_path", ""), private_row.get("telemetry_path", ""), private_row.get("rank_key", ""))
        if any(value and value in serialized for value in hidden_values):
            errors.append("PRIVATE_VALUE_EXPOSED_IN_SAFE_SHEET")
    if any(token in INSTRUCTIONS.read_text(encoding="utf-8") for token in ("canonical_parent_key", "first_emit_step", "physical_criticality", "gripper_closing_state", "Student probabilities")):
        errors.append("FORBIDDEN_VALUE_OR_FIELD_IN_OWNER_INSTRUCTIONS")
    if safe.get("status") != "BLANK_SAFE_OWNER_SHEET" or safe.get("next_gate") != "OWNER_MANUAL_CONTACT_LABELS_REQUIRED":
        errors.append("SAFE_JSON_STATUS_INVALID")

    old_seal = load(D1M0_SEAL)
    if sha(D1M0_SUMS) != old_seal.get("sha256sums", {}).get("sha256"):
        errors.append("D1M0_SUMS_CHANGED")
    sidecar_text = D1M0_SEAL_SIDE.read_text(encoding="utf-8").split()[0]
    if sha(D1M0_SEAL) != sidecar_text:
        errors.append("D1M0_ROOT_SEAL_SIDEcar_MISMATCH")
    for relative in OLD_ARTIFACTS:
        path = REPO / relative
        if not path.is_file():
            errors.append(f"D1M0_ARTIFACT_CHANGED:{relative}")
        elif sha(path) != pr131_blob_sha(relative):
            errors.append(f"D1M0_ARTIFACT_CHANGED:{relative}")
    fail(errors)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_rows = [{"kind": "repo", "path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(path)} for path in (MAPPING, RENDER, D1M0_SUMS, D1M0_SEAL, SAFE_CSV, SAFE_JSON, INSTRUCTIONS)]
    audit = {
        "schema": "STAGE_X_X1R_T1D1M0R_HUMAN_SHEET_AUDIT_V1",
        "status": "STAGE_X_X1R_T1D1M0R_HUMAN_SHEET_PASS",
        "source_before_evidence_outputs": source,
        "ancestry": {"pr131_head": "6dbf1ee287660a5f70ee90a0680e7c859fa155a8", "pr131_tree": EXPECTED_TREE, "pr130_head": EXPECTED_BASE},
        "d1m0_bindings": {"candidate_ledger_sha256": EXPECTED_LEDGER_SHA, "mapping_sha256": EXPECTED_MAPPING_SHA, "review_order_digest": EXPECTED_ORDER_DIGEST, "d1m0_artifacts_byte_identical": True},
        "projection": {"candidate_count": 14, "order_unchanged": True, "task_instruction_source": "frozen private mapping", "review_copy_source": "frozen render manifest", "private_mapping_not_copied": True, "owner_labels_present": False, "clips_rerendered": False, "raw_videos_changed": False},
        "human_fields": FIELDS,
        "protected_boundary": {"attack_authorized": False, "eval160": "UNREAD", "protected_evaluation": "UNREAD", "model_inference": 0, "student_inference": 0, "env_steps": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0},
        "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED",
        "artifact_rows": artifact_rows,
    }
    audit_path = output_dir / OUT_AUDIT.name
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sum_rows = sorted(artifact_rows + [{"kind": "repo", "path": str(audit_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(audit_path)}], key=lambda row: (row["kind"], row["path"]))
    sums_path = output_dir / "STAGE_X_X1R_T1D1M0R_SHA256SUMS.txt"
    sums_path.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in sum_rows), encoding="utf-8")
    sums_sha = sha(sums_path)
    (output_dir / "STAGE_X_X1R_T1D1M0R_SHA256SUMS.sha256").write_text(f"{sums_sha}  STAGE_X_X1R_T1D1M0R_SHA256SUMS.txt\n", encoding="utf-8")
    seal = {
        "schema": "STAGE_X_X1R_T1D1M0R_ROOT_SEAL_V1",
        "status": "STAGE_X_X1R_T1D1M0R_HUMAN_SHEET_PASS",
        "source_before_evidence_outputs": source,
        "ancestry": audit["ancestry"],
        "candidate_set_unchanged": True,
        "review_order_unchanged": True,
        "rubric_unchanged": True,
        "raw_review_videos_unchanged": True,
        "owner_labels_present": False,
        "private_mapping_not_copied_to_human_sheet": True,
        "human_sheet": {"csv": str(SAFE_CSV.relative_to(REPO)).replace("\\", "/"), "csv_sha256": sha(SAFE_CSV), "json": str(SAFE_JSON.relative_to(REPO)).replace("\\", "/"), "json_sha256": sha(SAFE_JSON), "task_instruction_projected": True},
        "audit": {"path": str(audit_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(audit_path)},
        "sha256sums": {"path": str(sums_path.relative_to(REPO)).replace("\\", "/"), "sha256": sums_sha},
        "protected_boundary": audit["protected_boundary"],
        "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED",
    }
    seal_path = output_dir / "STAGE_X_X1R_T1D1M0R_ROOT_SEAL.json"
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_sha = sha(seal_path)
    (output_dir / "STAGE_X_X1R_T1D1M0R_ROOT_SEAL.sha256").write_text(f"{seal_sha}  STAGE_X_X1R_T1D1M0R_ROOT_SEAL.json\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "safe_csv_sha256": sha(SAFE_CSV), "safe_json_sha256": sha(SAFE_JSON), "audit_sha256": sha(audit_path), "sha256sums_sha256": sums_sha, "root_seal_sha256": seal_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
