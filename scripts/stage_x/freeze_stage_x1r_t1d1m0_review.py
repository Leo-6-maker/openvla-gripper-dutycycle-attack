#!/usr/bin/env python3
"""Freeze the D1M0 candidate set and blind order from sealed receipts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/STAGE_X_X1R_T1D1R_SCREENING_CLEAN_PROTOCOL_V1.json"
M0_PROTOCOL = REPO / "configs/STAGE_X_X1R_T1D1M0_MANUAL_CONTACT_VALIDITY_PROTOCOL_V1.json"
ROOT_SEAL = REPO / "reports/STAGE_X_X1R_T1D1R_ROOT_SEAL.json"
ROOT_CENSUS = REPO / "reports/STAGE_X_X1R_T1D1R_CENSUS_AUDIT_V1.json"
OUT_LEDGER = REPO / "reports/STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json"
OUT_MAPPING = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json"
OUT_FORM_CSV = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.csv"
OUT_FORM_JSON = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.json"
OUT_FREEZE = REPO / "reports/STAGE_X_X1R_T1D1M0_PREVIDEO_FREEZE_V1.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = load(PROTOCOL)
    m0 = load(M0_PROTOCOL)
    seal = load(ROOT_SEAL)
    census = load(ROOT_CENSUS)
    if seal.get("status") != "PASS_D1R_CONTINUATION_CENSUS_PRE_MANUAL_REVIEW" or census.get("status") != seal.get("status"):
        raise SystemExit("D1R_CENSUS_NOT_SEALED")
    if m0.get("status") != "FROZEN_BEFORE_VIDEO_PIXEL_ACCESS":
        raise SystemExit("D1M0_PROTOCOL_NOT_FROZEN")
    if protocol.get("student", {}).get("checkpoint_sha256") != census.get("student", {}).get("checkpoint_sha256"):
        raise SystemExit("D1R_STUDENT_BINDING_MISMATCH")

    parents = root / "parents"
    receipts = sorted(parents.glob("*/attempt_*/parent_receipt.json"))
    candidates: list[dict[str, Any]] = []
    for receipt_path in receipts:
        receipt = load(receipt_path)
        emit = receipt.get("first_emit_step")
        legal = emit is not None and int(emit) + 5 + 10 <= int(receipt.get("policy_horizon", -1))
        if not (
            receipt.get("status") == "PASS_SCREENING_CLEAN_EPISODE"
            and receipt.get("clean_success") is True
            and receipt.get("student_status") == "PASS_CAUSAL_TRACE"
            and emit is not None
            and legal
        ):
            continue
        manifest_path = receipt_path.parent / "episode_manifest.json"
        telemetry_path = receipt_path.parent / "step_telemetry.jsonl"
        video_path = Path(str(receipt["video"]["path"]))
        manifest = load(manifest_path)
        key = str(receipt["canonical_parent_key"])
        candidates.append({
            "ordinal": int(receipt["ordinal"]),
            "canonical_parent_key": key,
            "suite": str(receipt["suite"]),
            "task_idx": int(receipt["task_idx"]),
            "state_id": int(receipt["state_id"]),
            "expected_clean_seed": int(receipt["expected_clean_seed"]),
            "clean_success": True,
            "student_status": str(receipt["student_status"]),
            "first_emit_step": int(emit),
            "policy_horizon": int(receipt["policy_horizon"]),
            "legal_horizon": True,
            "policy_steps_executed": int(receipt["policy_steps_executed"]),
            "task_instruction": str(manifest.get("instruction", "")),
            "parent_receipt_path": str(receipt_path),
            "parent_receipt_sha256": sha(receipt_path),
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": sha(telemetry_path),
            "raw_clean_video_path": str(video_path),
            "raw_clean_video_sha256": sha(video_path),
            "raw_clean_video_bytes": int(receipt["video"]["bytes"]),
            "context_start": max(0, int(emit) - 10),
            "context_end": min(int(receipt["policy_steps_executed"]) - 1, int(emit) + 14),
        })
    if len(candidates) != int(m0["eligibility_recompute"]["required_count"]):
        raise SystemExit(f"STAGE_X_X1R_T1D1M0_HOLD_ELIGIBLE_LEDGER_MISMATCH:{len(candidates)}")
    candidates.sort(key=lambda row: (int(row["ordinal"]), row["canonical_parent_key"]))
    ledger = {
        "schema": "STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1",
        "status": "FROZEN_RECEIPT_DERIVED_BEFORE_VIDEO_PIXEL_ACCESS",
        "source": {"d1r_root": str(root), "d1r_census_sha256": sha(ROOT_CENSUS), "d1r_root_seal_sha256": sha(ROOT_SEAL), "candidate_formula": m0["eligibility_recompute"]["formula"]},
        "candidate_count": len(candidates),
        "rows": candidates,
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attack_authorized": False},
        "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED",
    }
    OUT_LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    salt = str(m0["blinding"]["order_salt"])
    ordered = []
    for row in candidates:
        rank_key = hashlib.sha256(f"{salt}|{row['canonical_parent_key']}".encode()).hexdigest()
        ordered.append({"rank_key": rank_key, **row})
    ordered.sort(key=lambda row: (row["rank_key"], row["canonical_parent_key"]))
    mapping_rows = []
    for index, row in enumerate(ordered, start=1):
        mapping_rows.append({"review_id": f"M{index:03d}", **row})
    mapping = {
        "schema": "STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1",
        "status": "FROZEN_BLINDED_ORDER_BEFORE_VIDEO_PIXEL_ACCESS",
        "order_salt": salt,
        "rank_formula": "SHA256(order_salt + '|' + canonical_parent_key)",
        "rows": mapping_rows,
        "human_sheet_fields": ["review_id", "task_instruction", "review_clip", "review_frame_strip", "contact_label", "reason_code", "reviewer", "review_timestamp", "optional_short_note"],
        "hidden_from_human_sheet": ["suite", "task_idx", "state_id", "ordinal", "canonical_parent_key", "rank_key", "Student probabilities", "physical scores", "attack/V_phys results"],
    }
    OUT_MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    form_rows = [{"review_id": row["review_id"], "contact_label": "", "reason_code": "", "reviewer": "", "review_timestamp": "", "optional_short_note": ""} for row in mapping_rows]
    fields = ["review_id", "contact_label", "reason_code", "reviewer", "review_timestamp", "optional_short_note"]
    with OUT_FORM_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(form_rows)
    OUT_FORM_JSON.write_text(json.dumps({"schema": "STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1", "status": "BLANK_OWNER_FORM", "rows": form_rows, "allowed_contact_labels": ["PASS", "FAIL", "ABSTAIN"], "allowed_reason_codes": m0["rubric"]["fail_reason_codes"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    order_digest = hashlib.sha256(json.dumps([{"review_id": row["review_id"], "rank_key": row["rank_key"], "canonical_parent_key": row["canonical_parent_key"]} for row in mapping_rows], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    freeze = {
        "schema": "STAGE_X_X1R_T1D1M0_PREVIDEO_FREEZE_V1",
        "status": "PASS_PREVIDEO_FREEZE",
        "video_pixels_opened": False,
        "candidate_count": len(candidates),
        "candidate_ledger_sha256": sha(OUT_LEDGER),
        "mapping_sha256": sha(OUT_MAPPING),
        "order_digest": order_digest,
        "rows": [{"review_id": row["review_id"], "ordinal": row["ordinal"], "canonical_parent_key": row["canonical_parent_key"], "first_emit_step": row["first_emit_step"], "context_start": row["context_start"], "context_end": row["context_end"], "raw_clean_video_sha256": row["raw_clean_video_sha256"]} for row in mapping_rows],
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attack_authorized": False},
        "next_phase": "FIXED_REVIEW_COPY_RENDER_ONLY",
    }
    OUT_FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": freeze["status"], "candidate_count": len(candidates), "candidate_ledger_sha256": freeze["candidate_ledger_sha256"], "order_digest": order_digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
