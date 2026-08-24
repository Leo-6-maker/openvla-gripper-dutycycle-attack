#!/usr/bin/env python3
"""Audit the frozen D1M0 manual-review packet without assigning labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
M0 = REPO / "configs/STAGE_X_X1R_T1D1M0_MANUAL_CONTACT_VALIDITY_PROTOCOL_V1.json"
LEDGER = REPO / "reports/STAGE_X_X1R_T1D1M0_PREMANUAL_ELIGIBLE_LEDGER_V1.json"
MAPPING = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json"
FORM_CSV = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.csv"
FORM_JSON = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_FORM_V1.json"
FREEZE = REPO / "reports/STAGE_X_X1R_T1D1M0_PREVIDEO_FREEZE_V1.json"
RENDER = REPO / "reports/STAGE_X_X1R_T1D1M0_REVIEW_RENDER_MANIFEST_V1.json"
D1R_CENSUS = REPO / "reports/STAGE_X_X1R_T1D1R_CENSUS_AUDIT_V1.json"
D1R_SEAL = REPO / "reports/STAGE_X_X1R_T1D1R_ROOT_SEAL.json"
EXCLUDED = {1, 11, 20, 30}


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


def frame_count(path: Path) -> int:
    out = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
    ], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    raise RuntimeError(f"FRAME_COUNT_UNAVAILABLE:{path}")


def fail(errors: list[str]) -> None:
    if errors:
        raise SystemExit("STAGE_X_X1R_T1D1M0_HOLD_PACKET_AUDIT\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "reports")
    args = parser.parse_args()
    errors: list[str] = []
    source = {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "branch": git("branch", "--show-current"), "status_porcelain": git("status", "--porcelain")}
    root = args.root.resolve()
    m0, ledger, mapping, freeze, render, d1r_census, d1r_seal = (load(path) for path in (M0, LEDGER, MAPPING, FREEZE, RENDER, D1R_CENSUS, D1R_SEAL))
    if source["status_porcelain"]:
        errors.append("WORKTREE_NOT_CLEAN_BEFORE_PACKET_AUDIT")
    if m0.get("status") != "FROZEN_BEFORE_VIDEO_PIXEL_ACCESS":
        errors.append("M0_PROTOCOL_NOT_FROZEN")
    if ledger.get("status") != "FROZEN_RECEIPT_DERIVED_BEFORE_VIDEO_PIXEL_ACCESS" or ledger.get("candidate_count") != 14:
        errors.append("CANDIDATE_LEDGER_NOT_FROZEN_14")
    if mapping.get("status") != "FROZEN_BLINDED_ORDER_BEFORE_VIDEO_PIXEL_ACCESS" or len(mapping.get("rows", [])) != 14:
        errors.append("MAPPING_NOT_FROZEN_14")
    if freeze.get("status") != "PASS_PREVIDEO_FREEZE" or freeze.get("video_pixels_opened") is not False:
        errors.append("PREVIDEO_FREEZE_INVALID")
    if render.get("status") != "PASS_FIXED_REVIEW_COPIES_RENDERED" or render.get("candidate_count") != 14:
        errors.append("RENDER_MANIFEST_INVALID")
    if d1r_census.get("status") != "PASS_D1R_CONTINUATION_CENSUS_PRE_MANUAL_REVIEW" or d1r_seal.get("status") != d1r_census.get("status"):
        errors.append("D1R_SEAL_INVALID")

    ledger_by_key = {row["canonical_parent_key"]: row for row in ledger.get("rows", [])}
    mapping_rows = mapping.get("rows", [])
    if [row.get("review_id") for row in mapping_rows] != [f"M{i:03d}" for i in range(1, 15)]:
        errors.append("REVIEW_ID_SEQUENCE_INVALID")
    salt = str(m0["blinding"]["order_salt"])
    expected_order = sorted(ledger_by_key, key=lambda key: (hashlib.sha256(f"{salt}|{key}".encode()).hexdigest(), key))
    if [row.get("canonical_parent_key") for row in mapping_rows] != expected_order:
        errors.append("BLINDED_ORDER_INVALID")
    if freeze.get("candidate_count") != 14 or freeze.get("order_digest") != hashlib.sha256(json.dumps([{"review_id": row["review_id"], "rank_key": row["rank_key"], "canonical_parent_key": row["canonical_parent_key"]} for row in mapping_rows], sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        errors.append("PREVIDEO_ORDER_DIGEST_INVALID")

    render_by_id = {row["review_id"]: row for row in render.get("rows", [])}
    if set(render_by_id) != {f"M{i:03d}" for i in range(1, 15)}:
        errors.append("RENDER_ID_SET_INVALID")
    artifact_rows: list[dict[str, Any]] = []
    parent_keys: set[str] = set()
    for row in mapping_rows:
        review_id = str(row["review_id"])
        key = str(row["canonical_parent_key"])
        parent_keys.add(key)
        if int(row["ordinal"]) in EXCLUDED:
            errors.append(f"EXCLUDED_D1_ORDINAL_PRESENT:{review_id}")
        if row.get("clean_success") is not True or row.get("student_status") != "PASS_CAUSAL_TRACE" or row.get("legal_horizon") is not True:
            errors.append(f"CANDIDATE_GATE_INVALID:{review_id}")
        receipt = Path(str(row["parent_receipt_path"]))
        telemetry = Path(str(row["telemetry_path"]))
        raw = Path(str(row["raw_clean_video_path"]))
        for label, path, expected_sha in (("receipt", receipt, row["parent_receipt_sha256"]), ("telemetry", telemetry, row["telemetry_sha256"]), ("raw_video", raw, row["raw_clean_video_sha256"])):
            if not path.is_file():
                errors.append(f"{label.upper()}_MISSING:{review_id}")
            elif sha(path) != expected_sha:
                errors.append(f"{label.upper()}_SHA_MISMATCH:{review_id}")
            else:
                artifact_rows.append({"kind": label, "path": str(path), "sha256": expected_sha})
        receipt_data = load(receipt)
        if receipt_data.get("canonical_parent_key") != key or receipt_data.get("status") != "PASS_SCREENING_CLEAN_EPISODE":
            errors.append(f"RECEIPT_BINDING_INVALID:{review_id}")
        if receipt_data.get("first_emit_step") != row.get("first_emit_step") or receipt_data.get("policy_steps_executed") != row.get("policy_steps_executed"):
            errors.append(f"RECEIPT_TIMING_BINDING_INVALID:{review_id}")
        video = render_by_id.get(review_id, {})
        if video.get("raw_clean_video_sha256") != row.get("raw_clean_video_sha256") or video.get("context_start") != row.get("context_start") or video.get("context_end") != row.get("context_end") or video.get("emit_step") != row.get("first_emit_step"):
            errors.append(f"RENDER_BINDING_INVALID:{review_id}")
        clip = Path(str(video.get("review_clip_path", "")))
        strip = Path(str(video.get("review_frame_strip_path", "")))
        if not clip.is_file() or not strip.is_file():
            errors.append(f"REVIEW_COPY_MISSING:{review_id}")
        else:
            if sha(clip) != video.get("review_clip_sha256") or sha(strip) != video.get("review_frame_strip_sha256"):
                errors.append(f"REVIEW_COPY_SHA_MISMATCH:{review_id}")
            if frame_count(raw) != int(row["policy_steps_executed"]):
                errors.append(f"RAW_FRAME_MAPPING_INVALID:{review_id}")
            if frame_count(clip) != int(row["context_end"]) - int(row["context_start"]) + 1:
                errors.append(f"CLIP_FRAME_MAPPING_INVALID:{review_id}")
            artifact_rows.extend([
                {"kind": "review_clip", "path": str(clip), "sha256": video["review_clip_sha256"]},
                {"kind": "review_frame_strip", "path": str(strip), "sha256": video["review_frame_strip_sha256"]},
            ])
    if len(parent_keys) != 14 or any(int(row["ordinal"]) in EXCLUDED for row in mapping_rows):
        errors.append("CANDIDATE_IDENTITY_CLOSURE_INVALID")
    packet_root = Path(str(render["review_root"]))
    packet_dirs = sorted(path.name for path in packet_root.iterdir() if path.is_dir()) if packet_root.is_dir() else []
    if packet_dirs != [f"M{i:03d}" for i in range(1, 15)]:
        errors.append("PACKET_DIRECTORY_SET_INVALID")
    for directory in packet_root.iterdir() if packet_root.is_dir() else []:
        if directory.is_dir() and sorted(path.name for path in directory.iterdir()) != ["review_clip.mp4", "review_frame_strip.png"]:
            errors.append(f"PACKET_DIRECTORY_CONTENT_INVALID:{directory.name}")

    with FORM_CSV.open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    if len(csv_rows) != 14 or any(row.get(field, "") for row in csv_rows for field in ("contact_label", "reason_code", "reviewer", "review_timestamp", "optional_short_note")):
        errors.append("CSV_FORM_NOT_BLANK")
    form_json = load(FORM_JSON)
    if form_json.get("status") != "BLANK_OWNER_FORM" or any(any(row.get(field, "") for field in ("contact_label", "reason_code", "reviewer", "review_timestamp", "optional_short_note")) for row in form_json.get("rows", [])):
        errors.append("JSON_FORM_NOT_BLANK")
    human_text = FORM_CSV.read_text(encoding="utf-8") + FORM_JSON.read_text(encoding="utf-8")
    if any(token in human_text for token in ("student_probabilities", "physical_criticality", "gripper_closing_state", "V_phys")):
        errors.append("HUMAN_FORM_LEAKS_HIDDEN_FIELDS")
    if any(int(row["ordinal"]) in EXCLUDED for row in ledger.get("rows", []) if row.get("canonical_parent_key") in parent_keys):
        errors.append("D1_INVALID_ORDINAL_LEAK")
    fail(errors)

    artifact_rows.extend({"kind": "repo", "path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(path)} for path in (M0, LEDGER, MAPPING, FORM_CSV, FORM_JSON, FREEZE, RENDER, D1R_CENSUS, D1R_SEAL))
    artifact_rows.sort(key=lambda row: (row["kind"], row["path"]))
    counts = {"candidate_count": 14, "all_clean_success": True, "all_student_trace_pass": True, "all_first_emit_legal": True, "labels_present": False, "d1_excluded_ordinals_present": False}
    audit = {
        "schema": "STAGE_X_X1R_T1D1M0_REVIEW_PACKET_AUDIT_V1",
        "status": "STAGE_X_X1R_T1D1M0_REVIEW_PACKET_PASS",
        "scope": "pre-registered manual contact-validity packet only",
        "source_before_evidence_outputs": source,
        "ancestry": {"pr130_head": "f4da1c6683860757cc9775b573a158dd89505b15", "pr130_tree": "2d1b1cf94c0cd5322431eb0a864f1e00afd087f1", "d1r_runtime_commit": "d74b8b7aff311c4ebbd51bf83ff026efe48d0236", "d1r_runtime_tree": "2ee7425fc9177d70abb61f12b644833ec20d0a06"},
        "d1r_census_sha256": sha(D1R_CENSUS),
        "d1r_root_seal_sha256": sha(D1R_SEAL),
        "candidate_ledger_sha256": sha(LEDGER),
        "mapping_sha256": sha(MAPPING),
        "order_digest": freeze["order_digest"],
        "review_root": str(packet_root),
        "counts": counts,
        "packet_rules": {"fixed_window": "[t_emit-10,t_emit+14] clipped to recorded policy frames", "emit_marker": "T_EMIT", "raw_videos_unchanged": True, "human_form_blank": True, "labels_must_be_owner_supplied": True},
        "protected_boundary": {"eval160": "UNREAD", "protected_evaluation": "UNREAD", "attack_authorized": False, "model_inference": 0, "student_inference": 0, "env_steps": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "attack_outcome_reads": 0},
        "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED",
        "artifact_rows": artifact_rows,
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "STAGE_X_X1R_T1D1M0_REVIEW_PACKET_AUDIT_V1.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_entries = [{"kind": "repo", "path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(path)} for path in (M0, LEDGER, MAPPING, FORM_CSV, FORM_JSON, FREEZE, RENDER, D1R_CENSUS, D1R_SEAL, audit_path)]
    sums_rows = sorted(artifact_rows + repo_entries, key=lambda row: (row["kind"], row["path"]))
    sums_path = out / "STAGE_X_X1R_T1D1M0_SHA256SUMS.txt"
    sums_path.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in sums_rows), encoding="utf-8")
    sums_sha = sha(sums_path)
    (out / "STAGE_X_X1R_T1D1M0_SHA256SUMS.sha256").write_text(f"{sums_sha}  STAGE_X_X1R_T1D1M0_SHA256SUMS.txt\n", encoding="utf-8")
    seal = {
        "schema": "STAGE_X_X1R_T1D1M0_ROOT_SEAL_V1",
        "status": "STAGE_X_X1R_T1D1M0_REVIEW_PACKET_PASS",
        "scope": "pre-registered manual contact-validity review packet; no labels",
        "source_before_evidence_outputs": source,
        "ancestry": audit["ancestry"],
        "candidate_ledger_sha256": sha(LEDGER),
        "mapping_sha256": sha(MAPPING),
        "order_digest": freeze["order_digest"],
        "packet_audit": {"path": str(audit_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(audit_path)},
        "sha256sums": {"path": str(sums_path.relative_to(REPO)).replace("\\", "/"), "sha256": sums_sha},
        "review_root": str(packet_root),
        "counts": counts,
        "protected_boundary": audit["protected_boundary"],
        "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED",
        "seal_note": "No manual label was inferred or entered; this seal freezes only the review package and blank form.",
    }
    seal_path = out / "STAGE_X_X1R_T1D1M0_ROOT_SEAL.json"
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_sha = sha(seal_path)
    (out / "STAGE_X_X1R_T1D1M0_ROOT_SEAL.sha256").write_text(f"{seal_sha}  STAGE_X_X1R_T1D1M0_ROOT_SEAL.json\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "candidate_count": 14, "candidate_ledger_sha256": sha(LEDGER), "order_digest": freeze["order_digest"], "packet_audit_sha256": sha(audit_path), "sha256sums_sha256": sums_sha, "root_seal_sha256": seal_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
