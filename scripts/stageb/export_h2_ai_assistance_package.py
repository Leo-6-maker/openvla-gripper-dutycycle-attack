#!/usr/bin/env python3
"""Export the H2 v2.1 AI-assistance media package.

This is a packaging and validation utility only. It does not populate official
Reviewer A/B fields and does not inspect detector, attack, VIS/RAND, or task
success artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


EXPECTED_REVIEW_ROUND_ID = "h2_diagnostic_review_round_v2_1_20260621"
EXPECTED_REVIEWER_A_SHA256 = (
    "0340c6aa37bcbe9a92239f3c40c8cd7dbcb4a31e8dbf0552284295a3fbeac3df"
)
EXPECTED_MEDIA_MANIFEST_SHA256 = (
    "01d951d5fc6523a15137f84fcfe6a8cade58bcfdf1c86d75397f1b52713fb3ba"
)
OFFICIAL_HUMAN_FIELDS = {
    "reviewer_id",
    "review_timestamp",
    "object_identity_valid",
    "target_identity_valid",
    "event_exists",
    "close_onset_valid",
    "grasp_established_valid",
    "lift_onset_valid",
    "stable_carry_valid",
    "window_start_valid",
    "anchor_valid",
    "window_end_valid",
    "release_separation_valid",
    "false_positive_carry",
    "abstain_or_fail_closed_correct",
    "corrected_object_body",
    "corrected_target_body_or_site",
    "corrected_close_onset",
    "corrected_grasp_established",
    "corrected_lift_onset",
    "corrected_stable_carry_start",
    "corrected_window_start",
    "corrected_anchor",
    "corrected_window_end",
    "corrected_release_onset",
    "disagreement_reason",
    "reviewer_notes",
}
FORBIDDEN_EXPORT_SUBSTRINGS = (
    "detector",
    "attack",
    "vis",
    "rand",
    "shuffled",
    "task_success",
    "success",
)
ASSISTANT_COLUMNS = [
    "review_round_id",
    "review_id",
    "assistant_suggested_object_identity_valid",
    "assistant_suggested_target_identity_valid",
    "assistant_suggested_event_exists",
    "assistant_suggested_close_onset_valid",
    "assistant_suggested_grasp_established_valid",
    "assistant_suggested_lift_onset_valid",
    "assistant_suggested_stable_carry_valid",
    "assistant_suggested_window_start_valid",
    "assistant_suggested_anchor_valid",
    "assistant_suggested_window_end_valid",
    "assistant_suggested_release_separation_valid",
    "assistant_suggested_false_positive_carry",
    "assistant_suggested_abstain_or_fail_closed_correct",
    "assistant_confidence",
    "assistant_visual_evidence",
    "assistant_timing_evidence",
    "assistant_uncertainty_reason",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def count_nonempty_human_fields(rows: List[Dict[str, str]]) -> int:
    count = 0
    for row in rows:
        for field in OFFICIAL_HUMAN_FIELDS:
            if row.get(field, "").strip():
                count += 1
    return count


def rel_for_artifact(artifact_type: str, review_id: str, src: Path) -> Path:
    suffix = src.suffix
    if artifact_type == "raw_video":
        return Path("raw_videos") / f"{review_id}_raw{suffix}"
    if artifact_type == "teacher_overlay_mp4":
        return Path("teacher_overlays") / f"{review_id}_teacher_overlay{suffix}"
    if artifact_type == "teacher_timeline_csv":
        return Path("teacher_timelines") / f"{review_id}_teacher_timeline{suffix}"
    raise ValueError(f"unexpected artifact_type={artifact_type!r}")


def copy_verified_artifacts(
    media_rows: List[Dict[str, str]],
    package_root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    copied: List[Dict[str, str]] = []
    counts = {"raw_video": 0, "teacher_overlay_mp4": 0, "teacher_timeline_csv": 0}
    for row in media_rows:
        artifact_type = row["artifact_type"]
        if artifact_type not in counts:
            raise RuntimeError(f"Forbidden or unexpected artifact type: {artifact_type}")
        src = Path(row["path"])
        if not src.is_file():
            raise RuntimeError(f"Missing media artifact: {src}")
        expected_size = int(row["size_bytes"])
        actual_size = src.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {src}: {actual_size} != {expected_size}")
        actual_sha = sha256_file(src)
        if actual_sha != row["sha256"]:
            raise RuntimeError(f"SHA mismatch for {src}: {actual_sha} != {row['sha256']}")
        dst_rel = rel_for_artifact(artifact_type, row["review_id"], src)
        dst = package_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_row = dict(row)
        copied_row["package_relpath"] = dst_rel.as_posix()
        copied.append(copied_row)
        counts[artifact_type] += 1
    return copied, counts


def write_sha256s(package_root: Path) -> Tuple[Path, List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(p for p in package_root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        rel = path.relative_to(package_root).as_posix()
        rows.append({"sha256": sha256_file(path), "path": rel, "size_bytes": str(path.stat().st_size)})
    sums_path = package_root / "SHA256SUMS.txt"
    with sums_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(f"{row['sha256']}  {row['path']}\n")
    rows.append(
        {
            "sha256": sha256_file(sums_path),
            "path": "SHA256SUMS.txt",
            "size_bytes": str(sums_path.stat().st_size),
        }
    )
    return sums_path, rows


def zip_package(package_root: Path, zip_path: Path) -> str:
    if zip_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in package_root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(package_root).as_posix())
    return sha256_file(zip_path)


def make_read_only(path: Path) -> None:
    for p in sorted(path.rglob("*"), reverse=True):
        try:
            mode = p.stat().st_mode
            if p.is_dir():
                p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
            else:
                p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except PermissionError:
            pass
    try:
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except PermissionError:
        pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer-a-source", required=True, type=Path)
    ap.add_argument("--media-manifest", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--receipt-out", required=True, type=Path)
    ap.add_argument("--assistant-notes-out", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="Remove an existing output root before writing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    reviewer_a_source = args.reviewer_a_source
    media_manifest = args.media_manifest
    output_root = args.output_root

    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)

    reviewer_a_sha = sha256_file(reviewer_a_source)
    media_manifest_sha = sha256_file(media_manifest)
    if reviewer_a_sha != EXPECTED_REVIEWER_A_SHA256:
        raise RuntimeError(f"Reviewer A SHA mismatch: {reviewer_a_sha}")
    if media_manifest_sha != EXPECTED_MEDIA_MANIFEST_SHA256:
        raise RuntimeError(f"Media manifest SHA mismatch: {media_manifest_sha}")

    review_rows = read_csv(reviewer_a_source)
    if len(review_rows) != 36:
        raise RuntimeError(f"Expected 36 Reviewer A rows, found {len(review_rows)}")
    if {r.get("review_round_id") for r in review_rows} != {EXPECTED_REVIEW_ROUND_ID}:
        raise RuntimeError("Reviewer A rows contain unexpected review_round_id")
    human_nonempty = count_nonempty_human_fields(review_rows)
    if human_nonempty != 0:
        raise RuntimeError(f"Reviewer source already has nonempty human fields: {human_nonempty}")

    media_rows = read_csv(media_manifest)
    if len(media_rows) != 108:
        raise RuntimeError(f"Expected 108 media rows, found {len(media_rows)}")
    review_ids = {r["review_id"] for r in review_rows}
    media_ids = {r["review_id"] for r in media_rows}
    if media_ids != review_ids:
        raise RuntimeError("Media manifest review IDs do not match Reviewer A rows")

    forbidden_field_count = 0
    for row in review_rows:
        for key in row:
            lowered = key.lower()
            if any(s in lowered for s in FORBIDDEN_EXPORT_SUBSTRINGS):
                forbidden_field_count += 1
    # The v2.1 review queue intentionally includes no detector/attack/task-success fields.
    if forbidden_field_count != 0:
        raise RuntimeError(f"Forbidden field count in reviewer source: {forbidden_field_count}")

    shutil.copy2(reviewer_a_source, output_root / "reviewer_A_all36_blank_v2_1.csv")
    copied_media, media_counts = copy_verified_artifacts(media_rows, output_root)
    write_csv(
        output_root / "media_manifest_all36.csv",
        list(copied_media[0].keys()),
        copied_media,
    )
    metadata_root = output_root / "metadata"
    metadata_root.mkdir()
    shutil.copy2(media_manifest, metadata_root / "frozen_media_manifest_source.csv")

    assistant_rows = [
        {"review_round_id": EXPECTED_REVIEW_ROUND_ID, "review_id": row["review_id"]}
        for row in review_rows
    ]
    write_csv(args.assistant_notes_out, ASSISTANT_COLUMNS, assistant_rows)
    shutil.copy2(args.assistant_notes_out, metadata_root / "assistant_pre_review_notes_v2_1_blank.csv")

    package_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_round_id": EXPECTED_REVIEW_ROUND_ID,
        "reviewer_a_source": str(reviewer_a_source),
        "reviewer_a_source_sha256": reviewer_a_sha,
        "media_manifest_source": str(media_manifest),
        "media_manifest_sha256": media_manifest_sha,
        "row_count": len(review_rows),
        "raw_video_count": media_counts["raw_video"],
        "overlay_video_count": media_counts["teacher_overlay_mp4"],
        "timeline_count": media_counts["teacher_timeline_csv"],
        "human_fields_nonempty_count": human_nonempty,
        "forbidden_field_count": forbidden_field_count,
        "official_review_completion": "NOT_RUN",
        "assistant_notes_status": "NONOFFICIAL_BLANK_SCHEMA",
        "forbidden_uses": [
            "assistant_as_reviewer",
            "assistant_as_adjudicator",
            "assistant_counts_toward_completed_review_rows",
            "final_blind_selection",
            "full_clean300_resolver",
            "layer2_real_train_eval",
            "gpu_or_attack_execution",
        ],
    }
    package_manifest_path = output_root / "package_manifest.json"
    package_manifest_path.write_text(json.dumps(package_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _, sha_rows = write_sha256s(output_root)
    zip_path = output_root.with_suffix(".zip")
    package_zip_sha = zip_package(output_root, zip_path)
    package_zip_size = zip_path.stat().st_size
    make_read_only(output_root)

    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_round_id": EXPECTED_REVIEW_ROUND_ID,
        "reviewer_A_source": str(reviewer_a_source),
        "reviewer_A_source_sha256": reviewer_a_sha,
        "media_manifest_source": str(media_manifest),
        "media_manifest_sha256": media_manifest_sha,
        "row_count": len(review_rows),
        "raw_video_count": media_counts["raw_video"],
        "overlay_video_count": media_counts["teacher_overlay_mp4"],
        "timeline_count": media_counts["teacher_timeline_csv"],
        "human_fields_nonempty_count": human_nonempty,
        "forbidden_field_count": forbidden_field_count,
        "package_root": str(output_root),
        "package_zip": str(zip_path),
        "package_zip_sha256": package_zip_sha,
        "package_zip_size_bytes": package_zip_size,
        "package_file_count": len(sha_rows),
        "assistant_notes_path": str(args.assistant_notes_out),
        "assistant_notes_status": "NONOFFICIAL_BLANK_SCHEMA",
        "official_human_review": "INCOMPLETE",
        "approve_gate_h2": "NOT_GRANTED",
        "final_blind_selection": "NO_GO",
        "full_clean300_resolver": "NO_GO",
        "layer2_real_train_eval": "NO_GO",
        "gpu_attack_rollouts": "NO_GO",
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
