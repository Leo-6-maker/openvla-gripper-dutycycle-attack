#!/usr/bin/env python3
"""Prepare and validate H2 diagnostic human-review forms.

This is a CPU-only packaging helper. It copies resolver proposal metadata into
reviewer-facing forms and leaves every human judgment field blank. It must not
read detector telemetry, run the resolver, choose a final blind set, or infer
human judgments.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REVIEW_ROUND_ID = "h2_diagnostic_review_round_v2_20260620"
PROPOSAL_VERSION = "h2_diagnostic_review_package_v2"
ACCEPTED_STATUS = "ELIGIBLE_EVENT"
STRATA = {"DEV_CANARY", "DIAGNOSTIC_HOLDOUT"}
JUDGMENT_ENUM = {"", "YES", "NO", "UNCERTAIN", "NA"}
FORBIDDEN_REVIEWER_FIELDS = [
    "task_success",
    "detector",
    "emit",
    "probability",
    "pred_phase",
    "predicted_phase",
    "vis",
    "rand",
    "shuffled",
    "oracle",
    "attack",
]

PROPOSAL_FIELDS = [
    "review_round_id",
    "review_stratum",
    "proposal_version",
    "resolver_commit",
    "ontology_sha256",
    "teacher_schema_sha256",
    "physics_config_sha256",
    "timing_contract_sha256",
    "source_queue_sha256",
    "teacher_overlay_manifest_sha256",
    "review_id",
    "episode_key",
    "suite",
    "task_idx",
    "state_id",
    "mechanism_type",
    "teacher_status",
    "object_binding_status",
    "target_binding_status",
    "abstain_reason",
    "event_id",
    "proposed_object_body",
    "proposed_target_body_or_site",
    "proposed_close_onset",
    "proposed_grasp_established",
    "proposed_lift_onset",
    "proposed_stable_carry_start",
    "proposed_window_start",
    "proposed_anchor",
    "proposed_window_end",
    "proposed_release_onset",
    "raw_video_path",
    "teacher_only_timeline_path",
    "teacher_only_timeline_status",
    "teacher_only_overlay_path",
    "teacher_only_overlay_status",
    "video_status",
]

HUMAN_FIELDS = [
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
]

REVIEW_FIELDS = PROPOSAL_FIELDS + HUMAN_FIELDS
EVENT_JUDGMENT_FIELDS = [
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
]
JUDGMENT_FIELDS = EVENT_JUDGMENT_FIELDS + ["abstain_or_fail_closed_correct"]
CORRECTION_BY_JUDGMENT = {
    "object_identity_valid": "corrected_object_body",
    "target_identity_valid": "corrected_target_body_or_site",
    "close_onset_valid": "corrected_close_onset",
    "grasp_established_valid": "corrected_grasp_established",
    "lift_onset_valid": "corrected_lift_onset",
    "stable_carry_valid": "corrected_stable_carry_start",
    "window_start_valid": "corrected_window_start",
    "anchor_valid": "corrected_anchor",
    "window_end_valid": "corrected_window_end",
    "release_separation_valid": "corrected_release_onset",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def nonempty_human_field_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows for field in HUMAN_FIELDS if row.get(field, ""))


def teacher_timeline_rows(label: dict[str, str], event: dict[str, str]) -> list[dict[str, str]]:
    markers = [
        ("close_onset", event.get("close_onset_step", "")),
        ("grasp_established", event.get("grasp_established_step", "")),
        ("lift_onset", event.get("lift_onset_step", "")),
        ("stable_carry", event.get("stable_carry_start", "")),
        ("window_start", event.get("teacher_window_start", "")),
        ("anchor", event.get("teacher_anchor_step", "")),
        ("window_end", event.get("teacher_window_end", "")),
        ("release", event.get("release_onset_step", "")),
    ]
    rows = []
    for marker, step in markers:
        if str(step) != "":
            rows.append(
                {
                    "episode_key": label.get("episode_key", ""),
                    "teacher_status": label.get("teacher_status", ""),
                    "event_id": event.get("event_id", ""),
                    "marker": marker,
                    "step": str(step),
                    "object_body": event.get("object_body_name", ""),
                    "target_body_or_site": event.get("target_body_or_site_name", ""),
                }
            )
    if not rows:
        rows.append(
            {
                "episode_key": label.get("episode_key", ""),
                "teacher_status": label.get("teacher_status", ""),
                "event_id": event.get("event_id", ""),
                "marker": "no_teacher_event",
                "step": "",
                "object_body": "",
                "target_body_or_site": "",
            }
        )
    return rows


def maybe_write_timeline(path: Path, label: dict[str, str], event: dict[str, str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, teacher_timeline_rows(label, event), [
        "episode_key",
        "teacher_status",
        "event_id",
        "marker",
        "step",
        "object_body",
        "target_body_or_site",
    ])
    return "WROTE"


def maybe_write_overlay(raw_video: Path, output_path: Path, label: dict[str, str], event: dict[str, str]) -> str:
    if not raw_video.exists():
        return "RAW_VIDEO_MISSING"
    try:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        from scripts.stageb.cross_suite_layer1_resolver import write_teacher_overlay_video
    except Exception as exc:
        return f"OVERLAY_HELPER_UNAVAILABLE:{type(exc).__name__}"
    return write_teacher_overlay_video(raw_video, output_path, label, event)


def base_blank_row() -> dict[str, str]:
    return {field: "" for field in REVIEW_FIELDS}


def event_by_key(event_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {}
    for event in event_rows:
        result[event.get("episode_key", "")] = event
    return result


def dev_rows_to_review_rows(
    *,
    dev_manifest_rows: list[dict[str, str]],
    episode_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
    media_root: Path,
    source_queue_sha: str,
    constants: dict[str, str],
    write_media: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    labels = {row["episode_key"]: row for row in episode_rows}
    events = event_by_key(event_rows)
    review_rows = []
    media_manifest = []
    for idx, source in enumerate(dev_manifest_rows):
        key = source["canonical_key"]
        label = labels.get(key, {})
        event = events.get(key, {})
        review_id = f"v2_dev_{idx:03d}_event_00"
        raw_video = Path(source.get("episode_path", "")) / "rollout_raw.mp4"
        timeline_path = media_root / "teacher_timelines" / f"{review_id}_teacher_timeline.csv"
        overlay_path = media_root / "teacher_overlays" / f"{review_id}_teacher_overlay.mp4"
        timeline_status = maybe_write_timeline(timeline_path, label, event) if write_media else "PATH_DECLARED"
        overlay_status = maybe_write_overlay(raw_video, overlay_path, label, event) if write_media else "PATH_DECLARED"
        row = base_blank_row()
        row.update(constants)
        row.update(
            {
                "review_stratum": "DEV_CANARY",
                "source_queue_sha256": source_queue_sha,
                "review_id": review_id,
                "episode_key": key,
                "suite": source.get("suite", label.get("suite", "")),
                "task_idx": source.get("task_idx", label.get("task_idx", "")),
                "state_id": source.get("state_id", label.get("state_id", "")),
                "mechanism_type": source.get("mechanism_type", label.get("mechanism_type", "")),
                "teacher_status": label.get("teacher_status", ""),
                "object_binding_status": label.get("object_binding_status", ""),
                "target_binding_status": label.get("target_binding_status", ""),
                "abstain_reason": label.get("abstain_reason", ""),
                "event_id": event.get("event_id", ""),
                "proposed_object_body": event.get("object_body_name", ""),
                "proposed_target_body_or_site": event.get("target_body_or_site_name", ""),
                "proposed_close_onset": event.get("close_onset_step", ""),
                "proposed_grasp_established": event.get("grasp_established_step", ""),
                "proposed_lift_onset": event.get("lift_onset_step", ""),
                "proposed_stable_carry_start": event.get("stable_carry_start", ""),
                "proposed_window_start": event.get("teacher_window_start", ""),
                "proposed_anchor": event.get("teacher_anchor_step", ""),
                "proposed_window_end": event.get("teacher_window_end", ""),
                "proposed_release_onset": event.get("release_onset_step", ""),
                "raw_video_path": str(raw_video),
                "teacher_only_timeline_path": str(timeline_path),
                "teacher_only_timeline_status": timeline_status,
                "teacher_only_overlay_path": str(overlay_path) if overlay_status == "WROTE" else "",
                "teacher_only_overlay_status": overlay_status,
                "video_status": "source_path",
            }
        )
        media_manifest.extend(media_rows(row, timeline_path, overlay_path))
        review_rows.append(row)
    return review_rows, media_manifest


def diagnostic_rows_to_review_rows(
    *,
    diagnostic_queue_rows: list[dict[str, str]],
    source_queue_sha: str,
    constants: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows = []
    media_manifest = []
    for source in diagnostic_queue_rows:
        row = base_blank_row()
        row.update(constants)
        row.update(
            {
                "review_stratum": "DIAGNOSTIC_HOLDOUT",
                "source_queue_sha256": source_queue_sha,
                "review_id": source.get("review_id", ""),
                "episode_key": source.get("episode_key", ""),
                "suite": source.get("suite", ""),
                "task_idx": source.get("task_idx", ""),
                "state_id": source.get("state_id", ""),
                "mechanism_type": source.get("mechanism_type", ""),
                "teacher_status": source.get("teacher_status", ""),
                "event_id": source.get("event_id", ""),
                "proposed_object_body": source.get("proposed_object_body", ""),
                "proposed_target_body_or_site": source.get("proposed_target_body_or_site", ""),
                "proposed_close_onset": source.get("proposed_close_onset", ""),
                "proposed_grasp_established": source.get("proposed_grasp_established", ""),
                "proposed_lift_onset": source.get("proposed_lift_onset", ""),
                "proposed_stable_carry_start": source.get("proposed_stable_carry_start", ""),
                "proposed_window_start": source.get("proposed_window_start", ""),
                "proposed_anchor": source.get("proposed_anchor", ""),
                "proposed_window_end": source.get("proposed_window_end", ""),
                "proposed_release_onset": source.get("proposed_release_onset", ""),
                "raw_video_path": source.get("blind_video_path", source.get("raw_video_path", "")),
                "teacher_only_timeline_path": source.get("teacher_only_timeline_path", ""),
                "teacher_only_timeline_status": source.get("teacher_only_timeline_status", ""),
                "teacher_only_overlay_path": source.get("teacher_only_overlay_path", ""),
                "teacher_only_overlay_status": source.get("teacher_only_overlay_status", ""),
                "video_status": source.get("video_status", ""),
            }
        )
        media_manifest.extend(
            media_rows(
                row,
                Path(row["teacher_only_timeline_path"]) if row["teacher_only_timeline_path"] else None,
                Path(row["teacher_only_overlay_path"]) if row["teacher_only_overlay_path"] else None,
            )
        )
        review_rows.append(row)
    return review_rows, media_manifest


def media_rows(row: dict[str, str], timeline: Path | None, overlay: Path | None) -> list[dict[str, str]]:
    rows = []
    for artifact_type, path in [
        ("raw_video", Path(row["raw_video_path"]) if row.get("raw_video_path") else None),
        ("teacher_timeline_csv", timeline),
        ("teacher_overlay_mp4", overlay),
    ]:
        if path is None:
            rows.append({"artifact_type": artifact_type, "review_id": row["review_id"], "path": "", "size_bytes": "", "sha256": "", "exists": "false"})
            continue
        exists = path.exists()
        rows.append(
            {
                "artifact_type": artifact_type,
                "review_id": row["review_id"],
                "path": str(path),
                "size_bytes": str(path.stat().st_size) if exists else "",
                "sha256": sha256_file(path) if exists and path.is_file() else "",
                "exists": str(exists).lower(),
            }
        )
    return rows


def count_by(rows: list[dict[str, str]], *fields: str) -> dict[str, int]:
    counts = Counter("|".join(row.get(field, "") for field in fields) for row in rows)
    return dict(sorted(counts.items()))


def build_constants(args: argparse.Namespace, overlay_manifest_sha: str = "") -> dict[str, str]:
    return {
        "review_round_id": args.review_round_id,
        "proposal_version": PROPOSAL_VERSION,
        "resolver_commit": args.resolver_commit,
        "ontology_sha256": sha256_file(Path(args.ontology)),
        "teacher_schema_sha256": sha256_file(Path(args.teacher_schema)),
        "physics_config_sha256": sha256_file(Path(args.physics_config)),
        "timing_contract_sha256": sha256_file(Path(args.timing_contract)),
        "teacher_overlay_manifest_sha256": overlay_manifest_sha,
    }


def forbidden_columns(fields: list[str]) -> list[str]:
    bad = []
    for field in fields:
        lower = field.lower()
        for token in FORBIDDEN_REVIEWER_FIELDS:
            if token in lower:
                bad.append(field)
                break
    allowed = {"teacher_status", "teacher_only_timeline_path", "teacher_only_timeline_status", "teacher_only_overlay_path", "teacher_only_overlay_status"}
    return [field for field in bad if field not in allowed]


def validate_review_rows(rows: list[dict[str, str]], *, require_completed: bool = False) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    field_set = set(rows[0].keys()) if rows else set()
    missing_fields = [field for field in REVIEW_FIELDS if field not in field_set]
    if missing_fields:
        errors.append("missing_fields:" + "|".join(missing_fields))
    forbidden = forbidden_columns(list(field_set))
    if forbidden:
        errors.append("forbidden_reviewer_fields:" + "|".join(sorted(forbidden)))

    accepted = 0
    abstain_or_fail_closed = 0
    completed = 0
    nonempty_human = nonempty_human_field_count(rows)
    seen_completed_keys = set()
    for idx, row in enumerate(rows):
        status = row.get("teacher_status", "")
        is_accepted = status == ACCEPTED_STATUS
        if is_accepted:
            accepted += 1
        else:
            abstain_or_fail_closed += 1
        reviewer_id = row.get("reviewer_id", "")
        if reviewer_id:
            completed += 1
            key = (row.get("review_round_id", ""), row.get("review_id", ""), reviewer_id)
            if key in seen_completed_keys:
                errors.append(f"row{idx}.duplicate_completed_review_key:{key}")
            seen_completed_keys.add(key)

        for field in JUDGMENT_FIELDS:
            value = row.get(field, "")
            if value not in JUDGMENT_ENUM:
                errors.append(f"row{idx}.{field}:invalid_enum:{value}")

        if row.get("review_stratum") not in STRATA:
            errors.append(f"row{idx}.review_stratum:invalid:{row.get('review_stratum')}")

        if is_accepted:
            required_proposals = [
                "event_id",
                "proposed_object_body",
                "proposed_target_body_or_site",
                "proposed_close_onset",
                "proposed_grasp_established",
                "proposed_lift_onset",
                "proposed_stable_carry_start",
                "proposed_window_start",
                "proposed_anchor",
                "proposed_window_end",
                "raw_video_path",
                "teacher_only_timeline_path",
                "teacher_only_overlay_path",
            ]
            for field in required_proposals:
                if not row.get(field):
                    errors.append(f"row{idx}.{field}:accepted_proposal_required")

        if require_completed:
            if not reviewer_id:
                errors.append(f"row{idx}.reviewer_id:required")
            if not row.get("review_timestamp", ""):
                errors.append(f"row{idx}.review_timestamp:required")
            if is_accepted:
                for field in EVENT_JUDGMENT_FIELDS:
                    if not row.get(field):
                        errors.append(f"row{idx}.{field}:required_for_{status}")
                if row.get("abstain_or_fail_closed_correct") != "NA":
                    errors.append(f"row{idx}.abstain_or_fail_closed_correct:must_be_NA_for_{status}")
            else:
                if row.get("abstain_or_fail_closed_correct") not in {"YES", "NO", "UNCERTAIN"}:
                    errors.append(f"row{idx}.abstain_or_fail_closed_correct:required_for_{status}")
                for field in EVENT_JUDGMENT_FIELDS:
                    if row.get(field) != "NA":
                        errors.append(f"row{idx}.{field}:must_be_NA_for_{status}")

            for field, corrected in CORRECTION_BY_JUDGMENT.items():
                if row.get(field) == "NO" and not (row.get(corrected) or row.get("reviewer_notes")):
                    errors.append(f"row{idx}.{field}:NO_requires_{corrected}_or_notes")
            if row.get("event_exists") == "NO" and not row.get("reviewer_notes"):
                errors.append(f"row{idx}.event_exists:NO_requires_notes")
            if row.get("false_positive_carry") == "NO" and not row.get("reviewer_notes"):
                errors.append(f"row{idx}.false_positive_carry:NO_requires_notes")
            for field in JUDGMENT_FIELDS:
                if row.get(field) == "UNCERTAIN" and not row.get("reviewer_notes"):
                    errors.append(f"row{idx}.{field}:UNCERTAIN_requires_notes")

    summary = {
        "row_count": len(rows),
        "accepted_event_rows": accepted,
        "abstain_or_fail_closed_rows": abstain_or_fail_closed,
        "completed_review_rows": completed,
        "nonempty_human_field_count": nonempty_human,
        "validation_mode": "completed" if require_completed else "template_or_partial",
        "require_completed": require_completed,
        "validation_error_count": len(errors),
        "validation_status": "PASS" if not errors else "FAIL",
    }
    return errors, summary


def build_reviewer_b_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if row.get("teacher_status") == ACCEPTED_STATUS:
            selected.append(row)
            continue
        binding_text = "|".join(
            [
                row.get("teacher_status", ""),
                row.get("object_binding_status", ""),
                row.get("target_binding_status", ""),
            ]
        )
        if "AMBIGUOUS" in binding_text or "FAILED" in binding_text:
            selected.append(row)
    return selected


def write_round_manifest(args: argparse.Namespace, rows: list[dict[str, str]], media_manifest: list[dict[str, str]], summary: dict[str, Any]) -> None:
    manifest = {
        "review_round_id": args.review_round_id,
        "proposal_version": PROPOSAL_VERSION,
        "creation_timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "pr_head": current_head(),
        "resolver_commit": args.resolver_commit,
        "resolver_source_sha256": sha256_file(Path(args.resolver_source)),
        "ontology_sha256": sha256_file(Path(args.ontology)),
        "teacher_schema_sha256": sha256_file(Path(args.teacher_schema)),
        "physics_config_sha256": sha256_file(Path(args.physics_config)),
        "timing_contract_sha256": sha256_file(Path(args.timing_contract)),
        "dev_source_queue_sha256": sha256_file(Path(args.dev_manifest_csv)),
        "diagnostic_source_queue_sha256": sha256_file(Path(args.diagnostic_queue_csv)),
        "review_form_sha256": sha256_file(Path(args.output_csv)),
        "reviewer_a_form_sha256": sha256_file(Path(args.reviewer_a_csv)),
        "reviewer_b_initial_form_sha256": sha256_file(Path(args.reviewer_b_csv)),
        "timeline_manifest_sha256": sha256_file(Path(args.timeline_manifest_csv)),
        "overlay_manifest_sha256": sha256_file(Path(args.media_manifest_csv)),
        "teacher_overlay_manifest_sha256": sha256_file(Path(args.media_manifest_csv)),
        "row_counts": {
            "total": len(rows),
            "by_stratum": count_by(rows, "review_stratum"),
            "by_status": count_by(rows, "teacher_status"),
            "by_suite": count_by(rows, "suite"),
            "by_stratum_status": count_by(rows, "review_stratum", "teacher_status"),
        },
        "media_counts": count_by(media_manifest, "artifact_type", "exists"),
        "validation_summary": summary,
        "final_blind_selection": "NOT_RUN",
        "layer2_or_attack_execution": "NOT_RUN",
    }
    write_json(Path(args.manifest_json), manifest)


def build_v2_round(args: argparse.Namespace) -> dict[str, Any]:
    dev_manifest = read_csv(Path(args.dev_manifest_csv))
    dev_episode = read_csv(Path(args.dev_episode_csv))
    dev_event = read_csv(Path(args.dev_event_csv))
    diagnostic_queue = read_csv(Path(args.diagnostic_queue_csv))

    constants = build_constants(args)
    dev_source_sha = sha256_file(Path(args.dev_manifest_csv))
    diagnostic_source_sha = sha256_file(Path(args.diagnostic_queue_csv))
    dev_rows, dev_media = dev_rows_to_review_rows(
        dev_manifest_rows=dev_manifest,
        episode_rows=dev_episode,
        event_rows=dev_event,
        media_root=Path(args.server_package_root),
        source_queue_sha=dev_source_sha,
        constants=constants,
        write_media=args.write_media,
    )
    diagnostic_rows, diagnostic_media = diagnostic_rows_to_review_rows(
        diagnostic_queue_rows=diagnostic_queue,
        source_queue_sha=diagnostic_source_sha,
        constants=constants,
    )
    rows = dev_rows + diagnostic_rows
    media_manifest = dev_media + diagnostic_media
    write_csv(Path(args.timeline_manifest_csv), [row for row in media_manifest if row["artifact_type"] == "teacher_timeline_csv"], ["artifact_type", "review_id", "path", "size_bytes", "sha256", "exists"])
    write_csv(Path(args.media_manifest_csv), media_manifest, ["artifact_type", "review_id", "path", "size_bytes", "sha256", "exists"])
    overlay_manifest_sha = sha256_file(Path(args.media_manifest_csv))
    timeline_manifest_sha = sha256_file(Path(args.timeline_manifest_csv))
    for row in rows:
        row["teacher_overlay_manifest_sha256"] = overlay_manifest_sha
    write_csv(Path(args.output_csv), rows, REVIEW_FIELDS)
    write_csv(Path(args.reviewer_a_csv), rows, REVIEW_FIELDS)
    write_csv(Path(args.reviewer_b_csv), build_reviewer_b_rows(rows), REVIEW_FIELDS)
    errors, summary = validate_review_rows(rows, require_completed=False)
    summary.update(
        {
            "timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            "review_round_id": args.review_round_id,
            "proposal_version": PROPOSAL_VERSION,
            "dev_rows": len(dev_rows),
            "diagnostic_rows": len(diagnostic_rows),
            "reviewer_a_rows": len(rows),
            "reviewer_b_initial_rows": len(build_reviewer_b_rows(rows)),
            "timeline_manifest_sha256": timeline_manifest_sha,
            "overlay_manifest_sha256": overlay_manifest_sha,
            "final_blind_selection": "NOT_RUN",
            "layer2_or_attack_execution": "NOT_RUN",
            "errors": errors[:50],
        }
    )
    write_json(Path(args.summary_json), summary)
    write_round_manifest(args, rows, media_manifest, summary)
    if errors:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-v2-round", action="store_true")
    parser.add_argument("--queue-csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--review-round-id", default=REVIEW_ROUND_ID)
    parser.add_argument("--dev-manifest-csv")
    parser.add_argument("--dev-episode-csv")
    parser.add_argument("--dev-event-csv")
    parser.add_argument("--diagnostic-queue-csv")
    parser.add_argument("--reviewer-a-csv")
    parser.add_argument("--reviewer-b-csv")
    parser.add_argument("--manifest-json")
    parser.add_argument("--media-manifest-csv")
    parser.add_argument("--timeline-manifest-csv")
    parser.add_argument("--server-package-root", default="")
    parser.add_argument("--resolver-commit", default="")
    parser.add_argument("--resolver-source", default="scripts/stageb/cross_suite_layer1_resolver.py")
    parser.add_argument("--ontology", default="configs/cross_suite_task_ontology_v1.yaml")
    parser.add_argument("--teacher-schema", default="docs/schemas/cross_suite_teacher_label_schema_v1.md")
    parser.add_argument("--physics-config", default="configs/cross_suite_teacher_physics_v1.yaml")
    parser.add_argument("--timing-contract", default="reports/layer1_h2_20260620/timing_alignment_contract_20260620.md")
    parser.add_argument("--write-media", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_v2_round:
        summary = build_v2_round(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if not args.queue_csv:
        raise SystemExit("--queue-csv is required unless --build-v2-round is used")
    rows = read_csv(Path(args.queue_csv))
    errors, summary = validate_review_rows(rows, require_completed=args.require_completed)
    summary.update(
        {
            "timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            "queue_csv": str(Path(args.queue_csv)),
            "output_csv": str(Path(args.output_csv)),
            "final_blind_selection": "NOT_RUN",
            "layer2_or_attack_execution": "NOT_RUN",
            "errors": errors[:50],
        }
    )
    if not args.validate_only:
        write_csv(Path(args.output_csv), rows, REVIEW_FIELDS)
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
