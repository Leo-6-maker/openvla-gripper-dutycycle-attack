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
import json
from pathlib import Path
from typing import Any


PROPOSAL_FIELDS = [
    "review_id",
    "episode_key",
    "suite",
    "task_idx",
    "state_id",
    "mechanism_type",
    "teacher_status",
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
    "blind_video_path",
    "teacher_only_timeline_path",
    "teacher_only_overlay_path",
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
JUDGMENT_FIELDS = [
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
]
JUDGMENT_ENUM = {"", "YES", "NO", "UNCERTAIN", "NA"}
ACCEPTED_STATUS = "ELIGIBLE_EVENT"


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


def build_review_template(queue_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for source in queue_rows:
        row = {field: source.get(field, "") for field in PROPOSAL_FIELDS}
        for field in HUMAN_FIELDS:
            row[field] = ""
        rows.append(row)
    return rows


def validate_review_rows(rows: list[dict[str, str]], *, require_completed: bool = False) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    field_set = set(rows[0].keys()) if rows else set()
    missing_fields = [field for field in REVIEW_FIELDS if field not in field_set]
    if missing_fields:
        errors.append("missing_fields:" + "|".join(missing_fields))

    accepted = 0
    abstain_or_fail_closed = 0
    completed = 0
    for idx, row in enumerate(rows):
        status = row.get("teacher_status", "")
        if status == ACCEPTED_STATUS:
            accepted += 1
        else:
            abstain_or_fail_closed += 1
        if row.get("reviewer_id"):
            completed += 1
        for field in JUDGMENT_FIELDS:
            value = row.get(field, "")
            if value not in JUDGMENT_ENUM:
                errors.append(f"row{idx}.{field}:invalid_enum:{value}")
        if require_completed:
            if not row.get("reviewer_id"):
                errors.append(f"row{idx}.reviewer_id:required")
            required_for_status = (
                [
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
                if status == ACCEPTED_STATUS
                else ["abstain_or_fail_closed_correct"]
            )
            for field in required_for_status:
                if not row.get(field):
                    errors.append(f"row{idx}.{field}:required_for_{status}")
    summary = {
        "row_count": len(rows),
        "accepted_event_rows": accepted,
        "abstain_or_fail_closed_rows": abstain_or_fail_closed,
        "completed_review_rows": completed,
        "require_completed": require_completed,
        "validation_error_count": len(errors),
        "validation_status": "PASS" if not errors else "FAIL",
    }
    return errors, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_rows = read_csv(Path(args.queue_csv))
    rows = input_rows if args.validate_only else build_review_template(input_rows)
    errors, summary = validate_review_rows(rows, require_completed=args.require_completed)
    summary.update(
        {
            "timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            "queue_csv": str(Path(args.queue_csv)),
            "output_csv": str(Path(args.output_csv)),
            "codex_populated_human_judgments": False,
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
