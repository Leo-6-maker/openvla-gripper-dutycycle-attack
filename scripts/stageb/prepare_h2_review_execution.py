#!/usr/bin/env python3
"""Prepare H2 diagnostic review execution artifacts.

This helper is deliberately CPU-only. It verifies the frozen v2.1 reviewer
package, writes a pre-review integrity receipt, and creates independent blank
working copies for human reviewers. It must not populate human judgments.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.prepare_h2_human_review_forms import (
    FORBIDDEN_REVIEWER_FIELDS,
    HUMAN_FIELDS,
    forbidden_columns,
)


DEFAULT_ROUND = "h2_diagnostic_review_round_v2_1_20260621"
DEFAULT_FILE_STEM = "h2_diagnostic_review_round_v2_1"
TABLE_ROOT = Path("tables/layer1_h2_20260620")
REPORT_ROOT = Path("reports/layer1_h2_20260620")
IN_PROGRESS_ROOT = Path("human_reviews/in_progress")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def count_human_fields(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows for field in HUMAN_FIELDS if row.get(field, ""))


def media_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('artifact_type', '')}|exists={row.get('exists', '')}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def copy_with_sha(src: Path, dst: Path) -> dict[str, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source_path": str(src),
        "working_copy_path": str(dst),
        "source_sha256": sha256_file(src),
        "working_copy_sha256": sha256_file(dst),
    }


def verify_and_prepare(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    round_id = args.review_round_id

    file_stem = args.file_stem
    form_path = TABLE_ROOT / f"{file_stem}_form_template.csv"
    reviewer_a_path = TABLE_ROOT / f"reviewer_A_{file_stem}.csv"
    reviewer_b_path = TABLE_ROOT / f"reviewer_B_initial_{file_stem}.csv"
    media_manifest_path = TABLE_ROOT / f"{file_stem}_media_manifest.csv"
    overlay_manifest_path = TABLE_ROOT / f"{file_stem}_overlay_only_manifest.csv"
    timeline_manifest_path = TABLE_ROOT / f"{file_stem}_timeline_manifest.csv"

    files = {
        "review_form_sha256": form_path,
        "reviewer_a_form_sha256": reviewer_a_path,
        "reviewer_b_initial_form_sha256": reviewer_b_path,
        "media_manifest_sha256": media_manifest_path,
        "overlay_only_manifest_sha256": overlay_manifest_path,
        "timeline_manifest_sha256": timeline_manifest_path,
    }
    errors: list[str] = []
    computed_sha: dict[str, str] = {}
    for field, path in files.items():
        if not path.exists():
            errors.append(f"{field}:missing:{path}")
            continue
        computed = sha256_file(path)
        computed_sha[field] = computed
        expected = manifest.get(field, "")
        if computed != expected:
            errors.append(f"{field}:sha_mismatch:expected={expected}:actual={computed}")

    form_rows = read_csv(form_path)
    reviewer_a_rows = read_csv(reviewer_a_path)
    reviewer_b_rows = read_csv(reviewer_b_path)
    media_rows = read_csv(media_manifest_path)
    overlay_rows = read_csv(overlay_manifest_path)
    timeline_rows = read_csv(timeline_manifest_path)

    if len(form_rows) != 36:
        errors.append(f"review_form_rows:expected=36:actual={len(form_rows)}")
    if len(reviewer_a_rows) != 36:
        errors.append(f"reviewer_a_rows:expected=36:actual={len(reviewer_a_rows)}")
    if len(reviewer_b_rows) != 20:
        errors.append(f"reviewer_b_initial_rows:expected=20:actual={len(reviewer_b_rows)}")
    if count_human_fields(form_rows) != 0 or count_human_fields(reviewer_a_rows) != 0 or count_human_fields(reviewer_b_rows) != 0:
        errors.append("human_fields_nonempty")

    for label, rows in [("form", form_rows), ("reviewer_a", reviewer_a_rows), ("reviewer_b", reviewer_b_rows)]:
        forbidden = forbidden_columns(list(rows[0].keys()) if rows else [])
        if forbidden:
            errors.append(f"{label}:forbidden_columns:" + "|".join(forbidden))

    if len(media_rows) != 108:
        errors.append(f"media_manifest_rows:expected=108:actual={len(media_rows)}")
    if len(overlay_rows) != 36 or any(row.get("artifact_type") != "teacher_overlay_mp4" for row in overlay_rows):
        errors.append("overlay_only_manifest:not_36_teacher_overlay_mp4")
    if len(timeline_rows) != 36 or any(row.get("artifact_type") != "teacher_timeline_csv" for row in timeline_rows):
        errors.append("timeline_manifest:not_36_teacher_timeline_csv")
    for idx, row in enumerate(media_rows):
        if row.get("exists") != "true" or not row.get("sha256") or int(row.get("size_bytes") or "0") <= 0:
            errors.append(f"media_manifest.row{idx}:invalid_artifact_entry")

    copy_results: dict[str, dict[str, str]] = {}
    if not errors:
        copy_results["reviewer_a_working_copy"] = copy_with_sha(
            reviewer_a_path,
            IN_PROGRESS_ROOT / f"reviewer_A_{file_stem}_WORKING_COPY.csv",
        )
        copy_results["reviewer_b_initial_working_copy"] = copy_with_sha(
            reviewer_b_path,
            IN_PROGRESS_ROOT / f"reviewer_B_initial_{file_stem}_WORKING_COPY.csv",
        )

    receipt = {
        "review_round_id": round_id,
        "timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest_json": str(manifest_path),
        "computed_sha256": computed_sha,
        "reviewer_a_rows": len(reviewer_a_rows),
        "reviewer_b_initial_rows": len(reviewer_b_rows),
        "human_fields_nonempty_count": {
            "form": count_human_fields(form_rows),
            "reviewer_a": count_human_fields(reviewer_a_rows),
            "reviewer_b_initial": count_human_fields(reviewer_b_rows),
        },
        "media_counts": media_counts(media_rows),
        "forbidden_field_tokens_checked": FORBIDDEN_REVIEWER_FIELDS,
        "working_copies": copy_results,
        "layer2_or_attack_execution": "NOT_RUN",
        "final_blind_selection": "NOT_RUN",
    }
    write_json(Path(args.receipt_json), receipt)
    if errors:
        raise SystemExit(1)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-round-id", default=DEFAULT_ROUND)
    parser.add_argument("--file-stem", default=DEFAULT_FILE_STEM)
    parser.add_argument(
        "--manifest-json",
        default=str(REPORT_ROOT / f"{DEFAULT_FILE_STEM}_manifest.json"),
    )
    parser.add_argument(
        "--receipt-json",
        default=str(REPORT_ROOT / f"{DEFAULT_ROUND}_pre_review_integrity_receipt.json"),
    )
    return parser.parse_args()


def main() -> None:
    receipt = verify_and_prepare(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
