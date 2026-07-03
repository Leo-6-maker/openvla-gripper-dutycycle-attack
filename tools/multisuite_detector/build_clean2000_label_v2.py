#!/usr/bin/env python3
"""Build and validate CLEAN2000 Label V2 from frozen ledger CSVs.

Formal execution is implemented but remains subject to a separate authorization
record. This module performs no model inference, rollout, training, GPU work, or
live source mutation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


AVAILABILITY_COLUMNS = [
    "suite", "task_id", "episode_key", "canonical_index_label",
    "real_source_label_found", "source_label_path", "source_label_sha256",
    "source_anchor", "source_window_start", "source_window_end",
    "source_confidence", "source_event_id", "matches_canonical", "notes",
    "source_record_found", "source_schema_valid",
    "source_positive_anchor_valid", "source_no_event",
    "source_explicit_abstention", "source_clean_failure_no_event",
    "shared_fields_comparable", "shared_fields_match",
    "uncomparable_due_to_missing_fields", "source_timing_fields_present",
    "source_mechanism_eligible_schema_valid",
]
EPISODE_CENSUS_COLUMNS = [
    "episode_key", "parent_key", "suite", "task_id", "task_name", "state_id",
    "outcome_class", "mechanism_scope_class", "cohort_class",
    "label_record_present", "record_schema_valid",
    "teacher_positive_label_valid", "positive_anchor_valid",
    "explicit_abstention_valid", "timing_signal_usable",
    "teacher_anchor_step", "teacher_window_start", "teacher_window_end",
    "teacher_confidence", "teacher_event_id", "abstain_reason",
    "feature_schema_sha256", "source_manifest_sha256",
    "artifact_inventory_sha256", "n_steps", "n_valid_steps",
    "first_valid_step", "invalid_feature_steps", "feature_25d_join_ok",
    "cohort_set", "model_split", "parent_leakage_status",
    "task_leakage_status", "normalization_source_status",
]
OUTPUT_COLUMNS = [
    "episode_key", "parent_key", "suite", "task_id", "cohort_class",
    "clean_success", "mechanism_eligible", "event_present",
    "anchor_absolute_step", "window_start", "window_end", "event_source",
    "source_path", "source_sha256", "builder_git_sha", "builder_sha256",
    "invalid_reason", "abstain_reason", "mechanism_type", "event_id",
    "segment_id", "event_rank", "coordinate_semantics", "trace_length",
    "source_schema_version", "teacher_confidence", "confidence_available",
    "confidence_provenance", "event_id_provenance",
    "source_semantics_authority", "source_jsonl_check_mode", "window_valid",
    "label_validity_status", "manual_audit_status", "manual_audit_reason",
]
MANUAL_COLUMNS = [
    "suite", "task_id", "episode_key", "cohort_class", "clean_success",
    "mechanism_eligible", "event_present", "label_validity_status",
    "requested_priority", "actual_selected_category", "fallback_used",
    "fallback_reason", "sampling_seed",
]
CROSSTAB_COLUMNS = ["cohort_class", "source_positive", "source_no_event", "total"]

COORDINATE_SEMANTICS = "zero_based_observation_before_action_start_inclusive_end_exclusive_full_trajectory"
SOURCE_SCHEMA_VERSION = "source_availability_ledger_v1"
SOURCE_SEMANTICS_AUTHORITY = "SOURCE_AVAILABILITY_LEDGER"
SOURCE_JSONL_CHECK_MODE = "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ"
V2_EPISODE_TABLE_SCOPE = "PRIMARY_EVENT_ONLY"
MULTI_EVENT_POLICY = "MULTI_EVENT_TABLE_SEPARATE_ARTIFACT"
CONFIDENCE_POLICY = "SOURCE_AVAILABILITY_ONLY_NO_CENSUS_BACKFILL"
EVENT_ID_POLICY = "SOURCE_EVENT_ID_OR_EPISODE_PRIMARY_EVENT_FALLBACK"
SOURCE_WINDOW_END_SEMANTICS = "INCLUSIVE_CONVERTED_TO_V2_EXCLUSIVE"
SCHEMA_VERSION = "clean2000_label_v2_episode_primary_event_v1"
MANUAL_AUDIT_SEED = 20260703
MAX_SYNTHETIC_ROWS = 200
FORMAL_ROW_COUNT = 2000
FORMAL_MANUAL_SAMPLE_N = 160
FORMAL_SUITE_TASK_UNITS = 40
FORMAL_EXPECTED_COUNTS = {
    "PRIMARY_SUCCESS_ELIGIBLE": {"positive": 772, "no_event": 271, "total": 1043},
    "ELIGIBLE_CLEAN_FAILURE": {"positive": 31, "no_event": 276, "total": 307},
    "MECHANISM_INELIGIBLE_ABSTENTION": {"positive": 0, "no_event": 650, "total": 650},
}
ALLOWED_COHORTS = set(FORMAL_EXPECTED_COUNTS)
MECHANISM_TYPE_BY_SCOPE = {
    "MECHANISM_ELIGIBLE": "GRIPPER_TRANSFER_ELIGIBLE",
    "MECHANISM_INELIGIBLE": "MECHANISM_UNSUPPORTED",
}
KNOWN_INVALID_REASONS = {
    "MISSING_SOURCE_RECORD", "SOURCE_SCHEMA_INVALID", "ANCHOR_INVALID",
    "WINDOW_COORDINATE_INVALID", "TRUNCATED_TRACE",
}
SENTINEL_NAME = ".label_v2_synthetic_fixture.json"
SENTINEL_SHA256 = "dae3e444c0c8693d5a80e20fd0761ddd4d559ae038ef7ecb6cfef054ab69f482"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
MANUAL_PRIORITIES = [
    "positive_clean_success", "eligible_no_event", "failure_or_boundary",
    "abstention_or_ineligible",
]
MANUAL_FALLBACK_ORDER = {
    "positive_clean_success": [
        "positive_clean_success", "eligible_no_event", "failure_or_boundary",
        "abstention_or_ineligible",
    ],
    "eligible_no_event": [
        "eligible_no_event", "failure_or_boundary", "abstention_or_ineligible",
        "positive_clean_success",
    ],
    "failure_or_boundary": [
        "failure_or_boundary", "abstention_or_ineligible",
        "positive_clean_success", "eligible_no_event",
    ],
    "abstention_or_ineligible": [
        "abstention_or_ineligible", "positive_clean_success",
        "eligible_no_event", "failure_or_boundary",
    ],
}


class BuildError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha_arg(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value):
        fail(f"{label} must be 64 lowercase hex characters")


def validate_git_sha_arg(value: str, label: str) -> None:
    if not GIT_SHA_RE.fullmatch(value):
        fail(f"{label} must be 40 lowercase hex characters")


def parse_bool(value: str, field: str, episode: str) -> bool:
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    fail(f"{episode}: illegal bool for {field}: {value}")


def parse_int(value: str, field: str, episode: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        fail(f"{episode}: illegal int for {field}: {value}")
    if value.strip() != str(parsed):
        fail(f"{episode}: illegal int for {field}: {value}")
    return parsed


def parse_confidence(value: str, episode: str) -> str:
    if value == "UNKNOWN":
        return value
    try:
        parsed = float(value)
    except ValueError:
        fail(f"{episode}: illegal float for source_confidence: {value}")
    if not math.isfinite(parsed):
        fail(f"{episode}: illegal float for source_confidence: {value}")
    return str(parsed)


def read_csv_strict(
    path: Path,
    columns: list[str],
    allow_empty: set[str] | None = None,
) -> list[dict[str, str]]:
    allow_empty = allow_empty or set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            fail(f"{path.name}: expected columns {columns}, got {reader.fieldnames}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: row has extra cells")
            missing = [key for key, value in row.items() if value is None]
            if missing:
                fail(f"{path.name}:{line_no}: row has missing cells: {missing}")
            empty = [key for key, value in row.items() if value == "" and key not in allow_empty]
            if empty:
                fail(f"{path.name}:{line_no}: required fields are empty: {empty}")
            rows.append(row)
    return rows


def reject_path_traversal(text: str) -> None:
    if any(part == ".." for part in Path(text).parts):
        fail(f"path traversal is not allowed: {text}")


def reject_symlink(path: Path) -> None:
    probe = path
    while True:
        if probe.is_symlink():
            fail(f"symlink path is not allowed: {probe}")
        if probe.parent == probe:
            break
        probe = probe.parent


def require_descendant(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        fail(f"{label} must be under declared synthetic root: {path}")


def require_sentinel(root: Path) -> None:
    sentinel = root / SENTINEL_NAME
    reject_symlink(sentinel)
    if not sentinel.is_file():
        fail(f"synthetic fixture sentinel missing: {sentinel}")
    if sha256_file(sentinel) != SENTINEL_SHA256:
        fail("synthetic fixture sentinel SHA256 mismatch")


def git_sha(repo: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        fail("unable to determine git HEAD")
    if not GIT_SHA_RE.fullmatch(value):
        fail(f"git HEAD is not a 40-hex SHA: {value}")
    return value


def require_clean_worktree(repo: Path) -> None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo, text=True,
        )
    except Exception:
        fail("unable to check git worktree cleanliness")
    if status.strip():
        fail("git worktree must be clean for formal ledger build")


def unique_by_episode(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        episode = row["episode_key"]
        if episode in result:
            fail(f"duplicate {label} episode_key: {episode}")
        result[episode] = row
    return result


def bool_field(row: dict[str, str], field: str) -> bool:
    return parse_bool(row[field], field, row["episode_key"])


def event_present_from(row: dict[str, str]) -> bool:
    episode = row["episode_key"]
    positive = bool_field(row, "source_positive_anchor_valid")
    no_event = bool_field(row, "source_no_event")
    abstention = bool_field(row, "source_explicit_abstention")
    failure_no_event = bool_field(row, "source_clean_failure_no_event")
    if positive and (no_event or abstention or failure_no_event):
        fail(f"{episode}: positive/no-event source flags conflict")
    if positive:
        return True
    if no_event or abstention or failure_no_event:
        return False
    fail(f"{episode}: no source event disposition flag is set")


def clean_success_from(row: dict[str, str]) -> bool:
    if row["outcome_class"] == "CLEAN_SUCCESS":
        return True
    if row["outcome_class"] == "CLEAN_FAILURE":
        return False
    fail(f"{row['episode_key']}: unsupported outcome_class: {row['outcome_class']}")


def mechanism_eligible_from(row: dict[str, str]) -> bool:
    if row["mechanism_scope_class"] == "MECHANISM_ELIGIBLE":
        return True
    if row["mechanism_scope_class"] == "MECHANISM_INELIGIBLE":
        return False
    fail(f"{row['episode_key']}: unsupported mechanism_scope_class")


def validate_cohort(
    census: dict[str, str],
    clean_success: bool,
    mechanism_eligible: bool,
    event_present: bool,
) -> None:
    episode = census["episode_key"]
    cohort = census["cohort_class"]
    if cohort not in ALLOWED_COHORTS:
        fail(f"{episode}: unsupported cohort_class: {cohort}")
    if cohort == "PRIMARY_SUCCESS_ELIGIBLE" and (not clean_success or not mechanism_eligible):
        fail(f"{episode}: cohort invariant failed for PRIMARY_SUCCESS_ELIGIBLE")
    if cohort == "ELIGIBLE_CLEAN_FAILURE" and (clean_success or not mechanism_eligible):
        fail(f"{episode}: cohort invariant failed for ELIGIBLE_CLEAN_FAILURE")
    if cohort == "MECHANISM_INELIGIBLE_ABSTENTION" and (mechanism_eligible or event_present):
        fail(f"{episode}: cohort invariant failed for MECHANISM_INELIGIBLE_ABSTENTION")


def validate_source_disposition(
    availability: dict[str, str],
    census: dict[str, str],
    clean_success: bool,
    event_present: bool,
) -> None:
    episode = census["episode_key"]
    cohort = census["cohort_class"]
    no_event = bool_field(availability, "source_no_event")
    abstention = bool_field(availability, "source_explicit_abstention")
    failure_no_event = bool_field(availability, "source_clean_failure_no_event")
    if event_present:
        if no_event or abstention or failure_no_event:
            fail(f"{episode}: positive event has a no-event disposition flag")
        return
    if cohort == "PRIMARY_SUCCESS_ELIGIBLE":
        if not no_event or abstention or failure_no_event:
            fail(f"{episode}: primary-success no-event disposition is unexplained")
    elif cohort == "ELIGIBLE_CLEAN_FAILURE":
        if clean_success or not failure_no_event or abstention:
            fail(f"{episode}: eligible clean-failure no-event disposition is unexplained")
    elif cohort == "MECHANISM_INELIGIBLE_ABSTENTION":
        if not abstention or failure_no_event:
            fail(f"{episode}: mechanism-ineligible disposition is unexplained")


def invalid_reason_from(availability: dict[str, str], event_present: bool) -> str:
    if not bool_field(availability, "real_source_label_found") or not bool_field(availability, "source_record_found"):
        return "MISSING_SOURCE_RECORD"
    if not bool_field(availability, "source_schema_valid"):
        return "SOURCE_SCHEMA_INVALID"
    if event_present and not bool_field(availability, "source_positive_anchor_valid"):
        return "ANCHOR_INVALID"
    if event_present and not bool_field(availability, "source_timing_fields_present"):
        return "WINDOW_COORDINATE_INVALID"
    return ""


def adapt_rows(
    availability_rows: list[dict[str, str]],
    census_rows: list[dict[str, str]],
    builder_sha: str,
    git_head: str,
    max_rows: int,
) -> list[dict[str, str]]:
    if len(census_rows) > max_rows:
        fail(f"input row count exceeds {max_rows}: {len(census_rows)}")
    availability = unique_by_episode(availability_rows, "availability")
    census = unique_by_episode(census_rows, "census")
    if set(availability) != set(census):
        fail("episode_key set mismatch between availability and census")
    output = []
    for episode in sorted(census):
        a = availability[episode]
        c = census[episode]
        if a["suite"] != c["suite"] or a["task_id"] != c["task_id"]:
            fail(f"{episode}: availability/census suite-task mismatch")
        validate_sha_arg(a["source_label_sha256"], f"{episode} source_label_sha256")
        reject_path_traversal(a["source_label_path"])
        clean_success = clean_success_from(c)
        mechanism_eligible = mechanism_eligible_from(c)
        event_present = event_present_from(a)
        validate_cohort(c, clean_success, mechanism_eligible, event_present)
        validate_source_disposition(a, c, clean_success, event_present)
        invalid_reason = invalid_reason_from(a, event_present)
        if event_present:
            anchor = parse_int(a["source_anchor"], "source_anchor", episode)
            start = parse_int(a["source_window_start"], "source_window_start", episode)
            end = parse_int(a["source_window_end"], "source_window_end", episode) + 1
            source_event_id = a["source_event_id"]
            if source_event_id and source_event_id != "UNKNOWN":
                event_id = f"{episode}#{source_event_id}"
                event_id_provenance = "SOURCE_AVAILABILITY"
            else:
                event_id = f"{episode}#event_1"
                event_id_provenance = "EPISODE_PRIMARY_EVENT_FALLBACK"
            event_source = "source_availability"
            segment_id = f"{episode}#segment_1"
            event_rank = "1"
        else:
            anchor = start = end = -1
            event_id = "NO_EVENT"
            event_id_provenance = "NOT_APPLICABLE"
            event_source = ""
            segment_id = "NO_EVENT"
            event_rank = "0"
        confidence = parse_confidence(a["source_confidence"], episode)
        output.append({
            "episode_key": episode,
            "parent_key": c["parent_key"],
            "suite": c["suite"],
            "task_id": c["task_id"],
            "cohort_class": c["cohort_class"],
            "clean_success": "true" if clean_success else "false",
            "mechanism_eligible": "true" if mechanism_eligible else "false",
            "event_present": "true" if event_present else "false",
            "anchor_absolute_step": str(anchor),
            "window_start": str(start),
            "window_end": str(end),
            "event_source": event_source,
            "source_path": a["source_label_path"],
            "source_sha256": a["source_label_sha256"],
            "builder_git_sha": git_head,
            "builder_sha256": builder_sha,
            "invalid_reason": invalid_reason,
            "abstain_reason": c["abstain_reason"] or ("MECHANISM_INELIGIBLE" if not mechanism_eligible else ""),
            "mechanism_type": MECHANISM_TYPE_BY_SCOPE[c["mechanism_scope_class"]],
            "event_id": event_id,
            "segment_id": segment_id,
            "event_rank": event_rank,
            "coordinate_semantics": COORDINATE_SEMANTICS,
            "trace_length": c["n_steps"],
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "teacher_confidence": confidence,
            "confidence_available": "false" if confidence == "UNKNOWN" else "true",
            "confidence_provenance": "UNAVAILABLE" if confidence == "UNKNOWN" else "SOURCE_AVAILABILITY",
            "event_id_provenance": event_id_provenance,
            "source_semantics_authority": SOURCE_SEMANTICS_AUTHORITY,
            "source_jsonl_check_mode": SOURCE_JSONL_CHECK_MODE,
        })
    return output


def validate_and_transform(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        episode = row["episode_key"]
        event_present = parse_bool(row["event_present"], "event_present", episode)
        invalid_reason = row["invalid_reason"]
        if invalid_reason and invalid_reason not in KNOWN_INVALID_REASONS:
            fail(f"{episode}: unknown invalid reason: {invalid_reason}")
        if event_present:
            anchor = parse_int(row["anchor_absolute_step"], "anchor_absolute_step", episode)
            start = parse_int(row["window_start"], "window_start", episode)
            end = parse_int(row["window_end"], "window_end", episode)
            trace_length = parse_int(row["trace_length"], "trace_length", episode)
            window_valid = (
                not invalid_reason and start >= 0 and end > start and end <= trace_length
                and start <= anchor < end
            )
        else:
            if {row["anchor_absolute_step"], row["window_start"], row["window_end"]} != {"-1"}:
                fail(f"{episode}: no-event coordinates must be -1")
            window_valid = not invalid_reason
        transformed = dict(row)
        transformed["window_valid"] = "true" if window_valid else "false"
        transformed["label_validity_status"] = "VALID" if window_valid else "INVALID_WINDOW"
        transformed["manual_audit_status"] = "PENDING"
        transformed["manual_audit_reason"] = ""
        output.append(transformed)
    return output


def validate_counts(
    rows: list[dict[str, str]],
    crosstab_rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    counts = {cohort: {"positive": 0, "no_event": 0, "total": 0} for cohort in ALLOWED_COHORTS}
    for row in rows:
        cohort = row["cohort_class"]
        counts[cohort]["total"] += 1
        key = "positive" if row["event_present"] == "true" else "no_event"
        counts[cohort][key] += 1
    seen = set()
    for row in crosstab_rows:
        cohort = row["cohort_class"]
        if cohort in seen:
            fail(f"duplicate crosstab cohort: {cohort}")
        seen.add(cohort)
        if cohort not in counts:
            fail(f"unsupported crosstab cohort: {cohort}")
        expected = counts[cohort]
        actual = (
            parse_int(row["source_positive"], "source_positive", cohort),
            parse_int(row["source_no_event"], "source_no_event", cohort),
            parse_int(row["total"], "total", cohort),
        )
        if actual != (expected["positive"], expected["no_event"], expected["total"]):
            fail(f"crosstab mismatch for {cohort}")
    if seen != set(counts):
        fail("crosstab cohort set mismatch")
    return counts


def validate_formal_closure(rows: list[dict[str, str]], counts: dict[str, dict[str, int]]) -> None:
    if len(rows) != FORMAL_ROW_COUNT:
        fail(f"formal ledger build requires {FORMAL_ROW_COUNT} rows, got {len(rows)}")
    if counts != FORMAL_EXPECTED_COUNTS:
        fail(f"formal cohort counts mismatch: expected {FORMAL_EXPECTED_COUNTS}, got {counts}")
    units = {(row["suite"], row["task_id"]) for row in rows}
    if len(units) != FORMAL_SUITE_TASK_UNITS:
        fail(f"formal manual audit requires {FORMAL_SUITE_TASK_UNITS} suite-task units, got {len(units)}")


def row_category(row: dict[str, str]) -> str:
    if row["mechanism_eligible"] == "false":
        return "abstention_or_ineligible"
    if row["clean_success"] == "false" or row["label_validity_status"] != "VALID":
        return "failure_or_boundary"
    if row["event_present"] == "true":
        return "positive_clean_success"
    if row["event_present"] == "false":
        return "eligible_no_event"
    return "other"


def manual_audit_sample(
    rows: list[dict[str, str]],
    enforce_quota: bool = False,
    expected_n: int | None = None,
) -> list[dict[str, str]]:
    rng = random.Random(MANUAL_AUDIT_SEED)
    by_task: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault((row["suite"], row["task_id"]), []).append(row)
    if enforce_quota and len(by_task) != 40:
        fail(f"manual audit quota requires 40 suite-task units, got {len(by_task)}")
    picked = []
    for suite, task in sorted(by_task):
        task_rows = sorted(by_task[(suite, task)], key=lambda row: row["episode_key"])
        rng.shuffle(task_rows)
        used = set()
        for requested in MANUAL_PRIORITIES:
            match = None
            actual = ""
            for allowed in MANUAL_FALLBACK_ORDER[requested]:
                match = next((row for row in task_rows if row["episode_key"] not in used and row_category(row) == allowed), None)
                if match is not None:
                    actual = allowed
                    break
            if match is None:
                continue
            used.add(match["episode_key"])
            fallback = actual != requested
            picked.append({
                "suite": suite,
                "task_id": task,
                "episode_key": match["episode_key"],
                "cohort_class": match["cohort_class"],
                "clean_success": match["clean_success"],
                "mechanism_eligible": match["mechanism_eligible"],
                "event_present": match["event_present"],
                "label_validity_status": match["label_validity_status"],
                "requested_priority": requested,
                "actual_selected_category": actual,
                "fallback_used": "true" if fallback else "false",
                "fallback_reason": "" if not fallback else f"missing_{requested}_used_{actual}",
                "sampling_seed": str(MANUAL_AUDIT_SEED),
            })
        if enforce_quota and len(used) != 4:
            fail(f"manual audit quota requires 4 rows for {suite}/{task}, got {len(used)}")
    if expected_n is not None and len(picked) != expected_n:
        fail(f"manual audit sample count mismatch: expected {expected_n}, got {len(picked)}")
    return sorted(picked, key=lambda row: (row["suite"], row["task_id"], row["requested_priority"], row["episode_key"]))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.synthetic and args.dry_run:
        return "synthetic-dry-run"
    fail("specify --mode synthetic-dry-run or --mode formal-ledger-build")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic-dry-run", "formal-ledger-build", "validate-formal-output", "self-test-closeout"])
    parser.add_argument("--source-manifest")
    parser.add_argument("--episode-census")
    parser.add_argument("--source-crosstab")
    parser.add_argument("--output-root")
    parser.add_argument("--synthetic-fixture-root")
    parser.add_argument("--synthetic-output-root")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-census-sha256", required=True)
    parser.add_argument("--expected-crosstab-sha256", required=True)
    parser.add_argument("--expected-git-commit-sha")
    parser.add_argument("--expected-builder-sha256")
    parser.add_argument("--require-clean-worktree", action="store_true")
    parser.add_argument("--expected-manual-sample-n", type=int)
    parser.add_argument("--enforce-manual-quota", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def compare_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        fail(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def verify_paths(args: argparse.Namespace, mode: str) -> tuple[Path, Path, Path, Path]:
    for field in ["source_manifest", "episode_census", "source_crosstab", "output_root"]:
        if not getattr(args, field):
            fail(f"{field.replace('_', '-')} is required")
        reject_path_traversal(getattr(args, field))
    source = Path(args.source_manifest)
    census = Path(args.episode_census)
    crosstab = Path(args.source_crosstab)
    output = Path(args.output_root)
    for path in [source, census, crosstab]:
        reject_symlink(path)
        if not path.is_file():
            fail(f"input file does not exist: {path}")
    reject_symlink(output)
    if mode == "synthetic-dry-run":
        if not args.synthetic or not args.dry_run:
            fail("synthetic-dry-run requires --synthetic --dry-run")
        if not args.synthetic_fixture_root or not args.synthetic_output_root:
            fail("synthetic-dry-run requires synthetic roots")
        fixture_root = Path(args.synthetic_fixture_root)
        output_base = Path(args.synthetic_output_root)
        require_sentinel(fixture_root)
        for path in [source, census, crosstab]:
            require_descendant(path, fixture_root, "input")
        require_descendant(output, output_base, "output root")
        if output.exists() and any(output.iterdir()):
            fail(f"output root must be empty: {output}")
    else:
        if args.synthetic or args.dry_run:
            fail("formal-ledger-build must not use --synthetic or --dry-run")
        if not output.is_absolute():
            fail("formal-ledger-build output root must be absolute")
        repo = Path(__file__).resolve().parents[2]
        try:
            output.resolve().relative_to(repo.resolve())
            fail("formal-ledger-build output root must be outside the git repository")
        except ValueError:
            pass
        if output.exists() and any(output.iterdir()):
            fail(f"output root must be empty: {output}")
    return source, census, crosstab, output


def build_once(args: argparse.Namespace, mode: str) -> int:
    source, census, crosstab, output = verify_paths(args, mode)
    for value, label in [
        (args.expected_source_sha256, "expected-source-sha256"),
        (args.expected_census_sha256, "expected-census-sha256"),
        (args.expected_crosstab_sha256, "expected-crosstab-sha256"),
    ]:
        validate_sha_arg(value, label)
    source_sha = compare_sha(source, args.expected_source_sha256, "source manifest")
    census_sha = compare_sha(census, args.expected_census_sha256, "episode census")
    crosstab_sha = compare_sha(crosstab, args.expected_crosstab_sha256, "source crosstab")

    builder_path = Path(__file__).resolve()
    builder_sha = sha256_file(builder_path)
    repo = builder_path.parents[2]
    head = git_sha(repo)
    if mode == "formal-ledger-build":
        if not args.expected_git_commit_sha:
            fail("formal-ledger-build requires --expected-git-commit-sha")
        if not args.expected_builder_sha256:
            fail("formal-ledger-build requires --expected-builder-sha256")
        validate_git_sha_arg(args.expected_git_commit_sha, "expected-git-commit-sha")
        validate_sha_arg(args.expected_builder_sha256, "expected-builder-sha256")
        if head != args.expected_git_commit_sha:
            fail(f"git HEAD mismatch: expected {args.expected_git_commit_sha}, got {head}")
        if builder_sha != args.expected_builder_sha256:
            fail(f"builder SHA256 mismatch: expected {args.expected_builder_sha256}, got {builder_sha}")
        if not args.require_clean_worktree:
            fail("formal-ledger-build requires --require-clean-worktree")
        require_clean_worktree(repo)

    availability = read_csv_strict(source, AVAILABILITY_COLUMNS, {"source_event_id", "notes"})
    census_rows = read_csv_strict(census, EPISODE_CENSUS_COLUMNS, {
        "teacher_event_id", "abstain_reason", "model_split",
        "parent_leakage_status", "task_leakage_status",
        "normalization_source_status",
    })
    crosstab_rows = read_csv_strict(crosstab, CROSSTAB_COLUMNS)
    max_rows = MAX_SYNTHETIC_ROWS if mode == "synthetic-dry-run" else FORMAL_ROW_COUNT
    adapted = adapt_rows(availability, census_rows, builder_sha, head, max_rows)
    output_rows = validate_and_transform(adapted)
    counts = validate_counts(adapted, crosstab_rows)
    if mode == "formal-ledger-build":
        validate_formal_closure(output_rows, counts)
    enforce = args.enforce_manual_quota or mode == "formal-ledger-build"
    expected_n = FORMAL_MANUAL_SAMPLE_N if mode == "formal-ledger-build" else args.expected_manual_sample_n
    manual_rows = manual_audit_sample(output_rows, enforce, expected_n)

    target = output
    staging = None
    if mode == "formal-ledger-build":
        if output.exists():
            if any(output.iterdir()):
                fail(f"output root must be empty: {output}")
            output.rmdir()
        staging = output.parent / f".{output.name}.staging-{os.getpid()}"
        if staging.exists():
            fail(f"staging output already exists: {staging}")
        target = staging
    try:
        target.mkdir(parents=True, exist_ok=False)
        label_path = target / "label_v2.csv"
        manual_path = target / "manual_audit_sample_manifest.csv"
        summary_path = target / "validation_summary.json"
        manifest_path = target / "build_manifest.json"
        sums_path = target / "SHA256SUMS"
        write_csv(label_path, OUTPUT_COLUMNS, output_rows)
        write_csv(manual_path, MANUAL_COLUMNS, manual_rows)
        summary = {
            "status": "PASS", "mode": mode, "row_count": len(output_rows),
            "counts": counts,
            "invalid_window_rows": sum(row["label_validity_status"] == "INVALID_WINDOW" for row in output_rows),
            "manual_audit_sample_n": len(manual_rows),
            "unexplained_disposition_rows": 0,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "synthetic_only": mode == "synthetic-dry-run",
            "builder_git_sha": head,
            "builder_sha256": builder_sha,
            "source_semantics_authority": SOURCE_SEMANTICS_AUTHORITY,
            "source_jsonl_check_mode": SOURCE_JSONL_CHECK_MODE,
            "v2_episode_table_scope": V2_EPISODE_TABLE_SCOPE,
            "multi_event_policy": MULTI_EVENT_POLICY,
            "confidence_policy": CONFIDENCE_POLICY,
            "event_id_policy": EVENT_ID_POLICY,
            "source_window_end_semantics": SOURCE_WINDOW_END_SEMANTICS,
            "manual_fallback_policy": MANUAL_FALLBACK_ORDER,
            "formal_output_root": str(output) if mode == "formal-ledger-build" else str(target),
            "atomic_publish": mode == "formal-ledger-build",
            "inputs": {
                "source_manifest": {"path": str(source), "sha256": source_sha},
                "episode_census": {"path": str(census), "sha256": census_sha},
                "source_crosstab": {"path": str(crosstab), "sha256": crosstab_sha},
            },
            "outputs": ["label_v2.csv", "build_manifest.json", "validation_summary.json", "manual_audit_sample_manifest.csv", "SHA256SUMS"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sums = [f"{sha256_file(path)}  {path.name}" for path in [label_path, manifest_path, summary_path, manual_path]]
        sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")
        if staging is not None:
            staging.rename(output)
        return 0
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mode = resolve_mode(args)
        if mode == "self-test-closeout":
            if any(set(order) != set(MANUAL_PRIORITIES) for order in MANUAL_FALLBACK_ORDER.values()):
                fail("manual fallback matrix is not total")
            print("Label V2 closeout self-test: PASS")
            return 0
        if mode == "validate-formal-output":
            fail("validate-formal-output requires the finalized authorization command")
        return build_once(args, mode)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
