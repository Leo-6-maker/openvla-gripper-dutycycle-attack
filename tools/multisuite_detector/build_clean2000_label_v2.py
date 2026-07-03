#!/usr/bin/env python3
"""Build CLEAN2000 Label V2 from synthetic fixtures only.

This implementation is intentionally limited to Gate A1 CPU-only validation:
it refuses non-synthetic inputs and never reads CLEAN2000 live/backup sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path


AVAILABILITY_COLUMNS = [
    "suite",
    "task_id",
    "episode_key",
    "canonical_index_label",
    "real_source_label_found",
    "source_label_path",
    "source_label_sha256",
    "source_anchor",
    "source_window_start",
    "source_window_end",
    "source_confidence",
    "source_event_id",
    "matches_canonical",
    "notes",
    "source_record_found",
    "source_schema_valid",
    "source_positive_anchor_valid",
    "source_no_event",
    "source_explicit_abstention",
    "source_clean_failure_no_event",
    "shared_fields_comparable",
    "shared_fields_match",
    "uncomparable_due_to_missing_fields",
    "source_timing_fields_present",
    "source_mechanism_eligible_schema_valid",
]

EPISODE_CENSUS_COLUMNS = [
    "episode_key",
    "parent_key",
    "suite",
    "task_id",
    "task_name",
    "state_id",
    "outcome_class",
    "mechanism_scope_class",
    "cohort_class",
    "label_record_present",
    "record_schema_valid",
    "teacher_positive_label_valid",
    "positive_anchor_valid",
    "explicit_abstention_valid",
    "timing_signal_usable",
    "teacher_anchor_step",
    "teacher_window_start",
    "teacher_window_end",
    "teacher_confidence",
    "teacher_event_id",
    "abstain_reason",
    "feature_schema_sha256",
    "source_manifest_sha256",
    "artifact_inventory_sha256",
    "n_steps",
    "n_valid_steps",
    "first_valid_step",
    "invalid_feature_steps",
    "feature_25d_join_ok",
    "cohort_set",
    "model_split",
    "parent_leakage_status",
    "task_leakage_status",
    "normalization_source_status",
]

OUTPUT_COLUMNS = [
    "episode_key",
    "parent_key",
    "suite",
    "task_id",
    "clean_success",
    "mechanism_eligible",
    "event_present",
    "anchor_absolute_step",
    "window_start",
    "window_end",
    "event_source",
    "source_path",
    "source_sha256",
    "builder_git_sha",
    "builder_sha256",
    "invalid_reason",
    "abstain_reason",
    "mechanism_type",
    "event_id",
    "segment_id",
    "event_rank",
    "coordinate_semantics",
    "trace_length",
    "source_schema_version",
    "teacher_confidence",
    "confidence_available",
    "confidence_provenance",
    "event_id_provenance",
    "source_semantics_authority",
    "source_jsonl_check_mode",
    "window_valid",
    "label_validity_status",
    "manual_audit_status",
    "manual_audit_reason",
]

CROSSTAB_COLUMNS = ["cohort_class", "source_positive", "source_no_event", "total"]
COORDINATE_SEMANTICS = "zero_based_observation_before_action_start_inclusive_end_exclusive_full_trajectory"
SOURCE_SCHEMA_VERSION = "source_availability_v1_presence_only_jsonl_v1"
SOURCE_SEMANTICS_AUTHORITY = "SOURCE_AVAILABILITY_LEDGER"
SOURCE_JSONL_CHECK_MODE = "PROVENANCE_PRESENCE_ONLY"
V2_EPISODE_TABLE_SCOPE = "PRIMARY_EVENT_ONLY"
MULTI_EVENT_POLICY = "MULTI_EVENT_TABLE_SEPARATE_ARTIFACT"
CONFIDENCE_POLICY = "SOURCE_AVAILABILITY_ONLY_NO_CENSUS_BACKFILL"
EVENT_ID_POLICY = "SOURCE_EVENT_ID_OR_EPISODE_PRIMARY_EVENT_FALLBACK"
MANUAL_AUDIT_SEED = 20260703
MAX_SYNTHETIC_ROWS = 200
SENTINEL_NAME = ".label_v2_synthetic_fixture.json"
SENTINEL_SHA256 = "dae3e444c0c8693d5a80e20fd0761ddd4d559ae038ef7ecb6cfef054ab69f482"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ALLOWED_COHORTS = {
    "PRIMARY_SUCCESS_ELIGIBLE",
    "ELIGIBLE_CLEAN_FAILURE",
    "MECHANISM_INELIGIBLE_ABSTENTION",
}
MECHANISM_TYPE_BY_SCOPE = {
    "MECHANISM_ELIGIBLE": "SINGLE_OBJECT_TRANSFER",
    "MECHANISM_INELIGIBLE": "UNSUPPORTED",
}
INVALID_PRECEDENCE = [
    "MISSING_SOURCE_RECORD",
    "SOURCE_SCHEMA_INVALID",
    "ANCHOR_INVALID",
    "WINDOW_COORDINATE_INVALID",
    "TRUNCATED_TRACE",
]
MANUAL_COLUMNS = [
    "suite",
    "task_id",
    "episode_key",
    "requested_priority",
    "actual_selected_category",
    "fallback_used",
    "fallback_reason",
    "sampling_seed",
]


class BuildError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise BuildError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_strict(path: Path, columns: list[str], allow_empty: set[str] | None = None) -> list[dict[str, str]]:
    allow_empty = allow_empty or set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
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


def reject_path_traversal(path_text: str) -> None:
    if any(part == ".." for part in Path(path_text).parts):
        fail(f"path traversal is not allowed: {path_text}")


def reject_symlink(path: Path) -> None:
    probe = path
    candidates = []
    while True:
        candidates.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    for candidate in candidates:
        if candidate.is_symlink():
            fail(f"symlink path is not allowed: {candidate}")


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
    actual = sha256_file(sentinel)
    if actual != SENTINEL_SHA256:
        fail(f"synthetic fixture sentinel SHA256 mismatch: {actual}")


def parse_bool(value: str, field: str, episode_key: str) -> bool:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    fail(f"{episode_key}: illegal bool for {field}: {value}")


def parse_int(value: str, field: str, episode_key: str) -> int:
    try:
        if value.strip() != str(int(value)):
            fail(f"{episode_key}: illegal int for {field}: {value}")
        return int(value)
    except ValueError:
        fail(f"{episode_key}: illegal int for {field}: {value}")
    raise AssertionError("unreachable")


def parse_float(value: str, field: str, episode_key: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        fail(f"{episode_key}: illegal float for {field}: {value}")
    if not math.isfinite(parsed):
        fail(f"{episode_key}: illegal float for {field}: {value}")
    return parsed


def parse_confidence(value: str, field: str, episode_key: str) -> str:
    if value == "UNKNOWN":
        return value
    return str(parse_float(value, field, episode_key))


def git_sha(repo: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", head):
            fail(f"git HEAD is not a 40-hex SHA: {head}")
        return head
    except Exception:
        fail("unable to determine git HEAD")
    raise AssertionError("unreachable")


def unique_by_episode(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        episode = row["episode_key"]
        if episode in out:
            fail(f"duplicate {label} episode_key: {episode}")
        out[episode] = row
    return out


def bool_field(row: dict[str, str], field: str) -> bool:
    return parse_bool(row[field], field, row["episode_key"])


def clean_success_from(census: dict[str, str]) -> bool:
    outcome = census["outcome_class"]
    if outcome == "CLEAN_SUCCESS":
        return True
    if outcome == "CLEAN_FAILURE":
        return False
    fail(f"{census['episode_key']}: unsupported outcome_class: {outcome}")
    raise AssertionError("unreachable")


def mechanism_eligible_from(census: dict[str, str]) -> bool:
    scope = census["mechanism_scope_class"]
    if scope == "MECHANISM_ELIGIBLE":
        return True
    if scope == "MECHANISM_INELIGIBLE":
        return False
    fail(f"{census['episode_key']}: unsupported mechanism_scope_class: {scope}")
    raise AssertionError("unreachable")


def validate_cohort(census: dict[str, str], clean_success: bool, mechanism_eligible: bool, event_present: bool) -> None:
    cohort = census["cohort_class"]
    episode = census["episode_key"]
    if cohort not in ALLOWED_COHORTS:
        fail(f"{episode}: unsupported cohort_class: {cohort}")
    if cohort == "PRIMARY_SUCCESS_ELIGIBLE" and (not clean_success or not mechanism_eligible):
        fail(f"{episode}: cohort invariant failed for PRIMARY_SUCCESS_ELIGIBLE")
    if cohort == "ELIGIBLE_CLEAN_FAILURE" and (clean_success or not mechanism_eligible):
        fail(f"{episode}: cohort invariant failed for ELIGIBLE_CLEAN_FAILURE")
    if cohort == "MECHANISM_INELIGIBLE_ABSTENTION" and (mechanism_eligible or event_present):
        fail(f"{episode}: cohort invariant failed for MECHANISM_INELIGIBLE_ABSTENTION")


def event_present_from(availability: dict[str, str]) -> bool:
    episode = availability["episode_key"]
    positive = bool_field(availability, "source_positive_anchor_valid")
    no_event = bool_field(availability, "source_no_event")
    explicit_abstention = bool_field(availability, "source_explicit_abstention")
    clean_failure_no_event = bool_field(availability, "source_clean_failure_no_event")
    if positive and (no_event or explicit_abstention or clean_failure_no_event):
        fail(f"{episode}: positive/no-event source flags conflict")
    if positive:
        return True
    if no_event or explicit_abstention or clean_failure_no_event:
        return False
    fail(f"{episode}: no source event disposition flag is set")
    raise AssertionError("unreachable")


def invalid_reason_from(availability: dict[str, str], event_present: bool) -> str:
    episode = availability["episode_key"]
    if not bool_field(availability, "real_source_label_found") or not bool_field(availability, "source_record_found"):
        return "MISSING_SOURCE_RECORD"
    if not bool_field(availability, "source_schema_valid"):
        return "SOURCE_SCHEMA_INVALID"
    if event_present and not bool_field(availability, "source_positive_anchor_valid"):
        return "ANCHOR_INVALID"
    if event_present and not bool_field(availability, "source_timing_fields_present"):
        return "WINDOW_COORDINATE_INVALID"
    try:
        canonical = json.loads(availability["canonical_index_label"])
    except json.JSONDecodeError:
        return "SOURCE_SCHEMA_INVALID"
    reason = str(canonical.get("teacher_invalid_reason", ""))
    if reason:
        if reason not in INVALID_PRECEDENCE:
            fail(f"{episode}: UNKNOWN_INVALID_REASON: {reason}")
        return reason
    return ""


def load_source_records(path: Path, cache: dict[Path, dict[str, dict]]) -> dict[str, dict]:
    if path in cache:
        return cache[path]
    records = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path.name}:{line_no}: invalid source JSONL: {exc}")
            episode = record.get("episode_key")
            if not episode:
                fail(f"{path.name}:{line_no}: missing episode_key")
            if episode in records:
                fail(f"{path.name}:{line_no}: duplicate source episode_key: {episode}")
            records[episode] = record
    cache[path] = records
    return records


def validate_source_file(availability: dict[str, str], fixture_root: Path, cache: dict[Path, dict[str, dict]]) -> dict:
    source_path = availability["source_label_path"]
    source_sha = availability["source_label_sha256"]
    episode = availability["episode_key"]
    validate_sha_arg(source_sha, f"{episode} source_label_sha256")
    reject_path_traversal(source_path)
    path = fixture_root / source_path
    reject_symlink(path)
    require_descendant(path, fixture_root, "source label")
    if not path.is_file():
        fail(f"{episode}: source label file does not exist: {source_path}")
    actual = sha256_file(path)
    if actual != source_sha:
        fail(f"{episode}: source label SHA256 mismatch: expected {source_sha}, got {actual}")
    records = load_source_records(path, cache)
    if episode not in records:
        fail(f"{episode}: source label record not found in {source_path}")
    return records[episode]


def validate_availability_census_audit_flags(availability: dict[str, str], census: dict[str, str]) -> None:
    episode = census["episode_key"]
    pairs = [
        ("source_record_found", "label_record_present"),
        ("source_schema_valid", "record_schema_valid"),
        ("source_positive_anchor_valid", "positive_anchor_valid"),
    ]
    for availability_field, census_field in pairs:
        if bool_field(availability, availability_field) != bool_field(census, census_field):
            fail(f"{episode}: availability/census audit flag mismatch: {availability_field} vs {census_field}")


def adapt_rows(
    availability_rows: list[dict[str, str]],
    census_rows: list[dict[str, str]],
    builder_sha: str,
    git_head: str,
    fixture_root: Path,
) -> list[dict[str, str]]:
    if len(census_rows) > MAX_SYNTHETIC_ROWS:
        fail(f"synthetic fixture row count exceeds {MAX_SYNTHETIC_ROWS}: {len(census_rows)}")
    availability = unique_by_episode(availability_rows, "availability")
    census = unique_by_episode(census_rows, "census")
    source_cache: dict[Path, dict[str, dict]] = {}
    if set(availability) != set(census):
        fail("episode_key set mismatch between availability and census")

    output = []
    for episode in sorted(census):
        c = census[episode]
        a = availability[episode]
        if a["suite"] != c["suite"] or a["task_id"] != c["task_id"]:
            fail(f"{episode}: availability/census suite-task mismatch")
        validate_source_file(a, fixture_root, source_cache)
        validate_availability_census_audit_flags(a, c)

        clean_success = clean_success_from(c)
        mechanism_eligible = mechanism_eligible_from(c)
        event_present = event_present_from(a)
        validate_cohort(c, clean_success, mechanism_eligible, event_present)

        invalid_reason = invalid_reason_from(a, event_present)
        if event_present:
            anchor = parse_int(a["source_anchor"], "source_anchor", episode)
            window_start = parse_int(a["source_window_start"], "source_window_start", episode)
            source_window_end = parse_int(a["source_window_end"], "source_window_end", episode)
            window_end = source_window_end + 1
            event_source = "source_availability"
        else:
            anchor = window_start = window_end = -1
            event_source = ""

        event_rank = 1 if event_present else 0
        if event_present:
            if a["source_event_id"] and a["source_event_id"] != "UNKNOWN":
                event_id = a["source_event_id"]
                event_id_provenance = "SOURCE_AVAILABILITY"
            else:
                event_id = f"{episode}#event_{event_rank}"
                event_id_provenance = "EPISODE_PRIMARY_EVENT_FALLBACK"
        else:
            event_id = "NO_EVENT"
            event_id_provenance = "NOT_APPLICABLE"

        confidence = parse_confidence(a["source_confidence"], "source_confidence", episode)
        confidence_available = confidence != "UNKNOWN"
        confidence_provenance = "SOURCE_AVAILABILITY" if confidence_available else "UNAVAILABLE"

        output.append(
            {
                "episode_key": episode,
                "parent_key": c["parent_key"],
                "suite": c["suite"],
                "task_id": c["task_id"],
                "cohort_class": c["cohort_class"],
                "clean_success": "true" if clean_success else "false",
                "mechanism_eligible": "true" if mechanism_eligible else "false",
                "event_present": "true" if event_present else "false",
                "anchor_absolute_step": str(anchor),
                "window_start": str(window_start),
                "window_end": str(window_end),
                "event_source": event_source,
                "source_path": a["source_label_path"],
                "source_sha256": a["source_label_sha256"],
                "builder_git_sha": git_head,
                "builder_sha256": builder_sha,
                "invalid_reason": invalid_reason,
                "abstain_reason": c["abstain_reason"] or ("MECHANISM_INELIGIBLE" if not mechanism_eligible else ""),
                "mechanism_type": MECHANISM_TYPE_BY_SCOPE[c["mechanism_scope_class"]],
                "event_id": event_id,
                "segment_id": f"{episode}#segment_{event_rank}" if event_present else "NO_EVENT",
                "event_rank": str(event_rank),
                "coordinate_semantics": COORDINATE_SEMANTICS,
                "trace_length": c["n_steps"],
                "source_schema_version": SOURCE_SCHEMA_VERSION,
                "teacher_confidence": confidence,
                "confidence_available": "true" if confidence_available else "false",
                "confidence_provenance": confidence_provenance,
                "event_id_provenance": event_id_provenance,
                "source_semantics_authority": SOURCE_SEMANTICS_AUTHORITY,
                "source_jsonl_check_mode": SOURCE_JSONL_CHECK_MODE,
            }
        )
    return output


def validate_and_transform(rows: list[dict[str, str]], builder_sha: str, git_head: str) -> list[dict[str, str]]:
    seen = set()
    out = []
    if len(rows) > MAX_SYNTHETIC_ROWS:
        fail(f"synthetic fixture row count exceeds {MAX_SYNTHETIC_ROWS}: {len(rows)}")
    for row in rows:
        episode = row["episode_key"]
        if not episode:
            fail("empty episode_key")
        if episode in seen:
            fail(f"duplicate episode_key: {episode}")
        seen.add(episode)

        clean_success = parse_bool(row["clean_success"], "clean_success", episode)
        mechanism_eligible = parse_bool(row["mechanism_eligible"], "mechanism_eligible", episode)
        event_present = parse_bool(row["event_present"], "event_present", episode)
        anchor = parse_int(row["anchor_absolute_step"], "anchor_absolute_step", episode)
        window_start = parse_int(row["window_start"], "window_start", episode)
        window_end = parse_int(row["window_end"], "window_end", episode)
        event_rank = parse_int(row["event_rank"], "event_rank", episode)
        trace_length = parse_int(row["trace_length"], "trace_length", episode)
        confidence = parse_confidence(row["teacher_confidence"], "teacher_confidence", episode)

        if row["source_schema_version"] != SOURCE_SCHEMA_VERSION:
            fail(f"{episode}: source_schema_version must be {SOURCE_SCHEMA_VERSION}")
        if row["coordinate_semantics"] != COORDINATE_SEMANTICS:
            fail(f"{episode}: unexpected coordinate_semantics")
        if row.get("source_semantics_authority") != SOURCE_SEMANTICS_AUTHORITY:
            fail(f"{episode}: source_semantics_authority must be {SOURCE_SEMANTICS_AUTHORITY}")
        if row.get("source_jsonl_check_mode") != SOURCE_JSONL_CHECK_MODE:
            fail(f"{episode}: source_jsonl_check_mode must be {SOURCE_JSONL_CHECK_MODE}")
        if row.get("confidence_provenance") not in {"SOURCE_AVAILABILITY", "UNAVAILABLE"}:
            fail(f"{episode}: invalid confidence_provenance")
        if parse_bool(row.get("confidence_available", ""), "confidence_available", episode) != (confidence != "UNKNOWN"):
            fail(f"{episode}: confidence_available does not match teacher_confidence")
        if confidence != "UNKNOWN" and not 0.0 <= float(confidence) <= 1.0:
            fail(f"{episode}: teacher_confidence outside [0, 1]")
        if event_rank < 0:
            fail(f"{episode}: negative event_rank")
        if event_present and row["abstain_reason"]:
            fail(f"{episode}: positive event cannot have abstain_reason")
        if not event_present and row["event_source"]:
            fail(f"{episode}: no-event row cannot have event_source")
        if not mechanism_eligible and event_present:
            fail(f"{episode}: mechanism-ineligible row cannot be positive")
        if not mechanism_eligible and not row["abstain_reason"]:
            fail(f"{episode}: mechanism-ineligible row requires abstain_reason")

        if row["invalid_reason"] and row["invalid_reason"] not in INVALID_PRECEDENCE:
            fail(f"{episode}: UNKNOWN_INVALID_REASON: {row['invalid_reason']}")
        if row["invalid_reason"] in INVALID_PRECEDENCE:
            window_valid = False
        elif event_present:
            window_valid = (
                trace_length > 0
                and window_start >= 0
                and window_end > window_start
                and window_end <= trace_length
                and window_start <= anchor < window_end
            )
        else:
            if (anchor, window_start, window_end) != (-1, -1, -1):
                fail(f"{episode}: no-event row has non-empty anchor/window")
            window_valid = True

        output = {name: row[name] for name in OUTPUT_COLUMNS if name in row}
        output["builder_git_sha"] = git_head
        output["builder_sha256"] = builder_sha
        output["window_valid"] = "true" if window_valid else "false"
        output["label_validity_status"] = "VALID" if window_valid else "INVALID_WINDOW"
        output["manual_audit_status"] = "PENDING"
        output["manual_audit_reason"] = ""
        if not window_valid and not output["invalid_reason"]:
            output["invalid_reason"] = "INVALID_WINDOW"
        out.append(output)
    return out


def validate_counts(rows: list[dict[str, str]], crosstab_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        cohort = row["cohort_class"]
        bucket = counts.setdefault(cohort, {"positive": 0, "no_event": 0, "total": 0})
        if row["event_present"] == "true":
            bucket["positive"] += 1
        elif row["event_present"] == "false":
            bucket["no_event"] += 1
        else:
            fail(f"{row['episode_key']}: event_present must be validated before counts")
        bucket["total"] += 1

    seen_crosstab = set()
    for row in crosstab_rows:
        cohort = row["cohort_class"]
        if cohort in seen_crosstab:
            fail(f"duplicate crosstab cohort: {cohort}")
        seen_crosstab.add(cohort)
        expected = counts.get(cohort)
        if expected is None:
            fail(f"crosstab unknown cohort: {cohort}")
        got_positive = parse_int(row["source_positive"], "source_positive", cohort)
        got_no_event = parse_int(row["source_no_event"], "source_no_event", cohort)
        got_total = parse_int(row["total"], "total", cohort)
        if (got_positive, got_no_event, got_total) != (
            expected["positive"],
            expected["no_event"],
            expected["total"],
        ):
            fail(f"crosstab mismatch for {cohort}")
    if {row["cohort_class"] for row in crosstab_rows} != set(counts):
        fail("crosstab cohort set mismatch")
    return counts


def row_category(row: dict[str, str]) -> str:
    if row["mechanism_eligible"] == "false":
        return "abstention_or_ineligible"
    if row["event_present"] == "true" and row["clean_success"] == "true":
        return "positive_clean_success"
    if row["event_present"] == "false" and row["mechanism_eligible"] == "true":
        return "eligible_no_event"
    if row["clean_success"] == "false" or row["label_validity_status"] != "VALID":
        return "failure_or_boundary"
    return "other"


def manual_audit_sample(rows: list[dict[str, str]], enforce_quota: bool = False, expected_n: int | None = None) -> list[dict[str, str]]:
    rng = random.Random(MANUAL_AUDIT_SEED)
    by_task: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault((row["suite"], row["task_id"]), []).append(row)

    picked = []
    if enforce_quota and len(by_task) != 40:
        fail(f"manual audit quota requires 40 suite-task units, got {len(by_task)}")
    for suite, task in sorted(by_task):
        task_rows = sorted(by_task[(suite, task)], key=lambda r: r["episode_key"])
        rng.shuffle(task_rows)
        priorities = [
            ("positive_clean_success", lambda r: row_category(r) == "positive_clean_success"),
            ("eligible_no_event", lambda r: row_category(r) == "eligible_no_event"),
            ("failure_or_boundary", lambda r: row_category(r) == "failure_or_boundary"),
            ("abstention_or_ineligible", lambda r: row_category(r) == "abstention_or_ineligible"),
        ]
        used = set()
        for label, predicate in priorities:
            match = next((r for r in task_rows if r["episode_key"] not in used and predicate(r)), None)
            fallback = False
            if match is None:
                match = next((r for r in task_rows if r["episode_key"] not in used), None)
                fallback = match is not None
            if match is None:
                continue
            used.add(match["episode_key"])
            actual = row_category(match)
            picked.append(
                {
                    "suite": suite,
                    "task_id": task,
                    "episode_key": match["episode_key"],
                    "requested_priority": label,
                    "actual_selected_category": actual,
                    "fallback_used": "true" if fallback else "false",
                    "fallback_reason": "" if not fallback else f"missing_{label}",
                    "sampling_seed": str(MANUAL_AUDIT_SEED),
                }
            )
        if enforce_quota and len(used) != 4:
            fail(f"manual audit quota requires 4 rows for {suite}/{task}, got {len(used)}")
    if expected_n is not None and len(picked) != expected_n:
        fail(f"manual audit sample count mismatch: expected {expected_n}, got {len(picked)}")
    return sorted(picked, key=lambda r: (r["suite"], r["task_id"], r["requested_priority"], r["episode_key"]))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_sha_arg(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value):
        fail(f"{label} must be 64 lowercase hex characters")


def verify_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    for text in [args.source_manifest, args.episode_census, args.source_crosstab, args.output_root, args.synthetic_fixture_root, args.synthetic_output_root]:
        reject_path_traversal(text)
    if not args.synthetic or not args.dry_run:
        fail("Gate A1 implementation allows only --synthetic --dry-run")
    for value, label in [
        (args.expected_source_sha256, "expected-source-sha256"),
        (args.expected_census_sha256, "expected-census-sha256"),
        (args.expected_crosstab_sha256, "expected-crosstab-sha256"),
    ]:
        validate_sha_arg(value, label)
    source = Path(args.source_manifest)
    census = Path(args.episode_census)
    crosstab = Path(args.source_crosstab)
    output_root = Path(args.output_root)
    fixture_root = Path(args.synthetic_fixture_root)
    output_base = Path(args.synthetic_output_root)
    reject_symlink(fixture_root)
    reject_symlink(output_base)
    require_sentinel(fixture_root)
    for path in [source, census, crosstab]:
        reject_symlink(path)
        require_descendant(path, fixture_root, "input")
        if not path.is_file():
            fail(f"input file does not exist: {path}")
    reject_symlink(output_root)
    require_descendant(output_root, output_base, "output root")
    if output_root.exists() and any(output_root.iterdir()):
        fail(f"output root must be empty: {output_root}")
    return source, census, crosstab, output_root, fixture_root


def compare_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        fail(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--episode-census", required=True)
    parser.add_argument("--source-crosstab", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--synthetic-fixture-root", required=True)
    parser.add_argument("--synthetic-output-root", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-census-sha256", required=True)
    parser.add_argument("--expected-crosstab-sha256", required=True)
    parser.add_argument("--expected-manual-sample-n", type=int)
    parser.add_argument("--enforce-manual-quota", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        source, census, crosstab, output_root, fixture_root = verify_inputs(args)
        source_sha = compare_sha(source, args.expected_source_sha256, "source manifest")
        census_sha = compare_sha(census, args.expected_census_sha256, "episode census")
        crosstab_sha = compare_sha(crosstab, args.expected_crosstab_sha256, "source crosstab")

        builder_path = Path(__file__).resolve()
        builder_sha = sha256_file(builder_path)
        repo = builder_path.parents[2]
        git_head = git_sha(repo)

        availability_rows = read_csv_strict(
            source,
            AVAILABILITY_COLUMNS,
            {"source_event_id", "notes"},
        )
        census_rows = read_csv_strict(
            census,
            EPISODE_CENSUS_COLUMNS,
            {
                "teacher_event_id",
                "abstain_reason",
                "model_split",
                "parent_leakage_status",
                "task_leakage_status",
                "normalization_source_status",
            },
        )
        crosstab_rows = read_csv_strict(crosstab, CROSSTAB_COLUMNS)
        adapted_rows = adapt_rows(availability_rows, census_rows, builder_sha, git_head, fixture_root)
        output_rows = validate_and_transform(adapted_rows, builder_sha, git_head)
        counts = validate_counts(adapted_rows, crosstab_rows)

        output_root.mkdir(parents=True, exist_ok=True)
        label_path = output_root / "label_v2.csv"
        summary_path = output_root / "validation_summary.json"
        manifest_path = output_root / "build_manifest.json"
        manual_path = output_root / "manual_audit_sample_manifest.csv"
        sums_path = output_root / "SHA256SUMS"

        write_csv(label_path, OUTPUT_COLUMNS, output_rows)
        manual_rows = manual_audit_sample(output_rows, args.enforce_manual_quota, args.expected_manual_sample_n)
        write_csv(manual_path, MANUAL_COLUMNS, manual_rows)
        summary = {
            "status": "PASS",
            "mode": "synthetic_dry_run",
            "row_count": len(output_rows),
            "counts": counts,
            "invalid_window_rows": sum(1 for r in output_rows if r["label_validity_status"] == "INVALID_WINDOW"),
            "manual_audit_sample_n": len(read_csv_strict(manual_path, MANUAL_COLUMNS, {"fallback_reason"})),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "mode": "synthetic_dry_run",
            "synthetic_only": True,
            "builder_git_sha": git_head,
            "builder_sha256": builder_sha,
            "source_semantics_authority": SOURCE_SEMANTICS_AUTHORITY,
            "source_jsonl_check_mode": SOURCE_JSONL_CHECK_MODE,
            "v2_episode_table_scope": V2_EPISODE_TABLE_SCOPE,
            "multi_event_policy": MULTI_EVENT_POLICY,
            "confidence_policy": CONFIDENCE_POLICY,
            "event_id_policy": EVENT_ID_POLICY,
            "inputs": {
                "source_manifest": {"path": str(source), "sha256": source_sha},
                "episode_census": {"path": str(census), "sha256": census_sha},
                "source_crosstab": {"path": str(crosstab), "sha256": crosstab_sha},
            },
            "outputs": ["label_v2.csv", "build_manifest.json", "validation_summary.json", "manual_audit_sample_manifest.csv", "SHA256SUMS"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        sums = []
        for path in [label_path, manifest_path, summary_path, manual_path]:
            sums.append(f"{sha256_file(path)}  {path.name}")
        sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")
        return 0
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
