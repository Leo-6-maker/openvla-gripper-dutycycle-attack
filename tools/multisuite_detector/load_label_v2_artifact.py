#!/usr/bin/env python3
"""Read-only Label V2 five-file artifact validator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


class LabelV2ArtifactError(ValueError):
    pass


OUTPUT_FILES = {
    "label_v2.csv",
    "build_manifest.json",
    "validation_summary.json",
    "manual_audit_sample_manifest.csv",
    "SHA256SUMS",
}
LABEL_COLUMNS = [
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
FORMAL_COUNTS = {
    "PRIMARY_SUCCESS_ELIGIBLE": {"positive": 772, "no_event": 271, "total": 1043},
    "ELIGIBLE_CLEAN_FAILURE": {"positive": 31, "no_event": 276, "total": 307},
    "MECHANISM_INELIGIBLE_ABSTENTION": {"positive": 0, "no_event": 650, "total": 650},
}
MANUAL_CATEGORIES = {
    "positive_clean_success",
    "eligible_no_event",
    "failure_or_boundary",
    "abstention_or_ineligible",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def fail(message: str) -> None:
    raise LabelV2ArtifactError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_symlink(path: Path) -> None:
    probe = path
    while True:
        if probe.is_symlink():
            fail(f"symlink path is not allowed: {probe}")
        if probe.parent == probe:
            break
        probe = probe.parent


def reject_path_traversal(value: str, field: str) -> None:
    if any(part == ".." for part in Path(value).parts):
        fail(f"{field} contains path traversal: {value}")


def read_csv_strict(path: Path, columns: list[str], allow_empty: set[str] | None = None) -> list[dict[str, str]]:
    allow_empty = allow_empty or set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            fail(f"{path.name}: header mismatch")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            missing = [key for key, value in row.items() if value is None]
            if missing:
                fail(f"{path.name}:{line_no}: missing cells: {missing}")
            empty = [key for key, value in row.items() if value == "" and key not in allow_empty]
            if empty:
                fail(f"{path.name}:{line_no}: empty required fields: {empty}")
            rows.append(row)
    return rows


def parse_bool(value: str, field: str, episode: str = "") -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    fail(f"{episode}: {field} must be lowercase true/false")


def parse_int(value: str, field: str, episode: str = "") -> int:
    try:
        parsed = int(value)
    except ValueError:
        fail(f"{episode}: {field} must be an integer")
    if str(parsed) != value:
        fail(f"{episode}: {field} must be canonical integer text")
    return parsed


def row_category(row: dict[str, str]) -> str:
    if row["mechanism_eligible"] == "false":
        return "abstention_or_ineligible"
    if row["clean_success"] == "false" or row["label_validity_status"] != "VALID":
        return "failure_or_boundary"
    if row["event_present"] == "true":
        return "positive_clean_success"
    return "eligible_no_event"


def verify_file_set(root: Path) -> None:
    reject_symlink(root)
    if not root.is_dir():
        fail(f"artifact root is not a directory: {root}")
    names = {path.name for path in root.iterdir()}
    if names != OUTPUT_FILES:
        fail(f"artifact file set mismatch: {sorted(names)}")
    for path in root.iterdir():
        reject_symlink(path)
        if not path.is_file():
            fail(f"artifact entry is not a file: {path.name}")


def verify_sums(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
            fail(f"malformed SHA256SUMS line: {line}")
        digest, name = parts
        if name in entries:
            fail(f"duplicate SHA256SUMS entry: {name}")
        reject_path_traversal(name, "SHA256SUMS entry")
        entries[name] = digest
    expected = OUTPUT_FILES - {"SHA256SUMS"}
    if set(entries) != expected:
        fail("SHA256SUMS entry set mismatch")
    for name, expected_digest in entries.items():
        actual = sha256_file(root / name)
        if actual != expected_digest:
            fail(f"SHA256 mismatch for {name}")
    return entries


def validate_rows(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, int]], set[tuple[str, str]], str, str]:
    seen: set[str] = set()
    counts = {cohort: {"positive": 0, "no_event": 0, "total": 0} for cohort in FORMAL_COUNTS}
    suite_tasks: set[tuple[str, str]] = set()
    builder_git_sha = ""
    builder_sha256 = ""
    for row in rows:
        episode = row["episode_key"]
        if episode in seen:
            fail(f"duplicate episode_key: {episode}")
        seen.add(episode)
        reject_path_traversal(row["source_path"], "source_path")
        if not SHA256_RE.fullmatch(row["source_sha256"]):
            fail(f"{episode}: source_sha256 must be 64 lowercase hex")
        if not GIT_SHA_RE.fullmatch(row["builder_git_sha"]):
            fail(f"{episode}: builder_git_sha must be 40 lowercase hex")
        if not SHA256_RE.fullmatch(row["builder_sha256"]):
            fail(f"{episode}: builder_sha256 must be 64 lowercase hex")
        builder_git_sha = builder_git_sha or row["builder_git_sha"]
        builder_sha256 = builder_sha256 or row["builder_sha256"]
        if row["builder_git_sha"] != builder_git_sha or row["builder_sha256"] != builder_sha256:
            fail("builder identity is not uniform across label rows")

        clean_success = parse_bool(row["clean_success"], "clean_success", episode)
        mechanism_eligible = parse_bool(row["mechanism_eligible"], "mechanism_eligible", episode)
        event_present = parse_bool(row["event_present"], "event_present", episode)
        window_valid = parse_bool(row["window_valid"], "window_valid", episode)
        confidence_available = parse_bool(row["confidence_available"], "confidence_available", episode)
        if row["cohort_class"] not in counts:
            fail(f"{episode}: unknown cohort_class")
        if row["cohort_class"] == "PRIMARY_SUCCESS_ELIGIBLE" and (not clean_success or not mechanism_eligible):
            fail(f"{episode}: PRIMARY_SUCCESS_ELIGIBLE invariant failed")
        if row["cohort_class"] == "ELIGIBLE_CLEAN_FAILURE" and (clean_success or not mechanism_eligible):
            fail(f"{episode}: ELIGIBLE_CLEAN_FAILURE invariant failed")
        if row["cohort_class"] == "MECHANISM_INELIGIBLE_ABSTENTION" and (mechanism_eligible or event_present):
            fail(f"{episode}: MECHANISM_INELIGIBLE_ABSTENTION invariant failed")
        if row["source_semantics_authority"] != "SOURCE_AVAILABILITY_LEDGER":
            fail(f"{episode}: wrong source semantics authority")
        if row["source_jsonl_check_mode"] != "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ":
            fail(f"{episode}: wrong source JSONL check mode")

        trace_length = parse_int(row["trace_length"], "trace_length", episode)
        if trace_length <= 0:
            fail(f"{episode}: trace_length must be positive")
        anchor = parse_int(row["anchor_absolute_step"], "anchor_absolute_step", episode)
        start = parse_int(row["window_start"], "window_start", episode)
        end = parse_int(row["window_end"], "window_end", episode)
        rank = parse_int(row["event_rank"], "event_rank", episode)
        if event_present:
            if window_valid and not (0 <= start <= anchor < end <= trace_length):
                fail(f"{episode}: event window must be start <= anchor < exclusive end <= trace_length")
            if row["event_id"] == "NO_EVENT" or row["segment_id"] == "NO_EVENT" or rank < 1:
                fail(f"{episode}: event identifiers are inconsistent")
        else:
            if (anchor, start, end) != (-1, -1, -1):
                fail(f"{episode}: no-event coordinates must be -1")
            if row["event_id"] != "NO_EVENT" or row["segment_id"] != "NO_EVENT" or rank != 0:
                fail(f"{episode}: no-event identifiers are inconsistent")
        if row["label_validity_status"] not in {"VALID", "INVALID_WINDOW"}:
            fail(f"{episode}: unknown label_validity_status")
        if (row["label_validity_status"] == "VALID") != window_valid:
            fail(f"{episode}: window_valid/status mismatch")
        if confidence_available and row["teacher_confidence"] == "UNKNOWN":
            fail(f"{episode}: confidence availability mismatch")

        counts[row["cohort_class"]]["total"] += 1
        counts[row["cohort_class"]]["positive" if event_present else "no_event"] += 1
        suite_tasks.add((row["suite"], row["task_id"]))
    return counts, suite_tasks, builder_git_sha, builder_sha256


def validate_manual(rows: list[dict[str, str]], label_rows: list[dict[str, str]], formal: bool) -> None:
    by_episode = {row["episode_key"]: row for row in label_rows}
    by_unit: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        episode = row["episode_key"]
        label = by_episode.get(episode)
        if label is None:
            fail(f"manual row references missing episode: {episode}")
        for field in ["suite", "task_id", "cohort_class", "clean_success", "mechanism_eligible", "event_present", "label_validity_status"]:
            if row[field] != label[field]:
                fail(f"{episode}: manual {field} mismatch")
        if row["requested_priority"] not in MANUAL_CATEGORIES or row["actual_selected_category"] not in MANUAL_CATEGORIES:
            fail(f"{episode}: unknown manual category")
        actual = row_category(label)
        if row["actual_selected_category"] != actual:
            fail(f"{episode}: manual actual category mismatch")
        fallback = parse_bool(row["fallback_used"], "fallback_used", episode)
        if fallback != (row["requested_priority"] != row["actual_selected_category"]):
            fail(f"{episode}: manual fallback flag mismatch")
        if fallback and not row["fallback_reason"]:
            fail(f"{episode}: manual fallback reason is required")
        if not fallback and row["fallback_reason"]:
            fail(f"{episode}: manual fallback reason must be empty")
        unit = (row["suite"], row["task_id"])
        by_unit.setdefault(unit, set())
        if episode in by_unit[unit]:
            fail(f"{episode}: duplicate manual episode within suite-task unit")
        by_unit[unit].add(episode)
    if formal:
        if len(rows) != 160:
            fail("formal manual sample must contain 160 rows")
        if len(by_unit) != 40 or any(len(episodes) != 4 for episodes in by_unit.values()):
            fail("formal manual sample must contain 40 suite-task units with four rows each")


def validate_manifest(
    manifest: dict[str, object],
    *,
    expected_mode: str,
    expected_builder_git_sha: str | None,
    expected_builder_sha256: str | None,
    builder_git_sha: str,
    builder_sha256: str,
) -> None:
    required = {
        "mode": expected_mode,
        "builder_git_sha": builder_git_sha,
        "builder_sha256": builder_sha256,
        "source_semantics_authority": "SOURCE_AVAILABILITY_LEDGER",
        "source_jsonl_check_mode": "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ",
        "atomic_publish": expected_mode == "formal-ledger-build",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            fail(f"build_manifest.json mismatch: {key}")
    if manifest.get("synthetic_only") != (expected_mode == "synthetic-dry-run"):
        fail("build_manifest.json synthetic_only mismatch")
    if manifest.get("outputs") != sorted(OUTPUT_FILES):
        fail("build_manifest.json outputs mismatch")
    if expected_builder_git_sha and expected_builder_git_sha != builder_git_sha:
        fail("builder git SHA mismatch")
    if expected_builder_sha256 and expected_builder_sha256 != builder_sha256:
        fail("builder file SHA256 mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"source_manifest", "episode_census", "source_crosstab"}:
        fail("build_manifest.json inputs mismatch")
    for name, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            fail(f"build_manifest.json malformed input record: {name}")
        if not isinstance(record["path"], str) or not record["path"]:
            fail(f"build_manifest.json missing input path: {name}")
        reject_path_traversal(record["path"], f"input path {name}")
        if not isinstance(record["sha256"], str) or not SHA256_RE.fullmatch(record["sha256"]):
            fail(f"build_manifest.json malformed input SHA256: {name}")


def validate_summary(
    summary: dict[str, object],
    *,
    expected_mode: str,
    row_count: int,
    counts: dict[str, dict[str, int]],
    manual_count: int,
) -> None:
    if summary.get("status") != "PASS" or summary.get("mode") != expected_mode:
        fail("validation_summary.json status/mode mismatch")
    if summary.get("row_count") != row_count or summary.get("counts") != counts:
        fail("validation_summary.json recomputed closure mismatch")
    if summary.get("manual_audit_sample_n") != manual_count:
        fail("validation_summary.json manual sample mismatch")
    if summary.get("unexplained_disposition_rows") != 0:
        fail("validation_summary.json unexplained disposition mismatch")


def validate_label_v2_artifact(
    artifact_root: str | Path,
    *,
    expected_mode: str,
    expected_builder_git_sha: str | None = None,
    expected_builder_sha256: str | None = None,
) -> dict[str, object]:
    root = Path(artifact_root)
    if expected_mode not in {"synthetic-dry-run", "formal-ledger-build"}:
        fail(f"unsupported expected_mode: {expected_mode}")
    verify_file_set(root)
    verify_sums(root)
    rows = read_csv_strict(
        root / "label_v2.csv",
        LABEL_COLUMNS,
        {"event_source", "invalid_reason", "abstain_reason", "manual_audit_reason"},
    )
    manual = read_csv_strict(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, {"fallback_reason"})
    counts, suite_tasks, builder_git_sha, builder_sha256 = validate_rows(rows)
    formal = expected_mode == "formal-ledger-build"
    if formal:
        if len(rows) != 2000:
            fail("formal artifact must contain exactly 2000 rows")
        if counts != FORMAL_COUNTS:
            fail("formal cohort counts mismatch")
        if len(suite_tasks) != 40:
            fail("formal artifact must contain 40 suite-task units")
    validate_manual(manual, rows, formal)
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        expected_mode=expected_mode,
        expected_builder_git_sha=expected_builder_git_sha,
        expected_builder_sha256=expected_builder_sha256,
        builder_git_sha=builder_git_sha,
        builder_sha256=builder_sha256,
    )
    validate_summary(
        summary,
        expected_mode=expected_mode,
        row_count=len(rows),
        counts=counts,
        manual_count=len(manual),
    )
    return {
        "status": "PASS",
        "five_file_internal_closure": "PASS",
        "source_ledger_reverification": "NOT_PERFORMED_BY_THIS_LOADER",
        "source_jsonl_runtime_read": "NOT_PERFORMED",
        "mode": expected_mode,
        "row_count": len(rows),
        "counts": counts,
        "suite_task_units": len(suite_tasks),
        "manual_audit_sample_n": len(manual),
        "builder_git_sha": builder_git_sha,
        "builder_sha256": builder_sha256,
    }


def load_label_v2_artifact(
    artifact_root: str | Path,
    *,
    expected_mode: str,
    expected_builder_git_sha: str | None = None,
    expected_builder_sha256: str | None = None,
) -> dict[str, object]:
    report = validate_label_v2_artifact(
        artifact_root,
        expected_mode=expected_mode,
        expected_builder_git_sha=expected_builder_git_sha,
        expected_builder_sha256=expected_builder_sha256,
    )
    root = Path(artifact_root)
    return {
        "report": report,
        "label_rows": read_csv_strict(
            root / "label_v2.csv",
            LABEL_COLUMNS,
            {"event_source", "invalid_reason", "abstain_reason", "manual_audit_reason"},
        ),
        "manual_audit_rows": read_csv_strict(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, {"fallback_reason"}),
        "manifest": json.loads((root / "build_manifest.json").read_text(encoding="utf-8")),
        "validation_summary": json.loads((root / "validation_summary.json").read_text(encoding="utf-8")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-mode", required=True, choices=["synthetic-dry-run", "formal-ledger-build"])
    parser.add_argument("--expected-builder-git-sha")
    parser.add_argument("--expected-builder-sha256")
    args = parser.parse_args(argv)
    try:
        report = validate_label_v2_artifact(
            args.artifact_root,
            expected_mode=args.expected_mode,
            expected_builder_git_sha=args.expected_builder_git_sha,
            expected_builder_sha256=args.expected_builder_sha256,
        )
    except LabelV2ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
