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


SOURCE_COLUMNS = [
    "episode_key",
    "parent_key",
    "suite",
    "task_id",
    "cohort_class",
    "clean_success",
    "mechanism_eligible",
    "event_present",
    "anchor_absolute_step",
    "window_start",
    "window_end",
    "event_source",
    "source_path",
    "source_sha256",
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
    "window_valid",
    "label_validity_status",
    "manual_audit_status",
    "manual_audit_reason",
]

CENSUS_COLUMNS = ["cohort_class", "total"]
CROSSTAB_COLUMNS = ["cohort_class", "source_positive", "source_no_event", "total"]
COORDINATE_SEMANTICS = "zero_based_observation_before_action_start_inclusive_end_exclusive_full_trajectory"
MANUAL_AUDIT_SEED = 20260703
MAX_SYNTHETIC_ROWS = 200
SENTINEL_NAME = ".label_v2_synthetic_fixture.json"
SENTINEL_SHA256 = "dae3e444c0c8693d5a80e20fd0761ddd4d559ae038ef7ecb6cfef054ab69f482"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
    existing = []
    while True:
        existing.append(probe)
        if probe.exists() or probe.parent == probe:
            break
        probe = probe.parent
    for candidate in existing:
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
    if value == "true":
        return True
    if value == "false":
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


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


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
        confidence = parse_float(row["teacher_confidence"], "teacher_confidence", episode)

        if row["source_schema_version"] != "synthetic_v1":
            fail(f"{episode}: source_schema_version must be synthetic_v1")
        if row["coordinate_semantics"] != COORDINATE_SEMANTICS:
            fail(f"{episode}: unexpected coordinate_semantics")
        if not 0.0 <= confidence <= 1.0:
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

        if event_present:
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


def validate_counts(rows: list[dict[str, str]], census_rows: list[dict[str, str]], crosstab_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
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

    census = {}
    for row in census_rows:
        cohort = row["cohort_class"]
        if cohort in census:
            fail(f"duplicate census cohort: {cohort}")
        census[cohort] = parse_int(row["total"], "total", cohort)
    for cohort, bucket in counts.items():
        if census.get(cohort) != bucket["total"]:
            fail(f"census mismatch for {cohort}: expected {census.get(cohort)}, got {bucket['total']}")
    if set(census) != set(counts):
        fail(f"census cohort set mismatch: expected {sorted(census)}, got {sorted(counts)}")

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
    if row["event_present"] == "true" and row["clean_success"] == "true":
        return "positive_clean_success"
    if row["event_present"] == "false" and row["mechanism_eligible"] == "true":
        return "eligible_no_event"
    if row["clean_success"] == "false" or row["label_validity_status"] != "VALID":
        return "failure_or_boundary"
    if row["mechanism_eligible"] == "false":
        return "abstention_or_ineligible"
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
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_sha_arg(value: str, label: str) -> None:
    if not SHA256_RE.fullmatch(value):
        fail(f"{label} must be 64 lowercase hex characters")


def verify_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
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
    return source, census, crosstab, output_root


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
        source, census, crosstab, output_root = verify_inputs(args)
        source_sha = compare_sha(source, args.expected_source_sha256, "source manifest")
        census_sha = compare_sha(census, args.expected_census_sha256, "episode census")
        crosstab_sha = compare_sha(crosstab, args.expected_crosstab_sha256, "source crosstab")

        builder_path = Path(__file__).resolve()
        builder_sha = sha256_file(builder_path)
        repo = builder_path.parents[2]
        git_head = git_sha(repo)

        source_rows = read_csv_strict(source, SOURCE_COLUMNS, {"event_source", "invalid_reason", "abstain_reason"})
        census_rows = read_csv_strict(census, CENSUS_COLUMNS)
        crosstab_rows = read_csv_strict(crosstab, CROSSTAB_COLUMNS)
        output_rows = validate_and_transform(source_rows, builder_sha, git_head)
        counts = validate_counts(source_rows, census_rows, crosstab_rows)

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
