#!/usr/bin/env python3
"""Hardened CLEAN2000 Label V2 builder and independent validator.

The reviewed implementation body is loaded from an immutable parent commit.
This entrypoint applies the final Gate-A1 closeout rules, stages formal outputs
atomically, and provides an input-plus-output validator. Formal execution still
requires a separate authorization record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path


_CORE_COMMIT = "35e17855c57277f866142c34129425e0259ece5b"
_CORE_REPO_PATH = "tools/multisuite_detector/build_clean2000_label_v2.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "clean2000_label_v2_episode_primary_event_v1"
GATE_A1_INPUT_SHA256 = {
    "source_manifest": "268ec095aae19a5aca62141b162c0719706b885c96c84122174fe425493426e4",
    "episode_census": "6d3696465f3e09cd736677f25ac57d83135774229bd75c5a17b38801c7e956ba",
    "source_crosstab": "0b78c0749cdf4a17c93ce28859094c0733741f9892f0b9493894bece26cb25a1",
}

try:
    _CORE_SOURCE = subprocess.check_output(
        ["git", "show", f"{_CORE_COMMIT}:{_CORE_REPO_PATH}"],
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception as exc:
    raise RuntimeError(
        f"unable to load immutable Label V2 core {_CORE_COMMIT}:{_CORE_REPO_PATH}"
    ) from exc

_core = types.ModuleType("_label_v2_core")
_core.__file__ = __file__
exec(compile(_CORE_SOURCE, __file__, "exec"), _core.__dict__)

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


MANUAL_FALLBACK_ORDER = {
    "positive_clean_success": [
        "positive_clean_success",
        "eligible_no_event",
        "failure_or_boundary",
        "abstention_or_ineligible",
    ],
    "eligible_no_event": [
        "eligible_no_event",
        "failure_or_boundary",
        "abstention_or_ineligible",
        "positive_clean_success",
    ],
    "failure_or_boundary": [
        "failure_or_boundary",
        "abstention_or_ineligible",
        "positive_clean_success",
        "eligible_no_event",
    ],
    "abstention_or_ineligible": [
        "abstention_or_ineligible",
        "positive_clean_success",
        "eligible_no_event",
        "failure_or_boundary",
    ],
}
_core.MANUAL_FALLBACK_ORDER = MANUAL_FALLBACK_ORDER

_original_validate_source_disposition_invariants = _core.validate_source_disposition_invariants


def validate_source_disposition_invariants(
    availability: dict[str, str],
    census: dict[str, str],
    clean_success: bool,
    event_present: bool,
) -> None:
    """Enforce a total, cohort-specific source disposition partition."""

    _original_validate_source_disposition_invariants(
        availability,
        census,
        clean_success,
        event_present,
    )
    episode = census["episode_key"]
    cohort = census["cohort_class"]
    source_no_event = _core.bool_field(availability, "source_no_event")
    explicit_abstention = _core.bool_field(
        availability,
        "source_explicit_abstention",
    )
    clean_failure_no_event = _core.bool_field(
        availability,
        "source_clean_failure_no_event",
    )

    if event_present:
        if source_no_event or explicit_abstention or clean_failure_no_event:
            _core.fail(f"{episode}: positive event has a no-event disposition flag")
        return

    if cohort == "PRIMARY_SUCCESS_ELIGIBLE":
        if not source_no_event or explicit_abstention or clean_failure_no_event:
            _core.fail(
                f"{episode}: primary-success no-event requires the generic "
                "source_no_event disposition only"
            )
        return

    if cohort == "ELIGIBLE_CLEAN_FAILURE":
        if not clean_failure_no_event or explicit_abstention:
            _core.fail(
                f"{episode}: eligible clean-failure no-event requires "
                "source_clean_failure_no_event"
            )
        return

    if cohort == "MECHANISM_INELIGIBLE_ABSTENTION":
        if not explicit_abstention or clean_failure_no_event:
            _core.fail(
                f"{episode}: mechanism-ineligible row requires explicit source "
                "abstention"
            )
        return

    _core.fail(f"{episode}: unsupported disposition cohort: {cohort}")


_core.validate_source_disposition_invariants = validate_source_disposition_invariants


def _arg_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _replace_arg(argv: list[str], flag: str, value: str) -> list[str]:
    updated = list(argv)
    try:
        updated[updated.index(flag) + 1] = value
    except (ValueError, IndexError) as exc:
        raise _core.BuildError(f"missing required argument: {flag}") from exc
    return updated


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _formal_target_preflight(output_root: Path) -> None:
    if not output_root.is_absolute():
        _core.fail("formal-ledger-build output root must be absolute")
    _core.reject_symlink(output_root)
    if _is_within(output_root, _REPO_ROOT):
        _core.fail("formal-ledger-build output root must be outside the git repository")
    if output_root.exists():
        _core.fail("formal-ledger-build output root must not already exist")
    if not output_root.parent.is_dir():
        _core.fail("formal-ledger-build output parent must already exist")


def _formal_disposition_summary(label_path: Path) -> dict[str, object]:
    counts = {
        "primary_success_positive": 0,
        "primary_success_no_event": 0,
        "eligible_clean_failure_positive": 0,
        "eligible_clean_failure_no_event": 0,
        "mechanism_ineligible_abstention": 0,
    }
    with label_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        cohort = row["cohort_class"]
        event_present = row["event_present"] == "true"
        if cohort == "PRIMARY_SUCCESS_ELIGIBLE":
            key = (
                "primary_success_positive"
                if event_present
                else "primary_success_no_event"
            )
        elif cohort == "ELIGIBLE_CLEAN_FAILURE":
            key = (
                "eligible_clean_failure_positive"
                if event_present
                else "eligible_clean_failure_no_event"
            )
        elif cohort == "MECHANISM_INELIGIBLE_ABSTENTION":
            key = "mechanism_ineligible_abstention"
        else:
            _core.fail(f"unexpected cohort in formal output: {cohort}")
        counts[key] += 1
    return {
        "unexplained_disposition_rows": 0,
        "disposition_subtype_counts": counts,
    }


def _postprocess_formal_outputs(output_root: Path, final_output_root: Path) -> None:
    label_path = output_root / "label_v2.csv"
    summary_path = output_root / "validation_summary.json"
    manifest_path = output_root / "build_manifest.json"
    manual_path = output_root / "manual_audit_sample_manifest.csv"
    sums_path = output_root / "SHA256SUMS"

    disposition = _formal_disposition_summary(label_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(disposition)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manual_fallback_policy": MANUAL_FALLBACK_ORDER,
            "source_disposition_closure": disposition,
            "builder_entrypoint": str(Path(__file__).resolve()),
            "core_source_commit": _CORE_COMMIT,
            "formal_output_root": str(final_output_root),
            "atomic_publish": True,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sums = [
        f"{_sha256_file(path)}  {path.name}"
        for path in [label_path, manifest_path, summary_path, manual_path]
    ]
    sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _core.fail(f"invalid JSON artifact: {path}: {exc}")
    if not isinstance(value, dict):
        _core.fail(f"JSON artifact must be an object: {path}")
    return value


def _validate_sums(output_root: Path) -> None:
    expected_names = {
        "label_v2.csv",
        "build_manifest.json",
        "validation_summary.json",
        "manual_audit_sample_manifest.csv",
    }
    lines = (output_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    entries: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            _core.fail(f"malformed SHA256SUMS line: {line}")
        digest, name = parts
        if name in entries:
            _core.fail(f"duplicate SHA256SUMS entry: {name}")
        entries[name] = digest
    if set(entries) != expected_names:
        _core.fail(f"SHA256SUMS file set mismatch: {set(entries)}")
    for name, expected in entries.items():
        actual = _sha256_file(output_root / name)
        if actual != expected:
            _core.fail(f"output SHA256 mismatch for {name}: {actual}")


def _validate_manual_sample(
    manual_rows: list[dict[str, str]],
    label_by_episode: dict[str, dict[str, str]],
) -> None:
    if len(manual_rows) != _core.FORMAL_MANUAL_SAMPLE_N:
        _core.fail("manual audit sample must contain exactly 160 rows")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen: set[str] = set()
    for row in manual_rows:
        episode = row["episode_key"]
        if episode in seen:
            _core.fail(f"manual audit duplicate episode: {episode}")
        seen.add(episode)
        if episode not in label_by_episode:
            _core.fail(f"manual audit episode missing from label table: {episode}")
        label = label_by_episode[episode]
        for field in [
            "suite",
            "task_id",
            "cohort_class",
            "clean_success",
            "mechanism_eligible",
            "event_present",
            "label_validity_status",
        ]:
            if row[field] != label[field]:
                _core.fail(f"manual audit context mismatch for {episode}: {field}")
        requested = row["requested_priority"]
        actual = _core.row_category(label)
        if requested not in _core.MANUAL_PRIORITIES:
            _core.fail(f"unknown manual requested priority: {requested}")
        if row["actual_selected_category"] != actual:
            _core.fail(f"manual actual category mismatch for {episode}")
        if actual not in MANUAL_FALLBACK_ORDER[requested]:
            _core.fail(f"forbidden manual fallback {requested} -> {actual}")
        fallback = requested != actual
        if row["fallback_used"] != ("true" if fallback else "false"):
            _core.fail(f"manual fallback flag mismatch for {episode}")
        expected_reason = "" if not fallback else f"missing_{requested}_used_{actual}"
        if row["fallback_reason"] != expected_reason:
            _core.fail(f"manual fallback reason mismatch for {episode}")
        if row["sampling_seed"] != str(_core.MANUAL_AUDIT_SEED):
            _core.fail(f"manual sampling seed mismatch for {episode}")
        groups.setdefault((row["suite"], row["task_id"]), []).append(row)

    if len(groups) != _core.FORMAL_SUITE_TASK_UNITS:
        _core.fail("manual audit must cover exactly 40 suite-task units")
    expected_requests = set(_core.MANUAL_PRIORITIES)
    for key, rows in groups.items():
        if len(rows) != 4:
            _core.fail(f"manual audit group {key} does not contain four rows")
        if {row["requested_priority"] for row in rows} != expected_requests:
            _core.fail(f"manual audit group {key} has incomplete request quotas")


def _validate_input_output_semantics(
    source: Path,
    census: Path,
    crosstab: Path,
    label_by_episode: dict[str, dict[str, str]],
) -> dict[str, dict[str, int]]:
    availability_rows = _core.read_csv_strict(
        source,
        _core.AVAILABILITY_COLUMNS,
        {"source_event_id", "notes"},
    )
    census_rows = _core.read_csv_strict(
        census,
        _core.EPISODE_CENSUS_COLUMNS,
        {
            "teacher_event_id",
            "abstain_reason",
            "model_split",
            "parent_leakage_status",
            "task_leakage_status",
            "normalization_source_status",
        },
    )
    crosstab_rows = _core.read_csv_strict(crosstab, _core.CROSSTAB_COLUMNS)
    availability = _core.unique_by_episode(availability_rows, "availability")
    census_map = _core.unique_by_episode(census_rows, "census")
    if set(availability) != set(census_map) or set(census_map) != set(label_by_episode):
        _core.fail("validator episode-key set mismatch")

    for episode in sorted(label_by_episode):
        a = availability[episode]
        c = census_map[episode]
        label = label_by_episode[episode]
        clean_success = _core.clean_success_from(c)
        mechanism_eligible = _core.mechanism_eligible_from(c)
        event_present = _core.event_present_from(a)
        validate_source_disposition_invariants(
            a,
            c,
            clean_success,
            event_present,
        )
        _core.validate_cohort(c, clean_success, mechanism_eligible, event_present)
        expected = {
            "suite": c["suite"],
            "task_id": c["task_id"],
            "cohort_class": c["cohort_class"],
            "clean_success": "true" if clean_success else "false",
            "mechanism_eligible": "true" if mechanism_eligible else "false",
            "event_present": "true" if event_present else "false",
            "trace_length": c["n_steps"],
            "source_path": a["source_label_path"],
            "source_sha256": a["source_label_sha256"],
        }
        if event_present:
            expected.update(
                {
                    "anchor_absolute_step": a["source_anchor"],
                    "window_start": a["source_window_start"],
                    "window_end": str(int(a["source_window_end"]) + 1),
                }
            )
        else:
            expected.update(
                {
                    "anchor_absolute_step": "-1",
                    "window_start": "-1",
                    "window_end": "-1",
                }
            )
        for field, value in expected.items():
            if label[field] != value:
                _core.fail(f"validator semantic mismatch for {episode}: {field}")

    counts = _core.validate_counts(list(label_by_episode.values()), crosstab_rows)
    _core.validate_formal_closure(list(label_by_episode.values()), counts)
    return counts


def validate_formal_output(argv: list[str]) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Validate a formal Label V2 artifact")
    parser.add_argument("--mode", choices=["validate-formal-output"], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--episode-census", required=True)
    parser.add_argument("--source-crosstab", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-census-sha256", required=True)
    parser.add_argument("--expected-crosstab-sha256", required=True)
    parser.add_argument("--expected-git-commit-sha", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute() or not output_root.is_dir():
        _core.fail("validator output root must be an absolute directory")
    _core.reject_symlink(output_root)
    if _is_within(output_root, _REPO_ROOT):
        _core.fail("validator output root must be outside the git repository")

    required = {
        "label_v2.csv",
        "build_manifest.json",
        "validation_summary.json",
        "manual_audit_sample_manifest.csv",
        "SHA256SUMS",
    }
    if {path.name for path in output_root.iterdir()} != required:
        _core.fail("formal output file set mismatch")

    inputs = {
        "source_manifest": (Path(args.source_manifest), args.expected_source_sha256),
        "episode_census": (Path(args.episode_census), args.expected_census_sha256),
        "source_crosstab": (Path(args.source_crosstab), args.expected_crosstab_sha256),
    }
    for name, (path, expected) in inputs.items():
        _core.validate_sha_arg(expected, f"expected-{name}-sha256")
        if expected != GATE_A1_INPUT_SHA256[name]:
            _core.fail(f"validator {name} SHA is not the Gate A1 binding")
        if _sha256_file(path) != expected:
            _core.fail(f"validator {name} SHA mismatch")

    _core.validate_git_sha_arg(args.expected_git_commit_sha, "expected-git-commit-sha")
    _core.validate_sha_arg(args.expected_builder_sha256, "expected-builder-sha256")
    _validate_sums(output_root)

    label_rows = _core.read_csv_strict(
        output_root / "label_v2.csv",
        _core.OUTPUT_COLUMNS,
        {"event_source", "invalid_reason", "abstain_reason", "manual_audit_reason"},
    )
    if len(label_rows) != _core.FORMAL_ROW_COUNT:
        _core.fail("formal label table must contain exactly 2000 rows")
    label_by_episode = _core.unique_by_episode(label_rows, "label output")
    for row in label_rows:
        if row["builder_git_sha"] != args.expected_git_commit_sha:
            _core.fail(f"label producer git SHA mismatch: {row['episode_key']}")
        if row["builder_sha256"] != args.expected_builder_sha256:
            _core.fail(f"label producer file SHA mismatch: {row['episode_key']}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]):
            _core.fail(f"invalid source SHA in label row: {row['episode_key']}")

    counts = _validate_input_output_semantics(
        inputs["source_manifest"][0],
        inputs["episode_census"][0],
        inputs["source_crosstab"][0],
        label_by_episode,
    )

    manual_rows = _core.read_csv_strict(
        output_root / "manual_audit_sample_manifest.csv",
        _core.MANUAL_COLUMNS,
        {"fallback_reason"},
    )
    _validate_manual_sample(manual_rows, label_by_episode)

    summary = _read_json(output_root / "validation_summary.json")
    if summary.get("status") != "PASS" or summary.get("row_count") != 2000:
        _core.fail("validation summary status/row count mismatch")
    if summary.get("counts") != counts:
        _core.fail("validation summary crosstab mismatch")
    if summary.get("manual_audit_sample_n") != 160:
        _core.fail("validation summary manual sample mismatch")
    if summary.get("unexplained_disposition_rows") != 0:
        _core.fail("validation summary has unexplained dispositions")

    manifest = _read_json(output_root / "build_manifest.json")
    if manifest.get("mode") != "formal-ledger-build" or manifest.get("synthetic_only") is not False:
        _core.fail("build manifest mode mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _core.fail("build manifest schema version mismatch")
    if manifest.get("builder_git_sha") != args.expected_git_commit_sha:
        _core.fail("build manifest git SHA mismatch")
    if manifest.get("builder_sha256") != args.expected_builder_sha256:
        _core.fail("build manifest builder SHA mismatch")
    if manifest.get("core_source_commit") != _CORE_COMMIT:
        _core.fail("build manifest immutable core mismatch")
    if manifest.get("formal_output_root") != str(output_root):
        _core.fail("build manifest output-root mismatch")
    if manifest.get("manual_fallback_policy") != MANUAL_FALLBACK_ORDER:
        _core.fail("build manifest fallback policy mismatch")
    closure = manifest.get("source_disposition_closure")
    if not isinstance(closure, dict) or closure.get("unexplained_disposition_rows") != 0:
        _core.fail("build manifest disposition closure mismatch")

    report = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "output_root": str(output_root),
        "row_count": len(label_rows),
        "manual_audit_sample_n": len(manual_rows),
        "counts": counts,
        "unexplained_disposition_rows": 0,
        "builder_git_sha": args.expected_git_commit_sha,
        "builder_sha256": args.expected_builder_sha256,
        "validator_git_sha": _core.git_sha(_REPO_ROOT),
        "validator_file_sha256": _sha256_file(Path(__file__).resolve()),
    }
    return report


def _self_test_closeout() -> None:
    categories = set(_core.MANUAL_PRIORITIES)
    if any(set(order) != categories or len(order) != 4 for order in MANUAL_FALLBACK_ORDER.values()):
        _core.fail("manual fallback matrix is not total")

    def make_row(episode: str, category: str) -> dict[str, str]:
        row = {
            "episode_key": episode,
            "suite": "Object",
            "task_id": "0",
            "cohort_class": "PRIMARY_SUCCESS_ELIGIBLE",
            "clean_success": "true",
            "mechanism_eligible": "true",
            "event_present": "false",
            "label_validity_status": "VALID",
        }
        if category == "positive_clean_success":
            row["event_present"] = "true"
        elif category == "failure_or_boundary":
            row.update(cohort_class="ELIGIBLE_CLEAN_FAILURE", clean_success="false")
        elif category == "abstention_or_ineligible":
            row.update(
                cohort_class="MECHANISM_INELIGIBLE_ABSTENTION",
                mechanism_eligible="false",
            )
        return row

    for category in sorted(categories):
        sample = _core.manual_audit_sample(
            [make_row(f"{category}_{index}", category) for index in range(4)],
            expected_n=4,
        )
        if len({row["episode_key"] for row in sample}) != 4:
            _core.fail(f"sparse fallback self-test failed: {category}")


def _run_formal_build(args: list[str]) -> int:
    output_text = _arg_value(args, "--output-root")
    if output_text is None:
        _core.fail("formal-ledger-build requires --output-root")
    output_root = Path(output_text)
    _formal_target_preflight(output_root)
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        _core.fail(f"staging output already exists: {staging}")
    staged_args = _replace_arg(args, "--output-root", str(staging))
    try:
        result = _core.main(staged_args)
        if result != 0:
            return result
        _postprocess_formal_outputs(staging, output_root)
        staging.rename(output_root)
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    mode = _arg_value(args, "--mode")
    try:
        if mode == "validate-formal-output":
            report = validate_formal_output(args)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if mode == "self-test-closeout":
            _self_test_closeout()
            print("Label V2 closeout self-test: PASS")
            return 0
        if mode == "formal-ledger-build":
            return _run_formal_build(args)
        return _core.main(args)
    except _core.BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
