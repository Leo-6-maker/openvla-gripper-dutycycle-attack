import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.multisuite_detector.load_label_v2_artifact import (
    LABEL_COLUMNS,
    MANUAL_COLUMNS,
    FORMAL_COUNTS,
    LabelV2ArtifactError,
    load_label_v2_artifact,
    validate_label_v2_artifact,
)


LOADER = ROOT / "tools" / "multisuite_detector" / "load_label_v2_artifact.py"
BUILDER_GIT = "a" * 40
BUILDER_SHA = "b" * 64
SOURCE_SHA = "c" * 64


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def counts_for(rows):
    counts = {cohort: {"positive": 0, "no_event": 0, "total": 0} for cohort in FORMAL_COUNTS}
    for row in rows:
        bucket = counts[row["cohort_class"]]
        bucket["total"] += 1
        bucket["positive" if row["event_present"] == "true" else "no_event"] += 1
    return counts


def label_row(suite, task, idx, cohort, event_present):
    clean_success = cohort != "ELIGIBLE_CLEAN_FAILURE"
    mechanism_eligible = cohort != "MECHANISM_INELIGIBLE_ABSTENTION"
    episode = f"{suite}_{task}_{idx:04d}"
    event = event_present == "true"
    return {
        "episode_key": episode,
        "parent_key": f"{suite}_{task}_parent_{idx:04d}",
        "suite": suite,
        "task_id": task,
        "cohort_class": cohort,
        "clean_success": "true" if clean_success else "false",
        "mechanism_eligible": "true" if mechanism_eligible else "false",
        "event_present": event_present,
        "anchor_absolute_step": "10" if event else "-1",
        "window_start": "5" if event else "-1",
        "window_end": "11" if event else "-1",
        "event_source": "source_availability" if event else "",
        "source_path": f"ledger/{episode}.jsonl",
        "source_sha256": SOURCE_SHA,
        "builder_git_sha": BUILDER_GIT,
        "builder_sha256": BUILDER_SHA,
        "invalid_reason": "",
        "abstain_reason": "MECHANISM_INELIGIBLE" if not mechanism_eligible else "",
        "mechanism_type": "GRIPPER_TRANSFER_ELIGIBLE" if mechanism_eligible else "MECHANISM_UNSUPPORTED",
        "event_id": f"{episode}#event_1" if event else "NO_EVENT",
        "segment_id": f"{episode}#segment_1" if event else "NO_EVENT",
        "event_rank": "1" if event else "0",
        "coordinate_semantics": "zero_based_observation_before_action_start_inclusive_end_exclusive_full_trajectory",
        "trace_length": "20",
        "source_schema_version": "source_availability_ledger_v1",
        "teacher_confidence": "0.9" if event else "UNKNOWN",
        "confidence_available": "true" if event else "false",
        "confidence_provenance": "SOURCE_AVAILABILITY" if event else "UNAVAILABLE",
        "event_id_provenance": "SOURCE_AVAILABILITY" if event else "NOT_APPLICABLE",
        "source_semantics_authority": "SOURCE_AVAILABILITY_LEDGER",
        "source_jsonl_check_mode": "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ",
        "window_valid": "true",
        "label_validity_status": "VALID",
        "manual_audit_status": "PENDING",
        "manual_audit_reason": "",
    }


def make_rows():
    suites = ["Object", "Spatial", "Goal", "LIBERO_10"]
    tasks = [f"task_{i:02d}" for i in range(10)]
    units = [(suite, task) for suite in suites for task in tasks]
    rows = []
    remaining = {
        ("PRIMARY_SUCCESS_ELIGIBLE", "true"): 772,
        ("PRIMARY_SUCCESS_ELIGIBLE", "false"): 271,
        ("ELIGIBLE_CLEAN_FAILURE", "true"): 31,
        ("ELIGIBLE_CLEAN_FAILURE", "false"): 276,
        ("MECHANISM_INELIGIBLE_ABSTENTION", "false"): 650,
    }
    base = [
        ("PRIMARY_SUCCESS_ELIGIBLE", "true"),
        ("PRIMARY_SUCCESS_ELIGIBLE", "false"),
        ("ELIGIBLE_CLEAN_FAILURE", "false"),
        ("MECHANISM_INELIGIBLE_ABSTENTION", "false"),
    ]
    counters = {unit: 0 for unit in units}
    for unit in units:
        for cohort, event_present in base:
            rows.append(label_row(*unit, counters[unit], cohort, event_present))
            counters[unit] += 1
            remaining[(cohort, event_present)] -= 1
    cursor = 0
    for (cohort, event_present), count in remaining.items():
        for _ in range(count):
            unit = units[cursor % len(units)]
            rows.append(label_row(*unit, counters[unit], cohort, event_present))
            counters[unit] += 1
            cursor += 1
    return sorted(rows, key=lambda row: row["episode_key"])


def row_category(row):
    if row["mechanism_eligible"] == "false":
        return "abstention_or_ineligible"
    if row["clean_success"] == "false" or row["label_validity_status"] != "VALID":
        return "failure_or_boundary"
    if row["event_present"] == "true":
        return "positive_clean_success"
    return "eligible_no_event"


def make_manual(rows):
    by_unit = {}
    for row in rows:
        by_unit.setdefault((row["suite"], row["task_id"]), []).append(row)
    out = []
    for suite, task in sorted(by_unit):
        used = set()
        for category in [
            "positive_clean_success",
            "eligible_no_event",
            "failure_or_boundary",
            "abstention_or_ineligible",
        ]:
            row = next(r for r in by_unit[(suite, task)] if r["episode_key"] not in used and row_category(r) == category)
            used.add(row["episode_key"])
            out.append({
                "suite": suite,
                "task_id": task,
                "episode_key": row["episode_key"],
                "cohort_class": row["cohort_class"],
                "clean_success": row["clean_success"],
                "mechanism_eligible": row["mechanism_eligible"],
                "event_present": row["event_present"],
                "label_validity_status": row["label_validity_status"],
                "requested_priority": category,
                "actual_selected_category": category,
                "fallback_used": "false",
                "fallback_reason": "",
                "sampling_seed": "20260703",
            })
    return out


def write_sums(root):
    lines = []
    for name in ["label_v2.csv", "build_manifest.json", "validation_summary.json", "manual_audit_sample_manifest.csv"]:
        lines.append(f"{sha256(root / name)}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_artifact(tmp_path, *, mode="formal-ledger-build"):
    root = tmp_path / "artifact"
    root.mkdir(parents=True)
    rows = make_rows()
    if mode == "synthetic-dry-run":
        rows = rows[:8]
    manual = make_manual(rows) if mode == "formal-ledger-build" else make_manual(rows)[:4]
    counts = counts_for(rows)
    write_csv(root / "label_v2.csv", LABEL_COLUMNS, rows)
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, manual)
    (root / "validation_summary.json").write_text(json.dumps({
        "status": "PASS",
        "mode": mode,
        "row_count": len(rows),
        "counts": counts,
        "invalid_window_rows": 0,
        "manual_audit_sample_n": len(manual),
        "unexplained_disposition_rows": 0,
    }, sort_keys=True) + "\n", encoding="utf-8")
    (root / "build_manifest.json").write_text(json.dumps({
        "schema_version": "clean2000_label_v2_episode_primary_event_v1",
        "created_at_utc": "2026-07-04T00:00:00+00:00",
        "mode": mode,
        "synthetic_only": mode == "synthetic-dry-run",
        "builder_git_sha": BUILDER_GIT,
        "builder_sha256": BUILDER_SHA,
        "source_semantics_authority": "SOURCE_AVAILABILITY_LEDGER",
        "source_jsonl_check_mode": "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ",
        "manual_fallback_policy": {},
        "formal_output_root": str(root),
        "atomic_publish": mode == "formal-ledger-build",
        "inputs": {
            "source_manifest": {"path": "ledger/source_manifest.csv", "sha256": "1" * 64},
            "episode_census": {"path": "ledger/episode_census.csv", "sha256": "2" * 64},
            "source_crosstab": {"path": "ledger/source_event_crosstab.csv", "sha256": "3" * 64},
        },
        "outputs": sorted({
            "label_v2.csv",
            "build_manifest.json",
            "validation_summary.json",
            "manual_audit_sample_manifest.csv",
            "SHA256SUMS",
        }),
    }, sort_keys=True) + "\n", encoding="utf-8")
    write_sums(root)
    return root


def expect_fail(root, text=None):
    with pytest.raises(LabelV2ArtifactError) as exc:
        validate_label_v2_artifact(root, expected_mode="formal-ledger-build")
    if text:
        assert text in str(exc.value)


def mutate_csv(path, mutate):
    rows = read_csv(path)
    mutate(rows)
    write_csv(path, rows[0].keys(), rows)


def test_valid_formal_artifact_loads(tmp_path):
    root = make_artifact(tmp_path)
    loaded = load_label_v2_artifact(root, expected_mode="formal-ledger-build")
    assert loaded["report"]["five_file_internal_closure"] == "PASS"
    assert loaded["report"]["source_ledger_reverification"] == "NOT_PERFORMED_BY_THIS_LOADER"
    assert loaded["report"]["source_jsonl_runtime_read"] == "NOT_PERFORMED"
    assert loaded["report"]["row_count"] == 2000
    assert loaded["report"]["counts"] == FORMAL_COUNTS
    assert len(loaded["manual_audit_rows"]) == 160


def test_valid_synthetic_artifact_loads(tmp_path):
    root = make_artifact(tmp_path, mode="synthetic-dry-run")
    report = validate_label_v2_artifact(root, expected_mode="synthetic-dry-run")
    assert report["status"] == "PASS"
    assert report["mode"] == "synthetic-dry-run"
    assert report["row_count"] == 8


@pytest.mark.parametrize("name", ["label_v2.csv", "build_manifest.json", "validation_summary.json", "manual_audit_sample_manifest.csv", "SHA256SUMS"])
def test_rejects_missing_files(tmp_path, name):
    root = make_artifact(tmp_path)
    (root / name).unlink()
    expect_fail(root, "file set")


def test_rejects_extra_file(tmp_path):
    root = make_artifact(tmp_path)
    (root / ".extra").write_text("x", encoding="utf-8")
    expect_fail(root, "file set")


def test_rejects_sha_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    (root / "label_v2.csv").write_text("corrupt\n", encoding="utf-8")
    expect_fail(root, "SHA256 mismatch")


def test_rejects_duplicate_sums_entry(tmp_path):
    root = make_artifact(tmp_path)
    with (root / "SHA256SUMS").open("a", encoding="utf-8") as handle:
        handle.write(f"{'0' * 64}  label_v2.csv\n")
    expect_fail(root, "duplicate")


def test_rejects_malformed_and_missing_sums_entries(tmp_path):
    root = make_artifact(tmp_path)
    (root / "SHA256SUMS").write_text("not-a-sha  label_v2.csv\n", encoding="utf-8")
    expect_fail(root, "malformed SHA256SUMS")

    root = make_artifact(tmp_path / "missing")
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    (root / "SHA256SUMS").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    expect_fail(root, "entry set")


def test_rejects_header_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "label_v2.csv")
    write_csv(root / "label_v2.csv", LABEL_COLUMNS[:-1], [{k: r[k] for k in LABEL_COLUMNS[:-1]} for r in rows])
    write_sums(root)
    expect_fail(root, "header")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("clean_success", "True", "lowercase"),
        ("window_end", "10", "exclusive end"),
        ("source_sha256", "BAD", "source_sha256"),
    ],
)
def test_rejects_bad_field_encoding_and_window(tmp_path, field, value, message):
    root = make_artifact(tmp_path)
    mutate_csv(root / "label_v2.csv", lambda rows: rows[0].update({field: value}))
    write_sums(root)
    expect_fail(root, message)


def test_rejects_duplicate_episode(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "label_v2.csv")
    rows[1]["episode_key"] = rows[0]["episode_key"]
    write_csv(root / "label_v2.csv", LABEL_COLUMNS, rows)
    write_sums(root)
    expect_fail(root, "duplicate episode")


def test_rejects_mode_and_builder_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    with pytest.raises(LabelV2ArtifactError):
        validate_label_v2_artifact(root, expected_mode="synthetic-dry-run")
    with pytest.raises(LabelV2ArtifactError):
        validate_label_v2_artifact(root, expected_mode="formal-ledger-build", expected_builder_sha256="d" * 64)


def test_rejects_schema_version_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "wrong"
    (root / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_sums(root)
    expect_fail(root, "schema_version")


def test_rejects_summary_crosstab_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    summary["counts"]["PRIMARY_SUCCESS_ELIGIBLE"]["positive"] = 1
    (root / "validation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_sums(root)
    expect_fail(root, "summary")


def test_rejects_invalid_window_count_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    summary["invalid_window_rows"] = 1
    (root / "validation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_sums(root)
    expect_fail(root, "invalid_window_rows")


def test_rejects_formal_exact_count_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "label_v2.csv")[:-1]
    write_csv(root / "label_v2.csv", LABEL_COLUMNS, rows)
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    summary["row_count"] = len(rows)
    summary["counts"] = counts_for(rows)
    (root / "validation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_sums(root)
    expect_fail(root, "2000")


def test_rejects_no_event_coordinates(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "label_v2.csv")
    target = next(row for row in rows if row["event_present"] == "false")
    target["window_start"] = "0"
    write_csv(root / "label_v2.csv", LABEL_COLUMNS, rows)
    write_sums(root)
    expect_fail(root, "no-event coordinates")


def test_rejects_event_rank_other_than_one(tmp_path):
    root = make_artifact(tmp_path)
    mutate_csv(root / "label_v2.csv", lambda rows: rows[0].update({"event_rank": "2"}))
    write_sums(root)
    expect_fail(root, "event identifiers")


@pytest.mark.parametrize(
    "confidence,available,provenance",
    [
        ("0.9", "false", "UNAVAILABLE"),
        ("UNKNOWN", "true", "SOURCE_AVAILABILITY"),
        ("nan", "true", "SOURCE_AVAILABILITY"),
    ],
)
def test_rejects_confidence_availability_mismatch(tmp_path, confidence, available, provenance):
    root = make_artifact(tmp_path)
    mutate_csv(
        root / "label_v2.csv",
        lambda rows: rows[0].update({
            "teacher_confidence": confidence,
            "confidence_available": available,
            "confidence_provenance": provenance,
        }),
    )
    write_sums(root)
    expect_fail(root, "confidence")


def test_rejects_pending_manual_audit_reason(tmp_path):
    root = make_artifact(tmp_path)
    mutate_csv(root / "label_v2.csv", lambda rows: rows[0].update({"manual_audit_reason": "already reviewed"}))
    write_sums(root)
    expect_fail(root, "manual_audit_reason")


def test_rejects_manual_missing_and_context_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "manual_audit_sample_manifest.csv")
    rows[0]["episode_key"] = "missing"
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, rows)
    write_sums(root)
    expect_fail(root, "missing episode")

    root = make_artifact(tmp_path / "second")
    rows = read_csv(root / "manual_audit_sample_manifest.csv")
    rows[0]["suite"] = "Wrong"
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, rows)
    write_sums(root)
    expect_fail(root, "manual suite mismatch")


def test_rejects_manual_duplicate_and_quota_mismatch(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "manual_audit_sample_manifest.csv")
    rows[1] = dict(rows[0])
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, rows)
    write_sums(root)
    expect_fail(root, "duplicate manual episode")

    root = make_artifact(tmp_path / "quota")
    rows = read_csv(root / "manual_audit_sample_manifest.csv")[:-1]
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, rows)
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    summary["manual_audit_sample_n"] = len(rows)
    (root / "validation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_sums(root)
    expect_fail(root, "160")


def test_rejects_manual_duplicate_requested_priority_within_unit(tmp_path):
    root = make_artifact(tmp_path)
    rows = read_csv(root / "manual_audit_sample_manifest.csv")
    rows[1]["requested_priority"] = rows[0]["requested_priority"]
    rows[1]["fallback_used"] = "true"
    rows[1]["fallback_reason"] = "forced_duplicate_requested_priority"
    write_csv(root / "manual_audit_sample_manifest.csv", MANUAL_COLUMNS, rows)
    write_sums(root)
    expect_fail(root, "duplicate manual requested_priority")


def test_rejects_wrong_semantics_authority_or_jsonl_mode(tmp_path):
    root = make_artifact(tmp_path)
    mutate_csv(root / "label_v2.csv", lambda rows: rows[0].update({"source_jsonl_check_mode": "RUNTIME_READ"}))
    write_sums(root)
    expect_fail(root, "JSONL")

    root = make_artifact(tmp_path / "manifest")
    manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))
    manifest["source_semantics_authority"] = "OTHER"
    (root / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_sums(root)
    expect_fail(root, "source_semantics_authority")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("event_source", "source_availability", "no-event event_source"),
        ("mechanism_type", "MECHANISM_UNSUPPORTED", "mechanism_type"),
        ("event_id_provenance", "SOURCE_AVAILABILITY", "no-event event_id_provenance"),
        ("confidence_provenance", "SOURCE_AVAILABILITY", "unavailable confidence_provenance"),
        ("invalid_reason", "BAD_REASON", "invalid_reason"),
        ("coordinate_semantics", "wrong", "coordinate_semantics"),
        ("source_schema_version", "wrong", "source_schema_version"),
        ("manual_audit_status", "DONE", "manual_audit_status"),
    ],
)
def test_rejects_disposition_semantic_field_tamper(tmp_path, field, value, message):
    root = make_artifact(tmp_path)
    def change(rows):
        target = next(row for row in rows if row["event_present"] == "false" and row["mechanism_eligible"] == "true")
        target[field] = value
    mutate_csv(root / "label_v2.csv", change)
    write_sums(root)
    expect_fail(root, message)


def test_cli_reports_malformed_json_without_traceback(tmp_path):
    root = make_artifact(tmp_path)
    (root / "build_manifest.json").write_text("{bad", encoding="utf-8")
    write_sums(root)
    result = subprocess.run(
        [sys.executable, str(LOADER), "--artifact-root", str(root), "--expected-mode", "formal-ledger-build"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_rejects_root_symlink(tmp_path):
    root = make_artifact(tmp_path)
    link = tmp_path / "artifact_link"
    try:
        os.symlink(root, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    expect_fail(link, "symlink")


def test_rejects_child_symlink(tmp_path):
    root = make_artifact(tmp_path)
    (root / "label_v2.csv").unlink()
    try:
        os.symlink(root / "build_manifest.json", root / "label_v2.csv")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    expect_fail(root, "symlink")


def test_cli_prints_json_and_does_not_write(tmp_path):
    root = make_artifact(tmp_path)
    before = sorted(p.name for p in root.iterdir())
    result = subprocess.run(
        [
            sys.executable,
            str(LOADER),
            "--artifact-root",
            str(root),
            "--expected-mode",
            "formal-ledger-build",
            "--expected-builder-git-sha",
            BUILDER_GIT,
            "--expected-builder-sha256",
            BUILDER_SHA,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert result.stderr == ""
    assert sorted(p.name for p in root.iterdir()) == before
