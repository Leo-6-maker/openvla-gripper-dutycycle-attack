import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "multisuite_detector" / "build_clean2000_label_v2.py"
FIXTURE = ROOT / "tests" / "fixtures" / "label_v2_synthetic"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_fixture(tmp_path, name="label_v2_synthetic"):
    target = tmp_path / name
    shutil.copytree(FIXTURE, target)
    return target


def run_builder(fixture_dir, output_root, *extra, expect_ok=True):
    output_base = output_root.parent
    cmd = [
        sys.executable,
        str(BUILDER),
        "--source-manifest",
        str(fixture_dir / "source_manifest.csv"),
        "--episode-census",
        str(fixture_dir / "episode_census.csv"),
        "--source-crosstab",
        str(fixture_dir / "source_event_crosstab.csv"),
        "--output-root",
        str(output_root),
        "--synthetic-fixture-root",
        str(fixture_dir),
        "--synthetic-output-root",
        str(output_base),
        "--expected-source-sha256",
        sha256(fixture_dir / "source_manifest.csv"),
        "--expected-census-sha256",
        sha256(fixture_dir / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture_dir / "source_event_crosstab.csv"),
        "--synthetic",
        "--dry-run",
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok and result.returncode != 0:
        raise AssertionError(result.stderr)
    if not expect_ok and result.returncode == 0:
        raise AssertionError("builder unexpectedly passed")
    return result


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def tracked_worktree_dirty():
    return (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rewrite_source(path, transform):
    rows = read_rows(path)
    rows = transform(rows)
    write_rows(path, rows)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_synthetic_dry_run_outputs(tmp_path):
    fixture = copy_fixture(tmp_path)
    output = tmp_path / "out"

    run_builder(fixture, output)

    expected = {
        "label_v2.csv",
        "build_manifest.json",
        "validation_summary.json",
        "manual_audit_sample_manifest.csv",
        "SHA256SUMS",
    }
    assert expected == {p.name for p in output.iterdir()}
    rows = read_rows(output / "label_v2.csv")
    assert len(rows) == 6
    by_episode = {row["episode_key"]: row for row in rows}
    assert by_episode["ep_valid_positive"]["window_end"] == "14"
    assert by_episode["ep_valid_positive"]["source_sha256"] == sha256(fixture / "source_records.jsonl")
    assert by_episode["ep_valid_positive"]["mechanism_type"] == "GRIPPER_TRANSFER_ELIGIBLE"
    assert by_episode["ep_valid_positive"]["event_id"] == "ep_valid_positive#event_1"
    assert by_episode["ep_valid_positive"]["segment_id"] == "ep_valid_positive#segment_1"
    assert by_episode["ep_valid_positive"]["source_schema_version"] == "source_availability_ledger_v1"
    assert by_episode["ep_valid_positive"]["source_semantics_authority"] == "SOURCE_AVAILABILITY_LEDGER"
    assert by_episode["ep_valid_positive"]["source_jsonl_check_mode"] == "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ"
    assert by_episode["ep_eligible_no_event"]["event_id"] == "NO_EVENT"
    assert by_episode["ep_eligible_no_event"]["teacher_confidence"] == "UNKNOWN"
    assert by_episode["ep_eligible_no_event"]["confidence_available"] == "false"
    assert by_episode["ep_mechanism_ineligible"]["mechanism_type"] == "MECHANISM_UNSUPPORTED"
    assert {r["episode_key"] for r in rows if r["label_validity_status"] == "INVALID_WINDOW"} == {"ep_invalid_window"}
    assert all(r["builder_sha256"] for r in rows)
    manual = read_rows(output / "manual_audit_sample_manifest.csv")
    assert manual
    assert "suite" in manual[0]
    assert {"requested_priority", "actual_selected_category", "fallback_used", "fallback_reason"} <= set(manual[0])
    assert "label_v2.csv" in (output / "SHA256SUMS").read_text(encoding="utf-8")

    manifest = json.loads((output / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_semantics_authority"] == "SOURCE_AVAILABILITY_LEDGER"
    assert manifest["source_jsonl_check_mode"] == "LEDGER_PROVENANCE_ONLY_NO_RUNTIME_READ"
    assert manifest["v2_episode_table_scope"] == "PRIMARY_EVENT_ONLY"
    assert manifest["source_window_end_semantics"] == "INCLUSIVE_CONVERTED_TO_V2_EXCLUSIVE"


def test_trace_length_uses_full_trajectory_not_valid_step_count(tmp_path):
    fixture = copy_fixture(tmp_path)
    availability_rows = read_rows(fixture / "source_manifest.csv")
    census_rows = read_rows(fixture / "episode_census.csv")
    for row in availability_rows:
        if row["episode_key"] == "ep_valid_positive":
            row["source_anchor"] = "109"
            row["source_window_start"] = "109"
            row["source_window_end"] = "118"
    for row in census_rows:
        if row["episode_key"] == "ep_valid_positive":
            row["teacher_anchor_step"] = "109"
            row["teacher_window_start"] = "109"
            row["teacher_window_end"] = "119"
            row["n_steps"] = "175"
            row["n_valid_steps"] = "109"
    write_rows(fixture / "source_manifest.csv", availability_rows)
    write_rows(fixture / "episode_census.csv", census_rows)

    output = tmp_path / "out"
    run_builder(fixture, output)

    row = {r["episode_key"]: r for r in read_rows(output / "label_v2.csv")}["ep_valid_positive"]
    assert row["trace_length"] == "175"
    assert row["window_end"] == "119"
    assert row["label_validity_status"] == "VALID"


def test_missing_source_event_id_gets_episode_scoped_primary_fallback(tmp_path):
    fixture = copy_fixture(tmp_path)
    rewrite_source(
        fixture / "source_manifest.csv",
        lambda rows: [dict(r, source_event_id="UNKNOWN") if r["episode_key"] == "ep_valid_positive" else r for r in rows],
    )

    output = tmp_path / "out"
    run_builder(fixture, output)

    row = {r["episode_key"]: r for r in read_rows(output / "label_v2.csv")}["ep_valid_positive"]
    assert row["event_id"] == "ep_valid_positive#event_1"
    assert row["event_id_provenance"] == "EPISODE_PRIMARY_EVENT_FALLBACK"


@pytest.mark.parametrize(
    "mutator, expected_error",
    [
        (
            lambda p: rewrite_source(p, lambda rows: rows + [dict(rows[0])]),
            "duplicate availability episode_key",
        ),
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8").replace(",source_mechanism_eligible_schema_valid\n", "\n"),
                encoding="utf-8",
            ),
            "expected columns",
        ),
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8").replace(
                    "source_mechanism_eligible_schema_valid\n",
                    "source_mechanism_eligible_schema_valid,extra\n",
                ),
                encoding="utf-8",
            ),
            "expected columns",
        ),
        (
            lambda p: p.write_text(p.read_text(encoding="utf-8").replace(",True\n", ",True,extra\n", 1), encoding="utf-8"),
            "extra cells",
        ),
        (
            lambda p: p.write_text(p.read_text(encoding="utf-8").replace(",True\n", "\n", 1), encoding="utf-8"),
            "missing cells",
        ),
        (
            lambda p: rewrite_source(p, lambda rows: [dict(r, real_source_label_found="maybe") if i == 0 else r for i, r in enumerate(rows)]),
            "illegal bool",
        ),
        (
            lambda p: rewrite_source(
                p,
                lambda rows: [
                    dict(r, source_no_event="True")
                    if i == 0
                    else r
                    for i, r in enumerate(rows)
                ],
            ),
            "positive/no-event source flags conflict",
        ),
    ],
)
def test_fail_closed_source_mutations(tmp_path, mutator, expected_error):
    fixture = copy_fixture(tmp_path)
    mutator(fixture / "source_manifest.csv")

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert expected_error in result.stderr


def test_wrong_input_sha_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    output = tmp_path / "out"
    cmd = [
        sys.executable,
        str(BUILDER),
        "--source-manifest",
        str(fixture / "source_manifest.csv"),
        "--episode-census",
        str(fixture / "episode_census.csv"),
        "--source-crosstab",
        str(fixture / "source_event_crosstab.csv"),
        "--output-root",
        str(output),
        "--synthetic-fixture-root",
        str(fixture),
        "--synthetic-output-root",
        str(output.parent),
        "--expected-source-sha256",
        "0" * 64,
        "--expected-census-sha256",
        sha256(fixture / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture / "source_event_crosstab.csv"),
        "--synthetic",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr


def test_non_empty_output_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("occupied", encoding="utf-8")

    result = run_builder(fixture, output, expect_ok=False)

    assert "output root must be empty" in result.stderr


def test_missing_expected_sha_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source-manifest",
            str(fixture / "source_manifest.csv"),
            "--episode-census",
            str(fixture / "episode_census.csv"),
            "--source-crosstab",
            str(fixture / "source_event_crosstab.csv"),
            "--output-root",
            str(tmp_path / "out"),
            "--synthetic-fixture-root",
            str(fixture),
            "--synthetic-output-root",
            str(tmp_path),
            "--synthetic",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected-source-sha256" in result.stderr


def test_input_outside_declared_fixture_root_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    other = copy_fixture(tmp_path, "other_fixture")

    cmd = [
        sys.executable,
        str(BUILDER),
        "--source-manifest",
        str(other / "source_manifest.csv"),
        "--episode-census",
        str(fixture / "episode_census.csv"),
        "--source-crosstab",
        str(fixture / "source_event_crosstab.csv"),
        "--output-root",
        str(tmp_path / "out"),
        "--synthetic-fixture-root",
        str(fixture),
        "--synthetic-output-root",
        str(tmp_path),
        "--expected-source-sha256",
        sha256(other / "source_manifest.csv"),
        "--expected-census-sha256",
        sha256(fixture / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture / "source_event_crosstab.csv"),
        "--synthetic",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "under declared synthetic root" in result.stderr


def test_missing_sentinel_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    (fixture / ".label_v2_synthetic_fixture.json").unlink()

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert "sentinel missing" in result.stderr


def test_output_outside_declared_root_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    output = tmp_path / "out"
    cmd = [
        sys.executable,
        str(BUILDER),
        "--source-manifest",
        str(fixture / "source_manifest.csv"),
        "--episode-census",
        str(fixture / "episode_census.csv"),
        "--source-crosstab",
        str(fixture / "source_event_crosstab.csv"),
        "--output-root",
        str(output),
        "--synthetic-fixture-root",
        str(fixture),
        "--synthetic-output-root",
        str(tmp_path / "approved"),
        "--expected-source-sha256",
        sha256(fixture / "source_manifest.csv"),
        "--expected-census-sha256",
        sha256(fixture / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture / "source_event_crosstab.csv"),
        "--synthetic",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "output root must be under declared synthetic root" in result.stderr


def test_path_traversal_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)

    result = run_builder(fixture, tmp_path / "nested" / ".." / "out", expect_ok=False)

    assert "path traversal" in result.stderr


def test_crosstab_mismatch_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    crosstab = fixture / "source_event_crosstab.csv"
    crosstab.write_text(crosstab.read_text(encoding="utf-8").replace("PRIMARY_SUCCESS_ELIGIBLE,3,1,4", "PRIMARY_SUCCESS_ELIGIBLE,2,2,4"), encoding="utf-8")

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert "crosstab mismatch" in result.stderr


def test_per_row_source_sha_format_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    rewrite_source(
        fixture / "source_manifest.csv",
        lambda rows: [dict(r, source_label_sha256="not-a-sha") if i == 0 else r for i, r in enumerate(rows)],
    )

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert "source_label_sha256 must be 64 lowercase hex" in result.stderr


def test_cohort_invariant_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    census = fixture / "episode_census.csv"
    rows = read_rows(census)
    rows[0]["outcome_class"] = "CLEAN_FAILURE"
    write_rows(census, rows)

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert "cohort invariant failed" in result.stderr


def test_canonical_invalid_reason_does_not_control_v2_validity(tmp_path):
    fixture = copy_fixture(tmp_path)
    rewrite_source(
        fixture / "source_manifest.csv",
        lambda rows: [
            dict(r, canonical_index_label='{"teacher_invalid_reason":"TRUNCATED_TRACE"}')
            if r["episode_key"] == "ep_valid_positive"
            else r
            for r in rows
        ],
    )

    output = tmp_path / "out"
    run_builder(fixture, output)

    row = {r["episode_key"]: r for r in read_rows(output / "label_v2.csv")}["ep_valid_positive"]
    assert row["invalid_reason"] == ""
    assert row["label_validity_status"] == "VALID"


def test_source_no_event_with_canonical_positive_uses_source_semantics(tmp_path):
    fixture = copy_fixture(tmp_path)
    census = fixture / "episode_census.csv"
    rows = read_rows(census)
    for row in rows:
        if row["episode_key"] == "ep_eligible_no_event":
            row["teacher_positive_label_valid"] = "True"
            row["positive_anchor_valid"] = "True"
            row["teacher_anchor_step"] = "12"
            row["teacher_window_start"] = "10"
            row["teacher_window_end"] = "14"
    write_rows(census, rows)

    output = tmp_path / "out"
    run_builder(fixture, output)

    row = {r["episode_key"]: r for r in read_rows(output / "label_v2.csv")}["ep_eligible_no_event"]
    assert row["event_present"] == "false"
    assert row["event_id"] == "NO_EVENT"
    assert row["label_validity_status"] == "VALID"


def test_duplicate_census_episode_and_crosstab_cohort_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    census = fixture / "episode_census.csv"
    rows = read_rows(census)
    with census.open("a", encoding="utf-8") as f:
        f.write(",".join(rows[0].values()) + "\n")
    result = run_builder(fixture, tmp_path / "out1", expect_ok=False)
    assert "duplicate census episode_key" in result.stderr

    fixture = copy_fixture(tmp_path / "second")
    crosstab = fixture / "source_event_crosstab.csv"
    crosstab.write_text(crosstab.read_text(encoding="utf-8") + "PRIMARY_SUCCESS_ELIGIBLE,3,1,4\n", encoding="utf-8")
    result = run_builder(fixture, tmp_path / "out2", expect_ok=False)
    assert "duplicate crosstab cohort" in result.stderr


def make_160_fixture(tmp_path):
    fixture = tmp_path / "label_v2_synthetic"
    fixture.mkdir()
    shutil.copy(FIXTURE / ".label_v2_synthetic_fixture.json", fixture / ".label_v2_synthetic_fixture.json")
    suites = ["Object", "Spatial", "Goal", "LIBERO-10"]
    suffixes = ["pos", "noevent", "failure", "ineligible"]
    source_records = fixture / "source_records.jsonl"
    source_records.write_text(
        "".join(
            f'{{"episode_key":"{suite}_task_{task_idx:02d}_{suffix}"}}\n'
            for suite in suites
            for task_idx in range(10)
            for suffix in suffixes
        ),
        encoding="utf-8",
    )
    source_record_sha = sha256(source_records)
    availability_rows = []
    census_rows = []
    for suite in suites:
        for task_idx in range(10):
            task = f"task_{task_idx:02d}"
            variants = [
                ("pos", "CLEAN_SUCCESS", "MECHANISM_ELIGIBLE", "PRIMARY_SUCCESS_ELIGIBLE", True, False, False, False, "10", "8", "13", "event_1", ""),
                ("noevent", "CLEAN_SUCCESS", "MECHANISM_ELIGIBLE", "PRIMARY_SUCCESS_ELIGIBLE", False, True, False, False, "-1", "-1", "-1", "", "NO_TEACHER_EVENT"),
                ("failure", "CLEAN_FAILURE", "MECHANISM_ELIGIBLE", "ELIGIBLE_CLEAN_FAILURE", True, False, False, False, "11", "9", "14", "event_2", ""),
                ("ineligible", "CLEAN_SUCCESS", "MECHANISM_INELIGIBLE", "MECHANISM_INELIGIBLE_ABSTENTION", False, False, True, False, "-1", "-1", "-1", "", "UNSUPPORTED_MECHANISM"),
            ]
            for suffix, outcome, scope, cohort, positive, no_event, abstention, clean_failure_no_event, anchor, start, end, event_id, abstain in variants:
                episode = f"{suite}_{task}_{suffix}"
                availability_rows.append({
                    "suite": suite,
                    "task_id": task,
                    "episode_key": episode,
                    "canonical_index_label": '{"teacher_invalid_reason":""}',
                    "real_source_label_found": "True",
                    "source_label_path": "source_records.jsonl",
                    "source_label_sha256": source_record_sha,
                    "source_anchor": anchor,
                    "source_window_start": start,
                    "source_window_end": end,
                    "source_confidence": "0.9" if positive else "UNKNOWN",
                    "source_event_id": event_id,
                    "matches_canonical": "True",
                    "notes": "",
                    "source_record_found": "True",
                    "source_schema_valid": "True",
                    "source_positive_anchor_valid": str(positive),
                    "source_no_event": str(no_event),
                    "source_explicit_abstention": str(abstention),
                    "source_clean_failure_no_event": str(clean_failure_no_event),
                    "shared_fields_comparable": "True",
                    "shared_fields_match": "True",
                    "uncomparable_due_to_missing_fields": "False",
                    "source_timing_fields_present": str(positive),
                    "source_mechanism_eligible_schema_valid": "True",
                })
                census_rows.append({
                    "episode_key": episode,
                    "parent_key": f"{suite}_{task}_{suffix}_parent",
                    "suite": suite,
                    "task_id": task,
                    "task_name": "synthetic_task",
                    "state_id": "0",
                    "outcome_class": outcome,
                    "mechanism_scope_class": scope,
                    "cohort_class": cohort,
                    "label_record_present": "True",
                    "record_schema_valid": "True",
                    "teacher_positive_label_valid": str(positive),
                    "positive_anchor_valid": str(positive),
                    "explicit_abstention_valid": str(abstention),
                    "timing_signal_usable": str(positive),
                    "teacher_anchor_step": anchor,
                    "teacher_window_start": start,
                    "teacher_window_end": str(int(end) + 1) if positive else "-1",
                    "teacher_confidence": "0.9" if positive else "0.0",
                    "teacher_event_id": event_id,
                    "abstain_reason": abstain,
                    "feature_schema_sha256": "a" * 64,
                    "source_manifest_sha256": "b" * 64,
                    "artifact_inventory_sha256": "c" * 64,
                    "n_steps": "50",
                    "n_valid_steps": "50",
                    "first_valid_step": "0",
                    "invalid_feature_steps": "0",
                    "feature_25d_join_ok": "True",
                    "cohort_set": "SYNTHETIC",
                    "model_split": "UNKNOWN",
                    "parent_leakage_status": "UNKNOWN",
                    "task_leakage_status": "UNKNOWN",
                    "normalization_source_status": "UNKNOWN",
                })
    with (fixture / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = (FIXTURE / "source_manifest.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(availability_rows)
    with (fixture / "episode_census.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = (FIXTURE / "episode_census.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(census_rows)
    (fixture / "source_event_crosstab.csv").write_text(
        "cohort_class,source_positive,source_no_event,total\nPRIMARY_SUCCESS_ELIGIBLE,40,40,80\nELIGIBLE_CLEAN_FAILURE,40,0,40\nMECHANISM_INELIGIBLE_ABSTENTION,0,40,40\n",
        encoding="utf-8",
    )
    return fixture


def make_formal_fixture(tmp_path):
    fixture = tmp_path / "formal_ledger"
    fixture.mkdir()
    suites = ["Object", "Spatial", "Goal", "LIBERO-10"]
    groups = [(suite, f"task_{task_idx:02d}") for suite in suites for task_idx in range(10)]
    availability_rows = []
    census_rows = []

    def add_row(group_idx, suffix, outcome, scope, cohort, positive, no_event, abstention, clean_failure_no_event):
        suite, task = groups[group_idx % len(groups)]
        idx = len(census_rows)
        episode = f"{suite}_{task}_{suffix}_{idx:04d}"
        anchor = str(10 + idx % 20) if positive else "-1"
        start = str(8 + idx % 20) if positive else "-1"
        end = str(13 + idx % 20) if positive else "-1"
        event_id = f"event_{idx:04d}" if positive else ""
        availability_rows.append({
            "suite": suite,
            "task_id": task,
            "episode_key": episode,
            "canonical_index_label": '{"teacher_invalid_reason":"TRUNCATED_TRACE"}' if idx == 0 else '{"teacher_invalid_reason":""}',
            "real_source_label_found": "True",
            "source_label_path": "CLEAN2000_SUPERVISION_AUTH_V1_2/TEACHER_EVENT_INDEX.jsonl",
            "source_label_sha256": "d" * 64,
            "source_anchor": anchor,
            "source_window_start": start,
            "source_window_end": end,
            "source_confidence": "0.9" if positive else "UNKNOWN",
            "source_event_id": event_id,
            "matches_canonical": "False" if no_event else "True",
            "notes": "",
            "source_record_found": "True",
            "source_schema_valid": "True",
            "source_positive_anchor_valid": str(positive),
            "source_no_event": str(no_event),
            "source_explicit_abstention": str(abstention),
            "source_clean_failure_no_event": str(clean_failure_no_event),
            "shared_fields_comparable": "True",
            "shared_fields_match": "False" if no_event else "True",
            "uncomparable_due_to_missing_fields": "False",
            "source_timing_fields_present": str(positive),
            "source_mechanism_eligible_schema_valid": "True",
        })
        census_rows.append({
            "episode_key": episode,
            "parent_key": f"{suite}_{task}_{suffix}_parent_{idx:04d}",
            "suite": suite,
            "task_id": task,
            "task_name": "formal_synthetic_task",
            "state_id": str(idx),
            "outcome_class": outcome,
            "mechanism_scope_class": scope,
            "cohort_class": cohort,
            "label_record_present": "True",
            "record_schema_valid": "True",
            "teacher_positive_label_valid": "True" if (positive or no_event) else "False",
            "positive_anchor_valid": "True" if (positive or no_event) else "False",
            "explicit_abstention_valid": str(abstention),
            "timing_signal_usable": str(positive),
            "teacher_anchor_step": anchor if positive else "11",
            "teacher_window_start": start if positive else "9",
            "teacher_window_end": str(int(end) + 1) if positive else "14",
            "teacher_confidence": "0.9" if positive else "0.0",
            "teacher_event_id": event_id,
            "abstain_reason": "UNSUPPORTED_MECHANISM" if abstention else ("NO_TEACHER_EVENT" if (no_event or clean_failure_no_event) else ""),
            "feature_schema_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "artifact_inventory_sha256": "c" * 64,
            "n_steps": "50",
            "n_valid_steps": "40",
            "first_valid_step": "0",
            "invalid_feature_steps": "0",
            "feature_25d_join_ok": "True",
            "cohort_set": "FORMAL_SYNTHETIC",
            "model_split": "UNKNOWN",
            "parent_leakage_status": "UNKNOWN",
            "task_leakage_status": "UNKNOWN",
            "normalization_source_status": "UNKNOWN",
        })

    baseline = [
        ("primary_pos", "CLEAN_SUCCESS", "MECHANISM_ELIGIBLE", "PRIMARY_SUCCESS_ELIGIBLE", True, False, False, False),
        ("primary_noevent", "CLEAN_SUCCESS", "MECHANISM_ELIGIBLE", "PRIMARY_SUCCESS_ELIGIBLE", False, True, False, False),
        ("failure_noevent", "CLEAN_FAILURE", "MECHANISM_ELIGIBLE", "ELIGIBLE_CLEAN_FAILURE", False, False, False, True),
        ("ineligible", "CLEAN_SUCCESS", "MECHANISM_INELIGIBLE", "MECHANISM_INELIGIBLE_ABSTENTION", False, False, True, False),
    ]
    for group_idx in range(40):
        for row in baseline:
            add_row(group_idx, *row)
    extras = [
        (732, "primary_pos", "CLEAN_SUCCESS", "MECHANISM_ELIGIBLE", "PRIMARY_SUCCESS_ELIGIBLE", True, False, False, False),
        (231, "primary_noevent", "CLEAN_SUCCESS", "MECHANISM_ELIGIBLE", "PRIMARY_SUCCESS_ELIGIBLE", False, True, False, False),
        (31, "failure_pos", "CLEAN_FAILURE", "MECHANISM_ELIGIBLE", "ELIGIBLE_CLEAN_FAILURE", True, False, False, False),
        (236, "failure_noevent", "CLEAN_FAILURE", "MECHANISM_ELIGIBLE", "ELIGIBLE_CLEAN_FAILURE", False, False, False, True),
        (610, "ineligible", "CLEAN_SUCCESS", "MECHANISM_INELIGIBLE", "MECHANISM_INELIGIBLE_ABSTENTION", False, False, True, False),
    ]
    for count, *row in extras:
        for i in range(count):
            add_row(i, *row)

    with (fixture / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(FIXTURE / "source_manifest.csv").read_text(encoding="utf-8").splitlines()[0].split(","))
        writer.writeheader()
        writer.writerows(availability_rows)
    with (fixture / "episode_census.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(FIXTURE / "episode_census.csv").read_text(encoding="utf-8").splitlines()[0].split(","))
        writer.writeheader()
        writer.writerows(census_rows)
    (fixture / "source_event_crosstab.csv").write_text(
        "cohort_class,source_positive,source_no_event,total\n"
        "PRIMARY_SUCCESS_ELIGIBLE,772,271,1043\n"
        "ELIGIBLE_CLEAN_FAILURE,31,276,307\n"
        "MECHANISM_INELIGIBLE_ABSTENTION,0,650,650\n",
        encoding="utf-8",
    )
    return fixture


def test_manual_audit_uses_suite_task_key_and_enforces_160_quota(tmp_path):
    fixture = make_160_fixture(tmp_path)
    output = tmp_path / "approved" / "out"

    run_builder(fixture, output, "--enforce-manual-quota", "--expected-manual-sample-n", "160")

    manual = read_rows(output / "manual_audit_sample_manifest.csv")
    assert len(manual) == 160
    groups = {}
    for row in manual:
        groups.setdefault((row["suite"], row["task_id"]), []).append(row)
    assert len(groups) == 40
    assert all(len(rows) == 4 for rows in groups.values())
    assert len([key for key in groups if key[1] == "task_00"]) == 4


def test_formal_mode_requires_producer_identity(tmp_path):
    fixture = copy_fixture(tmp_path)
    cmd = [
        sys.executable,
        str(BUILDER),
        "--mode",
        "formal-ledger-build",
        "--source-manifest",
        str(fixture / "source_manifest.csv"),
        "--episode-census",
        str(fixture / "episode_census.csv"),
        "--source-crosstab",
        str(fixture / "source_event_crosstab.csv"),
        "--output-root",
        str(tmp_path / "formal-out"),
        "--expected-source-sha256",
        sha256(fixture / "source_manifest.csv"),
        "--expected-census-sha256",
        sha256(fixture / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture / "source_event_crosstab.csv"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "expected-git-commit-sha" in result.stderr


def test_formal_mode_closes_2000_row_ledger(tmp_path):
    if tracked_worktree_dirty():
        pytest.skip("formal mode requires a clean tracked worktree")
    fixture = make_formal_fixture(tmp_path)
    output = tmp_path / "formal-out"
    cmd = [
        sys.executable,
        str(BUILDER),
        "--mode",
        "formal-ledger-build",
        "--source-manifest",
        str(fixture / "source_manifest.csv"),
        "--episode-census",
        str(fixture / "episode_census.csv"),
        "--source-crosstab",
        str(fixture / "source_event_crosstab.csv"),
        "--output-root",
        str(output),
        "--expected-source-sha256",
        sha256(fixture / "source_manifest.csv"),
        "--expected-census-sha256",
        sha256(fixture / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture / "source_event_crosstab.csv"),
        "--expected-git-commit-sha",
        git_head(),
        "--expected-builder-sha256",
        sha256(BUILDER),
        "--require-clean-worktree",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    rows = read_rows(output / "label_v2.csv")
    manual = read_rows(output / "manual_audit_sample_manifest.csv")
    summary = json.loads((output / "validation_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 2000
    assert len(manual) == 160
    assert summary["counts"]["PRIMARY_SUCCESS_ELIGIBLE"] == {"positive": 772, "no_event": 271, "total": 1043}
    assert summary["counts"]["ELIGIBLE_CLEAN_FAILURE"] == {"positive": 31, "no_event": 276, "total": 307}
    assert summary["counts"]["MECHANISM_INELIGIBLE_ABSTENTION"] == {"positive": 0, "no_event": 650, "total": 650}


def test_manual_category_prefers_ineligible_over_failure():
    spec = importlib.util.spec_from_file_location("builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert builder.row_category(
        {
            "event_present": "false",
            "clean_success": "false",
            "mechanism_eligible": "false",
            "label_validity_status": "INVALID_WINDOW",
        }
    ) == "abstention_or_ineligible"


def test_symlink_input_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    real_source = fixture / "source_manifest.csv"
    target = fixture / "source_target.csv"
    real_source.rename(target)
    try:
        real_source.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    cmd = [
        sys.executable,
        str(BUILDER),
        "--source-manifest",
        str(real_source),
        "--episode-census",
        str(fixture / "episode_census.csv"),
        "--source-crosstab",
        str(fixture / "source_event_crosstab.csv"),
        "--output-root",
        str(tmp_path / "out"),
        "--synthetic-fixture-root",
        str(fixture),
        "--synthetic-output-root",
        str(tmp_path),
        "--expected-source-sha256",
        "0" * 64,
        "--expected-census-sha256",
        sha256(fixture / "episode_census.csv"),
        "--expected-crosstab-sha256",
        sha256(fixture / "source_event_crosstab.csv"),
        "--synthetic",
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode != 0
    assert "symlink path is not allowed" in result.stderr
