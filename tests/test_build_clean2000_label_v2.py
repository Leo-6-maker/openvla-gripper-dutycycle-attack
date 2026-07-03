import csv
import hashlib
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


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rewrite_source(path, transform):
    rows = read_rows(path)
    rows = transform(rows)
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
    assert {r["episode_key"] for r in rows if r["label_validity_status"] == "INVALID_WINDOW"} == {
        "ep_invalid_window",
        "ep_truncated_trace",
    }
    assert all(r["builder_sha256"] for r in rows)
    manual = read_rows(output / "manual_audit_sample_manifest.csv")
    assert manual
    assert "suite" in manual[0]
    assert {"requested_priority", "actual_selected_category", "fallback_used", "fallback_reason"} <= set(manual[0])
    assert "label_v2.csv" in (output / "SHA256SUMS").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutator, expected_error",
    [
        (
            lambda p: rewrite_source(p, lambda rows: rows + [dict(rows[0])]),
            "duplicate episode_key",
        ),
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8").replace(",teacher_confidence\n", "\n"), encoding="utf-8"
            ),
            "expected columns",
        ),
        (
            lambda p: p.write_text(
                p.read_text(encoding="utf-8").replace("teacher_confidence\n", "teacher_confidence,extra\n"),
                encoding="utf-8",
            ),
            "expected columns",
        ),
        (
            lambda p: p.write_text(p.read_text(encoding="utf-8").replace("0.95\n", "0.95,extra\n", 1), encoding="utf-8"),
            "extra cells",
        ),
        (
            lambda p: p.write_text(p.read_text(encoding="utf-8").replace(",0.95\n", "\n", 1), encoding="utf-8"),
            "missing cells",
        ),
        (
            lambda p: rewrite_source(p, lambda rows: [dict(r, clean_success="maybe") if i == 0 else r for i, r in enumerate(rows)]),
            "illegal bool",
        ),
        (
            lambda p: rewrite_source(
                p,
                lambda rows: [
                    dict(r, event_present="false", anchor_absolute_step="3", window_start="2", window_end="5", event_source="")
                    if i == 0
                    else r
                    for i, r in enumerate(rows)
                ],
            ),
            "no-event row has non-empty anchor/window",
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


def test_duplicate_census_and_crosstab_cohort_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    census = fixture / "episode_census.csv"
    census.write_text(census.read_text(encoding="utf-8") + "PRIMARY_SUCCESS_ELIGIBLE,4\n", encoding="utf-8")
    result = run_builder(fixture, tmp_path / "out1", expect_ok=False)
    assert "duplicate census cohort" in result.stderr

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
    rows = []
    for suite in suites:
        for task_idx in range(10):
            task = f"task_{task_idx:02d}"
            base = {
                "suite": suite,
                "task_id": task,
                "coordinate_semantics": "zero_based_observation_before_action_start_inclusive_end_exclusive_full_trajectory",
                "trace_length": "50",
                "source_schema_version": "synthetic_v1",
                "mechanism_type": "single_object_transfer",
                "segment_id": f"{suite}_{task}",
            }
            variants = [
                ("pos", "PRIMARY_SUCCESS_ELIGIBLE", "true", "true", "true", "10", "8", "14", "teacher_rule", "", "", "positive_clean_success"),
                ("noevent", "PRIMARY_SUCCESS_ELIGIBLE", "true", "true", "false", "-1", "-1", "-1", "", "", "NO_TEACHER_EVENT", "eligible_no_event"),
                ("failure", "ELIGIBLE_CLEAN_FAILURE", "false", "true", "true", "11", "9", "15", "teacher_rule", "", "", "failure_or_boundary"),
                ("ineligible", "MECHANISM_INELIGIBLE_ABSTENTION", "true", "false", "false", "-1", "-1", "-1", "", "", "UNSUPPORTED_MECHANISM", "abstention_or_ineligible"),
            ]
            for suffix, cohort, clean, eligible, present, anchor, start, end, event_source, invalid, abstain, _category in variants:
                episode = f"{suite}_{task}_{suffix}"
                rows.append({
                    **base,
                    "episode_key": episode,
                    "parent_key": f"{suite}_{task}_{suffix}_parent",
                    "cohort_class": cohort,
                    "clean_success": clean,
                    "mechanism_eligible": eligible,
                    "event_present": present,
                    "anchor_absolute_step": anchor,
                    "window_start": start,
                    "window_end": end,
                    "event_source": event_source,
                    "source_path": f"synthetic/{episode}.json",
                    "source_sha256": "a" * 64,
                    "invalid_reason": invalid,
                    "abstain_reason": abstain,
                    "event_id": "event_1" if present == "true" else "event_none",
                    "event_rank": "1" if present == "true" else "0",
                    "teacher_confidence": "0.9" if present == "true" else "0.0",
                })
    with (fixture / "source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = (FIXTURE / "source_manifest.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (fixture / "episode_census.csv").write_text(
        "cohort_class,total\nPRIMARY_SUCCESS_ELIGIBLE,80\nELIGIBLE_CLEAN_FAILURE,40\nMECHANISM_INELIGIBLE_ABSTENTION,40\n",
        encoding="utf-8",
    )
    (fixture / "source_event_crosstab.csv").write_text(
        "cohort_class,source_positive,source_no_event,total\nPRIMARY_SUCCESS_ELIGIBLE,40,40,80\nELIGIBLE_CLEAN_FAILURE,40,0,40\nMECHANISM_INELIGIBLE_ABSTENTION,0,40,40\n",
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
