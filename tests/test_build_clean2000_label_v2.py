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
    assert read_rows(output / "manual_audit_sample_manifest.csv")
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
        "--expected-source-sha256",
        "0" * 64,
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


def test_non_synthetic_input_rejected(tmp_path):
    fixture = copy_fixture(tmp_path, "not_synthetic")

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert "non-synthetic path input" in result.stderr


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


def test_symlink_input_rejected(tmp_path):
    fixture = copy_fixture(tmp_path)
    real_source = fixture / "source_manifest.csv"
    link = fixture / "source_link.csv"
    try:
        link.symlink_to(real_source)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    real_source.unlink()
    link.rename(real_source)

    result = run_builder(fixture, tmp_path / "out", expect_ok=False)

    assert "symlink path is not allowed" in result.stderr
