import json
import csv
from pathlib import Path

from detector.build_b3_fit_census import build_census, expected_identities, write_census


def test_expected_fit_census_is_complete_and_task_balanced():
    identities = expected_identities()
    assert len(identities) == 800
    assert len({row["canonical_parent_key"] for row in identities}) == 800
    assert {row["split"] for row in identities} == {"FIT"}
    for suite in {row["suite"] for row in identities}:
        suite_rows = [row for row in identities if row["suite"] == suite]
        assert len(suite_rows) == 200
        for task_idx in range(10):
            assert sum(row["task_idx"] == task_idx for row in suite_rows) == 20


def test_empty_source_is_a_complete_800_identity_census_without_teacher_reads(tmp_path: Path):
    rows, summary = build_census(None, tmp_path / "empty-source")
    assert summary["status"] == "CENSUS_COMPLETE"
    assert summary["identity_count"] == 800
    assert summary["status_counts"] == {"MISSING": 800}
    assert summary["teacher_labels_read"] is False
    assert summary["teacher_files_opened"] is False

    output = tmp_path / "census"
    write_census(rows, summary, output)
    written = list(csv_row for csv_row in (output / "B3_FIT_CENSUS_V1.csv").read_text().splitlines())
    assert len(written) == 801
    assert json.loads((output / "B3_FIT_CENSUS_SUMMARY.json").read_text())["status"] == "CENSUS_COMPLETE"


def test_metadata_identity_mismatch_is_protocol_hold(tmp_path: Path):
    source = tmp_path / "source"
    artifact = source / "one"
    artifact.mkdir(parents=True)
    (artifact / "episode_metadata.json").write_text(json.dumps({
        "suite": "libero_object",
        "task_idx": 0,
        "state_id": 0,
        "canonical_parent_key": "libero_object/task_00/state_01",
        "condition": "CLEAN",
        "split": "FIT",
        "runtime_valid": True,
    }) + "\n", encoding="utf-8")
    (artifact / "runtime_audit.json").write_text(json.dumps({"runtime_valid": True}) + "\n", encoding="utf-8")

    # The canonical key points at state_01, but the numeric metadata claims
    # state_00.  The census must hold the identity instead of trusting the key.
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["suite", "task_idx", "state_id", "canonical_parent_key", "split"])
        writer.writeheader()
        writer.writerows(expected_identities())
    rows, _ = build_census(manifest, source)
    target = next(row for row in rows if row["canonical_parent_key"] == "libero_object/task_00/state_01")
    assert target["status"] == "PROTOCOL_HOLD"
    assert target["reason"] == "IDENTITY_MISMATCH"
