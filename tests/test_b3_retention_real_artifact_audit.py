import json
from pathlib import Path

import pytest

from detector import audit_b3_retention_real_artifacts as real_audit
from test_b3_retention_materializer import _build_source_artifact, _reseal, _write_json, _write_jsonl


def _rewrite_identity(root: Path, task_idx: int) -> None:
    key = f"libero_10/task_{task_idx:02d}/state_019"
    metadata_path = root / "episode_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"task_idx": task_idx, "canonical_parent_key": key})
    _write_json(metadata_path, metadata)
    for name in ("step_records.jsonl", "policy_intent_records.jsonl", "privileged_teacher_sidecar.jsonl"):
        path = root / name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.update({"task_idx": task_idx, "canonical_parent_key": key})
        _write_jsonl(path, rows)
    _reseal(root)


def test_fit_preflight_rejects_late_alias_before_any_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = source_root / "first"
    second = source_root / "second"
    _build_source_artifact(first)
    _build_source_artifact(second)
    _rewrite_identity(second, 3)

    step_path = second / "step_records.jsonl"
    rows = [json.loads(line) for line in step_path.read_text().splitlines()]
    rows[3]["action_raw"][-1] = 0.2
    _write_jsonl(step_path, rows)
    _reseal(second)

    selection = tmp_path / "selection.csv"
    selection.write_text("artifact,runtime_valid\nfirst,true\nsecond,true\n", encoding="utf-8")
    config = Path(__file__).parents[1] / "configs" / "B3_RETENTION_PROTOCOL_V1.json"
    output = tmp_path / "fit-output"
    monkeypatch.setattr(
        real_audit,
        "_git_provenance",
        lambda repo, expected: ("head", True, True, "scripts/detector/audit_b3_retention_real_artifacts.py"),
    )

    with pytest.raises(ValueError, match="action_raw alias mismatch"):
        real_audit.run(
            source_root,
            output,
            config,
            selection,
            "fit-label-materialization",
            "head",
            tmp_path,
            False,
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".staging").exists()


def test_fit_success_promotes_staging_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = source_root / "only"
    _build_source_artifact(artifact)
    selection = tmp_path / "selection.csv"
    selection.write_text("artifact,runtime_valid\nonly,true\n", encoding="utf-8")
    config = Path(__file__).parents[1] / "configs" / "B3_RETENTION_PROTOCOL_V1.json"
    output = tmp_path / "fit-output"
    monkeypatch.setattr(
        real_audit,
        "_git_provenance",
        lambda repo, expected: ("head", True, True, "scripts/detector/audit_b3_retention_real_artifacts.py"),
    )

    report = real_audit.run(
        source_root,
        output,
        config,
        selection,
        "fit-label-materialization",
        "head",
        tmp_path,
        False,
    )

    assert report["status"] == "PASS"
    assert report["fit_transactional_preflight"] is True
    assert report["fit_output_promoted"] is True
    assert output.is_dir()
    assert not output.with_name(output.name + ".staging").exists()
    assert len(list((output / "episodes").glob("*/teacher_retention_records.jsonl"))) == 1
