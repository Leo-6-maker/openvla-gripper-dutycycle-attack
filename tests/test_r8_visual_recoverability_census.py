from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "detector_v4" / "census_r8_official_v3_artifacts.py"
    spec = importlib.util.spec_from_file_location("r8_visual_census", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(root: Path, identity: str, *, late_visual_field: bool = False) -> Path:
    episode = root.joinpath(*identity.split("/"))
    episode.mkdir(parents=True)
    for name in (
        "episode_metadata.json",
        "episode_summary.json",
        "runtime_audit.json",
        "condition_config.json",
        "attack_config.json",
    ):
        (episode / name).write_text(json.dumps({"schema": "test", "step_count": 2}), encoding="utf-8")

    step_rows = [{"step": 0, "features_25d": [0.0] * 25}, {"step": 1, "features_25d": [0.0] * 25}]
    if late_visual_field:
        step_rows[1]["visual_embedding"] = [1.0, 2.0]
    (episode / "step_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in step_rows), encoding="utf-8"
    )
    (episode / "policy_intent_records.jsonl").write_text(
        json.dumps({"step": 0, "clean_policy_intent_9d": [0.0] * 9}) + "\n",
        encoding="utf-8",
    )
    (episode / "privileged_teacher_sidecar.jsonl").write_text(
        json.dumps({"step": 0, "robot0_eef_pos": [0.0, 0.0, 0.0]}) + "\n",
        encoding="utf-8",
    )
    (episode / "artifact_sha256.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    return episode


def test_field_census_scans_every_identity_and_every_jsonl_row(tmp_path: Path):
    module = _module()
    identities = ["libero_object/task_00/state_00", "libero_goal/task_01/state_01"]
    _episode(tmp_path, identities[0])
    _episode(tmp_path, identities[1], late_visual_field=True)

    result = module.field_census(tmp_path, identities)

    assert result["identities_scanned"] == 2
    assert result["missing_stream_count"] == 0
    assert result["keyword_hits"]["visual"] >= 1
    assert result["keyword_hits"]["embedding"] >= 1
    example = result["details"]["visual"][0]
    assert example["identity"] == identities[1]
    assert example["row"] == 2


def test_artifact_census_is_recursive_and_catches_binary_carrier(tmp_path: Path):
    module = _module()
    identity = "libero_spatial/task_00/state_00"
    episode = _episode(tmp_path, identity)
    nested = episode / "cache"
    nested.mkdir()
    (nested / "agentview_frames.npz").write_bytes(b"not-an-npz")

    result = module.census_artifacts(tmp_path, [identity])

    assert result["identity_count"] == 1
    assert result["n_binary"] == 1
    assert result["binary_files"][0]["path"] == "cache/agentview_frames.npz"
    assert result["identity_rows"][0]["nested_file_count"] == 1
    assert result["identity_rows"][0]["exact_expected_file_set"] is False


def test_classification_requires_full_exhaustive_closure():
    module = _module()
    identities = [f"libero_object/task_00/state_{index:02d}" for index in range(800)]
    artifacts = {
        "identity_count": 800,
        "exact_expected_file_set_count": 800,
        "n_binary": 0,
        "filename_keyword_hits": {},
    }
    fields = {
        "identities_scanned": 800,
        "missing_stream_count": 0,
        "keyword_hit_total": 0,
    }

    status, errors = module.classify_census(identities, artifacts, fields)
    assert status == "NO_VISUAL_ASSET"
    assert errors == []

    fields["identities_scanned"] = 799
    status, errors = module.classify_census(identities, artifacts, fields)
    assert status == "HOLD_INCOMPLETE_CENSUS"
    assert errors


def test_teacher_geometry_uses_union_not_first_row_only():
    module = _module()
    summary = module.teacher_geometry_summary(
        {
            "stream_field_unions": {
                "privileged_teacher_sidecar.jsonl": [
                    "step",
                    "robot0_eef_pos",
                    "mujoco_contact_pairs",
                    "object_state",
                ]
            }
        }
    )
    assert summary["geometry_field_count"] == 3
    assert "object_state" in summary["geometry_field_paths"]
