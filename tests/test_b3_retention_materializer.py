import json
from pathlib import Path

import pytest

from detector.audit_b3_retention_materialization import audit
from detector.materialize_b3_retention_episode import json_sha, materialize, sha256_file


IDENTITY = {
    "suite": "libero_10",
    "task_idx": 2,
    "state_id": 30,
    "canonical_parent_key": "libero_10/task_02/state_030",
}


def _protocol() -> dict:
    return json.loads((Path(__file__).parents[1] / "configs" / "B3_RETENTION_PROTOCOL_V1.json").read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _build_source_artifact(root: Path, *, steps: int = 18) -> None:
    root.mkdir()
    _write_json(root / "episode_metadata.json", {
        "schema": "OPENVLA_OFFICIAL_CLEAN_EPISODE_V2",
        "protocol_id": "OPENVLA_LIBERO_OFFICIAL_V1",
        "condition": "CLEAN",
        "runtime_valid": True,
        "env_reset_called": True,
        "checkpoint_binding_pass": True,
        "generation_passes_per_step": 1,
        "num_steps_wait": 10,
        "official_horizon": 520,
        "task_language": "synthetic task",
        "split": "FINAL_EVAL_CANDIDATE",
        "initial_state_sha256": "a" * 64,
        "official_execution_adapter": "OfficialOpenVLAActionAdapter.predict_action",
        "score_adapter": "OfficialOpenVLAActionAdapter.predict_action_with_scores",
        "env_success": False,
        "success": False,
        "feature_names_25d": _protocol()["feature_names_25d"],
        "policy_intent_feature_names_9d": _protocol()["policy_intent_feature_names_9d"],
        **IDENTITY,
    })
    _write_json(root / "episode_summary.json", {"condition": "CLEAN", "clean": True, "success": False})
    _write_json(root / "runtime_audit.json", {
        "runtime_valid": True,
        "env_reset_called": True,
        "checkpoint_binding_pass": True,
        "official_horizon": 520,
    })
    _write_json(root / "condition_config.json", {"condition": "CLEAN", "protocol_id": "OPENVLA_LIBERO_OFFICIAL_V1"})
    _write_json(root / "attack_config.json", {"attack_enabled": False})

    step_rows = []
    policy_rows = []
    sidecar_rows = []
    for step in range(steps):
        close = step >= 3
        qpos = [-0.05, 0.05] if close else [0.2, 0.2]
        raw_action = [0.0] * 6 + [0.1 if close else 0.8]
        env_action = [0.0] * 6 + [1.0 if close else -1.0]
        tokens = list(range(7))
        score_summary = [{"top_token": token, "top_probability": 0.5} for token in tokens]
        step_rows.append({
            "suite": IDENTITY["suite"], "task_idx": IDENTITY["task_idx"], "state_id": IDENTITY["state_id"],
            "step": step,
            "official_execution": True,
            "features_25d": [float(step), *([0.0] * 24)],
            "raw_close": close,
            "gripper_qpos": sum(qpos),
            "gripper_opening_proxy": sum(abs(value) for value in qpos),
            "clean_policy_intent_9d": [0.0] * 9,
            "clean_action_raw_7d": raw_action,
            "applied_action_7d": env_action,
            "action_raw": raw_action,
            "action_env": env_action,
            "action_token_ids": tokens,
            "score_head_summary": score_summary,
            "score_adapter_parity_pass": True,
            "score_adapter_action_max_abs_error": 0.0,
        })
        policy_rows.append({
            "step": step,
            "clean_policy_intent_9d": [0.0] * 9,
            "action_token_ids": tokens,
            "score_adapter_parity_pass": True,
        })
        sidecar_rows.append({
            "suite": IDENTITY["suite"], "task_idx": IDENTITY["task_idx"], "state_id": IDENTITY["state_id"],
            "step": step,
            "task_language": "synthetic task",
            "robot0_eef_pos": [step * 0.01, 0.0, 0.3],
            "robot0_gripper_qpos": qpos,
        })
    _write_jsonl(root / "step_records.jsonl", step_rows)
    _write_jsonl(root / "policy_intent_records.jsonl", policy_rows)
    _write_jsonl(root / "privileged_teacher_sidecar.jsonl", sidecar_rows)

    rows = []
    for path in sorted(root.iterdir()):
        if path.name == "artifact_sha256.json":
            continue
        rows.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    _write_json(root / "artifact_sha256.json", {"files": rows, "recursive_sha256": json_sha(rows)})


def test_materializer_writes_separate_student_teacher_streams_and_audits(tmp_path: Path):
    source = tmp_path / "episode"
    output = tmp_path / "materialized"
    _build_source_artifact(source)

    config = Path(__file__).parents[1] / "configs" / "B3_RETENTION_PROTOCOL_V1.json"
    manifest = materialize(source, output, config)
    result = audit(output)

    assert manifest["formal_training_ready"] is False
    assert manifest["formal_attack_ready"] is False
    assert result["status"] == "PASS"
    student = [json.loads(line) for line in (output / "student_input_records.jsonl").read_text().splitlines()]
    teacher = [json.loads(line) for line in (output / "teacher_retention_records.jsonl").read_text().splitlines()]
    assert len(student) == len(teacher) == 18
    assert "event_id" not in student[0]
    assert "retention_continuation_t10" in teacher[0]
    assert teacher[0]["retention_unknown_mask"] is False
    with pytest.raises(ValueError, match="output is non-empty"):
        materialize(source, output, config)


def test_materializer_rejects_strict_join_identity_mismatch(tmp_path: Path):
    source = tmp_path / "episode"
    _build_source_artifact(source)
    policy_path = source / "policy_intent_records.jsonl"
    rows = [json.loads(line) for line in policy_path.read_text().splitlines()]
    rows[4]["state_id"] = 31
    _write_jsonl(policy_path, rows)

    # The checksum closure is intentionally rebuilt so the failure comes from
    # the identity join, not from an earlier checksum check.
    checksums = []
    for path in sorted(source.iterdir()):
        if path.name == "artifact_sha256.json":
            continue
        checksums.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    _write_json(source / "artifact_sha256.json", {"files": checksums, "recursive_sha256": json_sha(checksums)})

    config = Path(__file__).parents[1] / "configs" / "B3_RETENTION_PROTOCOL_V1.json"
    with pytest.raises(ValueError, match="identity mismatch"):
        materialize(source, tmp_path / "materialized", config)


def test_materializer_rejects_duplicate_field_conflict(tmp_path: Path):
    source = tmp_path / "episode"
    _build_source_artifact(source)
    policy_path = source / "policy_intent_records.jsonl"
    rows = [json.loads(line) for line in policy_path.read_text().splitlines()]
    rows[4]["clean_policy_intent_9d"][0] = 0.25
    _write_jsonl(policy_path, rows)

    checksums = []
    for path in sorted(source.iterdir()):
        if path.name == "artifact_sha256.json":
            continue
        checksums.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    _write_json(source / "artifact_sha256.json", {"files": checksums, "recursive_sha256": json_sha(checksums)})

    config = Path(__file__).parents[1] / "configs" / "B3_RETENTION_PROTOCOL_V1.json"
    with pytest.raises(ValueError, match="JOIN_CONFLICT_HOLD"):
        materialize(source, tmp_path / "materialized", config)
