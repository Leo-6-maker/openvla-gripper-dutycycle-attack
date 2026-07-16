#!/usr/bin/env python3
"""Materialize one FIT episode for the causal 25D B3 S1 line.

This is an offline, versioned path.  It deliberately does not open the
policy-intent stream: the student is 25D-only.  The privileged sidecar is
used only to reconstruct the separate Teacher stream and to validate robot
evidence.  The existing retention materializer remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_PATH.parent))

from gripper_attack.b3_causal_25d import (  # noqa: E402
    B3Causal25DMultieventV1,
    FEATURE_NAMES,
    LEGACY_SOURCE_FEATURE_NAMES_25D,
    LEGACY_SOURCE_FEATURE_ORDER_SHA256,
    SCHEMA as CAUSAL_SCHEMA,
    SOURCE_SCHEMA,
    serialize_student_25d,
)
from gripper_attack.b3_retention import rebuild_retention_features  # noqa: E402
from materialize_b3_retention_episode import (  # noqa: E402
    IDENTITY_FIELDS,
    _identity_check,
    _merge_stream_rows,
    _write_jsonl,
    _write_output_checksums,
    load_jsonl,
    load_protocol_config,
    sha256_file,
    verify_robot_evidence_contract,
    verify_source_artifact,
    verify_source_contract,
)


OUTPUT_SCHEMA = "B3_CAUSAL_25D_S1_MATERIALIZED_EPISODE_V1"
MODE = "fit-label-materialization-25d-causal"
FIT_MAX_STATE = 19


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    )


def _feature_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal feature config must be an object")
    expected = {
        "schema": "B3_CAUSAL_25D_MULTIEVENT_V1",
        "source_schema": SOURCE_SCHEMA,
        "status": "FEATURE_CONTRACT_ONLY",
        "formal_training_authorized": False,
        "attack_authorized": False,
        "feature_names": list(FEATURE_NAMES),
        "legacy_source_feature_names": list(LEGACY_SOURCE_FEATURE_NAMES_25D),
        "legacy_source_feature_order_sha256": LEGACY_SOURCE_FEATURE_ORDER_SHA256,
        "required_measured_action_fields": ["action_raw", "action_env"],
        "action_parity_tolerance": 1e-6,
        "robot_qpos_parity_tolerance": 1e-6,
        "robot_eef_parity_tolerance": 1e-3,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"causal feature config mismatch: {key}")
    return payload


def _materialization_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal S1 materialization config must be an object")
    expected = {
        "schema": "B3_CAUSAL_25D_S1_MATERIALIZATION_V1",
        "status": "PREPARATION_ONLY",
        "fit_state_range": [0, 19],
        "student_policy_intent_read": False,
        "student_stream": "25D_CAUSAL_ONLY",
        "teacher_stream": "SEPARATE_PRIVILEGED_RECONSTRUCTION",
        "teacher_event_stream": "SEPARATE_FILE",
        "unknown_is_negative": False,
        "formal_training_authorized": False,
        "attack_authorized": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"causal S1 materialization config mismatch: {key}")
    return payload


def _student_only_join(root: Path, meta: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join only step records and the privileged robot sidecar.

    ``policy_intent_records.jsonl`` is intentionally not opened here.  Its
    checksum is still covered by ``artifact_sha256.json`` through
    ``verify_source_artifact``.
    """
    streams = {
        "step_records": load_jsonl(root / "step_records.jsonl"),
        "privileged_sidecar": load_jsonl(root / "privileged_teacher_sidecar.jsonl"),
    }
    expected_steps: list[int] | None = None
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for name, rows in streams.items():
        if not rows:
            raise ValueError(f"{name} is empty")
        steps = [int(row.get("step", row.get("step_idx", index))) for index, row in enumerate(rows)]
        if steps != list(range(len(rows))):
            raise ValueError(f"{name} steps are not contiguous from zero")
        for row in rows:
            _identity_check(row, meta, source=name)
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError(f"strict student-only step join mismatch: {name}")
        indexed[name] = {step: row for step, row in zip(steps, rows)}

    assert expected_steps is not None
    float_merges: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for step in expected_steps:
        row = _merge_stream_rows(
            step,
            [
                ("step_records", indexed["step_records"][step]),
                ("privileged_sidecar", indexed["privileged_sidecar"][step]),
            ],
            float_merges,
        )
        for name in IDENTITY_FIELDS:
            row[name] = meta[name]
        # The source episode stores the legacy feature vector but not always
        # the per-row order binding.  Bind it from immutable metadata before
        # passing the row to the causal builder.
        row["feature_names_25d"] = list(LEGACY_SOURCE_FEATURE_NAMES_25D)
        row["feature_order_sha256"] = LEGACY_SOURCE_FEATURE_ORDER_SHA256
        merged.append(row)
    return merged, float_merges


def _verify_step_source(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if row.get("official_execution") is not True:
            raise ValueError(f"step {index}: official execution flag missing")
        if not _finite_vector(row.get("features_25d"), 25):
            raise ValueError(f"step {index}: invalid source 25D vector")
        for name in ("clean_action_raw_7d", "applied_action_7d"):
            if not _finite_vector(row.get(name), 7):
                raise ValueError(f"step {index}: invalid {name}")
        if not isinstance(row.get("action_token_ids"), list) or len(row["action_token_ids"]) != 7:
            raise ValueError(f"step {index}: action-token count is not seven")
        if not isinstance(row.get("score_head_summary"), list) or len(row["score_head_summary"]) != 7:
            raise ValueError(f"step {index}: score count is not seven")
        if row.get("score_adapter_parity_pass") is not True:
            raise ValueError(f"step {index}: score/action parity is not verified")
        error = row.get("score_adapter_action_max_abs_error")
        if not isinstance(error, (int, float)) or not math.isfinite(float(error)) or float(error) > 1e-6:
            raise ValueError(f"step {index}: score/action error exceeds tolerance")
        if not isinstance(row.get("robot0_eef_pos"), list) or len(row["robot0_eef_pos"]) != 3:
            raise ValueError(f"step {index}: missing EEF sidecar")
        if not isinstance(row.get("robot0_gripper_qpos"), list) or len(row["robot0_gripper_qpos"]) < 2:
            raise ValueError(f"step {index}: missing gripper qpos sidecar")


def _prepare(
    artifact_root: Path,
    source_protocol_path: Path,
    feature_config_path: Path,
    materialization_config_path: Path,
    *,
    materialize_teacher: bool,
) -> dict[str, Any]:
    source_protocol, retention_config = load_protocol_config(source_protocol_path)
    feature_config = _feature_config(feature_config_path)
    materialization_config = _materialization_config(materialization_config_path)
    meta = json.loads((artifact_root / "episode_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("episode metadata must be an object")
    if meta.get("split") != "FIT" or not isinstance(meta.get("state_id"), int) or meta["state_id"] > FIT_MAX_STATE:
        raise ValueError("causal FIT materialization requires split=FIT and state_id<=19")

    source_json = {
        name: json.loads((artifact_root / name).read_text(encoding="utf-8"))
        for name in ("episode_summary.json", "runtime_audit.json", "condition_config.json", "attack_config.json")
    }
    verify_source_contract(
        artifact_root,
        meta,
        source_json["episode_summary.json"],
        source_json["runtime_audit.json"],
        source_json["condition_config.json"],
        source_json["attack_config.json"],
        source_protocol,
    )
    if meta.get("feature_names_25d") != list(LEGACY_SOURCE_FEATURE_NAMES_25D):
        raise ValueError("source metadata legacy 25D order mismatch")
    source_sha = verify_source_artifact(artifact_root)
    merged, float_merges = _student_only_join(artifact_root, meta)
    _verify_step_source(merged)
    robot_errors = verify_robot_evidence_contract(merged)

    causal = B3Causal25DMultieventV1().rebuild(merged)
    if any(row.get("valid") is not True for row in causal["rows"]):
        invalid = [row.get("step") for row in causal["rows"] if row.get("valid") is not True]
        raise ValueError(f"causal feature reconstruction invalid rows: {invalid[:5]}")
    for row in causal["rows"]:
        serialize_student_25d({
            "schema": row["schema"],
            "source_schema": row["source_schema"],
            "valid": row["valid"],
            "features_25d": row["features_25d"],
        })

    teacher = None
    if materialize_teacher:
        teacher = rebuild_retention_features(merged, retention_config)
        if len(teacher["rows"]) != len(merged):
            raise ValueError("Teacher/student step count mismatch")
    return {
        "meta": meta,
        "source_sha": source_sha,
        "merged": merged,
        "float_merges": float_merges,
        "robot_errors": robot_errors,
        "causal": causal,
        "teacher": teacher,
        "retention_config": retention_config,
        "source_protocol": source_protocol,
        "source_protocol_sha256": sha256_file(source_protocol_path),
        "feature_config": feature_config,
        "feature_config_sha256": sha256_file(feature_config_path),
        "materialization_config": materialization_config,
        "materialization_config_sha256": sha256_file(materialization_config_path),
    }


def validate_materialization_inputs(
    artifact_root: Path,
    source_protocol_path: Path,
    feature_config_path: Path,
    materialization_config_path: Path,
) -> dict[str, Any]:
    """Run the full causal FIT preflight without creating output files."""
    prepared = _prepare(
        artifact_root.resolve(),
        source_protocol_path.resolve(),
        feature_config_path.resolve(),
        materialization_config_path.resolve(),
        materialize_teacher=False,
    )
    return {
        "source_artifact_sha256": prepared["source_sha"],
        "step_count": len(prepared["merged"]),
        "causal_event_count": len(prepared["causal"]["events"]),
        "robot_evidence_error_summary": prepared["robot_errors"],
        "student_policy_intent_read": False,
        "teacher_labels_materialized": False,
    }


def materialize(
    artifact_root: Path,
    output_root: Path,
    source_protocol_path: Path,
    feature_config_path: Path,
    materialization_config_path: Path,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"materialization output is non-empty: {output_root}")
    prepared = _prepare(
        artifact_root,
        source_protocol_path.resolve(),
        feature_config_path.resolve(),
        materialization_config_path.resolve(),
        materialize_teacher=True,
    )
    meta = prepared["meta"]
    identity = {name: meta.get(name) for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
    output_root.mkdir(parents=True, exist_ok=True)

    student_rows = []
    for row in prepared["causal"]["rows"]:
        projection = {
            "schema": row["schema"],
            "source_schema": row["source_schema"],
            "valid": row["valid"],
            "features_25d": row["features_25d"],
        }
        vector = serialize_student_25d(projection)
        student_rows.append({**identity, "step": row["step"], **projection, "features_25d": list(vector)})
    _write_jsonl(output_root / "student_input_records.jsonl", student_rows)

    teacher_rows = [
        {**identity, **{key: value for key, value in row.items() if key not in {"object_state", "mujoco_contact_pairs"}}}
        for row in prepared["teacher"]["rows"]
    ]
    _write_jsonl(output_root / "teacher_retention_records.jsonl", teacher_rows)
    (output_root / "retention_events.json").write_text(
        json.dumps(prepared["teacher"]["events"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "causal_event_summary.json").write_text(
        json.dumps(prepared["causal"]["events"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_after = verify_source_artifact(artifact_root)
    output_files = [
        "student_input_records.jsonl",
        "teacher_retention_records.jsonl",
        "retention_events.json",
        "causal_event_summary.json",
    ]
    file_rows = [
        {"path": name, "size": (output_root / name).stat().st_size, "sha256": sha256_file(output_root / name)}
        for name in output_files
    ]
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "mode": MODE,
        "source_schema": SOURCE_SCHEMA,
        "derived_schema": CAUSAL_SCHEMA,
        "source_contract_verified": True,
        "official_protocol_id": meta["protocol_id"],
        "source_artifact_sha256": prepared["source_sha"],
        "source_recursive_sha256_before": prepared["source_sha"],
        "source_recursive_sha256_after": source_after,
        "source_unchanged": prepared["source_sha"] == source_after,
        "source_identity": identity,
        "source_protocol_config_sha256": prepared["source_protocol_sha256"],
        "feature_config_sha256": prepared["feature_config_sha256"],
        "config_sha256": prepared["materialization_config_sha256"],
        "materialization_config_sha256": prepared["materialization_config_sha256"],
        "effective_retention_config": {
            key: getattr(prepared["retention_config"], key)
            for key in prepared["retention_config"].__dataclass_fields__
        },
        "causal_builder_sha256": sha256_file(REPO_ROOT / "src" / "gripper_attack" / "b3_causal_25d.py"),
        "teacher_rebuilder_sha256": sha256_file(REPO_ROOT / "src" / "gripper_attack" / "b3_retention.py"),
        "materializer_sha256": sha256_file(SCRIPT_PATH),
        "step_count": len(student_rows),
        "causal_event_count": len(prepared["causal"]["events"]),
        "teacher_event_count": len(prepared["teacher"]["events"]),
        "student_feature_names": list(FEATURE_NAMES),
        "student_projection_keys": ["schema", "source_schema", "valid", "features_25d"],
        "student_policy_intent_read": False,
        "student_policy_intent_present": False,
        "student_forbidden_fields_absent": True,
        "label_statistics": None,
        "teacher_materialization": "COMPLETED",
        "teacher_label_semantics": "ROBOT_CENTRIC_PROXY_LABELS_NOT_GRASP_GROUND_TRUTH",
        "unknown_is_negative": False,
        "join_float_tolerance_merges": prepared["float_merges"],
        "robot_evidence_contract_verified": True,
        "robot_evidence_error_summary": prepared["robot_errors"],
        "files": file_rows,
        "output_recursive_sha256": _json_sha(file_rows),
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }
    (output_root / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_output_checksums(output_root)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--materialization-config", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(
        args.artifact_root,
        args.output_root,
        args.source_protocol,
        args.feature_config,
        args.materialization_config,
    )
    print(json.dumps({"status": "PASS", "schema": manifest["schema"], "step_count": manifest["step_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
