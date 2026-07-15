#!/usr/bin/env python3
"""Strict, offline B3-Retention materialization for one sealed CLEAN episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.b3_retention import RetentionConfig, rebuild_retention_features  # noqa: E402


REQUIRED = (
    "episode_metadata.json",
    "episode_summary.json",
    "runtime_audit.json",
    "condition_config.json",
    "attack_config.json",
    "step_records.jsonl",
    "policy_intent_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
    "artifact_sha256.json",
)
HEADS = (
    "grasp_support",
    "retention_active",
    "retention_continuation_t10",
    "release_imminent",
)
IDENTITY_FIELDS = ("suite", "task_idx", "state_id", "canonical_parent_key")
PROTOCOL_SCHEMA = "c2g.b3_retention.protocol.v1"
PROTOCOL_STATUS = "PREPARATION_ONLY"
OFFICIAL_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
OFFICIAL_SPLITS = {
    "FIT": range(0, 24),
    "CAL": range(24, 27),
    "CHECK": range(27, 30),
    "FINAL_EVAL_CANDIDATE": range(30, 50),
}
MERGE_TOLERANCE = 1e-6


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected object")
            rows.append(value)
    return rows


def _step(row: dict[str, Any], fallback: int) -> int:
    value = row.get("step", row.get("step_idx", fallback))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid step {value!r}") from exc


def _identity_check(row: dict[str, Any], meta: dict[str, Any], *, source: str) -> None:
    for name in IDENTITY_FIELDS:
        # Metadata is the identity authority.  Streams may omit identity
        # columns by schema; if present, a stream value must agree.
        if name not in row:
            continue
        expected = meta.get(name)
        actual = row.get(name)
        if name in {"task_idx", "state_id"}:
            try:
                expected, actual = int(expected), int(actual)
            except (TypeError, ValueError):
                pass
        if actual != expected:
            raise ValueError(f"{source} identity mismatch for {name}: {actual!r} != {expected!r}")


def _values_equal(left: Any, right: Any, tolerance: float = MERGE_TOLERANCE) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_values_equal(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key], tolerance) for key in left
        )
    return left == right


def _max_float_delta(left: Any, right: Any) -> float:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right))
    if isinstance(left, list) and isinstance(right, list):
        return max((_max_float_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, dict) and isinstance(right, dict):
        return max((_max_float_delta(left[key], right[key]) for key in left if key in right), default=0.0)
    return 0.0


def _merge_stream_rows(
    step: int,
    sources: list[tuple[str, dict[str, Any]]],
    float_merges: list[dict[str, Any]],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source, row in sources:
        for key, value in row.items():
            if key not in merged:
                merged[key] = value
                continue
            if not _values_equal(merged[key], value):
                raise ValueError(f"JOIN_CONFLICT_HOLD: step={step} field={key} source={source}")
            if merged[key] != value:
                float_merges.append({
                    "step": step,
                    "field": key,
                    "source": source,
                    "max_abs_delta": _max_float_delta(merged[key], value),
                })
    merged["step"] = step
    return merged


def verify_source_artifact(root: Path) -> str:
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing source artifact files: {missing}")
    payload = json.loads((root / "artifact_sha256.json").read_text(encoding="utf-8"))
    rows = payload.get("files")
    if not isinstance(rows, list) or payload.get("recursive_sha256") != json_sha(rows):
        raise ValueError("invalid source artifact recursive checksum")
    seen: set[str] = set()
    required_hashed = set(REQUIRED) - {"artifact_sha256.json"}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise ValueError("invalid source artifact checksum row")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == "artifact_sha256.json":
            raise ValueError(f"unsafe source artifact path: {row.get('path')}")
        if relative.as_posix() in seen:
            raise ValueError(f"duplicate source artifact path: {row.get('path')}")
        seen.add(relative.as_posix())
        path = root / relative
        if not path.is_file() or ("size" in row and int(path.stat().st_size) != int(row["size"])) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"source artifact checksum mismatch: {row.get('path')}")
    if not required_hashed.issubset(seen):
        raise ValueError(f"source artifact checksum omits required files: {sorted(required_hashed - seen)}")
    return str(payload["recursive_sha256"])


def strict_join(
    root: Path,
    meta: dict[str, Any],
    streams: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    streams = streams or {
        "step_records": load_jsonl(root / "step_records.jsonl"),
        "policy_intent": load_jsonl(root / "policy_intent_records.jsonl"),
        "privileged_sidecar": load_jsonl(root / "privileged_teacher_sidecar.jsonl"),
    }
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    expected_steps: list[int] | None = None
    for name, rows in streams.items():
        if not rows:
            raise ValueError(f"{name} is empty")
        steps = [_step(row, index) for index, row in enumerate(rows)]
        if steps != list(range(len(rows))):
            raise ValueError(f"{name} steps are not contiguous from zero")
        if len(set(steps)) != len(steps):
            raise ValueError(f"{name} contains duplicate steps")
        for row in rows:
            _identity_check(row, meta, source=name)
        current = {step: row for step, row in zip(steps, rows)}
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError(f"strict step join mismatch: {name}")
        indexed[name] = current

    assert expected_steps is not None
    merged = []
    float_merges: list[dict[str, Any]] = []
    for step in expected_steps:
        row = _merge_stream_rows(
            step,
            [
                ("step_records", indexed["step_records"][step]),
                ("policy_intent", indexed["policy_intent"][step]),
                ("privileged_sidecar", indexed["privileged_sidecar"][step]),
            ],
            float_merges,
        )
        for name in IDENTITY_FIELDS:
            row[name] = meta[name]
        features = row.get("features_25d")
        intent = row.get("clean_policy_intent_9d")
        if not isinstance(features, list) or len(features) != 25:
            raise ValueError(f"step {step}: missing 25D student features")
        if not isinstance(intent, list) or len(intent) != 9:
            raise ValueError(f"step {step}: missing 9D policy intent")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in features + intent):
            raise ValueError(f"step {step}: non-finite student feature")
        merged.append(row)
    return merged, float_merges


def _stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for head in HEADS:
        mask_name = f"{head}_mask" if head != "retention_continuation_t10" else "retention_unknown_mask"
        known = positive = negative = 0
        for row in rows:
            masked = row.get(mask_name) is False if head == "retention_continuation_t10" else row.get(mask_name) is True
            value = row.get(head)
            if not masked or value is None:
                continue
            known += 1
            positive += int(bool(value))
            negative += int(not bool(value))
        result[head] = {"known": known, "positive": positive, "negative": negative}
    return result


def load_protocol_config(config_path: Path) -> tuple[dict[str, Any], RetentionConfig]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected B3 protocol schema")
    if payload.get("status") != PROTOCOL_STATUS:
        raise ValueError("B3 protocol is not preparation-only")
    params = payload.get("retention_teacher_parameters")
    required = {
        "n_close",
        "n_open",
        "stability_window",
        "qpos_range_max",
        "opening_range_max",
        "min_transport_steps",
        "min_transport_displacement",
        "release_lookahead",
        "t10",
    }
    if not isinstance(params, dict) or set(params) != required:
        raise ValueError("retention_teacher_parameters are not fully frozen")
    try:
        config = RetentionConfig(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid retention_teacher_parameters") from exc
    if config.t10 != 10 or config.release_lookahead < 1:
        raise ValueError("unsupported B3 retention horizon")
    feature_names = payload.get("feature_names_25d")
    policy_names = payload.get("policy_intent_feature_names_9d")
    if not isinstance(feature_names, list) or len(feature_names) != 25:
        raise ValueError("frozen 25D feature order is missing")
    if not isinstance(policy_names, list) or len(policy_names) != 9:
        raise ValueError("frozen 9D policy feature order is missing")
    return payload, config


def _finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value
    )


def _expected_split(state_id: int) -> str:
    for split, states in OFFICIAL_SPLITS.items():
        if state_id in states:
            return split
    raise ValueError(f"state outside official split: {state_id}")


def verify_source_contract(
    root: Path,
    meta: dict[str, Any],
    summary: dict[str, Any],
    runtime: dict[str, Any],
    condition: dict[str, Any],
    attack: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    if meta.get("schema") != "OPENVLA_OFFICIAL_CLEAN_EPISODE_V2" or meta.get("condition") != "CLEAN":
        raise ValueError("source is not an Official CLEAN V2 artifact")
    if meta.get("runtime_valid") is not True or runtime.get("runtime_valid") is not True:
        raise ValueError("runtime_valid is not true in metadata and runtime audit")
    if meta.get("protocol_id") != protocol.get("official_protocol_id", "OPENVLA_LIBERO_OFFICIAL_V1"):
        raise ValueError("official protocol id mismatch")
    if condition.get("condition") != "CLEAN" or condition.get("protocol_id") != meta.get("protocol_id"):
        raise ValueError("condition config is not official CLEAN")
    if attack.get("attack_enabled") is not False:
        raise ValueError("CLEAN artifact has attack enabled")
    if not isinstance(meta.get("success"), bool) or meta.get("env_success") != meta.get("success"):
        raise ValueError("metadata success semantics are inconsistent")
    if summary.get("clean") is not True or summary.get("success") != meta.get("success"):
        raise ValueError("summary success semantics are inconsistent")
    for key in ("env_reset_called", "checkpoint_binding_pass"):
        if meta.get(key) is not True or runtime.get(key) is not True:
            raise ValueError(f"source contract field {key} is not verified")
    if meta.get("official_execution_adapter") != "OfficialOpenVLAActionAdapter.predict_action":
        raise ValueError("official execution adapter is not predict_action")
    if meta.get("score_adapter") != "OfficialOpenVLAActionAdapter.predict_action_with_scores":
        raise ValueError("official score adapter is not the instrumented predict_action path")
    if meta.get("generation_passes_per_step") != 1:
        raise ValueError("generation passes per step is not exactly one")

    suite = meta.get("suite")
    task_idx = meta.get("task_idx")
    state_id = meta.get("state_id")
    if suite not in OFFICIAL_HORIZONS or not isinstance(task_idx, int) or not 0 <= task_idx < 10:
        raise ValueError("invalid official suite/task identity")
    if not isinstance(state_id, int) or not 0 <= state_id < 50:
        raise ValueError("invalid official state identity")
    if meta.get("split") != _expected_split(state_id):
        raise ValueError("source split does not match frozen state split")
    expected_horizon = protocol.get("official_horizons", OFFICIAL_HORIZONS).get(suite, OFFICIAL_HORIZONS[suite])
    if meta.get("official_horizon") != expected_horizon or runtime.get("official_horizon") != expected_horizon:
        raise ValueError("official horizon mismatch")
    if meta.get("num_steps_wait") != protocol.get("num_steps_wait", 10):
        raise ValueError("official wait-step mismatch")
    initial_sha = meta.get("initial_state_sha256")
    if not isinstance(initial_sha, str) or len(initial_sha) != 64:
        raise ValueError("initial-state SHA is missing or invalid")
    if meta.get("feature_names_25d") != protocol.get("feature_names_25d"):
        raise ValueError("metadata 25D feature order mismatch")
    if meta.get("policy_intent_feature_names_9d") != protocol.get("policy_intent_feature_names_9d"):
        raise ValueError("metadata 9D policy feature order mismatch")

    for name, value in (("task_language", meta.get("task_language")), ("canonical_parent_key", meta.get("canonical_parent_key"))):
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata missing {name}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def verify_step_contract(
    steps: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    sidecar: list[dict[str, Any]],
) -> None:
    if not steps or len(steps) != len(intents) or len(steps) != len(sidecar):
        raise ValueError("source stream counts are inconsistent")
    for index, (step, intent, privileged) in enumerate(zip(steps, intents, sidecar)):
        if step.get("official_execution") is not True:
            raise ValueError(f"step {index}: official execution flag missing")
        if not _finite_vector(step.get("features_25d"), 25) or not _finite_vector(step.get("clean_policy_intent_9d"), 9):
            raise ValueError(f"step {index}: invalid student feature vector")
        if not _finite_vector(step.get("clean_action_raw_7d"), 7) or not _finite_vector(step.get("applied_action_7d"), 7):
            raise ValueError(f"step {index}: invalid raw/env action vector")
        if not isinstance(step.get("action_token_ids"), list) or len(step["action_token_ids"]) != 7:
            raise ValueError(f"step {index}: action-token count is not seven")
        if not isinstance(step.get("score_head_summary"), list) or len(step["score_head_summary"]) != 7:
            raise ValueError(f"step {index}: score count is not seven")
        if step.get("score_adapter_parity_pass") is not True:
            raise ValueError(f"step {index}: score/action parity is not verified")
        if (
            not isinstance(step.get("score_adapter_action_max_abs_error"), (int, float))
            or not math.isfinite(float(step["score_adapter_action_max_abs_error"]))
            or float(step["score_adapter_action_max_abs_error"]) > 1e-6
        ):
            raise ValueError(f"step {index}: score/action error exceeds tolerance")
        if not _finite_vector(intent.get("clean_policy_intent_9d"), 9):
            raise ValueError(f"intent {index}: invalid policy vector")
        if intent.get("action_token_ids") != step.get("action_token_ids"):
            raise ValueError(f"intent {index}: action-token mismatch")
        if intent.get("score_adapter_parity_pass") is not True:
            raise ValueError(f"intent {index}: score/action parity is not verified")
        if not isinstance(privileged.get("robot0_eef_pos"), list) or len(privileged["robot0_eef_pos"]) != 3:
            raise ValueError(f"sidecar {index}: missing EEF position")
        if not isinstance(privileged.get("robot0_gripper_qpos"), list) or len(privileged["robot0_gripper_qpos"]) < 2:
            raise ValueError(f"sidecar {index}: missing gripper qpos")


def _write_sha_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def _write_output_checksums(root: Path, manifest_name: str = "materialization_manifest.json") -> None:
    manifest = root / manifest_name
    manifest_sidecar = root / f"{manifest_name}.sha256"
    _write_sha_sidecar(manifest)
    names = sorted(
        path.name for path in root.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(root / name)}  {name}\n" for name in names), encoding="utf-8")
    _write_sha_sidecar(sums)


def materialize(artifact_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ValueError(f"missing B3 protocol config: {config_path}")
    protocol, retention_config = load_protocol_config(config_path)
    meta = json.loads((artifact_root / "episode_metadata.json").read_text(encoding="utf-8"))
    missing_identity = [name for name in IDENTITY_FIELDS if name not in meta]
    if missing_identity:
        raise ValueError(f"source metadata missing identity fields: {missing_identity}")
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
        protocol,
    )
    source_sha = verify_source_artifact(artifact_root)
    streams = {
        "step_records": load_jsonl(artifact_root / "step_records.jsonl"),
        "policy_intent": load_jsonl(artifact_root / "policy_intent_records.jsonl"),
        "privileged_sidecar": load_jsonl(artifact_root / "privileged_teacher_sidecar.jsonl"),
    }
    verify_step_contract(streams["step_records"], streams["policy_intent"], streams["privileged_sidecar"])
    merged, float_merges = strict_join(artifact_root, meta, streams)
    rebuilt = rebuild_retention_features(merged, retention_config)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"materialization output is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    identity = {name: meta.get(name) for name in ("suite", "task_idx", "state_id", "canonical_parent_key")}
    student_rows = [
        {**identity, "step": row["step"], "features_25d": row["features_25d"], "clean_policy_intent_9d": row["clean_policy_intent_9d"]}
        for row in merged
    ]
    teacher_rows = [
        {**identity, **{key: value for key, value in row.items() if key not in {"object_state", "mujoco_contact_pairs"}}}
        for row in rebuilt["rows"]
    ]
    _write_jsonl(output_root / "student_input_records.jsonl", student_rows)
    _write_jsonl(output_root / "teacher_retention_records.jsonl", teacher_rows)
    (output_root / "retention_events.json").write_text(json.dumps(rebuilt["events"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_files = ["student_input_records.jsonl", "teacher_retention_records.jsonl", "retention_events.json"]
    file_rows = [{"path": name, "size": (output_root / name).stat().st_size, "sha256": sha256_file(output_root / name)} for name in output_files]
    manifest = {
        "schema": "B3_RETENTION_MATERIALIZED_EPISODE_V1",
        "source_schema": "OFFICIAL_25D_V1",
        "derived_schema": rebuilt["schema"],
        "source_contract_verified": True,
        "official_protocol_id": meta["protocol_id"],
        "source_artifact_sha256": source_sha,
        "source_identity": identity,
        "config_sha256": sha256_file(config_path),
        "effective_retention_config": asdict(retention_config),
        "rebuilder_sha256": sha256_file(REPO_ROOT / "src" / "gripper_attack" / "b3_retention.py"),
        "materializer_sha256": sha256_file(Path(__file__).resolve()),
        "step_count": len(teacher_rows),
        "label_statistics": _stats(rebuilt["rows"]),
        "head_roles": {
            "grasp_support": "TRAINING_AUXILIARY",
            "retention_active": "RUNTIME_PRIMARY",
            "retention_continuation_t10": "RUNTIME_PRIMARY",
            "release_imminent": "RUNTIME_PRIMARY",
        },
        "mask_semantics": {
            "*_mask": "true_means_known",
            "retention_unknown_mask": "true_means_unknown",
        },
        "unknown_is_negative": False,
        "join_float_tolerance_merges": float_merges,
        "label_semantics": "ROBOT_CENTRIC_PROXY_LABELS_NOT_GRASP_GROUND_TRUTH",
        "files": file_rows,
        "output_recursive_sha256": json_sha(file_rows),
        "student_forbidden_fields_absent": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }
    (output_root / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_output_checksums(output_root)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(args.artifact_root, args.output_root, args.config)
    print(json.dumps({"status": "PASS", "schema": manifest["schema"], "step_count": manifest["step_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
