#!/usr/bin/env python3
"""Audit official CLEAN artifacts for detector-retraining input completeness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gripper_attack.official_detector_features import CLEAN_POLICY_FEATURE_NAMES, CANONICAL_25D_FEATURES


REQUIRED_ARTIFACT_FILES = (
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
OFFICIAL_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def artifact_checksum_valid(path: Path) -> bool:
    manifest = path / "artifact_sha256.json"
    if not manifest.is_file():
        return False
    try:
        payload = read_json(manifest)
        rows = payload["files"]
        paths = {str(row["path"]) for row in rows}
        if not (set(REQUIRED_ARTIFACT_FILES) - {"artifact_sha256.json"}) <= paths:
            return False
        if payload["recursive_sha256"] != json_sha(rows):
            return False
        return all(
            (path / row["path"]).is_file()
            and int((path / row["path"]).stat().st_size) == int(row["size"])
            and sha256_file(path / row["path"]) == row["sha256"]
            for row in rows
        )
    except Exception:
        return False


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(value)
    return rows


def finite_vector(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError):
        return False


def audit_episode(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        missing = [name for name in REQUIRED_ARTIFACT_FILES if not (path / name).is_file()]
        if missing:
            errors.extend(f"artifact_missing:{name}" for name in missing)
        meta = read_json(path / "episode_metadata.json")
        summary = read_json(path / "episode_summary.json")
        runtime = read_json(path / "runtime_audit.json")
        condition = read_json(path / "condition_config.json")
        attack = read_json(path / "attack_config.json")
        steps = read_jsonl(path / "step_records.jsonl")
        intents = read_jsonl(path / "policy_intent_records.jsonl")
        privileged = read_jsonl(path / "privileged_teacher_sidecar.jsonl")
    except Exception as exc:
        return {"episode": str(path), "input_schema_pass": False, "errors": [str(exc)]}

    if not artifact_checksum_valid(path):
        errors.append("artifact_checksum_invalid")

    for key in (
        "protocol_id",
        "canonical_parent_key",
        "initial_state_sha256",
        "official_horizon",
        "feature_names_25d",
        "policy_intent_feature_names_9d",
        "privileged_teacher_sidecar",
        "env_reset_called",
        "checkpoint_binding_pass",
        "single_generation_parity_pass",
        "generation_passes_per_step",
        "runtime_valid",
        "env_success",
        "success",
        "student_allowed_modalities",
        "student_forbidden_modalities",
    ):
        if key not in meta:
            errors.append(f"metadata_missing:{key}")
    if meta.get("detector_retraining_input_ready") is not True:
        errors.append("metadata_detector_retraining_input_not_ready")
    if meta.get("condition") != "CLEAN" or condition.get("condition") != "CLEAN":
        errors.append("condition_not_clean")
    if meta.get("runtime_valid") is not True or runtime.get("runtime_valid") is not True:
        errors.append("runtime_not_valid")
    if not isinstance(meta.get("success"), bool) or meta.get("env_success") != meta.get("success"):
        errors.append("success_semantics_invalid")
    if summary.get("clean") is not True or summary.get("success") != meta.get("success"):
        errors.append("summary_semantics_invalid")
    if attack.get("attack_enabled") is not False:
        errors.append("clean_attack_enabled")
    if meta.get("env_reset_called") is not True:
        errors.append("env_reset_not_recorded")
    if meta.get("checkpoint_binding_pass") is not True:
        errors.append("checkpoint_binding_not_verified")
    if meta.get("single_generation_parity_pass") is not True:
        errors.append("single_generation_parity_not_verified")
    if meta.get("generation_passes_per_step") != 1:
        errors.append("generation_pass_count_not_one")
    if not isinstance(meta.get("official_horizon"), int) or meta["official_horizon"] <= 0:
        errors.append("official_horizon_invalid")
    if meta.get("official_horizon") != OFFICIAL_HORIZONS.get(meta.get("suite")):
        errors.append("official_horizon_mismatch")
    if meta.get("num_steps_wait") != 10:
        errors.append("num_steps_wait_mismatch")
    if not isinstance(meta.get("task_idx"), int) or not 0 <= meta["task_idx"] < 10:
        errors.append("task_identity_invalid")
    if not isinstance(meta.get("state_id"), int) or not 0 <= meta["state_id"] < 50:
        errors.append("state_identity_invalid")
    if not meta.get("canonical_parent_key") or not meta.get("initial_state_sha256"):
        errors.append("identity_incomplete")
    if tuple(meta.get("student_allowed_modalities", ())) != ("features_25d", "clean_policy_intent_9d", "task_language"):
        errors.append("student_allowed_modalities_mismatch")
    if tuple(meta.get("student_forbidden_modalities", ())) != ("object_state", "mujoco_contact_pairs", "attack_outcome"):
        errors.append("student_forbidden_modalities_mismatch")
    if tuple(meta.get("feature_names_25d", ())) != tuple(CANONICAL_25D_FEATURES):
        errors.append("metadata_feature_order_mismatch")
    if tuple(meta.get("policy_intent_feature_names_9d", ())) != tuple(CLEAN_POLICY_FEATURE_NAMES):
        errors.append("metadata_policy_feature_order_mismatch")

    if not steps:
        errors.append("step_records_empty")
    if len(steps) != len(intents):
        errors.append("step_intent_count_mismatch")
    if [row.get("step") for row in steps] != [row.get("step") for row in intents]:
        errors.append("step_intent_identity_mismatch")
    if [row.get("step") for row in steps] != list(range(len(steps))):
        errors.append("step_index_not_contiguous")
    if len(privileged) != len(steps):
        errors.append("step_privileged_count_mismatch")
    identity = (meta.get("suite"), meta.get("task_idx"), meta.get("state_id"))
    for index, row in enumerate(steps):
        if (row.get("suite"), row.get("task_idx"), row.get("state_id")) != identity:
            errors.append(f"step_{index}_identity_mismatch")
        if not finite_vector(row.get("features_25d"), len(CANONICAL_25D_FEATURES)):
            errors.append(f"step_{index}_features_25d_invalid")
        if not finite_vector(row.get("clean_policy_intent_9d"), len(CLEAN_POLICY_FEATURE_NAMES)):
            errors.append(f"step_{index}_policy_intent_9d_invalid")
        if not finite_vector(row.get("clean_action_raw_7d"), 7):
            errors.append(f"step_{index}_raw_action_invalid")
        if not finite_vector(row.get("applied_action_7d"), 7):
            errors.append(f"step_{index}_applied_action_invalid")
        if not isinstance(row.get("clean_action_token_top_ids"), list) or not row["clean_action_token_top_ids"]:
            errors.append(f"step_{index}_top_token_ids_missing")
        if not isinstance(row.get("clean_action_token_top_logits"), list) or not row["clean_action_token_top_logits"]:
            errors.append(f"step_{index}_top_token_logits_missing")
        if not isinstance(row.get("action_token_ids"), list) or len(row["action_token_ids"]) != 7:
            errors.append(f"step_{index}_action_token_ids_not_7")
        if row.get("score_adapter_parity_pass") is not True or row.get("single_generation_parity_pass") is not True:
            errors.append(f"step_{index}_action_parity_not_verified")
        if row.get("generation_passes_per_step") != 1:
            errors.append(f"step_{index}_generation_pass_count_not_one")
        if any(key in row for key in ("object_state", "mujoco_contact_pairs", "attack_outcome")):
            errors.append(f"step_{index}_student_forbidden_field_present")
        try:
            if float(row.get("score_adapter_action_max_abs_error")) > 1e-6:
                errors.append(f"step_{index}_action_parity_error")
        except (TypeError, ValueError):
            errors.append(f"step_{index}_action_parity_error_missing")
    for index, row in enumerate(intents):
        if row.get("step") != index:
            errors.append(f"intent_{index}_step_identity_mismatch")
        if not finite_vector(row.get("clean_policy_intent_9d"), len(CLEAN_POLICY_FEATURE_NAMES)):
            errors.append(f"intent_{index}_policy_intent_9d_invalid")
        if not isinstance(row.get("action_token_ids"), list) or len(row["action_token_ids"]) != 7:
            errors.append(f"intent_{index}_action_token_ids_not_7")
        if row.get("score_adapter_parity_pass") is not True:
            errors.append(f"intent_{index}_action_parity_not_verified")

    input_pass = not errors
    return {
        "episode": str(path),
        "canonical_parent_key": meta.get("canonical_parent_key", ""),
        "input_schema_pass": input_pass,
        "teacher_labels_materialized": bool(meta.get("teacher_labels_materialized", False)),
        "steps": len(steps),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.episode_root.resolve()
    metadata = [root / "episode_metadata.json"] if (root / "episode_metadata.json").is_file() else sorted(root.rglob("episode_metadata.json"))
    episodes = [audit_episode(path.parent) for path in metadata]
    report = {
        "schema": "OPENVLA_OFFICIAL_CLEAN_SCHEMA_AUDIT_V2",
        "episodes": len(episodes),
        "input_schema_pass": bool(episodes) and all(bool(row["input_schema_pass"]) for row in episodes),
        "teacher_labels_materialized_count": sum(bool(row["teacher_labels_materialized"]) for row in episodes),
        "rows": episodes,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["input_schema_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
