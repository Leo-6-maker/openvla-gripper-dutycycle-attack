#!/usr/bin/env python3
"""Audit official CLEAN artifacts for detector-retraining input completeness."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gripper_attack.official_detector_features import CLEAN_POLICY_FEATURE_NAMES, CANONICAL_25D_FEATURES


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
        meta = read_json(path / "episode_metadata.json")
        steps = read_jsonl(path / "step_records.jsonl")
        intents = read_jsonl(path / "policy_intent_records.jsonl")
    except Exception as exc:
        return {"episode": str(path), "input_schema_pass": False, "errors": [str(exc)]}

    for key in (
        "protocol_id",
        "canonical_parent_key",
        "initial_state_sha256",
        "official_horizon",
        "feature_names_25d",
        "policy_intent_feature_names_9d",
        "privileged_teacher_sidecar",
    ):
        if key not in meta:
            errors.append(f"metadata_missing:{key}")
    if meta.get("detector_retraining_input_ready") is not True:
        errors.append("metadata_detector_retraining_input_not_ready")
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
    for index, row in enumerate(steps):
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
    for index, row in enumerate(intents):
        if not finite_vector(row.get("clean_policy_intent_9d"), len(CLEAN_POLICY_FEATURE_NAMES)):
            errors.append(f"intent_{index}_policy_intent_9d_invalid")

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
        "schema": "OPENVLA_OFFICIAL_CLEAN_SCHEMA_AUDIT_V1",
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
