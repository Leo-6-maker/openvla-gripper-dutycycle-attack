#!/usr/bin/env python3
"""Fail-closed audit for one B3-Retention materialized episode."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from materialize_b3_retention_episode import HEADS, json_sha, load_jsonl, sha256_file  # noqa: E402


FORBIDDEN_STUDENT_FIELDS = {"object_state", "mujoco_contact_pairs", "attack_outcome", "event_id", "event_ordinal"}


def audit(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "materialization_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "B3_RETENTION_MATERIALIZED_EPISODE_V1":
        raise ValueError("unexpected materialization schema")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("output_recursive_sha256") != json_sha(files):
        raise ValueError("invalid materialization recursive checksum")
    if manifest.get("formal_training_ready") is not False or manifest.get("formal_attack_ready") is not False:
        raise ValueError("materialization cannot advertise formal training or attack readiness")
    if manifest.get("source_contract_verified") is not True or manifest.get("unknown_is_negative") is not False:
        raise ValueError("source contract or unknown-mask boundary is not closed")
    if manifest.get("label_semantics") != "ROBOT_CENTRIC_PROXY_LABELS_NOT_GRASP_GROUND_TRUTH":
        raise ValueError("proxy-label semantics are not explicit")
    effective = manifest.get("effective_retention_config")
    if not isinstance(effective, dict) or effective.get("t10") != 10 or effective.get("release_lookahead") != 3:
        raise ValueError("effective retention config is incomplete")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, dict) or any(
        source_identity.get(name) in (None, "")
        for name in ("suite", "task_idx", "state_id", "canonical_parent_key")
    ):
        raise ValueError("missing source identity in materialization manifest")
    source_sha = manifest.get("source_artifact_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("invalid source artifact SHA")
    for row in files:
        path = root / str(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"output checksum mismatch: {row.get('path')}")
    manifest_sidecar = root / "materialization_manifest.json.sha256"
    sums = root / "SHA256SUMS"
    sums_sidecar = root / "SHA256SUMS.sha256"
    if not manifest_sidecar.is_file() or not sums.is_file() or not sums_sidecar.is_file():
        raise ValueError("materialization checksum sidecars are incomplete")
    if manifest_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(root / 'materialization_manifest.json')}  materialization_manifest.json":
        raise ValueError("manifest SHA sidecar mismatch")
    if sums_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError("SHA256SUMS sidecar mismatch")
    sum_rows = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not digest or not name:
            raise ValueError("invalid SHA256SUMS row")
        sum_rows[name] = digest
    expected_sum_names = {str(row["path"]) for row in files} | {
        "materialization_manifest.json",
        "materialization_manifest.json.sha256",
    }
    if set(sum_rows) != expected_sum_names:
        raise ValueError("SHA256SUMS file set mismatch")
    for name, digest in sum_rows.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"SHA256SUMS digest mismatch: {name}")

    student = load_jsonl(root / "student_input_records.jsonl")
    teacher = load_jsonl(root / "teacher_retention_records.jsonl")
    if len(student) != len(teacher) or len(student) != int(manifest["step_count"]):
        raise ValueError("student/teacher step count mismatch")
    for student_row, teacher_row in zip(student, teacher):
        if FORBIDDEN_STUDENT_FIELDS.intersection(student_row):
            raise ValueError("privileged/event field leaked into student record")
        if student_row.get("step") != teacher_row.get("step"):
            raise ValueError("student/teacher step mismatch")
        for name in ("suite", "task_idx", "state_id", "canonical_parent_key"):
            if student_row.get(name) != source_identity.get(name) or teacher_row.get(name) != source_identity.get(name):
                raise ValueError(f"identity mismatch for {name}")
        features = student_row.get("features_25d")
        intent = student_row.get("clean_policy_intent_9d")
        if not isinstance(features, list) or len(features) != 25 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in features
        ):
            raise ValueError("invalid student 25D row")
        if not isinstance(intent, list) or len(intent) != 9 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in intent
        ):
            raise ValueError("invalid student 9D row")
        for head in HEADS:
            if head not in teacher_row:
                raise ValueError(f"missing teacher head {head}")
        for mask in ("grasp_support_mask", "retention_active_mask", "retention_unknown_mask", "release_imminent_mask"):
            if not isinstance(teacher_row.get(mask), bool):
                raise ValueError(f"invalid teacher mask {mask}")

    return {
        "status": "PASS",
        "schema": manifest["schema"],
        "step_count": len(student),
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "label_statistics": manifest.get("label_statistics", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.materialized_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
