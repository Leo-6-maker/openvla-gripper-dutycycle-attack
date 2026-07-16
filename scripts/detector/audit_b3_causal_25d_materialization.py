#!/usr/bin/env python3
"""Fail-closed audit for one causal-25D FIT materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from audit_b3_teacher_invariants import audit_episode  # noqa: E402
from materialize_b3_causal_25d_episode import sha256_file  # noqa: E402
from gripper_attack.b3_causal_25d import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA as CAUSAL_SCHEMA,
    SOURCE_SCHEMA,
    STUDENT_FORBIDDEN_FEATURE_NAMES,
    serialize_student_25d,
)


OUTPUT_SCHEMA = "B3_CAUSAL_25D_S1_MATERIALIZED_EPISODE_V1"
MODE = "fit-label-materialization-25d-causal"
STUDENT_FORBIDDEN_TOP_LEVEL = {
    "clean_policy_intent_9d",
    "object_state",
    "mujoco_contact_pairs",
    "attack_outcome",
    "event_id",
    "event_ordinal",
    "retention_continuation_t10",
    "retention_unknown_mask",
    "release_imminent",
}


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected object")
            rows.append(value)
    return rows


def _validate_checksums(root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("output_recursive_sha256") != _json_sha(files):
        raise ValueError("invalid materialization output checksum manifest")
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("invalid materialization output file row")
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"materialization output checksum mismatch: {row.get('path')}")
        if int(path.stat().st_size) != int(row.get("size", -1)):
            raise ValueError(f"materialization output size mismatch: {row.get('path')}")

    manifest_sidecar = root / "materialization_manifest.json.sha256"
    sums = root / "SHA256SUMS"
    sums_sidecar = root / "SHA256SUMS.sha256"
    if not manifest_sidecar.is_file() or not sums.is_file() or not sums_sidecar.is_file():
        raise ValueError("materialization checksum sidecars are incomplete")
    expected_manifest_sidecar = f"{sha256_file(root / 'materialization_manifest.json')}  materialization_manifest.json"
    if manifest_sidecar.read_text(encoding="utf-8").strip() != expected_manifest_sidecar:
        raise ValueError("materialization manifest SHA sidecar mismatch")
    if sums_sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError("SHA256SUMS sidecar mismatch")
    expected_names = {str(row["path"]) for row in files} | {
        "materialization_manifest.json",
        "materialization_manifest.json.sha256",
    }
    sum_rows = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not digest or not name:
            raise ValueError("invalid SHA256SUMS row")
        sum_rows[name] = digest
    if set(sum_rows) != expected_names:
        raise ValueError("SHA256SUMS file set mismatch")
    for name, digest in sum_rows.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"SHA256SUMS digest mismatch: {name}")


def _validate_student(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    identity = manifest.get("source_identity")
    if not isinstance(identity, dict):
        raise ValueError("missing source identity")
    if len(rows) != manifest.get("step_count"):
        raise ValueError("student step count mismatch")
    for index, row in enumerate(rows):
        if row.get("step") != index:
            raise ValueError(f"student steps are not contiguous at {index}")
        for name in ("suite", "task_idx", "state_id", "canonical_parent_key"):
            if row.get(name) != identity.get(name):
                raise ValueError(f"student identity mismatch for {name}")
        if STUDENT_FORBIDDEN_TOP_LEVEL.intersection(row):
            raise ValueError(f"student privileged/event field leaked at step {index}")
        projection = {name: row.get(name) for name in ("schema", "source_schema", "valid", "features_25d")}
        if set(row) - set(identity) - {"step", "schema", "source_schema", "valid", "features_25d"}:
            raise ValueError(f"student row has unregistered fields at step {index}")
        serialize_student_25d(projection)
        if row.get("schema") != CAUSAL_SCHEMA or row.get("source_schema") != SOURCE_SCHEMA:
            raise ValueError(f"student causal schema mismatch at step {index}")
        values = row.get("features_25d")
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"student causal features invalid at step {index}")
    if manifest.get("student_feature_names") != list(FEATURE_NAMES):
        raise ValueError("student feature order is not bound")
    if manifest.get("student_projection_keys") != ["schema", "source_schema", "valid", "features_25d"]:
        raise ValueError("student projection contract is not strict")
    if any(name in STUDENT_FORBIDDEN_FEATURE_NAMES for name in FEATURE_NAMES):
        raise ValueError("forbidden feature name entered causal student")


def _validate_events(path: Path, expected_count: Any) -> None:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list) or len(events) != expected_count:
        raise ValueError("causal event summary count mismatch")
    previous_end = -1
    for expected_id, event in enumerate(events):
        if event.get("event_id") != expected_id:
            raise ValueError("causal event IDs are not contiguous")
        start, end = event.get("start_step"), event.get("end_step")
        if not isinstance(start, int) or not isinstance(end, int) or end < start or start <= previous_end:
            raise ValueError("causal event interval is invalid or overlapping")
        previous_end = end


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "materialization_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != OUTPUT_SCHEMA or manifest.get("mode") != MODE:
        raise ValueError("unexpected causal FIT materialization schema")
    if manifest.get("formal_training_ready") is not False or manifest.get("formal_attack_ready") is not False:
        raise ValueError("causal S1 materialization cannot advertise training or attack readiness")
    if manifest.get("source_unchanged") is not True:
        raise ValueError("source changed during causal materialization")
    if manifest.get("student_policy_intent_read") is not False or manifest.get("student_policy_intent_present") is not False:
        raise ValueError("causal student path is not 25D-only")
    if manifest.get("unknown_is_negative") is not False:
        raise ValueError("unknown labels cannot be treated as negative")
    _validate_checksums(root, manifest)
    student = _load_jsonl(root / "student_input_records.jsonl")
    _validate_student(student, manifest)
    teacher = _load_jsonl(root / "teacher_retention_records.jsonl")
    if len(teacher) != len(student):
        raise ValueError("teacher/student step count mismatch")
    teacher_audit = audit_episode(teacher, json.loads((root / "retention_events.json").read_text(encoding="utf-8")))
    if teacher_audit.get("status") != "PASS":
        raise ValueError(f"Teacher invariant audit failed: {teacher_audit.get('violations')}")
    _validate_events(root / "causal_event_summary.json", manifest.get("causal_event_count"))
    return {
        "schema": "B3_CAUSAL_25D_S1_MATERIALIZATION_AUDIT_V1",
        "status": "PASS",
        "materialized_root": str(root),
        "source_identity": manifest["source_identity"],
        "step_count": len(student),
        "causal_event_count": manifest["causal_event_count"],
        "teacher_event_count": manifest["teacher_event_count"],
        "student_policy_intent_read": False,
        "student_projection_pass": True,
        "teacher_invariant_pass": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.materialized_root)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"audit output already exists: {output}")
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.with_name(output.name + ".sha256").write_text(
            f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
        )
    print(json.dumps({"status": report["status"], "step_count": report["step_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
