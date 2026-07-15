#!/usr/bin/env python3
"""Strict, offline B3-Retention materialization for one sealed CLEAN episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.b3_retention import rebuild_retention_features  # noqa: E402


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
        if name not in row:
            raise ValueError(f"{source} missing identity field {name}")
        expected = meta.get(name)
        actual = row.get(name)
        if name in {"task_idx", "state_id"}:
            try:
                expected, actual = int(expected), int(actual)
            except (TypeError, ValueError):
                pass
        if actual != expected:
            raise ValueError(f"{source} identity mismatch for {name}: {actual!r} != {expected!r}")


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
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"source artifact checksum mismatch: {row.get('path')}")
    if not required_hashed.issubset(seen):
        raise ValueError(f"source artifact checksum omits required files: {sorted(required_hashed - seen)}")
    return str(payload["recursive_sha256"])


def strict_join(root: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    streams = {
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
    for step in expected_steps:
        row = dict(indexed["step_records"][step])
        row.update(indexed["policy_intent"][step])
        row.update(indexed["privileged_sidecar"][step])
        row["step"] = step
        features = row.get("features_25d")
        intent = row.get("clean_policy_intent_9d")
        if not isinstance(features, list) or len(features) != 25:
            raise ValueError(f"step {step}: missing 25D student features")
        if not isinstance(intent, list) or len(intent) != 9:
            raise ValueError(f"step {step}: missing 9D policy intent")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in features + intent):
            raise ValueError(f"step {step}: non-finite student feature")
        merged.append(row)
    return merged


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def materialize(artifact_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ValueError(f"missing B3 protocol config: {config_path}")
    meta = json.loads((artifact_root / "episode_metadata.json").read_text(encoding="utf-8"))
    if meta.get("schema") != "OPENVLA_OFFICIAL_CLEAN_EPISODE_V2" or meta.get("condition") != "CLEAN":
        raise ValueError("source is not an Official CLEAN V2 artifact")
    if meta.get("runtime_valid") is not True:
        raise ValueError("source runtime_valid is not true")
    missing_identity = [name for name in IDENTITY_FIELDS if name not in meta]
    if missing_identity:
        raise ValueError(f"source metadata missing identity fields: {missing_identity}")
    source_sha = verify_source_artifact(artifact_root)
    merged = strict_join(artifact_root, meta)
    rebuilt = rebuild_retention_features(merged)
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
        "source_artifact_sha256": source_sha,
        "source_identity": identity,
        "config_sha256": sha256_file(config_path),
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
        "files": file_rows,
        "output_recursive_sha256": json_sha(file_rows),
        "student_forbidden_fields_absent": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }
    (output_root / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
