#!/usr/bin/env python3
"""Read-only, split-scoped Factorized V2 runtime bundle builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory  # noqa: E402
from gripper_attack.factorized_runtime import (  # noqa: E402
    FactorizedRuntimeError,
    build_runtime_record,
    exact_runtime_step_join,
    validate_runtime_record,
)

EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stream(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FactorizedRuntimeError(f"STREAM_MISSING:{'/'.join(names)}")


def _identity(row: dict[str, Any]) -> str:
    value = row.get("episode", row.get("canonical_parent_key"))
    if not isinstance(value, str) or value.count("/") != 2:
        raise FactorizedRuntimeError("IDENTITY_MISSING")
    return value


def _episode_root(root: Path, identity: str, required: str) -> Path:
    suite, task, state = identity.split("/")
    candidates = [root / suite / task / state, root / "episodes" / suite / task / state, root / "episodes" / identity]
    matches = [path for path in candidates if (path / required).is_file()]
    if len(matches) != 1:
        raise FactorizedRuntimeError(f"EPISODE_ROOT_AMBIGUOUS:{identity}:{len(matches)}")
    return matches[0]


def _source_binding(root: Path) -> dict[str, Any]:
    for name in ("source_binding.json", "input_manifest.json", "manifest.json"):
        path = root / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    raise FactorizedRuntimeError("SOURCE_BINDING_MISSING")


def _find(binding: Any, names: set[str]) -> Any:
    pending = [binding]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in names:
                    return value
                pending.append(value)
        elif isinstance(item, list):
            pending.extend(item)
    return None


def _spec(value: str) -> dict[str, Any]:
    name, separator, path = value.partition("=")
    if not separator or name not in EXPECTED_SPLITS or not path:
        raise FactorizedRuntimeError("SPLIT_SPEC_INVALID")
    item = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise FactorizedRuntimeError("SPLIT_SPEC_OBJECT_REQUIRED")
    item = dict(item)
    item["name"] = name
    required = {"name", "prediction_root", "student_root", "runtime_root", "checkpoint", "source_commit", "feature_order_sha256"}
    if set(item) != required:
        raise FactorizedRuntimeError("SPLIT_SPEC_FIELD_SET")
    return item


def _validate_source_manifest(root: Path, binding: dict[str, Any], *, expected_feature: str, expected_runtime_seal: str | None = None) -> None:
    actual_feature = _find(binding, {"feature_order_sha256", "student_feature_order_sha256"})
    if actual_feature is None:
        raise FactorizedRuntimeError("STUDENT_FEATURE_ORDER_BINDING_MISSING")
    if str(actual_feature).lower() != expected_feature.lower():
        raise FactorizedRuntimeError("FEATURE_ORDER_MANIFEST_MISMATCH")
    if expected_runtime_seal is not None:
        declared = _find(binding, {"runtime_action_seal", "runtime_artifact_seal", "action_runtime_seal"})
        if declared is not None and str(declared).lower() != expected_runtime_seal.lower():
            raise FactorizedRuntimeError("RUNTIME_ACTION_SEAL_MISMATCH")
        certified = _find(binding, {"runtime_action_semantic_certified", "action_semantic_certified", "action_semantics_parity_pass"})
        if certified is not True:
            raise FactorizedRuntimeError("RUNTIME_ACTION_SEMANTIC_CERTIFICATION_MISSING")


def _prepare_split(item: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    prediction_root = Path(item["prediction_root"]).resolve()
    student_root = Path(item["student_root"]).resolve()
    runtime_root = Path(item["runtime_root"]).resolve()
    checkpoint = Path(item["checkpoint"]).resolve()
    for root in (prediction_root, student_root, runtime_root):
        verify_sealed_directory(root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    prediction_binding = _source_binding(prediction_root)
    if str(prediction_binding.get("checkpoint_sha256", "")).lower() != checkpoint_sha:
        raise FactorizedRuntimeError("CHECKPOINT_SHA_MISMATCH")
    if prediction_binding.get("source_commit") != item["source_commit"]:
        raise FactorizedRuntimeError("SOURCE_COMMIT_MISMATCH")
    prediction_seal = sha256_file(prediction_root / "SHA256SUMS")
    student_seal = sha256_file(student_root / "SHA256SUMS")
    runtime_seal = sha256_file(runtime_root / "SHA256SUMS")
    student_binding = _source_binding(student_root)
    runtime_manifest = _source_binding(runtime_root)
    _validate_source_manifest(student_root, student_binding, expected_feature=item["feature_order_sha256"])
    _validate_source_manifest(runtime_root, runtime_manifest, expected_feature=item["feature_order_sha256"], expected_runtime_seal=runtime_seal)
    predictions = _jsonl(_stream(prediction_root, ("prediction_records.jsonl", "factorized_step_predictions.jsonl", "heldout_step_predictions.jsonl")))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(_identity(row), []).append(row)
    output: list[dict[str, Any]] = []
    for identity, prediction_rows in sorted(grouped.items()):
        student_episode = _episode_root(student_root, identity, "student_input_records.jsonl")
        runtime_episode = _episode_root(runtime_root, identity, "step_records.jsonl")
        joined = exact_runtime_step_join(prediction_rows, _jsonl(student_episode / "student_input_records.jsonl"), _jsonl(runtime_episode / "step_records.jsonl"))
        for prediction, student, runtime in joined:
            row = build_runtime_record(
                prediction, student, runtime,
                checkpoint_sha256=checkpoint_sha,
                source_commit=item["source_commit"],
                prediction_artifact_seal=prediction_seal,
                runtime_artifact_seal=runtime_seal,
                feature_order_sha256=item["feature_order_sha256"],
                runtime_manifest=runtime_manifest,
            )
            row["split"] = item["name"]
            validate_runtime_record(row)
            output.append(row)
    return item["name"], output, {
        "split": item["name"],
        "checkpoint_sha256": checkpoint_sha,
        "source_commit": item["source_commit"],
        "feature_order_sha256": item["feature_order_sha256"],
        "prediction_seal": prediction_seal,
        "student_input_seal": student_seal,
        "runtime_action_seal": runtime_seal,
        "identity_count": len(grouped),
        "record_count": len(output),
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_enabled": False,
    }


def _seal_tree(root: Path) -> None:
    names = []
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).parts[0] not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            names.append(path.relative_to(root).as_posix())
    names.sort()
    sums = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def verify_sealed_tree(root: Path) -> None:
    root = root.resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise FactorizedRuntimeError("TREE_SEAL_INVALID")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        path = Path(name)
        if not separator or len(digest) != 64 or path.is_absolute() or ".." in path.parts or name in listed:
            raise FactorizedRuntimeError("TREE_CHECKSUM_ROW_INVALID")
        target = root / path
        if not target.is_file() or sha256_file(target) != digest.lower():
            raise FactorizedRuntimeError("TREE_CHECKSUM_MISMATCH")
        listed.add(path.as_posix())
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != listed | {"SHA256SUMS", "SHA256SUMS.sha256"}:
        raise FactorizedRuntimeError("TREE_FILE_SET_MISMATCH")


def rematerialize(items: list[dict[str, Any]], output_root: Path, *, require_twelve: bool = True) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output_root}")
    names = [item.get("name") for item in items]
    if len(names) != len(set(names)):
        raise FactorizedRuntimeError("DUPLICATE_SPLIT_NAME")
    if any(name not in EXPECTED_SPLITS for name in names):
        raise FactorizedRuntimeError("UNEXPECTED_SPLIT_NAME")
    if require_twelve and tuple(sorted(names)) != EXPECTED_SPLITS:
        raise FactorizedRuntimeError("EXACT_12_SPLIT_CLOSURE_REQUIRED")
    prepared = [_prepare_split(item) for item in sorted(items, key=lambda value: value["name"])]
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        bindings: list[dict[str, Any]] = []
        total_records = 0
        for name, rows, binding in prepared:
            split_root = staging / name
            split_root.mkdir()
            (split_root / "runtime_scheduler_inputs.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            (split_root / "manifest.json").write_text(json.dumps({
                "schema": "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_SPLIT_V1",
                **binding,
                "record_count": len(rows),
                "identity_count": binding["identity_count"],
                "teacher_fields_consumed": False,
                "future_fields_consumed": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            seal_directory(split_root)
            bindings.append(dict(binding, split_seal_sha256=sha256_file(split_root / "SHA256SUMS")))
            total_records += len(rows)
        (staging / "split_bindings.json").write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_BUNDLE_V2",
            "split_names": list(EXPECTED_SPLITS),
            "split_count": len(bindings),
            "record_count": total_records,
            "formal_selection_eligible": False,
            "training_authorized": False,
            "attack_enabled": False,
            "teacher_fields_consumed": False,
            "future_fields_consumed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _seal_tree(staging)
        verify_sealed_tree(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "split_names": list(EXPECTED_SPLITS), "record_count": total_records, "output_root": str(output_root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, help="NAME=sealed_split_spec.json; repeat exactly 12 times")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(rematerialize([_spec(value) for value in args.split], args.output_root), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
