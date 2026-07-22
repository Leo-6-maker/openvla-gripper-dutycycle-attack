#!/usr/bin/env python3
"""Build Factorized V2 runtime scheduler inputs without inference."""

from __future__ import annotations

import argparse
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
)


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
    if not isinstance(value, str):
        raise FactorizedRuntimeError("IDENTITY_MISSING")
    return value


def _episode_root(root: Path, identity: str, required: str) -> Path:
    suite, task, state = identity.split("/")
    candidates = [
        root / suite / task / state,
        root / "episodes" / suite / task / state,
        root / "episodes" / identity,
    ]
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


def _spec(value: str) -> dict[str, Any]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
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
    if prediction_binding.get("checkpoint_sha256", "").lower() != checkpoint_sha:
        raise FactorizedRuntimeError("CHECKPOINT_SHA_MISMATCH")
    if prediction_binding.get("source_commit") != item["source_commit"]:
        raise FactorizedRuntimeError("SOURCE_COMMIT_MISMATCH")
    prediction_seal = sha256_file(prediction_root / "SHA256SUMS")
    runtime_seal = sha256_file(runtime_root / "SHA256SUMS")
    runtime_manifest = _source_binding(runtime_root)
    predictions = _jsonl(_stream(prediction_root, ("prediction_records.jsonl", "factorized_step_predictions.jsonl", "heldout_step_predictions.jsonl")))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(_identity(row), []).append(row)
    output: list[dict[str, Any]] = []
    for identity, prediction_rows in sorted(grouped.items()):
        student_episode = _episode_root(student_root, identity, "student_input_records.jsonl")
        runtime_episode = _episode_root(runtime_root, identity, "step_records.jsonl")
        joined = exact_runtime_step_join(
            prediction_rows,
            _jsonl(student_episode / "student_input_records.jsonl"),
            _jsonl(runtime_episode / "step_records.jsonl"),
        )
        for prediction, student, runtime in joined:
            output.append(build_runtime_record(
                prediction,
                student,
                runtime,
                checkpoint_sha256=checkpoint_sha,
                source_commit=item["source_commit"],
                prediction_artifact_seal=prediction_seal,
                runtime_artifact_seal=runtime_seal,
                feature_order_sha256=item["feature_order_sha256"],
                runtime_manifest=runtime_manifest,
            ))
    return item["name"], output, {
        "prediction_root_sha256s_sha256": prediction_seal,
        "student_root_sha256s_sha256": sha256_file(student_root / "SHA256SUMS"),
        "runtime_root_sha256s_sha256": runtime_seal,
        "checkpoint_sha256": checkpoint_sha,
        "source_commit": item["source_commit"],
        "feature_order_sha256": item["feature_order_sha256"],
        "identity_count": len(grouped),
        "record_count": len(output),
    }


def rematerialize(items: list[dict[str, Any]], output_root: Path, *, require_twelve: bool = True) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output_root}")
    if require_twelve and len(items) != 12:
        raise FactorizedRuntimeError("EXACT_12_SPLIT_CLOSURE_REQUIRED")
    names = [item.get("name") for item in items]
    if len(names) != len(set(names)):
        raise FactorizedRuntimeError("DUPLICATE_SPLIT_NAME")
    prepared = [_prepare_split(item) for item in items]
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        records: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        for name, rows, binding in prepared:
            records.extend(rows)
            split_rows.append(dict({"split": name}, **binding))
        (staging / "runtime_scheduler_inputs.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
        )
        (staging / "split_bindings.json").write_text(json.dumps(split_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_BUNDLE_V1",
            "split_count": len(items),
            "record_count": len(records),
            "formal_selection_eligible": False,
            "training_authorized": False,
            "attack_enabled": False,
            "teacher_fields_consumed": False,
            "future_fields_consumed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "split_count": len(items), "record_count": len(records), "output_root": str(output_root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, help="NAME=sealed_split_spec.json; repeat exactly 12 times for production")
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
