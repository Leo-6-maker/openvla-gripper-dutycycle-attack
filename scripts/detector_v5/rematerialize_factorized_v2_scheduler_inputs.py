#!/usr/bin/env python3
"""Read-only Factorized V2 -> scheduler-ready rematerialization.

The current Factorized V2 stream intentionally fails because it has no
authoritative utility/regrasp heads.  This entrypoint exists to make that
boundary explicit; it never substitutes another Factorized head.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory, verify_sealed_directory  # noqa: E402
from gripper_attack.factorized_scheduler_bridge import (  # noqa: E402
    SchedulerBridgeError,
    build_scheduler_ready_record,
    exact_step_join,
    sha256_file,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_root(root: Path, identity: str) -> Path:
    suite, task, state = identity.split("/")
    candidates = [
        root / suite / task / state,
        root / "episodes" / suite / task / state,
        root / "episodes" / identity,
    ]
    matches = [path for path in candidates if (path / "student_input_records.jsonl").is_file() and (path / "step_records.jsonl").is_file()]
    if len(matches) != 1:
        raise SchedulerBridgeError(f"SOURCE_EPISODE_ROOT_AMBIGUOUS:{identity}:{len(matches)}")
    return matches[0]


def _prediction_path(root: Path) -> Path:
    for name in ("heldout_step_predictions.jsonl", "prediction_records.jsonl", "factorized_step_predictions.jsonl"):
        path = root / name
        if path.is_file():
            return path
    raise SchedulerBridgeError("PREDICTION_STREAM_MISSING")


def _source_manifest(root: Path) -> dict:
    for name in ("input_manifest.json", "source_binding.json", "manifest.json"):
        path = root / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    return {}


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        identity = row.get("canonical_parent_key", row.get("episode"))
        if not isinstance(identity, str):
            raise SchedulerBridgeError("PREDICTION_IDENTITY_MISSING")
        grouped.setdefault(identity, []).append(row)
    return grouped


def rematerialize(args: argparse.Namespace) -> dict:
    prediction_root = args.prediction_root.resolve()
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output}")
    verify_sealed_directory(prediction_root)
    verify_sealed_directory(source_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)

    binding_path = prediction_root / "source_binding.json"
    if not binding_path.is_file():
        raise SchedulerBridgeError("PREDICTION_SOURCE_BINDING_MISSING")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    bound_checkpoint = binding.get("checkpoint_sha256")
    if bound_checkpoint is None:
        raise SchedulerBridgeError("PREDICTION_CHECKPOINT_SHA_MISSING")
    if bound_checkpoint.lower() != checkpoint_sha:
        raise SchedulerBridgeError("CHECKPOINT_SHA_MISMATCH")
    if binding.get("source_commit") != args.source_commit:
        raise SchedulerBridgeError("SOURCE_COMMIT_MISMATCH")

    predictions = _group(_jsonl(_prediction_path(prediction_root)))
    source_manifest = _source_manifest(source_root)
    output_rows: list[dict] = []
    for identity, prediction_rows in sorted(predictions.items()):
        episode = _episode_root(source_root, identity)
        student_rows = _jsonl(episode / "student_input_records.jsonl")
        runtime_rows = _jsonl(episode / "step_records.jsonl")
        joined = exact_step_join(prediction_rows, student_rows, runtime_rows)
        for prediction, student, runtime in joined:
            output_rows.append(build_scheduler_ready_record(
                prediction,
                student,
                runtime,
                checkpoint_sha256=checkpoint_sha,
                source_commit=args.source_commit,
                input_artifact_seal=sha256_file(source_root / "SHA256SUMS"),
                feature_order_sha256=args.feature_order_sha256,
                runtime_manifest=source_manifest,
            ))

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "scheduler_ready_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8"
        )
        (staging / "source_binding.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_SCHEDULER_READY_SOURCE_BINDING_V1",
            "prediction_root_sha256s_sha256": sha256_file(prediction_root / "SHA256SUMS"),
            "source_root_sha256s_sha256": sha256_file(source_root / "SHA256SUMS"),
            "checkpoint_sha256": checkpoint_sha,
            "source_commit": args.source_commit,
            "feature_order_sha256": args.feature_order_sha256,
            "formal_selection_eligible": False,
            "attack_enabled": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_SCHEDULER_READY_PREDICTION_BUNDLE_V1",
            "record_count": len(output_rows),
            "identity_count": len(predictions),
            "formal_selection_eligible": False,
            "attack_enabled": False,
            "teacher_fields_consumed": False,
            "future_fields_consumed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "record_count": len(output_rows), "identity_count": len(predictions), "output_root": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--feature-order-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    try:
        print(json.dumps(rematerialize(parser.parse_args()), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
