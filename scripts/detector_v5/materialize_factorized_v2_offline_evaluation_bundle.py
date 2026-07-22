#!/usr/bin/env python3
"""Materialize a label-only evaluation bundle from an existing sealed Teacher root.

No model, runtime action, or Student input is read.  The command is intended
for a future mounted-root authorization and fails closed on missing roots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory, sha256_file, verify_sealed_directory  # noqa: E402


class EvaluationBundleError(ValueError):
    pass


def _identity_values(values: object) -> set[str]:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
        or len(values) != len(set(values))
    ):
        raise EvaluationBundleError("IDENTITY_MANIFEST_EMPTY_INVALID_OR_DUPLICATE")
    return set(values)


def _identities(path: Path) -> set[str]:
    if not path.is_file():
        raise EvaluationBundleError("IDENTITY_MANIFEST_MISSING")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            for key in ("identities", "identity_list", "episodes"):
                if key in value:
                    return _identity_values(value[key])
        if isinstance(value, list):
            return _identity_values(value)
        raise EvaluationBundleError("IDENTITY_MANIFEST_SCHEMA")
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise EvaluationBundleError("IDENTITY_MANIFEST_ROW_INVALID")
            identity = item.get("episode") or item.get("identity") or item.get("canonical_parent_key")
            values.append(identity)
        return _identity_values(values)


def _row_paths(root: Path) -> list[Path]:
    top_level = [
        root / name
        for name in (
            "teacher_records.jsonl",
            "teacher_labels.jsonl",
            "student_teacher_records.jsonl",
            "episode_records.jsonl",
        )
        if (root / name).is_file()
    ]
    recursive = sorted(
        path for path in root.rglob("*.jsonl")
        if "factorized_teacher" in path.name
    )
    if top_level and recursive:
        raise EvaluationBundleError("AMBIGUOUS_TEACHER_LABEL_STREAM")
    if len(top_level) > 1:
        raise EvaluationBundleError("AMBIGUOUS_TEACHER_LABEL_STREAM")
    paths = top_level or recursive
    if not paths:
        raise EvaluationBundleError("TEACHER_LABEL_STREAM_MISSING")
    return paths


def materialize(
    teacher_root: Path,
    identity_manifest: Path,
    feature_root: Path,
    output_root: Path,
    *,
    split: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output_root}")
    if not re.fullmatch(r"o[0-3]_i[0-2]", split):
        raise EvaluationBundleError("SPLIT_INVALID")
    lowered = "/".join(str(path).replace("\\", "/").lower() for path in (teacher_root, identity_manifest, feature_root, output_root))
    if any(marker in lowered.split("/") for marker in ("fit-dev", "fit_dev", "cal", "check", "cs200", "attack")):
        raise EvaluationBundleError("PROTECTED_SPLIT_PATH")
    verify_sealed_directory(teacher_root)
    verify_sealed_directory(feature_root)
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise EvaluationBundleError("CHECKPOINT_SHA_INVALID")
    try:
        int(checkpoint_sha256, 16)
    except ValueError as exc:
        raise EvaluationBundleError("CHECKPOINT_SHA_INVALID") from exc
    identities = _identities(identity_manifest)
    streams = _row_paths(teacher_root)
    teacher_seal = sha256_file(teacher_root / "SHA256SUMS")
    output_rows = []
    seen = set()
    steps_by_identity = {identity: [] for identity in identities}
    for stream in streams:
        for line in stream.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            if not isinstance(source, dict):
                raise EvaluationBundleError("TEACHER_ROW_INVALID")
            identity = source.get("episode") or source.get("identity") or source.get("canonical_parent_key")
            step = source.get("step", source.get("step_index"))
            if identity not in identities:
                continue
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise EvaluationBundleError("TEACHER_STEP_INVALID")
            key = (identity, step)
            if key in seen:
                raise EvaluationBundleError("DUPLICATE_EVALUATION_STEP")
            seen.add(key)
            steps_by_identity[identity].append(step)
            if "strict_k10_feasible" not in source or "strict_k10_known_mask" not in source:
                raise EvaluationBundleError("STRICT_K10_FIELDS_MISSING")
            feasible = source["strict_k10_feasible"]
            known = source["strict_k10_known_mask"]
            if not isinstance(feasible, bool) or not isinstance(known, bool):
                raise EvaluationBundleError("STRICT_K10_FIELD_TYPE_INVALID")
            eligible = source.get("eligible_start", feasible)
            if not isinstance(eligible, bool):
                raise EvaluationBundleError("ELIGIBLE_START_TYPE_INVALID")
            output_rows.append({
                "episode": identity,
                "step": step,
                "strict_k10_feasible": feasible,
                "strict_k10_known_mask": known,
                "teacher_label_seal": teacher_seal,
                "eligible_start": eligible,
            })
    observed_identities = {row["episode"] for row in output_rows}
    if observed_identities != identities:
        raise EvaluationBundleError(
            f"EVALUATION_IDENTITY_CLOSURE_FAIL:missing={sorted(identities - observed_identities)}"
        )
    for identity, steps in steps_by_identity.items():
        if sorted(steps) != list(range(len(steps))):
            raise EvaluationBundleError(f"EVALUATION_STEP_CLOSURE_FAIL:{identity}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "evaluation_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(output_rows, key=lambda row: (row["episode"], row["step"]))), encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_OFFLINE_EVALUATION_BUNDLE_V1",
            "split": split,
            "data_filename": "evaluation_records.jsonl",
            "record_stream": "evaluation_records.jsonl",
            "record_count": len(output_rows),
            "identity_count": len({row["episode"] for row in output_rows}),
            "checkpoint_sha256": checkpoint_sha256.lower(),
            "teacher_label_seal_sha256": teacher_seal,
            "feature_input_seal_sha256": sha256_file(feature_root / "SHA256SUMS"),
            "fields": [
                "strict_k10_feasible",
                "strict_k10_known_mask",
                "identity",
                "step",
                "teacher_label_seal",
                "eligible_start_contract",
            ],
            "teacher_root_sha256s_sha256": teacher_seal,
            "identity_manifest_sha256": sha256_file(identity_manifest),
            "runtime_fields_consumed": False,
            "student_input_consumed": False,
            "formal_selection_eligible": False,
            "training_authorized": False,
            "attack_authorized": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        verify_sealed_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "output_root": str(output_root), "record_count": len(output_rows), "sha256s_sha256": sha256_file(output_root / "SHA256SUMS")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(
            args.teacher_root.resolve(),
            args.identity_manifest.resolve(),
            args.feature_root.resolve(),
            args.output_root.resolve(),
            split=args.split,
            checkpoint_sha256=args.checkpoint_sha256,
        ), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
