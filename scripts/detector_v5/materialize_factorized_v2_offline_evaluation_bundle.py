#!/usr/bin/env python3
"""Materialize a label-only evaluation bundle from an existing sealed Teacher root.

No model, runtime action, or Student input is read.  The command is intended
for a future mounted-root authorization and fails closed on missing roots.
"""

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


class EvaluationBundleError(ValueError):
    pass


def _identities(path: Path) -> set[str]:
    if not path.is_file():
        raise EvaluationBundleError("IDENTITY_MANIFEST_MISSING")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("identities", "identity_list", "episodes"):
                if isinstance(value.get(key), list):
                    return {str(item) for item in value[key] if isinstance(item, str)}
        if isinstance(value, list):
            return {str(item) for item in value if isinstance(item, str)}
    except json.JSONDecodeError:
        pass
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                identity = item.get("episode") or item.get("identity") or item.get("canonical_parent_key")
                if isinstance(identity, str):
                    result.add(identity)
    if not result:
        raise EvaluationBundleError("IDENTITY_MANIFEST_EMPTY")
    return result


def _rows(root: Path) -> Path:
    for name in ("teacher_records.jsonl", "teacher_labels.jsonl", "student_teacher_records.jsonl", "episode_records.jsonl"):
        path = root / name
        if path.is_file():
            return path
    raise EvaluationBundleError("TEACHER_LABEL_STREAM_MISSING")


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
    stream = _rows(teacher_root)
    teacher_seal = sha256_file(teacher_root / "SHA256SUMS")
    output_rows = []
    seen = set()
    for line in stream.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        identity = source.get("episode") or source.get("identity") or source.get("canonical_parent_key")
        step = source.get("step", source.get("step_index"))
        if identity not in identities:
            continue
        if not isinstance(step, int) or step < 0:
            raise EvaluationBundleError("TEACHER_STEP_INVALID")
        key = (identity, step)
        if key in seen:
            raise EvaluationBundleError("DUPLICATE_EVALUATION_STEP")
        seen.add(key)
        if "strict_k10_feasible" not in source or "strict_k10_known_mask" not in source:
            raise EvaluationBundleError("STRICT_K10_FIELDS_MISSING")
        output_rows.append({
            "episode": identity,
            "step": step,
            "strict_k10_feasible": source["strict_k10_feasible"],
            "strict_k10_known_mask": source["strict_k10_known_mask"],
            "teacher_label_seal": teacher_seal,
            "eligible_start": bool(source.get("eligible_start", source.get("strict_k10_feasible", False))),
        })
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
