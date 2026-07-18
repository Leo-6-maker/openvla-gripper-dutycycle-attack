#!/usr/bin/env python3
"""Build a sealed V5 clean-only utility-proxy Teacher derivative."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import sha256_file
from gripper_attack.v5_teacher import HIGH_VALUE_RETENTION_WINDOW_MIN_STEPS, convert_teacher_row


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _recursive_seal(root: Path) -> dict[str, str]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"source Teacher root is not sealed: {root}")
    expected = f"{sha256_file(sums)}  SHA256SUMS"
    if sidecar.read_text(encoding="utf-8").strip() != expected:
        raise ValueError("source Teacher SHA256SUMS sidecar mismatch")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not name or name in listed or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"invalid source checksum row: {name}")
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"source checksum mismatch: {name}")
        listed[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(listed) | {"SHA256SUMS", "SHA256SUMS.sha256"}:
        raise ValueError("source Teacher file-set closure mismatch")
    return {"sha256sums_sha256": sha256_file(sums), "file_count": str(len(listed))}


def _write_recursive_seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in payloads),
        encoding="utf-8",
    )
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def build(source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_seal = _recursive_seal(source_root)
    identities = 0
    steps = 0
    known_steps = 0
    phase_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        for labels_path in sorted(source_root.rglob("teacher_v213_labels.jsonl")):
            identity = "/".join(labels_path.relative_to(source_root).parts[:-1])
            target = staging / labels_path.relative_to(source_root).parent
            target.mkdir(parents=True, exist_ok=True)
            converted = [convert_teacher_row(row, identity, index) for index, row in enumerate(_jsonl(labels_path))]
            (target / "v5_teacher_utility.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in converted), encoding="utf-8"
            )
            identities += 1
            steps += len(converted)
            known_steps += sum(bool(row["known_mask"]) for row in converted)
            phase_counts.update(row["phase_name"] for row in converted)
            tier_counts.update(str(row["utility_tier"]) for row in converted if row["utility_tier"] is not None)
        if identities != 800:
            raise ValueError(f"V5 Teacher requires 800 FIT identities, got {identities}")
        manifest = {
            "schema": "DETECTOR_V5_TEACHER_UTILITY_V1_MANIFEST",
            "source_root_sha256s_sha256": source_seal["sha256sums_sha256"],
            "source_schema": "DETECTOR_V4_TEACHER_V213_V1_MANIFEST",
            "identity_count": identities,
            "step_count": steps,
            "known_step_count": known_steps,
            "phase_counts": dict(sorted(phase_counts.items())),
            "utility_tier_counts": dict(sorted(tier_counts.items())),
            "high_value_proxy_rule": f"VALID_RETENTION window length >= {HIGH_VALUE_RETENTION_WINDOW_MIN_STEPS} and no release/veto",
            "teacher_is_clean_only_proxy": True,
            "student_future_leakage": False,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        (staging / "v5_teacher_utility_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_recursive_seal(staging)
        os.replace(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
